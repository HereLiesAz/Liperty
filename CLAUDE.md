# CLAUDE.md

This file provides guidance to Claude (and other AI coding assistants) when working with code in this repository.

## Project Overview

**Liperty** is a real-time, on-device Visual Speech Recognition (VSR) Android application. It uses deep learning to convert lip movements into text (lipreading) or synthesized speech (Silent Speech Interface). It targets Deaf, Hard-of-Hearing, and speech-impaired communities.

### Key Characteristics
- **Platform**: Android (Kotlin-first, minSdk 26 / targetSdk 36)
- **ML Stack**: TensorFlow Lite (LiteRT 2.17.0), MediaPipe Tasks Vision, OpenCV 4.10.0 (C++ via NDK)
- **UI**: Jetpack Compose + Material 3
- **Privacy**: Fully offline; zero cloud dependencies. Biometric data lives only in RAM.
- **Hardware acceleration**: GPU Delegate (primary), NNAPI/Hexagon (fallback), CPU (last resort)

---

## Repository Structure

```
Liperty/
├── app/
│   ├── build.gradle.kts               # App-level build config (AGP 9.2.1)
│   └── src/
│       ├── main/
│       │   ├── AndroidManifest.xml
│       │   ├── assets/                # .tflite models, .task files (not in git)
│       │   ├── cpp/
│       │   │   └── CMakeLists.txt     # OpenCV NDK integration (C++17)
│       │   ├── java/com/hereliesaz/liperty/
│       │   │   ├── LipertyApplication.kt
│       │   │   ├── MainActivity.kt    # App entry point
│       │   │   ├── camera/
│       │   │   │   └── CameraManager.kt
│       │   │   ├── dsp/
│       │   │   │   └── VibraPhoneDSP.kt    # Experimental silent-speech DSP
│       │   │   ├── ml/
│       │   │   │   ├── ModelEngine.kt       # Interface for inference backends
│       │   │   │   ├── TFLiteEngine.kt      # TFLite executor + GPU fallback
│       │   │   │   ├── VSRInference.kt      # Pipeline orchestrator
│       │   │   │   ├── FrameBuffer.kt       # Rolling frame window (capacity set at construction time)
│       │   │   │   ├── FaceLandmarkerHelper.kt  # MediaPipe wrapper
│       │   │   │   ├── BeamSearchDecoder.kt
│       │   │   │   ├── GreedyDecoder.kt
│       │   │   │   ├── HomopheneCorrector.kt
│       │   │   │   ├── LanguageModel.kt
│       │   │   │   ├── LipReadingModel.kt
│       │   │   │   ├── HandGestureHelper.kt
│       │   │   │   └── OnDeviceTrainer.kt
│       │   │   ├── ui/
│       │   │   │   ├── LipertyApp.kt        # Compose root
│       │   │   │   ├── OverlayView.kt       # Landmark/text overlay
│       │   │   │   ├── GestureListener.kt
│       │   │   │   ├── SettingsActivity.kt
│       │   │   │   └── TranscriptionManager.kt
│       │   │   └── utils/
│       │   │       ├── ImageUtils.kt
│       │   │       ├── BitmapPool.kt
│       │   │       └── PerformanceMonitor.kt
│       │   └── res/                   # Android resources
│       ├── test/java/                 # Robolectric unit tests (14 files)
│       └── androidTest/java/          # Espresso instrumented tests
├── VALLR/                             # Research model (Python, ICCV 2025)
│   ├── Models/
│   └── Data/
├── docs/
│   ├── ARCHITECTURE.md
│   ├── TODO.md                        # 8-phase roadmap
│   ├── RESEARCH.md
│   ├── USER_GUIDE.md
│   ├── LEGAL.md
│   └── MODEL_CONVERSION.md
├── tools/                             # Python model-generation scripts
├── gradle/
│   └── libs.versions.toml            # Centralized version catalog
├── build.gradle.kts                   # Root Gradle config
├── settings.gradle.kts                # Multi-module (OpenCV included dynamically)
├── gradle.properties                  # JVM heap = 2 GB
├── version.properties                 # Major=0, Minor=1, Patch=0
├── setup_libs.sh                      # Downloads OpenCV + MediaPipe model
└── AGENTS.md                          # Guidance for Warp (warp.dev)
```

---

## Development Environment & Setup

### Prerequisites
- **JDK 17** (enforced by CI/CD and `build.gradle.kts`)
- **Android SDK** compileSdk 36, buildToolsVersion matching AGP 9.2.1
- **CMake 3.22.1+** and NDK (for C++ OpenCV integration)
- **Python 3** (for `tools/` model scripts and `VALLR/`)

### 1. Dependency Initialization (Required before first build)
OpenCV binaries and ML model files are **not committed to git**.

```bash
./setup_libs.sh
```

This script:
- Downloads OpenCV Android SDK v4.10.0 and patches its `build.gradle` for AGP 9 / Java 17 compatibility
- Downloads the MediaPipe Face Landmarker `.task` file into `app/src/main/assets/`
- Generates a dummy VSR TFLite stub if a real model is absent

### 2. Build Commands

```bash
# Debug APK
./gradlew assembleDebug

# Release APK (requires keystore secrets)
./gradlew assembleRelease

# Unit tests (Robolectric, no device needed)
./gradlew testDebugUnitTest

# Instrumented tests (requires connected device or emulator)
./gradlew connectedDebugAndroidTest
```

### 3. Device Testing Notes
- **Physical device strongly preferred** — emulators lack NPU/GPU delegate support and CameraX telephoto lens enumeration.
- Minimum API 26 (Android 8.0).
- Camera and (optional) Microphone permissions are required at runtime.

---

## Inference Pipeline (Data Flow)

```
CameraX frame (25–30 FPS)
        │
        ▼
FaceLandmarkerHelper  ──► 468 MediaPipe landmarks
        │
        ▼
ImageUtils.alignAndCropMouth()  ──► grayscale 96×96 mouth ROI
        │
        ▼
FrameBuffer  ──► rolling window of frames (~2 s at default capacity)
        │
        ▼
TFLiteEngine  ──► .tflite model (GPU Delegate → CPU fallback)
        │
        ▼
BeamSearchDecoder / GreedyDecoder  ──► phoneme probabilities → text
        │
        ▼
HomopheneCorrector + LanguageModel  ──► post-processed transcript
        │
        ▼
TranscriptionManager  ──► displayed in OverlayView / Compose UI
```

---

## Key Coding Guidelines

### Machine Learning & TFLite

- **Delegate initialization** must be wrapped in `try-catch`. Device delegate support varies widely (GPU, NNAPI, Hexagon, CPU). `TFLiteEngine` cascades from GPU → CPU automatically.
- **Memory**: Use `ByteBuffer.allocateDirect()` for model I/O to avoid GC pressure during camera callbacks. Reuse `Bitmap` objects through `BitmapPool`.
- **Model interface**: New inference backends must implement the `ModelEngine` interface, not touch `TFLiteEngine` directly.
- **Frame window**: `FrameBuffer` capacity is configurable at construction time and may vary based on the model's input shape. Do not change the capacity without retraining and converting the model to match.
- **Input shape**: The VSR model's expected input shape depends on the loaded model; `[1, 50, 64, 128, 3]` (batch, frames, H, W, channels) are the fallback defaults. Preprocessing in `VSRInference` must produce the shape the loaded model declares.

### Camera & Computer Vision

- **Lens selection**: Never hardcode `LENS_FACING_BACK`. `CameraManager` queries `CameraCharacteristics` and prefers longer focal lengths (telephoto) to reduce perspective distortion — preserve this behavior.
- **Rotation**: VSR models are sensitive to frame orientation. Frames must be rotation-corrected before cropping.
- **OpenCV**: Used exclusively for low-level image processing in C++ (`app/src/main/cpp/`). Do not import `org.opencv` in Kotlin where a pure-Kotlin or MediaPipe solution exists.

### Legal & Privacy (Critical)

- **Biometric Identifiers**: Face meshes and lip landmarks are legally classified as biometric data under BIPA (Illinois), GDPR (EU), and similar laws. See `docs/LEGAL.md`.
- **No persistent storage of raw biometrics**: Never write video frames, face meshes, or landmarked images to disk/NAND unless the user explicitly triggers a debug export.
- **RAM-only processing**: All biometric data must remain in volatile memory only.
- **Consent gate**: Do **not** bypass `checkConsentAndStart()` in `MainActivity`. This legal-consent dialog is mandatory on first launch.

### UI / UX

- **Primary audience**: Deaf/Hard-of-Hearing users. Visual feedback (text overlays, Compose UI) is the primary modality.
- **Secondary audience**: Hearing interlocutors. TTS output is secondary.
- **Accessibility**: Maintain high-contrast text overlays and adequate touch target sizes.
- **Compose conventions**: State lives in `ViewModel` / `TranscriptionManager`. UI is stateless and observes `StateFlow`s.
- **Navigation**: Custom `AzNavRail` (com.github.HereLiesAz:AzNavRail) is used in place of standard `NavigationRail`. Do not replace it with the Compose built-in unless the library is removed.

### Concurrency

- All camera callbacks and ML inference run on background coroutine dispatchers via `lifecycleScope` / `viewModelScope`.
- Do not perform blocking operations on `Dispatchers.Main`.
- `FrameBuffer` uses its own `Mutex` for thread-safe access; respect it.

---

## Testing Conventions

### Unit Tests (`app/src/test/`)
- Framework: **JUnit 4 + Robolectric 4.16.1**
- Cover every class in `ml/` individually (see `TFLiteEngineTest`, `BeamSearchDecoderTest`, etc.)
- Use mock TFLite models (stubs/fakes) — never load real `.tflite` assets in unit tests.
- Run without a device: `./gradlew testDebugUnitTest`

### Instrumented Tests (`app/src/androidTest/`)
- Framework: **Espresso + JUnit 4**
- `IntegrationTest.kt` — end-to-end pipeline validation on device.
- `PrivacyTest.kt` — verifies biometric data is never written to disk.
- Run on device: `./gradlew connectedDebugAndroidTest`

### What to Test When Adding Features
| Area changed | Tests to add/update |
|---|---|
| `ml/` classes | Corresponding `*Test.kt` in `test/java/.../ml/` |
| Camera pipeline | `CameraManagerTest` |
| UI/transcription | `TranscriptionManagerTest` |
| Biometric data handling | `PrivacyTest` |
| End-to-end | `IntegrationTest` |

---

## CI/CD Pipeline (`.github/workflows/build.yml`)

- **Triggers**: All pushes and PRs to `main` / `master`
- **JDK**: 17
- **Steps**:
  1. `./setup_libs.sh` — download OpenCV + model files
  2. Inject `google-services.json` from GitHub Secrets
  3. Generate release keystore from Secrets (`KEYSTORE_PRIVATE`, `KEYSTORE_CHAIN`, `KEYSTORE_PASSWORD`, `KEY_ALIAS`, `KEY_PASSWORD`)
  4. `./gradlew assembleRelease`
  5. Publish signed APK to GitHub Releases with semantic version `MAJOR.MINOR.PATCH.BUILD`
  6. Report failures to Jules (AI agent integration)

**Version scheme**: `version.properties` holds `Major`, `Minor`, `Patch`. Build number is computed from `git rev-list --count HEAD`.

---

## Dependency Management

All versions are centralized in `gradle/libs.versions.toml`. When adding or upgrading a dependency:
1. Add/update the version in `[versions]` table.
2. Declare the artifact in `[libraries]`.
3. Reference via the generated accessor (e.g., `libs.tensorflow.lite`) in `build.gradle.kts`.
4. Do not hardcode version strings in `build.gradle.kts`.

### Key Versions (as of v0.1.0)
| Dependency | Version |
|---|---|
| AGP | 9.2.1 |
| Kotlin | 2.2.10 (bundled with AGP 9) |
| Compose BOM | 2026.05.00 |
| CameraX | 1.6.0 |
| TFLite (LiteRT) | 2.1.4 |
| MediaPipe Tasks Vision | 0.20230731 |
| OpenCV | 4.5.3.0 |
| Robolectric | 4.16.1 |

---

## Architecture Decisions & Constraints

- **Offline-only**: There is no network stack. Do not add `INTERNET` permission or HTTP clients.
- **ModelEngine interface**: Keeps inference backends swappable (TFLite today, future ONNX/ExecuTorch).
- **VALLR model** (`VALLR/`): Reference Python implementation of the ICCV 2025 paper used for training. Conversion to TFLite happens via `docs/MODEL_CONVERSION.md` tooling. Do not modify VALLR Python code without understanding the paper.
- **OpenCV via NDK**: OpenCV is included as an Android module (not AAR) and linked into a shared library via CMake. The `settings.gradle.kts` dynamically includes it after `setup_libs.sh` runs.
- **No cloud ML services**: ARCore API key is present for optional future AR overlays only, not for inference.

---

## Common Pitfalls

1. **Building without `setup_libs.sh`**: The OpenCV module won't exist; Gradle sync will fail.
2. **GPU delegate on emulator**: Will silently fall back to CPU; inference will be very slow. This is expected.
3. **Frame count mismatch**: Changing `FrameBuffer` capacity without retraining and reconverting the TFLite model causes shape mismatch errors at runtime. The capacity must match the model's declared input shape.
4. **Bypassing consent dialog**: Will break BIPA/GDPR compliance and `PrivacyTest` will fail.
5. **Writing landmarks to SharedPreferences**: Illegal under biometric data laws; `PrivacyTest` catches this.
6. **Hardcoding back camera**: Breaks telephoto selection; use `CameraManager` API.

---

## Project Status (v0.1.0)

| Phase | Status |
|---|---|
| 1–2: Core infrastructure | Complete |
| 3: Computer vision pipeline | Partial |
| 4: ML / model optimization | In progress |
| 5–8: Hardware opt, UI, privacy hardening | Pending |

See `docs/TODO.md` for the detailed 8-phase roadmap.
