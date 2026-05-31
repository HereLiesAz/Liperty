# CLAUDE.md

This file provides guidance to Claude (and other AI coding assistants) when working with code in this repository.

## Project Overview

**Liperty** is a real-time, on-device Visual Speech Recognition (VSR) Android application. It uses deep learning to convert lip movements into text (lipreading) or reconstructed speech (via Bone Conduction / Electrolarynx voice reconstruction). It targets Deaf, Hard-of-Hearing, and speech-impaired communities.

### Key Characteristics
- **Platform**: Android (Kotlin-first, minSdk 26 / targetSdk 37)
- **ML Stack**: ONNX Runtime Mobile (primary; serves the SyncVSR visual encoder + attention decoder), TensorFlow Lite / LiteRT (secondary; legacy + small auxiliary models), MediaPipe Tasks Vision, OpenCV 4.13.0 (C++ via NDK)
- **UI**: Jetpack Compose + Material 3
- **Privacy**: Offline *after first-launch setup* — no inference/user data ever leaves the device. The app declares `INTERNET` solely so `setup/ModelDownloadManager.kt` can download ML models from HuggingFace once on first launch into app-private storage; all recognition then runs locally. Biometric data lives only in RAM (never persisted) — the sole exception is the explicitly-consented on-device personalization store, which the user can delete/revoke. Disclose the first-launch network use in the Play Store Data Safety form.
- **Hardware acceleration**: ONNX Runtime CPU (XNNPACK), TFLite GPU Delegate (primary for TFLite path), NNAPI/Hexagon (fallback), CPU (last resort)
- **Production VSR model**: **SyncVSR** (visual-only E2E, LRS3-trained, 5049-token unigram SentencePiece vocab) — encoder + attention decoder (seq2seq) exported to ONNX and pulled at runtime from `HereLiesAz/liperty-syncvsr-onnx`. Selected by `MainActivity.VSR_BACKEND = BACKEND_SYNC_VSR` with `SYNCVSR_USE_SEQ2SEQ = true`. The **Auto-AVSR** model (`Amanvir/LRS3_V_WER19.1`, ESPnet visual-only Conformer + CTC, 5050-token vocab, `HereLiesAz/liperty-autoavsr-onnx`) is an **alternate/legacy backend** selectable via `VSR_BACKEND = BACKEND_AUTO_AVSR`. Lineage of both: Auto-AVSR / Chaplin.
  - ⚠️ **Accuracy is not yet validated on in-domain data.** The only eval run (`docs/EVAL_RESULTS_2026-05-13.md`) scored 100% WER on an out-of-distribution dataset (GRID 16-frame clips). Real WER/CER on a SyncVSR-matched (full-utterance LRS3) test set, and an on-device qualitative pass, are still pending — see the audit roadmap.

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
│       │   │   │   └── VibraPhoneDSP.kt    # BC/EL voice reconstruction DSP
│       │   │   ├── ml/
│       │   │   │   ├── ModelEngine.kt          # Interface for inference backends
│       │   │   │   ├── TFLiteEngine.kt         # TFLite executor + GPU fallback (legacy/aux)
│       │   │   │   ├── OnnxModelEngine.kt      # ONNX Runtime executor — SyncVSR / Auto-AVSR backends
│       │   │   │   ├── VSRInference.kt         # Pipeline orchestrator (parameterized: pixelMean, pixelStd, customDecoder)
│       │   │   │   ├── FrameBuffer.kt          # Rolling frame window + slideAndGetFrames(retainCount)
│       │   │   │   ├── FaceLandmarkerHelper.kt # MediaPipe wrapper
│       │   │   │   ├── VocabularyLoader.kt     # Parses ESPnet unigram5000_units.txt
│       │   │   │   ├── SubwordCTCDecoder.kt    # Greedy subword CTC
│       │   │   │   ├── SubwordCtcBeamDecoder.kt# Beam-search subword CTC (logsumexp prefix merge)
│       │   │   │   ├── GreedyDecoder.kt        # Phoneme greedy CTC (legacy)
│       │   │   │   ├── BeamSearchDecoder.kt    # Phoneme beam search (legacy)
│       │   │   │   ├── HomopheneCorrector.kt
│       │   │   │   ├── LanguageModel.kt
│       │   │   │   ├── LlmTextCleaner.kt       # Opt-in 2nd-stage on-device LLM cleanup (MediaPipe Tasks GenAI)
│       │   │   │   ├── LanguageModelScorer.kt  # Interface + NoopLanguageModelScorer for rescoring
│       │   │   │   ├── KenLmScorer.kt          # KenLM n-gram LM via JNI (see docs/LM_RESCORING.md)
│       │   │   │   ├── VisemeMap.kt            # ARPABET phoneme -> viseme class loader (asset: viseme_map.txt)
│       │   │   │   ├── VisemeIndex.kt          # Word -> viseme-equivalent words lookup (asset: viseme_index.json)
│       │   │   │   ├── VisemeRescorer.kt       # Viseme-aware post-CTC rescorer ("Chaplin's second AI" for visual ASR)
│       │   │   │   ├── AvHubertEncoderSession.kt   # V3 encoder ONNX session (interface: EncoderSession)
│       │   │   │   ├── AvHubertDecoderSession.kt   # V3 seq2seq decoder ONNX session (interface: DecoderSession)
│       │   │   │   ├── Seq2SeqGreedyDecoder.kt # V3 autoregressive greedy loop
│       │   │   │   ├── BpeDetokenizer.kt       # SentencePiece detokenizer (▁ -> space)
│       │   │   │   ├── AvHubertSeq2SeqInference.kt # V3 backend top-level orchestrator (USE_V3_BACKEND=false default)
│       │   │   │   ├── HandGestureHelper.kt
│       │   │   │   ├── MLConstants.kt
│       │   │   │   ├── CalibrationManager.kt   # Per-user pre-deployment calibration
│       │   │   │   └── OnDeviceTrainer.kt
│       │   │   ├── personalization/      # On-device personalization (see docs/PERSONALIZATION.md)
│       │   │   │   ├── PairedTrainingRecord.kt # (audio, lip-video frames, optional transcript) sample
│       │   │   │   ├── PairedTrainingStore.kt  # On-disk store with delete-all retention controls
│       │   │   │   └── VideoFrameExtractor.kt  # MediaMetadataRetriever-based frame extraction
│       │   │   ├── ui/
│       │   │   │   ├── LipertyApp.kt        # Compose root
│       │   │   │   ├── OverlayView.kt       # Landmark/text overlay
│       │   │   │   ├── GestureListener.kt
│       │   │   │   ├── SettingsActivity.kt
│       │   │   │   └── TranscriptionManager.kt
│       │   │   ├── voicebox/
│       │   │   │   ├── VoiceManager.kt        # TTS orchestrator (system + cloned)
│       │   │   │   ├── LaryngealSensor.kt     # BC/EL voice reconstruction sensor
│       │   │   │   ├── AudioRouter.kt          # Audio device routing & full-duplex
│       │   │   │   ├── ArtificialLarynx.kt    # BC carrier output management
│       │   │   │   ├── GlottalCarrierGenerator.kt  # Glottal pulse waveform (80-200 Hz)
│       │   │   │   ├── PocketTTSEngine.kt     # Voice cloning TTS engine
│       │   │   │   ├── VoiceViewModel.kt      # Voice UI state management
│       │   │   │   ├── BluetoothLEAudioManager.kt  # BLE Audio / LC3 codec
│       │   │   │   ├── TrambaProcessor.kt     # Bandwidth expansion model
│       │   │   │   └── cloning/
│       │   │   │       ├── VoiceStore.kt       # Profile persistence (JSON + binary)
│       │   │   │       ├── VoiceProfileBuilder.kt  # Builds profiles from samples
│       │   │   │       ├── AudioPreprocessor.kt    # VAD, resampling, segmentation
│       │   │   │       └── SpeakerClusterer.kt     # Multi-speaker identification
│       │   │   └── utils/
│       │   │       ├── ImageUtils.kt
│       │   │       ├── BitmapPool.kt
│       │   │       └── PerformanceMonitor.kt
│       │   └── res/                   # Android resources
│       ├── test/java/                 # Robolectric unit tests
│       └── androidTest/java/          # Espresso instrumented tests
├── docs/                              # All project documentation
├── docker/                            # Dockerfiles for KenLM/V3 export builds
├── tools/                             # Notebook generators + .ipynb pipelines (Kaggle/Colab)
│   ├── _build_*.py                    # Python generators → .ipynb
│   ├── *.ipynb                        # Generated notebooks (training, export, eval)
│   └── *.py                           # Standalone utilities (build_viseme_*, convert_*, etc.)
├── gradle/
│   └── libs.versions.toml            # Centralized version catalog
├── build.gradle.kts                   # Root Gradle config
├── settings.gradle.kts                # Multi-module (OpenCV included dynamically)
├── gradle.properties                  # JVM heap = 4 GB (-Xmx4096m)
├── version.properties                 # Major=0, Minor=1, Patch=0
└── setup_libs.sh                      # Downloads OpenCV, MediaPipe, all ML models from HF
```

---

## Development Environment & Setup

### Prerequisites
- **JDK 17** (enforced by CI/CD and `build.gradle.kts`)
- **Android SDK** compileSdk 36, buildToolsVersion matching AGP 9.2.1
- **CMake 3.22.1+** and NDK (for C++ OpenCV integration)
- **Python 3** (for `tools/` model scripts)

### 1. Dependency Initialization (Required before first build)
OpenCV binaries and ML model files are **not committed to git**.

```bash
./setup_libs.sh
```

This script:
- Downloads OpenCV Android SDK v4.13.0 and patches its `build.gradle` for AGP 9 / Java 17 compatibility
- Downloads MediaPipe Face & Hand Landmarker `.task` files
- Pulls Auto-AVSR ONNX model + vocab from `HereLiesAz/liperty-autoavsr-onnx`
- Pulls SyncVSR ONNX models from `HereLiesAz/liperty-syncvsr-onnx`
- Pulls AV-HuBERT V3 encoder/decoder from `HereLiesAz/liperty-avhubert-encoder`
- Pulls KenLM binary + NDK prebuilts from `HereLiesAz/liperty-lm`
- Pulls PocketTTS voice cloning ONNX models from `HereLiesAz/liperty-pocket-tts`
- Extracts legacy TFLite stubs + viseme data from a Google Drive assets bundle

**All binary assets in `app/src/main/assets/` are gitignored.** Only small hand-crafted text files (`homophones.json`, `viseme_map*.txt`) are tracked.

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
- Camera, Microphone, and Bluetooth Connect (API 31+) permissions are required at runtime.

---

## Inference Pipeline (Data Flow)

The production path is the **SyncVSR backend** (`VSR_BACKEND = BACKEND_SYNC_VSR`). With `SYNCVSR_USE_SEQ2SEQ = true` (the default), the active route is the **encoder + attention decoder (seq2seq)** — `syncvsr_lrs3_encoder.onnx` (~759 MB) feeding `syncvsr_lrs3_decoder.onnx` (~273 MB), driven by `Seq2SeqGreedyDecoder` and `BpeDetokenizer`. The **CTC head** (`syncvsr_lrs3_visual_ctc_fp16.onnx`, ~370 MB, the path drawn in the diagram below) is the graceful-degradation fallback when the seq2seq ONNX can't load (e.g. OOM on low-RAM devices). The **Auto-AVSR** backend (CTC only) and the legacy phoneme TFLite path remain wired in `VSRInference` and are reachable via `VSR_BACKEND` / config flags for experiments. All paths can be optionally cleaned by an on-device LLM.

> Naming caveat: the companion constants are still named `AUTOAVSR_MODEL` / `AUTOAVSR_VOCAB` / `AUTOAVSR_*` for historical reasons, but they **resolve to the SyncVSR files** when `VSR_BACKEND == BACKEND_SYNC_VSR` (see `MainActivity.kt:102-135`).

```
CameraX frame (25–30 FPS)
        │
        ▼
FaceLandmarkerHelper  ──► 468 MediaPipe landmarks
        │
        ▼
extractLipBoundingBox + ImageUtils.alignAndCropMouth()  ──► grayscale 88×88 mouth ROI
                                                            (SyncVSR / Auto-AVSR; legacy 96×96 for the phoneme TFLite path)
        │
        ▼
FrameBuffer  ──► rolling window; slideAndGetFrames(retainCount=8) yields a copy
                 of the last N frames + retains the tail for streaming inference
        │
        ▼
OnnxModelEngine (SyncVSR / Auto-AVSR CTC)   OR  TFLiteEngine (legacy phoneme path)
        │                                              │
        │  per-channel normalize                       │  greyscale normalize
        │  (mean=0.421, std=0.165)                     │
        ▼                                              ▼
ONNX encoder + CTC head ─► (1, T_out, V)       .tflite ─► (1, T_out, V_phonemes)
  (SyncVSR V=5049, NTCHW; Auto-AVSR V=5050, NCTHW)
  [seq2seq default routes encoder→attention decoder instead — see prose above]
        │                                              │
        ▼                                              ▼
SubwordCtcBeamDecoder (beamWidth=8)            BeamSearchDecoder / GreedyDecoder
+ optional KenLM rescoring at end of beam     phoneme probabilities → text
SentencePiece (▁) → words                              │
        │                                              │
        ▼                                              ▼
                  HomopheneCorrector + LanguageModel (word-level)
        │
        ▼
VisemeRescorer (CTC text → viseme-equivalent alternatives → LM-rescored)
The "Chaplin's-second-AI" for visual ASR: swaps viseme-confusable words
(e.g. "tasty" → "nasty") when the LM prefers the alternative in context.
        │
        ▼
TranscriptionManager  ──► getCurrentSentence() (raw assembled)
                          getCleanedSentence() (after optional transformSentence,
                                                e.g. LlmTextCleaner.clean wrapper —
                                                cached, only re-runs on transcript change)
        │
        ▼
OverlayView / Compose UI
```

**Rescoring stack status:** the LM and viseme paths are wired and unit-tested end-to-end. The KenLM JNI/native build IS in place (`app/src/main/cpp/kenlm_jni.cpp` + `CMakeLists.txt`; arm64 `.a` prebuilts pulled by `setup_libs.sh` from `HereLiesAz/liperty-lm`). Scoring is active at runtime only when **both** (a) the arm64 prebuilts were present at build time so `KenLmScorer.isNativeLoaded` is true, and (b) the LM model (`librispeech_3gram.bin`) is available (downloaded by `ModelDownloadManager`). Otherwise both rescorers run as a no-op input-bias tiebreaker (original CTC output wins). **CI must run `setup_libs.sh` before `assembleRelease` and fail the release build if the prebuilts are absent**, so a silent no-op can't ship — see `docs/LM_RESCORING.md` and `.github/workflows/build.yml`.

**V3 backend (research-only, off by default):** `MainActivity.USE_V3_BACKEND = false`. Flipping the flag swaps the CTC pipeline above for `AvHubertSeq2SeqInference` — AV-HuBERT encoder ONNX + Transformer-decoder seq2seq + BPE detokenization. See `docs/AVHUBERT_V3_BACKEND.md` for the multi-attempt research log and `app/src/main/java/com/hereliesaz/liperty/ml/AvHubertSeq2SeqInference.kt` for the orchestrator.

---

## Key Coding Guidelines

### Machine Learning (ONNX + TFLite)

- **Backend selection**: `OnnxModelEngine` is the production backend, serving **SyncVSR** (seq2seq encoder/decoder by default; CTC head as fallback). `VSR_BACKEND` in the `MainActivity` companion picks SyncVSR vs Auto-AVSR; `SYNCVSR_USE_SEQ2SEQ` picks seq2seq vs CTC within SyncVSR. `TFLiteEngine` is retained for the legacy phoneme path and small auxiliary models. Both implement `ModelEngine`; new backends must too — never touch a concrete engine class directly from `VSRInference`.
- **Delegate initialization** for TFLite must be wrapped in `try-catch`. Device delegate support varies widely (GPU, NNAPI, Hexagon, CPU). `TFLiteEngine` cascades from GPU → CPU automatically. ONNX Runtime defaults to XNNPACK CPU.
- **Memory**: Use `ByteBuffer.allocateDirect()` for TFLite I/O. ONNX I/O uses `OnnxTensor.createTensor` with NIO buffers. Reuse `Bitmap` objects through `BitmapPool`.
- **Frame window**: `FrameBuffer` capacity is configurable at construction time (production uses capacity 16). The VSR backends use streaming inference: `FrameBuffer.slideAndGetFrames(retainCount=8)` returns a copy of the current window and retains the last 8 frames as the seed for the next inference (continuity for the encoder).
- **Input shape**: The model's expected input shape and layout depend on the active backend. **SyncVSR** ONNX is **NTCHW** `(1, T, 1, 88, 88)` float32; **Auto-AVSR** ONNX is **NCTHW** `(1, 1, T, 88, 88)` float32. The legacy TFLite fallback defaults to `(1, 50, 64, 128, 3)` NTHWC. `VSRInference` accepts an `InputLayout` enum (NTHWC, NCTHW, NTCHW) and per-channel `pixelMean`/`pixelStd` so the same orchestrator handles all backends; `MainActivity.VSR_INPUT_LAYOUT` is pre-resolved off `VSR_BACKEND`.
- **VSR constants** (in `MainActivity` companion — note the legacy `AUTOAVSR_*` names resolve to whichever backend `VSR_BACKEND` selects): `AUTOAVSR_MODEL`, `AUTOAVSR_VOCAB`, `AUTOAVSR_CROP_SIZE=88`, `AUTOAVSR_PIXEL_MEAN=0.421f`, `AUTOAVSR_PIXEL_STD=0.165f`, `AUTOAVSR_SLIDE_RETAIN=8`. SyncVSR and Auto-AVSR share the 88×88 crop and mean/std (both LRS3-trained with Chaplin-equivalent `transforms.VideoTransform` preprocessing). Changing them silently divorces deployment from the trained model.
- **Decoders**: For SyncVSR seq2seq (the production default), `Seq2SeqGreedyDecoder` + `BpeDetokenizer` drive the autoregressive attention decoder. For the CTC paths (SyncVSR CTC fallback and Auto-AVSR), `SubwordCtcBeamDecoder` (beam width 8, prefix merge via logsumexp) is the production decoder and `SubwordCTCDecoder` is the greedy variant; both consume the token list parsed by `VocabularyLoader` (SyncVSR: `syncvsr_unigram_units.txt`, 5049 tokens, blank=0, eos=5048; Auto-AVSR: `unigram5000_units.txt`, 5050 tokens; SentencePiece `▁` = word boundary). `BeamSearchDecoder` / `GreedyDecoder` are the legacy phoneme decoders.
- **Optional 2nd-stage cleanup**: `LlmTextCleaner` wraps a small on-device LLM (e.g. Gemma-2B-it via MediaPipe Tasks GenAI's `LlmInference`) to clean noisy CTC output. Wired into `TranscriptionManager` via the optional `transformSentence` constructor callback; result is cached against the assembled transcript so it only re-runs on new content. **Opt-in**: the model file is not bundled in the APK (~1.3 GB); cleaner falls back to raw text when absent.

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
- Cover every class in `ml/` individually (see `TFLiteEngineTest`, `SubwordCtcBeamDecoderTest`, `LlmTextCleanerTest`, `VocabularyLoaderTest`, `FrameBufferSlidingTest`, `VSRInferenceParamsTest`, etc.)
- Use mock models (stubs/fakes) — never load real `.tflite` / `.onnx` assets in unit tests.
- **Robolectric SDK 37 workaround**: `targetSdkVersion=37` confuses Robolectric. Tests that need an Android `Context` should use `@RunWith(AndroidJUnit4::class) + @Config(sdk=[34])` instead of the default RobolectricTestRunner. See `TranscriptionManagerTransformTest` and `LlmTextCleanerTest` for the pattern.
- Pure-JVM tests (no Android `Context`) — like `LlmTextCleanerTest` — should avoid Android Log calls entirely; use silent failure paths instead.
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
| UI/transcription | `TranscriptionManagerTest`, `TranscriptionManagerTransformTest` |
| `voicebox/` classes | `VoiceManagerTest`, `LaryngealSensorTest`, `AudioRouterTest`, `VoiceStoreTest` |
| `voicebox/cloning/` | `AudioPreprocessorTest`, `SpeakerClustererTest`, `VoiceProfileBuilderTest` |
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
| OpenCV | 4.13.0 |
| Robolectric | 4.16.1 |

---

## Architecture Decisions & Constraints

- **Offline after first-launch setup**: The app ships with the `INTERNET` permission and a one-time model downloader (`setup/ModelDownloadManager.kt`) that pulls the ML models from HuggingFace into app-private storage on first launch — currently **~1.4 GB of *required* models** (SyncVSR CTC ~370 MB + encoder ~759 MB + decoder ~273 MB + face landmarker + vocab) plus up to **~1.1 GB optional** (voice-cloning, personalization training artifacts, LM). After that, all inference runs fully offline and no user/inference data ever leaves the device. Do NOT add network calls on the inference/recognition path. The first-launch download must be disclosed in the Play Store Data Safety form. (Training pipelines run off-device on Kaggle/Colab and use HuggingFace Hub as a cross-account state store: `HereLiesAz/liperty-syncvsr-onnx`, `HereLiesAz/liperty-autoavsr-onnx`, `HereLiesAz/liperty-grid-preprocessed`, `HereLiesAz/liperty-tcd-preprocessed`, `e1lephant/lrs3-landmark`. The Android app consumes the resulting artifacts, embedded via `setup_libs.sh` at build time or downloaded at first launch.)
- **ModelEngine interface**: Keeps inference backends swappable. Production: `OnnxModelEngine` (SyncVSR seq2seq, CTC fallback). Alternate/legacy: Auto-AVSR (also `OnnxModelEngine`), `TFLiteEngine` (phoneme path). Future: ExecuTorch.
- **Production model & WER status**: The deployed backend is **SyncVSR** (visual-only E2E, LRS3-trained, 5049-token unigram SentencePiece). Its accuracy on Liperty's deployment configuration is **not yet validated** — see the ⚠️ note in Project Overview and `docs/EVAL_RESULTS_2026-05-13.md`. The documented **Auto-AVSR alternate** (`Amanvir/LRS3_V_WER19.1`, ESPnet Conformer + CTC, 5050 tokens) has a headline 19.1% WER on LRS3, but that requires beam search with CTC + attention scorer + external LM — none of which exports cleanly to ONNX; an encoder+CTC-only export gives roughly 30–50% WER, partly recovered by the optional `LlmTextCleaner` pass. Treat all on-device WER numbers as unmeasured until the in-domain eval (audit Phase 1) lands.
- **Training notebooks** in `tools/` (resumable, run on Kaggle P100 or Colab):
  - `train_grid_tcd_resumable.ipynb` — pixel V1 baseline (GRID + TCD-TIMIT)
  - `train_landmark_lrs3_resumable.ipynb` — landmark-only V2 (uses `e1lephant/lrs3-landmark` shards; the point of the LRS3 landmark releases is exactly that landmark-only training is competitive)
  - `train_personal_lora_resumable.ipynb` — per-user LoRA adaptation on top of a pretrained encoder
  - `eval_autoavsr.ipynb` — offline WER/CER evaluation against held-out shards, mirrors deployment exactly (same ONNX, vocab, mean/std, decoders)
- **OpenCV via NDK**: OpenCV is included as an Android module (not AAR) and linked into a shared library via CMake. The `settings.gradle.kts` dynamically includes it after `setup_libs.sh` runs.
- **No cloud ML services**: ARCore API key is present for optional future AR overlays only, not for inference.

---

## Common Pitfalls

1. **Building without `setup_libs.sh`**: The OpenCV module won't exist; Gradle sync will fail. Also: the SyncVSR ONNX models won't be in `assets/` (and aren't bundled in the APK regardless — they're pruned and downloaded at runtime), so on a fresh build the production VSR backend has nothing to load until `ModelDownloadManager` completes the first-launch download.
2. **GPU delegate on emulator**: Will silently fall back to CPU; inference will be very slow. This is expected.
3. **Frame count / input shape mismatch**: Changing `FrameBuffer` capacity, the 88×88 crop size, or the per-channel mean/std without re-exporting the model causes silent quality cliffs (or shape errors). The VSR preprocessing is calibrated to Chaplin-equivalent `transforms.VideoTransform` (SyncVSR and Auto-AVSR share it) — match those constants or retrain.
4. **Bypassing consent dialog**: Will break BIPA/GDPR compliance and `PrivacyTest` will fail.
5. **Writing landmarks to SharedPreferences**: Illegal under biometric data laws; `PrivacyTest` catches this.
6. **Hardcoding back camera**: Breaks telephoto selection; use `CameraManager` API.
7. **Robolectric SDK 37 trap**: Default `RobolectricTestRunner` chokes on `targetSdk=37`. New tests that need `Context` must use `AndroidJUnit4 + @Config(sdk=[34])` (see `TranscriptionManagerTransformTest`).
8. **Pasting code from chat/markdown**: Copy-pasting code blocks from Claude/ChatGPT into a notebook can corrupt indentation (markdown auto-rendering of em-dashes, smart quotes, no-break spaces). When editing the notebook generators, sanitize ASCII-only and regenerate with `python tools/_build_*.py`.
9. **Per-notebook Kaggle secrets**: `HF_TOKEN` is per-notebook on Kaggle, not account-global. Toggle it ON in the notebook's Add-ons → Secrets pane before running.

---

## Project Status (v0.1.0)

| Phase | Status |
|---|---|
| 1–2: Core infrastructure | Complete |
| 3: Computer vision pipeline | Partial |
| 4: ML / model optimization | SyncVSR backend wired (ONNX seq2seq encoder/decoder, CTC fallback); Auto-AVSR alternate + legacy phoneme path also wired; subword CTC beam search + sliding-window inference live; KenLM + viseme rescoring build-complete; LLM cleanup hook in place; offline eval notebook ready. **⚠️ In-domain accuracy still unmeasured** (only eval run = 100% WER on out-of-distribution data — see `docs/EVAL_RESULTS_2026-05-13.md`). Training pipelines (GRID/TCD pixel, LRS3 landmark, personal LoRA) running on Kaggle |
| 5–6: Hardware opt, UI/UX | Partial |
| 7: Voice Reconstruction (BC/EL) | Complete |
| 8: Bone Conduction hardware | Partial |
| 9–13: Advanced modalities | Pending |

See `docs/TODO.md` for the detailed roadmap.
