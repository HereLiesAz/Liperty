# AGENTS.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

**Liperty** is a real-time, on-device Visual Speech Recognition (VSR) Android application. It uses deep learning to convert lip movements into text (lipreading) or synthesized speech (silent speech interface).

### Key Characteristics
- **Platform**: Android (Native Kotlin)
- **ML Stack**: TensorFlow Lite (LiteRT), MediaPipe, OpenCV (C++ integration)
- **Privacy**: Fully offline processing; no cloud dependencies.
- **Hardware**: Heavy reliance on GPU/NPU acceleration via TFLite Delegates.

## Development Environment & Setup

Before attempting to build or run the application, the environment must be correctly provisioned.

### 1. Dependency Initialization
The project relies on external binaries (OpenCV) and model files that are **not** committed to Git.
You must run the setup script before building:
```bash
./setup_libs.sh
```
This script:
- Downloads and configures the OpenCV Android SDK (v4.10.0+).
- Patches OpenCV's `build.gradle` for compatibility with AGP 9.0 and Java 17.
- Downloads the **Project Bundle** (VALLR, tools/external, and data) from Google Drive.
- Downloads the MediaPipe Face Landmarker task.
- Generates a dummy VSR TFLite model if a real one is not present.

### 2. Build Commands
- **Build Debug APK**:
  ```bash
  ./gradlew assembleDebug
  ```
- **Run Unit Tests**:
  ```bash
  ./gradlew testDebugUnitTest
  ```
- **Run Instrumented Tests**:
  *Note: These require a connected device/emulator.*
  ```bash
  ./gradlew connectedDebugAndroidTest
  ```

### 3. Running on Device
- **Emulator Limitations**: The Android Emulator has limited support for NPU delegates and specific CameraX lens configurations. **Physical devices are strongly recommended** for testing ML inference and CameraX features.
- **Permissions**: The app requires Camera and Microphone (optional) permissions. It also enforces a strict "Legal Consent" dialog on first launch due to biometric privacy laws.

## Architecture & Code Structure

The application follows a modular architecture separating UI, Camera logic, and ML Inference.

### High-Level Data Flow
1.  **Camera Input**: `CameraManager` captures frames via CameraX. It prioritizes **Telephoto lenses** to minimize perspective distortion.
2.  **Face Detection**: `FaceLandmarkerHelper` (MediaPipe) detects 468 facial landmarks.
3.  **Preprocessing**: Frames are cropped to the mouth Region of Interest (ROI), converted to grayscale, and normalized.
4.  **Inference**:
    - Frames are buffered in `FrameBuffer` (rolling window of ~50 frames).
    - `VSRInference` passes the buffer to `TFLiteEngine`.
    - `TFLiteEngine` executes the `.tflite` model (using GPU Delegate if available).
5.  **Decoding**: The model output (probabilities) is decoded into text via `BeamSearchDecoder` or `GreedyDecoder`.
6.  **Post-Processing**: `HomopheneCorrector` applies language modeling or user corrections.

### Key Directories
- **`app/src/main/java/com/HereLiesAz/Liperty/`**:
    - **`camera/`**: `CameraManager.kt` (CameraX setup, lens selection logic).
    - **`ml/`**: Core Machine Learning logic.
        - `ModelEngine.kt`: Interface for inference engines.
        - `TFLiteEngine.kt`: TensorFlow Lite implementation with GPU fallback.
        - `VSRInference.kt`: Orchestrates the inference pipeline (pre-processing -> run -> decode).
        - `FaceLandmarkerHelper.kt`: MediaPipe wrapper.
    - **`ui/`**: Activity and View components.
        - `OverlayView.kt`: Draws landmarks/text over the camera preview.
        - `GestureListener.kt`: Handles swipe gestures for homophene correction.
    - **`utils/`**: Helper classes (`BitmapPool`, `ImageUtils`).
- **`app/src/main/cpp/`**: Native C++ code (OpenCV integration).
- **`app/src/main/assets/`**: Location for `.tflite` models and `.task` files.
- **`tools/`**: Python scripts for model generation and debugging.

## Coding Guidelines

### Machine Learning & NPU
- **Delegates**: Always wrap TFLite delegate initialization in `try-catch` blocks. Different devices support different delegates (GPU, NNAPI, Hexagon). The `TFLiteEngine` currently falls back to CPU if GPU fails.
- **Memory Management**: Use `ByteBuffer.allocateDirect` for model inputs to avoid GC overhead. Reuse `Bitmap` objects via `BitmapPool` where possible to prevent memory churn during high-frequency camera callbacks.

### Camera & Vision
- **Lens Selection**: Do not hardcode `LENS_FACING_BACK`. Use `CameraManager`'s logic to query `CameraCharacteristics` and prefer longer focal lengths (Telephoto).
- **Orientation**: VSR models are sensitive to rotation. Ensure input frames are strictly oriented (usually portrait or landscape aligned to the face).

### Legal & Privacy (Crucial)
- **Biometric Data**: Code interacting with `FaceLandmarker` handles "Biometric Identifiers" (under laws like BIPA).
- **Storage**: **Never** write raw video frames or extracted face meshes to persistent storage (Disk/NAND) unless explicitly triggered by a user "Save" action for debugging. Keep data in volatile memory (RAM) only.
- **Consent**: Do not bypass the `checkConsentAndStart()` flow in `MainActivity`.

### UI/UX
- **Accessibility**: This app is for Deaf/Hard-of-Hearing users. Visual feedback (text, overlays) is primary. Audio feedback (TTS) is for the secondary audience (hearing interlocutors).
