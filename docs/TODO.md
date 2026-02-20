# Comprehensive Todo List

This document provides a granular breakdown of tasks required to build the Liperty application, from initial research to deployment.

## Phase 1: Research & Prerequisites (Completed)

- [x] Read Research Document (docs/RESEARCH.md)
- [x] Define System Architecture (docs/ARCHITECTURE.md)
- [x] Establish Legal Guidelines (docs/LEGAL.md)
- [x] Create Repository Skeleton

## Phase 2: Core Infrastructure

- [ ] **Project Setup**
    - [ ] Create `settings.gradle.kts`
    - [ ] Create root `build.gradle.kts`
    - [ ] Create `app/` module
    - [ ] Add dependencies:
        - [ ] `androidx.camera:camera-core` (1.3.0+)
        - [ ] `androidx.camera:camera-camera2`
        - [ ] `androidx.camera:camera-lifecycle`
        - [ ] `androidx.camera:camera-view`
        - [ ] `com.google.mlkit:face-detection` (or MediaPipe Face Mesh)
        - [ ] `org.tensorflow:tensorflow-lite` (2.14.0+)
        - [ ] `org.tensorflow:tensorflow-lite-gpu`
        - [ ] `org.tensorflow:tensorflow-lite-support`
        - [ ] `org.jetbrains.kotlinx:kotlinx-coroutines-android`

- [ ] **Android Manifest & Permissions**
    - [ ] `android.permission.CAMERA`
    - [ ] `android.hardware.camera.autofocus`
    - [ ] `android.hardware.camera.flash` (optional)
    - [ ] Handle runtime permission requests (ActivityResultContracts).

- [ ] **CameraX Implementation**
    - [ ] Initialize `ProcessCameraProvider`
    - [ ] Implement `Preview` use case
    - [ ] Implement `ImageAnalysis` use case (Backpressure strategy: STRATEGY_KEEP_ONLY_LATEST)
    - [ ] **Critical:** Implement `CameraSelector` logic to prefer **Telephoto** lens for rear camera (research requirement).
    - [ ] Handle Lifecycle (bind/unbind).

## Phase 3: Computer Vision Pipeline (MediaPipe & OpenCV)

- [ ] **Face Landmark Detection**
    - [ ] Integrate MediaPipe Face Mesh (468 landmarks).
    - [ ] Extract Lip landmarks (Indices: 0, 13, 14, 17, 37, 39, 40, 61, 146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 317, 318, 321, 375, 402, 405, 409).
    - [ ] Implement Head Pose Estimation (Roll, Pitch, Yaw).

- [ ] **ROI Extraction & Normalization**
    - [ ] Calculate Affine Transformation Matrix to align mouth horizontally.
    - [ ] Crop mouth region (Square: 96x96, 112x112, or 128x128).
    - [ ] Convert `ImageProxy` (YUV) to `Bitmap` (ARGB) -> Grayscale.
    - [ ] Apply Gaussian Blur (Kernel 3x3 or 5x5).
    - [ ] Apply Contrast Stretching / Histogram Equalization.
    - [ ] **Optimization:** Use RenderScript or Vulkan for image processing if CPU is too slow.

## Phase 4: Machine Learning (VSR & LLM)

- [ ] **Model Selection & Conversion**
    - [ ] **Option A (DeepLip):** Train/Fine-tune CNN-LSTM on LRW/LRS3 dataset.
        - [ ] Convert to TFLite (fp16 quantization).
    - [ ] **Option B (VALLR):** Train Transformer-based Phoneme predictor.
        - [ ] Convert Encoder to TFLite.
        - [ ] Convert Decoder (LLM) to TFLite (int8 quantization).
    - [ ] Place `.tflite` models in `app/src/main/assets/`.

- [ ] **Inference Engine (LiteRT)**
    - [ ] Initialize `Interpreter` with `GpuDelegate` (for Vision/Encoder) and `NnApiDelegate` (for Decoder).
    - [ ] Implement `runInference(inputBuffer: ByteBuffer): OutputBuffer`.
    - [ ] Handle Threading (run on background thread, post to UI).

- [ ] **Decoding Logic**
    - [ ] Implement CTC Beam Search (if using CTC model).
    - [ ] Implement Greedy Decoder (for simple testing).
    - [ ] **Homophene Correction Logic:**
        - [ ] Dictionary lookup for homophenes (e.g., p/b/m).
        - [ ] Contextual scoring (Bigram/Trigram or LLM).

## Phase 5: User Interface (UI/UX)

- [ ] **Main Screen**
    - [ ] Camera Preview Surface.
    - [ ] **OverlayView:** Draw bounding box (Green = Face Detected, Blue = Lips Tracked).
    - [ ] **Transcription Text:** Real-time scrolling text view.
    - [ ] **Status Indicator:** "Listening...", "Processing...", "Error".

- [ ] **Gesture Controls**
    - [ ] **Swipe Left/Right:** On a word to cycle alternative homophenes.
    - [ ] **Swipe Up:** Speak current sentence (TTS).
    - [ ] **Double Tap:** Clear transcript.

- [ ] **Settings**
    - [ ] Toggle Rear/Front Camera (SSI Mode vs. Lipreading Mode).
    - [ ] Adjust Font Size.
    - [ ] Enable/Disable Telephoto Lens Preference.

## Phase 6: Testing & Optimization

- [ ] **Unit Tests**
    - [ ] Test Image Processing algorithms (Grayscale, Crop).
    - [ ] Test TFLite Interpreter wrapper.
- [ ] **Integration Tests**
    - [ ] Test Camera -> Face Mesh pipeline latency.
    - [ ] Test End-to-End VSR accuracy on sample videos.
- [ ] **Performance Profiling**
    - [ ] Measure Inference Time (ms).
    - [ ] Measure CPU/GPU/NPU Usage.
    - [ ] Measure Battery Drain.
    - [ ] Optimize Bitmap allocations (Object Pooling).

## Phase 7: Legal & Deployment

- [ ] **Legal Review**
    - [ ] Verify "Recording" indicator visibility.
    - [ ] Add "Consent" checkbox in onboarding flow.
    - [ ] Ensure no data persistence (files delete on exit).
- [ ] **Documentation**
    - [ ] Write USER_GUIDE.md.
    - [ ] Generate JavaDocs/KDocs.
