# Comprehensive Todo List & Implementation Roadmap

This document serves as the master source of truth for the Liperty project. It maps research findings to engineering tasks to ensure a production-ready Visual Speech Recognition (VSR) and Silent Speech Interface (SSI) application.

---

## 🔍 Phase 1-2: Core Infrastructure (Completed/Verified)

- [x] **Project Skeleton:** Kotlin 2.3.10, AGP 9.1.0, Version Catalog (`libs.versions.toml`).
- [x] **Permission Management:** Runtime CAMERA, RECORD_AUDIO, and VIBRATE permissions.
- [x] **Basic CameraX:** Implementation of Preview and ImageAnalysis with fixed front-camera default.
- [x] **Legal Foundation:** Consent dialog and ephemeral RAM-only processing logic.

---

## 🛠️ Phase 3: Computer Vision Pipeline (The "Stabilization & Normalization" Engine)

*Ref: RESEARCH.md Section "The Real-Time Computer Vision and Preprocessing Pipeline"*

- [x] **Dense Facial Landmark Tracking:** Integrate MediaPipe Face Mesh (468 landmarks).
- [x] **Lip ROI Isolation:** Map specific indices for mouth tracking.
- [x] **Software Video Stabilization:**
    - [x] **Kalman Filter:** Bounding box smoothing to counter hand jitter (Implemented in `KalmanFilter.kt`).
    - [ ] **Optical Flow:** Calculate inter-frame movement to maintain mouth centering.
- [ ] **Advanced Spatial Normalization:**
    - [x] **Affine Transformation:** Matrix calculation for roll/pitch/yaw neutralization.
    - [x] **Standardized Cropping:** 88x88 square cropping for model input.
    - [ ] **Zero-Allocation Pipeline:** Refine `BitmapPool` to ensure `ImageProxy` -> `ROI` conversion involves no GC churn.
- [x] **Photometric Normalization:**
    - [x] **Native Acceleration:** Gaussian Blur and Histogram Equalization implemented in JNI/OpenCV (`image_utils_jni.cpp`).

---

## 🧠 Phase 4: Machine Learning (The "Inference & Reasoning" Engine)

*Ref: RESEARCH.md Section "State-of-the-Art Neural Architectures for Mobile VSR"*

- [/] **Model Selection & Deployment:**
    - [ ] **VALLR Architecture (Production):** Integrate real Video Transformer and LLM Decoder.
    - [x] **Mock Engine:** Deployment of `vsr_model.tflite` (Dummy weights with correct I/O shapes).
- [ ] **Multi-View Robustness:**
    - [ ] Train/Integrate pose-invariant feature extractors for off-axis (30°-60°) lipreading.
- [/] **Advanced Decoding:**
    - [x] **CTC Beam Search:** Prefix merging implementation in `BeamSearchDecoder.kt`.
    - [ ] **Dynamic Language Model:** Replace placeholder `HomopheneCorrector` with a probabilistic dictionary.
- [/] **On-Device Personalization:**
    - [x] **Calibration UI:** "Tweak" flow for capturing user-specific articulatory patterns.
    - [x] **LoRA Infrastructure:** TFLite training signature support in `OnDeviceTrainer.kt`.
    - [ ] **Real Fine-Tuning:** Replace dummy label mapping with actual phoneme-to-index alignment.

---

## 📷 Phase 5: Hardware & Optical Optimization

- [x] **Lens Selection Logic:** Prioritize Telephoto lens (2x/3x) when using the rear camera.
- [ ] **Optical Pacing & Locking:**
    - [ ] Enforce deterministic 25 FPS stream.
    - [ ] Implement Exposure/Focus lock during active inference.

---

## 🖐️ Phase 6: Accessibility-Driven UI/UX

- [x] **Gesture-Driven Correction:**
    - [x] **Touch Swipes:** Cycle through homophenes via word block interaction.
    - [x] **Hand Gestures:** `WAVE_PAUSE` and `AIR_SWIPE` (Horizontal) for hands-free control.
- [x] **Visual Feedback:**
    - [x] **Direct Overlay:** Transcription rendered as tappable word blocks over camera view.
    - [x] **Pinch-to-Zoom:** Dynamic font scaling (12sp–120sp) via Compose gestures.
- [ ] **Confidence Heatmap:**
    - [ ] Implement word-level background shading based on model softmax confidence.

---

## 🎙️ Phase 7: Experimental Modality (Silent Speech Interface)

- [ ] **VibraPhone Implementation (Phase 2):**
    - [ ] Capture Back-EMF from LRA via NDK (Requires specialized hardware/drivers).
- [x] **Artificial Larynx:**
    - [x] **Carrier Generation:** Continuous high-intensity vibration using `VibratorManager` (Multi-motor support).
- [/] **Laryngeal Sensing Pipeline:**
    - [x] **Dual-Stream Sync:** Synchronized capture of contact-mic and accelerometer.
    - [x] **Multimodal VAD:** Accelerometer-gated audio processing.
    - [/] **DSP Reconstruction:** FFT and Equalization implemented; **Formant Extrapolation is currently a dummy spectral folding mockup.**

---

## 🗣️ Phase 9: Voice Management & synthesis

- [/] **Voice Cloning (Pocket TTS):**
    - [x] **Import/Record UI:** Support for capturing samples or selecting existing `.wav` files.
    - [x] **ONNX Boilerplate:** Infrastructure for executing Pocket-TTS models in `PocketTTSEngine.kt`.
    - [ ] **Real Inference:** Replace dummy embedding extraction and white-noise generation with actual model outputs.

---

## 🎯 Immediate Next Steps for AI Agent

1. [x] **PHONEME MAPPING:** Replace dummy 40-char vocabulary with the VALLR 38-phoneme set.
2. [x] **DSP REFINEMENT:** Improve `voiceSourceExpansion` in `VibraPhoneDSP` using a non-linear excitation model instead of simple folding.
3. [x] **POCKET-TTS INTEGRATION:** Implement the actual ONNX session execution for voice cloning and audio generation.
