# Comprehensive Todo List & Implementation Roadmap

This document serves as the master source of truth for the Liperty project. It maps research findings to engineering tasks to ensure a production-ready Visual Speech Recognition (VSR) and Silent Speech Interface (SSI) application.

---

## 🔍 Phase 1-2: Core Infrastructure (Completed/Verified)

- [x] **Project Skeleton:** Kotlin 2.3.10, AGP 9.0.1, Version Catalog (`libs.versions.toml`).
- [x] **Permission Management:** Runtime CAMERA permission with ActivityResultContracts.
- [x] **Basic CameraX:** Implementation of Preview and ImageAnalysis with `STRATEGY_KEEP_ONLY_LATEST`.
- [x] **Legal Foundation:** Consent dialog and "Recording" indicator logic.

---

## 🛠️ Phase 3: Computer Vision Pipeline (The "Stabilization & Normalization" Engine)

*Ref: RESEARCH.md Section "The Real-Time Computer Vision and Preprocessing Pipeline"*

- [x] **Dense Facial Landmark Tracking:** Integrate MediaPipe Face Mesh (468 landmarks).
- [x] **Lip ROI Isolation:** Map specific indices (0, 13, 14, 17, 37, 39, 40, 61, 146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 317, 318, 321, 375, 402, 405, 409).
- [x] **Software Video Stabilization:**
    - [x] Implement Kalman Filter for bounding box smoothing to counter hand jitter (Implemented in `KalmanFilter.kt`).
    - [ ] Calculate inter-frame optical flow to maintain mouth centering.
- [ ] **Advanced Spatial Normalization:**
    - [x] Calculate Affine Transformation Matrix for roll/pitch/yaw neutralization.
    - [x] Perform 96x96 to 128x128 square cropping.
    - [ ] **Zero-Allocation Pipeline:** Refine `BitmapPool` to ensure `ImageProxy` -> `ROI` conversion involves no GC churn.
- [x] **Photometric Normalization:**
    - [x] Grayscale conversion.
    - [x] Gaussian Blur (Kernel 3x3/5x5) for sensor noise reduction (Native).
    - [x] Contrast Stretching / Histogram Equalization (Native).
    - [x] **Hardware Acceleration:** Migrate manual Kotlin loops in `ImageUtils` to OpenCV NDK (Implemented in `image_utils_jni.cpp`).

---

## 🧠 Phase 4: Machine Learning (The "Inference & Reasoning" Engine)

*Ref: RESEARCH.md Section "State-of-the-Art Neural Architectures for Mobile VSR"*

- [ ] **Model Selection & Deployment:**
    - [ ] **VALLR Architecture (Preferred):**
        - [ ] **Stage 1 (Video Transformer):** Predict 38 phonetic classes from lip ROIs.
        - [ ] **Stage 2 (LLM Decoder):** Reconstruct text from phoneme stream using contextual linguistic reasoning.
    - [ ] **Asset Management:** Deploy `vsr_model.tflite` (Encoder) and `decoder_model.tflite` (LLM) to assets.
- [ ] **Multi-View Robustness:**
    - [ ] Train/Integrate pose-invariant feature extractors (using MV-LRS dataset) for off-axis (30°-60°) lipreading.
- [ ] **Advanced Decoding:**
    - [x] CTC Beam Search implementation.
    - [ ] **Dynamic Language Model:** Replace placeholder `HomopheneCorrector` with a probabilistic dictionary and Bigram/Trigram scoring.
- [ ] **On-Device Personalization:**
    - [ ] Implement **Calibration Phase** (User mouths "The quick brown fox...").
    - [ ] Integrate **Low-Rank Adaptation (LoRA)** via LiteRT training signatures for speaker-adaptive fine-tuning.

---

## 📷 Phase 5: Hardware & Optical Optimization

*Ref: RESEARCH.md Section "Optical Physics, Perspective Distortion, and Camera Hardware Constraints"*

- [x] **Lens Selection Logic:** Programmatically prioritize Telephoto lens (2x/3x) over Wide-angle to minimize "moustache distortion."
- [ ] **Optical Pacing & Locking:**
    - [ ] Enforce deterministic 25 FPS stream for temporal consistency.
    - [ ] Implement Manual Exposure/Focus lock during active "Listening" state to prevent photometric shifting.

---

## 🖐️ Phase 6: Accessibility-Driven UI/UX

*Ref: RESEARCH.md Section "Gesture-Driven Homophene Correction Interface"*

- [x] **Gesture-Driven Correction:**
    - [x] **Horizontal Swipe:** Cycle through top-N homophene candidates for the selected word.
    - [x] **Vertical Swipe Up:** Trigger Text-to-Speech (TTS).
    - [x] **Multi-Finger Swipe Down:** Clear buffer/transcript.
- [x] **Visual Hand Gestures (Waving):**
    - [x] **Wave-to-Pause:** Implement hand wave detection in `FaceLandmarkerHelper` to act as a pause button for TTS and inference (Integrated in `MainActivity.kt`).
    - [ ] **Air-Swipe:** Map horizontal hand movement directions to homophene cycling (simulating touch swipes).
- [ ] **Visual Feedback:**
    - [x] Draw bounding boxes (Green=Face, Blue=Lips).
    - [ ] **Confidence Heatmap:** Color-code transcribed words based on model confidence (e.g., Red for high ambiguity).

---

## 🎙️ Phase 7: Experimental Modality (Silent Speech Interface)

*Ref: RESEARCH.md Section "Alternative Modalities: Hardware-Native Laryngeal Sensing"*

- [ ] **VibraPhone Implementation (Prototype):**
    - [ ] Capture Back-EMF from Linear Resonant Actuators (LRA) via NDK.
    - [x] **Artificial Larynx:** Continuous vibration mode to provide a sound source for mute users (Implemented in `ArtificialLarynx.kt`).
    - [x] **DSP Pipeline:** Spectral Subtraction, Frequency Equalization, and Formant Extrapolation (Implemented in `VibraPhoneDSP.kt`).
    - [x] **Real-time Laryngeal Sensing:** Dual-stream capture of contact-mic and accelerometer (Implemented in `LaryngealSensor.kt`).
    - [x] **Multi-Modal Fusion:** Use Accelerometer (0-400Hz) for high-precision VAD to gate the contact-mic stream.

---

## ⚖️ Phase 8: Privacy & Legal Governance

*Ref: RESEARCH.md Section "Legal, Ethical, and Privacy Governance"*

- [x] **Ephemeral Processing:** Ensure zero-persistence of biometric data (RAM-only processing).
- [x] **Visual Notice:** "Live Transcription Active" banner on interlocutor-facing screen.
- [ ] **KDoc/Security Audit:** Fully document biometric handling for compliance with BIPA/GDPR.

---

## 🎯 Immediate Next Steps for AI Agent

1. **PRODUCE MODEL:** You cannot proceed without `vsr_model.tflite`. Either locate the model or provide a script to generate a dummy TFLite with the correct I/O shapes [1, 50, 88, 88, 1] -> [1, 50, 40] to allow pipeline testing.
2. **NDK INTEGRATION:** Create a JNI layer for OpenCV to handle the high-frequency image normalization in C++.
3. **EXPAND HOMOPHENES:** Transform `HomopheneCorrector` into a JSON-backed dictionary covering the top 5,000 English words.
