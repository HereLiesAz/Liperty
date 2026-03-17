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
    - [x] **Optical Flow:** Calculate inter-frame movement to maintain mouth centering.
- [x] **Advanced Spatial Normalization:**
    - [x] **Affine Transformation:** Matrix calculation for roll/pitch/yaw neutralization.
    - [x] **Standardized Cropping:** 88x88 square cropping for model input.
    - [x] **Zero-Allocation Pipeline:** Refine `BitmapPool` to ensure `ImageProxy` -> `ROI` conversion involves no GC churn.
- [x] **Photometric Normalization:**
    - [x] **Native Acceleration:** Gaussian Blur and Histogram Equalization implemented in JNI/OpenCV (`image_utils_jni.cpp`).

---

## 🧠 Phase 4: Machine Learning (The "Inference & Reasoning" Engine)

*Ref: RESEARCH.md Section "State-of-the-Art Neural Architectures for Mobile VSR"*

- [x] **Model Selection & Deployment:**
    - [x] **VALLR Architecture (Production):** Integrate real Video Transformer and LLM Decoder.
    - [x] **Mock Engine:** Deployment of `vsr_model.tflite` (Dummy weights with correct I/O shapes).
- [x] **Multi-View Robustness:**
    - [x] Integration of pose-invariant feature extractors for off-axis (30°-60°) lipreading.
- [x] **Advanced Decoding:**
    - [x] **CTC Beam Search:** Prefix merging implementation in `BeamSearchDecoder.kt`.
    - [x] **Dynamic Language Model:** Replace placeholder `HomopheneCorrector` with a probabilistic dictionary.
- [x] **On-Device Personalization:**
    - [x] **Calibration UI:** "Tweak" flow for capturing user-specific articulatory patterns.
    - [x] **LoRA Infrastructure:** TFLite training signature support in `OnDeviceTrainer.kt`.
    - [x] **Real Fine-Tuning:** Replace dummy label mapping with actual phoneme-to-index alignment.

---

## 📷 Phase 5: Hardware & Optical Optimization

- [x] **Lens Selection Logic:** Prioritize Telephoto lens (2x/3x) when using the rear camera.
- [x] **Optical Pacing & Locking:**
    - [x] Enforce deterministic 25 FPS stream.
    - [x] Implement Exposure/Focus lock during active inference.

---

## 🖐️ Phase 6: Accessibility-Driven UI/UX

- [x] **Gesture-Driven Correction:**
    - [x] **Touch Swipes:** Cycle through homophenes via word block interaction.
    - [x] **Hand Gestures:** `WAVE_PAUSE` and `AIR_SWIPE` (Horizontal) for hands-free control.
- [x] **Visual Feedback:**
    - [x] **Direct Overlay:** Transcription rendered as tappable word blocks over camera view.
    - [x] **Pinch-to-Zoom:** Dynamic font scaling (12sp–120sp) via Compose gestures.
- [x] **Confidence Heatmap:**
    - [x] Implement word-level background shading based on model softmax confidence.

---

## 🎙️ Phase 7: Experimental Modality (Silent Speech Interface)

- [x] **VibraPhone Implementation (Phase 2):**
    - [x] Capture Back-EMF from LRA via NDK (Requires specialized hardware/drivers).
- [x] **Artificial Larynx:**
    - [x] **Carrier Generation:** Continuous high-intensity vibration using `VibratorManager` (Multi-motor support).
- [x] **Laryngeal Sensing Pipeline:**
    - [x] **Dual-Stream Sync:** Synchronized capture of contact-mic and accelerometer.
    - [x] **Multimodal VAD:** Accelerometer-gated audio processing.
    - [x] **DSP Reconstruction:** FFT, Equalization, and Formant Extrapolation via LPC vocoder implemented.

---

## 🗣️ Phase 9: Voice Management & synthesis

- [x] **Voice Cloning (Pocket TTS):**
    - [x] **Import/Record UI:** Support for capturing samples or selecting existing `.wav` files.
    - [x] **ONNX Boilerplate:** Infrastructure for executing Pocket-TTS models in `PocketTTSEngine.kt`.
    - [x] **Real Inference:** Replace dummy embedding extraction and white-noise generation with actual model outputs.

---

## 🦴 Phase 8: Hardware Input & Pre-Processing (Bone Conduction)

*Ref: RESEARCH3.md Section "Hardware: Bone Conduction Microphones (BCMs) as Input Sensors"*

- [x] **Integrate Bone Conduction Microphone (BCM) Support:**
    - [x] Implement input routing to prioritize BCMs and head-worn accelerometers when connected.
- [x] **Audio Bandwidth Expansion (Super-Resolution):**
    - [x] Deploy a lightweight deep-learning model (e.g., TRAMBA architecture) to reconstruct high-frequency vocal components attenuated by tissue transmission.

---

## 🔕 Phase 9: Architecture 1 - Silent Speech Recognition (SSR)

*Ref: RESEARCH3.md Section "Architecture 1: Vibration-to-Text-to-Speech (Silent Speech Recognition)"*

- [x] **SSR Translation Module:**
    - [x] Integrate a CNN-based machine learning model to decode bone vibrations/sEMG into text strings.
- [x] **Ultra-Low Latency On-Device TTS:**
    - [x] Optimize the PocketTTS engine for streaming execution.
    - [x] Achieve first-word audio latency targets of ~130 milliseconds.

---

## 🤖 Phase 10: Architecture 2 - Electrolarynx Translator

*Ref: RESEARCH3.md Section "Architecture 2: The Smartphone as an Electrolarynx Translator"*

- [x] **Active Noise Suppression:**
    - [x] Implement pitch-synchronous generalized spectral subtraction to dynamically estimate and subtract the mechanical EL buzz (self-noise) from captured speech in real-time.
- [x] **Real-Time Voice Cloning (Intelligibility Enhancement):**
    - [x] Deploy an advanced voice conversion neural network (e.g., Respeecher-style architecture) to map cleaned EL speech to a high-fidelity target voice.

---

## ⚡ Phase 11: Wireless Latency & Native Audio Stack

*Ref: RESEARCH3.md Section "Overcoming the Wireless Latency Bottleneck"*

- [x] **Native C++ Audio Integration:**
    - [x] Bypass the standard Android Java/Kotlin audio framework.
    - [x] Implement AAudio or the Oboe C++ wrapper for audio playback and capture with MMAP buffers and low-latency performance modes.
- [x] **Bluetooth LE Audio & LC3 Codec:**
    - [x] Ensure the application is fully compatible with Bluetooth 5.2/5.3 LE Audio.
    - [x] Leverage Isochronous Channels (ISOC) and the LC3/LC3plus codec to reduce wireless transmission latency to 10-15 ms.

---

## 🎯 Immediate Next Steps for AI Agent

1. [x] **PHONEME MAPPING:** Replace dummy 40-char vocabulary with the VALLR 38-phoneme set.
2. [x] **DSP REFINEMENT:** Improve `voiceSourceExpansion` in `VibraPhoneDSP` using a non-linear excitation model instead of simple folding.
3. [x] **POCKET-TTS INTEGRATION:** Implement the actual ONNX session execution for voice cloning and audio generation.

---

## 🔬 Phase 10: RESEARCH3 Implementation Plan

- [x] **TRAMBA High-frequency bandwidth expansion model:** Restore attenuated high-frequency components from BCMs.
- [x] **Silent Speech Recognition (SSR) CNN:** Decode non-auditory physiological signals into text.
- [x] **Pitch-synchronous generalized spectral subtraction:** Suppress mechanical EL buzz and leakage noise via NDK.
- [x] **On-device Streaming TTS:** Synthesize speech incrementally as the SSR engine outputs tokens.
- [x] **Localized low-latency Voice Cloning:** Voice conversion neural network for high-fidelity target voices.
- [x] **Native C++ AAudio / Oboe low-latency playback:** Write audio data directly via MMAP with exclusive low-latency mode.
- [x] **Bluetooth LE Audio / LC3 Codec support:** Transmit high-fidelity data with ISOC channels.
