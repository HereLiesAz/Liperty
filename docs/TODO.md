# Comprehensive Todo List & Implementation Roadmap

This document serves as the master source of truth for the Liperty project. It maps research findings to engineering tasks to ensure a production-ready Visual Speech Recognition (VSR) and Voice Reconstruction (BC/EL) application.

---

## 🔍 Phase 1-2: Core Infrastructure (Completed/Verified)

- [x] **Project Skeleton:** Kotlin 2.3.10, AGP 9.1.0, Version Catalog (`libs.versions.toml`).
- [x] **Permission Management:** Runtime CAMERA, RECORD_AUDIO, and BLUETOOTH_CONNECT permissions.
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
    - [ ] **Affine Transformation:** Matrix calculation for roll/pitch/yaw neutralization.
    - [ ] **Standardized Cropping:** 88x88 square cropping for model input.
    - [ ] **Zero-Allocation Pipeline:** Refine `BitmapPool` to ensure `ImageProxy` -> `ROI` conversion involves no GC churn.
- [ ] **Photometric Normalization:**
    - [ ] **Native Acceleration:** Gaussian Blur and Histogram Equalization implemented in JNI/OpenCV (`image_utils_jni.cpp`).

---

## 🧠 Phase 4: Machine Learning (The "Inference & Reasoning" Engine)

*Refs: RESEARCH.md "State-of-the-Art Neural Architectures for Mobile VSR"; AVHUBERT_V3_BACKEND.md; LM_RESCORING.md*

- [x] **Production VSR backend (V2):**
    - [x] Auto-AVSR ONNX (`Amanvir/LRS3_V_WER19.1`, ESPnet Conformer + CTC, 5050 SentencePiece tokens).
    - [x] Streaming inference via `FrameBuffer.slideAndGetFrames(retainCount=8)`.
    - [x] `SubwordCtcBeamDecoder` (beam=8, prefix merge via logsumexp).
- [x] **Research VSR backend (V3, off by default):**
    - [x] AV-HuBERT base+vox+433h encoder exported to ONNX (parity verified, 392 MB).
    - [x] AV-HuBERT seq2seq decoder exported to ONNX (parity vs PyTorch verified text-equivalent on 5/5 clips, 240 MB).
    - [x] Kotlin orchestrator (`AvHubertSeq2SeqInference` + `Seq2SeqGreedyDecoder` + `BpeDetokenizer`).
    - [x] Wired into MainActivity behind `USE_V3_BACKEND` flag; 18 unit tests pass.
    - [ ] On-device WER measurement vs V2.
    - [ ] Seq2Seq beam search + LM fusion for V3 (Phase A5).
- [x] **Phase A: LM rescoring + viseme-aware "Chaplin's second AI" (see LM_RESCORING.md)**:
    - [x] LibriSpeech 3-gram pruned 1e-7 → KenLM trie+q8 binary (27 MB), shipped via `setup_libs.sh`.
    - [x] `LanguageModelScorer` interface + `NoopLanguageModelScorer`.
    - [x] `SubwordCtcBeamDecoder` optional LM rescoring via `lmScorer` constructor param.
    - [x] Viseme map asset (9-class, Jeffers-Barley + Bear-Harvey).
    - [x] Viseme inverse index asset (126K cmudict words / 30K viseme sequences, 2.1 MB).
    - [x] `VisemeRescorer` with beam search over viseme-equivalent substitutions + input-bias tiebreaker.
    - [x] Wired in MainActivity post-CTC, pre-TranscriptionManager.
    - [ ] **`libkenlm.so` NDK build (gates all LM scoring effect — Phase A3b/c).**
    - [ ] Offline WER sweep V2-no-LM vs V2+KenLM vs V2+KenLM+VisemeRescorer (Phase A6).
- [ ] **Multi-view robustness:** pose-invariant feature extractors for off-axis (30°-60°) lipreading.
- [ ] **On-device personalization (see PERSONALIZATION.md):**
    - [x] **Step 1a: PairedTrainingRecord + PairedTrainingStore + 8 unit tests.**
    - [x] **Step 1b: VideoFrameExtractor (MediaMetadataRetriever-based).**
    - [ ] **Step 1c-f:** Android SpeechRecognizer transcript labels, voice-import hook, separate consent dialog, Settings UI for delete-all.
    - [x] **Step 3 PoC partial:** `tools/build_avhubert_training_artifacts.py`; ONNX export validated (382.9 MB). Blocker: `onnxruntime-training` package needs adding to v3-export docker.
    - [ ] **Step 3 on-device trainer:** Kotlin/JNI ORT Training Session, adapter-aware inference (post-PoC validation, ~6-8 weeks).
    - [ ] **Step 2:** Personal n-gram LM, personal viseme confusion matrix, auto-tuned hyperparameters (queued behind Step 1).
- [x] **Optional 2nd-stage LLM cleanup:** `LlmTextCleaner` wrapping Gemma-2B-it (MediaPipe Tasks GenAI). Opt-in.

---

## 📷 Phase 5: Hardware & Optical Optimization

- [ ] **Lens Selection Logic:** Prioritize Telephoto lens (2x/3x) when using the rear camera.
- [ ] **Optical Pacing & Locking:**
    - [ ] Enforce deterministic 25 FPS stream.
    - [ ] Implement Exposure/Focus lock during active inference.

---

## 🖐️ Phase 6: Accessibility-Driven UI/UX

- [ ] **Gesture-Driven Correction:**
    - [ ] **Touch Swipes:** Cycle through homophenes via word block interaction.
    - [ ] **Hand Gestures:** `WAVE_PAUSE` and `AIR_SWIPE` (Horizontal) for hands-free control.
- [ ] **Visual Feedback:**
    - [ ] **Direct Overlay:** Transcription rendered as tappable word blocks over camera view.
    - [ ] **Pinch-to-Zoom:** Dynamic font scaling (12sp–120sp) via Compose gestures.
- [ ] **Confidence Heatmap:**
    - [ ] Implement word-level background shading based on model softmax confidence.

---

## 🎙️ Phase 7: Voice Reconstruction (BC/EL Pipeline)

- [x] **BC Mode (Bone Conduction Larynx):**
    - [x] **Carrier Generation:** `GlottalCarrierGenerator` produces glottal pulse carrier (80–200 Hz) via `ArtificialLarynx`.
    - [x] **Audio Routing:** `AudioRouter` manages full-duplex (mic input + BC headphone output), forces built-in mic.
    - [x] **DSP Pipeline:** Spectral subtraction → frequency equalization → mel spectrogram → VoiceConverter → inverse mel.
- [x] **EL Mode (Electrolarynx Translator):**
    - [x] **Capture:** Built-in mic captures external EL buzz; same DSP pipeline (no carrier generation).
- [ ] **Future:**
    - [ ] Native C++ Oboe/AAudio for ultra-low-latency audio playback.
    - [ ] TRAMBA high-frequency bandwidth expansion model for BC input.

---

## 🗣️ Phase 9: Voice Management & Synthesis

- [ ] **Voice Cloning (Pocket TTS):**
    - [ ] **Import/Record UI:** Support for capturing samples or selecting existing `.wav` files.
    - [ ] **ONNX Boilerplate:** Infrastructure for executing Pocket-TTS models in `PocketTTSEngine.kt`.
    - [ ] **Real Inference:** Replace dummy embedding extraction and white-noise generation with actual model outputs.

---

## 🦴 Phase 8: Hardware Input & Pre-Processing (Bone Conduction)

*Ref: RESEARCH3.md Section "Hardware: Bone Conduction Microphones (BCMs) as Input Sensors"*

- [ ] **Integrate Bone Conduction Microphone (BCM) Support:**
    - [ ] Implement input routing to prioritize BCMs and head-worn accelerometers when connected.
- [ ] **Audio Bandwidth Expansion (Super-Resolution):**
    - [ ] Deploy a lightweight deep-learning model (e.g., TRAMBA architecture) to reconstruct high-frequency vocal components attenuated by tissue transmission.

---

## 🔕 Phase 10: Architecture 1 - Silent Speech Recognition (SSR)

*Ref: RESEARCH3.md Section "Architecture 1: Vibration-to-Text-to-Speech (Silent Speech Recognition)"*

- [ ] **SSR Translation Module:**
    - [ ] Integrate a CNN-based machine learning model to decode bone vibrations/sEMG into text strings.
- [ ] **Ultra-Low Latency On-Device TTS:**
    - [ ] Optimize the PocketTTS engine for streaming execution.
    - [ ] Achieve first-word audio latency targets of ~130 milliseconds.

---

## 🤖 Phase 11: Architecture 2 - Electrolarynx Translator

*Ref: RESEARCH3.md Section "Architecture 2: The Smartphone as an Electrolarynx Translator"*

- [ ] **Active Noise Suppression:**
    - [ ] Implement pitch-synchronous generalized spectral subtraction to dynamically estimate and subtract the mechanical EL buzz (self-noise) from captured speech in real-time.
- [ ] **Real-Time Voice Cloning (Intelligibility Enhancement):**
    - [ ] Deploy an advanced voice conversion neural network (e.g., Respeecher-style architecture) to map cleaned EL speech to a high-fidelity target voice.

---

## ⚡ Phase 12: Wireless Latency & Native Audio Stack

*Ref: RESEARCH3.md Section "Overcoming the Wireless Latency Bottleneck"*

- [ ] **Native C++ Audio Integration:**
    - [ ] Bypass the standard Android Java/Kotlin audio framework.
    - [ ] Implement AAudio or the Oboe C++ wrapper for audio playback and capture with MMAP buffers and low-latency performance modes.
- [ ] **Bluetooth LE Audio & LC3 Codec:**
    - [ ] Ensure the application is fully compatible with Bluetooth 5.2/5.3 LE Audio.
    - [ ] Leverage Isochronous Channels (ISOC) and the LC3/LC3plus codec to reduce wireless transmission latency to 10-15 ms.

---

## 🎯 Immediate Next Steps for AI Agent

In rough priority order (latest first):

1. [ ] **KenLM JNI / `libkenlm.so` NDK build.** Gates ALL LM scoring effect — the entire rescoring stack (Phase A) runs as no-op until this lands. ~4-8 hours of NDK work. See `docs/LM_RESCORING.md` § "Phase A3b/c: building libkenlm.so".
2. [ ] **Step 3 PoC unblock.** Add `onnxruntime-training` to `docker/v3-export/Dockerfile`, re-run `tools/build_avhubert_training_artifacts.py`, confirm the 4 ORT training artifacts generate cleanly for AV-HuBERT. ~1 hour.
3. [ ] **Step 1c-f: voice import hook + consent dialog + Settings UI.** Foundation for both Step 2 (statistical personalization) and Step 3 (encoder LoRA). See `docs/PERSONALIZATION.md` § "Step 1".
4. [ ] **Phase A6: offline WER sweep.** Once libkenlm.so works, measure V2-no-LM vs V2+KenLM vs V2+KenLM+VisemeRescorer on a held-out clip set. Pick α weight.
5. [ ] **V3 device validation.** Flip `USE_V3_BACKEND=true` on a debug build, run on a real device, compare V2 vs V3 WER on the same input.
6. [ ] **DSP refinement:** improve `voiceSourceExpansion` in `VibraPhoneDSP` using a non-linear excitation model instead of simple folding.

---

## 🔬 Phase 13: RESEARCH3 Implementation Plan

- [ ] **TRAMBA High-frequency bandwidth expansion model:** Restore attenuated high-frequency components from BCMs.
- [ ] **Silent Speech Recognition (SSR) CNN:** Decode non-auditory physiological signals into text.
- [ ] **Pitch-synchronous generalized spectral subtraction:** Suppress mechanical EL buzz and leakage noise via NDK.
- [ ] **On-device Streaming TTS:** Synthesize speech incrementally as the SSR engine outputs tokens.
- [ ] **Localized low-latency Voice Cloning:** Voice conversion neural network for high-fidelity target voices.
- [ ] **Native C++ AAudio / Oboe low-latency playback:** Write audio data directly via MMAP with exclusive low-latency mode.
- [ ] **Bluetooth LE Audio / LC3 Codec support:** Transmit high-fidelity data with ISOC channels.
