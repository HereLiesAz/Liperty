# Liperty

**Liperty** is a real-time, on-device Visual Speech Recognition (VSR) and Voice Reconstruction application for the Deaf, Hard-of-Hearing, and speech-impaired communities. It converts lip movements into text (lipreading) or reconstructed speech, running **fully offline after a one-time first-launch model download** to protect privacy.

> ⚠️ **Accuracy status:** The production lipreading model (SyncVSR) is **not yet validated on in-domain data.** The only evaluation run scored 100% WER on an out-of-distribution dataset (GRID 16-frame clips) — see [`docs/EVAL_RESULTS_2026-05-13.md`](docs/EVAL_RESULTS_2026-05-13.md). Real WER/CER on a SyncVSR-matched (full-utterance LRS3) test set is still pending. Treat lipreading output as experimental.

## What works today

* **Lipreading (VSR):** Camera → MediaPipe face mesh → 88×88 mouth ROI → **SyncVSR** ONNX (visual-only, LRS3-trained) → subword decoding → optional language-model + viseme-aware rescoring → text overlay.
* **Voice Reconstruction (BC/EL):** "Artificial Larynx" bone-conduction mode (phone vibration as a glottal carrier the user modulates silently) and electrolarynx-buzz capture, both through a shared DSP pipeline.
* **Multimodal laryngeal sensing:** contact-mic + accelerometer fusion for voice-activity detection.
* **Voice cloning (PocketTTS):** local ONNX-based personalized voice profiles.
* **Privacy-first:** all recognition/DSP/TTS runs on-device; biometric data (face mesh, lip motion) lives in RAM only. The sole persisted-biometric path is the explicitly-consented, user-deletable on-device personalization store.

## Planned / partial (see [`docs/TODO.md`](docs/TODO.md))

* Accessibility UI: tappable transcription word-blocks, pinch-to-zoom font scaling, hand-gesture (wave/air-swipe) control, confidence heatmap.
* On-device personalization (per-user encoder LoRA) — capture infra in place; training pipeline in progress.
* Bone-conduction microphone input + TRAMBA bandwidth expansion; electrolarynx noise-suppression voice conversion.
* Native C++ AAudio/Oboe low-latency audio + Bluetooth LE Audio (LC3) — not yet implemented.

## ML stack

* **ONNX Runtime Mobile** — production VSR (SyncVSR seq2seq encoder + attention decoder; CTC head as a low-RAM fallback) and PocketTTS voice cloning. CPU/XNNPACK by default.
* **MediaPipe Tasks Vision** — 468-landmark face mesh, hand landmarker.
* **OpenCV 4.13.0** (C++ via NDK) — lip-crop / image normalization.
* **LiteRT / TFLite** — legacy phoneme path + small auxiliary models only (not the production VSR).
* **KenLM** (NDK) + viseme-aware rescoring — n-gram + visual-confusability post-correction.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full technical breakdown and [`CLAUDE.md`](CLAUDE.md) for the maintained contributor guide.

## Setup & build

1. Clone the repository.
2. Run the initialization script (downloads the OpenCV Android SDK, MediaPipe `.task` files, ML models, and patches dependencies for AGP 9 / JDK 17):
   ```bash
   ./setup_libs.sh
   ```
   The large ML models are **pruned from the APK** and downloaded into app-private storage on first launch by `setup/ModelDownloadManager.kt` (~1.4 GB required; the app declares `INTERNET` solely for this).
3. Open in Android Studio (JDK 17), sync Gradle.
4. Connect a **physical Android device** (minSdk 26; emulators lack NPU/GPU-delegate and telephoto-lens support).
5. Build & run: `./gradlew assembleDebug`.

## Legal & privacy

Liperty processes biometric data (facial landmarks). See [`docs/LEGAL.md`](docs/LEGAL.md) (BIPA/GDPR/wiretap posture) and [`docs/PRIVACY_POLICY.md`](docs/PRIVACY_POLICY.md). The first-launch model download must be disclosed in the Play Store Data Safety form.

## Research

Architecture and methods: [`docs/RESEARCH_PAPER.md`](docs/RESEARCH_PAPER.md). Literature surveys: [`docs/RESEARCH.md`](docs/RESEARCH.md).
