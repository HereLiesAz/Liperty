# AGENTS.md

Guidance for AI agents (WARP, Claude, etc.) working in this repo.

> **The maintained, authoritative contributor guide is [`CLAUDE.md`](CLAUDE.md).** Read it first — it has the full architecture, conventions, pitfalls, and current model/backend facts. This file is a short orientation only.

## Project overview

**Liperty** is a real-time, on-device Visual Speech Recognition (VSR) and Voice Reconstruction app: it converts lip movements into text, or reconstructs speech via bone-conduction / electrolarynx pipelines.

### Key facts
- **Platform**: Android (Kotlin-first, C++/NDK for OpenCV + KenLM). minSdk 26.
- **Production VSR**: **SyncVSR** (visual-only, LRS3-trained), served by `OnnxModelEngine` via **ONNX Runtime Mobile** — seq2seq encoder + attention decoder by default, CTC head as a low-RAM fallback. Auto-AVSR is a legacy/alternate ONNX backend; the TFLite phoneme path is legacy/auxiliary only. (Backend selected by `MainActivity.VSR_BACKEND`.)
  - ⚠️ Accuracy is **not yet validated on in-domain data** (only eval = 100% WER on out-of-distribution GRID — see `docs/EVAL_RESULTS_2026-05-13.md`).
- **Vision**: MediaPipe face mesh (468 landmarks) + JNI/OpenCV 4.13.0 lip-crop/normalization.
- **Voice reconstruction / SSI**: multimodal contact-mic + accelerometer fusion; glottal-carrier BC mode + electrolarynx capture.
- **Privacy**: no cloud inference; biometric data is RAM-only except the explicitly-consented, user-deletable on-device personalization store. The app declares `INTERNET` solely for the one-time first-launch model download (`setup/ModelDownloadManager.kt`).

## Setup & build
```bash
./setup_libs.sh          # OpenCV SDK + MediaPipe .task + models; patches AGP 9 / JDK 17
./gradlew assembleDebug  # build (JDK 17)
./gradlew testDebugUnitTest
```
Large ML models are pruned from the APK and downloaded on first launch. Physical devices are required for real ML/sensor/vibrator testing.

## Coding guidelines
- **Backends are swappable via `ModelEngine`** — never touch a concrete engine from `VSRInference`. Wrap delegate init in try/catch with CPU fallback.
- **Zero-allocation** camera callbacks: reuse buffers (`BitmapPool`); use JNI for >30 Hz frame work.
- **Compose** UI; state in `ViewModel`/`TranscriptionManager` (StateFlow).
- **Legal/BIPA/GDPR**: never write raw frames, landmarks, or audio to persistent storage; never bypass the consent gate. `PrivacyTest` enforces this.

Consult [`docs/TODO.md`](docs/TODO.md) for what's shipped vs planned before starting a task.
