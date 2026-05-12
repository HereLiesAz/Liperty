# System Architecture

Liperty is a real-time, on-device Visual Speech Recognition (VSR) and Voice Reconstruction app. Two operational requirements drive the architecture:

1. **Low-latency inference** — sub-200 ms end-to-end from camera frame to transcribed text.
2. **Absolute privacy** — biometric data (face mesh, lip motion, raw audio) lives in RAM only by default. The opt-in personalization feature is the only path that persists biometric data, with a separate consent flow.

---

## 1. Vision & VSR Pipeline (production: Auto-AVSR ONNX)

### Camera capture

`CameraManager` defaults to the front camera. For the rear camera it queries `CameraCharacteristics` and prefers longer focal lengths (telephoto) to minimize perspective distortion — never hardcoded `LENS_FACING_BACK`.

### Frame preprocessing (NDK + Kotlin)

- **Face landmarks** — `FaceLandmarkerHelper` wraps MediaPipe's Face Mesh (468 landmarks).
- **Lip ROI** — `extractLipBoundingBox()` derives a stabilized lip bounding box, smoothed by `RectKalmanFilter` against hand jitter.
- **Crop + normalize** — `ImageUtils.alignAndCropMouth()` (JNI/OpenCV) extracts 88×88 grayscale mouth ROIs and applies per-channel normalization `(pixel/255 - mean) / std` where mean=0.421, std=0.165 (Chaplin's Auto-AVSR pipeline).
- **Buffering** — `FrameBuffer` holds a rolling window. `slideAndGetFrames(retainCount = 8)` yields the last N frames + retains the tail seed for streaming inference, so each new frame is seen by exactly two inference windows.

### Inference (ONNX Runtime — primary)

`OnnxModelEngine` runs the Auto-AVSR Conformer + CTC head (`autoavsr_lrs3_visual_ctc.onnx`), pulled by `setup_libs.sh` from `HereLiesAz/liperty-autoavsr-onnx`. Input shape `(1, 1, T, 88, 88)` NCTHW float32; output `(1, T_out, 5050)` log-softmax over SentencePiece subword tokens. `XNNPACK` CPU provider by default.

`TFLiteEngine` is the legacy phoneme path. Retained for experiments and small auxiliary models, not the production VSR.

### Decoding

`SubwordCtcBeamDecoder` (beam width 8) over the 5050-token vocab parsed by `VocabularyLoader` from `unigram5000_units.txt`. Implements CTC prefix-merging via logsumexp.

### Rescoring stack

Two layers stacked after the CTC beam:

1. **LM rescoring inside the CTC beam.** When `lmScorer` is non-null, the surviving top-K beams are re-ranked by `CTC_logprob + lmWeight · LM.score(words)` before the winner is picked.
2. **Viseme-aware post-rescoring.** `VisemeRescorer` consumes the best-beam sentence, expands each word to viseme-equivalent candidates (cmudict-derived 9-class viseme groupings), beam-searches over per-position substitutions scored by the same LM, and replaces words where the LM strictly prefers a visually-confusable alternative.

Both layers are wired and unit-tested; the LM scoring is gated on `KenLmScorer.isNativeLoaded` (requires `libkenlm.so` packaged in the APK — NDK build pending). See [`LM_RESCORING.md`](LM_RESCORING.md) for the architecture and current build state.

### Optional 2nd-stage LLM cleanup

`LlmTextCleaner` wraps a small on-device LLM (Gemma-2B-it via MediaPipe Tasks GenAI). Opt-in (the model file isn't bundled in the APK — ~1.3 GB). When configured, `TranscriptionManager.transformSentence` post-cleans the rescored output.

### Pipeline summary

```
Camera frame (25–30 FPS)
  → MediaPipe landmarks
  → lip-ROI crop + 88×88 grayscale + normalize
  → FrameBuffer (sliding window, retain 8 of 16)
  → OnnxModelEngine (Auto-AVSR encoder + CTC) → (1, T_out, 5050)
  → SubwordCtcBeamDecoder (beam 8, optional LM rescoring inside beam)
  → VisemeRescorer (viseme-equivalent substitution + LM rescoring)
  → optional LlmTextCleaner
  → TranscriptionManager → OverlayView / Compose UI
```

### V3 research backend (off by default)

`MainActivity.USE_V3_BACKEND = false`. Flipping it swaps the entire CTC pipeline above for `AvHubertSeq2SeqInference`:

```
Camera frame
  → preprocessing (identical to V2)
  → AvHubertEncoderSession (AV-HuBERT base+vox+433h encoder ONNX) → (1, T, 768) features
  → AvHubertDecoderSession + Seq2SeqGreedyDecoder
      (autoregressive over 1000-token SentencePiece BPE, BOS → argmax → EOS)
  → BpeDetokenizer (▁ → space)
  → VisemeRescorer (same post-rescoring as V2)
  → TranscriptionManager
```

Status: code shipped + 18 unit tests pass. Not validated on device. Research log: [`AVHUBERT_V3_BACKEND.md`](AVHUBERT_V3_BACKEND.md).

---

## 2. Voice Reconstruction (BC/EL) Pipeline

Unchanged from prior versions. See `app/src/main/java/com/hereliesaz/liperty/voicebox/` for the implementation and `docs/RESEARCH3.md` for the research basis.

- **BC mode:** `GlottalCarrierGenerator` produces a glottal pulse carrier (80-200 Hz) routed to BC headphones via `ArtificialLarynx` + `AudioRouter`. Phone built-in mic captures the modulated result.
- **EL mode:** Captures external electrolarynx buzz via the phone mic. No carrier generation.
- **Shared DSP** in `VibraPhoneDSP`: spectral subtraction, EQ (300 Hz – 3.5 kHz), Mel-spectrogram → `VoiceConverter` → inverse Mel.
- **Routing** in `AudioRouter`: full-duplex configuration; forces phone built-in mic in BC mode to avoid carrier capture.

---

## 3. Voice Management

- **PocketTTS Engine** (`PocketTTSEngine`) executes ONNX voice cloning models locally.
- **VoiceStore** persists voice profiles (JSON + binary embeddings).
- **Voice import** flow (`VoiceImportWizardScreen` + `VoiceViewModel`) accepts both audio and video files; the audio is segmented via `AudioPreprocessor`'s VAD and clustered by speaker.

When the user opts into personalization (see [`PERSONALIZATION.md`](PERSONALIZATION.md)), the voice import flow also harvests synchronized lip-cropped video frames into [`PairedTrainingStore`](../app/src/main/java/com/hereliesaz/liperty/personalization/PairedTrainingStore.kt) for downstream visual-encoder personalization. Stored data is biometric, is gated on a separate consent flow, and is user-deletable from Settings.

---

## 4. On-Device Personalization (in progress)

Three layers, all on-device:

1. **Recording capture infrastructure** — harvest paired (audio, video, optional transcript) from voice cloning sessions.
2. **Statistical personalization** — personal n-gram LM, viseme confusion matrix, auto-tuned hyperparameters.
3. **Encoder LoRA fine-tune** — ONNX Runtime On-Device Training of a LoRA adapter on top of the frozen AV-HuBERT base encoder.

Bias rationale and full details: [`PERSONALIZATION.md`](PERSONALIZATION.md).

---

## Hardware Acceleration Strategy

| Layer | Acceleration |
|---|---|
| MediaPipe landmarks | GPU delegate (MediaPipe-managed) |
| ONNX Runtime (Auto-AVSR encoder) | XNNPACK CPU (primary); NNAPI/Hexagon attempted but device-dependent |
| TFLite (auxiliary models) | GPU delegate → CPU fallback via `try-catch` cascade |
| OpenCV (lip crop, NDK) | SIMD via OpenCV native builds |
| KenLM (when shipped) | CPU memory-map; mostly memory-bound, fast |

---

## Technical Stack

| Layer | Tech |
|---|---|
| Language | Kotlin (primary) / C++ (NDK for OpenCV, eventually KenLM) |
| ML inference | ONNX Runtime Mobile (primary), LiteRT/TFLite (legacy), MediaPipe Tasks Vision |
| ML training (planned) | ONNX Runtime On-Device Training (Android, AAR) |
| UI | Jetpack Compose + Material 3, custom `AzNavRail` |
| Concurrency | Kotlin Coroutines + StateFlow |
| Sensors | Camera2 / CameraX, Android Sensor Framework, Vibrator API |
| Audio | AudioTrack / AudioRecord + AAudio (planned for low-latency) |

---

## Module map (high level)

| Package | Responsibility |
|---|---|
| `liperty.camera` | CameraX wrapper + lens-selection heuristics |
| `liperty.ml` | All VSR-related ML: engines, decoders, rescorers, vocabulary, V3 seq2seq components |
| `liperty.personalization` | On-device per-user training infrastructure (paired-recording store, frame extractor) |
| `liperty.voicebox` | Voice reconstruction, voice cloning, audio routing, BC/EL hardware |
| `liperty.voicebox.cloning` | Voice clone pipeline (VAD, speaker clustering, profile persistence) |
| `liperty.dsp` | Signal processing primitives (VibraPhoneDSP, glottal carrier) |
| `liperty.ui` | Jetpack Compose UI + overlay rendering |
| `liperty.utils` | Image helpers, BitmapPool, performance monitor |

---

## Related documents

- [`RESEARCH_PAPER.md`](RESEARCH_PAPER.md) — **the system paper.** Comprehensive description of architecture, methods, experimental results (including the GRID negative result), limitations, and future work. Start here for the full picture.
- [`AVHUBERT_V3_BACKEND.md`](AVHUBERT_V3_BACKEND.md) — research log for the V3 seq2seq backend (chronological, 10+ attempts).
- [`LM_RESCORING.md`](LM_RESCORING.md) — KenLM + viseme-aware rescoring architecture and current build state.
- [`PERSONALIZATION.md`](PERSONALIZATION.md) — on-device per-user training plan.
- [`PRIVACY_POLICY.md`](PRIVACY_POLICY.md) — privacy contract and consent flows.
- [`LEGAL.md`](LEGAL.md) — BIPA/GDPR posture for biometric data.
- [`TODO.md`](TODO.md) — full roadmap.
