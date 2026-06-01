# On-Device Personalization

This document covers Liperty's plan to make the visual ASR pipeline adapt to each user, on-device only. The motivating problem isn't engineering — it's that the pretrained encoder (SyncVSR in production; the Step-3 LoRA proof-of-concept below was prototyped on AV-HuBERT) was trained on demographically biased corpora (LRS3 = TED talks, VoxCeleb2 = YouTube celebrity clips) that systematically under-represent the population Liperty exists to serve. The paired-recording capture and crop constants (88×88, mean 0.421, std 0.165) are backend-shared, so the captured data applies regardless of which encoder is personalized.

---

## The bias problem (why this work matters)

LRS3 and VoxCeleb2 skew hard toward:
- White, English-native, "broadcast standard" articulation
- Speakers without speech impairments
- Speakers without atypical mouth morphology
- WEIRD (Western/Educated/Industrialized/Rich/Democratic) populations

Liperty's actual user base is, by design, the population least represented in this training data:
- Deaf and Hard-of-Hearing speakers (whose own articulation may differ from hearing speakers — their auditory feedback loop is different)
- People with speech impairments (dysarthria, ALS, post-stroke, cerebral palsy)
- Speakers across the global demographic distribution rather than the Anglosphere-academic subset

For these users, the encoder's representations themselves are worse — not just downstream layers. An LM rescorer or viseme rescorer (see [`LM_RESCORING.md`](LM_RESCORING.md)) can recover from word-choice mistakes, but it can't recover from "the encoder literally can't see this user's mouth as well as it sees a TED speaker's."

**The fix is per-user fine-tuning, and Liperty's privacy posture forces it on-device.**

---

## The free training data we didn't realize we had

Voice cloning is already a feature: the user uploads or records video of themselves talking, and Liperty builds a TTS voice profile from the audio track. The same recording contains synchronized lip motion — exactly the kind of paired (lip-video, transcript) data that's normally so hard to obtain that the entire rest of the research log has been working around its absence.

**One recording session → three artifacts:**
1. Voice clone model (original purpose).
2. Per-user lipreading personalization data (this document's subject).
3. Eventually, a personal visual-encoder LoRA (Step 3 below).

The user pays the recording cost once; the data feeds multiple downstream uses.

---

## Three layers of personalization

In order of increasing engineering cost. Earlier layers are useful even if later ones are never built; later ones build on the data infrastructure of earlier ones.

### Step 1 — Recording capture infrastructure (foundation)

What it does: harvest synchronized (audio, lip-cropped video, optional ASR transcript) tuples from the voice cloning import pipeline. Persist on-device. Make user-deletable.

This is the foundation Steps 2 and 3 both need. **Doesn't itself improve WER** — it just produces the dataset.

| Component | File | Status |
|---|---|---|
| Paired-record data class | [`PairedTrainingRecord.kt`](../app/src/main/java/com/hereliesaz/liperty/personalization/PairedTrainingRecord.kt) | ✓ shipped |
| On-disk store with delete-all controls | [`PairedTrainingStore.kt`](../app/src/main/java/com/hereliesaz/liperty/personalization/PairedTrainingStore.kt) | ✓ shipped (8 unit tests pass) |
| Video frame extractor (MediaMetadataRetriever) | [`VideoFrameExtractor.kt`](../app/src/main/java/com/hereliesaz/liperty/personalization/VideoFrameExtractor.kt) | ✓ shipped |
| Android SpeechRecognizer integration for transcript labels | (planned) | pending |
| Hook into `VoiceViewModel.startImportProcessing` | (planned) | pending |
| Consent dialog (separate from app-launch consent gate) | (planned) | pending |
| Settings entry to view/delete stored training data | (planned) | pending |

### Step 2 — Cheap statistical personalization (no gradient descent)

Once Step 1 produces recordings, three quick wins land immediately:

1. **Personal n-gram LM.** Train a KenLM on the user's own utterances (voice-clone prompts they read + corrections during normal use). Interpolate with the LibriSpeech LM at runtime: `final = (1-λ) · LibriSpeech(words) + λ · Personal(words)`. KenLM `lmplz` runs on a phone in milliseconds for a 1000-sentence corpus. Personal LM is ~1-5 MB.

2. **Personal viseme confusion matrix.** From paired (lip-motion, audio-derived phoneme) data, measure which viseme classes THIS user actually conflates. The shipped 9-class viseme map is a population average; emit a personalized `viseme_map.txt` per user. ~5 KB on disk.

3. **Auto-tuned hyperparameters.** Per-user sweet spots for `KENLM_WEIGHT` (α), viseme `candidatesPerWord`, confidence thresholds. Measure WER on held-out clips from the recording session, grid-search the knobs, pick best. ~1 KB JSON.

All three are pure-statistics — no backprop, no training-framework dependencies. Each is days of work, not weeks.

### Step 3 — Encoder LoRA fine-tune (the bias fix)

Train a LoRA adapter on the visual encoder using the user's paired recordings. This is the layer that actually addresses the encoder-bias problem; Step 2 is downstream-only and can't fix encoder representation gaps.

**Constraint:** off-device training is not an option (privacy + UX). Therefore the architecture must do gradient descent on the phone.

**The path that's real (not hypothetical):**

| Stage | Where | What |
|---|---|---|
| One-time build prep | Build server (NOT per-user) | Take base AV-HuBERT, add LoRA modules to attention layers, export ONNX Runtime training artifacts. Bundle in APK. |
| Recording session | On device | User records paired data via Step 1 infrastructure. |
| Training | On device | Load training artifacts, run gradient descent on user's data. Output: LoRA adapter weights (~3-5 MB). Overnight while charging. |
| Inference | On device | Load base encoder + personal LoRA adapter at runtime. Encoder runs personalized. |

**The on-device training pipeline uses [ONNX Runtime On-Device Training](https://onnxruntime.ai/docs/get-started/training-on-device.html).** Validated by Microsoft's [MobileTransformers](https://martinkorelic.github.io/mobiletransformers-docs/) project running 500M-1B parameter LLM fine-tunes on a Pixel 6. AV-HuBERT base (95M params) is well within the demonstrated envelope.

**Build-time tool (validated end-to-end on Kaggle, 2026-05-12):**

[`tools/kaggle_build_training_artifacts.py`](../tools/kaggle_build_training_artifacts.py) — paste-driven Kaggle cell that:

1. Pulls the already-published encoder ONNX from `HereLiesAz/liperty-avhubert-encoder` (skips fairseq entirely — no Docker needed).
2. Modifies the ONNX graph: adds a trainable Linear adapter (W: 768×768 identity-init, b: 768 zero-init) as new initializers + MatMul/Add nodes, prunes the 4 diagnostic outputs from V3 parity debugging, swaps the primary output for `adapted_features`.
3. Calls `onnxruntime.training.artifacts.generate_artifacts(...)` with `requires_grad=["adapter.W", "adapter.b"]`, `loss=MSELoss`, `optimizer=AdamW`.
4. Uploads the 4 artifacts to `HereLiesAz/liperty-v3-training-artifacts`.

**PoC outcome — the key question is answered:** ORT On-Device Training accepts AV-HuBERT's operator set. The graph-transformer trace during artifact generation shows the standard ORT optimization passes (ConstantSharing, LayerNormFusion, GeluFusion, BiasGeluFusion, ReshapeFusion) ran cleanly over the entire encoder. The 4 artifacts are live at [`HereLiesAz/liperty-v3-training-artifacts`](https://huggingface.co/HereLiesAz/liperty-v3-training-artifacts) and ready for the on-device half of the pipeline.

Artifact sizes:
- `training_model.onnx` — 410.6 MB (forward + loss + gradient subgraph)
- `eval_model.onnx` — 410.8 MB
- `optimizer_model.onnx` — 538 B (AdamW step on only 2 trainable params is tiny)
- `checkpoint` — 2.4 MB
- `nominal_checkpoint` — slim per-param state

**Five issues solved during the PoC iteration**, now baked into the script:
1. PyPI has no `onnxruntime-training` wheels for Python 3.12 (Kaggle's default) → bootstrap a Python 3.10 venv via `uv` and install there.
2. uv venvs ship without pip → use `uv pip install --python ...`.
3. `onnxruntime.training` has an import-time dependency on `torch` → install it alongside the other deps.
4. The published encoder ONNX exposes 5 diagnostic outputs (`features`, `feat_a`, `feat_v`, `feats_ln`, `feats_proj`) from the V3 parity-debugging phase → prune all but `features` and pass `loss_input_names=["adapted_features"]` to `generate_artifacts`.
5. `onnxblock` writes intermediate temp files with `save_as_external_data=True` hardcoded; for a 413 MB model that produces a `temp.onnx` + missing `temp.onnx.data` and the validator fails → monkey-patch `onnx.save_model` to force `save_as_external_data=False` process-wide. Safe because the model is under the 2 GB protobuf limit.

**Memory/time budget on phone (estimated, not yet measured):**
- AV-HuBERT base: 95M params, 392 MB fp32
- LoRA rank-8 on attention layers: ~600K trainable params, ~3 MB
- Activation memory per 50-frame clip: ~5 MB
- Optimizer state (Adam): 2× LoRA params, ~6 MB
- Peak training memory: ~50-100 MB beyond model weights — well within phone RAM
- Training time: 30 min – 2 hours on a mid-range phone for ~1000 steps. "Overnight while charging" covers it.

---

## Privacy posture

Per [`docs/LEGAL.md`](LEGAL.md) and [`docs/PRIVACY_POLICY.md`](PRIVACY_POLICY.md), Liperty's default is RAM-only processing; biometric data (face meshes, lip landmarks, lip motion samples) is never persisted. The personalization feature **requires** persisting biometric data on-device for training and therefore needs explicit additional consent, separate from the app-launch consent gate.

Mandatory invariants:

1. **Separate consent flow** specific to this feature. Different decision from "use the app." User can deny personalization while still using Liperty.
2. **On-device only.** No biometric data ever leaves the device. Step 3's training is on-device for this reason; the build-time artifact generation does NOT touch user data.
3. **Physical deletion** via Settings. `PairedTrainingStore.deleteAll()` unlinks the files; the directory remains so the system can rebuild on next opt-in.
4. **Retention defaults** to "keep until user deletes" but the UI must surface storage size and let the user delete subsets (per-source URI, per-date-range, or all).
5. **No biometric data in logs.** Diagnostic logging may reference counts and timings but never frames or transcripts.
6. **Per-user adapter is encoded.** The trained LoRA weights are an information-theoretic compression of the user's lip motion; not raw biometrics. Still treated as user data, but with a lower privacy bar than raw frames.

---

## Roadmap & sequencing

The user has gated this on the constraint "off-device is not an option" and the framing "Step 1 + Step 3 in parallel; Step 2 starts when Step 1 is done."

Recommended order:

```
                  recording capture infrastructure (Step 1)
                  ┌───────────────────────────────────────┐
                  ▼                                       ▼
      Step 2 (statistical personalization)    Step 3 (encoder LoRA)
      ┌───────────────────────────────────┐   ┌────────────────────────┐
      ├── personal n-gram LM               │   ├── build training artifacts
      ├── personal viseme confusion matrix │   │   (off-device script, validated PoC)
      └── auto-tuned hyperparameters       │   ├── on-device trainer
      ▼                                    │   │   (ORT Training Session on Android)
      Ship as soon as ready                │   └── adapter-aware inference
                                           │       (load base + LoRA at runtime)
                                           ▼
                                           Ship after on-device WER beats baseline
```

Step 2 lands incrementally on top of the recording infra; users get value before Step 3 completes. Step 3 is the bigger commitment (~6-8 weeks) and is the layer that actually fixes the bias problem — but Step 2's wins are real and worth shipping first.

---

## Current state of progress

| Item | Status |
|---|---|
| Step 1: `PairedTrainingRecord` data class | ✓ shipped |
| Step 1: `PairedTrainingStore` (on-disk persistence, 8 unit tests) | ✓ shipped |
| Step 1: `VideoFrameExtractor` (MediaMetadataRetriever) | ✓ shipped |
| Step 1: Android SpeechRecognizer transcript labels | pending |
| Step 1: hook into `VoiceViewModel.startImportProcessing` | pending |
| Step 1: separate consent dialog | pending |
| Step 1: Settings UI for view/delete | pending |
| Step 3 build-time: training-artifact generator on Kaggle | **✓ validated end-to-end** — see below |
| Step 3 on-device: Kotlin TrainingSession + adapter-aware inference | pending |
| Step 2: personal n-gram LM | queued behind Step 1 |
| Step 2: personal viseme confusion matrix | queued behind Step 1 |
| Step 2: auto-tuned hyperparameters | queued behind Step 1 |
