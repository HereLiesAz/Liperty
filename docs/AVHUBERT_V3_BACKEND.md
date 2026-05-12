# AV-HuBERT (V3) Backend — Research Plan

This document captures the in-progress research path for swapping
Liperty's deployed visual-speech encoder from **Auto-AVSR (V2, ESPnet
visual-only Conformer + CTC, 19.1% headline WER on LRS3)** to
**AV-HuBERT large (V3, Meta's audio-visual self-supervised encoder,
trained on LRS3 + VoxCeleb2)**.

**Status: research-only. Not deployed. Not on a release timeline.**
The Auto-AVSR backend stays the production path until V3 demonstrates
a concrete WER improvement on real Liperty input on real hardware.

> **Update 2026-05-12.** The seq2seq orchestrator originally built for
> V3 (`AvHubertSeq2SeqInference`) has been generalized: it now also
> drives the **SyncVSR** seq2seq path (KAIST-AILab Vox+LRS2+LRS3,
> NTCHW input, attention decoder exported alongside the encoder by
> `tools/syncvsr_export_stage3_decoder.py` + `..._stage4_encoder.py`).
> Selected by `MainActivity.SYNCVSR_USE_SEQ2SEQ`; falls back to the
> SyncVSR CTC path if the new ONNX files aren't present. The class
> name still says "AvHubert" because moving it would churn imports
> across the codebase without changing behavior — the rename is a
> follow-up task.

---

## Why bother

- **Bigger pretraining corpus.** AV-HuBERT large was self-supervised on
  LRS3 (~433h) + VoxCeleb2 (~2400h, English subset). Auto-AVSR's
  encoder saw only LRS3.
- **Better downstream WER on paper.** [LMD-VSR (ICCV 2023)](
  https://github.com/JeongHun0716/lmd-vsr) reports **12.6% WER** on
  English LRS2 test using AV-HuBERT large + an LMDecoder, vs Auto-AVSR's
  19.1% headline.
- **Public weights.** Both `base_vox_iter5.pt` and `large_vox_iter5.pt`
  are downloadable from `dl.fbaipublicfiles.com/avhubert/...` — no
  academic registration required to obtain the encoder, even though
  re-training would need LRS3 access.

## Why this is a research project, not a 1-week swap

1. **Stack mismatch.** AV-HuBERT runs on `fairseq`, Meta's research
   framework. Exporting fairseq transformers to ONNX is fragile —
   custom layer norm placements, dynamic shapes, AMP-only ops. Auto-AVSR's
   ESPnet export was already painful; AV-HuBERT is harder.

2. **The 12.6% WER assumes the LMDecoder runs at inference.** That's a
   second fairseq transformer trained on LRS2+LRS3 transcripts. Without
   it, the encoder alone is closer to 25-30% WER, which is no improvement
   over Auto-AVSR. Both encoder *and* decoder need to ONNX-export and
   ship on-device for V3 to beat V2.

3. **Architecture incompatibility with the existing pipeline.**
   - Auto-AVSR: 88×88 grayscale, `pixelMean=0.421, pixelStd=0.165`
   - AV-HuBERT: 96×96 grayscale, mouth ROI cropped via the AV-HuBERT
     mean-face alignment (see [`avhubert/preparation/align_mouth.py`](
     https://github.com/JeongHun0716/lmd-vsr/tree/main/avhubert/preparation))
   - `MainActivity.AUTOAVSR_*` constants and the `FrameBuffer` crop
     pipeline both need to change for V3.

4. **No way to validate WER without LRS2/LRS3 test data.** Both are
   academically gated. Until the user has access, V3 is flying blind on
   whether the expected WER actually materializes for Liperty's
   deployment context.

5. **License obligation.** The AV-HuBERT weights are licensed
   non-commercial research only. Liperty is a research/accessibility
   project, which qualifies, but the LICENSE file must ship with any
   redistribution and Liperty's own LICENSE/about-screen must
   acknowledge Meta as the upstream author.

---

## Pieces already in hand

| Piece | Where | Status |
|---|---|---|
| `large_vox_iter5.pt` (3.91 GB) | `HereLiesAz/liperty-avhubert-encoder` on HF (public mirror) | Done |
| Original Meta URL | https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/clean-pretrain/large_vox_iter5.pt | n/a |
| AV-HuBERT license | bundled in the HF mirror as `LICENSE.txt` | Done |
| Reference mouth-alignment pipeline | https://github.com/JeongHun0716/lmd-vsr/tree/main/avhubert/preparation (`align_mouth.py` + mean face) | external |
| LMDecoder (LRS2+LRS3 trained, English, 12.6 WER) | https://www.dropbox.com/scl/fo/zxnycpjlffd18ok5bg7ob/AKwd8lxvbx_q_BECGnTI2Pc | not mirrored yet |
| LMD-VSR full English VSR (LRS2 only, 23.8 WER) | https://www.dropbox.com/scl/fo/60xihdj518w44ujnixp8p/AKhdf0TxhPL5MLjQLtX8zdc | not mirrored yet |

## Pieces NOT yet in hand

- Working ONNX export of the AV-HuBERT encoder
- Working ONNX export of the LMDecoder
- Adapted Liperty preprocessing pipeline that produces 96×96 mouth ROIs
  via the AV-HuBERT mean-face alignment
- Android `AvHubertModelEngine.kt` implementing `ModelEngine`
- Eval pipeline that scores the V3 backend on a held-out set

---

## Attempt log

### 2026-05-12 (eleventh): Step 3 PoC end-to-end success on Kaggle

After the docker-based attempt at `tools/build_avhubert_training_artifacts.py` got blocked on `onnxruntime-training` pip-pin issues against the 2022 nvcr.io base image, the restructured Kaggle-native version succeeded end-to-end. See [`tools/kaggle_build_training_artifacts.py`](../tools/kaggle_build_training_artifacts.py) for the working script and [`docs/PERSONALIZATION.md`](PERSONALIZATION.md) for the five issues we iterated through.

**Outcome:** ORT On-Device Training accepts AV-HuBERT's operator set. The graph-transformer trace during artifact generation shows the standard ORT optimization passes ran cleanly over the entire encoder — `ConstantSharing`, `LayerNormFusion`, `GeluFusion`, `BiasGeluFusion`, `ReshapeFusion` all reported `modified: 1 with status: OK`.

The 4 ORT training artifacts are live at [`HereLiesAz/liperty-v3-training-artifacts`](https://huggingface.co/HereLiesAz/liperty-v3-training-artifacts):

| File | Size | Purpose |
|---|---|---|
| `training_model.onnx` | 410.6 MB | Forward + loss + gradient subgraph (TrainStep input) |
| `eval_model.onnx` | 410.8 MB | Forward + loss, no gradient nodes |
| `optimizer_model.onnx` | 538 B | AdamW step (small — only 2 trainable params) |
| `checkpoint` | 2.4 MB | Initial trainable parameter values |
| `nominal_checkpoint` | slim | Per-param state for memory-constrained sessions |

**What this unlocks:** the build-side of Phase 3 personalization (Step 3 in [PERSONALIZATION.md](PERSONALIZATION.md)) is done. The remaining work is on-device:

- New Kotlin class around ONNX Runtime's Android `TrainingSession` API.
- Wire the user's [`PairedTrainingRecord`](../app/src/main/java/com/hereliesaz/liperty/personalization/PairedTrainingRecord.kt) data through `TrainStep` → `OptimizerStep` → `LazyResetGrad` over the recorded clips.
- Save the resulting checkpoint as the user's personal adapter, load alongside the base encoder at inference time.

Estimated 2–3 weeks of focused Android work — a real chunk but no longer flying blind.

### 2026-05-12 (tenth): on-device personalization plan + Step 1 foundation + Step 3 PoC

The user pushed back on my framing that "encoder LoRA gives marginal gains" — correctly. LRS3 and VoxCeleb2 are demographically biased corpora; Liperty's actual users (Deaf/HoH, speech-impaired, non-Anglo demographics) are systematically under-represented. For them, the encoder itself is the bottleneck, not the downstream layers. Phase A (LM + viseme rescoring) is necessary but not sufficient.

A new track opened: on-device per-user personalization. Three layers in increasing engineering cost, all 100% on-device (off-device training is not allowed per Liperty's privacy posture). See [`docs/PERSONALIZATION.md`](PERSONALIZATION.md) for the full plan; the V3 backend log records the parts that touch the V3 stack.

**Free training data:** voice cloning already invites the user to record themselves talking. The synchronized lip motion is exactly the paired (video, text) data the entire V3 backend log has been working around the absence of. One session, three artifacts: voice clone (original purpose), personal lipreading statistics, eventually a personal encoder LoRA.

**Step 1 — recording capture infrastructure (in progress)**

The data foundation Steps 2 and 3 both need. Shipped this session:

- [`PairedTrainingRecord`](../app/src/main/java/com/hereliesaz/liperty/personalization/PairedTrainingRecord.kt) — `(audio_pcm, lip_video_frames, optional_transcript, metadata)` data class. Video frames are 88×88 grayscale (encoder-input format), NOT raw camera frames — the lip crop happens during capture so we never persist identifying features.
- [`PairedTrainingStore`](../app/src/main/java/com/hereliesaz/liperty/personalization/PairedTrainingStore.kt) — on-disk persistence with `delete(id)` and `deleteAll()` for biometric data retention controls. 8 unit tests pass.
- [`VideoFrameExtractor`](../app/src/main/java/com/hereliesaz/liperty/personalization/VideoFrameExtractor.kt) — MediaMetadataRetriever wrapper for extracting frames from imported video URIs at VAD-detected speech-segment time windows.

Still pending in Step 1: Android `SpeechRecognizer` integration for transcript labels, hook into `VoiceViewModel.startImportProcessing`, separate consent dialog (NOT the app-launch one), Settings UI for view/delete.

**Step 3 PoC — on-device encoder LoRA training feasibility**

Validated the architecture works through Microsoft's [MobileTransformers](https://martinkorelic.github.io/mobiletransformers-docs/) project running 500M-1B LLM fine-tunes on a Pixel 6 via [ONNX Runtime On-Device Training](https://onnxruntime.ai/docs/get-started/training-on-device.html). AV-HuBERT base (95M params) is well within the demonstrated envelope.

Workflow:

| Stage | Where | What |
|---|---|---|
| Build prep (one-time, NOT per-user) | Build server | Take base AV-HuBERT, add LoRA modules to attention layers, export ONNX Runtime training artifacts. Bundle in APK. |
| Recording session | On device | User records paired data via Step 1 infrastructure. |
| Training | On device | Load training artifacts; run gradient descent. Output: LoRA adapter weights (~3-5 MB). Overnight while charging. |
| Inference | On device | Load base encoder + personal LoRA adapter. Encoder runs personalized. |

Build-prep tool: [`tools/build_avhubert_training_artifacts.py`](../tools/build_avhubert_training_artifacts.py). Wraps the existing AV-HuBERT encoder with a frozen-base + trainable-adapter shell (LoRA-style bottleneck Linear, zero-initialized so it starts as no-op), exports to ONNX, then calls `onnxruntime.training.artifacts.generate_artifacts(...)` with only the adapter params marked trainable.

**PoC status:** ONNX export step verified working in the v3-export docker (output 382.9 MB, includes adapter params). Artifact generation step blocked on `onnxruntime-training` package not being in the docker image — single-line Dockerfile fix pending. After that, the on-device side (Kotlin/JNI ORT training session) is the next ~2-3 week chunk.

**Memory/time budget on phone (estimated, not yet measured):**

- AV-HuBERT base: 95M params, 392 MB fp32 (frozen, no gradients)
- LoRA rank-8 on attention: ~600K trainable params, ~3 MB
- Activation memory per 50-frame clip: ~5 MB
- Optimizer state (Adam): 2× LoRA params, ~6 MB
- Peak training memory: ~50-100 MB beyond model weights — fits on a mid-range phone
- Training time: 30 min – 2 hours for ~1000 steps. Plugged-in overnight covers it.

**Step 2 (queued behind Step 1) — cheap statistical personalization**

While Step 3 builds, ship the wins that don't need gradient descent:

- Personal n-gram LM (KenLM `lmplz` on user's own utterances; interpolate with LibriSpeech LM)
- Personal viseme confusion matrix (measure THIS user's actual viseme conflations, override the population-default `viseme_map.txt`)
- Auto-tuned hyperparameters (per-user α, viseme `candidatesPerWord`, confidence thresholds)

All days-not-weeks engineering; ship as soon as Step 1's recording infrastructure produces data.

### 2026-05-12 (ninth): Phase A landed — KenLM rescoring + viseme-aware "Chaplin's second AI"

After Phase 3 (V3 seq2seq decoder wired into Android) it was clear the encoder + decoder alone aren't enough; we need the LM-scoring step Chaplin/Auto-AVSR uses to get to 19.1% WER. This session built that, *and* extended it specifically for visual ASR.

**The LM artifact**

- LibriSpeech 3-gram pruned 1e-7 → KenLM trie+q8 binary, 27 MB on disk.
- Built via a new `docker/kenlm/Dockerfile` (kpu/kenlm pinned at HEAD, Debian bookworm-slim base, ~5 min cold build).
- Important gotcha: KenLM's `trie` format creates an unlinked temp file during conversion, and Docker Desktop's Windows bind mount doesn't preserve the file handle after unlink. Workaround: `build_binary -T /tmp/kenlm_temp ...` puts the temp on the container's tmpfs.
- LibriSpeech LM vocabulary is **UPPERCASE-normalized**. The Kotlin scorer uppercases inputs automatically; documented in [`KenLmScorer.kt`](../app/src/main/java/com/hereliesaz/liperty/ml/KenLmScorer.kt).
- Hosted at [`HereLiesAz/liperty-lm`](https://huggingface.co/HereLiesAz/liperty-lm), pulled by `setup_libs.sh`.

**The Kotlin LM stack**

- [`LanguageModelScorer`](../app/src/main/java/com/hereliesaz/liperty/ml/LanguageModelScorer.kt) — interface + `NoopLanguageModelScorer` for missing-LM degradation.
- [`KenLmScorer`](../app/src/main/java/com/hereliesaz/liperty/ml/KenLmScorer.kt) — wraps a future `libkenlm.so` via JNI. **JNI not yet built** (`Phase A3b/c` pending); `KenLmScorer.isNativeLoaded` returns false and the scorer no-ops. The rest of the rescoring pipeline is shipped and wired but currently outputs identical text to pure CTC.
- [`SubwordCtcBeamDecoder`](../app/src/main/java/com/hereliesaz/liperty/ml/SubwordCtcBeamDecoder.kt) gained optional `lmScorer`/`lmWeight` params. When set, the surviving top-K CTC beams are re-ranked by `CTC_logprob + lmWeight · LM.score(words)`. Backward-compat verified by a test that asserts identical output when LM is absent.

**Viseme-aware rescoring — the "Chaplin's-second-AI" specialized for lip-reading**

The breakthrough framing this session: a generic English LM is the wrong second AI for visual ASR. Lip-readers' systematic errors are **viseme confusions** (b/p/m, f/v, t/n, etc. — phonemes that look identical from outside the mouth). A text-only LM doesn't know about visemes; it just penalizes ungrammatical English. We need a rescorer that *uses* viseme equivalence as its candidate set, then scores those candidates with the LM.

Pipeline:

1. CTC beam search produces a hypothesis, e.g. "bird flew tasty".
2. For each word, look up its viseme sequence (cmudict phonemize → phoneme→viseme map).
3. Look up other words with the *same* viseme sequence (the visual ambiguity class). "tasty" and "nasty" share `V4 V1 V4 V4 V1` — alveolar / vowel / alveolar / alveolar / vowel.
4. Beam-search over per-position candidate sets, scoring each candidate sentence with the LM. Input-bias tiebreaker so unchanged input wins LM-indifferent ties (the "do no harm" prior).
5. Adopt the winning rescored sentence only if it *strictly* beats the input's LM score.

Worked example: "bird flew tasty" → "bird flew nasty" (more English-plausible). Or: "I want a ban" → "I want a pan" if context is food-related.

Components (all unit-tested):
- [`viseme_map.txt`](../app/src/main/assets/viseme_map.txt) — ARPABET → viseme class. 9 classes (Jeffers-Barley 1971 + Bear-Harvey 2017 adjustments).
- [`tools/build_viseme_index.py`](../tools/build_viseme_index.py) — offline tool: fetches cmudict, phonemizes 126K words, builds the inverse index. Output is 2.1 MB JSON shipped as asset.
- [`VisemeMap.kt`](../app/src/main/java/com/hereliesaz/liperty/ml/VisemeMap.kt) — asset parser.
- [`VisemeIndex.kt`](../app/src/main/java/com/hereliesaz/liperty/ml/VisemeIndex.kt) — runtime lookup, builds bi-directional maps at load time, returns viseme-equivalent words for any input word in O(1).
- [`VisemeRescorer.kt`](../app/src/main/java/com/hereliesaz/liperty/ml/VisemeRescorer.kt) — the beam-search rescorer over candidate substitutions.
- [`MainActivity`](../app/src/main/java/com/hereliesaz/liperty/MainActivity.kt) — constructs the rescorer and runs it post-CTC, pre-TranscriptionManager.

Index stats from cmudict (135K entries):
| | |
|---|---|
| Words mapped | 126,052 |
| Unique viseme sequences | 29,856 |
| Candidates per sequence (median) | 1 |
| Candidates per sequence (p95) | 11 |
| Candidates per sequence (max) | 1,673 |

The max (1,673 candidates for the most-ambiguous sequence) is why `candidatesFor(word, maxCandidates=10)` caps the per-word candidate set — rescoring cost is `O(N · K · B)` LM calls per sentence (N=words, K=candidates, B=beam) and we want sub-second latency on device.

Honest design trade-offs (worth re-evaluating once we have on-device WER numbers):
- **9 viseme classes is a guess**, not measured. Coarser groupings make the rescorer's candidate set richer at the cost of false positives. Bear-Harvey 2017 validates 9 on LRS2; Liperty's deployment may want different.
- **Single pronunciation per word** — cmudict has variants (e.g. "READ" = /R EH D/ or /R IY D/), but we keep only the first. Costs some candidates for poly-pronunciation words.
- **No frequency weighting** — alphabetical sort within candidate sets means common words don't always come first. KenLM scoring catches this back on the LM side, but a frequency tiebreaker would tighten the candidate cap.
- **All rescoring is gated on KenLM being native-loaded.** With the JNI not yet built, the rescorer runs but the LM scores everything 0, so input-bias tiebreaker keeps the original. End-to-end ready; one phase from real WER gains.

**What's left in Phase A**

- A3b/c: vendor KenLM C++ in `app/src/main/cpp/`, write CMakeLists.txt entry for Android NDK arm64-v8a, JNI bridge `kenlm_jni.cpp`. ~4-8 hours of careful work. Until this lands, all LM scoring is no-op.
- A5: extend `Seq2SeqGreedyDecoder` (V3 path) to beam-search + LM fusion. Currently V3 is greedy only.
- A6: offline WER sweep, V2-no-LM vs V2+KenLM vs V2+KenLM+VisemeRescorer. Pick α (LM weight). Pick K (candidates per word).

### 2026-05-12 (eighth): seq2seq decoder exported + Android path wired (USE_V3_BACKEND=false)

After the seventh run produced a head with 53% val CER on GRID — a corpus-mismatch failure that no amount of GRID-side training can fix — the next move was to skip a custom head entirely and reuse AV-HuBERT's own fine-tuned Transformer decoder from `base_vox_433h.pt`. That checkpoint's decoder is an autoregressive seq2seq (cross-attention into encoder features) over a 1000-token SentencePiece BPE vocab trained on LRS3 + VoxCeleb2 transcripts — i.e. the right language distribution for the encoder it sits on.

**Decoder ONNX export** ([`tools/kaggle_avhubert_export_decoder.py`](../tools/kaggle_avhubert_export_decoder.py))

Reused the same Docker image as the encoder export. Decoder is 6 layers × 768 dim × 4 attention heads, output 1000 BPE tokens. Two ONNX-specific landmines:

- av_hubert's custom `TransformerDecoder` reads `encoder_out["padding_mask"]` as a flat tensor (NOT `encoder_out["encoder_padding_mask"]` as a list, like stock fairseq). Wrong key dict = `KeyError` mid-trace.
- `decoder.buffered_future_mask` caches a `(dim, dim)` causal mask sized to the dummy input at trace time. Legacy tracer captures `dim = tensor.size(0)` as a Python int and bakes the shape in. Result: ONNX errors at runtime when `T_dec ≠ T_dummy`. Fixed by monkey-patching `buffered_future_mask` to an `arange`-based dynamic mask: `mask[i,j] = -inf if i > j else 0`, built from `torch.arange(sz)` so the op is dynamic-shape-friendly.

Final artifact: [`avhubert_base_vox_433h_decoder.onnx`](https://huggingface.co/HereLiesAz/liperty-avhubert-encoder/blob/main/avhubert_base_vox_433h_decoder.onnx) (239.5 MB) + [`avhubert_base_vox_433h_dict.txt`](https://huggingface.co/HereLiesAz/liperty-avhubert-encoder/blob/main/avhubert_base_vox_433h_dict.txt) (1000 lines).

**Decoder ONNX↔PyTorch parity** ([`tools/verify_v3_decoder_e2e.py`](../tools/verify_v3_decoder_e2e.py))

Per-position random-tensor `max_diff` scales with `T_dec` and reaches 0.34 at `T_dec=20` — concerning. But the actual *argmax* check on five real GRID-encoder-derived feature tensors found **0/5 mismatches** between PyTorch and ONNX token streams. The fp32 accumulation noise doesn't flip top-token decisions for the autoregressive loop, so ONNX is text-equivalent to PT.

(The decoded text on those 5 GRID clips is trivial — `[<bos>, 'but', <eos>]` or just `[<bos>, <eos>]` — because the GRID feature cache was built with 16-frame clips, far below AV-HuBERT's trained 50-frame window. Output quality is a context-length issue, not a decoder bug. Liperty's `FrameBuffer.slideAndGetFrames(retainCount=8)` produces the proper 50-frame window on-device.)

**Android wiring (research path; off by default)**

New components, all unit-tested:

- [`BpeDetokenizer`](../app/src/main/java/com/hereliesaz/liperty/ml/BpeDetokenizer.kt) — `IntArray` of token ids → text, handling SentencePiece `▁` word boundaries and dropping `<s>`/`<pad>`/`</s>`/`<unk>`.
- [`Seq2SeqGreedyDecoder`](../app/src/main/java/com/hereliesaz/liperty/ml/Seq2SeqGreedyDecoder.kt) — autoregressive loop (BOS → argmax → append → EOS or budget). Pure JVM, step function injected.
- [`EncoderSession`](../app/src/main/java/com/hereliesaz/liperty/ml/AvHubertEncoderSession.kt) / [`DecoderSession`](../app/src/main/java/com/hereliesaz/liperty/ml/AvHubertDecoderSession.kt) — interfaces (concrete: `AvHubertEncoderSession`, `AvHubertDecoderSession`) wrapping `ai.onnxruntime` sessions with typed I/O. The decoder extracts and returns only the last position's `(V,)` logits, not the full `(B, T_dec, V)`.
- [`AvHubertSeq2SeqInference`](../app/src/main/java/com/hereliesaz/liperty/ml/AvHubertSeq2SeqInference.kt) — orchestrator. Preprocess → encode once → autoregressive decode → BPE detokenize, returns a [`VSRResult`](../app/src/main/java/com/hereliesaz/liperty/ml/VSRInference.kt) so it's drop-in compatible with the rest of the pipeline.

Wired into [`MainActivity`](../app/src/main/java/com/hereliesaz/liperty/MainActivity.kt) behind a compile-time `USE_V3_BACKEND = false` switch. When the flag is on, the camera pipeline dispatches frames to the V3 orchestrator instead of V2's `VSRInference`; when off (default), V2 is unchanged. Missing V3 asset files are a soft fail — `avhubertV3` stays `null` and the pipeline falls back to V2.

[`setup_libs.sh`](../setup_libs.sh) now downloads the V3 encoder ONNX, decoder ONNX, and dict from `HereLiesAz/liperty-avhubert-encoder` into `app/src/main/assets/`. The repo is public so no HF token is required, but `huggingface_hub` must be installed.

18 new unit tests pass (`BpeDetokenizerTest`, `Seq2SeqGreedyDecoderTest`, `AvHubertSeq2SeqInferenceTest`). The orchestrator test uses interface-based fakes so it never touches `LipertyApplication.onCreate` — no Bluetooth-init crashes in Robolectric.

**What's deliberately not done yet**

- On-device benchmarking (Phase 4). Need a physical device to compare V2 vs V3 WER on the same camera input.
- No Settings UI toggle — `USE_V3_BACKEND` is a const for now. Promote to `BuildConfig` or settings once V3 actually beats V2.
- No beam search — greedy is the baseline. Add beam if greedy underperforms in Phase 4.
- No KV-cache decoder export — the current decoder re-runs over all `T_dec` tokens each step (`O(N²)` per utterance). For typical Liperty utterances (<50 tokens) the overhead is acceptable. Revisit if device latency becomes a problem.
- APK size impact: adding the V3 ONNXes pushes total assets to ~990 MB (V2 ~350 MB + V3 encoder 392 MB + V3 decoder 240 MB + dict). For a research build that's fine; for a public release we'd ship one backend, not both, via a Gradle flavor.

### 2026-05-11 (seventh): scaled to 6 train + 1 val — GRID overfits, head won't generalize

After the sixth run produced a head trained on 3 speakers with loss collapsing to zero (no val split), the training script grew a held-out validation speaker, periodic val loss + greedy-CTC CER reporting, and a `head_best.pt` saved on each val-loss improvement.

Run with `LIPERTY_MAX_SPEAKERS=7  LIPERTY_VAL_SPEAKERS=1` (sorted alphabetically that picks `s1, s10, s11, s12, s13, s14` for training, `s15` for validation).

Total wall time on local CPU: ~31 min (8 min feature extraction for the four new speakers — ~7 clips/sec on the CPU ONNX provider — and ~23 min training).

| step | train_loss | val_loss | val_cer | ckpt |
|---:|---:|---:|---:|:---|
| 500  | 0.557 | **0.645** | 53.1% | **best** |
| 1500 | 0.190 | 0.837 | 52.7% | |
| 2500 | 0.020 | 1.152 | 53.3% | |
| 3500 | 0.002 | 1.313 | 52.6% | |
| 5000 | 0.000 | 1.482 | 52.9% | last |

Train loss collapses to zero by step 4000, val loss climbs from 0.65 to 1.48, **val CER stays pinned around 53% for the entire run** — the model is memorizing speaker-specific lip geometry, not learning phoneme-invariant features. Greedy decode over 40 tokens with phoneme bigram structure has a chance baseline somewhere around 50-60% CER, so 53% means this head provides essentially no signal on an unseen GRID speaker.

The pipeline did what it's supposed to: it picked the step-500 checkpoint as best-by-val and exported THAT to ONNX (the previous run's memorized step-5000 weights would have been even worse on a real held-out speaker, just not measured). Parity still passed at max_diff `2e-6`.

Artifact pushed to [`HereLiesAz/liperty-v3-phoneme-head`](https://huggingface.co/HereLiesAz/liperty-v3-phoneme-head/blob/main/avhubert_v3_phoneme_head.onnx). It is **not** deployment-quality.

#### Why this happened and what it means

GRID is the wrong training corpus for V3 fine-tuning:

- Fixed 51-word grammar (`bin|lay|place|set + blue|green|red|white + at|by|in|with + A-Z (no W) + 1-9,zero + now|please|again|soon`). The phonetic distribution is a tiny English subset.
- Only 33 unique speakers exist in GRID total; sampling 6 for training is statistically thin.
- Phoneme labels are derived from cmudict lookups on the constrained vocabulary, so the head only ever sees those words' phoneme sequences and has nothing to generalize from.
- AV-HuBERT was pretrained on LRS3 + VoxCeleb2, both of which are continuous unconstrained English speech. Its features encode English phonetics, but mapping those features to phonemes from 6 GRID speakers can't undo the corpus mismatch.

A bigger model wouldn't fix this; more speakers (within GRID) wouldn't fix it; longer training (already proved) makes it worse. The fix is a different training corpus:

- **TCD-TIMIT** — same phonetic-balance design as the V1 baseline used, ~60 speakers, full TIMIT sentence set, public.
- **LRS3** — same corpus AV-HuBERT was pretrained on. Academically gated and we don't have access.
- **Skip the head entirely** — the encoder ONNX paired with the optional `LlmTextCleaner` pass (an on-device Gemma) could go straight from `(T, 768)` features to text via a learned linear projection to BPE tokens (re-using AV-HuBERT's `large_vox_iter5.pt` trained classifier head, not training a new one on GRID).

The current artifact stays on HF as a **pipeline-validation milestone**. Don't wire it into `OnnxModelEngine` for end-user inference — Android integration should wait until a non-GRID training run produces a head with a non-trivial val CER.

### 2026-05-11 (sixth): downstream head trained + exported end-to-end

With both pretrained and fine-tuned AV-HuBERT encoder ONNXes live on
[`HereLiesAz/liperty-avhubert-encoder`](https://huggingface.co/HereLiesAz/liperty-avhubert-encoder),
the downstream task can be trained **without any fairseq dependency**:
the encoder is consumed as a frozen ONNX graph via `onnxruntime`,
producing `(T, 768)` features per clip, and a tiny PyTorch head learns
the phoneme mapping.

[`tools/kaggle_avhubert_head_train.py`](../tools/kaggle_avhubert_head_train.py)
implements the V3 phoneme head:

- **Architecture**: `LayerNorm(768) → BiLSTM(512, 2 layers, bidirectional, dropout=0.1) → Linear(40)` — 11.6 M params, matches Liperty's `MLConstants.PHONEME_VOCAB` (40-symbol ARPABET).
- **Training data**: GRID shards from `HereLiesAz/liperty-grid-preprocessed` (same source the V1 baseline used).
- **Strategy**: features cached to disk after a single pass through the frozen encoder, then 5 000 CTC steps with AdamW + cosine schedule on the cached tensors.

First end-to-end run (local CPU, GRID s1+s10+s11, 5 000 steps, ~24 min):

```
step= 500 loss=0.435   step=1500 loss=0.037   step=3500 loss=0.000   step=5000 loss=0.000
```

Loss collapsed by step 3500. **This is heavy overfitting** — 3 speakers × 1 000 utterances is far too narrow for a CTC head with ~12 M params, and there's no held-out validation split yet. The artifact is useful only as a pipeline-smoke-test, not as a deployable model. Next run must scale to ≥ 6 speakers and reserve one for validation.

ONNX export landmines hit (torch 2.10 on Windows / Python 3.13):

1. The default `torch.onnx.export` now uses the dynamo path, which (a) requires `onnxscript` (not in the script's pinned deps) and (b) infers a static `T=50` from the dummy input and then errors out when `dynamic_axes` tries to mark that dim dynamic.
2. Even after `pip install onnxscript`, the same Python process has the failed import cached and re-raises on retry.
3. Torch's exporter prints a `❌` emoji on failure, which `cp1252`-codec stdout on Windows can't encode, so the failure message itself crashes.

Workaround: [`tools/finalize_v3_head.py`](../tools/finalize_v3_head.py) loads the `head_step5000.pt` checkpoint and re-exports with `dynamo=False` (legacy TorchScript tracer). The legacy path handles LSTM `dynamic_axes` natively and doesn't need `onnxscript`. Parity passed with max_diff `2e-6`.

Artifact: [`HereLiesAz/liperty-v3-phoneme-head/avhubert_v3_phoneme_head.onnx`](https://huggingface.co/HereLiesAz/liperty-v3-phoneme-head/blob/main/avhubert_v3_phoneme_head.onnx) (46.4 MB).

**V3 backend now has both ends present as public ONNXes on HF.** Android integration and on-device WER measurement are the next concrete steps; this run does not by itself say anything about whether V3 beats V2 on real input.

### 2026-05-11 (fifth): parity passed (apex FusedLayerNorm was the culprit)

After the fourth attempt produced an ONNX with all-zero output, layer-by-layer parity bisect identified the divergence: `feat_a` and `feat_v` (the frontends) matched perfectly, but `feats_ln` (the post-transpose layer norm) was all zeros in ONNX vs `[-2.86, 1.98]` in PyTorch.

Root cause: fairseq's `LayerNorm()` factory returns `apex.normalization.FusedLayerNorm` when CUDA + apex are available (both true on the NGC `pytorch:22.12-py3` base image). The fused CUDA kernel has no ONNX equivalent, so the legacy tracer silently emits zeros for it. fairseq has an `export=True` flag on the factory that disables the fused path, but it's not engaged by `torch.onnx.export` automatically.

Fix: after loading the checkpoint, walk the model and replace any `apex.normalization.FusedLayerNorm` instance with `torch.nn.LayerNorm` of matching shape/eps/elementwise_affine, copying the weight + bias parameters (they're directly compatible). 26 modules got swapped on AV-HuBERT base.

Result with the swap (run #9):
```
features     PT [-1.181, 1.129]   ORT [-1.181, 1.129]   max_diff=0.000410   ← MAIN OUTPUT
feat_a       diff=0.000000   feat_v diff=0.000088
feats_ln     diff=0.001823   feats_proj diff=0.006467
```

All within fp32 numerical noise. **The ONNX is valid and ready for V3 backend integration.**

Artifact: `out/avhubert_visual_encoder.onnx` (391.7 MB, AV-HuBERT base
visual encoder, sm_70+ compatible, dynamic time axis, input
`(1, 1, T, 88, 88)` float32, output `(1, T_out, 768)` features).

### 2026-05-11 (fourth): Docker route succeeds at build + trace,
### fails at parity

After the conda env attempt died on omegaconf 2.0.x py3.9
unavailability, switched to Docker (`docker/v3-export/`) based on
`nvcr.io/nvidia/pytorch:22.12-py3`. The Docker route bypassed the
dep-stack rot entirely (the base image freezes Python 3.8 + torch
1.14 + numpy 1.22 at build time, so pip never has to re-resolve).

Six build iterations to get a working image — each failure was a
distinct dep / API issue documented in commit messages on
[`818ccaa`](https://github.com/HereLiesAz/Liperty/commit/818ccaa):
1. fairseq's `libnat_cuda` extension needs `THC/THC.h` (removed
   in torch 1.11+)
2. After stripping CUDA extension, setup.py still tries to
   Cython-build deleted `.pyx` files
3. protobuf 4.x rejects old `_pb2.py` files until
   `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` env var is set
4. `avhubert/hubert.py` uses bare imports (`from hubert_pretraining
   import ...`) requiring `av_hubert/avhubert/` (the inner dir)
   on PYTHONPATH
5. Adding `av_hubert/` AND `av_hubert/avhubert/` to PYTHONPATH
   makes `import avhubert` and `import hubert` double-register
   the `av_hubert` model in fairseq's registry. Fixed by treating
   avhubert as a flat module collection (Meta's own
   `infer_s2s.py` pattern).

Plus runtime iterations:
- API mismatch: `AVHubertModel.extract_features` doesn't take
  `features_only`; the dict-source needs both `audio` and `video`
  keys; audio shape is `(B, 104, T_video)` not `(B, 104, T_video*4)`
  (the `stack_order_audio=4` happens at preprocessing, not at
  model input time).
- Output disk-full on Google Drive bind mount, fixed by mounting
  a regular C:\ path.

Final state of run #6:
- AV-HuBERT base loaded cleanly from Meta's CDN
- PyTorch forward: clean output `(1, 50, 768)` in 1.0s, range `[-1.14, 1.12]`
- `torch.onnx.export` completed in 25s, wrote 411 MB ONNX file
- **Parity check FAILED**: ONNX output is all-zeros, max abs diff
  to PyTorch = 1.14, mean = 0.19. The traced graph is structurally
  valid (onnxruntime loads + runs it cleanly) but captured some
  wrong control-flow path during the trace.
- The standard fixes tried (`training=TrainingMode.EVAL` explicit,
  forcing eval on the wrapper) didn't change anything.

The ONNX file is durable at
`C:\Users\azrie\v3-export-out\avhubert_visual_encoder.onnx` and at
`out/avhubert_visual_encoder.onnx` once the user copies it back to
the repo. **It is NOT yet usable for V3 inference** — its outputs
are zeros, so any downstream pipeline that consumes its features
would produce garbage.

Next-step research for whoever picks this up:
1. Add intermediate-activation logging at each fairseq layer
   (`forward_features`, `layer_norm`, `post_extract_proj`,
   `encoder.layers[0..N]`) inside both the PyTorch path and an
   onnxruntime session, and binary-search where the values diverge
   from PyTorch's path.
2. Common culprits the warnings flagged: the `assert` statements
   in `multihead_attention.py` got traced as `aten::Bool` ops,
   which may be feeding into the conditional that picks zero
   vs non-zero output paths.
3. Alternatives: torch.export.export with strict=False; or rewrite
   the encoder forward in plain PyTorch (no fairseq imports) and
   transfer the state_dict to the rewritten module.

The Dockerfile + scripts + 411 MB ONNX artifact are all durable.
The research question of *why* this particular trace produces
zero output is what remains.

### 2026-05 (third): conda env path also blocked at omegaconf

After the first two warm-kernel pip attempts failed at fairseq import,
the docs recommended a conda env with the pinned 2022 stack. Tried
exactly that. Got further but hit a different unresolvable wall.

What worked:
- Bootstrapped Miniconda inline (Kaggle's /usr/local/bin/mamba is an
  unrelated Python script, not the conda mamba — verified via
  `head -3` and missing `which conda` / `which micromamba`. So
  downloaded https://repo.anaconda.com/miniconda/Miniconda3-py39_24.7.1-0-Linux-x86_64.sh
  and ran it into /kaggle/working/miniconda3 with `-b -p`).
- `conda create -p /kaggle/working/v3_env -y -c conda-forge python=3.9 pip`
- `pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117`
- `pip install --editable /kaggle/working/work/av_hubert/fairseq --no-deps`
- `pip install "numpy==1.23.5"` (essential — `np.float` was removed in 1.24)

Where it died:
- `pip install "omegaconf==2.0.6"` → "No matching distribution found for omegaconf==2.0.6 (from versions: 1.0.3, ..., 1.0.13, ..., 2.4.0.dev8 Requires-Python >=3.10, ...)"
  - omegaconf 2.0.6's PyPI metadata excludes Python 3.9. Hard.
- `pip install "omegaconf>=2.0,<2.1"` → picks 2.0.0 (the *only* 2.0.x stable available for py39)
- omegaconf 2.0.0 has a known `issubclass()` bug on Python 3.9+:
  ```
  File "/kaggle/working/v3_env/lib/python3.9/site-packages/omegaconf/omegaconf.py", line 642, in _node_wrap
      elif issubclass(type_, Enum):
  TypeError: issubclass() arg 1 must be a class
  ```
  fixed in 2.0.5+, but those don't install on py39.
- Plus a separate constraint conflict: hydra-core 1.0.4 wants
  omegaconf>=2.0.5, while fairseq wants omegaconf<2.1. The narrow
  intersection [2.0.5, 2.1) is exactly what PyPI doesn't ship for py39.

So three independent attempts now hit three different walls in the
same dep stack. The 2022-era research stack's transitive dep graph
genuinely doesn't compose anymore on a modern PyPI index. **The
remaining viable path is Docker, and the Dockerfile + run
instructions for it live at [`docker/v3-export/`](../docker/v3-export/README.md)**.
It starts from a 2022-vintage NVIDIA PyTorch image
(`nvcr.io/nvidia/pytorch:22.12-py3`) where the entire stack
was tested and frozen at build time, not re-resolved by pip.
Kaggle supports custom-container kernels through its Datasets
feature for paid tiers; alternatively this can be done locally
or on a separate cloud machine with Docker + an NVIDIA GPU.

### 2026-05 (second): cascading dep rot, not just torch

After the first attempt's docs/fixes, retried in the same warm
Liperty kernel. Force-reinstalled torch 2.2.2 (the oldest torch
with cu118 wheels for Python 3.12), did `del sys.modules['torch']`
+ re-import, and fairseq's editable install was redone.

Two new failure layers came up in sequence:

1. **fairseq `__file__ = None` even with torch 2.2.2.** The pip
   `--editable` install registers the *outer repo directory*
   (`av_hubert/fairseq/`) on sys.path, but the actual fairseq
   Python package is at `av_hubert/fairseq/fairseq/`. So Python
   loads the outer dir as a PEP 420 namespace package and never
   finds an `__init__.py`. **Fix:** manually
   `sys.path.insert(0, '/kaggle/working/work/av_hubert/fairseq')`
   so the inner `fairseq/__init__.py` gets resolved.

2. **omegaconf API drift.** With fairseq finally importing,
   `from omegaconf import II` fails — `II` (interpolation
   indicator) was renamed/removed in omegaconf 2.1+. Kaggle's
   image has omegaconf 2.x; fairseq's vendored commit needs
   omegaconf<2.1.

This is the canonical "old research repo on a 2026 image"
cascade — each fix unblocks the next failure layer (next would
likely be hydra-core, then numpy 2.x ABI breaks). Fighting this
incrementally in a notebook console burns 15+ minutes per layer.

**Verdict:** the V3 export is not just blocked on "torch too new"
— it needs the **entire 2022 dependency stack** that fairseq
commit `afc77bdf` was tested against. Kaggle's base-image churn
makes in-place install fights pointless. Future attempts must
either:

- (a) Build a Docker container starting from `nvcr.io/nvidia/pytorch:23.04-py3`
  or similar circa-2022 image and run the export there.
- (b) Use Kaggle's "Add Container" feature to pin a known-working
  base image with locked dep versions.
- (c) Use Conda environment isolation: `conda create -n v3 python=3.10`
  then `pip install -r av_hubert/fairseq/requirements.txt`
  before the editable install. Kaggle does support conda envs but
  switching the kernel to use one is non-trivial via the UI.

### 2026-05 (first): first ONNX export attempt blocked at fairseq install

Ran `tools/kaggle_avhubert_export.py` against a Kaggle T4 session
with torch 2.10.0+cu128. The script cloned `av_hubert`,
`pip install --editable`'d its vendored fairseq submodule (commit
`afc77bdf4bb51453ce76f1572ef2ee6ddcda8eeb`), and tried to
`import avhubert.hubert`. Hit:

```
File "/kaggle/working/work/av_hubert/avhubert/hubert.py", line 16, in <module>
    from fairseq import utils
ImportError: cannot import name 'utils' from 'fairseq' (unknown location)
```

Probing `fairseq.__file__` after the editable install returned
`None`. That means Python loaded `fairseq` as a **PEP 420 namespace
package**, not as a regular package — `__init__.py` was either not
written or not located on the path. The vendored fairseq commit was
tested against torch 1.10ish and its setup machinery doesn't quite
finish on torch 2.10.

The encoder-trace risk (transformer with dynamic control flow not
ONNX-able) is downstream of this — we never even reached it.

**Verdict:** the warm-kernel `fetch+exec` pattern that works for
Liperty's other Kaggle scripts doesn't work here because the
existing kernel's torch 2.10 is incompatible with the vendored
fairseq commit. Future attempts need a **fresh Kaggle kernel with
deliberate torch downgrade** before installing fairseq:

```bash
# Required first step in any V3 export attempt:
pip install -q torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118
# Then: clone av_hubert, pip install --editable its fairseq, retry.
```

The mirror at `HereLiesAz/liperty-avhubert-encoder` and the export
artifacts (`tools/_build_export_avhubert_notebook.py`,
`tools/export_avhubert_to_onnx.ipynb`,
`tools/kaggle_avhubert_export.py`) are durable — only the *running*
of them is blocked.

## Next steps (in rough order, not committed to a schedule)

1. **Open `tools/export_avhubert_to_onnx.ipynb` in a fresh Kaggle
   kernel** (not the long-lived Liperty training session). Edit cell 2
   to prepend `pip install -q torch==2.0.1 ...` before the fairseq
   editable install. Run All. If fairseq imports cleanly, proceed.
2. **Trace the encoder.** This is the *next* risky step; if fairseq's
   transformer doesn't trace cleanly, V3 stops here until someone
   finds a workaround (e.g. via torch.export.export with dynamic
   shapes, or via a manual model-class reimplementation in plain
   PyTorch).

3. **Validate the exported ONNX matches PyTorch on a sample.** Same
   parity-check pattern as the Auto-AVSR export — feed a dummy video
   tensor through both and confirm max abs diff < 1e-3.

4. **Build a 96×96 mouth-crop pipeline.** Either adapt
   `ImageUtils.alignAndCropMouth` to use AV-HuBERT's mean face, or run
   the AV-HuBERT preparation pipeline's `align_mouth.py` on Liperty's
   incoming frames.

5. **Score the encoder + a simple CTC head** on held-out GRID. This
   gives a lower bound on V3 quality. If the CTC-only result is no
   better than V2, the LMDecoder must be exported too — back to step 2
   for the decoder.

6. **Decide on V3 viability based on real numbers.** No production swap
   without (a) proven WER improvement on Liperty-style input and
   (b) all required pieces (encoder + decoder + preprocessing) cleanly
   exporting to ONNX and running within the existing app's RAM/latency
   budget.

---

## What to do right now

Nothing. The mirror exists, the docs exist, the next step (ONNX
export) is a research notebook that needs a 4+ hour focused session
on Kaggle to attempt. Don't merge the V3 backend into `MainActivity`,
`OnnxModelEngine`, or `setup_libs.sh` until the export is proven.

If you want a small thing to ship between now and then, the legacy
TFLite phoneme path can be marked deprecated (it's only retained as a
fallback). Or ship the eval-vs-Auto-AVSR notebook properly.

## Attribution

> AV-HuBERT is licensed under the AV-HuBERT license, Copyright (c)
> Meta Platforms, Inc. All Rights Reserved.

Source repository: https://github.com/facebookresearch/av_hubert
Paper: "Learning Audio-Visual Speech Representation by Masked
Multimodal Cluster Prediction" (Shi et al., ICLR 2022).
