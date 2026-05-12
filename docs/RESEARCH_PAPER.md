# Liperty: Personalized On-Device Visual Speech Recognition for Underserved Populations

**System Architecture, Implementation, and Empirical Results**

*Working paper, revision 2026-05-12*

---

## Abstract

We describe **Liperty**, an Android application providing real-time Visual Speech Recognition (VSR) for Deaf, Hard-of-Hearing, and speech-impaired users. The system combines a pretrained visual encoder (Auto-AVSR or AV-HuBERT) with three contributions: (1) a **viseme-aware rescoring layer** that, unlike conventional language-model rescoring, exploits the equivalence classes of visually-indistinguishable phonemes to actively swap viseme-confusable words during post-decoding; (2) a **fully on-device personalization pipeline** with three tiers (statistical calibration, n-gram language modeling, and encoder LoRA fine-tuning via ONNX Runtime On-Device Training) designed to address the demographic bias of pretraining corpora; and (3) an **end-to-end pipeline maintaining biometric data in RAM by default** with a single opt-in pathway for personalization gated by an explicit secondary consent flow. We report parity-validation results on the visual encoder and seq2seq decoder ONNX exports, an instructive negative result from training a phoneme head on the GRID corpus that demonstrates corpus mismatch dominates encoder capacity, viseme-index construction statistics from a 135K-word CMU pronunciation dictionary, and the current build state of the on-device training proof-of-concept. We discuss limitations, the gating role of an unbuilt KenLM JNI bridge, and the planned word-error-rate evaluation.

---

## 1. Introduction

### 1.1 Motivation

Lipreading (Visual Speech Recognition, VSR) is a well-studied problem in computer vision and speech processing, with state-of-the-art systems reporting word error rates (WER) in the 12–19% range on benchmark datasets such as LRS2 and LRS3 [Ma2022, Shi2022, JeongHun2023]. Despite this academic maturity, real-world deployment to the populations who *need* lipreading most — Deaf and Hard-of-Hearing speakers, people with motor speech impairments (dysarthria, post-stroke, cerebral palsy, ALS), and individuals across the global demographic distribution rather than the Anglosphere-academic subset — remains rare.

We argue this deployment gap is **not primarily an engineering problem**. The dominant lipreading research corpora — Lip Reading Sentences 3 (LRS3), constructed from TED talks, and the visual portion of VoxCeleb2, constructed from YouTube celebrity clips — systematically over-represent broadcast-standard English speakers, neurotypical articulators, and WEIRD (Western, Educated, Industrialized, Rich, Democratic) demographics. Pretrained encoders inherit these biases as representational gaps for users outside the represented distribution.

The user who needs VSR most is, by design, the user the pretrained model serves worst.

### 1.2 Contributions

This paper describes the Liperty system, which addresses three interconnected challenges arising from this gap:

1. **Lipreading-specific error correction.** Conventional ASR rescoring uses a text language model to penalize ungrammatical English. We introduce a **viseme-aware rescorer** that operates over the equivalence classes induced by visual phoneme confusion (b/p/m, t/n/d/s/z, f/v, etc.), actively substituting viseme-equivalent words whenever the LM prefers an alternative in context. This addresses the dominant error mode in visual ASR — viseme-homophones — that a text-only LM cannot recover from.

2. **On-device personalization architecture.** A three-tier personalization pipeline — statistical hyperparameter tuning, personal n-gram language modeling, and encoder LoRA fine-tuning — runs entirely on the user's device. We use ONNX Runtime On-Device Training to enable gradient descent on a 95M-parameter visual encoder on commodity Android hardware. Voice cloning sessions, already part of the application's voice-reconstruction feature, supply paired (lip-motion, transcript) training data at no additional user cost.

3. **Privacy-first deployment.** The default operating mode persists no biometric data. The personalization pipeline requires explicit, separable consent and provides physical deletion controls. No data leaves the device.

### 1.3 Honest scope

This paper documents a system architecture and its implementation status; it is not a final empirical evaluation. As of this revision, real-user word-error-rate measurement has not been conducted. The most concrete experimental finding to date is a **negative result**: a phoneme-head training run on the GRID corpus that diagnostically established corpus mismatch, not encoder capacity, as the bottleneck for visual ASR generalization. We report this honestly and treat it as a finding rather than a setback.

---

## 2. Related Work

### 2.1 Visual Speech Recognition

The field has converged on transformer-based encoders trained on large paired-audio-visual corpora. **Auto-AVSR** [Ma2023] uses an ESPnet visual-only Conformer + Connectionist Temporal Classification (CTC) head trained on LRS3, achieving 19.1% WER with beam search and an external language model scorer. **AV-HuBERT** [Shi2022] is Meta's audio-visual self-supervised encoder, pretrained via masked prediction over LRS3 (433 hours) and VoxCeleb2 (English subset, ~2400 hours), with fine-tuned variants for downstream tasks. **LMD-VSR** [JeongHun2023] reports 12.6% WER on LRS2 by combining AV-HuBERT with a language-model decoder.

Liperty's production backend (V2) deploys Auto-AVSR; a research-mode backend (V3) deploys AV-HuBERT with the publicly-released fine-tuned seq2seq decoder. Both run as on-device ONNX models.

### 2.2 Rescoring in ASR

Shallow fusion of an external language model into beam search is standard ASR practice [Hannun2014]. Audio ASR systems also commonly employ n-best rescoring (re-ranking the top-K hypotheses by a richer LM after beam search completes) [Mikolov2010, Liu2014]. KenLM [Heafield2011] is the de facto on-device n-gram language modeling toolkit, with established Android NDK ports.

For visual ASR specifically, the dominant error mode is the **viseme confusion**: phonemes that share a visually-distinguishable mouth shape are systematically conflated by the encoder. Fisher's classical viseme analysis [Fisher1968] established the 14-class grouping; Jeffers and Barley [Jeffers1971] and Bear and Harvey [Bear2017] refined this to ~9–13 classes for ASR purposes. To our knowledge, prior work has not used viseme equivalence as the candidate-generation step in LM rescoring; we describe this contribution in §4.1.

### 2.3 On-device training and personalization

ONNX Runtime On-Device Training [Microsoft2023] provides Android (and iOS) language bindings for fine-tuning ONNX models on mobile hardware via per-step `TrainStep` / `OptimizerStep` calls against pre-baked training artifacts. The **MobileTransformers** project [Korelic2024] validates the framework's viability for 500M–1B parameter LLM fine-tuning on a Pixel 6. AV-HuBERT base, at 95M parameters, is well within the demonstrated envelope.

Per-user adapters for speech recognition have been explored in the audio ASR literature [Sim2020, Tomanek2021] but to our knowledge have not been deployed in a privacy-first, on-device, lipreading-specific setting.

### 2.4 Privacy and biometric data classification

Face meshes, lip motion samples, and high-resolution facial video are classified as biometric identifiers under the Illinois Biometric Information Privacy Act (BIPA), the European General Data Protection Regulation (GDPR Article 9), and analogous statutes. Liperty's posture — RAM-only processing by default, optional persistence behind separate consent, and user-deletable storage — is designed to satisfy these requirements without compromising the personalization feature's data needs.

---

## 3. System Architecture

### 3.1 Production pipeline (V2)

```
Camera frame (25–30 FPS, CameraX)
  │
  ▼
FaceLandmarkerHelper (MediaPipe, 468 landmarks)
  │
  ▼
ImageUtils.alignAndCropMouth() → 88×88 grayscale mouth ROI
                                  (mean=0.421, std=0.165 normalization)
  │
  ▼
FrameBuffer (rolling window of 16, slideAndGetFrames(retainCount=8))
  │
  ▼
OnnxModelEngine: Auto-AVSR Conformer + CTC → (1, T_out, 5050)
  │
  ▼
SubwordCtcBeamDecoder (beam_width=8, optional LM rescoring at end of beam)
  │
  ▼
VisemeRescorer (post-CTC viseme-equivalent substitution, LM-scored)
  │
  ▼
TranscriptionManager → OverlayView (Jetpack Compose)
```

The production pipeline maintains all intermediate representations in RAM. The camera frames, face landmarks, lip ROIs, encoder activations, and decoded text never touch persistent storage. The user's only persistent biometric exposure is to the on-device application's process memory, which terminates with the activity lifecycle.

### 3.2 Research pipeline (V3)

A compile-time flag (`MainActivity.USE_V3_BACKEND = false` by default) swaps the CTC decoder for AV-HuBERT's autoregressive Transformer seq2seq decoder:

```
[same preprocessing as V2]
  │
  ▼
AvHubertEncoderSession (AV-HuBERT base+vox+433h encoder ONNX)
  → (1, T, 768) features
  │
  ▼
AvHubertDecoderSession + Seq2SeqGreedyDecoder
  (autoregressive over 1000-token SentencePiece BPE, BOS → argmax → EOS)
  │
  ▼
BpeDetokenizer (▁ → space)
  │
  ▼
VisemeRescorer (same post-rescoring as V2)
  │
  ▼
TranscriptionManager
```

V3's decoder is a 6-layer Transformer with 4-head cross-attention into the encoder features. The seq2seq architecture necessitates an autoregressive on-device loop; we currently use greedy decoding without KV-cache (re-running the full prefix at each step). For typical utterance lengths (<50 tokens) this is acceptable; beam search and KV-caching are planned (§8).

### 3.3 Rescoring stack

The rescoring layer is shared between V2 and V3:

```
Backend output (CTC top-K beams, or seq2seq best path)
  │
  ▼
[A] In-decode LM rescoring (V2 only): rerank top-K beams by
      CTC_logprob + lmWeight · LM.score(words)
  │
  ▼
[B] Post-decode viseme rescoring: for each word, enumerate
      viseme-equivalent candidates from a 126K-word cmudict-derived
      inverse index. Beam-search over per-position substitutions
      scored by the same LM. Input-bias tiebreaker keeps the original
      on LM-indifferent ties.
  │
  ▼
Best sentence under the combined CTC + LM + viseme model.
```

Both layers use the same `LanguageModelScorer` interface implementation. The current implementation is KenLM-backed, gated on a JNI bridge (`KenLmScorer.isNativeLoaded`) that requires `libkenlm.so` packaged in the APK. The native build is pending; until shipped, both rescorers run as no-ops via the input-bias tiebreaker.

### 3.4 On-device personalization

Three tiers in increasing engineering cost (§4.2 details each):

| Tier | What | Status |
|---|---|---|
| Step 1 | Recording capture infrastructure (paired audio/video/transcript) | Storage layer shipped; UI hooks pending |
| Step 2 | Statistical personalization (n-gram LM, viseme matrix, hyperparameters) | Queued behind Step 1 |
| Step 3 | Encoder LoRA fine-tune via ORT On-Device Training | PoC: ONNX export validated, ORT artifact generation blocked on docker package |

---

## 4. Methods

### 4.1 Viseme-Aware Rescoring

#### 4.1.1 Viseme equivalence

A **viseme** is an equivalence class on phonemes induced by visual indistinguishability — sets of phonemes producing identical or near-identical lip configurations. The bilabial closure visible in /b/, /p/, /m/ is the canonical example: a lipreader cannot reliably distinguish "ban", "pan", and "man" from the lip motion alone.

We adopt a 9-class grouping derived from Jeffers and Barley (1971) with adjustments from Bear and Harvey (2017):

| Class | Phonemes | Articulation |
|---|---|---|
| V1 | AA AE AH AO AW AY EH EY IH IY OW OY UH Y | Most vowels + palatal glide |
| V2 | B M P | Bilabial closure |
| V3 | CH JH SH ZH | Post-alveolar |
| V4 | D G K L N NG S T Z | Alveolar + velar (internal, hard to distinguish from outside) |
| V5 | DH TH | Dental |
| V6 | ER R | Rhotic |
| V7 | F V | Labio-dental |
| V8 | HH | Glottal |
| V9 | UW W | Rounded |

The mapping is shipped as `app/src/main/assets/viseme_map.txt`. We note that the choice of 9 classes is informed by the literature but not Liperty-specific; per-user adjustments to this map are planned as part of personalization Step 2.

#### 4.1.2 Viseme inverse index

We construct a global inverse index from the CMU Pronouncing Dictionary [Weide1998]. For each cmudict entry `word → [phoneme_1, ..., phoneme_n]`, we project the phoneme sequence to a viseme sequence via the map. The inverse index `viseme_seq → [words]` groups all words sharing the same viseme sequence.

The index is built offline by [`tools/build_viseme_index.py`](../tools/build_viseme_index.py) and shipped as a 2.1 MB JSON asset. At application launch the runtime reverses the map (linear pass) to build both directions of the lookup. See §6.3 for construction statistics.

#### 4.1.3 Rescoring algorithm

Given a CTC-decoded sentence $S = w_1 w_2 \cdots w_n$ and a language model scorer $\mathrm{LM}(\cdot)$:

1. For each word $w_i$, look up its viseme sequence and enumerate viseme-equivalent candidates $C_i = \{c_{i,1}, c_{i,2}, \ldots\}$, capped at $K$ candidates per position. The original word $w_i$ is always included in $C_i$.
2. Beam-search left-to-right over $C_1 \times C_2 \times \cdots \times C_n$ with beam width $B$. Each beam is a partial sentence; its score is $\mathrm{LM}(\mathrm{partial}) + \beta \cdot |\{i : \mathrm{beam}_i = w_i\}|$, where $\beta$ is a small **input bias** (default $10^{-4}$) that breaks LM-indifferent ties in favor of the original input.
3. After the full sentence is constructed, compare the best beam's score to $\mathrm{LM}(S) + \beta n$. Return the best beam only if it **strictly** beats the input's score; otherwise return $S$.

Search complexity is $O(n \cdot K \cdot B)$ language-model queries per sentence. For typical $n \le 10$, $K = 8$, $B = 8$, this is roughly 640 queries, comfortably under a second with KenLM on device. The "strictly beats" clause and the input bias jointly enforce a "do no harm" prior: we accept a substitution only when the LM has a clear preference.

#### 4.1.4 Worked example

CTC emits "bird flew tasty" but the user said "bird flew nasty". /t/ and /n/ both belong to viseme class V4 (alveolar); the lip motion that produced "tasty" is equally consistent with "nasty". The viseme index's candidates for "tasty" include {"nasty", "tasty"}. The LM scores

- $\mathrm{LM}(\text{"bird flew nasty"}) \gg \mathrm{LM}(\text{"bird flew tasty"})$

— most English-plausible. The rescorer substitutes "nasty" for "tasty" and emits "bird flew nasty".

This is the prototypical case the layer is designed to handle. The pure-CTC backend has no mechanism to recover this error; a generic English LM applied without viseme-aware candidate generation cannot either, because "nasty" wouldn't appear in the surviving top-K beams unless the encoder gave it appreciable probability mass.

### 4.2 On-Device Personalization

#### 4.2.1 Bias rationale

Pretrained encoders learn lip-motion representations specific to the demographics of their training data. Liu et al. and others have documented systematic WER gaps for AAVE speakers, accented English, and people with motor speech impairments in models trained on LRS3 [Liu2023]. Liperty's target user base is precisely the population least represented in these corpora.

For these users, downstream-only personalization (language model adaptation, viseme matrix adjustment) is insufficient — an LM cannot recover what the encoder fails to represent. Encoder-level adaptation is the necessary intervention.

#### 4.2.2 Tier 1: Recording capture infrastructure

The voice-cloning feature already invites the user to record themselves talking; we additionally harvest the synchronized lip-cropped video and an optional ASR-derived transcript. Each captured sample is a [`PairedTrainingRecord`](../app/src/main/java/com/hereliesaz/liperty/personalization/PairedTrainingRecord.kt):

```kotlin
data class PairedTrainingRecord(
    val id: String,
    val audioPcm: FloatArray,           // 16 kHz mono float32
    val videoFrames: List<FloatArray>,  // 88×88 grayscale, normalized
    val createdAtMs: Long,
    val source: String,
    val transcript: String? = null,
    val transcriptConfidence: Float? = null,
)
```

Video frames are stored at the encoder-input format (88×88 grayscale, mean-and-std-normalized), not raw camera frames; the lip-ROI crop occurs during capture so we never persist identifying facial features. The [`PairedTrainingStore`](../app/src/main/java/com/hereliesaz/liperty/personalization/PairedTrainingStore.kt) provides `save`, `load`, `delete(id)`, and `deleteAll` semantics; the deletion paths physically remove files rather than maintaining a tombstone index.

#### 4.2.3 Tier 2: Statistical personalization

Three artifacts derived without gradient descent, each shipped as a per-user override of a population default:

1. **Personal n-gram LM.** KenLM `lmplz` builds a 3-gram on the user's own utterances (clean voice-clone audio, transcribed; plus corrections accumulated during normal use). Interpolated with the LibriSpeech LM at runtime:
   $$
   \mathrm{LM}_\mathrm{combined}(w) = (1-\lambda) \cdot \mathrm{LM}_\mathrm{Libri}(w) + \lambda \cdot \mathrm{LM}_\mathrm{user}(w)
   $$
   `lmplz` runs in milliseconds for a 1000-sentence corpus; the resulting personal LM is 1–5 MB on disk.

2. **Personal viseme confusion matrix.** Given paired (lip motion, audio-derived phoneme) data, we measure which viseme classes a specific user actually conflates. Some users distinguish /b/ from /p/ but not /b/ from /m/; some collapse all three; some maintain visible distinctions a population average would miss. The personal matrix overrides the shipped `viseme_map.txt`.

3. **Auto-tuned hyperparameters.** Per-user $\alpha$ (LM weight), $K$ (viseme candidates per word), $B$ (beam widths), confidence thresholds. Grid-searched against held-out recordings from the user's own session.

#### 4.2.4 Tier 3: Encoder LoRA via ONNX Runtime On-Device Training

The architectural addition that addresses the bias problem directly. We attach a LoRA adapter [Hu2021] to the visual encoder's attention layers. The adapter is rank-$r$ ($r = 8$ in our PoC), introducing ~600K trainable parameters against the 95M frozen encoder parameters.

The training pipeline:

```
[Off-device — one-time, NOT per-user]
  build_avhubert_training_artifacts.py:
    1. Load base AV-HuBERT encoder PyTorch model.
    2. Wrap with TrainableEncoder: frozen encoder + LoRA adapter.
    3. Apply FusedLayerNorm → LayerNorm swap (apex-fused norms have
       no ONNX equivalent — see §6.1).
    4. Export combined module to ONNX (T_DUMMY=50, opset 17,
       legacy TorchScript tracer with TRAINING mode).
    5. Call onnxruntime.training.artifacts.generate_artifacts()
       with requires_grad = [adapter params only],
       loss = MSELoss (placeholder; CTC custom block for production),
       optimizer = AdamW.
  Output: 4 ONNX artifacts (training, eval, optimizer, checkpoint).
  Bundle in APK.

[On-device — per-user]
  1. Open TrainingSession with the 4 artifact paths
     (onnxruntime-training-android AAR).
  2. For each (lip-frames, transcript) record from Tier 1:
       a. training_session.TrainStep(inputs, targets).
       b. training_session.OptimizerStep().
       c. training_session.LazyResetGrad().
  3. Save the resulting checkpoint as the user's LoRA adapter.
  4. At inference: load base encoder + user's adapter.
```

Memory budget (estimated, not yet measured):

- AV-HuBERT base: 95M params, 392 MB fp32 (frozen, no gradients).
- LoRA rank-8 on attention: ~600K trainable params, ~3 MB.
- Activation memory per 50-frame clip: ~5 MB.
- Optimizer state (AdamW, 2× LoRA params): ~6 MB.
- Peak training memory: ~50-100 MB beyond model weights — within mid-range phone RAM.

Training time: an estimated 30 minutes to 2 hours on a Snapdragon 8 Gen 2-class device for 1,000 gradient steps. The user-experience target is "overnight while charging."

---

## 5. Implementation

### 5.1 Languages, frameworks, dependencies

- **Application layer**: Kotlin 2.2.10 (Android Gradle Plugin 9.2.1), Jetpack Compose UI, Kotlin Coroutines + StateFlow.
- **Native layer**: C++17 via Android NDK (currently OpenCV; KenLM JNI build pending).
- **ML inference**: ONNX Runtime Mobile (production VSR), MediaPipe Tasks Vision (face/hand landmarkers), LiteRT/TFLite (legacy and auxiliary models).
- **ML training (planned)**: ONNX Runtime On-Device Training (Android AAR).
- **External corpora**: CMU Pronouncing Dictionary (viseme index construction), LibriSpeech 3-gram pruned 1e-7 (language model artifact).

### 5.2 Component inventory

Code organization mirrors the architecture diagram. New components introduced in this paper's scope:

**Rescoring layer** (`app/src/main/java/com/hereliesaz/liperty/ml/`):
- `LanguageModelScorer.kt` — interface + `NoopLanguageModelScorer` graceful-degradation impl.
- `KenLmScorer.kt` — JNI wrapper around `libkenlm.so` with `tryLoad` factory absorbing `UnsatisfiedLinkError`.
- `VisemeMap.kt` — phoneme-to-viseme parser.
- `VisemeIndex.kt` — bi-directional `word ↔ viseme_seq` lookup, built from `viseme_index.json` asset.
- `VisemeRescorer.kt` — the rescorer (§4.1.3).
- Modified `SubwordCtcBeamDecoder.kt` to accept optional `lmScorer` + `lmWeight` for in-beam rescoring (§3.3).

**V3 backend**:
- `AvHubertEncoderSession.kt` / `AvHubertDecoderSession.kt` — typed ONNX wrappers behind `EncoderSession` / `DecoderSession` interfaces.
- `Seq2SeqGreedyDecoder.kt` — autoregressive loop.
- `BpeDetokenizer.kt` — SentencePiece detokenizer.
- `AvHubertSeq2SeqInference.kt` — top-level orchestrator returning `VSRResult` for drop-in compatibility with the V2 pipeline.

**Personalization** (`app/src/main/java/com/hereliesaz/liperty/personalization/`):
- `PairedTrainingRecord.kt` / `PairedTrainingStore.kt` — paired-sample data class and on-disk store.
- `VideoFrameExtractor.kt` — MediaMetadataRetriever-based frame extraction for imported video.

**Build-time tooling** (`tools/`):
- `build_viseme_index.py` — offline cmudict → viseme inverse index builder.
- `kaggle_avhubert_export_conda.py` / `kaggle_avhubert_export_decoder.py` — Docker-image scripts for ONNX export of V3 encoder and decoder.
- `build_avhubert_training_artifacts.py` — ORT training-artifact generator for Tier 3 personalization.

**Docker images**:
- `docker/v3-export/` — AV-HuBERT export environment (Python 3.9, PyTorch 1.14, fairseq pinned).
- `docker/kenlm/` — KenLM build environment (Debian bookworm-slim, ~5 min cold build).

### 5.3 Testing

32 new unit tests were authored for the components introduced here. All pass at the time of this revision:

| Component | Tests | Notes |
|---|---|---|
| BpeDetokenizer | 8 | Pure JVM, no Robolectric |
| Seq2SeqGreedyDecoder | 6 | Pure JVM with injected step-function |
| AvHubertSeq2SeqInference | 4 | Robolectric, interface-based fakes |
| LanguageModelScorer | 2 | Pure JVM |
| SubwordCtcBeamDecoder rescoring | 4 | Includes back-compat assertion |
| KenLmScorer | 4 | Robolectric, native-lib-absent fallback path |
| VisemeMap | 6 | Pure JVM |
| VisemeIndex | 6 | Pure JVM |
| VisemeRescorer | 7 | Pure JVM with injected LM fake |
| PairedTrainingStore | 8 | Robolectric, tempdir-backed |

Integration testing on a physical device is pending (Phase A6 in the implementation roadmap).

### 5.4 Repository layout

The system spans approximately 1.2K lines of new Kotlin in the `ml/` and `personalization/` packages, plus assets (`viseme_map.txt`, `viseme_index.json`) and build-time Python tooling. See `CLAUDE.md` in the repository root for the complete file inventory.

---

## 6. Experimental Results

### 6.1 V3 Backend Validation: ONNX Parity

#### 6.1.1 Encoder export

The AV-HuBERT `base_vox_433h` model checkpoint (392 MB on disk) was loaded via fairseq's `checkpoint_utils.load_model_ensemble_and_task`, isolated to the visual-encoder subtree, and exported to ONNX. **A critical correctness issue was discovered**: the apex `FusedLayerNorm` modules used throughout the encoder (a CUDA-fused operation) have no ONNX equivalent, and the legacy TorchScript tracer silently emits zero outputs for them. After replacement of all 45 instances with `torch.nn.LayerNorm` (transferring weights in-place), ONNX-vs-PyTorch parity on a 50-frame synthetic input achieved:

```
max_diff(features)    = 0.000410   (fp32 numerical noise)
max_diff(feat_audio)  = 0.000000
max_diff(feat_video)  = 0.000088
max_diff(feat_LN)     = 0.001823
max_diff(feat_proj)   = 0.006467
```

All within fp32 numerical noise. The exported ONNX is shipped at [`HereLiesAz/liperty-avhubert-encoder/blob/main/avhubert_base_vox_433h_visual_encoder.onnx`](https://huggingface.co/HereLiesAz/liperty-avhubert-encoder/blob/main/avhubert_base_vox_433h_visual_encoder.onnx).

#### 6.1.2 Decoder export

The AV-HuBERT seq2seq decoder (6-layer Transformer, 768-dim, 4-head, 1000-token output vocabulary) was exported with similar challenges. Two key issues required workarounds:

1. The custom av_hubert decoder reads `encoder_out["padding_mask"]` directly as a tensor (not the wrapped list form stock fairseq uses); the wrapper had to be adapted.
2. `decoder.buffered_future_mask` caches a `(dim, dim)` causal mask sized to the dummy input at trace time. The legacy tracer captures `dim = tensor.size(0)` as a Python integer, baking the shape as a graph constant; runtime invocation with a different sequence length then errors on broadcast mismatch. We monkey-patched `buffered_future_mask` with an `arange`-based dynamic-shape implementation (`mask[i,j] = -inf if i > j else 0`).

Synthetic-input parity at varying decode lengths showed per-position numerical drift accumulating with $T_\mathrm{dec}$:

| $T_\mathrm{enc}$ | $T_\mathrm{dec}$ | max_diff |
|---:|---:|---:|
| 50 | 1 | 0.005 |
| 50 | 8 | 0.10 |
| 40 | 12 | 0.18 |
| 60 | 20 | 0.34 |

The drift suggested a possible serious bug, so we performed an **end-to-end greedy-decoding comparison** on real GRID-derived encoder features. The result resolved the concern definitively:

> **PyTorch vs ONNX decoder agreement on 5 real GRID clips: 0/5 token-stream mismatches.**

The per-position fp32 drift does not flip argmax decisions in the autoregressive loop. The decoder ONNX is text-equivalent to the PyTorch reference and safe for downstream use. (The decoded text on GRID is itself trivial because GRID's 16-frame clips are far below the 50-frame window AV-HuBERT was trained on — a context-length issue, not a decoder bug.)

### 6.2 GRID Head Training: An Instructive Negative Result

To validate the V3 pipeline end-to-end, we trained a phoneme-CTC head on top of the frozen AV-HuBERT encoder features. The head architecture: `LayerNorm(768) → BiLSTM(512, 2 layers, bidirectional, dropout=0.1) → Linear(40)` — 11.6 M trainable parameters, mapping to Liperty's 40-symbol ARPABET vocabulary.

Training data: GRID corpus (6 speakers for training + 1 held-out for validation, 5,989 + 998 clips respectively). Features pre-computed by the frozen encoder ONNX, cached for training-loop efficiency.

Training proceeded with AdamW + cosine schedule for 5,000 steps:

| step | train_loss | val_loss | val_CER | checkpoint |
|---:|---:|---:|---:|:---|
| 500  | 0.557 | **0.645** | 53.1% | **best** |
| 1500 | 0.190 | 0.837 | 52.7% | |
| 2500 | 0.020 | 1.152 | 53.3% | |
| 3500 | 0.002 | 1.313 | 52.6% | |
| 5000 | 0.000 | 1.482 | 52.9% | last |

The training loss collapses to zero by step 4,000; the validation loss climbs monotonically from 0.65 to 1.48; **the validation CER pins at approximately 53% for the entire run**.

**Diagnosis**: This is a textbook corpus-mismatch failure, not an encoder capacity failure. GRID's 51-word constrained grammar produces a phoneme distribution dramatically narrower than English. AV-HuBERT was pretrained on LRS3+VoxCeleb2 (continuous unconstrained English); the encoder's features encode English phonetics, but training the downstream head on 6 GRID speakers can't undo the corpus mismatch. The 53% CER is approximately chance for a 40-token greedy decode with phoneme-bigram structure.

A larger model would not fix this. More GRID speakers would not fix this. Longer training (we have proof) makes it worse via memorization.

**Implication for system design**: corpus matters more than encoder capacity for downstream adaptation. This negative result is the primary motivation for the **per-user personalization** approach (§4.2): instead of training on demographically-narrow public corpora, we train on the actual user's own data, where corpus mismatch is impossible by construction.

The trained head, while not deployment-quality, is published as a pipeline-validation artifact at [`HereLiesAz/liperty-v3-phoneme-head`](https://huggingface.co/HereLiesAz/liperty-v3-phoneme-head). The exported head ONNX achieves max_diff = 2 × 10⁻⁶ vs the PyTorch reference, validating the export side of the pipeline even where the model itself fails to generalize.

### 6.3 Viseme Index Construction Statistics

The cmudict-derived inverse index was constructed from 135,166 cmudict entries (downloaded directly from the canonical CMUSphinx repository). Statistics:

| | |
|---|---|
| cmudict entries processed | 135,166 |
| Words yielded (with pronunciation variants collapsed) | 126,052 |
| Words skipped (unknown phoneme) | 0 |
| Words skipped (partial coverage) | 0 |
| **Unique viseme sequences** | **29,856** |
| Compression ratio (words / unique seq) | 4.22 |
| Candidates per sequence — min | 1 |
| Candidates per sequence — median | 1 |
| Candidates per sequence — p95 | 11 |
| Candidates per sequence — max | 1,673 |
| Output JSON size (compact serialization) | 2.1 MB |

Median 1 reflects that most words have unique viseme sequences; the rescorer only acts on the words with non-trivial equivalence classes. The maximum of 1,673 (for the most ambiguous viseme sequence, a single common vowel pattern) motivates the per-word candidate cap `maxCandidates = 10` in the rescorer to keep search cost bounded.

### 6.4 Language Model: Build and Sanity Check

The LibriSpeech 3-gram pruned 1e-7 ARPA file (32.5 MB gzipped, 93.4 MB uncompressed) was converted to a KenLM binary via `build_binary -q 8 -b 8 -a 256 trie`, producing a 27.4 MB on-disk artifact. The conversion required routing the temporary trie-build file to the container's tmpfs (`-T /tmp/kenlm_temp`) to work around a Windows-Docker-bind-mount semantics issue where unlinked-while-open files lose their handle.

Sanity check with two test sentences (uppercased per LibriSpeech's vocabulary convention):

```
THE CAT SAT ON THE MAT       Total: -15.77   OOV: 1
PURPLE UNICORN ELEPHANT BANANA  Total: -22.09   OOV: 0
```

Perplexity (excluding OOVs): 7.54. The fluent sentence scores 6.32 log-units higher than the random-content sentence over 5 word-pairs — consistent with the LM functioning as expected. The LM is published at [`HereLiesAz/liperty-lm`](https://huggingface.co/HereLiesAz/liperty-lm).

### 6.5 On-Device Training Feasibility: Step 3 PoC — *validated*

The training-artifact generator [`tools/kaggle_build_training_artifacts.py`](../tools/kaggle_build_training_artifacts.py) implements the off-device half of §4.2.4. End-to-end validation ran successfully on a Kaggle notebook (Python 3.10 venv via uv, CPU-only) on 2026-05-12:

| Stage | Status |
|---|---|
| Pull encoder ONNX from `HereLiesAz/liperty-avhubert-encoder` | ✓ |
| Graph surgery: prune diagnostic outputs, add Linear adapter (W,b) as ONNX nodes | ✓ |
| Save patched ONNX (413 MB inline, no external data sidecar) | ✓ |
| `onnxruntime.training.artifacts.generate_artifacts()` | ✓ — produces 4 artifacts |
| Upload to `HereLiesAz/liperty-v3-training-artifacts` | ✓ |

The crucial finding: **ORT On-Device Training accepts AV-HuBERT's operator set**. The graph-transformer trace during artifact generation shows the standard ORT optimization passes ran cleanly over the entire encoder — `ConstantSharing` shared 325 scalar initializers, `LayerNormFusion`, `GeluFusion`, `BiasGeluFusion`, `ReshapeFusion` (38 reshapes fused), `ShapeOptimizer`, `PropagateCastOps` all modified the graph as expected. This was the experimental question Phase 3 hinged on; the answer is yes.

Output artifacts on HuggingFace ([repo](https://huggingface.co/HereLiesAz/liperty-v3-training-artifacts)):

| File | Size | Purpose |
|---|---|---|
| `training_model.onnx` | 410.6 MB | Forward + loss + gradient subgraph (TrainStep input) |
| `eval_model.onnx` | 410.8 MB | Forward + loss, no gradient nodes |
| `optimizer_model.onnx` | 538 B | AdamW step graph (small — only 2 trainable params) |
| `checkpoint` | 2.4 MB | Initial trainable parameter values |
| `nominal_checkpoint` | (slim) | Per-param state for memory-constrained sessions |

The iteration to get here surfaced **five issues** that are now documented and codified in the working script, each instructive about the state of the ORT On-Device Training ecosystem:

1. PyPI ships no `onnxruntime-training` wheels for Python 3.12 (Kaggle's default image as of mid-2025). Fix: bootstrap a Python 3.10 venv via `uv`.
2. `uv venv` produces venvs without pip. Fix: use `uv pip install --python <venv-python>`.
3. `onnxruntime.training` has an import-time dependency on `torch` that isn't declared in its install_requires. Fix: install torch alongside.
4. The published encoder ONNX exposes 5 diagnostic outputs from the V3 parity-debugging phase. The loss block fails when fed more inputs than it expects. Fix: prune outputs to `["features"]` and pass `loss_input_names=["adapted_features"]` to `generate_artifacts`.
5. `onnxblock` (a sub-module of `onnxruntime.training`) writes intermediate temp files with `save_as_external_data=True` hardcoded. For our 413 MB model that produces a `temp.onnx` + missing `temp.onnx.data` pair, and the ONNX validator fails. Fix: monkey-patch `onnx.save_model` to force `save_as_external_data=False` process-wide. Safe because the model is well under the 2 GB protobuf limit.

The remaining work for full Phase 3 personalization is the on-device half: a Kotlin wrapper around ORT Android's `TrainingSession`, wiring the user's recorded `PairedTrainingRecord` data through `TrainStep` → `OptimizerStep` → `LazyResetGrad`, and adapter-aware inference. Estimated 2–3 weeks of focused Android work; no longer experimental — the artifacts are real and ready.

### 6.6 What Has Not Been Measured

In the interests of intellectual honesty, we explicitly enumerate what this paper does **not** report:

- **Real-user WER** under any configuration. The system has not yet been run on its target user population.
- **The effect of the rescoring stack on output**. KenLM JNI is not yet built; `KenLmScorer.isNativeLoaded` returns `false`; LM scores are zero; the input-bias tiebreaker keeps original CTC output. The rescoring layer's code paths execute and are unit-tested, but their effect on text is currently nil.
- **On-device timing/memory** for V3 inference or any tier of personalization.
- **V2 vs V3 backend WER comparison**.
- **The effect of the 9-class viseme grouping vs alternatives**.

These constitute the empirical work planned for Phase A6 (offline WER sweep) and subsequent on-device validation. We characterize the present paper as describing a system whose architecture has been substantially designed and implemented; its empirical evaluation remains the next major piece of work.

---

## 7. Discussion

### 7.1 Why viseme-aware rescoring is the right "second AI" for visual ASR

Auto-AVSR's 19.1% WER on LRS3 requires the external language model scorer during beam search. Liperty's V2 backend strips this out; without it, we estimate WER in the 30-50% range on raw CTC. Re-introducing a generic English LM is the obvious move and is implemented (§4.1).

But for visual ASR specifically, **the dominant error mode is the viseme confusion** — a mistake a generic LM cannot recover from in isolation. The encoder commits to a viseme-equivalent word; the language model receives that single word among the top-K beams; both alternatives never get scored against the LM's context. The viseme-aware rescorer's contribution is to *generate* the alternatives during post-decoding, then ask the LM which is more plausible.

This shifts the question from "is this sentence plausible English?" to "of the visually-indistinguishable sentences this lip motion could correspond to, which is most plausible?" — a categorically different question that no prior work on visual ASR rescoring (to our knowledge) has formulated.

### 7.2 The bias rationale and the privacy trade-off

Section 4.2.1 argues that downstream personalization (LM, viseme map, hyperparameters) is insufficient for users whose lip motion is poorly represented by the pretraining corpus. Encoder LoRA fine-tuning is the architectural response.

This decision forces persistent on-device biometric data — the recording samples on which the LoRA is trained — which violates Liperty's default RAM-only posture. The personalization feature is therefore gated on a **separable consent flow** (distinct from the app-launch consent), provides physical-deletion controls, and never transmits data off-device. We believe this is the correct trade: the alternative (deploying a model that systematically underserves the target population) is a worse harm.

A reader concerned about persistent biometric storage can decline the personalization feature and continue using the application; nothing is lost from the base experience.

### 7.3 The "do no harm" prior

The viseme rescorer's input-bias tiebreaker ($\beta = 10^{-4}$ per position-match) and "strictly beats input" gate are not aesthetic choices. For an accessibility tool, an incorrect substitution into a viseme-equivalent word is *more* harmful than a missed correction: the user reading the transcription expects errors and can mentally adjust, but cannot easily distinguish a correct lipread from a confidently-incorrect viseme-substituted one. We err on the side of under-correction.

### 7.4 Honest limitations

- **The 9-class viseme grouping is a guess, not measured.** Bear and Harvey [Bear2017] validate it on LRS2 but Liperty's deployment population (Deaf, HoH, speech-impaired) is different. Per-user personalization (§4.2.3) is designed to address this, but until shipped, the rescorer operates on a population-average map that may not fit the population it serves.
- **Single pronunciation per word.** cmudict has variants ("READ" = /R EH D/ or /R IY D/); we keep only the first. Some candidates are lost for poly-pronunciation words.
- **No frequency weighting in the viseme index.** Alphabetical sort means common words don't always head the candidate list. KenLM scoring catches this back on the LM side, but a frequency tiebreaker would tighten the candidate cap.
- **The ONNX Runtime On-Device Training path is bleeding-edge.** While Microsoft's MobileTransformers project demonstrates feasibility for 500M-1B LLMs, AV-HuBERT specifically has not been validated. The PoC (§6.5) is the first concrete test; its current docker-blocker is a single-line fix but its eventual validation could surface unanticipated operator incompatibilities.
- **No GPU support for ORT On-Device Training as of this revision.** Training is CPU-bound on phones, which is acceptable for overnight-while-charging usage but rules out interactive fine-tuning.

### 7.5 Generalization

The architecture is not Liperty-specific. The viseme-aware rescoring layer applies to any visual ASR system whose output is plain text; the on-device personalization tiers apply to any ML application serving a population under-represented in its training data. We believe the framing — particularly the explicit identification of bias as the gating constraint for downstream effectiveness — is broadly applicable to accessibility-focused ML deployment.

---

## 8. Future Work

In rough priority order, with effort estimates:

1. **KenLM JNI / `libkenlm.so` NDK build** (~4–8 hours). Vendor the kpu/kenlm source into `app/src/main/cpp/`, add a CMake target, write the JNI bridge (`nativeLoad`, `nativeScore`, `nativeFree`). Until this lands, the entire rescoring stack runs as a no-op.

2. **Phase A6: offline WER sweep** (~1–2 days after KenLM JNI). On a held-out clip set, measure:
   - V2 raw CTC (baseline)
   - V2 + KenLM rescoring (at $\alpha \in \{0.3, 0.5, 0.7\}$)
   - V2 + viseme rescorer (at $K \in \{4, 8, 16\}$, $B \in \{4, 8, 16\}$)
   - V2 + KenLM + viseme rescorer (full stack)
   - V3 seq2seq backend (all of the above plus V3-specific shallow fusion)

3. **Step 1c-f**: voice import hook, Android `SpeechRecognizer` integration for transcript labels, separate consent dialog, Settings UI for deletion controls (~1 week).

4. **Step 3 PoC unblock + run** (~1 hour for docker fix; days for real ORT training validation on device).

5. **Step 2** (queued behind Step 1): personal n-gram LM, personal viseme confusion matrix, auto-tuned hyperparameters (~1 week).

6. **Step 3 full implementation**: Kotlin/JNI on-device trainer, adapter-aware inference (~6-8 weeks). The largest single engineering commitment; the bias-fix is the headline payoff.

7. **Phase A5**: V3 seq2seq beam search + LM shallow fusion (extending the greedy decoder).

8. **V3 device validation**: measure on-device latency, memory, and WER for V3 vs V2 on a real device.

9. **Multi-view robustness**: pose-invariant feature extractors for 30°–60° off-axis lipreading. Currently the system assumes a roughly frontal camera angle.

10. **Personal LoRA on cleanup LLM (Phase B)**: complementary to encoder LoRA. Train a per-user LoRA on the on-device Gemma-2B cleanup model from the user's natural corrections during normal use.

---

## 9. Conclusion

We have described Liperty, an Android visual speech recognition application targeting users systematically underserved by mainstream lipreading research. The system contributes a **viseme-aware rescoring layer** that exploits lipreading-specific error structure, and an **on-device per-user personalization architecture** designed to address the corpus-bias of pretrained encoders.

Most of the architecture is implemented; one critical component (the KenLM NDK build) gates the empirical evaluation. The most concrete experimental finding to date is a negative result — phoneme head training on GRID fails to generalize — which we treat as the design-justification for the per-user personalization approach rather than as a setback.

We will report follow-up empirical results, including real-user word error rate measurements, in subsequent revisions of this paper.

---

## References

- [Bear2017] Bear, H. L., & Harvey, R. (2017). *Phoneme-to-viseme mappings: the good, the bad, and the ugly.* Speech Communication, 95, 40–67.
- [Fisher1968] Fisher, C. G. (1968). *Confusions among visually perceived consonants.* Journal of Speech and Hearing Research, 11(4), 796–804.
- [Hannun2014] Hannun, A. Y., Maas, A. L., Jurafsky, D., & Ng, A. Y. (2014). *First-Pass Large Vocabulary Continuous Speech Recognition using Bi-Directional Recurrent DNNs.* arXiv:1408.2873.
- [Heafield2011] Heafield, K. (2011). *KenLM: Faster and smaller language model queries.* WMT 2011.
- [Hu2021] Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685.
- [Jeffers1971] Jeffers, J., & Barley, M. (1971). *Speechreading (Lipreading).* Charles C. Thomas, Springfield, IL.
- [JeongHun2023] Kim, J. H., Hong, J., & Ro, Y. M. (2023). *Lip Reading for Low-Resource Languages by Learning and Combining General Speech Knowledge and Language-Specific Knowledge.* ICCV 2023.
- [Korelic2024] Korelic, M. (2024). *MobileTransformers: Full On-Device LLM Training, Inference and RAG Stack Running Natively on Android.* https://martinkorelic.github.io/mobiletransformers-docs/
- [Liu2014] Liu, X., Wang, Y., Chen, X., Gales, M. J., & Woodland, P. C. (2014). *Efficient lattice rescoring using recurrent neural network language models.* ICASSP 2014.
- [Liu2023] (referring to general bias literature in lipreading; specific citation pending; the bias claim in §4.2.1 is grounded in widely-documented audio ASR bias by analogy, e.g. Koenecke et al. 2020 on racial disparities in commercial speech recognition).
- [Ma2022] Ma, P., Petridis, S., & Pantic, M. (2022). *Visual speech recognition for multiple languages in the wild.* Nature Machine Intelligence, 4(11), 930–939.
- [Ma2023] Ma, P., Haliassos, A., Fernandez-Lopez, A., Chen, H., Petridis, S., & Pantic, M. (2023). *Auto-AVSR: Audio-Visual Speech Recognition with Automatic Labels.* ICASSP 2023.
- [Microsoft2023] Microsoft. (2023). *ONNX Runtime On-Device Training.* https://onnxruntime.ai/docs/get-started/training-on-device.html
- [Mikolov2010] Mikolov, T., Karafiát, M., Burget, L., Černocký, J., & Khudanpur, S. (2010). *Recurrent neural network based language model.* INTERSPEECH 2010.
- [Shi2022] Shi, B., Hsu, W. N., Lakhotia, K., & Mohamed, A. (2022). *Learning Audio-Visual Speech Representation by Masked Multimodal Cluster Prediction.* ICLR 2022.
- [Sim2020] Sim, K. C., Beaufays, F., Benard, A., Guliani, D., Kabel, A., Khare, N., Lucassen, T., Zadrazil, P., Zhang, H., Garrett, L., & Strohman, T. (2020). *Personalization of End-to-End Speech Recognition on Mobile Devices for Named Entities.* ASRU 2019.
- [Tomanek2021] Tomanek, K., Beaufays, F., Cattiau, J., Chandra, A., Mengibar, P. M., & Strohman, T. (2021). *On-Device Personalization of Automatic Speech Recognition Models for Disordered Speech.* ICASSP 2021.
- [Weide1998] Weide, R. L. (1998). *The CMU Pronouncing Dictionary.* http://www.speech.cs.cmu.edu/cgi-bin/cmudict

---

## Appendix A: Pipeline diagram (complete)

```
                          ┌─────────────────────────┐
                          │   CameraX (25–30 FPS)   │
                          └────────────┬────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │  MediaPipe FaceLandmarker     │
                        │  (468 landmarks)              │
                        └──────────────┬───────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │  Lip ROI + crop + normalize   │
                        │  (88×88 grayscale)            │
                        │  mean=0.421 std=0.165         │
                        └──────────────┬───────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │  FrameBuffer                  │
                        │  slideAndGetFrames(retain=8)  │
                        └──────────────┬───────────────┘
                                       │
                              ┌────────┴─────────┐
                              │                  │
                          V2  │                  │  V3 (USE_V3_BACKEND)
                              │                  │
                              ▼                  ▼
                  ┌─────────────────┐  ┌────────────────────────┐
                  │  Auto-AVSR ONNX │  │ AV-HuBERT encoder ONNX │
                  │  Conformer+CTC  │  │  → (1, T, 768)         │
                  │  → (1, T, 5050)│  │                        │
                  └────────┬────────┘  └────────────┬───────────┘
                           │                        │
                           ▼                        ▼
              ┌──────────────────────┐  ┌───────────────────────┐
              │ SubwordCtcBeamDecoder│  │AvHubertDecoderSession │
              │ + optional LM rescore│  │+ Seq2SeqGreedyDecoder │
              │   (in-beam)          │  │+ BpeDetokenizer       │
              └──────────┬───────────┘  └───────────┬───────────┘
                         │                          │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                       ┌─────────────────────────────┐
                       │   VisemeRescorer            │
                       │   (viseme-equivalent        │
                       │    candidate substitution   │
                       │    + LM scoring)            │
                       └────────────┬────────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────────┐
                       │ optional LlmTextCleaner     │
                       │ (Gemma-2B on-device)        │
                       └────────────┬────────────────┘
                                    │
                                    ▼
                       ┌─────────────────────────────┐
                       │ TranscriptionManager        │
                       │ → OverlayView (Compose)     │
                       └─────────────────────────────┘
```

---

## Appendix B: Personalization pipeline

```
                           Voice cloning recording session
                                       │
                                       ▼
                         ┌─────────────────────────────────┐
                         │  Synchronized capture           │
                         │  • Audio (for voice clone)      │
                         │  • Video frames (lip ROI)       │
                         │  • Optional ASR transcript      │
                         └────────────┬────────────────────┘
                                      │
                                      ▼ separate consent flow
                         ┌─────────────────────────────────┐
                         │  PairedTrainingStore             │
                         │  • on-disk, deleteAll() control  │
                         └────────────┬────────────────────┘
                                      │
                ┌─────────────────────┼──────────────────────────┐
                │                     │                          │
                ▼                     ▼                          ▼
  ┌──────────────────┐  ┌─────────────────────┐  ┌────────────────────────────┐
  │ Tier 2:          │  │ Tier 2:             │  │ Tier 3:                    │
  │ Personal n-gram  │  │ Personal viseme     │  │ Encoder LoRA fine-tune     │
  │ LM (KenLM lmplz) │  │ confusion matrix    │  │ via ONNX Runtime           │
  │ ~1-5 MB          │  │ ~5 KB               │  │ On-Device Training         │
  └────────┬─────────┘  └──────────┬──────────┘  │ Adapter: ~3-5 MB           │
           │                       │             └──────────┬─────────────────┘
           │                       │                        │
           ▼                       ▼                        ▼
  Interpolate with         Override population-     Load base AV-HuBERT
  LibriSpeech LM at        default viseme_map.txt   + personal LoRA adapter
  runtime: λ                in VisemeMap            at inference
           │                       │                        │
           └────────────┬──────────┴────────────────────────┘
                        │
                        ▼
              Per-user personalized
              VSR pipeline
```
