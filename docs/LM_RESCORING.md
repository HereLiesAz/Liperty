# LM Rescoring & Viseme-Aware Correction

Liperty's CTC backend (Auto-AVSR) produces a single best-beam transcription per inference window. This document describes the two **rescoring layers** stacked on top of that output to recover from CTC errors that the encoder alone can't avoid:

1. **Generic English LM (KenLM)** — penalizes ungrammatical English candidates among the top-K CTC beams.
2. **Viseme-aware rescorer** — specifically attacks the dominant visual-ASR failure mode: confusable viseme-equivalent words (b/p/m, t/n/d, f/v).

Both layers are wired and unit-tested end-to-end. Their actual *effect* is gated on the KenLM native library being packaged in the APK; until that lands, both run as no-ops (input-bias tiebreaker keeps the original CTC output).

---

## Why this exists

Auto-AVSR's deployed encoder ([`Amanvir/LRS3_V_WER19.1`](https://huggingface.co/Amanvir/LRS3_V_WER19.1)) hits 19.1% WER in the original paper. Liperty's V2 backend strips out the LM scorer that the paper uses during beam search — so we run closer to 30-50% WER on raw CTC. The two layers here are designed to recover most of that gap.

The viseme-aware framing matters: a vanilla text-LM treats "tasty" and "nasty" as unrelated. But /t/ and /n/ share viseme V4 (alveolar) — visually indistinguishable from outside the mouth. If the encoder commits to "tasty" and the user actually said "nasty", a plain LM rescorer can't undo that mistake unless "nasty" happens to make the surviving CTC beams. **The viseme-aware rescorer specifically generates the viseme-equivalent alternatives** as candidates, then asks the LM which is more English-plausible *in context*.

---

## Pipeline

```
CTC beam search (top-K beams by CTC score)
        │
        ▼
[A] LanguageModelScorer (n-best rescoring at end of beam loop)
    For each surviving beam, score `CTC + lmWeight · LM(words)`.
    Pick the highest combined score. Falls back to pure-CTC argmax
    when lmScorer is null or its weight is 0.
        │
        ▼  (single best sentence emerges)
[B] VisemeRescorer (per-position viseme-equivalent substitution)
    For each word in the CTC output, expand to viseme-equivalent
    candidates from a 126K-word cmudict-derived index. Beam-search
    over per-position substitutions; pick the global max-score
    sentence under the same LM. Input-bias tiebreaker keeps the
    original on LM-indifferent ties.
        │
        ▼
TranscriptionManager
```

Both [A] and [B] use the same `LanguageModelScorer` instance (KenLM under the hood). [A] catches "this CTC beam survived but isn't English"; [B] catches "this surviving beam is English-plausible but visually-confusable words score better."

---

## Components

### LanguageModelScorer interface

[`app/src/main/java/com/hereliesaz/liperty/ml/LanguageModelScorer.kt`](../app/src/main/java/com/hereliesaz/liperty/ml/LanguageModelScorer.kt)

```kotlin
interface LanguageModelScorer {
    fun score(words: List<String>): Float   // log10 prob (or any monotonic log-likelihood)
    fun close()
}

class NoopLanguageModelScorer : LanguageModelScorer   // always returns 0
```

Implementations: `KenLmScorer` (production) and test-only fakes.

### KenLmScorer

[`app/src/main/java/com/hereliesaz/liperty/ml/KenLmScorer.kt`](../app/src/main/java/com/hereliesaz/liperty/ml/KenLmScorer.kt)

Wraps a native KenLM model via JNI (Java methods: `nativeLoad`, `nativeScore`, `nativeFree`). The companion's `tryLoad(path)` absorbs `UnsatisfiedLinkError` and missing-file errors and returns a scorer whose `score()` is a no-op — so MainActivity can wire this in immediately and the rescoring path is gated only on `KenLmScorer.isNativeLoaded` becoming true (when `libkenlm.so` is packaged).

LibriSpeech LM vocabulary is **uppercase-normalized**. `score()` uppercases input automatically; callers don't need to remember.

### LM artifact

LibriSpeech 3-gram pruned 1e-7 → KenLM trie+q8 binary, 27 MB.

- Source: [OpenSLR resource 11](https://www.openslr.org/11/) (`3-gram.pruned.1e-7.arpa.gz`, 32 MB gzipped)
- Build: [`docker/kenlm/Dockerfile`](../docker/kenlm/Dockerfile) — builds `kpu/kenlm` from source on Debian bookworm-slim, ~5 min cold
- Conversion command: `build_binary -q 8 -b 8 -a 256 -T /tmp/kenlm_temp trie <arpa> <bin>`
  - `-T /tmp/kenlm_temp` is mandatory on Windows-bind-mounted Docker volumes: KenLM's trie build creates an unlinked temp file, and Windows mounts don't preserve the file handle after unlink. Routing the temp to the container tmpfs sidesteps it.
- Hosted at [`HereLiesAz/liperty-lm`](https://huggingface.co/HereLiesAz/liperty-lm); pulled by `setup_libs.sh`

### SubwordCtcBeamDecoder rescoring

[`app/src/main/java/com/hereliesaz/liperty/ml/SubwordCtcBeamDecoder.kt`](../app/src/main/java/com/hereliesaz/liperty/ml/SubwordCtcBeamDecoder.kt)

New optional constructor params:

```kotlin
SubwordCtcBeamDecoder(
    vocabulary = ...,
    beamWidth = 8,
    blankIndex = 0,
    lmScorer = kenLmScorer,            // nullable; null = no rescoring
    lmWeight = 0.5f,                   // conventional 0.3-0.7 for n-grams
)
```

When `lmScorer != null && lmWeight != 0f`, the decoder re-ranks the surviving top-K beams by `CTC_logprob + lmWeight · LM.score(words)` before picking the winner. Backward-compat verified by a test asserting identical output when LM is absent.

**Subtle bug we hit:** Kotlin's `maxByOrNull` short-circuits on single-element collections — the selector is never invoked. This is the correct optimization (LM can't change a 1-way race) but our test was generating only one beam, so the LM was never called. Tests now produce genuine beam competition.

### VisemeMap + VisemeIndex + VisemeRescorer

[`app/src/main/java/com/hereliesaz/liperty/ml/VisemeMap.kt`](../app/src/main/java/com/hereliesaz/liperty/ml/VisemeMap.kt) — ARPABET phoneme → viseme class loader. Asset: [`viseme_map.txt`](../app/src/main/assets/viseme_map.txt). 9-class grouping (Jeffers-Barley 1971 + Bear-Harvey 2017 adjustments).

[`app/src/main/java/com/hereliesaz/liperty/ml/VisemeIndex.kt`](../app/src/main/java/com/hereliesaz/liperty/ml/VisemeIndex.kt) — bidirectional `word ↔ viseme_seq ↔ [candidate_words]` lookup. Asset: [`viseme_index.json`](../app/src/main/assets/viseme_index.json), 2.1 MB, 126 052 cmudict words across 29 856 unique viseme sequences. Built offline by [`tools/build_viseme_index.py`](../tools/build_viseme_index.py).

| | |
|---|---|
| Words mapped | 126,052 |
| Unique viseme sequences | 29,856 |
| Candidates per sequence (median) | 1 |
| Candidates per sequence (p95) | 11 |
| Candidates per sequence (max) | 1,673 |

`candidatesFor(word, maxCandidates = 10)` caps the per-word candidate set to keep rescoring cost bounded.

[`app/src/main/java/com/hereliesaz/liperty/ml/VisemeRescorer.kt`](../app/src/main/java/com/hereliesaz/liperty/ml/VisemeRescorer.kt) — the rescorer:

1. Tokenize CTC output sentence into words.
2. Per position, look up viseme-equivalent candidates (capped at K).
3. Beam-search left-to-right (beam B), scoring each candidate sentence with the injected `LanguageModelScorer`.
4. Input-bias tiebreaker (`inputBias` = 1e-4 per position-match) so the original input wins LM-indifferent ties — the "do no harm" prior for an accessibility tool.
5. Return the winning sentence only if it *strictly* beats the input's LM score.

Search cost: `O(N · K · B)` LM calls per sentence. For typical N≤10, K=8, B=8 → ~640 LM calls per sentence. Well under a second with KenLM on device.

---

## Wiring in MainActivity

[`MainActivity.kt`](../app/src/main/java/com/hereliesaz/liperty/MainActivity.kt) constants:

```kotlin
const val KENLM_MODEL = "librispeech_3gram.bin"
const val KENLM_WEIGHT = 0.5f
const val VISEME_INDEX = "viseme_index.json"
```

Initialization order:

1. Stream `librispeech_3gram.bin` from assets to `filesDir`; call `KenLmScorer.tryLoad(path)`.
2. Pass the scorer to `SubwordCtcBeamDecoder(..., lmScorer = kenLmScorer, lmWeight = if (kenLmScorer?.isNativeLoaded) KENLM_WEIGHT else 0f)`.
3. Load `VisemeIndex.loadFromAssets(this, VISEME_INDEX)`.
4. Construct `VisemeRescorer(idx, kenLmScorer ?: NoopLanguageModelScorer())`.

The inference dispatch in `runInferenceOnBuffer`:

```kotlin
val ctcOutput = vsrInference.runInference(framesToProcess, landmarksToProcess).text
val rescored = visemeRescorer?.rescore(ctcOutput) ?: ctcOutput
transcriptionManager.appendText(rescored, ...)
```

---

## Current status

| Component | Status |
|---|---|
| LM binary build (Docker + KenLM) | ✓ shipped |
| `LanguageModelScorer` interface + Noop | ✓ shipped |
| `SubwordCtcBeamDecoder` rescoring | ✓ shipped (no-op when scorer absent) |
| `KenLmScorer` Kotlin skeleton with native gating | ✓ shipped |
| **KenLM JNI / `libkenlm.so`** | ✗ NOT BUILT — gates all LM-effect |
| Viseme map + index assets | ✓ shipped |
| `VisemeRescorer` | ✓ shipped |
| MainActivity wiring | ✓ shipped |
| Unit tests | ✓ 32 new tests pass |
| On-device WER measurement | ✗ not done — Phase A6 |

**The single remaining blocker for all of this to do anything useful:** the KenLM native library. The Kotlin code paths all execute today, but every LM score returns 0 until `libkenlm.so` is packaged in the APK.

---

## Phase A3b/c: building libkenlm.so

The work to unblock the rescoring stack:

1. **Vendor KenLM source** into `app/src/main/cpp/kenlm/`. The kpu/kenlm repo is small (~150 .cc/.hh files). Either submodule it or copy the headers + sources directly.
2. **Add a CMakeLists.txt target** alongside the existing OpenCV integration. Build the static parts of KenLM as a static lib, then link into `libkenlm.so` together with the JNI bridge.
3. **Write `kenlm_jni.cpp`** exposing the three native methods declared in `KenLmScorer.kt`:
   - `nativeLoad(modelPath: String) -> jlong` — load the binary into a `lm::ngram::Model`, return its pointer cast to `jlong`.
   - `nativeScore(handle: jlong, words: Array<String>) -> jfloat` — walk the words through `model.FullScoreForgotState`, sum the log10 probabilities.
   - `nativeFree(handle: jlong)` — delete the `Model`.
4. **Configure NDK build** in `app/build.gradle.kts` to produce arm64-v8a (primary) and armeabi-v7a (optional, deprecated).

Estimated effort: 4-8 hours of focused NDK work. The blockers are NDK cross-compile quirks rather than algorithmic complexity. KenLM has no exotic deps — just C++ stdlib + zlib for compressed model files.

---

## Tuning knobs

Once the LM is actually active, these are the parameters to sweep on a held-out set:

| Knob | Where | Reasonable range | Default |
|---|---|---|---|
| `KENLM_WEIGHT` (α) | MainActivity | 0.3 – 0.7 | 0.5 |
| Viseme `candidatesPerWord` | `VisemeRescorer` ctor | 4 – 16 | 8 |
| Viseme `beamWidth` | `VisemeRescorer` ctor | 4 – 16 | 8 |
| Viseme `inputBias` | `VisemeRescorer` ctor | 1e-5 – 1e-3 | 1e-4 |
| CTC `beamWidth` | `SubwordCtcBeamDecoder` ctor | 4 – 16 | 8 |

See Phase A6 in [`AVHUBERT_V3_BACKEND.md`](AVHUBERT_V3_BACKEND.md) for the planned WER sweep.

---

## Future: V3 seq2seq path

When `USE_V3_BACKEND = true`, the decoder side is replaced by AV-HuBERT's autoregressive Transformer decoder (`Seq2SeqGreedyDecoder`). The viseme rescorer applies identically to V3's output — it operates on the post-decode sentence, not the decoder's internals. The generic LM rescoring inside the CTC beam search doesn't apply (V3 is seq2seq, no CTC beam); instead Phase A5 plans to extend the V3 decoder to a beam-search variant that shallow-fuses the LM the same way as V2.
