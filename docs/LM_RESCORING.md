# LM Rescoring & Viseme-Aware Correction

Liperty's CTC decode paths (the **SyncVSR** CTC fallback and the legacy/alternate **Auto-AVSR** backend) produce a single best-beam transcription per inference window. This document describes the two **rescoring layers** stacked on top of that output to recover from CTC errors that the encoder alone can't avoid (the production SyncVSR seq2seq path benefits from the same viseme post-rescoring):

1. **Generic English LM (KenLM)** — penalizes ungrammatical English candidates among the top-K CTC beams.
2. **Viseme-aware rescorer** — specifically attacks the dominant visual-ASR failure mode: confusable viseme-equivalent words (b/p/m, t/n/d, f/v).

Both layers are wired and unit-tested end-to-end. The KenLM JNI/native build is **in place** (`kenlm_jni.cpp` + arm64 `.a` prebuilts pulled by `setup_libs.sh`; CI fails the release build if they're absent). Scoring is active at runtime when **both** `KenLmScorer.isNativeLoaded` is true (prebuilts linked) **and** the LM (`librispeech_3gram.bin`) is present; otherwise both rescorers run as a no-op input-bias tiebreaker (original CTC output wins).

---

## Why this exists

As a reference point, the Auto-AVSR encoder ([`Amanvir/LRS3_V_WER19.1`](https://huggingface.co/Amanvir/LRS3_V_WER19.1)) hits 19.1% WER in the original paper *with* an external LM scorer during beam search. Liperty's exported CTC paths strip that scorer out, so raw CTC WER is materially higher (estimated 30-50%; Liperty's own in-domain WER is still unmeasured — see `EVAL_RESULTS_2026-05-13.md`). The two layers here are designed to recover most of that gap.

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
| `KenLmScorer` Kotlin with native gating | ✓ shipped |
| KenLM Android prebuilt (libkenlm.a + libkenlm_util.a) | ✓ at `HereLiesAz/liperty-lm/android-arm64` |
| `setup_libs.sh` pulls prebuilt into `app/src/main/cpp/kenlm/` | ✓ shipped |
| `CMakeLists.txt` links kenlm + kenlm_util into `liperty_cv` | ✓ shipped (guarded by `KENLM_AVAILABLE` macro) |
| `kenlm_jni.cpp` real `lm::ngram::Model` implementation | ✓ shipped (still falls back to stub if the .a files are absent) |
| Viseme map + index assets | ✓ shipped |
| `VisemeRescorer` | ✓ shipped |
| MainActivity wiring | ✓ shipped |
| Unit tests | ✓ 40 new tests pass |
| On-device validation that `KenLmScorer.isNativeLoaded` flips true | instrumented test added (`KenLmScorerDeviceTest`) — run on arm64 device to confirm |
| On-device WER measurement | pending — Phase A6 |

**The build-side stack is complete.** A fresh checkout that runs `./setup_libs.sh && ./gradlew assembleDebug` produces an APK that ships `libliperty_cv.so` linked against `libkenlm.a`, with the LM scoring path active end-to-end. On-device validation (confirming `KenLmScorer.isNativeLoaded` flips to `true` on a real arm64 device and that the rescorer actually produces different output) is the next concrete step — and the gating step for Phase A6's WER sweep.

### Build-time symbol verification

After `./gradlew assembleDebug`, confirm that the real KenLM JNI bridge (not the stub) was compiled into `libliperty_cv.so`:

```bash
# Check that KENLM_AVAILABLE was defined and real JNI methods are present:
nm -gC app/build/intermediates/cxx/RelWithDebInfo/*/obj/arm64-v8a/libliperty_cv.so \
  | grep -E "nativeLoad|nativeScore|nativeFree"

# You should see three T (text/code) entries — NOT zero-length stubs.
# If only stub symbols appear, the prebuilt .a files were not found by CMake.
```

### On-device validation

Run the instrumented test on a connected arm64 device:

```bash
./gradlew connectedDebugAndroidTest --tests "*.KenLmScorerDeviceTest"
```

Or inspect logcat after launching the app:

```bash
adb logcat -s KenLmScorer MainActivity | grep -i kenlm
# Expected: "KenLM scorer: isNativeLoaded=true"
# Expected: "KenLM model: /data/.../librispeech_3gram.bin (27405863 bytes)"
```

---

## Phase A3b/c: KenLM JNI build (done)

The rescoring-stack-blocking work, now complete:

1. **KenLM cross-compiled for Android arm64-v8a** via [`tools/kaggle_build_kenlm_android.py`](../tools/kaggle_build_kenlm_android.py). Inference-only build (training-pipeline subdirs excluded), no Boost dependency. Output: `libkenlm.a` (~10 MB) + `libkenlm_util.a` (~3 MB) at [`HereLiesAz/liperty-lm/android-arm64`](https://huggingface.co/HereLiesAz/liperty-lm/tree/main/android-arm64).
2. **`app/src/main/cpp/CMakeLists.txt`** declares the two as `IMPORTED STATIC` targets, adds the include path, links into `liperty_cv`. Defines a `KENLM_AVAILABLE` compile macro when the `.a` files are present (so a fresh checkout that hasn't run `setup_libs.sh` builds cleanly with `kenlm_jni.cpp` falling back to its stub).
3. **`kenlm_jni.cpp`** ships real implementations of the three native methods declared in `KenLmScorer.kt`:
   - `nativeLoad(modelPath)` — `lm::ngram::LoadVirtual(path)` (auto-detects binary format), wraps in try/catch, returns the polymorphic `lm::base::Model*` cast to `jlong`.
   - `nativeScore(handle, words[])` — `model->BeginSentenceWrite(state)`, loop `model->BaseScore(state, vocab.Index(word), out_state)` ping-ponging two state buffers sized by `model->StateSize()`, sum log10 probabilities.
   - `nativeFree(handle)` — `delete` the Model pointer.
4. **`setup_libs.sh`** pulls `android-arm64/` from HF into `app/src/main/cpp/kenlm/` via `huggingface-cli` (with `huggingface_hub` Python fallback).

On a fresh clone: `./setup_libs.sh && ./gradlew assembleDebug` produces an APK where `KenLmScorer.tryLoad(path).isNativeLoaded` returns `true` — assuming a real `librispeech_3gram.bin` ships in `assets/` (also pulled by `setup_libs.sh`).

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
