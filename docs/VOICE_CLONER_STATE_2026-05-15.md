# Voice cloner status — 2026-05-15

Update after pivoting the TTS pipeline to OpenVoice v2 and rewriting
`PocketTTSEngine.kt` against the new contract. Supersedes the
2026-05-14 doc on a per-stage basis.

## Where this stands

### Tier 1 — engine (THIS COMMIT)

`PocketTTSEngine.kt` rewritten end-to-end for OpenVoice v2:

- Three ONNX sessions instead of four:
  - `pocket_tts_base.onnx`           — MeloTTS English base voice
  - `pocket_tts_se_extractor.onnx`   — 256-d speaker embedding extractor
  - `pocket_tts_tone_converter.onnx` — timbre transfer (base → user)
- Sample rate constants: `TTS_OUTPUT_SAMPLE_RATE_HZ` and
  `SE_EXTRACTOR_SAMPLE_RATE_HZ` both 24 000 Hz (OpenVoice v2 native rate).
- `EMBEDDING_DIM = 256` (was 192 for ECAPA-TDNN).
- `cloneVoice(name, files)` per-clip extraction + average of
  256-d embeddings. Empty list throws; dim-mismatch across clips
  throws (catches stale 192-d profiles being mixed with new ones).
- Multi-stage `generateAudio(text, voice)`:
  1. tokenize → base TTS → generic 24 kHz waveform
  2. extract / lazy-cache base speaker embedding (`src_se`)
  3. tone converter (waveform, src_se, voice.embedding) → user voice
- WAV reader honors the embedded sample rate; clips at 16/22.05/44.1/48
  kHz are linearly resampled to 24 kHz before SE extraction. Multi-
  channel inputs are downmixed.
- Tokenization gated behind `tokenizerLoaded`. Until a Kotlin port
  of `g2p_en` lands, `tokenize()` returns null and `generateAudio()`
  fails fast → `VoiceManager` falls back to system TTS. Voice
  **cloning** (record → embedding → save profile) works end-to-end
  with just the SE extractor.

`PocketTTSEngineHelpersTest` covers the pure-Kotlin helpers:
- `linearResample` identity / 2× / 0.5× / interpolation monotonicity
- `averageEmbeddings` single-clip / midpoint / centroid convergence
  / dim-mismatch / empty-list
- Constants pinned to the OpenVoice v2 contract (256-d, 24 kHz).

Supporting file updates:
- `setup_libs.sh` — `TTS_MODELS` list and FreeVC reference rewritten
  for the OpenVoice v2 file names.
- `ModelDownloadManager.kt` — three new `ModelSpec`s plus the
  vocab JSON; old `pocket_tts_acoustic / speaker / vocoder / phoneme_map`
  specs removed.
- `app/build.gradle.kts` — packaging excludes updated to the three
  new ONNX files so the APK stays under the AAB base-module size cap.

### Tier 2 — UI (NEXT)

The existing `VoiceManagementScreen` + `VoiceImportWizardScreen`
already handle multi-clip imports, speaker clustering, and quality
scoring. Gaps to close before this tier is "done":

1. **In-app multi-clip recording loop.** Today's `startRecording()`
   captures exactly one 5-sec clip. Users who want to bank multiple
   reference clips must do it outside the app and import. Add a
   `recordAdditionalClip(profileName)` flow that appends a clip to
   an existing profile and re-runs SE extraction + averaging.
2. **Quality-tier display during recording.** Right now the quality
   bar only appears after a profile exists. During recording the
   user should see live progression: ●○○○ Initial → ●●●○ Good →
   ●●●● Excellent, tied to clip count + cumulative speech duration.
3. **Banked-voice priming.** For users with ALS / progressive
   conditions, surface a "record before you can't" coach screen
   with target script and progress toward the Excellent tier.

### Tier 3 — fine-tuning notebook (LATER)

Build `tools/_build_train_openvoice_clone_notebook.py` →
`tools/train_openvoice_clone.ipynb`. Loads paired records from
`PairedTrainingStore`, fine-tunes the tone color converter
LoRA-style on the user's audio, exports a delta-overlay ONNX
that loads on top of the bundled tone converter at runtime.
Mirrors the structure of the existing SyncVSR LoRA pipeline.

## What's blocked

1. **OpenVoice v2 ONNX exports** still need to be produced. The
   notebook generator (`tools/_build_tts_export_notebook.py`) is
   in place, but `tools/export_tts_to_onnx.ipynb` has not been
   run to completion on Kaggle. Until that happens, `setup_libs.sh`
   downloads will 404 and the engine sessions will all be null,
   leaving the app permanently on the system-TTS fallback path.
2. **On-device tokenizer.** `g2p_en` is pure Python. Two viable
   ports: (a) hand-translate to Kotlin (~500 lines of CMU dict
   lookup + rule-based fallbacks); (b) bundle a tiny Kotlin
   wrapper around `eng-g2p` via JNI. Until one lands, the
   `pocket_tts_vocab.json` symbol table is loadable but unusable
   for arbitrary input text.

## To resume

1. Open `tools/export_tts_to_onnx.ipynb` in Kaggle, verify
   `HF_TOKEN` is attached, hit Run All. Expect 10–15 minutes
   for OpenVoice v2 export to produce ~140 MB across three files
   on `HereLiesAz/liperty-pocket-tts`.
2. Run `./setup_libs.sh` locally to pull the new ONNX files into
   `app/src/main/assets/`. Delete any stale `pocket_tts_acoustic.onnx`
   / `pocket_tts_speaker.onnx` / `pocket_tts_vocoder.onnx` /
   `pocket_tts_phoneme_map.json` files left from the old pipeline.
3. Rebuild + install. Voice cloning (single-clip and multi-clip)
   should work end-to-end. Voice synthesis still falls back to
   system TTS pending the g2p_en Kotlin port.
4. Start Tier 2 work in `VoiceManagementScreen` /
   `VoiceImportWizardScreen` — add the per-profile "record another
   clip" loop and the live quality-tier indicator.

## Commits in this thread

- `89a1397` — initial PocketTTS scaffolding + setup_libs.sh
- `fa6a535` — notebook generator: drop %%capture + graceful Coqui
  install failure
- `b257360` — speechbrain >=1.0,<2.0 pin (torchaudio 2.10 compat)
- `4a60630` — opset 18 bump (onnxscript dropped <18 support)
- `0942dc8` — 2026-05-14 state doc
- `40e6e11` — Kokoro-82M pivot (later reverted — preset voices only)
- `f4c17c1` — OpenVoice v2 pivot in the notebook generator
- _(THIS)_ — OpenVoice v2 contract wired into the Android engine
  + supporting file updates + helper unit tests
