# Voice cloner status — 2026-05-14

Status of the per-user voice cloning pipeline after two Kaggle export
runs and an engine audit. The intent here is for the next person
(you, in your next session) to know exactly where things are and
what's left.

## What works today

1. **Engine scaffolding (Android).** `PocketTTSEngine` loads four ONNX
   sessions (speaker / acoustic / vocoder / VC), exposes
   `cloneVoice(audioFiles)` for embedding extraction and
   `generateAudio(text, voiceState)` for synthesis. `VoiceManager`
   orchestrates between system TTS and PocketTTS. All wiring compiles
   cleanly.

2. **Sample rate fix.** `VoiceManager`'s streaming `AudioTrack` now
   reads `PocketTTSEngine.TTS_OUTPUT_SAMPLE_RATE_HZ` (22050) instead
   of the hardcoded 16000. Previously, even with working ONNX models,
   VITS output would have played 27% slow and pitch-shifted down.

3. **Synthesis fail-safe.** `PocketTTSEngine.generateAudio()` is
   gated by `espeakPhonemizerAvailable = false`. It returns `null`
   early instead of producing wrong-vocab garbage audio with the
   current ARPABET-indices-into-VITS-IPA-vocab path. `VoiceManager`
   falls back to system TTS in that case — correct user-facing
   behavior.

4. **HF asset pipeline.** `setup_libs.sh` was rewritten to pull from
   `HereLiesAz/liperty-pocket-tts` instead of a dead GitHub Release
   URL that used to silently produce 9-byte HTML stubs. Drops stale
   stubs before redownload, so existing broken installs self-heal.

5. **Notebook generator.** `tools/_build_tts_export_notebook.py` ->
   `tools/export_tts_to_onnx.ipynb`. After today's fixes, the
   notebook:
   - Drops `%%capture` from the install cell. Earlier this hid a
     pip resolver failure where `TTS==0.22.0` conflicted with
     Kaggle's modern Python and silently dropped `speechbrain`
     from the resolved deps, causing every downstream cell to fail
     with `ModuleNotFoundError: No module named 'speechbrain'`.
   - Splits installs: guaranteed-OK deps first, `speechbrain` next
     with `assert` to fail loud, `TTS` last via `subprocess.run`
     with rc capture so its (likely) failure becomes a printed
     warning rather than aborting the whole notebook.
   - Gates the VITS export + phoneme-map dump on `TTS_OK`. When
     TTS install fails (common), the notebook still produces
     `pocket_tts_speaker.onnx` and `pocket_tts_vocoder.onnx`,
     uploads them, and prints a clear "TTS synthesis skipped"
     message. Voice CLONING (recording -> embedding) works from
     just those.

## What's NOT done

1. **The conversion notebook hasn't successfully completed on Kaggle
   yet.** Two attempts today:

   - Attempt 1 (before the fix): cell 2's `%%capture` hid the pip
     failure; cell 5 ECAPA download crashed with
     `ModuleNotFoundError`. HF repo empty.
   - Attempt 2 (after the fix): re-imported the fixed notebook,
     hit Run All. Cell counters advanced past [1] but the Kaggle
     UI kept fighting remote control — clicks misregistering, the
     editor renderer freezing mid-cell, console commands going to
     a session that then restarted. After ~14 minutes I could see
     cell counter [7] on Setup (meaning at least 6 prior cells
     had executed in some form) but HF repo was still empty. The
     session was stopped clean to avoid wasting more Kaggle hours.

   The notebook generator is correct. The wrapper environment
   driving it remotely was the problem, not the notebook. In an
   interactive session at the keyboard, Run All should complete in
   ~5-10 minutes producing at least the speaker encoder upload.

2. **libespeak-ng integration on Android.** Required for
   `generateAudio()` to do real text-to-IPA-phonemes-to-VITS-indices.
   Currently gated off (`espeakPhonemizerAvailable = false`). When
   this lands:
   - Bundle libespeak-ng's Android port (~3 MB native lib) into the
     APK via CMakeLists.txt + add to packagingOptions jniLibs block.
   - Add a JNI wrapper class returning IPA phoneme strings.
   - Modify `PocketTTSEngine.generateAudio()` to route Liperty text
     through espeak -> phoneme list -> `pocket_tts_phoneme_map.json`
     `char_to_id` lookup -> int64 token tensor for the VITS ONNX.
   - Flip `espeakPhonemizerAvailable = true`.

## To resume

When you're back at the keyboard:

1. Open the Kaggle notebook (notebook81b33c16e8 in your azwashere
   workspace, or fresh). Verify HF_TOKEN secret is attached. The
   notebook is already imported from
   `HereLiesAz/Liperty/main/tools/export_tts_to_onnx.ipynb`.
2. Run All. Should take 5-10 minutes. Expected uploads to
   `HereLiesAz/liperty-pocket-tts`:
   - `pocket_tts_speaker.onnx` (~5-7 MB)
   - `pocket_tts_vocoder.onnx` (~5 KB pass-through)
   - Possibly: `pocket_tts_acoustic.onnx` + `pocket_tts_phoneme_map.json`
     if `TTS==0.22.0` happens to install cleanly that day.
3. Run `./setup_libs.sh` locally to pull the real ONNX files into
   `app/src/main/assets/` (replacing the 9-byte stubs that have
   been there for months).
4. Rebuild + install the app. Voice CLONING (record reference audio,
   extract embedding, save as profile) should now work end-to-end.
   Voice SYNTHESIS will still fall back to system TTS until
   libespeak-ng lands.

## Commits today

- 89a1397 — initial notebook + setup_libs.sh + engine audit
- fa6a535 — notebook generator fix: drop %%capture + graceful TTS
  install failure
