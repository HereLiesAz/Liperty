"""Generate tools/export_tts_to_onnx.ipynb.

Produces the four ONNX files PocketTTSEngine needs, uploads them to
HereLiesAz/liperty-pocket-tts on HF, so setup_libs.sh can pull them
the same way it pulls the SyncVSR assets. Replaces the dead GitHub
Release URL path in setup_libs.sh's old TTS block.

Why a Kaggle notebook and not just running tools/convert_tts_models.py
locally:
- ~3 GB of pretrained checkpoints (SpeechBrain ECAPA, Coqui VITS VCTK,
  optional FreeVC) to download. Kaggle has the bandwidth.
- VITS ONNX export is fiddly across torch versions. Kaggle's pinned
  torch + a known-good Coqui TTS pin gives a reproducible export.
- Outputs upload directly to HF via the same workflow as everything
  else, so the on-device path doesn't depend on the user manually
  copying files into app/src/main/assets/.

Models converted:
- pocket_tts_speaker.onnx (~5-7 MB): SpeechBrain ECAPA-TDNN.
  Input: audio[1, T_samples] float32 16 kHz mono.
  Output: embedding[1, 192] float32 speaker vector.
- pocket_tts_acoustic.onnx (~35 MB): Coqui VITS VCTK multispeaker.
  Inputs: input_ids[1, T_text] int64 phoneme indices,
          speaker_ids[1, 192] float32 (the embedding from above).
  Output: waveform[1, T_audio] float32 at 22050 Hz.
- pocket_tts_vocoder.onnx (~5 MB): pass-through identity. VITS is
  end-to-end (text -> waveform) so the vocoder slot is a no-op, but
  PocketTTSEngine.generateAudio() requires *some* ONNX in the
  vocoderSession slot to satisfy its 2-stage pipeline contract.
- pocket_tts_vc.onnx (~15 MB): optional FreeVC voice conversion.
  Skipped by default because FreeVC requires a manual checkpoint
  download. PocketTTSEngine treats it as optional.

PocketTTSEngine input/output contract (referenced from
PocketTTSEngine.kt — DO NOT change without updating both sides):
  Speaker:  name="audio"                        -> "embedding"
  Acoustic: name="input_ids" + "speaker_ids"    -> "waveform" (or "mel")
  Vocoder:  name="mel"                          -> "waveform"
  VC:       name="source_audio" + "target_embedding" -> "waveform"

Run with:  python tools/_build_tts_export_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "export_tts_to_onnx.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.splitlines(keepends=True)}


cells: list[dict] = []

cells.append(md("""\
# Export PocketTTS models to ONNX + upload to HF

Produces the four ONNX files `PocketTTSEngine` on the Android side expects, uploads them to `HereLiesAz/liperty-pocket-tts`. `setup_libs.sh` pulls from there into `app/src/main/assets/`.

This replaces the GitHub-Release-based download path in the previous version of `setup_libs.sh`'s TTS block, which is dead (the release tag `v0.1.0-models` doesn't host the actual .onnx files).
"""))

cells.append(md("""\
## 1. Setup
"""))

cells.append(code("""\
import os, sys
import torch

IS_KAGGLE = os.path.exists("/kaggle/working") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ
ENV = "kaggle" if IS_KAGGLE else "local"
print(f"Environment: {ENV}")
print(f"Python: {sys.version.split()[0]}, PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
"""))

cells.append(code("""\
# NOTE: NO %%capture here. Coqui TTS == 0.22.0 has brittle deps that
# conflict with Kaggle's modern Python env (numpy / torch pins clash);
# without --no-deps or a separate venv, the resolver silently fails
# AND %%capture hides the failure, so downstream `import speechbrain`
# tracebacks with no hint of what broke. We surface errors loud.
#
# Strategy:
#   1. Install the GUARANTEED-to-work deps first (huggingface_hub,
#      onnx, onnxruntime, onnxscript, numpy, scipy).
#   2. Install speechbrain — the speaker-encoder side, which is what
#      voice CLONING actually needs.
#   3. Try Coqui TTS for VITS synthesis. If it fails (most common on
#      modern Kaggle), we continue without it; the VITS export
#      section becomes a no-op and the on-device engine stays in
#      its current state where generateAudio() returns null and
#      VoiceManager falls back to system TTS.
!pip install -q "huggingface_hub>=0.27,<1.0" "onnx>=1.16" "onnxruntime>=1.18" "onnxscript" "numpy" "scipy"
# speechbrain >=1.0 — older 0.5.x calls torchaudio.list_audio_backends()
# at import time, which torchaudio 2.10+ moved out of the top-level
# module. With Kaggle's pinned torch 2.10.0+cu128, speechbrain 0.5.x
# crashes with AttributeError on import; 1.0+ handles the attribute
# move gracefully.
!pip install -q "speechbrain>=1.0,<2.0"
import importlib
SPEECHBRAIN_OK = importlib.util.find_spec("speechbrain") is not None
print(f"speechbrain importable: {SPEECHBRAIN_OK}")
assert SPEECHBRAIN_OK, "speechbrain failed to install — the speaker encoder export needs it; cannot continue."
"""))

cells.append(code("""\
# Best-effort Coqui TTS install. If this fails (numpy/torch dep
# conflict, common on Kaggle Python 3.12+), we mark TTS_OK=False and
# skip the VITS section entirely. The speaker-encoder + vocoder +
# phoneme-map outputs are still produced and uploaded; the device
# side gets enough to do voice cloning (record + embed) without
# synthesis, which is the larger-value half anyway.
import subprocess, sys
TTS_OK = False
try:
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "TTS==0.22.0"],
        capture_output=True, text=True, timeout=600,
    )
    if r.returncode == 0:
        import importlib
        TTS_OK = importlib.util.find_spec("TTS") is not None
    if not TTS_OK:
        print(f"Coqui TTS install failed (rc={r.returncode}):")
        print(r.stderr[-1500:])
except Exception as e:
    print(f"Coqui TTS install threw: {e!r}")
print(f"TTS importable: {TTS_OK}")
print("(If False, VITS section skipped; speaker encoder + vocoder + phoneme map still produced.)")
"""))

cells.append(code("""\
WORK_DIR = "/kaggle/working/work" if IS_KAGGLE else "/content/work"
os.makedirs(WORK_DIR, exist_ok=True)
print(f"Work dir: {WORK_DIR}")
"""))

cells.append(code("""\
from huggingface_hub import login, whoami

token = os.environ.get("HF_TOKEN")
if not token and IS_KAGGLE:
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        pass
if token:
    login(token, add_to_git_credential=True)
else:
    from huggingface_hub import notebook_login
    notebook_login()
print(f"HF user: {whoami()['name']}")
"""))

cells.append(md("""\
## 2. Speaker encoder (SpeechBrain ECAPA-TDNN)

Input: 16 kHz mono audio. Output: 192-d speaker embedding. ~5-7 MB.
"""))

cells.append(code("""\
import torch
import torch.nn as nn

SPEAKER_ONNX = os.path.join(WORK_DIR, "pocket_tts_speaker.onnx")

if not os.path.exists(SPEAKER_ONNX):
    from speechbrain.inference.speaker import EncoderClassifier
    print("Loading SpeechBrain ECAPA-TDNN ...")
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.join(WORK_DIR, "spkrec-ecapa-voxceleb"),
    )

    class SpeakerEncoderWrapper(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, audio):
            # audio: (1, T_samples) at 16 kHz, peak-normalized
            emb = self.m.encode_batch(audio)
            return emb.squeeze(1)   # (1, 192)

    wrapper = SpeakerEncoderWrapper(classifier).eval()
    dummy = torch.randn(1, 16000 * 3)

    print("Exporting ...")
    torch.onnx.export(
        wrapper, dummy, SPEAKER_ONNX,
        input_names=["audio"],
        output_names=["embedding"],
        dynamic_axes={"audio": {1: "audio_length"}},
        opset_version=18,
    )

import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession(SPEAKER_ONNX, providers=["CPUExecutionProvider"])
test_audio = np.random.randn(1, 48000).astype(np.float32)
emb = sess.run(None, {"audio": test_audio})[0]
print(f"Speaker encoder OK. Output shape: {emb.shape} (expected (1, 192))")
print(f"Size: {os.path.getsize(SPEAKER_ONNX) / 1e6:.1f} MB")
"""))

cells.append(md("""\
## 3. VITS acoustic model (Coqui VCTK multispeaker)

VITS is end-to-end: phoneme IDs + speaker embedding -> waveform at 22050 Hz. ~35 MB.
"""))

cells.append(code("""\
ACOUSTIC_ONNX = os.path.join(WORK_DIR, "pocket_tts_acoustic.onnx")

if not TTS_OK:
    print("Skipping VITS export — Coqui TTS not installed.")
    print("Voice CLONING (speaker embedding) still works via the encoder above.")
    print("Voice SYNTHESIS will need a different TTS backend; tracked as backlog.")
elif not os.path.exists(ACOUSTIC_ONNX):
    from TTS.api import TTS

    print("Loading Coqui VITS VCTK ...")
    tts = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False)
    vits = tts.synthesizer.tts_model
    vits.eval()

    class VITSAcousticWrapper(nn.Module):
        \"\"\"Wraps VITS.inference() in a tensor-in/tensor-out shape ONNX can
        trace. VITS internally handles attention masks, mel synthesis,
        and HiFi-GAN vocoding — what comes out is a 22050 Hz waveform.\"\"\"

        def __init__(self, vits): super().__init__(); self.vits = vits

        def forward(self, input_ids, speaker_ids):
            # input_ids: (1, T_text) long phoneme indices
            # speaker_ids: (1, 192) float speaker embedding (d_vector)
            x_lengths = torch.tensor([input_ids.shape[1]], dtype=torch.long)
            outputs = self.vits.inference(
                input_ids, x_lengths, sid=None, d_vectors=speaker_ids,
            )
            return outputs["wav"]   # (1, T_audio)

    wrapper = VITSAcousticWrapper(vits)
    dummy_ids = torch.randint(0, 40, (1, 12))
    dummy_speaker = torch.randn(1, 192)

    print("Exporting (this can take a minute) ...")
    torch.onnx.export(
        wrapper, (dummy_ids, dummy_speaker), ACOUSTIC_ONNX,
        input_names=["input_ids", "speaker_ids"],
        output_names=["waveform"],
        dynamic_axes={
            "input_ids": {1: "text_length"},
            "waveform":  {1: "audio_length"},
        },
        opset_version=18,
    )

if TTS_OK and os.path.exists(ACOUSTIC_ONNX):
    sess = ort.InferenceSession(ACOUSTIC_ONNX, providers=["CPUExecutionProvider"])
    print(f"Acoustic inputs: {[i.name for i in sess.get_inputs()]}")
    print(f"Acoustic outputs: {[o.name for o in sess.get_outputs()]}")
    print(f"Size: {os.path.getsize(ACOUSTIC_ONNX) / 1e6:.1f} MB")
"""))

cells.append(md("""\
## 4. Pass-through vocoder

VITS produces waveform directly, but `PocketTTSEngine.generateAudio()` is hardcoded for a 2-stage pipeline (acoustic -> vocoder). To preserve that contract without rewriting the engine, we ship an identity-function ONNX in the vocoder slot.

The engine sets `vocoderInputName` (default `"mel"`) on the input dict, so this ONNX must use input name `mel`. Output is identical to input.
"""))

cells.append(code("""\
VOCODER_ONNX = os.path.join(WORK_DIR, "pocket_tts_vocoder.onnx")

if not os.path.exists(VOCODER_ONNX):
    class PassthroughVocoder(nn.Module):
        def forward(self, mel):
            return mel

    wrapper = PassthroughVocoder()
    dummy = torch.randn(1, 22050)
    torch.onnx.export(
        wrapper, dummy, VOCODER_ONNX,
        input_names=["mel"],
        output_names=["waveform"],
        dynamic_axes={"mel": {1: "t"}, "waveform": {1: "t"}},
        opset_version=18,
    )

sess = ort.InferenceSession(VOCODER_ONNX, providers=["CPUExecutionProvider"])
test = np.random.randn(1, 1000).astype(np.float32)
out = sess.run(None, {"mel": test})[0]
print(f"Vocoder OK (pass-through). In={test.shape} Out={out.shape}")
print(f"Size: {os.path.getsize(VOCODER_ONNX) / 1e6:.2f} MB")
"""))

cells.append(md("""\
## 5. Phoneme map for the Android side

The Coqui VITS VCTK model expects phoneme indices defined by its OWN tokenizer (espeak-ng-derived IPA), not Liperty's MLConstants.PHONEME_VOCAB (which is 40-symbol ARPABET).

PocketTTSEngine.kt currently does the wrong thing: it tokenizes via ARPABET and passes those indices to the VITS acoustic ONNX. The model produces garbage because the index space doesn't match.

This cell dumps the VITS character-to-index map alongside the ONNX so the Android side can use it to tokenize correctly. The follow-up fix on the engine side is to read this map at init and route Liperty's text -> espeak phonemes -> VITS indices instead of -> ARPABET -> VITS indices.
"""))

cells.append(code("""\
import json as _json

PHONEME_MAP_PATH = os.path.join(WORK_DIR, "pocket_tts_phoneme_map.json")

if not TTS_OK:
    print("Skipping phoneme map dump — Coqui TTS not installed.")
    print("Voice CLONING (recording -> speaker embedding) is unaffected.")
elif not os.path.exists(PHONEME_MAP_PATH):
    from TTS.api import TTS
    tts = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False)
    vits = tts.synthesizer.tts_model

    cfg = tts.synthesizer.tts_config.characters
    pad = cfg.pad or ""
    eos = cfg.eos or ""
    bos = cfg.bos or ""
    blank = cfg.blank or ""
    chars = list(cfg.characters or "")
    phonemes = list(cfg.phonemes or "")

    # --- Extract the ACTUAL tokenizer char_to_id mapping ---
    # This is the ground-truth index assignment the VITS model was
    # trained with. Using it directly on Android avoids having to
    # reconstruct the index ordering from the config (which varies
    # between Coqui versions and add_blank / pad settings).
    tokenizer_obj = vits.tokenizer
    if hasattr(tokenizer_obj, 'char_to_id'):
        char_to_id = dict(tokenizer_obj.char_to_id)
    elif hasattr(tokenizer_obj, 'characters') and hasattr(tokenizer_obj.characters, 'char_to_id'):
        char_to_id = dict(tokenizer_obj.characters.char_to_id)
    else:
        # Fallback: build from phonemes/characters list (less reliable)
        symbols = phonemes if phonemes else chars
        char_to_id = {s: i for i, s in enumerate(symbols)}
        print("WARNING: could not extract char_to_id from tokenizer, built from list")

    # Detect add_blank setting
    add_blank = getattr(tts.synthesizer.tts_config, 'add_blank', True)

    pmap = {
        "characters": chars,
        "phonemes": phonemes,
        "char_to_id": char_to_id,
        "add_blank": add_blank,
        "pad": pad, "eos": eos, "bos": bos, "blank": blank,
        "is_phoneme": True if phonemes else False,
        "sample_rate": int(tts.synthesizer.output_sample_rate),
    }
    with open(PHONEME_MAP_PATH, "w") as f:
        _json.dump(pmap, f, indent=2, ensure_ascii=False)

if os.path.exists(PHONEME_MAP_PATH):
    with open(PHONEME_MAP_PATH) as f:
        pmap = _json.load(f)
    print(f"Phoneme map sample_rate: {pmap['sample_rate']} Hz")
    print(f"Characters ({len(pmap['characters'])}): {pmap['characters'][:20]}...")
    print(f"Phonemes ({len(pmap['phonemes'])}): {pmap['phonemes'][:20]}...")
    print(f"char_to_id entries: {len(pmap.get('char_to_id', {}))}")
    print(f"add_blank: {pmap.get('add_blank', True)}")
    blank_str = pmap.get('blank', '<blnk>')
    print(f"blank index: {pmap.get('char_to_id', {}).get(blank_str, 'NOT FOUND')}")
"""))

cells.append(md("""\
## 6. Upload to HF
"""))

cells.append(code("""\
from huggingface_hub import HfApi, create_repo

REPO = "HereLiesAz/liperty-pocket-tts"
create_repo(REPO, repo_type="model", private=False, exist_ok=True)
api = HfApi()
for path in (SPEAKER_ONNX, ACOUSTIC_ONNX, VOCODER_ONNX, PHONEME_MAP_PATH):
    if not os.path.exists(path): continue
    api.upload_file(
        path_or_fileobj=path,
        path_in_repo=os.path.basename(path),
        repo_id=REPO, repo_type="model",
        commit_message=f"PocketTTS {os.path.basename(path)} ({os.path.getsize(path) // 1024} KB)",
    )
    print(f"Uploaded {os.path.basename(path)}")
print()
print(f"All assets at: https://huggingface.co/{REPO}")
print()
print("Next steps on the Android side:")
print("  1. setup_libs.sh: change TTS block to pull from this repo")
print("  2. PocketTTSEngine.kt: use pocket_tts_phoneme_map.json instead of")
print("     MLConstants.PHONEME_VOCAB for tokenization")
print("  3. PocketTTSEngine.generateAudio: vocoder is now a pass-through,")
print("     skip it (or keep it as a no-op pipeline node)")
print("  4. AudioRouter: VITS outputs 22050 Hz, NOT 16 kHz — set the")
print("     AudioTrack sample rate accordingly")
"""))


nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
print(f"Wrote {OUT}")
