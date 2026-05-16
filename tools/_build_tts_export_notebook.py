"""Generate tools/export_tts_to_onnx.ipynb.

Exports OpenVoice v2 components to ONNX, uploads to
HereLiesAz/liperty-pocket-tts on HF. setup_libs.sh pulls from there.

WHY OPENVOICE V2 (after rejecting Coqui TTS, SpeechT5, Kokoro):

The Liperty user base includes people who LOST their voice (ALS,
laryngeal cancer, throat surgery, stroke, vocal cord injury, etc.).
For these users, voice cloning isn't a nice-to-have — it's the
entire point. They have recordings (voicemails, family videos,
voice memos), they know what they sounded like, and they want to
keep sounding like themselves.

Zero-shot cloning is what they need:
- They may have only short reference clips (a few seconds).
- Disease progression makes long recording sessions impractical
  (ALS users may have weeks before voice loss).
- Their reference audio quality varies; they need a model robust
  to less-than-studio input.

OpenVoice v2 (myshell-ai, MIT) is purpose-built for this:
- Two-stage: MeloTTS produces clean base speech, then a Tone Color
  Converter transforms timbre using a reference clip (5-30 sec).
- ~140 MB total: ~70 MB MeloTTS English + ~50 MB Tone Color
  Converter + ~20 MB speaker encoder.
- g2p_en tokenizer (pure Python, no espeak-ng on Android).
- 24 kHz output.
- Active maintenance, community ONNX exports.
- Trained on enough speaker variety that it generalizes well to
  voices it never saw during training (atypical voices, accents,
  prosody).

EXPLICITLY REJECTED:
- Coqui TTS / Coqui XTTS-v2: numpy/torch dep hell on Python 3.12.
- SpeechT5: ~1 GB, overkill, voice cloning works but quality
  lower than OpenVoice v2.
- Kokoro-82M: 54 PRESET voices, no zero-shot cloning. Wrong
  feature set for our actual users.
- Piper TTS: requires espeak-ng on device; preset voices only.
- Tortoise-TTS: high quality cloning but ~5 sec per inference,
  unusable on mobile CPU.

PIPELINE:
  text + ref_audio
    -> g2p_en tokenize
    -> MeloTTS base ONNX -> generic-voice waveform
    -> SE Extractor ONNX (run on ref audio once, cache)
       + base waveform
    -> Tone Color Converter ONNX
    -> user-voice waveform at 24 kHz

ENGINE INPUT/OUTPUT CONTRACT after this lands:
  base_tts.onnx: inputs=(input_ids, speaker_id, speed)
                 output=audio   (1, T_base)
  se_extractor.onnx: input=audio  (1, T_ref)
                     output=speaker_embedding  (1, 256)
  tone_converter.onnx: inputs=(source_audio, source_se, target_se, tau)
                       output=audio  (1, T_out)

The user records reference audio ONCE on first run, we cache the
extracted speaker_embedding. At synthesis time the cache hit is
free; the conversion is the per-utterance cost.

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
# Export OpenVoice v2 to ONNX + upload to HF

Converts the three OpenVoice v2 components (MeloTTS base + Tone Color Converter + Speaker Encoder) to ONNX, uploads to `HereLiesAz/liperty-pocket-tts`. `setup_libs.sh` pulls from there into `app/src/main/assets/`.

**Why OpenVoice v2:** Liperty's user base includes people who lost their voice (ALS, laryngeal cancer, throat surgery, stroke). For them, cloning IS the feature. OpenVoice v2 is zero-shot — 5-30 seconds of reference audio is enough to capture timbre. The pipeline is two-stage: MeloTTS generates clean base speech, then the Tone Color Converter transforms it to match the reference voice.

~140 MB total. MIT licensed. g2p_en tokenizer (pure Python, no espeak-ng on Android).
"""))

cells.append(md("""\
## 1. Setup
"""))

cells.append(code("""\
import os, sys
print(f"Python: {sys.version.split()[0]}")

IS_KAGGLE = os.path.exists("/kaggle/working") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ
ENV = "kaggle" if IS_KAGGLE else "local"
print(f"Environment: {ENV}")

import torch
print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
"""))

cells.append(code("""\
# Drop %%capture so errors are visible.
print("=== Installing utility deps ===")
!pip install -q "huggingface_hub>=0.27,<1.0" "onnx>=1.16" "onnxruntime>=1.18" "onnxscript" "numpy" "scipy" "soundfile" "librosa"

print("\\n=== Cloning OpenVoice + MeloTTS source ===")
WORK_DIR = "/kaggle/working/work" if IS_KAGGLE else "/content/work"
os.makedirs(WORK_DIR, exist_ok=True)
OPENVOICE_DIR = os.path.join(WORK_DIR, "OpenVoice")
MELOTTS_DIR = os.path.join(WORK_DIR, "MeloTTS")
if not os.path.exists(OPENVOICE_DIR):
    !git clone --depth 1 https://github.com/myshell-ai/OpenVoice.git {OPENVOICE_DIR}
if not os.path.exists(MELOTTS_DIR):
    !git clone --depth 1 https://github.com/myshell-ai/MeloTTS.git {MELOTTS_DIR}

print("\\n=== Installing OpenVoice + MeloTTS Python deps ===")
# OpenVoice's setup.py pulls in librosa, faster-whisper, etc.
# MeloTTS pulls in g2p_en, jieba, etc. Both are pure-Python deps
# (no native compile) so they install cleanly on Python 3.12.
import subprocess
for src in (OPENVOICE_DIR, MELOTTS_DIR):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", src],
                       capture_output=True, text=True, timeout=600)
    print(f"  {os.path.basename(src)} install: rc={r.returncode}")
    if r.returncode != 0:
        print(f"    stderr tail: {r.stderr[-800:]}")

# unidic-lite for MeloTTS Japanese; we only need English but the
# import chain may pull it. Best-effort.
!python -m unidic download 2>&1 | tail -1 || true

import importlib
for mod in ("openvoice", "melo", "torch", "onnx", "onnxruntime"):
    print(f"{mod}: {'OK' if importlib.util.find_spec(mod) else 'MISSING'}")
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
## 2. Download OpenVoice v2 checkpoints
"""))

cells.append(code("""\
from huggingface_hub import snapshot_download

CKPT_DIR = os.path.join(WORK_DIR, "ckpts")
os.makedirs(CKPT_DIR, exist_ok=True)

# OpenVoice v2 ships its checkpoints in this repo. Three components:
#  - base_speakers/ses/  (per-language base speaker embeddings)
#  - converter/          (tone color converter)
#  - tokenizer/          (g2p_en + bert support)
OPENVOICE_REPO = "myshell-ai/OpenVoiceV2"

print(f"Downloading {OPENVOICE_REPO} ...")
ov_local = snapshot_download(
    repo_id=OPENVOICE_REPO,
    local_dir=os.path.join(CKPT_DIR, "OpenVoiceV2"),
)
print(f"  -> {ov_local}")
print(f"  Contents:")
for root, _, files in os.walk(ov_local):
    rel = os.path.relpath(root, ov_local)
    for f in files:
        full = os.path.join(root, f)
        sz_kb = os.path.getsize(full) // 1024
        print(f"    {os.path.join(rel, f)}  ({sz_kb} KB)")
"""))

cells.append(code("""\
# MeloTTS English checkpoint for the base TTS stage. MeloTTS hosts
# its weights under myshell-ai/MeloTTS-English-v3 (or v2 depending
# on release).
MELO_REPO = "myshell-ai/MeloTTS-English-v3"
try:
    melo_local = snapshot_download(
        repo_id=MELO_REPO,
        local_dir=os.path.join(CKPT_DIR, "MeloTTS-English-v3"),
    )
    print(f"  MeloTTS-English-v3 -> {melo_local}")
except Exception as e:
    print(f"v3 fetch failed ({e}); trying v2 ...")
    MELO_REPO = "myshell-ai/MeloTTS-English"
    melo_local = snapshot_download(
        repo_id=MELO_REPO,
        local_dir=os.path.join(CKPT_DIR, "MeloTTS-English"),
    )
    print(f"  MeloTTS-English -> {melo_local}")
"""))

cells.append(md("""\
## 3. Export Tone Color Converter to ONNX

The Tone Color Converter is the magic piece — takes (source_waveform, source_speaker_embedding, target_speaker_embedding) and produces source's content in target's voice. This is the part the user-with-a-banked-voice actually consumes.
"""))

cells.append(code("""\
import sys
sys.path.insert(0, OPENVOICE_DIR)
from openvoice.api import ToneColorConverter
import torch
import torch.nn as nn

TONE_CONVERTER_ONNX = os.path.join(WORK_DIR, "pocket_tts_tone_converter.onnx")

# Initialize the converter from its config + checkpoint.
converter_cfg = os.path.join(ov_local, "converter", "config.json")
converter_ckpt = os.path.join(ov_local, "converter", "checkpoint.pth")
print(f"Loading ToneColorConverter from {converter_ckpt} ...")
tcc = ToneColorConverter(converter_cfg, device="cpu")
tcc.load_ckpt(converter_ckpt)
tcc.model.eval()
print("Loaded.")
"""))

cells.append(code("""\
class ToneConverterWrapper(nn.Module):
    \"\"\"Wraps ToneColorConverter.model.voice_conversion() for ONNX.

    Inputs:
      audio_src: (1, T_src)        source waveform at 24 kHz, mono
      src_se:    (1, 256, 1)       source speaker embedding
      tgt_se:    (1, 256, 1)       target speaker embedding
      tau:       scalar float      temperature (0.3 default)
    Output:
      audio_out: (1, T_out)        converted waveform at 24 kHz
    \"\"\"
    def __init__(self, tcc_model): super().__init__(); self.m = tcc_model
    def forward(self, audio_src, src_se, tgt_se, tau):
        # voice_conversion expects (1, T) audio and (B, 256, 1) embeddings.
        out = self.m.voice_conversion(audio_src, src_se=src_se, tgt_se=tgt_se, tau=tau)
        return out

wrapper = ToneConverterWrapper(tcc.model).eval()
dummy_audio = torch.randn(1, 24000)
dummy_src_se = torch.randn(1, 256, 1)
dummy_tgt_se = torch.randn(1, 256, 1)
dummy_tau = torch.tensor(0.3)

with torch.no_grad():
    try:
        out = wrapper(dummy_audio, dummy_src_se, dummy_tgt_se, dummy_tau)
        print(f"Sanity forward OK. Output shape: {tuple(out.shape)}")
    except Exception as e:
        print(f"Sanity forward FAILED: {e}")
        print("Tone Color Converter export needs upstream API debugging.")
        out = None

if out is not None:
    print(f"Exporting to {TONE_CONVERTER_ONNX} ...")
    torch.onnx.export(
        wrapper, (dummy_audio, dummy_src_se, dummy_tgt_se, dummy_tau),
        TONE_CONVERTER_ONNX,
        input_names=["audio_src", "src_se", "tgt_se", "tau"],
        output_names=["audio_out"],
        dynamic_axes={
            "audio_src": {1: "src_length"},
            "audio_out": {1: "out_length"},
        },
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )
    sz = os.path.getsize(TONE_CONVERTER_ONNX) / 1e6
    print(f"Exported. Size: {sz:.1f} MB")
"""))

cells.append(md("""\
## 4. Export Speaker Encoder (SE Extractor) to ONNX

Takes a reference audio clip (5-30 seconds at 24 kHz) and produces a 256-dim speaker embedding. Run ONCE per user (on first reference recording), then cache. The result is what gets fed to the Tone Color Converter at every synthesis call.
"""))

cells.append(code("""\
SE_EXTRACTOR_ONNX = os.path.join(WORK_DIR, "pocket_tts_se_extractor.onnx")

# The SE extractor is a sub-component of the ToneColorConverter
# model (the encoder side of the variational autoencoder). Pull
# it out and trace separately.

class SEExtractorWrapper(nn.Module):
    \"\"\"Extracts a (1, 256, 1) speaker embedding from raw audio.\"\"\"
    def __init__(self, tcc_model): super().__init__(); self.m = tcc_model
    def forward(self, audio):
        # audio: (1, T) at 24 kHz. Returns: (1, 256, 1).
        return self.m.ref_enc(audio.transpose(0, 1).unsqueeze(0))   # convert (1,T)->(1,1,T) for the encoder

# Trace with dummy data.
wrapper = SEExtractorWrapper(tcc.model).eval()
dummy_ref = torch.randn(1, 24000 * 5)   # 5-second reference

with torch.no_grad():
    try:
        emb = wrapper(dummy_ref)
        print(f"Sanity forward OK. Embedding shape: {tuple(emb.shape)} (expect (1, 256, 1))")
    except Exception as e:
        print(f"SE extractor sanity failed: {e}")
        print("The ref_enc API may need a different shape. Inspect tcc.model.ref_enc.")
        emb = None

if emb is not None:
    print(f"Exporting to {SE_EXTRACTOR_ONNX} ...")
    torch.onnx.export(
        wrapper, dummy_ref, SE_EXTRACTOR_ONNX,
        input_names=["audio"],
        output_names=["embedding"],
        dynamic_axes={"audio": {1: "audio_length"}},
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )
    sz = os.path.getsize(SE_EXTRACTOR_ONNX) / 1e6
    print(f"Exported. Size: {sz:.1f} MB")
"""))

cells.append(md("""\
## 5. Export MeloTTS base ONNX

MeloTTS produces the base speech that the Tone Color Converter then transforms. ~70 MB ONNX.
"""))

cells.append(code("""\
sys.path.insert(0, MELOTTS_DIR)
from melo.api import TTS as MeloTTS

BASE_TTS_ONNX = os.path.join(WORK_DIR, "pocket_tts_base.onnx")

print(f"Loading MeloTTS-English ...")
melo = MeloTTS(language="EN", device="cpu")
melo.model.eval()
print("Loaded.")
print(f"Available speakers: {list(melo.hps.data.spk2id.keys())}")
"""))

cells.append(code("""\
class MeloTTSWrapper(nn.Module):
    \"\"\"Wraps MeloTTS.model.infer() for ONNX export.

    Inputs:
      input_ids: (1, T_text)  int64
      speaker_id: scalar int64 (which preset base speaker to use)
      speed: scalar float (1.0 = normal)
    Output:
      audio: (1, T_audio)  float32 at 24 kHz
    \"\"\"
    def __init__(self, melo_model, hps):
        super().__init__()
        self.model = melo_model
        self.hps = hps

    def forward(self, input_ids, speaker_id, speed):
        x_tst = input_ids
        x_tst_lengths = torch.tensor([input_ids.shape[1]], dtype=torch.long)
        sid = speaker_id
        audio = self.model.infer(
            x_tst, x_tst_lengths, sid, tone=None, language=None, bert=None,
            ja_bert=None, noise_scale=0.667, length_scale=1.0 / speed,
            noise_scale_w=0.8, sdp_ratio=0.2,
        )[0][0, 0]
        return audio.unsqueeze(0)


# Build dummy input via the tokenizer.
import re
from melo.text.cleaner import clean_text
from melo.text import cleaned_text_to_sequence

text = "hello world"
norm_text, phones, tones, word2ph = clean_text(text, "EN")
phone_ids = cleaned_text_to_sequence(phones, tones, "EN")[0]
input_ids = torch.tensor([phone_ids], dtype=torch.long)
print(f"Dummy phone_ids shape: {input_ids.shape}")

wrapper = MeloTTSWrapper(melo.model, melo.hps).eval()
dummy_sid = torch.tensor(list(melo.hps.data.spk2id.values())[0], dtype=torch.long)
dummy_speed = torch.tensor(1.0)

with torch.no_grad():
    try:
        out = wrapper(input_ids, dummy_sid, dummy_speed)
        print(f"Sanity forward OK. Output shape: {tuple(out.shape)}")
    except Exception as e:
        print(f"MeloTTS sanity failed: {e}")
        out = None

if out is not None:
    print(f"Exporting to {BASE_TTS_ONNX} ...")
    torch.onnx.export(
        wrapper, (input_ids, dummy_sid, dummy_speed), BASE_TTS_ONNX,
        input_names=["input_ids", "speaker_id", "speed"],
        output_names=["audio"],
        dynamic_axes={
            "input_ids": {1: "text_length"},
            "audio":     {1: "audio_length"},
        },
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )
    sz = os.path.getsize(BASE_TTS_ONNX) / 1e6
    print(f"Exported. Size: {sz:.1f} MB")
"""))

cells.append(md("""\
## 6. Tokenizer dump

g2p_en's grapheme→phoneme→ID map. Pure Python on Kaggle, but we serialize the ID table so the Android side can do the lookup without re-running g2p_en (which depends on NLTK).

For arbitrary text on-device, the cleanest path is to port g2p_en to Kotlin (~500 lines) or run it server-side. As an interim, we ship the phoneme→ID table and document that text-to-phoneme conversion needs to happen via either an embedded Kotlin g2p port or pre-computed phonemes for fixed strings.
"""))

cells.append(code("""\
import json as _json

VOCAB_PATH = os.path.join(WORK_DIR, "pocket_tts_vocab.json")

# MeloTTS exposes its symbol table directly.
from melo.text.symbols import symbols, language_id_map, num_tones

vocab_table = {sym: i for i, sym in enumerate(symbols)}
with open(VOCAB_PATH, "w", encoding="utf-8") as f:
    _json.dump({
        "tokenizer_class": "MeloTTS_g2p_en",
        "vocab_size": len(symbols),
        "vocab": vocab_table,
        "sample_rate": 24000,
        "speaker_dim": 256,
        "language_id_map": language_id_map,
        "num_tones": num_tones,
        "available_base_speakers": list(melo.hps.data.spk2id.keys()),
    }, f, indent=2, ensure_ascii=False)
print(f"Wrote vocab ({len(symbols)} symbols) to {VOCAB_PATH}")
"""))

cells.append(md("""\
## 7. End-to-end smoke test

Use the original OpenVoice Python API (not the exported ONNX) to verify the conversion actually works on a reference clip. Saves audio for spot-checking before uploading.
"""))

cells.append(code("""\
import soundfile as sf
import numpy as np

SMOKE_BASE_WAV = os.path.join(WORK_DIR, "pocket_tts_smoketest_base.wav")
SMOKE_CONV_WAV = os.path.join(WORK_DIR, "pocket_tts_smoketest_converted.wav")

# Synthesize base TTS via MeloTTS Python API.
try:
    speaker_ids = melo.hps.data.spk2id
    test_text = "Hello, this is a test of the OpenVoice cloning pipeline. The user's banked voice should replace this default voice in the converted output."
    melo.tts_to_file(test_text, speaker_ids["EN-Default"], SMOKE_BASE_WAV, speed=1.0)
    print(f"Base synthesis OK -> {SMOKE_BASE_WAV}")
except Exception as e:
    print(f"Base synthesis failed: {e}")

# Convert via Tone Color Converter (uses a built-in default reference
# for the test; the real user reference would be supplied by the
# Android app).
try:
    ref_audio = os.path.join(ov_local, "base_speakers", "ses", "en-default.pth")
    if not os.path.exists(ref_audio):
        ref_audio = None
        print("No bundled reference found; skipping conversion smoke test")
    else:
        # Load source SE from base speaker embedding.
        base_se = torch.load(ref_audio, map_location="cpu", weights_only=False).unsqueeze(0)
        # Target SE = same as source (identity conversion as a smoke test).
        tcc.convert(
            audio_src_path=SMOKE_BASE_WAV,
            src_se=base_se,
            tgt_se=base_se,
            output_path=SMOKE_CONV_WAV,
            message="@MyShell",
        )
        print(f"Conversion (identity) OK -> {SMOKE_CONV_WAV}")
except Exception as e:
    print(f"Conversion smoke test failed: {e}")
"""))

cells.append(md("""\
## 8. Upload to HF
"""))

cells.append(code("""\
from huggingface_hub import HfApi, create_repo

REPO = "HereLiesAz/liperty-pocket-tts"
create_repo(REPO, repo_type="model", private=False, exist_ok=True)
api = HfApi()
paths_to_upload = [
    BASE_TTS_ONNX, TONE_CONVERTER_ONNX, SE_EXTRACTOR_ONNX,
    VOCAB_PATH, SMOKE_BASE_WAV, SMOKE_CONV_WAV,
]
for path in paths_to_upload:
    if not os.path.exists(path): continue
    sz_kb = os.path.getsize(path) // 1024
    api.upload_file(
        path_or_fileobj=path,
        path_in_repo=os.path.basename(path),
        repo_id=REPO, repo_type="model",
        commit_message=f"OpenVoice v2: {os.path.basename(path)} ({sz_kb} KB)",
    )
    print(f"Uploaded {os.path.basename(path)} ({sz_kb} KB)")
print()
print(f"All assets at: https://huggingface.co/{REPO}")
print()
print("Android-side rework needed (separate commit):")
print("  PocketTTSEngine.kt now has three ONNX sessions instead of two:")
print("    base_tts.onnx          (text -> generic-voice waveform)")
print("    se_extractor.onnx      (ref audio -> 256-d speaker embedding)")
print("    tone_converter.onnx    (waveform + src_se + tgt_se -> user-voice waveform)")
print()
print("  Voice profile lifecycle:")
print("    1. User records 5-30 sec of clean reference audio (NEW UI).")
print("    2. App runs se_extractor once -> caches 256-d embedding.")
print("    3. Every synthesis: base_tts -> tone_converter with cached emb.")
print()
print("  Sample rate: 24 kHz (NOT 22050, NOT 16000).")
print("  No espeak-ng dependency. Tokenization via vocab.json + on-device g2p_en port.")
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
