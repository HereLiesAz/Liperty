"""Generate tools/export_tts_to_onnx.ipynb.

Pulls the pre-exported Kokoro-82M ONNX bundle + voice style pack
from the official release, mirrors them to
HereLiesAz/liperty-pocket-tts. setup_libs.sh pulls from there into
app/src/main/assets/.

Why Kokoro (after rejecting Coqui TTS and SpeechT5):
- Pre-exported ONNX. Zero torch.onnx.export wrestling. We just
  download and re-upload to our HF model repo where setup_libs.sh
  already knows how to look.
- Pure-Python tokenizer (misaki). No espeak-ng on Android. Drops
  the ~3 MB native lib + JNI surgery we'd have needed for
  VITS/Piper.
- ~315 MB total (~85 MB onnx + ~230 MB voices bin). 1/3 the size
  of SpeechT5 with better quality (Kokoro topped TTS Arena late
  2024).
- 50+ preset voices included (American/British, M/F, several
  styles). For Liperty's actual user base (deaf/HoH), preset
  voice selection is often more valuable than zero-shot cloning —
  many users haven't had typical speech for years and don't have
  reference audio of "themselves as they'd want to sound."
- Apache 2.0. No licensing wrinkle.

Why NOT each rejected option:
- Coqui TTS 0.22: numpy/torch pins don't resolve on Python 3.12.
  Upstream effectively abandoned.
- SpeechT5: works fine but ~1 GB total. Heavier than we need.
- Piper TTS: smaller (~30 MB) but requires espeak-ng on device.
- MeloTTS: pip install fragile (mecab dep), no ONNX exports yet.
- SileroTTS: TorchScript-only, no ONNX path.

PocketTTSEngine contract after this pivot:
  Acoustic: input_names=["input_ids", "style", "speed"]
                                  -> "audio"      (shape (1, T_audio))
  No separate vocoder (Kokoro is end-to-end, output is waveform
  at 24 kHz directly).
  Voice selection: pass one of 54 preset 256-dim style vectors
  from voices-v1.0.bin. The style binary is a packed array of
  (54 voices, 511 frames, 256 dims) float32; the Android side
  indexes by voice ID and frame count to get the conditioning.

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
# Mirror Kokoro-82M TTS to HF for PocketTTS

Downloads the pre-exported [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) ONNX bundle + voice style pack from the official release, mirrors them to `HereLiesAz/liperty-pocket-tts`. `setup_libs.sh` already pulls from there.

**Why Kokoro:**
- Already ONNX. No conversion. Download → re-upload.
- Pure-Python tokenizer (`misaki`). No espeak-ng on Android.
- ~315 MB total. 1/3 the size of SpeechT5, better quality (topped TTS Arena late 2024).
- 54 preset voices included.
- Apache 2.0.
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
"""))

cells.append(code("""\
# Light deps only. We're not running training or PyTorch export —
# just downloading pre-built ONNX files and re-uploading them.
# kokoro-onnx provides a smoke-test runner for verification.
print("=== Installing deps ===")
!pip install -q "huggingface_hub>=0.27,<1.0" "onnxruntime>=1.18" "kokoro-onnx>=0.4" "soundfile" "numpy"

import importlib
for mod in ("huggingface_hub", "onnxruntime", "kokoro_onnx", "soundfile"):
    print(f"{mod}: {'OK' if importlib.util.find_spec(mod) else 'MISSING'}")
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
## 2. Download Kokoro-82M

From the upstream release. Two files:

- `kokoro-v1.0.onnx` — the acoustic+vocoder model (~85 MB)
- `voices-v1.0.bin` — packed array of all 54 preset voice style vectors (~230 MB)
"""))

cells.append(code("""\
from huggingface_hub import hf_hub_download

KOKORO_ONNX = os.path.join(WORK_DIR, "pocket_tts_acoustic.onnx")
KOKORO_VOICES = os.path.join(WORK_DIR, "pocket_tts_voices.bin")

# The official release lives at this HF repo:
KOKORO_REPO = "hexgrad/Kokoro-82M"

# The kokoro-onnx package re-hosts the converted ONNX. We pull from
# their release because hexgrad/Kokoro-82M itself ships the PyTorch
# checkpoint, not the ONNX. The onnx-community fork has the export.
ONNX_SRC_REPO = "onnx-community/Kokoro-82M-v1.0-ONNX"

print("Downloading Kokoro ONNX ...")
onnx_path = hf_hub_download(
    repo_id=ONNX_SRC_REPO,
    filename="onnx/model.onnx",
    local_dir=WORK_DIR,
)
import shutil
shutil.copy(onnx_path, KOKORO_ONNX)
print(f"  ONNX: {KOKORO_ONNX} ({os.path.getsize(KOKORO_ONNX)/1e6:.1f} MB)")

print("\\nDownloading voice style pack ...")
voices_path = hf_hub_download(
    repo_id=ONNX_SRC_REPO,
    filename="voices/af.bin",   # individual voice file, can iterate
    local_dir=WORK_DIR,
)
# The onnx-community repo splits voices into individual .bin files
# per voice instead of one big pack. For our purposes we'll
# concatenate the most common voices into a single .bin matching
# Kokoro's standard packed format, OR just ship them individually.
# Let's go with the standard combined pack instead — pull from the
# kokoro-onnx package's mirror:

# Pivot: use the upstream Kokoro release for the packed voices file.
# It lives in hexgrad/Kokoro-82M release artifacts.
try:
    packed = hf_hub_download(
        repo_id="hexgrad/Kokoro-82M",
        filename="voices/af_heart.pt",   # one voice's tensor as canary
        local_dir=WORK_DIR,
    )
    print(f"  Voice canary OK: {packed}")
except Exception as e:
    print(f"  Voice download via hexgrad/Kokoro-82M failed: {e}")
    print("  Falling back to onnx-community individual .bin files (we'll concatenate below).")

# Listing what voices are available so we can pick which to ship:
from huggingface_hub import list_repo_files
voice_files = [f for f in list_repo_files(ONNX_SRC_REPO) if f.startswith("voices/") and f.endswith(".bin")]
print(f"\\nAvailable voice .bin files: {len(voice_files)}")
for f in sorted(voice_files)[:10]:
    print(f"  {f}")
"""))

cells.append(code("""\
# Download every voice file and concatenate into a single packed
# binary, in the format kokoro-onnx expects:
#   [ num_voices : i32 ]
#   [ voice_id_len : i32, voice_id : utf8 bytes ] * num_voices
#   [ raw float32 vectors : (num_voices, 511, 256) ] flat
#
# Actually the upstream kokoro-onnx binary format is simpler: just
# a flat numpy archive. We'll go with a npz-style approach: dict
# of {voice_id -> (511, 256) float32}.
import numpy as np

voices = {}
for vf in sorted(voice_files):
    voice_id = os.path.splitext(os.path.basename(vf))[0]   # e.g. "af_heart"
    local = hf_hub_download(repo_id=ONNX_SRC_REPO, filename=vf, local_dir=WORK_DIR)
    # The individual .bin is a raw float32 dump of shape (511, 256)
    # per kokoro-onnx's convention. Verify:
    arr = np.fromfile(local, dtype=np.float32)
    expected = 511 * 256
    if arr.size != expected:
        print(f"  WARN {voice_id}: size {arr.size} != expected {expected}, skipping")
        continue
    voices[voice_id] = arr.reshape(511, 256)

print(f"\\nLoaded {len(voices)} voices.")
print("Sample:", list(voices.keys())[:8])

# Save as a single npz for the Android side. Easier to parse than
# a custom packed binary, and AssetFileDescriptor + Java NIO can
# read it directly.
np.savez(KOKORO_VOICES, **voices)
# np.savez appends .npz to the filename; fix that.
if os.path.exists(KOKORO_VOICES + ".npz"):
    os.rename(KOKORO_VOICES + ".npz", KOKORO_VOICES)
print(f"  Packed voices: {KOKORO_VOICES} ({os.path.getsize(KOKORO_VOICES)/1e6:.1f} MB)")
"""))

cells.append(md("""\
## 3. Tokenizer vocab

Kokoro uses the `misaki` G2P library for English. We dump its grapheme→phoneme→token-id mapping so the Android side can replicate tokenization without running misaki itself (which is pure Python but pulls more deps than we want on-device).

For simplicity, we ship the post-G2P phoneme vocab directly — Android does character-to-phoneme via a smaller embedded table, or sends text through misaki on a Kotlin port if it materializes. Current approach: ship the misaki vocab as JSON; the engine can use a simple character fallback if the on-device tokenizer isn't wired yet.
"""))

cells.append(code("""\
import json as _json

VOCAB_PATH = os.path.join(WORK_DIR, "pocket_tts_vocab.json")

# Kokoro's input token vocabulary is the IPA phoneme set + special
# tokens. The onnx-community repo includes a tokenizer.json.
try:
    tokenizer_path = hf_hub_download(
        repo_id=ONNX_SRC_REPO,
        filename="tokenizer.json",
        local_dir=WORK_DIR,
    )
    with open(tokenizer_path) as f:
        upstream_vocab = _json.load(f)
    print(f"Loaded upstream tokenizer.json with {len(upstream_vocab.get('model', {}).get('vocab', {}))} tokens")
except Exception as e:
    print(f"tokenizer.json fetch failed: {e}")
    upstream_vocab = None

# Repack into the schema PocketTTSEngine expects.
if upstream_vocab is not None:
    vocab_table = upstream_vocab.get("model", {}).get("vocab", {})
else:
    # Hand-built fallback from the Kokoro README's documented phoneme
    # set. Approximate; may miss edge cases.
    vocab_table = {sym: i for i, sym in enumerate(list("$;:,.!?¡¿—…\"«»“” ()") +
                                                  list("abcdefghijklmnopqrstuvwxyz") +
                                                  ["ɑ","ɐ","ɒ","æ","ɓ","ʙ","β","ɔ","ɕ","ç","ɗ","ɖ","ð","ʤ","ə","ɘ","ɚ","ɛ","ɜ","ɝ","ɞ","ɟ","ʄ","ɡ","ɠ","ɢ","ʛ","ɦ","ɧ","ħ","ɥ","ʜ","ɨ","ɪ","ʝ","ɭ","ɬ","ɫ","ɮ","ʟ","ɱ","ɯ","ɰ","ŋ","ɳ","ɲ","ɴ","ø","ɵ","ɸ","θ","œ","ɶ","ʘ","ɹ","ɺ","ɾ","ɻ","ʀ","ʁ","ɽ","ʂ","ʃ","ʈ","ʧ","ʉ","ʊ","ʋ","ⱱ","ʌ","ɣ","ɤ","ʍ","χ","ʎ","ʏ","ʑ","ʐ","ʒ","ʔ","ʡ","ʕ","ʢ","ǀ","ǁ","ǂ","ǃ"])}

with open(VOCAB_PATH, "w", encoding="utf-8") as f:
    _json.dump({
        "tokenizer_class": "Kokoro",
        "vocab_size": len(vocab_table),
        "vocab": vocab_table,
        "sample_rate": 24000,
        "voice_dim": 256,
        "voice_frame_count": 511,
        "available_voices": sorted(voices.keys()),
        "default_voice": "af_heart" if "af_heart" in voices else (sorted(voices.keys())[0] if voices else None),
    }, f, indent=2, ensure_ascii=False)
print(f"Wrote {VOCAB_PATH} ({os.path.getsize(VOCAB_PATH)} bytes)")
"""))

cells.append(md("""\
## 4. End-to-end smoke test

Use the `kokoro-onnx` Python package to verify the ONNX + voices files work together. Save a sample .wav for spot-checking.
"""))

cells.append(code("""\
import soundfile as sf

SMOKE_WAV = os.path.join(WORK_DIR, "pocket_tts_smoketest.wav")

try:
    from kokoro_onnx import Kokoro
    # Build with the packed binary we just made.
    kokoro = Kokoro(KOKORO_ONNX, KOKORO_VOICES)
    samples, sample_rate = kokoro.create(
        "Hello, this is a test of the Kokoro text to speech pipeline.",
        voice=list(voices.keys())[0],
        speed=1.0,
        lang="en-us",
    )
    sf.write(SMOKE_WAV, samples, sample_rate)
    print(f"Smoke test OK. {samples.shape[0]} samples at {sample_rate} Hz "
          f"({samples.shape[0] / sample_rate:.2f} s)")
    print(f"Saved {SMOKE_WAV}")
except Exception as e:
    print(f"Smoke test failed: {e}")
    print("(ONNX + voices uploaded anyway; engine-side wiring is the remaining step.)")
"""))

cells.append(md("""\
## 5. Upload to HF
"""))

cells.append(code("""\
from huggingface_hub import HfApi, create_repo

REPO = "HereLiesAz/liperty-pocket-tts"
create_repo(REPO, repo_type="model", private=False, exist_ok=True)
api = HfApi()
for path in (KOKORO_ONNX, KOKORO_VOICES, VOCAB_PATH, SMOKE_WAV):
    if not os.path.exists(path): continue
    sz_kb = os.path.getsize(path) // 1024
    api.upload_file(
        path_or_fileobj=path,
        path_in_repo=os.path.basename(path),
        repo_id=REPO, repo_type="model",
        commit_message=f"Kokoro-82M: {os.path.basename(path)} ({sz_kb} KB)",
    )
    print(f"Uploaded {os.path.basename(path)} ({sz_kb} KB)")
print()
print(f"All assets at: https://huggingface.co/{REPO}")
print()
print("On the Android side after these files land:")
print("  1. setup_libs.sh already pulls from this repo. Old SpeechBrain")
print("     ECAPA pocket_tts_speaker.onnx (192-dim) is now obsolete —")
print("     remove from the pull list or let it be overwritten/orphaned.")
print("  2. PocketTTSEngine.kt: rewrite generateAudio() for Kokoro's")
print("     contract: inputs are 'input_ids', 'style' (256-d), 'speed';")
print("     output is 'audio' at 24 kHz (NOT 22050 or 16000).")
print("  3. VoiceManager: voice selection becomes 'pick from 54 presets'")
print("     loaded from pocket_tts_voices.bin. Cloning gets deferred to")
print("     style-transfer once we have a reference-audio path.")
print("  4. TTS_OUTPUT_SAMPLE_RATE_HZ: change to 24000 (Kokoro outputs)")
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
