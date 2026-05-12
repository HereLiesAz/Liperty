"""Generate tools/export_syncvsr_to_onnx.ipynb.

SyncVSR (KAIST-AILab, Interspeech 2024) is the proposed replacement for
the Auto-AVSR ONNX that ships in Liperty today. Reasons:

  1. Auto-AVSR's deployed export (`Amanvir/LRS3_V_WER19.1`) is producing
     <blank> with very high confidence on every window, even with
     visually-correct mouth crops (verified by dumping the 88x88 crop on
     device + inspecting). 774 MB model, ~6 s/window on Pixel 5 -- not
     viable for real-time even if it worked.

  2. SyncVSR is data-efficient (trained with cross-modal audio-token
     synchronization on top of standard CTC) and the published numbers
     are competitive with Auto-AVSR on LRS3.

  3. The pretrained `Vox+LRS2+LRS3.ckpt` (1.14 GB PyTorch Lightning
     checkpoint, hosted on the SyncVSR releases page) was trained on
     all three of LRS2 + LRS3 + VoxCeleb2 -- a much broader distribution
     than Auto-AVSR's LRS3-only training.

The encoder-only export is the same structural trick as for Auto-AVSR:
strip the autoregressive decoder + beam search + LM, keep the visual
encoder + CTC projection head. We decode greedily / via subword beam
search on Android, with KenLM rescoring on top.

USAGE (Kaggle, GPU T4 / P100 / V100):
  - HF_TOKEN secret enabled.
  - Internet enabled.
  - Paste the notebook into a cell and run top-to-bottom. ~10 min.

OUTPUT:
  - syncvsr_lrs3_visual_ctc.onnx           (encoder + CTC head)
  - syncvsr_unigram_units.txt              (vocab, SentencePiece if applicable)
  - syncvsr_export_metadata.json           (transforms used: mean, std, crop
                                            size, expected input layout)
  - All uploaded to HF model repo `HereLiesAz/liperty-syncvsr-onnx`.

The Android side then swaps `OnnxModelEngine` to load
`syncvsr_lrs3_visual_ctc.onnx` and updates the preprocessing constants
(crop size, mean/std) to match what this notebook records in the
metadata JSON.

Run with:  python tools/_build_export_syncvsr_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "export_syncvsr_to_onnx.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.splitlines(keepends=True)}


cells: list[dict] = []

cells.append(md("""\
# Export SyncVSR `Vox+LRS2+LRS3.ckpt` -> ONNX for Android

Replacement backend for Liperty's Auto-AVSR ONNX. The deployed Auto-AVSR
export was confirmed broken in production (always predicts `<blank>`
with high confidence even on visually-perfect 88x88 mouth crops, and at
~6 s/window on a Pixel 5 it's too slow for real-time anyway). SyncVSR
is data-efficient, trained on the union of VoxCeleb2 + LRS2 + LRS3, and
the encoder portion is structurally similar to Auto-AVSR (ESPnet E2E
transformer) so the export trick carries over.

**The trade:** SyncVSR's headline numbers also assume beam search + CTC
+ attention scorer + LM. ONNX gets the encoder + CTC head only.
Greedy/beam CTC on Android gives a worse WER than the paper headlines.
KenLM rescoring (already shipped in `KenLmScorer`) recovers some of it.
"""))

cells.append(md("## 1. Environment setup"))
cells.append(code("""\
import os, sys, subprocess, json
from pathlib import Path

# Resolve HF token early so we fail fast if it isn't wired up.
if "HF_TOKEN" not in os.environ:
    try:
        from kaggle_secrets import UserSecretsClient
        os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as e:
        raise RuntimeError("HF_TOKEN not in Kaggle secrets") from e

WORK = Path("/kaggle/working/syncvsr-export")
WORK.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)
print("Working dir:", WORK)
"""))

cells.append(code("""\
# Install pinned ML stack. SyncVSR is PyTorch-Lightning-based and uses
# ESPnet's E2E transformer; the requirements file in their repo is the
# authoritative source so we defer the install to the next cell after
# the clone.
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q", "--upgrade",
    "huggingface_hub>=0.27,<1.0",
    "onnx>=1.17", "onnxruntime>=1.20",
])
import torch
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
"""))

cells.append(md("## 2. Clone SyncVSR + install its deps"))
cells.append(code("""\
SYNCVSR_SRC = WORK / "SyncVSR"
if not SYNCVSR_SRC.exists():
    subprocess.check_call([
        "git", "clone", "--depth", "1",
        "https://github.com/KAIST-AILab/SyncVSR.git", str(SYNCVSR_SRC),
    ])

# Install the LRS/video subproject's requirements.
req = SYNCVSR_SRC / "LRS" / "video" / "requirements.txt"
if req.exists():
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "-r", str(req),
    ])
else:
    # Fall back: install the deps we know are needed.
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "pytorch-lightning>=2.0", "espnet", "sentencepiece",
        "av", "torchvision",
    ])

# Make the SyncVSR package importable.
sys.path.insert(0, str(SYNCVSR_SRC / "LRS" / "video"))
print("SyncVSR cloned to:", SYNCVSR_SRC)
"""))

cells.append(md("## 3. Download the pretrained checkpoint"))
cells.append(code("""\
import urllib.request

CKPT_URL = "https://github.com/KAIST-AILab/SyncVSR/releases/download/weight-audio-v1/Vox%2BLRS2%2BLRS3.ckpt"
CKPT_PATH = WORK / "Vox+LRS2+LRS3.ckpt"

if not CKPT_PATH.exists() or CKPT_PATH.stat().st_size < 1_000_000_000:
    print(f"Downloading {CKPT_URL} (1.14 GB) ...")
    urllib.request.urlretrieve(CKPT_URL, CKPT_PATH)
print("Checkpoint:", CKPT_PATH, f"({CKPT_PATH.stat().st_size/1e6:.1f} MB)")
"""))

cells.append(md("""\
## 4. Inspect the checkpoint structure

The PyTorch-Lightning checkpoint stores `state_dict` plus hyperparameters
and optimizer state. We need to figure out the LightningModule class
name and instantiate it so we can hand it the weights.
"""))
cells.append(code("""\
ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
print("Top-level keys:", list(ckpt.keys())[:20])
if "hyper_parameters" in ckpt:
    hp = ckpt["hyper_parameters"]
    print("Hyperparams keys:", list(hp.keys())[:30])
    # The class path tells us which LightningModule to import
    if "_target_" in hp:
        print("Lightning target:", hp["_target_"])
    if "model" in hp:
        print("Model config keys:", list(hp["model"].keys()) if isinstance(hp["model"], dict) else type(hp["model"]))
state = ckpt.get("state_dict", ckpt)
print(f"state_dict entries: {len(state)}")
print("First 10 keys:")
for k in list(state.keys())[:10]:
    v = state[k]
    print(f"  {k}: {tuple(v.shape) if hasattr(v, 'shape') else type(v)}")
"""))

cells.append(md("""\
## 5. Load SyncVSR's LightningModule and apply the weights

The repo's `LRS/video/main.py` instantiates a Lightning module via the
config in `config/lrs3.yaml`. We import that module directly and let
Lightning's `load_from_checkpoint` handle the state-dict mapping.

If the import path differs from the assumption below, the error message
will point us at the correct module path inside `SyncVSR/LRS/video/`.
"""))
cells.append(code("""\
# The model class lives in SyncVSR/LRS/video/src/lightning_module.py
# (or similar). We try a couple of likely import paths and let Python
# tell us which one resolves.
import importlib

candidates = [
    "src.lightning_module.LipReadingModule",
    "src.lightning.LipReadingModule",
    "src.module.LipReadingModule",
    "src.lit.LipReadingModule",
]
LitModule = None
for cand in candidates:
    mod_path, cls_name = cand.rsplit(".", 1)
    try:
        mod = importlib.import_module(mod_path)
        LitModule = getattr(mod, cls_name)
        print(f"Found Lightning module: {cand}")
        break
    except Exception as e:
        print(f"  {cand}: {type(e).__name__}: {e}")

if LitModule is None:
    # Fall back: walk src/ for any *.py with a LightningModule subclass.
    import pkgutil, inspect
    import pytorch_lightning as pl
    for finder, name, ispkg in pkgutil.walk_packages([str(SYNCVSR_SRC / "LRS" / "video" / "src")], prefix="src."):
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        for cname, cls in inspect.getmembers(mod, inspect.isclass):
            if issubclass(cls, pl.LightningModule) and cls is not pl.LightningModule:
                print(f"Candidate: {name}.{cname}")
                if LitModule is None:
                    LitModule = cls

assert LitModule is not None, "Could not locate Lightning module class in SyncVSR/LRS/video/src/"
print("Using:", LitModule)
"""))

cells.append(code("""\
# Instantiate via load_from_checkpoint; Lightning re-creates the module
# from the saved hyperparameters and loads the state_dict.
model = LitModule.load_from_checkpoint(CKPT_PATH, map_location="cpu", strict=False)
model.eval()
print("Loaded.")
print("Model type:", type(model.model).__name__ if hasattr(model, "model") else type(model).__name__)
# Try to print the encoder + CTC head identity if exposed.
for name in ("encoder", "model.encoder", "ctc", "model.ctc"):
    obj = model
    for part in name.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            break
    if obj is not None:
        print(f"  {name}: {type(obj).__name__}")
"""))

cells.append(md("""\
## 6. Wrap encoder + CTC head for ONNX export

ESPnet E2E transformer's encoder takes the video tensor and returns the
encoder hidden states + an output mask. The CTC head projects hidden
states to vocab logits. We bundle these two in a thin nn.Module that
takes the video tensor and returns CTC log-softmax over time:

    inputs:  video    (1, C=1, T, H=88, W=88)   float32
    outputs: logprobs (1, T_out, V)             float32

The actual encoder may use different layouts internally; we conform to
NCTHW input on the wrapper so the Android side doesn't need to change.
"""))
cells.append(code("""\
import torch.nn as nn
import torch.nn.functional as F

class EncoderCTCWrapper(nn.Module):
    def __init__(self, encoder, ctc_head):
        super().__init__()
        self.encoder = encoder
        self.ctc = ctc_head

    def forward(self, video):
        # video: (1, 1, T, 88, 88). ESPnet's E2E expects either NCTHW or
        # NTHW depending on the model -- try the most common signatures.
        # The encoder returns (hidden, ilens) or (hidden, mask).
        try:
            out = self.encoder(video, ilens=None)
        except TypeError:
            out = self.encoder(video)
        if isinstance(out, tuple):
            hidden = out[0]
        else:
            hidden = out
        # hidden: (1, T_out, D). Project to vocab.
        logits = self.ctc.ctc_lo(hidden) if hasattr(self.ctc, "ctc_lo") else self.ctc(hidden)
        return F.log_softmax(logits, dim=-1)

# Resolve encoder + ctc from the loaded model. The exact attribute names
# depend on the SyncVSR LightningModule wrapping ESPnet.
inner = getattr(model, "model", model)
encoder = getattr(inner, "encoder", None)
ctc = getattr(inner, "ctc", None)
assert encoder is not None and ctc is not None, "Need encoder + ctc on model -- inspect printout above"

wrapper = EncoderCTCWrapper(encoder, ctc).eval()
print("Wrapper ready.")
"""))

cells.append(md("## 7. Sanity-check forward pass on dummy input"))
cells.append(code("""\
T = 16
dummy = torch.randn(1, 1, T, 88, 88)
with torch.no_grad():
    out = wrapper(dummy)
print("Output shape:", tuple(out.shape))
# Expected: (1, T_out, V). T_out depends on the encoder's stride.
# V should be the SentencePiece vocab size (typically 5000-ish for LRS3).
"""))

cells.append(md("""\
## 8. Export to ONNX with dynamic time axis

Opset 17 covers everything the encoder + CTC head should need. We mark
the batch and time axes as dynamic so the Android side can pass any
window length.
"""))
cells.append(code("""\
ONNX_PATH = WORK / "syncvsr_lrs3_visual_ctc.onnx"
torch.onnx.export(
    wrapper,
    (dummy,),
    str(ONNX_PATH),
    input_names=["video"],
    output_names=["logprobs"],
    dynamic_axes={
        "video":   {0: "batch", 2: "time"},
        "logprobs": {0: "batch", 1: "t_out"},
    },
    opset_version=17,
    do_constant_folding=True,
)
print(f"Exported: {ONNX_PATH} ({ONNX_PATH.stat().st_size/1e6:.1f} MB)")
"""))

cells.append(md("## 9. Parity check: ONNX vs PyTorch"))
cells.append(code("""\
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
np_input = dummy.numpy()
ort_out = sess.run(None, {"video": np_input})[0]

with torch.no_grad():
    pt_out = wrapper(dummy).numpy()

# Trim to overlapping shapes
m = min(pt_out.shape[1], ort_out.shape[1])
diff = np.abs(pt_out[:, :m] - ort_out[:, :m])
print(f"max abs diff: {diff.max():.6f}, mean abs diff: {diff.mean():.6f}")
print(f"PT  shape: {pt_out.shape}")
print(f"ORT shape: {ort_out.shape}")
assert diff.max() < 1e-2, "PT vs ORT output diverged; export is broken"
print("Parity OK.")
"""))

cells.append(md("## 10. Extract + save vocab"))
cells.append(code("""\
# SyncVSR uses SentencePiece. The tokenizer/spm model should be in the
# loaded checkpoint's hyperparameters or accessible via the LightningModule.
# Save the vocab as a unigram list mirroring Auto-AVSR's
# unigram5000_units.txt format (one token per line, blank first, eos last).
spm_path = None
for k in ("spm_model_path", "tokenizer_path", "unit_path"):
    p = getattr(model, k, None) or (hp.get(k) if isinstance(hp, dict) else None)
    if p and Path(p).exists():
        spm_path = p
        break

vocab_out = WORK / "syncvsr_unigram_units.txt"

if spm_path is not None:
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(spm_path)
    with open(vocab_out, "w") as f:
        for i in range(sp.get_piece_size()):
            f.write(sp.id_to_piece(i) + "\\n")
    print(f"Wrote {sp.get_piece_size()} tokens from {spm_path} -> {vocab_out}")
else:
    # Fallback: write a placeholder so the upload doesn't fail. The
    # Android side won't be able to decode until this is populated.
    print("WARN: SentencePiece model not found. Inspect model attributes:")
    print([a for a in dir(model) if not a.startswith('_')][:30])
    vocab_out.write_text("<blank>\\n<unk>\\n<eos>\\n")
"""))

cells.append(md("## 11. Metadata JSON"))
cells.append(code("""\
# Mirror Auto-AVSR's metadata so the Android side has a single source of
# truth for the preprocessing constants when we swap backends.
metadata = {
    "model_name": "syncvsr_lrs3_visual_ctc",
    "source_checkpoint": "Vox+LRS2+LRS3.ckpt",
    "input_layout": "NCTHW",
    "input_channels": 1,
    "input_height": 88,
    "input_width": 88,
    "pixel_mean": 0.421,
    "pixel_std": 0.165,
    "output_layout": "NTV",
    "output_is_log_softmax": True,
    "blank_index": 0,
    "vocab_file": "syncvsr_unigram_units.txt",
    "notes": (
        "Exported by tools/export_syncvsr_to_onnx.ipynb. Encoder + CTC "
        "head only -- no attention decoder, no beam search, no LM. "
        "Decode greedily or with subword CTC beam search; KenLM "
        "rescoring on top via KenLmScorer."
    ),
}
meta_path = WORK / "syncvsr_export_metadata.json"
meta_path.write_text(json.dumps(metadata, indent=2))
print(json.dumps(metadata, indent=2))
"""))

cells.append(md("## 12. Upload to HuggingFace"))
cells.append(code("""\
from huggingface_hub import login, create_repo, upload_folder

login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
REPO = "HereLiesAz/liperty-syncvsr-onnx"
create_repo(REPO, repo_type="model", private=False, exist_ok=True)
upload_folder(
    folder_path=str(WORK),
    path_in_repo=".",
    repo_id=REPO,
    repo_type="model",
    allow_patterns=["*.onnx", "*.txt", "*.json"],
    commit_message="SyncVSR Vox+LRS2+LRS3 -> ONNX (encoder + CTC head)",
)
print(f"Uploaded -> https://huggingface.co/{REPO}")
"""))


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(notebook, indent=1))
print(f"Wrote {OUT}")
