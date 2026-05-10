"""Generate tools/export_avhubert_to_onnx.ipynb.

Exports Meta's `large_vox_iter5.pt` AV-HuBERT visual encoder to ONNX
so Liperty can evaluate it as a candidate V3 backend (see
`docs/AVHUBERT_V3_BACKEND.md`).

The notebook:
1. Detects the environment (Kaggle / Colab / local).
2. Installs fairseq + AV-HuBERT deps. AV-HuBERT was developed against
   PyTorch 1.10ish via fairseq 0.12.x. On Kaggle's torch 2.10 image,
   we pin a known-compatible fairseq commit and fall back to a torch
   downgrade if the pinned commit doesn't import.
3. Clones `facebookresearch/av_hubert` (ships the model class and
   custom layers; fairseq doesn't include AV-HuBERT itself).
4. Pulls `large_vox_iter5.pt` from the public mirror at
   `HereLiesAz/liperty-avhubert-encoder` (no academic license needed
   for the *weights*, just for re-training data).
5. Loads the checkpoint via `fairseq.checkpoint_utils.load_model_ensemble_and_task`,
   which yields the full audio-visual model.
6. Builds a video-only wrapper around `model.encoder` so the ONNX
   trace doesn't include the audio branch / mask logic / output
   projections (we just want the visual hidden states).
7. Exports to ONNX with a dynamic time axis and a static
   (B=1, C=1, H=88, W=88) frame layout. AV-HuBERT was trained on
   88x88 mouth ROIs (matches Liperty's existing crop pipeline).
8. Parity-checks: PyTorch encoder vs ONNX runtime on a dummy
   tensor. Aborts loudly if max abs diff >1e-2 (some looseness is
   OK because of fp16 internals; large diff = export bug).
9. Uploads the resulting `.onnx` to `HereLiesAz/liperty-avhubert-encoder`.

Risk: this is the experimental gate for the V3 backend. fairseq's
transformer encoder uses dynamic control flow for masking and
positional encoding that often refuses to ONNX-trace. If that
happens, we either patch the model class to use static shapes, or
write a clean-room re-implementation of the encoder forward in plain
PyTorch (then transfer the state dict to it).

Run with:  python tools/_build_export_avhubert_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "export_avhubert_to_onnx.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.splitlines(keepends=True)}


cells: list[dict] = []

cells.append(md("""\
# Export AV-HuBERT large (`large_vox_iter5`) -> ONNX for Liperty V3

This is the experimental gate for the V3 backend (see `docs/AVHUBERT_V3_BACKEND.md`). If the encoder traces cleanly, we have a real candidate. If it doesn't, V3 needs a clean-room re-implementation of the encoder forward in plain PyTorch.

**Inputs:**
- `large_vox_iter5.pt` (3.91 GB) from `HereLiesAz/liperty-avhubert-encoder` (the public mirror of Meta's CDN copy).

**Outputs:**
- `avhubert_visual_encoder.onnx` uploaded back to `HereLiesAz/liperty-avhubert-encoder`.
- Console parity report (max abs diff between PyTorch and ONNX outputs on a dummy tensor).

**Caveats:**
- AV-HuBERT was developed against fairseq 0.12.x and torch 1.10ish. On Kaggle's torch 2.10 image we pin a known-compatible fairseq commit and downgrade torch if the import fails.
- This exports only the encoder, not the LMDecoder. The 12.6% WER reported by LMD-VSR (ICCV 2023) requires the LMDecoder at inference; without it, the encoder + a simple CTC head is closer to ~25-30% WER. A separate notebook will export the LMDecoder once this one works.
"""))

cells.append(md("""\
## 1. Environment + torch downgrade

**Required before fairseq install.** The vendored fairseq commit AV-HuBERT pins (`afc77bdf...`) does not finish its editable install on torch 2.5+, leaving `fairseq.__file__ = None` (PEP 420 namespace package). Downgrade torch FIRST, then restart the kernel before continuing.

If you skip this and run the notebook end-to-end, it will fail at the `import avhubert` step with `ImportError: cannot import name 'utils' from 'fairseq'`.
"""))

cells.append(code("""\
%%capture
!pip install -q torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \\
    --index-url https://download.pytorch.org/whl/cu118
print("torch downgraded. RESTART THE KERNEL NOW (Run -> Restart Session) before continuing.")
"""))

cells.append(md("""\
## 2. Environment detection (after restart)
"""))

cells.append(code("""\
import os, sys, subprocess
import torch

IS_KAGGLE = os.path.exists("/kaggle/working") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ
try:
    import google.colab  # noqa
    IS_COLAB = True
except ImportError:
    IS_COLAB = False
ENV = "kaggle" if IS_KAGGLE else "colab" if IS_COLAB else "local"
print(f"Environment: {ENV}")
print(f"Python: {sys.version.split()[0]}, PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
"""))

cells.append(md("""\
## 2. Clone AV-HuBERT (and its vendored fairseq) and install
"""))

cells.append(code("""\
WORK_DIR = "/kaggle/working/work" if IS_KAGGLE else "/content/work"
os.makedirs(WORK_DIR, exist_ok=True)

AVHUBERT_DIR = os.path.join(WORK_DIR, "av_hubert")
if not os.path.exists(AVHUBERT_DIR):
    !git clone --depth 1 https://github.com/facebookresearch/av_hubert.git {AVHUBERT_DIR}
    # Init the fairseq submodule that av_hubert ships
    !cd {AVHUBERT_DIR} && git submodule init && git submodule update
print(f"AV-HuBERT checkout: {AVHUBERT_DIR}")
print("  contents:", os.listdir(AVHUBERT_DIR)[:10])
"""))

cells.append(code("""\
%%capture
# fairseq from av_hubert's vendored submodule. AV-HuBERT pins a
# specific fairseq commit; using that one avoids API drift surprises.
!pip install --editable {AVHUBERT_DIR}/fairseq
!pip install -q \\
    "huggingface_hub>=0.27,<1.0" \\
    "onnx>=1.16" \\
    "onnxruntime>=1.18" \\
    "numpy>=1.24" \\
    "scipy" \\
    "scikit-learn" \\
    "python_speech_features" \\
    "sentencepiece"
print("Deps installed.")
"""))

cells.append(code("""\
# Add the av_hubert/ parent directory (NOT av_hubert/avhubert/) to
# PYTHONPATH so `import avhubert` resolves the package directory.
sys.path.insert(0, AVHUBERT_DIR)

# Sanity-check that fairseq is a proper package (not a namespace package).
import fairseq
if fairseq.__file__ is None:
    raise RuntimeError(
        "fairseq imported as a namespace package. The editable install "
        "didn't finish, almost always because torch is too new for the "
        "vendored fairseq commit. Did you skip the torch downgrade in "
        "cell 1? Restart the kernel and start over."
    )

import avhubert  # noqa: F401
print(f"avhubert package importable. fairseq.__file__ = {fairseq.__file__}")
"""))

cells.append(md("""\
## 3. HuggingFace auth + pull the checkpoint
"""))

cells.append(code("""\
from huggingface_hub import login, whoami, hf_hub_download

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

cells.append(code("""\
HF_REPO = "HereLiesAz/liperty-avhubert-encoder"
ckpt_path = hf_hub_download(repo_id=HF_REPO, filename="large_vox_iter5.pt", local_dir=WORK_DIR)
print(f"Checkpoint: {ckpt_path} ({os.path.getsize(ckpt_path) / 1e9:.2f} GB)")
"""))

cells.append(md("""\
## 4. Load the model via fairseq

`fairseq.checkpoint_utils.load_model_ensemble_and_task` is the canonical loader. It returns `(models, saved_cfg, task)`. The first model is what we want.

The AV-HuBERT large checkpoint expects a `data` directory containing tokenizer / dictionary files. For ONNX export we don't need the tokenizer, but fairseq insists on it being there. We synthesize a minimal one or pass `arg_overrides` to skip it where possible.
"""))

cells.append(code("""\
import json as _json

# fairseq's task setup wants a data directory with a tokenizer. AV-HuBERT
# at this stage hasn't been finetuned on a transcription task, so the
# pretrained checkpoint's task is `av_hubert_pretraining` and it doesn't
# need a vocab. Pass arg_overrides={'data': '<empty dir>'} just so the
# task instantiation doesn't trip on FileNotFoundError.
empty_data_dir = os.path.join(WORK_DIR, "empty_data")
os.makedirs(empty_data_dir, exist_ok=True)
# AV-HuBERT pretraining task expects nframes.{audio,video} to exist
# even if empty.
for fn in ["nframes.audio", "nframes.video", "test.tsv", "valid.tsv", "train.tsv"]:
    p = os.path.join(empty_data_dir, fn)
    if not os.path.exists(p):
        open(p, "w").close()

from fairseq import checkpoint_utils

print("Loading checkpoint via fairseq...")
models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
    [ckpt_path],
    arg_overrides={
        "data": empty_data_dir,
        "label_dir": empty_data_dir,
        "tokenizer_bpe_model": None,
    },
)
model = models[0]
model.eval()
print(f"Model class: {type(model).__name__}")
print(f"Encoder class: {type(model.encoder).__name__ if hasattr(model, 'encoder') else 'no encoder attr'}")
"""))

cells.append(md("""\
## 5. Build a video-only wrapper around the encoder

The full AV-HuBERT model takes both audio and video plus padding masks. For Liperty we only have video. The encoder itself accepts a `source` dict with `'video'` and `'audio'` keys; we feed `audio=None` so the model uses the visual branch only.

The wrapper takes `(B, 1, T, 88, 88)` float32 video and returns `(B, T_out, hidden)` encoder hidden states.
"""))

cells.append(code("""\
import torch.nn as nn

class AvHubertVisualEncoder(nn.Module):
    \"\"\"Video-only forward of AV-HuBERT large. Strips audio branch and the
    pretraining heads; returns just the encoder hidden states so a
    downstream CTC head or LMDecoder can consume them.\"\"\"
    def __init__(self, full_model):
        super().__init__()
        self.full = full_model

    def forward(self, video):
        # video: (B, 1, T, H, W) float32, mean=0.0 std=1.0 normalized 88x88 grayscale
        # AV-HuBERT's feature_extractor_video expects (B, C=1, T, H, W).
        # The encoder forward signature is `forward_features(source, padding_mask)`
        # but we want hidden states, so we route through extract_features.
        src = {"video": video, "audio": None}
        # AV-HuBERT's `extract_features` returns (features, padding_mask).
        feats, _ = self.full.extract_features(
            source=src,
            padding_mask=None,
            mask=False,
            features_only=True,
            output_layer=None,
        )
        # feats: (B, T_out, hidden=1024 for large)
        return feats


wrapper = AvHubertVisualEncoder(model).eval()

# Smoke-test forward
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wrapper = wrapper.to(device)
T_DUMMY = 50
dummy_video = torch.randn(1, 1, T_DUMMY, 88, 88, device=device, dtype=torch.float32)
with torch.no_grad():
    out_pt = wrapper(dummy_video)
print(f"PyTorch encoder output: {tuple(out_pt.shape)}  dtype={out_pt.dtype}")
"""))

cells.append(md("""\
## 6. Export to ONNX

Dynamic time axis (T can vary per inference). Hidden size and spatial dims are static.
"""))

cells.append(code("""\
ONNX_PATH = os.path.join(WORK_DIR, "avhubert_visual_encoder.onnx")

print("Tracing to ONNX...")
torch.onnx.export(
    wrapper,
    dummy_video,
    ONNX_PATH,
    input_names=["video"],
    output_names=["features"],
    opset_version=17,
    do_constant_folding=True,
    dynamic_axes={
        "video":    {2: "T"},   # batch=1 fixed, channel=1 fixed, T dynamic, H/W fixed
        "features": {1: "T_out"},
    },
    dynamo=False,   # torch 2.10's dynamo exporter is fragile in Kaggle;
                    # the legacy tracer handles AV-HuBERT's transformer better
)
print(f"ONNX written: {ONNX_PATH}  ({os.path.getsize(ONNX_PATH) / 1e6:.0f} MB)")
"""))

cells.append(md("""\
## 7. Parity check: PyTorch vs ONNX
"""))

cells.append(code("""\
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
print("ONNX inputs:")
for i in sess.get_inputs():  print(f"  {i.name}: {i.shape}  {i.type}")
print("ONNX outputs:")
for o in sess.get_outputs(): print(f"  {o.name}: {o.shape}  {o.type}")

# Same input as the PyTorch smoke test
dummy_np = dummy_video.detach().cpu().numpy().astype(np.float32)
ort_out = sess.run(["features"], {"video": dummy_np})[0]
pt_out  = out_pt.detach().cpu().numpy()

print(f"PyTorch out: {pt_out.shape}  range=[{pt_out.min():.3f}, {pt_out.max():.3f}]")
print(f"ONNX out:    {ort_out.shape}  range=[{ort_out.min():.3f}, {ort_out.max():.3f}]")
diff = np.abs(pt_out - ort_out)
print(f"Max abs diff: {diff.max():.6f}")
print(f"Mean abs diff: {diff.mean():.6f}")
assert diff.max() < 1e-2, "ONNX output diverges from PyTorch by >1e-2; export is broken."
print("Parity check PASSED.")
"""))

cells.append(md("""\
## 8. Upload ONNX to the public mirror repo
"""))

cells.append(code("""\
from huggingface_hub import upload_file

upload_file(
    path_or_fileobj=ONNX_PATH,
    path_in_repo="avhubert_visual_encoder.onnx",
    repo_id=HF_REPO,
    repo_type="model",
    commit_message="Add ONNX-exported visual encoder for V3 backend research",
)
print(f"Uploaded -> https://huggingface.co/{HF_REPO}/blob/main/avhubert_visual_encoder.onnx")
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
