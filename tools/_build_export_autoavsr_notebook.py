"""Generate tools/export_autoavsr_to_onnx.ipynb.

A Colab/Kaggle notebook that:
1. Clones Chaplin (vendors ESPnet) for the model definition.
2. Downloads `Amanvir/LRS3_V_WER19.1` model.json + model.pth from HuggingFace.
3. Loads the espnet E2E audio-visual transformer with the checkpoint.
4. Wraps it so visual-only input flows through encoder + CTC head only
   (drops the attention decoder + beam search; Android deploys greedy CTC).
5. Traces and exports the wrapper to ONNX with dynamic time axis.
6. Verifies ONNX vs PyTorch output parity on a dummy input.
7. Uploads ONNX + tokenizer to a private HF model repo so any of the
   user's accounts can pull it.

WER expectation: Chaplin's headline 19.1% WER is with beam search + CTC +
attention scorer + external LM. Greedy-CTC-only on Android will be
materially worse (30-50% WER, depending on the input). Mitigated downstream
by an on-device LLM cleanup step.

Run with:  python tools/_build_export_autoavsr_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "export_autoavsr_to_onnx.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.splitlines(keepends=True)}


cells: list[dict] = []

cells.append(md("""\
# Export `Amanvir/LRS3_V_WER19.1` (Auto-AVSR / VSR-ML) → ONNX for Android

One-shot notebook to materialise an ONNX file the Liperty Android app can load via `OnnxModelEngine`.

**The trade we're making:**

The headline 19.1% WER on LRS3 is achieved with **beam search using both the CTC and attention decoder scorers, plus an external language model**. None of that is going to ONNX cleanly -- the attention decoder uses dynamic control flow, beam search is non-traceable, and the external LM is a separate beast.

So this notebook exports **only the encoder + CTC projection head**. On Android we'll decode greedily over the 5000-unit subword vocabulary. Expected WER on LRS3 with greedy CTC alone: roughly **30-50%** (vs 19.1% headline). That's the price of fitting it on a phone.

A small on-device LLM cleanup pass (Chaplin's actual innovation, using Qwen-3:4B via Ollama) recovers a lot of that lost accuracy. Out of scope for this notebook -- we just need the ONNX file + vocab first.
"""))

cells.append(md("""\
## 1. Setup

Clone Chaplin (which vendors ESPnet's E2E AV transformer). Install deps. Auth with HF.
"""))

cells.append(code("""\
import os, sys, platform
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

cells.append(code("""\
%%capture
!pip install -q \\
    "huggingface_hub>=0.27,<1.0" \\
    "onnx>=1.16" \\
    "onnxruntime>=1.18" \\
    "numpy>=1.24" \\
    "torchvision" \\
    "sentencepiece"
print("Deps installed.")
"""))

cells.append(code("""\
WORK_DIR = "/kaggle/working/work" if IS_KAGGLE else "/content/work"
os.makedirs(WORK_DIR, exist_ok=True)

CHAPLIN_DIR = os.path.join(WORK_DIR, "chaplin")
if not os.path.exists(CHAPLIN_DIR):
    !git clone --depth 1 https://github.com/amanvirparhar/chaplin.git {CHAPLIN_DIR}
sys.path.insert(0, CHAPLIN_DIR)
print(f"Chaplin checkout: {CHAPLIN_DIR}")
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
if not token and IS_COLAB:
    try:
        from google.colab import userdata
        token = userdata.get("HF_TOKEN")
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
## 2. Run config

Set your HuggingFace username so the exported ONNX gets uploaded to a private repo for cross-account retrieval. The vocab file `unigram5000_units.txt` ships with Chaplin and we'll bundle it alongside.
"""))

cells.append(code("""\
# === EDIT ME ===
HF_USER = "CHANGE-ME"
# ===============
assert HF_USER != "CHANGE-ME", "Set HF_USER to your HuggingFace username."

# Destination for the exported ONNX + tokenizer.
HF_EXPORT_REPO = f"{HF_USER}/liperty-autoavsr-onnx"   # private model repo

# Source checkpoint
SRC_REPO = "Amanvir/LRS3_V_WER19.1"

# Export shape -- the model accepts variable-length time axis. We pick a
# representative T for tracing; ONNX dynamic_axes makes T flexible at runtime.
TRACE_TIME = 60          # 2.4 s @ 25 fps
TRACE_HEIGHT = 88
TRACE_WIDTH = 88
print(f"Tracing at T={TRACE_TIME}, H={TRACE_HEIGHT}, W={TRACE_WIDTH}")
"""))

cells.append(md("""\
## 3. Download the checkpoint
"""))

cells.append(code("""\
from huggingface_hub import hf_hub_download

ckpt_dir = os.path.join(WORK_DIR, "ckpt")
os.makedirs(ckpt_dir, exist_ok=True)

model_json = hf_hub_download(repo_id=SRC_REPO, filename="model.json", local_dir=ckpt_dir)
model_pth  = hf_hub_download(repo_id=SRC_REPO, filename="model.pth",  local_dir=ckpt_dir)

print(f"model.json:  {model_json}")
print(f"model.pth:   {model_pth} ({os.path.getsize(model_pth) / 1e6:.1f} MB)")

# Also pull the vocab file from Chaplin's checkout
import shutil
src_vocab = os.path.join(CHAPLIN_DIR, "pipelines", "tokens", "unigram5000_units.txt")
dst_vocab = os.path.join(ckpt_dir, "unigram5000_units.txt")
shutil.copy(src_vocab, dst_vocab)
with open(dst_vocab) as f:
    vocab_lines = [ln.strip() for ln in f if ln.strip()]
print(f"vocab: {len(vocab_lines)} entries (head: {vocab_lines[:5]}, tail: {vocab_lines[-5:]})")
"""))

cells.append(md("""\
## 4. Load the ESPnet model

Mirrors Chaplin's loading path (`pipelines/model.py:AVSR`). The ESPnet E2E model takes a YAML/JSON config of hyperparams, instantiates the transformer encoder + decoder + CTC head, and we then `load_state_dict` from the .pth file.

The model is audio-visual; we're going to use ONLY the visual branch via a wrapper.
"""))

cells.append(code("""\
import argparse
import json as _json

with open(model_json) as f:
    cfg_dict = _json.load(f)

# Some ESPnet config dumps wrap args in a key like "model_conf" or "config".
if not isinstance(cfg_dict, dict):
    raise TypeError("Expected dict from model.json, got " + type(cfg_dict).__name__)
for unwrap_key in ("config", "model_conf", "args"):
    if unwrap_key in cfg_dict and isinstance(cfg_dict[unwrap_key], dict):
        print("Unwrapping nested config key:", unwrap_key)
        cfg_dict = cfg_dict[unwrap_key]
        break

print("Config keys (first 15):", list(cfg_dict.keys())[:15])

args = argparse.Namespace(**cfg_dict)

from espnet.nets.pytorch_backend.e2e_asr_transformer_av import E2E

odim = len(vocab_lines)
print("odim (vocab size):", odim)

# Try common ESPnet E2E constructor signatures in order.
model = None
last_err = None
for sig_name, ctor in [
    ("E2E(odim=, args=)",          lambda: E2E(odim=odim, args=args)),
    ("E2E(idim=80, odim=, args=)", lambda: E2E(idim=80, odim=odim, args=args)),
    ("E2E(80, odim, args)",        lambda: E2E(80, odim, args)),
]:
    try:
        model = ctor()
        print("Constructed via:", sig_name)
        break
    except Exception as e:
        last_err = e
        print("  " + sig_name + " failed:", type(e).__name__, str(e)[:200])

if model is None:
    raise RuntimeError("All E2E constructor signatures failed; last error: " + str(last_err))

# Load weights. Checkpoints sometimes wrap state in 'state_dict' / 'model'.
state = torch.load(model_pth, map_location="cpu", weights_only=False)
if isinstance(state, dict):
    for outer_key in ("state_dict", "model", "model_state_dict"):
        if outer_key in state and isinstance(state[outer_key], dict):
            print("Unwrapping checkpoint key:", outer_key)
            state = state[outer_key]
            break

missing, unexpected = model.load_state_dict(state, strict=False)
print()
print("Loaded checkpoint. missing=" + str(len(missing)) + " unexpected=" + str(len(unexpected)))
if missing[:5]:    print("  first missing:   ", missing[:5])
if unexpected[:5]: print("  first unexpected:", unexpected[:5])
model.eval()
print("Model in eval mode.")
"""))

cells.append(md("""\
## 5. Build a visual-only wrapper

The full E2E model expects (audio, visual) input and runs encode → CTC + attention scorers → beam search. We want a single forward that:

1. Takes only **video** input: `(B, T, 1, 88, 88) float32`, normalized per Chaplin's transforms (mean=0.421, std=0.165).
2. Routes through the visual frontend (3D conv → ResNet → Conformer encoder).
3. Projects through the CTC head to get `(B, T_out, V)` log-softmax logits over the 5000-unit vocab.

We do this by introspecting which submodules ESPnet exposes (`encoder`, `ctc`, `frontend`, `feature_extractor`) and threading the visual path manually. The exact attribute names vary by ESPnet version -- print the model and adjust if needed.
"""))

cells.append(code("""\
# Inspect what's on the model
print("Top-level submodules of the loaded model:")
for name, _ in model.named_children():
    print(f"  {name}")

# Print a quick state-dict overview to find the visual branch
keys = list(state.keys()) if isinstance(state, dict) else []
print()
print("First 30 state-dict keys (helps locate visual submodules):")
for k in keys[:30]:
    print(f"  {k}")
print("...")
print("Last 10 keys:")
for k in keys[-10:]:
    print(f"  {k}")
"""))

cells.append(code("""\
import torch.nn as nn
import torch.nn.functional as F

class VisualOnlyCTC(nn.Module):
    \"\"\"Wraps the ESPnet AV E2E model for visual-only CTC inference.

    Adapt the attribute lookups below if `model.named_children()` shows
    different names. The pattern in modern ESPnet checkpoints is:
        - model.frontend or model.feature_extractor   (3D conv + ResNet)
        - model.encoder                                 (Conformer)
        - model.ctc.ctc_lo                              (CTC linear projection)
    Audio submodules (model.audio_*, model.frontend_audio) are ignored.
    \"\"\"
    def __init__(self, base):
        super().__init__()
        # Best-effort attribute resolution -- print and adjust if needed.
        self.frontend = (
            getattr(base, "frontend", None)
            or getattr(base, "feature_extractor", None)
            or getattr(base, "visual_frontend", None)
        )
        self.encoder = base.encoder
        self.ctc = base.ctc

    def forward(self, video):
        # video: (B, T, 1, 88, 88) float32, pre-normalized
        # ESPnet's conv3d frontend expects (B, T, C, H, W) and emits (B, T_sub, D)
        feats = self.frontend(video)
        # Some frontends return tuples (feats, lengths)
        if isinstance(feats, tuple):
            feats = feats[0]
        # Encoder
        # ESPnet encoder takes (xs_pad, masks=None) and returns (xs, masks)
        # We pass None mask to skip pad masking (not ONNX-friendly anyway).
        try:
            enc_out, _ = self.encoder(feats, None)
        except Exception:
            enc_out = self.encoder(feats)
            if isinstance(enc_out, tuple):
                enc_out = enc_out[0]
        # CTC projection -- ESPnet's CTC has `ctc_lo` (Linear) → softmax
        logits = self.ctc.ctc_lo(enc_out)
        # Return log_softmax for stability (caller does argmax -- equivalent)
        return F.log_softmax(logits, dim=-1)

wrapper = VisualOnlyCTC(model).eval()

# Dummy forward to confirm shapes
dummy = torch.randn(1, TRACE_TIME, 1, TRACE_HEIGHT, TRACE_WIDTH)
with torch.no_grad():
    out = wrapper(dummy)
print(f"Wrapper forward OK. Output: {out.shape}  (expected (1, T_sub, {len(vocab_lines)}))")
"""))

cells.append(md("""\
## 6. Trace and export to ONNX
"""))

cells.append(code("""\
onnx_path = os.path.join(ckpt_dir, "autoavsr_lrs3_visual_ctc.onnx")

with torch.no_grad():
    torch.onnx.export(
        wrapper,
        dummy,
        onnx_path,
        input_names=["video"],
        output_names=["log_probs"],
        dynamic_axes={
            "video":     {0: "batch", 1: "time"},
            "log_probs": {0: "batch", 1: "time_sub"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
print(f"Exported: {onnx_path} ({os.path.getsize(onnx_path) / 1e6:.1f} MB)")
"""))

cells.append(md("""\
## 7. Verify ONNX matches PyTorch
"""))

cells.append(code("""\
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

with torch.no_grad():
    pt_out = wrapper(dummy).numpy()
ort_out = session.run(["log_probs"], {"video": dummy.numpy()})[0]

print(f"PyTorch out: {pt_out.shape}")
print(f"ONNX out:    {ort_out.shape}")
diff = np.abs(pt_out - ort_out)
print(f"max abs diff: {diff.max():.6f}")
print(f"mean abs diff: {diff.mean():.6f}")
print()
if diff.max() < 1e-3:
    print("OK -- exports match within 1e-3.")
else:
    print("WARNING -- diff is larger than 1e-3. Inspect the wrapper's forward path.")

# Also test with a different time length to confirm dynamic axis works
T2 = 100
dummy2 = torch.randn(1, T2, 1, TRACE_HEIGHT, TRACE_WIDTH)
ort_out2 = session.run(["log_probs"], {"video": dummy2.numpy()})[0]
print(f"Dynamic time test (T={T2}): output shape {ort_out2.shape}")
"""))

cells.append(md("""\
## 8. Upload ONNX + vocab to your HF model repo

So any of your Kaggle/Colab accounts (or your local dev machine) can pull the artifacts via a single HF token.
"""))

cells.append(code("""\
from huggingface_hub import HfApi, create_repo, upload_file

api = HfApi()
try:
    create_repo(HF_EXPORT_REPO, repo_type="model", private=True, exist_ok=True)
except Exception as e:
    print(f"create_repo: {e}")

upload_file(
    path_or_fileobj=onnx_path,
    path_in_repo=os.path.basename(onnx_path),
    repo_id=HF_EXPORT_REPO, repo_type="model",
    commit_message="Auto-AVSR LRS3 visual-CTC ONNX export",
)
upload_file(
    path_or_fileobj=dst_vocab,
    path_in_repo="unigram5000_units.txt",
    repo_id=HF_EXPORT_REPO, repo_type="model",
    commit_message="Subword vocab",
)
print(f"\\nUploaded to https://huggingface.co/{HF_EXPORT_REPO}")
print("\\nFor Android: download the .onnx into app/src/main/assets/ and ship the")
print("vocab file alongside (small text file). See the README cell below for the")
print("Android-side integration plan.")
"""))

cells.append(md("""\
## 9. Android-side integration -- what changes

Once the ONNX is in `app/src/main/assets/autoavsr_lrs3_visual_ctc.onnx`, the deployed Android pipeline needs the following changes. Each of these is a separate diff worth committing on its own.

### 9.1 Preprocessing -- different from VideoMAE

| Stage | Current (VideoMAE) | Auto-AVSR target |
|---|---|---|
| ROI source | Full face (extractFaceBoundingBox) | **Mouth** (use existing `extractLipBoundingBox`) |
| Crop size | 224×224 | **88×88** |
| Channels | RGB (3) | **Grayscale (1)** |
| Frame count | Fixed 16 | **Variable** (use whatever the buffer has) |
| Normalization | `pixel / 255` | **`(pixel/255 - 0.421) / 0.165`** |

Concrete changes:
- `MainActivity.kt:535-538` -- `cropSize = 88`, switch back to `extractLipBoundingBox` and `alignAndCropMouth`. Convert to grayscale (R channel only).
- `VSRInference.kt` allocateBuffersIfNeeded -- `numChannels = 1`, fixed-size grayscale buffer.
- `VSRInference.kt` input-write loop -- apply Auto-AVSR normalization stats instead of `pixel/255`.

### 9.2 New `ModelEngine` class

`AutoAvsrEngine : ModelEngine` mirrors `OnnxModelEngine` but reports `getInputLayout() = NTHWC` (the Auto-AVSR ONNX takes `(B, T, 1, 88, 88)` which is NTCHW; we'll have a small permute or an alternative layout enum).

A simpler path: keep `OnnxModelEngine`, just point at the Auto-AVSR ONNX, and treat the input as NTHWC where C=1 (grayscale). VSRInference already handles `numChannels==1` in the NTHWC branch.

### 9.3 Decoder -- drop ARPABET, use subword units

`MLConstants.PHONEME_VOCAB` no longer applies. New:

- `MLConstants.AUTOAVSR_VOCAB` -- load `unigram5000_units.txt` from assets at startup, parse to `List<String>` with index 0 = blank.
- `BeamSearchDecoder` and `GreedyDecoder` -- they take a vocab list as a parameter already; just pass the new vocab.
- `HomopheneCorrector`, `LanguageModel`, `G2PConverter` -- these all assume ARPABET phonemes. **Disable them** when the Auto-AVSR engine is active. Add a config flag in `MainActivity` that selects between the phoneme pipeline (existing VideoMAE / future landmark model) and the subword pipeline (Auto-AVSR).
- Detokenization -- subword units use `▁` (U+2581) as a word-boundary marker. After greedy-CTC produces a sequence of subword indices, join into a single string and replace `▁` with space, then trim.

### 9.4 Expected effective output quality

With CTC-greedy + no LM rescoring:
- Per-character error rate roughly 30-50% on natural speech.
- Output will be partially garbled subword strings.
- Useful as a "draft" the user can edit, or as input to an on-device LLM cleanup step (Phi-3-mini quantized would do this in 1-2 sec on a flagship Snapdragon).

If the raw output is unusable as-is, the next steps in priority order:
1. Add a small rescoring CTC beam search in Kotlin (~50 LOC).
2. Bundle a quantized small LM and add a single-pass rescore.
3. Fine-tune the encoder on personal data (the existing landmark/pixel notebooks let you do this; you'd swap in this checkpoint as the encoder init).
"""))


nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": []},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB, {len(cells)} cells)")
