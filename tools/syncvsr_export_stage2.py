"""Stage 2 of SyncVSR ONNX export — picks up where the interactive
Kaggle cells left off, after `model` (the loaded ModelModule) is
populated. Wraps encoder + CTC, exports to ONNX, dumps vocab + meta,
and uploads to HuggingFace.

Run via:
    NB_URL = '...' # unused here
    exec(urllib.request.urlopen(
        'https://raw.githubusercontent.com/HereLiesAz/Liperty/<sha>/'
        'tools/syncvsr_export_stage2.py').read())

Requires in caller globals: model, ckpt, WORK (or defaults).
"""
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys

# Pull state from the caller (Kaggle bootstrap cell's globals).
_caller = sys._getframe(1).f_globals if hasattr(sys, "_getframe") else globals()
model = _caller.get("model")
ckpt = _caller.get("ckpt")
WORK = Path(_caller.get("WORK", "/kaggle/working/syncvsr-export"))
if model is None:
    raise RuntimeError("stage2: caller globals missing `model`")
WORK.mkdir(parents=True, exist_ok=True)


class EncoderCTCWrapper(nn.Module):
    """Encoder + CTC projection head from SyncVSR's E2E. Drops the
    autoregressive decoder, beam search, scorers, and the audio codec.
    Exports cleanly to ONNX with dynamic time + batch axes."""

    def __init__(self, e2e):
        super().__init__()
        self.encoder = e2e.encoder
        self.ctc = e2e.ctc

    def forward(self, video):
        # video: (N, C=1, T, 88, 88) float32, already normalized to the
        # mean/std Auto-AVSR/Chaplin trained on. Espnet's E2E encoder
        # for visual speech takes (xs_pad, ilens) and returns
        # (hs_pad, hs_mask). Pass ilens = T for every batch element
        # since we're working on full windows here.
        ilens = torch.full((video.size(0),), video.size(2),
                           dtype=torch.long, device=video.device)
        try:
            out = self.encoder(video, ilens)
        except (TypeError, ValueError):
            out = self.encoder(video)
        hs_pad = out[0] if isinstance(out, tuple) else out
        # CTC head linear: hidden_dim -> vocab. Returns log-softmax so
        # the Android decoder can sum logs directly (no second softmax).
        logits = self.ctc.ctc_lo(hs_pad)
        return F.log_softmax(logits, dim=-1)


print("[stage2] Building wrapper ...")
wrapper = EncoderCTCWrapper(model.model).eval()

# Sanity check: SyncVSR's bundled E2E expects NTCHW
# (batch, time, channel=1, 88, 88), confirmed from the encoder's
# first conv layer expecting input[*, *, 1, 88, 88] format.
print("[stage2] Sanity forward NTCHW (1, 16, 1, 88, 88) ...")
with torch.no_grad():
    dummy = torch.randn(1, 16, 1, 88, 88)
    try:
        out = wrapper(dummy)
        print(f"[stage2] Output shape: {tuple(out.shape)}")
    except Exception as e:
        print(f"[stage2] NTCHW forward failed ({type(e).__name__}): {e}")
        print("[stage2] Retrying with NCTHW (1, 1, 16, 88, 88) ...")
        dummy = torch.randn(1, 1, 16, 88, 88)
        out = wrapper(dummy)
        print(f"[stage2] NCTHW output shape: {tuple(out.shape)}")

# ONNX export with dynamic batch + time axes.
ONNX_PATH = WORK / "syncvsr_lrs3_visual_ctc.onnx"
print(f"[stage2] Exporting -> {ONNX_PATH}")
torch.onnx.export(
    wrapper,
    (dummy,),
    str(ONNX_PATH),
    input_names=["video"],
    output_names=["logprobs"],
    dynamic_axes={
        "video":    {0: "batch", 2: "time"},
        "logprobs": {0: "batch", 1: "t_out"},
    },
    opset_version=17,
    do_constant_folding=True,
)
print(f"[stage2] Exported: {ONNX_PATH} ({ONNX_PATH.stat().st_size/1e6:.1f} MB)")

# Parity check ORT vs PyTorch.
import onnxruntime as ort
import numpy as np

print("[stage2] Parity check ORT vs PyTorch ...")
sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
ort_out = sess.run(None, {"video": dummy.numpy()})[0]
with torch.no_grad():
    pt_out = wrapper(dummy).numpy()
m = min(pt_out.shape[1], ort_out.shape[1])
diff = np.abs(pt_out[:, :m] - ort_out[:, :m])
print(f"[stage2] PT shape: {pt_out.shape}, ORT shape: {ort_out.shape}")
print(f"[stage2] max abs diff: {diff.max():.6f}, mean: {diff.mean():.6f}")
if diff.max() >= 1e-2:
    print(f"[stage2] WARNING: parity drift larger than 1e-2 -- inspect manually")

# Vocab dump.
vocab_out = WORK / "syncvsr_unigram_units.txt"
with open(vocab_out, "w") as f:
    for tok in model.text_transform.token_list:
        f.write(f"{tok}\n")
print(f"[stage2] Wrote {len(model.text_transform.token_list)} tokens -> {vocab_out}")
print(f"[stage2] First 5: {model.text_transform.token_list[:5]}")
print(f"[stage2] Last 5: {model.text_transform.token_list[-5:]}")

# Metadata.
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
    "vocab_size": len(model.text_transform.token_list),
    "vocab_file": "syncvsr_unigram_units.txt",
    "notes": (
        "Exported by tools/syncvsr_export_stage2.py via Kaggle. "
        "Encoder + CTC head only from SyncVSR's E2E (Vox+LRS2+LRS3 "
        "checkpoint). Decoder + beam search + LM + audio codec all "
        "dropped. Decode greedily or with subword CTC beam search; "
        "KenLM rescoring on top via KenLmScorer."
    ),
}
meta_path = WORK / "syncvsr_export_metadata.json"
meta_path.write_text(json.dumps(metadata, indent=2))
print(f"[stage2] Wrote metadata -> {meta_path}")

# Upload to HF.
print("[stage2] Uploading to HereLiesAz/liperty-syncvsr-onnx ...")
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
print(f"[stage2] Uploaded -> https://huggingface.co/{REPO}")
print("[stage2] DONE")
