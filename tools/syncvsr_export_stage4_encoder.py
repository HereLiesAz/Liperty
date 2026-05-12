"""Stage 4: export SyncVSR encoder HIDDEN STATES only (no CTC head).

The existing `syncvsr_lrs3_visual_ctc.onnx` exported by stage 2 bakes
the encoder + CTC linear projection together: input video, output
log-softmax over the 5049-token vocab. That's exactly what the V2
greedy/beam CTC path on Android wants, but it's no use for the
attention decoder, which feeds on pre-CTC hidden features of shape
(B, T, D=768).

This stage exports a parallel ONNX that returns the encoder hidden
states directly:

  input:  video             (B, T, 1, 88, 88) float32   -- NTCHW (auto-detected)
  output: encoder_features  (B, T, D)         float32

Used by the Android seq2seq orchestrator alongside
`syncvsr_lrs3_decoder.onnx` (stage 3): encoder once per window,
then autoregressive decoder loop until EOS.

Run from a fresh Kaggle session where stage 1 (bootstrap) has
already loaded `model` into globals:

    import urllib.request; exec(urllib.request.urlopen(
        'https://raw.githubusercontent.com/HereLiesAz/Liperty/<sha>/'
        'tools/syncvsr_export_stage4_encoder.py').read())
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Pull state from the caller (Kaggle bootstrap cell's globals).
_caller = sys._getframe(1).f_globals if hasattr(sys, "_getframe") else globals()
model = _caller.get("model")
WORK = Path(_caller.get("WORK", "/kaggle/working/syncvsr-export"))
if model is None:
    raise RuntimeError("stage4: caller globals missing `model` (run stage1 first)")
WORK.mkdir(parents=True, exist_ok=True)

e2e = model.model
print(f"[stage4] Encoder type: {type(e2e.encoder).__name__}")


class EncoderHiddenWrapper(nn.Module):
    """Wraps espnet's encoder so its (hs_pad, hs_mask) tuple is reduced
    to just hs_pad -- the only thing ONNX export and the Android
    decoder need."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, video):
        t_dim = video.size(1) if video.size(1) > video.size(2) else video.size(2)
        ilens = torch.full((video.size(0),), t_dim,
                           dtype=torch.long, device=video.device)
        try:
            out = self.encoder(video, ilens)
        except (TypeError, ValueError):
            out = self.encoder(video)
        # out is (hs_pad, hs_mask) or just hs_pad.
        return out[0] if isinstance(out, tuple) else out


print("[stage4] Building encoder-hidden wrapper ...")
wrapper = EncoderHiddenWrapper(e2e.encoder).eval()

# Auto-detect NTCHW vs NCTHW the same way stage2 did.
for layout_name, dummy_shape in [("NTCHW", (1, 16, 1, 88, 88)), ("NCTHW", (1, 1, 16, 88, 88))]:
    print(f"[stage4] Trying {layout_name} dummy {dummy_shape} ...")
    dummy = torch.randn(*dummy_shape)
    try:
        with torch.no_grad():
            out = wrapper(dummy)
        print(f"[stage4]   {layout_name} worked. Hidden shape: {tuple(out.shape)}")
        INPUT_LAYOUT = layout_name
        break
    except RuntimeError as e:
        print(f"[stage4]   {layout_name} failed: {e}")
else:
    raise RuntimeError("Neither layout worked")

# ONNX export.
ENCODER_ONNX = WORK / "syncvsr_lrs3_encoder.onnx"
time_axis_idx = 1 if INPUT_LAYOUT == "NTCHW" else 2
print(f"[stage4] Exporting -> {ENCODER_ONNX}")
torch.onnx.export(
    wrapper,
    (dummy,),
    str(ENCODER_ONNX),
    input_names=["video"],
    output_names=["encoder_features"],
    dynamic_axes={
        "video":            {0: "batch", time_axis_idx: "time"},
        "encoder_features": {0: "batch", 1: "t_enc"},
    },
    opset_version=17,
    do_constant_folding=True,
    dynamo=False,
)
print(f"[stage4] Exported: {ENCODER_ONNX} ({ENCODER_ONNX.stat().st_size/1e6:.1f} MB)")

# Parity check.
import onnxruntime as ort
print("[stage4] Parity check ORT vs PyTorch ...")
sess = ort.InferenceSession(str(ENCODER_ONNX), providers=["CPUExecutionProvider"])
ort_out = sess.run(None, {"video": dummy.numpy()})[0]
with torch.no_grad():
    pt_out = wrapper(dummy).numpy()
diff = np.abs(pt_out - ort_out)
print(f"[stage4] PT shape: {pt_out.shape}, ORT shape: {ort_out.shape}")
print(f"[stage4] max abs diff: {diff.max():.6f}, mean: {diff.mean():.6f}")

# Upload.
print("[stage4] Uploading to HereLiesAz/liperty-syncvsr-onnx ...")
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
api.upload_file(
    path_or_fileobj=str(ENCODER_ONNX),
    path_in_repo=ENCODER_ONNX.name,
    repo_id="HereLiesAz/liperty-syncvsr-onnx",
    repo_type="model",
    commit_message=f"Add SyncVSR encoder hidden states ONNX (no CTC head, {ENCODER_ONNX.stat().st_size//1024//1024} MB)",
)
print(f"[stage4] Uploaded -> https://huggingface.co/HereLiesAz/liperty-syncvsr-onnx")
print(f"[stage4] DONE  (input layout: {INPUT_LAYOUT})")
