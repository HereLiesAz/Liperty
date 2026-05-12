"""Stage 3 of SyncVSR ONNX export: the attention DECODER.

Picks up from a Kaggle session where `model` (the loaded ModelModule)
is already in memory. Wraps the espnet Transformer decoder + its
output projection in a single-step no-KV-cache module:

  inputs:
    prev_tokens       (1, T_dec) int64   -- all previously emitted tokens
    encoder_features  (1, T_enc, D) float32 -- pre-computed encoder output
  output:
    logits            (1, T_dec, V) float32

The wrapper re-runs the full Transformer decoder over all previously
emitted tokens on every call -- N^2 cost per output token but trivial
to export and unchanged across positions. For 16-frame encoder windows
producing < 10 output tokens, the cost is dominated by the encoder
pass anyway.

The Kotlin side (existing AvHubertDecoderSession + Seq2SeqGreedyDecoder
from the V3 backend) consumes this contract verbatim. The orchestrator
calls runStep() in a loop, picking the argmax of logits[:, -1, :] as
the next token until EOS.

Run via the same Kaggle bootstrap pattern as stage2:
  exec(urllib.request.urlopen(
      'https://raw.githubusercontent.com/HereLiesAz/Liperty/<sha>/'
      'tools/syncvsr_export_stage3_decoder.py').read())

Outputs:
  /kaggle/working/syncvsr-export/syncvsr_lrs3_decoder.onnx
  Uploaded to HereLiesAz/liperty-syncvsr-onnx alongside the encoder.
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
ckpt = _caller.get("ckpt")
WORK = Path(_caller.get("WORK", "/kaggle/working/syncvsr-export"))
if model is None:
    raise RuntimeError("stage3: caller globals missing `model` (run stage1 / stage2 first)")
WORK.mkdir(parents=True, exist_ok=True)

# Locate the decoder + output projection on the inner E2E.
e2e = model.model
decoder = getattr(e2e, "decoder", None)
if decoder is None:
    raise RuntimeError("stage3: model.model.decoder not found -- inspect E2E layout")
print(f"[stage3] Decoder type: {type(decoder).__name__}")
print(f"[stage3] Vocab size from CTC head: {e2e.ctc.ctc_lo.out_features}")


class DecoderStepWrapper(nn.Module):
    """No-KV-cache single-step decoder. ONNX-friendly: pure tensor ops,
    no Python control flow that depends on input values, no Python
    loops over the decoder layers."""

    def __init__(self, espnet_decoder):
        super().__init__()
        self.decoder = espnet_decoder

    def forward(self, prev_tokens: torch.Tensor, encoder_features: torch.Tensor):
        # prev_tokens:      (B, T_dec)        int64
        # encoder_features: (B, T_enc, D)     float32
        #
        # espnet's TransformerDecoder.forward returns (decoded, olens):
        #   decoded:  (B, T_dec, V)  log_softmax-ed logits
        #   olens:    (B,) ignored at inference
        #
        # The signature differs across espnet variants. Newer ones take
        # (hs_pad, hlens, ys_in_pad, ys_in_lens). Older ones take
        # (memory, memory_mask, ys_in_pad, ys_in_mask). We try both.
        bsz = prev_tokens.size(0)
        t_dec = prev_tokens.size(1)
        t_enc = encoder_features.size(1)
        device = encoder_features.device

        hlens = torch.full((bsz,), t_enc, dtype=torch.long, device=device)
        ys_in_lens = torch.full((bsz,), t_dec, dtype=torch.long, device=device)

        try:
            out = self.decoder(encoder_features, hlens, prev_tokens, ys_in_lens)
        except (TypeError, AttributeError):
            # Older API: build masks explicitly.
            # memory_mask: (B, 1, T_enc) -- all-ones for inference.
            memory_mask = torch.ones(bsz, 1, t_enc, dtype=torch.bool, device=device)
            # ys_mask: (B, T_dec, T_dec) causal lower-triangular.
            ys_mask = torch.tril(
                torch.ones(t_dec, t_dec, dtype=torch.bool, device=device)
            ).unsqueeze(0).expand(bsz, -1, -1)
            out = self.decoder(encoder_features, memory_mask, prev_tokens, ys_mask)

        decoded = out[0] if isinstance(out, tuple) else out
        # decoded: (B, T_dec, V). Return as-is; the runner picks
        # logits[:, -1, :] for next-token argmax.
        return decoded


print("[stage3] Building DecoderStepWrapper ...")
wrapper = DecoderStepWrapper(decoder).eval()

# Probe encoder hidden dim by running the encoder over a 16-frame
# dummy. SyncVSR's encoder takes NTCHW (B, T, C, H, W).
print("[stage3] Probing encoder hidden dim via dummy forward ...")
with torch.no_grad():
    dummy_video = torch.randn(1, 16, 1, 88, 88)
    ilens = torch.full((1,), 16, dtype=torch.long)
    try:
        enc_out = e2e.encoder(dummy_video, ilens)
    except (TypeError, ValueError):
        enc_out = e2e.encoder(dummy_video)
    enc_hidden = enc_out[0] if isinstance(enc_out, tuple) else enc_out
    print(f"[stage3] Encoder output shape: {tuple(enc_hidden.shape)}")
    encoder_dim = enc_hidden.shape[-1]

# Sanity forward.
print("[stage3] Sanity forward through decoder wrapper ...")
SOS = 0  # placeholder; actual SOS id depends on vocab. Just need a
         # consistent value for the trace.
with torch.no_grad():
    prev_tokens = torch.tensor([[SOS]], dtype=torch.long)
    encoder_features = enc_hidden.detach()
    try:
        decoded = wrapper(prev_tokens, encoder_features)
        print(f"[stage3] Decoder wrapper output: {tuple(decoded.shape)}")
    except Exception as e:
        print(f"[stage3] Decoder forward FAILED: {type(e).__name__}: {e}")
        # Print the actual decoder signature so we can adjust.
        import inspect
        print("[stage3] decoder.forward signature:")
        print(inspect.signature(decoder.forward))
        raise

# ONNX export with dynamic axes for T_dec and T_enc.
DECODER_ONNX = WORK / "syncvsr_lrs3_decoder.onnx"
print(f"[stage3] Exporting -> {DECODER_ONNX}")
torch.onnx.export(
    wrapper,
    (prev_tokens, encoder_features),
    str(DECODER_ONNX),
    input_names=["prev_tokens", "encoder_features"],
    output_names=["logits"],
    dynamic_axes={
        "prev_tokens":      {0: "batch", 1: "t_dec"},
        "encoder_features": {0: "batch", 1: "t_enc"},
        "logits":           {0: "batch", 1: "t_dec"},
    },
    opset_version=17,
    do_constant_folding=True,
    dynamo=False,
)
print(f"[stage3] Exported: {DECODER_ONNX} ({DECODER_ONNX.stat().st_size/1e6:.1f} MB)")

# Parity check.
print("[stage3] Parity check ORT vs PyTorch ...")
import onnxruntime as ort
sess = ort.InferenceSession(str(DECODER_ONNX), providers=["CPUExecutionProvider"])
np_prev = prev_tokens.numpy()
np_enc = encoder_features.numpy()
ort_out = sess.run(None, {
    "prev_tokens": np_prev,
    "encoder_features": np_enc,
})[0]
with torch.no_grad():
    pt_out = wrapper(prev_tokens, encoder_features).numpy()
diff = np.abs(pt_out - ort_out)
print(f"[stage3] PT shape: {pt_out.shape}, ORT shape: {ort_out.shape}")
print(f"[stage3] max abs diff: {diff.max():.6f}, mean: {diff.mean():.6f}")

# Upload to HF alongside the encoder.
print("[stage3] Uploading to HereLiesAz/liperty-syncvsr-onnx ...")
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
api.upload_file(
    path_or_fileobj=str(DECODER_ONNX),
    path_in_repo=DECODER_ONNX.name,
    repo_id="HereLiesAz/liperty-syncvsr-onnx",
    repo_type="model",
    commit_message=f"Add SyncVSR attention decoder ({DECODER_ONNX.stat().st_size//1024//1024} MB)",
)
print(f"[stage3] Uploaded -> https://huggingface.co/HereLiesAz/liperty-syncvsr-onnx")
print("[stage3] DONE")
