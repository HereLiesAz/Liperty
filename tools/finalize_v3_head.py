"""Standalone finalization: load head_step5000.pt, export to ONNX, upload to HF.

Use when kaggle_avhubert_head_train.py completes training but crashes
at torch.onnx.export (e.g. missing onnxscript on Windows). The training
checkpoint is intact; we just need to redo the export + upload.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn

WORK_DIR = os.environ.get("LIPERTY_WORK_DIR", r"C:\Users\azrie\liperty-v3-train")
_CKPT_DIR = os.path.join(WORK_DIR, "head_ckpt")
# Prefer the validation-best checkpoint; fall back to the last-step ckpt only
# if a best wasn't recorded (e.g. training run without a val split).
_CKPT_BEST = os.path.join(_CKPT_DIR, "head_best.pt")
_CKPT_LAST = os.path.join(_CKPT_DIR, "head_step5000.pt")
CKPT     = os.environ.get("LIPERTY_HEAD_CKPT",
                          _CKPT_BEST if os.path.exists(_CKPT_BEST) else _CKPT_LAST)
ONNX_OUT = os.environ.get("LIPERTY_HEAD_ONNX", os.path.join(WORK_DIR, "avhubert_v3_phoneme_head.onnx"))
HEAD_REPO = os.environ.get("LIPERTY_HEAD_REPO", "HereLiesAz/liperty-v3-phoneme-head")

PHONEME_VOCAB = ["_", "AA","AE","AH","AO","AW","AY","B","CH","D","DH","EH","ER","EY",
                 "F","G","HH","IH","IY","JH","K","L","M","N","NG","OW","OY","P","R",
                 "S","SH","T","TH","UH","UW","V","W","Y","Z","ZH"]
VOCAB_SIZE = len(PHONEME_VOCAB)
HEAD_DIM, HEAD_LAYERS, HEAD_DROPOUT = 512, 2, 0.1


class PhonemeHead(nn.Module):
    def __init__(self, in_dim=768, hidden=HEAD_DIM, layers=HEAD_LAYERS,
                 vocab=VOCAB_SIZE, dropout=HEAD_DROPOUT):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.rnn = nn.LSTM(in_dim, hidden, num_layers=layers,
                           bidirectional=True, dropout=dropout, batch_first=True)
        self.proj = nn.Linear(2 * hidden, vocab)

    def forward(self, x):
        x = self.norm(x)
        x, _ = self.rnn(x)
        return self.proj(x)


print(f"Loading {CKPT}")
head = PhonemeHead()
state = torch.load(CKPT, map_location="cpu", weights_only=True)
head.load_state_dict(state)
head.eval()
n_params = sum(p.numel() for p in head.parameters())
print(f"Head: {n_params/1e6:.1f}M params")

# Force the legacy TorchScript-based exporter (dynamo=False). The torch 2.10
# default dynamo path needs onnxscript AND chokes on LSTM dynamic_axes (it
# infers static T=50 from the dummy input then errors on the dynamic dim).
# The legacy path handles LSTMs + dynamic_axes correctly and doesn't need
# onnxscript at all.
dummy_feat = torch.randn(1, 50, 768, dtype=torch.float32)
print(f"Exporting to {ONNX_OUT} (legacy tracer)")
torch.onnx.export(
    head,
    dummy_feat,
    ONNX_OUT,
    input_names=["features"],
    output_names=["phoneme_logits"],
    opset_version=17,
    do_constant_folding=True,
    dynamic_axes={"features": {1: "T"}, "phoneme_logits": {1: "T_out"}},
    training=torch.onnx.TrainingMode.EVAL,
    dynamo=False,
)

sz = os.path.getsize(ONNX_OUT) / 1e6
print(f"ONNX: {ONNX_OUT}  ({sz:.1f} MB)")

# Parity check
import onnxruntime as ort
with torch.no_grad():
    pt_out = head(dummy_feat).numpy()
sess = ort.InferenceSession(ONNX_OUT, providers=["CPUExecutionProvider"])
ort_out = sess.run(["phoneme_logits"], {"features": dummy_feat.numpy()})[0]
diff = np.abs(pt_out - ort_out)
print(f"Parity max_diff={diff.max():.6f}  mean_diff={diff.mean():.6f}")

# Upload
token = os.environ.get("HF_TOKEN")
if token:
    from huggingface_hub import login, create_repo, upload_file
    login(token=token, add_to_git_credential=False)
    create_repo(HEAD_REPO, repo_type="model", private=False, exist_ok=True)
    upload_file(
        path_or_fileobj=ONNX_OUT,
        path_in_repo="avhubert_v3_phoneme_head.onnx",
        repo_id=HEAD_REPO,
        repo_type="model",
        commit_message="V3 phoneme head (GRID 3 speakers, 5000 steps, base+vox+433h encoder; finalized post-training)",
    )
    print(f"Uploaded -> https://huggingface.co/{HEAD_REPO}/blob/main/avhubert_v3_phoneme_head.onnx")
else:
    print("No HF_TOKEN — skipping upload")

print("Done.")
