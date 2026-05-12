"""End-to-end verification: decoder ONNX vs decoder PyTorch on real
GRID clip features.

The encoder ONNX was already parity-checked (max_diff ~4e-4) at export
time. This script tests whether the decoder ONNX produces the same
greedy-decoded token stream (and therefore the same text) as the
fairseq PyTorch decoder, on actual cached features.

If token streams match, the per-position logit drift (max_diff up to
0.34 in the export-time random-tensor parity check) is harmless for
argmax-based decoding. If they diverge, there's a real bug to fix
before Android integration.

Runs inside the same liperty-v3-export Docker image used for the
decoder export. Expects:
  /work/onnx/avhubert_base_vox_433h_decoder.onnx       (mounted from host)
  /work/onnx/avhubert_base_vox_433h_dict.txt           (mounted from host)
  /ckpts/base_vox_433h.pt                              (mounted from host)
  /work/cache/sN_feats.pt                              (mounted from host)
"""

import os
import sys
import time

WORK_DIR     = os.environ.get("V3_WORK_DIR",     "/work")
AVHUBERT_DIR = os.environ.get("V3_AVHUBERT_DIR", os.path.join(WORK_DIR, "av_hubert"))
FAIRSEQ_DIR  = os.environ.get("V3_FAIRSEQ_DIR",  os.path.join(AVHUBERT_DIR, "fairseq"))
LOCAL_PT     = os.environ.get("V3_CKPT_PATH",    "/ckpts/base_vox_433h.pt")
ONNX_PATH    = os.environ.get("V3_DECODER_ONNX", "/work/onnx/avhubert_base_vox_433h_decoder.onnx")
DICT_PATH    = os.environ.get("V3_DECODER_DICT", "/work/onnx/avhubert_base_vox_433h_dict.txt")
CACHE_DIR    = os.environ.get("V3_FEATURE_CACHE","/work/cache")
N_CLIPS      = int(os.environ.get("V3_N_CLIPS", "5"))
MAX_DECODE   = int(os.environ.get("V3_MAX_DECODE", "30"))

os.environ.setdefault("WANDB_MODE", "disabled")

if FAIRSEQ_DIR not in sys.path:
    sys.path.insert(0, FAIRSEQ_DIR)
AVH_PKG = os.path.join(AVHUBERT_DIR, "avhubert")
if AVH_PKG not in sys.path:
    sys.path.insert(0, AVH_PKG)

import torch
import torch.nn as nn
import numpy as np
import onnxruntime as ort

import hubert_pretraining  # noqa: F401
import hubert              # noqa: F401
import hubert_asr          # noqa: F401
from fairseq import checkpoint_utils


# -------------------------------------------------------------------------
# 1. Load PyTorch decoder (full model + apex swap + mask patch — must match
#    the export script exactly so any divergence is from ONNX, not from us)
# -------------------------------------------------------------------------
import tempfile
empty = tempfile.mkdtemp(prefix="empty_avh_")
print(f"Loading PT model from {LOCAL_PT}...")
t0 = time.time()
models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
    [LOCAL_PT],
    arg_overrides={"data": empty, "label_dir": empty, "tokenizer_bpe_model": None},
)
print(f"  loaded in {time.time()-t0:.0f}s")
model = models[0].eval()

try:
    from apex.normalization import FusedLayerNorm as _AFLN
    def _swap(m):
        n = 0
        for nm, ch in list(m.named_children()):
            if isinstance(ch, _AFLN):
                stk = nn.LayerNorm(ch.normalized_shape, eps=ch.eps,
                                   elementwise_affine=ch.elementwise_affine)
                if ch.elementwise_affine:
                    stk.weight.data.copy_(ch.weight.data)
                    stk.bias.data.copy_(ch.bias.data)
                stk.to(next(ch.parameters()).device)
                setattr(m, nm, stk); n += 1
            else:
                n += _swap(ch)
        return n
    print(f"Swapped {_swap(model)} FusedLayerNorm modules.")
except ImportError:
    pass

# Causal mask monkey-patch (same as export — keeps PT path identical)
NEG_INF = float("-inf")
def _dyn_mask(self, t):
    sz = t.size(0)
    i = torch.arange(sz, device=t.device).unsqueeze(0)
    j = torch.arange(sz, device=t.device).unsqueeze(1)
    m = torch.zeros(sz, sz, dtype=t.dtype, device=t.device)
    return m.masked_fill(i > j, NEG_INF)
import types
model.decoder.buffered_future_mask = types.MethodType(_dyn_mask, model.decoder)

decoder = model.decoder
tgt_dict = decoder.dictionary if hasattr(decoder, "dictionary") else task.target_dictionary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
decoder = decoder.to(device)


class PtDecoderWrapper(nn.Module):
    def __init__(self, d): super().__init__(); self.d = d
    def forward(self, prev_tokens, enc_feats):
        enc_TBC = enc_feats.transpose(0, 1).contiguous()
        logits, _ = self.d(
            prev_output_tokens=prev_tokens,
            encoder_out={"encoder_out": enc_TBC, "padding_mask": None},
            features_only=False,
        )
        return logits

pt_dec = PtDecoderWrapper(decoder).eval().to(device)


# -------------------------------------------------------------------------
# 2. Load ONNX decoder
# -------------------------------------------------------------------------
print(f"Loading ONNX decoder from {ONNX_PATH}...")
sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])


# -------------------------------------------------------------------------
# 3. Load dictionary
# -------------------------------------------------------------------------
with open(DICT_PATH, "r", encoding="utf-8") as f:
    symbols = [line.rstrip("\n") for line in f]
print(f"Dict: {len(symbols)} tokens (bos={tgt_dict.bos()}={symbols[tgt_dict.bos()]!r}, "
      f"eos={tgt_dict.eos()}={symbols[tgt_dict.eos()]!r})")
BOS, EOS = tgt_dict.bos(), tgt_dict.eos()


def detokenize(token_ids):
    # SentencePiece: ▁ = word boundary -> space; everything else concatenates
    SP_SPACE = "▁"
    parts = []
    for t in token_ids:
        if t in (BOS, EOS, tgt_dict.pad()):
            continue
        if 0 <= t < len(symbols):
            parts.append(symbols[t])
    return "".join(parts).replace(SP_SPACE, " ").strip()


# -------------------------------------------------------------------------
# 4. Greedy decode (PyTorch and ONNX paths share the loop, differ only in
#    the underlying decoder call). max_diff per-step logged for the
#    interesting positions.
# -------------------------------------------------------------------------
@torch.no_grad()
def greedy_pt(enc_feats_np):
    enc = torch.from_numpy(enc_feats_np).to(device).unsqueeze(0)   # (1, T_enc, C)
    tokens = [BOS]
    for _ in range(MAX_DECODE):
        toks = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = pt_dec(toks, enc)        # (1, t, V)
        nxt = int(logits[0, -1].argmax().item())
        tokens.append(nxt)
        if nxt == EOS:
            break
    return tokens


def greedy_onnx(enc_feats_np):
    enc = enc_feats_np[None, ...].astype(np.float32)   # (1, T_enc, C)
    tokens = [BOS]
    for _ in range(MAX_DECODE):
        toks = np.array([tokens], dtype=np.int64)
        logits = sess.run(["logits"], {"prev_tokens": toks, "encoder_features": enc})[0]
        nxt = int(np.argmax(logits[0, -1]))
        tokens.append(nxt)
        if nxt == EOS:
            break
    return tokens


# -------------------------------------------------------------------------
# 5. Iterate over real clips from the feature cache
# -------------------------------------------------------------------------
print(f"\nScanning {CACHE_DIR} for cached feature shards...")
shards = sorted(p for p in os.listdir(CACHE_DIR) if p.endswith("_feats.pt"))
if not shards:
    raise RuntimeError(f"No *_feats.pt files in {CACHE_DIR}")
print(f"  found {shards}")

# Load just the first shard for now — enough for a sanity comparison.
sp_path = os.path.join(CACHE_DIR, shards[0])
print(f"  using {sp_path}")
d = torch.load(sp_path, map_location="cpu", weights_only=False)
feats_list = d["features"]
print(f"  {len(feats_list)} clips in shard, picking first {N_CLIPS}")


mismatches = 0
for i in range(min(N_CLIPS, len(feats_list))):
    feat = feats_list[i]
    feat_np = feat.numpy() if hasattr(feat, "numpy") else np.asarray(feat)
    print(f"\n--- clip {i}  enc_shape={feat_np.shape} ---")
    t0 = time.time()
    pt_toks   = greedy_pt(feat_np)
    pt_time   = time.time() - t0
    t0 = time.time()
    onnx_toks = greedy_onnx(feat_np)
    onnx_time = time.time() - t0
    same = pt_toks == onnx_toks
    print(f"  PT   ({pt_time:.1f}s, {len(pt_toks)} tok): {pt_toks}")
    print(f"  ONNX ({onnx_time:.1f}s, {len(onnx_toks)} tok): {onnx_toks}")
    print(f"  match={same}")
    print(f"  PT   text: {detokenize(pt_toks)!r}")
    print(f"  ONNX text: {detokenize(onnx_toks)!r}")
    if not same:
        mismatches += 1
        # Show divergence position
        for k in range(min(len(pt_toks), len(onnx_toks))):
            if pt_toks[k] != onnx_toks[k]:
                print(f"  first divergence at position {k}: PT={pt_toks[k]} ONNX={onnx_toks[k]}")
                break

print(f"\n==== Summary: {mismatches}/{min(N_CLIPS, len(feats_list))} clips diverged ====")
if mismatches == 0:
    print("Decoder ONNX is text-equivalent to PT. Safe to ship to Android.")
else:
    print("Decoder ONNX text-diverges from PT. Investigate before Android integration.")
