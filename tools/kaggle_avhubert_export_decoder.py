"""AV-HuBERT seq2seq DECODER ONNX export.

Companion to kaggle_avhubert_export_conda.py (which handles the
encoder). Loads the same `base_vox_433h.pt` checkpoint and exports
just the Transformer decoder as a single-step ONNX:

  inputs:
    prev_tokens     (B, T_dec) int64     — tokens decoded so far, including <bos>
    encoder_features (B, T_enc, C)        — NTC layout (transposed for fairseq inside)
  outputs:
    logits          (B, T_dec, V)        — per-position logits; Android takes [:, -1, :]

No KV cache. Each Android autoregressive step re-runs the decoder over
all previously emitted tokens. O(N^2) work per utterance but trivial
ONNX, no incremental_state tensors to wrangle. For typical Liperty
utterances (<50 tokens) the overhead is acceptable; can be optimized
later by exporting a KV-cached step if benchmarks demand it.

Designed to run inside the same liperty-v3-export Docker image as the
encoder export. Mount tools/ as /scripts and override entrypoint:

  docker run --rm --gpus all \\
      -e HF_TOKEN=$HF_TOKEN \\
      -v $(pwd)/out:/work/out \\
      -v "$(pwd)/tools:/scripts" \\
      --entrypoint python liperty-v3-export \\
      /scripts/kaggle_avhubert_export_decoder.py
"""

import os
import sys
import time

# -------------------------------------------------------------------------
# 0. Env (matches the Dockerfile's defaults; override at `docker run`)
# -------------------------------------------------------------------------
WORK_DIR     = os.environ.get("V3_WORK_DIR",     "/work")
AVHUBERT_DIR = os.environ.get("V3_AVHUBERT_DIR", os.path.join(WORK_DIR, "av_hubert"))
FAIRSEQ_DIR  = os.environ.get("V3_FAIRSEQ_DIR",  os.path.join(AVHUBERT_DIR, "fairseq"))
LOCAL_PT     = os.environ.get("V3_CKPT_PATH",    os.path.join(WORK_DIR, "base_vox_433h.pt"))
CKPT_FNAME   = os.environ.get("V3_CKPT_FILENAME","base_vox_433h.pt")
ONNX_OUT     = os.environ.get("V3_DECODER_ONNX_OUT",
                              os.path.join(WORK_DIR, "out", "avhubert_base_vox_433h_decoder.onnx"))
DICT_OUT     = os.environ.get("V3_DECODER_DICT_OUT",
                              os.path.join(WORK_DIR, "out", "avhubert_base_vox_433h_dict.txt"))
HF_REPO      = os.environ.get("V3_HF_REPO", "HereLiesAz/liperty-avhubert-encoder")

# Avoid fairseq's eager wandb/tensorboard imports.
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")

print(f"Python:    {sys.executable}")
print(f"Version:   {sys.version.split()[0]}")
print(f"work:      {WORK_DIR}")
print(f"avhubert:  {AVHUBERT_DIR}")
print(f"fairseq:   {FAIRSEQ_DIR}")
print(f"ckpt:      {LOCAL_PT}")
print(f"onnx-out:  {ONNX_OUT}")

os.makedirs(os.path.dirname(ONNX_OUT), exist_ok=True)


# -------------------------------------------------------------------------
# 1. sys.path dance so `import fairseq` and `import hubert*` both resolve
#    (see kaggle_avhubert_export_conda.py for the rationale)
# -------------------------------------------------------------------------
if FAIRSEQ_DIR not in sys.path:
    sys.path.insert(0, FAIRSEQ_DIR)
AVHUBERT_PKG_DIR = os.path.join(AVHUBERT_DIR, "avhubert")
if AVHUBERT_PKG_DIR not in sys.path:
    sys.path.insert(0, AVHUBERT_PKG_DIR)


# -------------------------------------------------------------------------
# 2. Imports
# -------------------------------------------------------------------------
import torch
import torch.nn as nn
import numpy as np

print(f"torch:     {torch.__version__}")
print(f"CUDA:      {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"           {torch.cuda.get_device_name(0)}")

# Eagerly import the avhubert flat modules so register_model/register_task
# fire BEFORE checkpoint_utils tries to instantiate them.
import hubert_pretraining  # noqa: F401
import hubert              # noqa: F401
import hubert_asr          # noqa: F401

import fairseq             # noqa: E402
print(f"fairseq:   {fairseq.__file__}")

from fairseq import checkpoint_utils  # noqa: E402


# -------------------------------------------------------------------------
# 3. Download checkpoint if missing
# -------------------------------------------------------------------------
def ensure_checkpoint():
    if os.path.exists(LOCAL_PT):
        sz = os.path.getsize(LOCAL_PT) / 1e6
        print(f"Checkpoint exists: {LOCAL_PT}  ({sz:.0f} MB)")
        return
    # First try HF (faster, no rate-limit headaches on FB CDN).
    token = os.environ.get("HF_TOKEN")
    try:
        from huggingface_hub import login, hf_hub_download
        if token:
            login(token=token, add_to_git_credential=False)
        print(f"Pulling {CKPT_FNAME} from {HF_REPO} (HF)...")
        p = hf_hub_download(repo_id=HF_REPO, repo_type="model",
                            filename=CKPT_FNAME, local_dir=WORK_DIR)
        # Move/link into LOCAL_PT if it landed elsewhere.
        if os.path.abspath(p) != os.path.abspath(LOCAL_PT):
            import shutil; shutil.copy(p, LOCAL_PT)
        return
    except Exception as e:
        print(f"HF pull failed: {e}")
    # Fall back to FB public files.
    fb_url = "https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/vsr/base_vox_433h.pt"
    print(f"Falling back to {fb_url}")
    import urllib.request
    urllib.request.urlretrieve(fb_url, LOCAL_PT)

ensure_checkpoint()


# -------------------------------------------------------------------------
# 4. Load the model via fairseq
# -------------------------------------------------------------------------
import tempfile
empty_data_dir = tempfile.mkdtemp(prefix="empty_avhubert_data_")
# fairseq's task setup wants these dirs to exist; the contents don't matter
# because we never run train/eval, only model construction + state dict load.

print("Loading checkpoint via fairseq...")
t0 = time.time()
models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
    [LOCAL_PT],
    arg_overrides={
        "data": empty_data_dir,
        "label_dir": empty_data_dir,
        "tokenizer_bpe_model": None,
    },
)
print(f"  loaded in {time.time()-t0:.0f}s")
model = models[0]
model.eval()
print(f"Model class: {type(model).__name__}")

if not hasattr(model, "decoder"):
    raise RuntimeError(
        f"{type(model).__name__} has no .decoder — this checkpoint isn't a "
        "seq2seq model. Did you load base_vox_iter5.pt (pretrained-only) by "
        "mistake?"
    )


# -------------------------------------------------------------------------
# 5. apex FusedLayerNorm swap (same fix as the encoder run #9)
# -------------------------------------------------------------------------
try:
    from apex.normalization import FusedLayerNorm as _ApexFLN

    def _swap_fused_layernorm(m: nn.Module) -> int:
        replaced = 0
        for name, child in list(m.named_children()):
            if isinstance(child, _ApexFLN):
                stock = nn.LayerNorm(
                    child.normalized_shape, eps=child.eps,
                    elementwise_affine=child.elementwise_affine,
                )
                if child.elementwise_affine:
                    stock.weight.data.copy_(child.weight.data)
                    stock.bias.data.copy_(child.bias.data)
                stock.to(next(child.parameters()).device)
                setattr(m, name, stock)
                replaced += 1
            else:
                replaced += _swap_fused_layernorm(child)
        return replaced

    n = _swap_fused_layernorm(model)
    print(f"Swapped {n} apex FusedLayerNorm modules for torch.nn.LayerNorm.")
except ImportError:
    print("apex not installed; assuming stock LayerNorm everywhere.")


# -------------------------------------------------------------------------
# 6. Inspect the decoder so we know what we're exporting
# -------------------------------------------------------------------------
decoder = model.decoder
print(f"Decoder class: {type(decoder).__name__}")

# fairseq.TransformerDecoder exposes .dictionary (or task.target_dictionary).
tgt_dict = getattr(decoder, "dictionary", None) or task.target_dictionary
vocab_size = len(tgt_dict)
print(f"Vocab size:   {vocab_size}")
print(f"  pad:  {tgt_dict.pad()} ('{tgt_dict[tgt_dict.pad()]}')")
print(f"  bos:  {tgt_dict.bos()} ('{tgt_dict[tgt_dict.bos()]}')")
print(f"  eos:  {tgt_dict.eos()} ('{tgt_dict[tgt_dict.eos()]}')")
print(f"  unk:  {tgt_dict.unk()} ('{tgt_dict[tgt_dict.unk()]}')")

# Save dictionary as text (one symbol per line, matches fairseq's dict.txt
# format: "<symbol> <count>" — but we drop the count since Android only
# needs the index->token map).
with open(DICT_OUT, "w", encoding="utf-8") as f:
    # Specials in fairseq's standard order: <s>=bos, <pad>, </s>=eos, <unk>
    # are placed at indices 0..3 by Dictionary's __init__. We need the FULL
    # symbol list in index order for Android.
    for i in range(vocab_size):
        f.write(tgt_dict[i] + "\n")
print(f"Dictionary written: {DICT_OUT}  ({vocab_size} lines)")

# Encoder feature dim (input dim of decoder cross-attention)
# AVHubertSeq2Seq decoder's cross-attention reads encoder_out["encoder_out"][0]
# which is the (T, B, C) features that the encoder produced. C = 768 for base.
ENC_DIM = decoder.embed_dim if hasattr(decoder, "embed_dim") else 768
print(f"Decoder embed dim: {ENC_DIM}")

# Decoder depth/heads for sanity
n_layers = len(decoder.layers) if hasattr(decoder, "layers") else "?"
print(f"Decoder layers:   {n_layers}")


# -------------------------------------------------------------------------
# 6b. Monkey-patch the decoder's causal mask
# -------------------------------------------------------------------------
# av_hubert/decoder.py:buffered_future_mask caches a (dim, dim) tensor
# created with utils.fill_with_neg_inf and torch.triu. The legacy ONNX
# tracer captures `dim = tensor.size(0)` as a Python int at trace time,
# baking the mask shape as a constant. At runtime with a different
# T_dec, onnxruntime then errors on a broadcast mismatch (e.g. trace
# T_dec=4, run T_dec=8 -> "Attempting to broadcast axis 4 by 8").
#
# Replace with an arange-based mask: torch.arange consumed by a
# comparison op, then masked_fill. Both ops support dynamic shape in
# ONNX opset 17.
NEG_INF = float("-inf")

def _dynamic_future_mask(self, tensor):
    sz = tensor.size(0)
    i = torch.arange(sz, device=tensor.device).unsqueeze(0)   # (1, sz)
    j = torch.arange(sz, device=tensor.device).unsqueeze(1)   # (sz, 1)
    # Causal: position j attends to positions 0..j -> mask out i > j.
    # masked_fill avoids the 0.0*-inf=NaN trap of a multiplicative mask.
    mask = torch.zeros(sz, sz, dtype=tensor.dtype, device=tensor.device)
    return mask.masked_fill(i > j, NEG_INF)

import types as _types
model.decoder.buffered_future_mask = _types.MethodType(_dynamic_future_mask, model.decoder)
print("Patched decoder.buffered_future_mask for dynamic T_dec.")


# -------------------------------------------------------------------------
# 7. ONNX-friendly wrapper
# -------------------------------------------------------------------------
class DecoderWrapper(nn.Module):
    """Adapts fairseq.TransformerDecoder to a flat (prev_tokens, enc_features)
    -> logits signature with Android-friendly NTC layout for encoder_features.

    Re-runs over all prev_tokens each call (no KV cache). Android takes
    logits[:, -1, :] to score the next token and appends to prev_tokens.
    """

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, prev_tokens, encoder_features):
        # encoder_features: (B, T_enc, C) — convert to fairseq's (T_enc, B, C)
        enc_TBC = encoder_features.transpose(0, 1).contiguous()
        # av_hubert's custom TransformerDecoder (avhubert/decoder.py) reads
        # encoder_out["encoder_out"] and encoder_out["padding_mask"] as FLAT
        # tensors (NOT lists, unlike stock fairseq). padding_mask=None means
        # "no positions are padded".
        encoder_out = {
            "encoder_out": enc_TBC,
            "padding_mask": None,
        }
        logits, _ = self.decoder(
            prev_output_tokens=prev_tokens,
            encoder_out=encoder_out,
            features_only=False,
        )
        return logits   # (B, T_dec, V)


wrapper = DecoderWrapper(decoder).eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wrapper = wrapper.to(device)


# -------------------------------------------------------------------------
# 8. Smoke-test PyTorch forward before attempting ONNX
# -------------------------------------------------------------------------
T_ENC_DUMMY = 50      # encoder output length (sliding window of ~50 video features)
T_DEC_DUMMY = 4       # decode position

dummy_enc = torch.randn(1, T_ENC_DUMMY, ENC_DIM, device=device, dtype=torch.float32)
dummy_tok = torch.tensor(
    [[tgt_dict.bos(), tgt_dict.unk(), tgt_dict.unk(), tgt_dict.eos()]],
    device=device, dtype=torch.long,
)

print("Smoke-testing PyTorch decoder...")
t0 = time.time()
with torch.no_grad():
    pt_logits = wrapper(dummy_tok, dummy_enc)
print(f"  forward OK in {time.time()-t0:.1f}s, logits shape {tuple(pt_logits.shape)}")
print(f"  range: [{pt_logits.min().item():.3f}, {pt_logits.max().item():.3f}]")


# -------------------------------------------------------------------------
# 9. ONNX export (legacy tracer; the new dynamo path can't handle fairseq's
#    incremental-state-aware decoder cleanly)
# -------------------------------------------------------------------------
print(f"Exporting decoder ONNX: {ONNX_OUT}")
torch.onnx.export(
    wrapper,
    (dummy_tok, dummy_enc),
    ONNX_OUT,
    input_names=["prev_tokens", "encoder_features"],
    output_names=["logits"],
    opset_version=17,
    do_constant_folding=True,
    dynamic_axes={
        "prev_tokens":      {0: "B", 1: "T_dec"},
        "encoder_features": {0: "B", 1: "T_enc"},
        "logits":           {0: "B", 1: "T_dec"},
    },
    training=torch.onnx.TrainingMode.EVAL,
)
sz = os.path.getsize(ONNX_OUT) / 1e6
print(f"ONNX written: {sz:.1f} MB")


# -------------------------------------------------------------------------
# 10. Parity check vs PyTorch on a couple of (T_enc, T_dec) shapes
# -------------------------------------------------------------------------
import onnxruntime as ort
sess = ort.InferenceSession(ONNX_OUT, providers=["CPUExecutionProvider"])

def parity(t_enc, t_dec):
    enc = torch.randn(1, t_enc, ENC_DIM, dtype=torch.float32)
    tok = torch.randint(low=0, high=vocab_size, size=(1, t_dec), dtype=torch.long)
    tok[0, 0] = tgt_dict.bos()
    enc_dev = enc.to(device); tok_dev = tok.to(device)
    with torch.no_grad():
        pt = wrapper(tok_dev, enc_dev).cpu().numpy()
    ort_out = sess.run(["logits"], {
        "prev_tokens": tok.numpy(),
        "encoder_features": enc.numpy(),
    })[0]
    diff = np.abs(pt - ort_out)
    print(f"  T_enc={t_enc} T_dec={t_dec}: max_diff={diff.max():.6f}  "
          f"mean_diff={diff.mean():.6f}  shape={ort_out.shape}")
    return diff.max()

print("Parity check:")
worst = max(parity(50, 1), parity(50, 8), parity(40, 12), parity(60, 20))
print(f"Worst max_diff across configs: {worst:.6f}")
if worst > 1e-2:
    print("WARNING: parity drift > 1e-2 — investigate before relying on this ONNX.")


# -------------------------------------------------------------------------
# 11. Upload ONNX + dictionary to HF
# -------------------------------------------------------------------------
token = os.environ.get("HF_TOKEN")
if token:
    from huggingface_hub import login, upload_file, create_repo
    login(token=token, add_to_git_credential=False)
    create_repo(HF_REPO, repo_type="model", private=False, exist_ok=True)
    for src, dst in [
        (ONNX_OUT, "avhubert_base_vox_433h_decoder.onnx"),
        (DICT_OUT, "avhubert_base_vox_433h_dict.txt"),
    ]:
        print(f"  uploading {os.path.basename(src)}")
        upload_file(
            path_or_fileobj=src, path_in_repo=dst,
            repo_id=HF_REPO, repo_type="model",
            commit_message=f"V3 seq2seq decoder export ({type(decoder).__name__}, "
                           f"{n_layers} layers, vocab={vocab_size})",
        )
    print(f"Uploaded -> https://huggingface.co/{HF_REPO}")
else:
    print("No HF_TOKEN; decoder ONNX + dict stay local at:")
    print(f"  {ONNX_OUT}")
    print(f"  {DICT_OUT}")

print("AV-HuBERT V3 decoder export complete.")
