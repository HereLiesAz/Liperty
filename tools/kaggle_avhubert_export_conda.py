"""AV-HuBERT ONNX export via an isolated conda env.

The 2026-05 attempts inside Kaggle's base-image Python failed at
cascading dep rot (torch 2.10 -> 2.2.2 didn't help, then omegaconf
2.x removed `II`, then hydra-core, etc.). The fix is a properly
isolated Python 3.9 environment with the entire 2022 dep stack
that fairseq commit afc77bdf was tested against.

This script runs INSIDE the conda env (call it from a subprocess
launched as `/kaggle/working/v3_env/bin/python tools/kaggle_avhubert_export_conda.py`),
not from the Kaggle kernel's Python.

It assumes:
- The env exists at ENV_PREFIX (default /kaggle/working/v3_env)
- The env has python 3.9, torch 1.13.1+cu117, omegaconf 2.0.6,
  hydra-core 1.0.7, numpy<2, and the rest of fairseq's pinned deps
- /kaggle/working/large_vox_iter5.pt exists
- /kaggle/working/work/av_hubert is cloned with the fairseq submodule
- HF_TOKEN is in the env (passed through from the Kaggle kernel)
"""

import os
import sys
import subprocess
import time

ENV_PREFIX = os.environ.get("V3_ENV_PREFIX", "/kaggle/working/v3_env")
# Default for Kaggle; override via env vars in the Docker entrypoint.
WORK_DIR = os.environ.get("V3_WORK_DIR", "/kaggle/working/work")
AVHUBERT_DIR = os.environ.get("V3_AVHUBERT_DIR", os.path.join(WORK_DIR, "av_hubert"))
FAIRSEQ_DIR = os.environ.get("V3_FAIRSEQ_DIR", os.path.join(AVHUBERT_DIR, "fairseq"))

LOCAL_PT = os.environ.get("V3_CKPT_PATH", "/kaggle/working/large_vox_iter5.pt")
ONNX_OUT = os.environ.get("V3_ONNX_OUT", os.path.join(WORK_DIR, "avhubert_visual_encoder.onnx"))
HF_REPO = os.environ.get("V3_HF_REPO", "HereLiesAz/liperty-avhubert-encoder")

print(f"Python:    {sys.executable}")
print(f"Version:   {sys.version.split()[0]}")
print(f"Env:       {ENV_PREFIX}")


# -------------------------------------------------------------------------
# 1. Add fairseq's INNER package dir to sys.path so `import fairseq` resolves
# -------------------------------------------------------------------------
# The pip --editable install of av_hubert/fairseq registers the wrong level
# (the outer repo dir). Manually prepend the right one.
if FAIRSEQ_DIR not in sys.path:
    sys.path.insert(0, FAIRSEQ_DIR)
# Put av_hubert/avhubert/ (the inner directory) on sys.path so that the
# avhubert modules import each other via bare imports like
# `from hubert_pretraining import ...`. DON'T also put av_hubert/ on
# sys.path: doing so makes `import avhubert` AND `import hubert` both
# load hubert.py, which double-runs the @register_model decorator and
# raises ValueError("Cannot register duplicate model (av_hubert)").
AVHUBERT_PKG_DIR = os.path.join(AVHUBERT_DIR, "avhubert")
if AVHUBERT_PKG_DIR not in sys.path:
    sys.path.insert(0, AVHUBERT_PKG_DIR)


# -------------------------------------------------------------------------
# 2. Sanity checks
# -------------------------------------------------------------------------
import torch                                                    # noqa: E402

print(f"torch:     {torch.__version__}")
print(f"CUDA:      {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"           {torch.cuda.get_device_name(0)}")

import fairseq                                                  # noqa: E402

if fairseq.__file__ is None:
    raise RuntimeError(
        f"fairseq imported as a namespace package. Expected {FAIRSEQ_DIR}/fairseq/__init__.py "
        f"to be found via sys.path[0] = {FAIRSEQ_DIR}, but Python loaded a namespace package "
        f"instead. The conda env is misconfigured."
    )
print(f"fairseq:   {fairseq.__file__}")

# omegaconf must be <2.1 (the vendored fairseq commit imports `II` from it)
import omegaconf                                                # noqa: E402

print(f"omegaconf: {omegaconf.__version__}")
from omegaconf import II  # noqa: F401, E402
print("  II imports OK")

# Import the avhubert modules as TOP-LEVEL modules (not as a package).
# This is how Meta's own avhubert/infer_s2s.py does it. The
# @register_model side-effects fire on import and register the avhubert
# classes with fairseq's model registry so checkpoint_utils can find them.
import hubert_pretraining   # noqa: E402, F401
import hubert               # noqa: E402, F401
import hubert_asr           # noqa: E402, F401

print(f"avhubert:  {hubert.__file__}")


# -------------------------------------------------------------------------
# 3. HF auth + checkpoint already on disk
# -------------------------------------------------------------------------
from huggingface_hub import login, whoami, upload_file          # noqa: E402

token = os.environ.get("HF_TOKEN")
if token:
    login(token, add_to_git_credential=True)
    print(f"HF user:   {whoami().get('name', '?')}")
else:
    print("HF user:   (no HF_TOKEN set; will skip upload at end)")

CKPT_FILENAME = os.environ.get("V3_CKPT_FILENAME", "large_vox_iter5.pt")

if not os.path.exists(LOCAL_PT):
    # In Docker, the .pt won't be there yet. Try the HF mirror first
    # (where large_vox_iter5.pt has been rehosted), then fall back to
    # Meta's CDN for the base model (which we haven't mirrored).
    print(f"{LOCAL_PT} not found locally; fetching {CKPT_FILENAME}...")
    try:
        from huggingface_hub import hf_hub_download
        LOCAL_PT = hf_hub_download(
            repo_id=HF_REPO,
            filename=CKPT_FILENAME,
            local_dir=os.path.dirname(LOCAL_PT) or "/work",
        )
    except Exception as e:
        # Not on HF; fall back to Meta's public CDN.
        meta_url = ("https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/clean-pretrain/"
                    + CKPT_FILENAME)
        print(f"  HF fetch failed ({type(e).__name__}); trying Meta CDN: {meta_url}")
        import urllib.request
        urllib.request.urlretrieve(meta_url, LOCAL_PT)
print(f"ckpt:      {LOCAL_PT}  ({os.path.getsize(LOCAL_PT) / 1e9:.2f} GB)")


# -------------------------------------------------------------------------
# 4. Empty data dir for fairseq's task setup
# -------------------------------------------------------------------------
empty_data_dir = os.path.join(WORK_DIR, "empty_data")
os.makedirs(empty_data_dir, exist_ok=True)
for fn in ["nframes.audio", "nframes.video", "test.tsv", "valid.tsv", "train.tsv"]:
    p = os.path.join(empty_data_dir, fn)
    if not os.path.exists(p):
        open(p, "w").close()


# -------------------------------------------------------------------------
# 5. Load the model
# -------------------------------------------------------------------------
from fairseq import checkpoint_utils                            # noqa: E402

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


# -------------------------------------------------------------------------
# 6. Video-only wrapper
# -------------------------------------------------------------------------
import torch.nn as nn                                           # noqa: E402


class AvHubertVisualEncoder(nn.Module):
    """Video-only wrapper that explicitly composes the minimal encoder
    forward, bypassing AVHubertModel.forward()'s conditional branches.

    The default extract_features -> forward() path goes through:
      - apply_input_mask (skipped when mask=False but the tracer still
        bakes the `if` decision in)
      - np.random.random() calls for modality dropout (in eval these
        do nothing but the random values still get captured as
        constants by the tracer)
      - forward_targets (skipped but conditional)
      - feature masking
      - target/loss computation

    For ONNX export we want a clean, branch-free forward:
      audio = zeros
      feat_a = feature_extractor_audio(audio)
      feat_v = feature_extractor_video(video)
      feats = cat([feat_a, feat_v], dim=1).transpose(1, 2)
      feats = layer_norm(feats)
      feats = post_extract_proj(feats)
      x, _ = encoder(feats)
      return x

    Run #6 produced all-zero ONNX output despite a clean PyTorch
    forward, suggesting one of the conditionals in the AVHubertModel.
    forward() path was traced incorrectly. This rewrite eliminates the
    suspect paths.
    """

    AUDIO_FEAT_DIM = 104

    def __init__(self, full_model):
        super().__init__()
        # Borrow the trained sub-modules. We don't subclass them, just
        # reference them so the state dict transfers transparently.
        self.feature_extractor_audio = full_model.feature_extractor_audio
        self.feature_extractor_video = full_model.feature_extractor_video
        self.layer_norm = full_model.layer_norm
        self.post_extract_proj = full_model.post_extract_proj  # may be None
        self.encoder = full_model.encoder

    def forward(self, video):
        # video: (B, 1, T, 88, 88) float32
        B = video.shape[0]
        T = video.shape[2]
        zero_audio = torch.zeros(
            B, self.AUDIO_FEAT_DIM, T,
            dtype=video.dtype, device=video.device,
        )
        feat_a = self.feature_extractor_audio(zero_audio)
        feat_v = self.feature_extractor_video(video)
        feats_cat = torch.cat([feat_a, feat_v], dim=1)
        feats_t = feats_cat.transpose(1, 2)
        feats_ln = self.layer_norm(feats_t)
        if self.post_extract_proj is not None:
            feats_proj = self.post_extract_proj(feats_ln)
        else:
            feats_proj = feats_ln
        x, _ = self.encoder(feats_proj, padding_mask=None, layer=None)
        # Multiple outputs let the parity check bisect the divergence
        # point: feat_a (audio frontend), feat_v (video frontend),
        # feats_ln (layer norm), feats_proj (linear proj), x (encoder).
        return x, feat_a, feat_v, feats_ln, feats_proj


# Replace any apex FusedLayerNorm in the loaded model with stock
# torch.nn.LayerNorm. apex's CUDA-fused layer norm has no ONNX
# equivalent so the legacy tracer silently emits zero outputs for it
# (this was the exact divergence point in run #8: feat_a/feat_v
# matched perfectly, the post-transpose layer_norm produced [0, 0]
# in ONNX vs [-2.89, 1.98] in PyTorch). stock LayerNorm is
# parameter-compatible (same weight/bias names + shapes) so we
# transplant in-place.
try:
    from apex.normalization import FusedLayerNorm as _ApexFLN

    def _swap_fused_layernorm(m: nn.Module) -> nn.Module:
        replaced = 0
        for name, child in list(m.named_children()):
            if isinstance(child, _ApexFLN):
                stock = nn.LayerNorm(
                    child.normalized_shape,
                    eps=child.eps,
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

wrapper = AvHubertVisualEncoder(model).eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wrapper = wrapper.to(device)

T_DUMMY = 50
dummy_video = torch.randn(1, 1, T_DUMMY, 88, 88, device=device, dtype=torch.float32)

print("Smoke-testing PyTorch forward...")
t0 = time.time()
with torch.no_grad():
    pt_outs = wrapper(dummy_video)
out_pt = pt_outs[0]  # The encoder output is the primary return
print(f"  PyTorch encoder output: {tuple(out_pt.shape)}  in {time.time()-t0:.1f}s")
for name, t in zip(["feat_a", "feat_v", "feats_ln", "feats_proj"], pt_outs[1:]):
    print(f"  PyTorch {name}: shape={tuple(t.shape)}  range=[{t.min():.3f}, {t.max():.3f}]")


# -------------------------------------------------------------------------
# 7. ONNX export
# -------------------------------------------------------------------------
ONNX_PATH = ONNX_OUT
os.makedirs(os.path.dirname(ONNX_PATH) or ".", exist_ok=True)
print("Tracing to ONNX (this is the risky step)...")
t0 = time.time()
torch.onnx.export(
    wrapper,
    dummy_video,
    ONNX_PATH,
    input_names=["video"],
    output_names=["features", "feat_a", "feat_v", "feats_ln", "feats_proj"],
    opset_version=17,
    do_constant_folding=True,
    dynamic_axes={
        "video":    {2: "T"},
        "features": {1: "T_out"},
        "feat_a":   {2: "T"},
        "feat_v":   {2: "T"},
        "feats_ln": {1: "T"},
        "feats_proj": {1: "T"},
    },
    training=torch.onnx.TrainingMode.EVAL,
)
print(f"  ONNX written: {ONNX_PATH}  ({os.path.getsize(ONNX_PATH) / 1e6:.0f} MB)  in {time.time()-t0:.0f}s")


# -------------------------------------------------------------------------
# 8. Parity check
# -------------------------------------------------------------------------
import onnxruntime as ort                                       # noqa: E402
import numpy as np                                              # noqa: E402

sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
dummy_np = dummy_video.detach().cpu().numpy().astype(np.float32)
all_ort_names = ["features", "feat_a", "feat_v", "feats_ln", "feats_proj"]
ort_outs = sess.run(all_ort_names, {"video": dummy_np})
pt_outs_np = [t.detach().cpu().numpy() for t in pt_outs]

print()
print("=== Layer-by-layer parity ===")
for name, pt_np, ort_np in zip(all_ort_names, pt_outs_np, ort_outs):
    diff = np.abs(pt_np - ort_np)
    print(f"{name:12s} PT range=[{pt_np.min():.3f},{pt_np.max():.3f}]  "
          f"ORT range=[{ort_np.min():.3f},{ort_np.max():.3f}]  "
          f"max_diff={diff.max():.6f}  mean_diff={diff.mean():.6f}")

primary_diff = np.abs(pt_outs_np[0] - ort_outs[0])
if primary_diff.max() >= 1e-2:
    print()
    print("WARNING: ONNX 'features' diverges from PyTorch by >=1e-2.")
    print("Look at the first layer above with non-trivial max_diff to localize.")


# -------------------------------------------------------------------------
# 9. Upload
# -------------------------------------------------------------------------
if token:
    print("Uploading ONNX to HF mirror...")
    upload_file(
        path_or_fileobj=ONNX_PATH,
        path_in_repo="avhubert_visual_encoder.onnx",
        repo_id=HF_REPO,
        repo_type="model",
        commit_message="ONNX-exported visual encoder (conda env, 2022 dep stack)",
    )
    print(f"Uploaded -> https://huggingface.co/{HF_REPO}/blob/main/avhubert_visual_encoder.onnx")
else:
    print(f"Upload skipped (no HF_TOKEN). ONNX is at {ONNX_PATH} on disk; upload manually with:")
    print(f"  huggingface-cli upload {HF_REPO} {ONNX_PATH} avhubert_visual_encoder.onnx")
print()
print("AV-HuBERT V3 encoder export complete.")
