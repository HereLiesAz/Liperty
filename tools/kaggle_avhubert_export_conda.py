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
# And put av_hubert/ on sys.path so `import avhubert` finds the package
if AVHUBERT_DIR not in sys.path:
    sys.path.insert(0, AVHUBERT_DIR)


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

import avhubert                                                 # noqa: E402

print(f"avhubert:  {avhubert.__file__}")


# -------------------------------------------------------------------------
# 3. HF auth + checkpoint already on disk
# -------------------------------------------------------------------------
from huggingface_hub import login, whoami, upload_file          # noqa: E402

token = os.environ.get("HF_TOKEN")
if token:
    login(token, add_to_git_credential=True)
    print(f"HF user:   {whoami().get('name', '?')}")

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
    def __init__(self, full_model):
        super().__init__()
        self.full = full_model

    def forward(self, video):
        src = {"video": video, "audio": None}
        feats, _ = self.full.extract_features(
            source=src,
            padding_mask=None,
            mask=False,
            features_only=True,
            output_layer=None,
        )
        return feats


wrapper = AvHubertVisualEncoder(model).eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wrapper = wrapper.to(device)

T_DUMMY = 50
dummy_video = torch.randn(1, 1, T_DUMMY, 88, 88, device=device, dtype=torch.float32)

print("Smoke-testing PyTorch forward...")
t0 = time.time()
with torch.no_grad():
    out_pt = wrapper(dummy_video)
print(f"  PyTorch output: {tuple(out_pt.shape)}  in {time.time()-t0:.1f}s")


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
    output_names=["features"],
    opset_version=17,
    do_constant_folding=True,
    dynamic_axes={
        "video":    {2: "T"},
        "features": {1: "T_out"},
    },
)
print(f"  ONNX written: {ONNX_PATH}  ({os.path.getsize(ONNX_PATH) / 1e6:.0f} MB)  in {time.time()-t0:.0f}s")


# -------------------------------------------------------------------------
# 8. Parity check
# -------------------------------------------------------------------------
import onnxruntime as ort                                       # noqa: E402
import numpy as np                                              # noqa: E402

sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
dummy_np = dummy_video.detach().cpu().numpy().astype(np.float32)
ort_out = sess.run(["features"], {"video": dummy_np})[0]
pt_out = out_pt.detach().cpu().numpy()

diff = np.abs(pt_out - ort_out)
print(f"PyTorch out: {pt_out.shape}  range=[{pt_out.min():.3f}, {pt_out.max():.3f}]")
print(f"ONNX out:    {ort_out.shape}  range=[{ort_out.min():.3f}, {ort_out.max():.3f}]")
print(f"Max abs diff: {diff.max():.6f}")
print(f"Mean abs diff: {diff.mean():.6f}")
if diff.max() >= 1e-2:
    print("WARNING: ONNX diverges from PyTorch by >=1e-2. Export may be subtly broken.")


# -------------------------------------------------------------------------
# 9. Upload
# -------------------------------------------------------------------------
print("Uploading ONNX to HF mirror...")
upload_file(
    path_or_fileobj=ONNX_PATH,
    path_in_repo="avhubert_visual_encoder.onnx",
    repo_id=HF_REPO,
    repo_type="model",
    commit_message="ONNX-exported visual encoder (conda env, 2022 dep stack)",
)
print(f"Uploaded -> https://huggingface.co/{HF_REPO}/blob/main/avhubert_visual_encoder.onnx")
print()
print("AV-HuBERT V3 encoder export complete.")
