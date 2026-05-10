"""In-kernel AV-HuBERT large -> ONNX export.

**REQUIRES A FRESH KAGGLE KERNEL.** The 2026-05 attempt against a
warm kernel with torch 2.10 hit:

  ImportError: cannot import name 'utils' from 'fairseq' (unknown location)
  fairseq.__file__ = None

i.e. fairseq's editable install left a PEP 420 namespace package
instead of a real package, because the vendored fairseq commit
(afc77bdf...) doesn't finish setup on torch 2.10. See
docs/AVHUBERT_V3_BACKEND.md "Attempt log" section.

The fix is to downgrade torch BEFORE installing fairseq. Open a
fresh Kaggle session and run this at the top of the kernel,
*before* fetching this script:

    !pip install -q torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \\
        --index-url https://download.pytorch.org/whl/cu118
    # restart the kernel here (Kaggle UI: Runtime -> Restart)

Then in the new kernel:

    import urllib.request
    exec(urllib.request.urlopen(
        'https://raw.githubusercontent.com/HereLiesAz/Liperty/main/tools/kaggle_avhubert_export.py'
    ).read())

Sets up:
- Clones facebookresearch/av_hubert (vendors fairseq as a submodule)
- pip install --editable on the vendored fairseq
- adds av_hubert/ to PYTHONPATH so `import avhubert` resolves the
  package directory (NOT av_hubert/avhubert/ which would shadow it)
- pulls large_vox_iter5.pt from the public mirror (or uses an existing
  /kaggle/working/large_vox_iter5.pt if present)
- loads via fairseq.checkpoint_utils.load_model_ensemble_and_task
- wraps model.extract_features for video-only forward
- exports to ONNX with dynamic time axis
- runs a parity check (PyTorch vs ONNX) on a dummy tensor
- uploads the ONNX back to HereLiesAz/liperty-avhubert-encoder
"""

import os
import sys
import subprocess
import time

IS_KAGGLE = os.path.exists("/kaggle/working") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ
WORK_DIR = "/kaggle/working/work" if IS_KAGGLE else "/content/work"
os.makedirs(WORK_DIR, exist_ok=True)


def _run(cmd, check=True, capture=False):
    print(f"$ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    if capture:
        return subprocess.run(cmd, shell=isinstance(cmd, str), check=check, capture_output=True, text=True)
    return subprocess.run(cmd, shell=isinstance(cmd, str), check=check)


# -------------------------------------------------------------------------
# 1. Clone av_hubert + init fairseq submodule
# -------------------------------------------------------------------------
AVHUBERT_DIR = os.path.join(WORK_DIR, "av_hubert")
if not os.path.exists(AVHUBERT_DIR):
    _run(["git", "clone", "--depth", "1", "https://github.com/facebookresearch/av_hubert.git", AVHUBERT_DIR])
    _run(f"cd {AVHUBERT_DIR} && git submodule init && git submodule update")
print(f"av_hubert dir: {AVHUBERT_DIR}")

FAIRSEQ_DIR = os.path.join(AVHUBERT_DIR, "fairseq")
if not os.path.exists(os.path.join(FAIRSEQ_DIR, "setup.py")):
    raise RuntimeError(
        f"fairseq submodule did not populate at {FAIRSEQ_DIR}. "
        "Try `cd av_hubert && git submodule update --init --recursive` manually."
    )


# -------------------------------------------------------------------------
# 2. Install fairseq + small extra deps
# -------------------------------------------------------------------------
def _ensure_pkg(name, install_args):
    try:
        __import__(name)
        return True
    except ImportError:
        pass
    print(f"Installing {name}...")
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", *install_args],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"FAILED to install {name}:")
        print(r.stderr[-2000:])
        return False
    print(f"  {name} installed.")
    return True


# fairseq editable install. AV-HuBERT pinned an older fairseq commit
# that may or may not coexist with torch 2.10. Try once; report
# loudly on failure.
print("Installing fairseq (editable, from av_hubert's vendored copy)...")
r = subprocess.run(
    [sys.executable, "-m", "pip", "install", "--editable", FAIRSEQ_DIR, "-q"],
    capture_output=True, text=True,
)
if r.returncode != 0:
    print("fairseq install FAILED. Last 2 KB of stderr:")
    print(r.stderr[-2000:])
    raise RuntimeError("fairseq install failed; cannot proceed with AV-HuBERT export.")
print("  fairseq installed.")

for pkg, args in [
    ("onnx", ["onnx>=1.16"]),
    ("onnxruntime", ["onnxruntime>=1.18"]),
    ("scipy", ["scipy"]),
    ("python_speech_features", ["python_speech_features"]),
]:
    _ensure_pkg(pkg, args)


# -------------------------------------------------------------------------
# 3. Make avhubert importable
# -------------------------------------------------------------------------
# Add the av_hubert/ directory itself (NOT av_hubert/avhubert/) so that
# `import avhubert` resolves to av_hubert/avhubert/__init__.py as a real
# package. The 2026-05 attempt accidentally added the inner directory,
# making `import avhubert` see the package's *files* as top-level
# modules instead of treating the directory as a package.
if AVHUBERT_DIR not in sys.path:
    sys.path.insert(0, AVHUBERT_DIR)
print(f"Added to sys.path: {AVHUBERT_DIR}")

# Sanity-check that fairseq is a proper package (not a namespace package).
# fairseq.__file__ = None means PEP 420 namespace package, which is what
# happens when torch>=2.5 breaks the vendored fairseq commit's setup.
import fairseq                                                  # noqa: E402
if fairseq.__file__ is None:
    raise RuntimeError(
        "fairseq imported as a namespace package (no __init__.py loaded). "
        "This means the editable install didn't finish — almost always "
        "because torch is too new for the vendored fairseq commit. "
        "Downgrade torch to 2.0.1 in a fresh kernel before retrying. "
        "See docs/AVHUBERT_V3_BACKEND.md attempt log."
    )

try:
    import avhubert  # noqa: F401
    print("avhubert package imported.")
except Exception as e:
    print(f"avhubert import FAILED: {type(e).__name__}: {e}")
    raise


# -------------------------------------------------------------------------
# 4. HF auth + checkpoint
# -------------------------------------------------------------------------
from huggingface_hub import login, whoami, hf_hub_download, upload_file   # noqa: E402

token = os.environ.get("HF_TOKEN")
if not token and IS_KAGGLE:
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        pass
if token:
    login(token, add_to_git_credential=True)
print(f"HF user: {whoami().get('name', '?')}")

HF_REPO = "HereLiesAz/liperty-avhubert-encoder"
LOCAL_PT = "/kaggle/working/large_vox_iter5.pt" if IS_KAGGLE else os.path.join(WORK_DIR, "large_vox_iter5.pt")

if not os.path.exists(LOCAL_PT):
    print("Pulling large_vox_iter5.pt from HF mirror...")
    LOCAL_PT = hf_hub_download(
        repo_id=HF_REPO, filename="large_vox_iter5.pt",
        local_dir=os.path.dirname(LOCAL_PT),
    )
print(f"Checkpoint: {LOCAL_PT}  ({os.path.getsize(LOCAL_PT) / 1e9:.2f} GB)")


# -------------------------------------------------------------------------
# 5. Load via fairseq
# -------------------------------------------------------------------------
import torch                                                    # noqa: E402
import torch.nn as nn                                           # noqa: E402

# AV-HuBERT pretraining task wants a data dir with a few empty manifest
# files. We provide an empty one so task instantiation doesn't trip on
# FileNotFoundError.
empty_data_dir = os.path.join(WORK_DIR, "empty_data")
os.makedirs(empty_data_dir, exist_ok=True)
for fn in ["nframes.audio", "nframes.video", "test.tsv", "valid.tsv", "train.tsv"]:
    p = os.path.join(empty_data_dir, fn)
    if not os.path.exists(p):
        open(p, "w").close()

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
ONNX_PATH = os.path.join(WORK_DIR, "avhubert_visual_encoder.onnx")
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
    dynamo=False,
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

print(f"PyTorch out: {pt_out.shape}  range=[{pt_out.min():.3f}, {pt_out.max():.3f}]")
print(f"ONNX out:    {ort_out.shape}  range=[{ort_out.min():.3f}, {ort_out.max():.3f}]")
diff = np.abs(pt_out - ort_out)
print(f"Max abs diff: {diff.max():.6f}")
print(f"Mean abs diff: {diff.mean():.6f}")
if diff.max() >= 1e-2:
    print("WARNING: ONNX diverges from PyTorch by >=1e-2. Export may be subtly broken.")


# -------------------------------------------------------------------------
# 9. Upload to HF
# -------------------------------------------------------------------------
print("Uploading ONNX to HF mirror...")
upload_file(
    path_or_fileobj=ONNX_PATH,
    path_in_repo="avhubert_visual_encoder.onnx",
    repo_id=HF_REPO,
    repo_type="model",
    commit_message="ONNX-exported visual encoder for V3 backend research",
)
print(f"Uploaded -> https://huggingface.co/{HF_REPO}/blob/main/avhubert_visual_encoder.onnx")
print()
print("AV-HuBERT V3 encoder export complete.")
