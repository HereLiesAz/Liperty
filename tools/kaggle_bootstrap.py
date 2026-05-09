"""Fresh-kernel bootstrap for Kaggle GRID training.

After a session restart (e.g. switching accelerator), the IPython
kernel loses all namespace state. The notebook still has cells 1-5
that would set things up, but cell 5's `mp.solutions.face_detection`
crashes on Kaggle's Python 3.12 mediapipe wheel — so re-running the
notebook top-to-bottom is a trap.

This script replaces cells 1-5 (env detect, deps, HF auth, run config,
shared utilities) with a mediapipe-free version, leaving the kernel
in the same state cell 5 would have on success. Then exec
kaggle_haar_fix.py (face detection + GRID preprocessing utilities)
and kaggle_grid_train.py (dataset / model / training loop) on top.

Run from the Kaggle IPython console after the session restarts:

    import urllib.request
    exec(urllib.request.urlopen(
        'https://raw.githubusercontent.com/HereLiesAz/Liperty/main/tools/kaggle_bootstrap.py'
    ).read())
    exec(urllib.request.urlopen(
        'https://raw.githubusercontent.com/HereLiesAz/Liperty/main/tools/kaggle_grid_train.py'
    ).read())
    run_training(budget_min=360)

(kaggle_haar_fix is only needed if you'll re-run preprocessing in
this session. Skip it for training-only.)

The HF_TOKEN secret must be enabled for this notebook on Kaggle
(Add-ons -> Secrets) before running.
"""

import os, sys

# ---------------------------------------------------------------------------
# Cell 1 equiv — environment detection
# ---------------------------------------------------------------------------
IS_KAGGLE = os.path.exists("/kaggle/working") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ
try:
    import google.colab  # noqa
    IS_COLAB = True
except ImportError:
    IS_COLAB = False
ENV = "kaggle" if IS_KAGGLE else "colab" if IS_COLAB else "local"

import torch
print(f"Environment: {ENV}")
print(f"Python: {sys.version.split()[0]}, PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}", end="")
if torch.cuda.is_available():
    print(f"  {torch.cuda.get_device_name(0)}  cap={torch.cuda.get_device_capability(0)}")
else:
    print()


# ---------------------------------------------------------------------------
# Cell 2 equiv — install deps (skip mediapipe)
# ---------------------------------------------------------------------------
import subprocess
def _pip(*args):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  pip install {args} failed:\n{r.stderr[-500:]}")
        return False
    return True

print("Installing dependencies...")
_pip(
    "transformers>=4.46,<5.0",
    "huggingface_hub>=0.27,<1.0",
    "datasets>=3.0,<4.0",
    "accelerate>=1.0,<2.0",
    "pronouncing>=0.2.0",
    "decord>=0.6.0",
    "av>=12.0",
    "jiwer>=3.0",
)
print("  done.")


# ---------------------------------------------------------------------------
# Cell 3 equiv — HF auth
# ---------------------------------------------------------------------------
from huggingface_hub import login, whoami

_token = os.environ.get("HF_TOKEN")
if not _token and IS_KAGGLE:
    try:
        from kaggle_secrets import UserSecretsClient
        _token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        pass
if _token:
    login(_token, add_to_git_credential=True)
    print(f"HF user: {whoami()['name']}")
else:
    print("No HF_TOKEN — Add-ons -> Secrets -> enable HF_TOKEN, then re-run this script.")


# ---------------------------------------------------------------------------
# Cell 4 equiv — run config
# ---------------------------------------------------------------------------
HF_USER = "HereLiesAz"
HF_DATA_REPO_GRID = f"{HF_USER}/liperty-grid-preprocessed"
HF_DATA_REPO_TCD  = f"{HF_USER}/liperty-tcd-preprocessed"
HF_CKPT_REPO      = f"{HF_USER}/liperty-vsr-checkpoints"

RUN_NAME = "vmae-base-grid-tcd-v1"

TIME_BUDGET_MIN = 480 if IS_KAGGLE else 200

NUM_FRAMES = 16
IMG_SIZE   = 224
TIME_UPSAMPLE = 4

BATCH_SIZE   = 2
GRAD_ACCUM   = 8
LR           = 1e-4
WEIGHT_DECAY = 1e-2
WARMUP_STEPS = 500
NUM_WORKERS  = 2
USE_FP16     = True

CKPT_EVERY_STEPS = 200
LOG_EVERY_STEPS  = 20
KEEP_LAST_CKPTS  = 3

if IS_KAGGLE:
    WORK_DIR = "/kaggle/working/work"
    DATA_DIR = "/kaggle/working/data"
    CKPT_DIR = "/kaggle/working/ckpt"
else:
    WORK_DIR = "/content/work"
    DATA_DIR = "/content/data"
    CKPT_DIR = "/content/ckpt"

PHONEME_VOCAB = ["_", "AA","AE","AH","AO","AW","AY","B","CH","D","DH","EH","ER","EY",
                 "F","G","HH","IH","IY","JH","K","L","M","N","NG","OW","OY","P",
                 "R","S","SH","T","TH","UH","UW","V","W","Y","Z","ZH"]
VOCAB_SIZE = len(PHONEME_VOCAB)
BLANK_IDX  = 0
PHONEME_TO_IDX = {p: i for i, p in enumerate(PHONEME_VOCAB)}

for _d in (WORK_DIR, DATA_DIR, CKPT_DIR):
    os.makedirs(_d, exist_ok=True)


# ---------------------------------------------------------------------------
# Cell 5 equiv (mediapipe-free) — repo + shared imports
# ---------------------------------------------------------------------------
import re, time, json, io, shutil, urllib.request, hashlib
from pathlib import Path
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import cv2

from huggingface_hub import HfApi, create_repo, upload_file, hf_hub_download, snapshot_download

api = HfApi()
for _repo_id, _kind in [
    (HF_DATA_REPO_GRID, "dataset"),
    (HF_DATA_REPO_TCD,  "dataset"),
    (HF_CKPT_REPO,      "model"),
]:
    try:
        create_repo(repo_id=_repo_id, repo_type=_kind, private=True, exist_ok=True)
    except Exception as e:
        print(f"create_repo({_repo_id}): {e}")


print()
print(f"Bootstrap complete.")
print(f"  Run:           {RUN_NAME}")
print(f"  Budget:        {TIME_BUDGET_MIN} min")
print(f"  Effective bs:  {BATCH_SIZE * GRAD_ACCUM}")
print(f"  Vocab size:    {VOCAB_SIZE}")
print(f"  CTC timesteps: {NUM_FRAMES // 2 * TIME_UPSAMPLE}")
print()
print("Next: exec kaggle_grid_train.py, then run_training()")
