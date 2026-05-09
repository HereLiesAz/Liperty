"""Generate tools/preprocess_landmarks_resumable.ipynb.

Per-speaker landmark extraction from the existing pixel preprocessed shards
(`liperty-grid-preprocessed`, `liperty-tcd-preprocessed`). Reads the 16-frame
RGB face crops, runs dlib's HOG face detector + 68-point shape predictor on
every frame, and packs per-speaker .pt shards into a new HF dataset
(`liperty-grid-landmarks`, `liperty-tcd-landmarks`) in a format the landmark
training notebook can consume.

Resumable per-speaker. Re-running after a session kick skips already-uploaded
shards.

Run with:  python tools/_build_landmark_preprocess_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "preprocess_landmarks_resumable.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.splitlines(keepends=True)}


cells: list[dict] = []

cells.append(md("""\
# Liperty — Landmark extraction from pixel shards

Companion to `train_landmark_lrs3_resumable.ipynb`. Reads the GRID / TCD-TIMIT pixel preprocessed shards (output of `train_grid_tcd_resumable.ipynb`'s preprocess cells), runs dlib face detection + 68-point landmark prediction over every frame, and writes per-speaker landmark shards to a new HF dataset.

**Why this exists.** The landmark training notebook can consume `e1lephant/lrs3-landmark` directly (LRS3 landmarks at full temporal resolution, MIT-licensed) — but no LRS3 transcripts. By extracting landmarks from GRID + TCD-TIMIT in the same format, we get a **trainable-today** landmark dataset (~40k clips with phoneme labels) that the same training notebook handles uniformly. When LRS3 transcripts eventually arrive, you flip one config line and add 150k more clips.

**Output format** per speaker shard (one `.pt` file):

```python
{
    "landmarks": [                          # one entry per clip
        [ndarray(68, 2) | None, ...],       # per-frame dlib 68-point landmarks
        ...
    ],
    "phonemes": [[int, ...], ...],          # ARPABET phoneme indices, copied from pixel shard
    "texts":    [str, ...],                 # raw transcripts, copied from pixel shard
    "speaker":  <id>,
}
```

Same per-frame None-for-missing convention as `e1lephant/lrs3-landmark`. The landmark training notebook's loader treats both formats uniformly.

**Resumable.** Per-speaker. Skips speakers already on HF Hub. Time-budgeted.
"""))

cells.append(md("""\
## 1. Environment + deps
"""))

cells.append(code("""\
import os, sys, platform
import torch

IS_KAGGLE = os.path.exists("/kaggle/working") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ
try:
    import google.colab  # noqa
    IS_COLAB = True
except ImportError:
    IS_COLAB = False
ENV = "kaggle" if IS_KAGGLE else "colab" if IS_COLAB else "local"
print(f"Environment: {ENV}")
!nvidia-smi 2>&1 | head -10
print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
"""))

cells.append(code("""\
%%capture
# dlib needs build tools; Colab + Kaggle have them pre-installed.
!pip install -q \\
    "dlib>=19.24" \\
    "huggingface_hub>=0.27,<1.0" \\
    "numpy>=1.24"
print("Deps installed.")
"""))

cells.append(md("""\
## 2. HF auth
"""))

cells.append(code("""\
from huggingface_hub import login, whoami

token = os.environ.get("HF_TOKEN")
if not token and IS_KAGGLE:
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as e:
        print(f"Kaggle Secrets: {e}")
if not token and IS_COLAB:
    try:
        from google.colab import userdata
        token = userdata.get("HF_TOKEN")
    except Exception:
        pass

if token:
    login(token, add_to_git_credential=True)
else:
    from huggingface_hub import notebook_login
    notebook_login()

print(f"HF user: {whoami()['name']}")
"""))

cells.append(md("""\
## 3. Config

Two pairs of repos: pixel input → landmark output. Skip a corpus by setting `RUN_<X> = False`.
"""))

cells.append(code("""\
# === EDIT ME ===
HF_USER = "CHANGE-ME"
# ===============
assert HF_USER != "CHANGE-ME", "Set HF_USER to your HuggingFace username."

# Source pixel shards (output of train_grid_tcd_resumable.ipynb preprocessing)
HF_PIXEL_REPO_GRID = f"{HF_USER}/liperty-grid-preprocessed"
HF_PIXEL_REPO_TCD  = f"{HF_USER}/liperty-tcd-preprocessed"

# Destination landmark shards
HF_LM_REPO_GRID = f"{HF_USER}/liperty-grid-landmarks"
HF_LM_REPO_TCD  = f"{HF_USER}/liperty-tcd-landmarks"

RUN_GRID = True
RUN_TCD  = True

# Time budget. Kaggle GPU sessions cap at ~9h, Colab free at ~4h. Leave 30 min
# buffer for upload of in-flight shard.
TIME_BUDGET_MIN = 480 if IS_KAGGLE else 200

# Working dir
if IS_KAGGLE:
    WORK_DIR = "/kaggle/working/work"
else:
    WORK_DIR = "/content/work"
os.makedirs(WORK_DIR, exist_ok=True)

print(f"GRID:  {HF_PIXEL_REPO_GRID}  →  {HF_LM_REPO_GRID}")
print(f"TCD:   {HF_PIXEL_REPO_TCD}  →  {HF_LM_REPO_TCD}")
print(f"Budget: {TIME_BUDGET_MIN} min")
"""))

cells.append(md("""\
## 4. Download dlib's 68-point shape predictor

The model file is ~99 MB, distributed by Davis King under the dlib examples. One-time per session.
"""))

cells.append(code("""\
import urllib.request
import bz2
from pathlib import Path

DLIB_MODEL_URL = "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2"
bz2_path = Path(WORK_DIR) / "shape_predictor_68_face_landmarks.dat.bz2"
dat_path = bz2_path.with_suffix("")

if not dat_path.exists():
    if not bz2_path.exists():
        print(f"Downloading {DLIB_MODEL_URL}...")
        urllib.request.urlretrieve(DLIB_MODEL_URL, bz2_path)
    print("Decompressing...")
    with bz2.open(bz2_path, "rb") as src, open(dat_path, "wb") as dst:
        dst.write(src.read())
    bz2_path.unlink()

print(f"dlib model: {dat_path} ({dat_path.stat().st_size / 1e6:.0f} MB)")
"""))

cells.append(md("""\
## 5. Landmark extraction primitives

Per-frame: dlib HOG face detector (fast, CPU) → 68-point shape predictor. Returns `(68, 2) float32` in pixel coords, or `None` if no face was found. Sticky-bbox fallback: if a frame's detection fails but the previous frame had a face, we run the predictor on the previous frame's bbox.
"""))

cells.append(code("""\
import numpy as np
import dlib

_detector = dlib.get_frontal_face_detector()
_predictor = dlib.shape_predictor(str(dat_path))

def extract_landmarks_frame(frame_rgb, fallback_rect=None):
    \"\"\"frame_rgb: (H, W, 3) uint8. Returns ((68,2) float32, dlib.rectangle) or (None, None).\"\"\"
    rects = _detector(frame_rgb, 0)
    rect = None
    if rects:
        rect = max(rects, key=lambda r: r.width() * r.height())
    elif fallback_rect is not None:
        rect = fallback_rect
    else:
        return None, None
    shape = _predictor(frame_rgb, rect)
    arr = np.array([[p.x, p.y] for p in shape.parts()], dtype=np.float32)
    return arr, rect

def extract_landmarks_clip(clip_frames):
    \"\"\"clip_frames: (T, H, W, 3) uint8. Returns List[Optional[(68,2) float32]] of length T.\"\"\"
    out = []
    last_rect = None
    for t in range(clip_frames.shape[0]):
        frame = clip_frames[t]
        if hasattr(frame, "numpy"):
            frame = frame.numpy()
        # Ensure contiguous uint8
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        lm, rect = extract_landmarks_frame(frame, fallback_rect=last_rect)
        out.append(lm)
        if rect is not None:
            last_rect = rect
    return out

# Smoke test on a synthetic image
test = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
lm, rect = extract_landmarks_frame(test)
print(f"Synthetic test: rect={rect} (None expected unless lucky), landmarks present={lm is not None}")
"""))

cells.append(md("""\
## 6. Per-speaker preprocessing loop

For each pending speaker shard:

1. Download the pixel shard from `liperty-<corpus>-preprocessed`.
2. Iterate clips → iterate frames → run dlib → collect `List[Optional[ndarray(68,2)]]` per clip.
3. Save a `.pt` shard with landmarks + phonemes + texts + speaker.
4. Upload to `liperty-<corpus>-landmarks`.
5. Delete local data, move on.

A 16-frame pixel shard with ~1000 clips processes in ~3-5 min on a T4 (CPU-bound; the GPU is unused). Per-corpus full preprocessing is therefore ~2-3 hours of CPU work — well under one Kaggle session budget.
"""))

cells.append(code("""\
import time, re, shutil
from huggingface_hub import HfApi, create_repo, upload_file, hf_hub_download, list_repo_files

api = HfApi()

def existing_shards(repo_id):
    try:
        files = list_repo_files(repo_id, repo_type="dataset")
    except Exception:
        return set()
    return {f for f in files if f.endswith(".pt")}

def ensure_dataset_repo(repo_id):
    try:
        create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    except Exception as e:
        print(f"create_repo({repo_id}): {e}")

def list_source_shards(repo_id):
    try:
        return sorted(f for f in list_repo_files(repo_id, repo_type="dataset") if f.endswith(".pt"))
    except Exception as e:
        print(f"list_repo_files({repo_id}): {e}")
        return []

def process_corpus(name, src_repo, dst_repo, deadline_s, t0):
    if not RUN_GRID and name == "GRID": return
    if not RUN_TCD  and name == "TCD":  return
    print(f"\\n=== {name}: {src_repo}  →  {dst_repo} ===")
    ensure_dataset_repo(dst_repo)
    src_files = list_source_shards(src_repo)
    if not src_files:
        print(f"  no source shards found in {src_repo}; skip.")
        return
    done = existing_shards(dst_repo)
    pending = [f for f in src_files if f not in done]
    print(f"  source shards: {len(src_files)}   already done: {len(done)}   pending: {len(pending)}")

    for shard_name in pending:
        elapsed_min = (time.time() - t0) / 60
        if elapsed_min > deadline_s / 60 - 5:
            print(f"  within 5 min of budget; stopping. Re-run to continue.")
            return
        print(f"  [{name}] {shard_name}: downloading pixel shard...")
        try:
            local_pixel = hf_hub_download(
                repo_id=src_repo, repo_type="dataset",
                filename=shard_name, local_dir=WORK_DIR,
            )
        except Exception as e:
            print(f"  [{name}] {shard_name}: download failed: {e}; skip.")
            continue
        try:
            data = torch.load(local_pixel, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"  [{name}] {shard_name}: load failed: {e}; skip.")
            continue

        frames_all = data["frames"]      # (N, T, H, W, C) uint8
        phonemes   = data.get("phonemes", [])
        texts      = data.get("texts", [])
        speaker    = data.get("speaker", shard_name.removesuffix(".pt"))
        N = frames_all.shape[0]

        landmarks_per_clip = []
        n_face_failures = 0
        n_total_frames  = 0
        for i in range(N):
            clip = frames_all[i]
            lms = extract_landmarks_clip(clip)
            landmarks_per_clip.append(lms)
            n_face_failures += sum(1 for x in lms if x is None)
            n_total_frames  += len(lms)
            if (i + 1) % 200 == 0:
                miss = n_face_failures / max(1, n_total_frames)
                print(f"    [{name}] {shard_name} clip {i+1}/{N} (frame miss rate so far: {miss:.1%})")

        out_path = Path(WORK_DIR) / f"lm-{shard_name}"
        torch.save({
            "landmarks": landmarks_per_clip,
            "phonemes":  phonemes,
            "texts":     texts,
            "speaker":   speaker,
            "source":    f"{src_repo}#{shard_name}",
        }, out_path)
        sz_mb = out_path.stat().st_size / 1e6
        miss = n_face_failures / max(1, n_total_frames)
        print(f"  [{name}] {shard_name}: {N} clips, miss rate {miss:.1%}, packed {sz_mb:.1f} MB; uploading...")
        upload_file(
            path_or_fileobj=str(out_path),
            path_in_repo=shard_name,
            repo_id=dst_repo, repo_type="dataset",
            commit_message=f"{name} {shard_name}: {N} clips, miss {miss:.1%}",
        )
        try: Path(local_pixel).unlink()
        except: pass
        out_path.unlink()
        print(f"  [{name}] {shard_name}: done.")

deadline_s = TIME_BUDGET_MIN * 60
t0 = time.time()
process_corpus("GRID", HF_PIXEL_REPO_GRID, HF_LM_REPO_GRID, deadline_s, t0)
process_corpus("TCD",  HF_PIXEL_REPO_TCD,  HF_LM_REPO_TCD,  deadline_s, t0)

print(f"\\nFinished. Elapsed: {(time.time() - t0)/60:.1f} min.")
print(f"GRID landmark shards on HF: {sorted(existing_shards(HF_LM_REPO_GRID))}")
print(f"TCD  landmark shards on HF: {sorted(existing_shards(HF_LM_REPO_TCD))}")
"""))

cells.append(md("""\
## 7. Troubleshooting

**"dlib face detector finds nothing for many frames."** GRID is recorded with the speaker filling the frame; TCD-TIMIT clips are also tight. The 30%-margin face crop in the pixel shards keeps the face large in the 224×224 image, but dlib's HOG detector can still miss. Two mitigations are already in the loop:
- **Sticky bbox**: if a frame's detection fails but a previous frame had a face, we re-run the predictor on the previous frame's bbox. Captures most blink-frame failures.
- **Per-clip landmark interpolation** happens later in the training-time dataset class (`landmark_train` notebook cell 6), not here. We just record `None` and pass through.

If the miss rate per shard is >40%, the source pixel shard's face crops are likely off-center. Inspect the source `frames` tensor and verify the face is roughly centered.

**"dlib install fails on Kaggle / Colab."** Both have `cmake`, `build-essential`, and `libopenblas-dev` pre-installed, so `pip install dlib` should work in 1-3 min (it builds from source). If it doesn't:

```
!apt-get install -y build-essential cmake libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev
!pip install dlib
```

**"This is CPU-bound and the GPU sits idle."** Yes — dlib HOG is CPU-only. You can run two preprocessing notebooks in parallel on different Kaggle accounts without contending for the GPU. Or run a CPU-only Kaggle kernel (12h limit) for this step.

**"My time budget hit and a shard was mid-flight."** The shard was either fully uploaded (`upload_file` is atomic on HF) or wasn't, in which case the next session re-processes it. No partial state to recover.
"""))


nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": []},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB, {len(cells)} cells)")
