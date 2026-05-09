"""In-session fix for the Kaggle preprocessing notebook.

The notebook's setup cell crashed with
    AttributeError: module 'mediapipe' has no attribute 'solutions'
because the mediapipe wheel that resolves on Kaggle's Python 3.12 image
is missing the legacy `solutions` submodule. This module replaces the
mediapipe face detector with OpenCV's Haar cascade (which ships with
every cv2 install) and restages the rest of the cell-5 / cell-6
utilities so the GRID preprocessing loop can run.

Run from the Kaggle notebook's IPython console:

    import urllib.request
    exec(urllib.request.urlopen(
        'https://raw.githubusercontent.com/HereLiesAz/Liperty/main/tools/kaggle_haar_fix.py'
    ).read())

Picks up these names from the kernel namespace (already created by
cells 4 + the surviving prefix of cell 5):
    cv2, np, torch, os, sys, re, time, subprocess, urllib, shutil
    Path, glob, pronouncing
    HfApi, create_repo, upload_file, hf_hub_download, snapshot_download
    WORK_DIR, HF_DATA_REPO_GRID, HF_DATA_REPO_TCD
    NUM_FRAMES, IMG_SIZE, PHONEME_TO_IDX, TIME_BUDGET_MIN
"""

import cv2
import numpy as np
import torch
import os, re, time, shutil, subprocess
import urllib.request
from pathlib import Path
from glob import glob

import pronouncing
from huggingface_hub import HfApi, upload_file


# Ensure HfApi instance exists (cell 5 did this before the crash, but be safe).
try:
    api  # noqa: F821
except NameError:
    api = HfApi()


# ---------------------------------------------------------------------------
# 1. Face detection — OpenCV Haar cascade, no mediapipe.
# ---------------------------------------------------------------------------
_haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade = cv2.CascadeClassifier(_haar_path)
if _face_cascade.empty():
    raise RuntimeError(f"Failed to load Haar cascade from {_haar_path}")


def detect_face_bbox(frame_bgr):
    """Square face bounding box with 30% margin, in pixel coords. Returns
    (x1, y1, x2, y2) or None."""
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=4, minSize=(60, 60)
    )
    if len(faces) == 0:
        return None
    fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
    x1, y1, x2, y2 = fx, fy, fx + fw, fy + fh
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    side = int(max(x2 - x1, y2 - y1) * 1.30)
    half = side // 2
    return (
        max(0, cx - half), max(0, cy - half),
        min(w, cx + half), min(h, cy + half),
    )


# ---------------------------------------------------------------------------
# 2. text -> phoneme ids (uses PHONEME_TO_IDX already in kernel namespace)
# ---------------------------------------------------------------------------
def text_to_phoneme_ids(text):
    ids = []
    for word in text.lower().split():
        word = re.sub(r"[^a-z]", "", word)
        if not word:
            continue
        phs = pronouncing.phones_for_word(word)
        if not phs:
            continue
        for ph in phs[0].split():
            ph_clean = re.sub(r"\d+", "", ph)
            if ph_clean in PHONEME_TO_IDX:  # noqa: F821 — supplied by kernel
                ids.append(PHONEME_TO_IDX[ph_clean])
    return ids


# ---------------------------------------------------------------------------
# 3. preprocess_video (depends on Haar detect_face_bbox)
# ---------------------------------------------------------------------------
def preprocess_video(path, num_frames=None, size=None):
    """Decode -> sample N frames evenly -> face-crop (sticky bbox) -> 224 RGB.
    Returns (T,H,W,C) uint8 or None on failure.
    """
    if num_frames is None:
        num_frames = NUM_FRAMES   # noqa: F821
    if size is None:
        size = IMG_SIZE           # noqa: F821
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < num_frames:
        cap.release(); return None
    indices = np.linspace(0, total - 1, num_frames).astype(int)
    out = np.zeros((num_frames, size, size, 3), dtype=np.uint8)
    last_bbox = None
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            cap.release(); return None
        bbox = detect_face_bbox(frame) or last_bbox
        if bbox is None:
            cap.release(); return None
        last_bbox = bbox
        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            cap.release(); return None
        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        out[i] = crop
    cap.release()
    return out


# ---------------------------------------------------------------------------
# 4. GRID parsing + per-speaker preprocessing
# ---------------------------------------------------------------------------
GRID_VIDEO_URL = "https://zenodo.org/records/3625687/files/s{sp}.zip?download=1"

_GRID_CMD   = {"b": "bin", "l": "lay", "p": "place", "s": "set"}
_GRID_COLOR = {"b": "blue", "g": "green", "r": "red", "w": "white"}
_GRID_PREP  = {"a": "at", "b": "by", "i": "in", "w": "with"}
_GRID_DIGIT = {"1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
               "6": "six", "7": "seven", "8": "eight", "9": "nine",
               "0": "zero", "z": "zero"}
_GRID_ADV   = {"a": "again", "n": "now", "p": "please", "s": "soon"}


def parse_grid_filename(stem):
    if len(stem) < 6:
        return None
    cmd = _GRID_CMD.get(stem[0])
    col = _GRID_COLOR.get(stem[1])
    prp = _GRID_PREP.get(stem[2])
    let = stem[3] if stem[3].isalpha() else None
    dig = _GRID_DIGIT.get(stem[4])
    adv = _GRID_ADV.get(stem[5])
    if any(x is None for x in (cmd, col, prp, let, dig, adv)):
        return None
    return f"{cmd} {col} {prp} {let} {dig} {adv}"


def existing_grid_speakers():
    try:
        files = api.list_repo_files(HF_DATA_REPO_GRID, repo_type="dataset")  # noqa: F821
    except Exception:
        return set()
    out = set()
    for f in files:
        m = re.match(r"s(\d+)\.pt$", f)
        if m:
            out.add(int(m.group(1)))
    return out


def preprocess_grid_speaker(sp):
    work = Path(WORK_DIR) / f"grid_s{sp}"   # noqa: F821
    work.mkdir(parents=True, exist_ok=True)
    zip_path = work / f"s{sp}.zip"
    if not zip_path.exists():
        url = GRID_VIDEO_URL.format(sp=sp)
        print(f"  [s{sp}] downloading...")
        try:
            urllib.request.urlretrieve(url, zip_path)
        except Exception as e:
            print(f"  [s{sp}] download failed: {e}")
            shutil.rmtree(work, ignore_errors=True); return False
    extract = work / "videos"
    if not extract.exists():
        extract.mkdir()
        subprocess.run(["unzip", "-q", "-o", str(zip_path), "-d", str(extract)], check=True)
    video_files = sorted(glob(str(extract / "**" / "*.mpg"), recursive=True))
    if not video_files:
        print(f"  [s{sp}] no .mpg files in zip")
        shutil.rmtree(work, ignore_errors=True); return False
    print(f"  [s{sp}] {len(video_files)} clips found, processing...")
    frames_list, ph_list, text_list = [], [], []
    for i, vid in enumerate(video_files):
        text = parse_grid_filename(Path(vid).stem)
        if text is None: continue
        ph = text_to_phoneme_ids(text)
        if not ph: continue
        f = preprocess_video(vid)
        if f is None: continue
        frames_list.append(f); ph_list.append(ph); text_list.append(text)
        if (i + 1) % 200 == 0:
            print(f"    [s{sp}] {i+1}/{len(video_files)}")
    if not frames_list:
        print(f"  [s{sp}] no usable clips")
        shutil.rmtree(work, ignore_errors=True); return False
    out_path = work / f"s{sp}.pt"
    torch.save({
        "frames":   torch.from_numpy(np.stack(frames_list)),   # (N, T, H, W, C) uint8
        "phonemes": ph_list,
        "texts":    text_list,
        "speaker":  sp,
    }, out_path)
    sz_mb = out_path.stat().st_size / 1e6
    print(f"  [s{sp}] uploading {sz_mb:.0f} MB...")
    upload_file(
        path_or_fileobj=str(out_path),
        path_in_repo=f"s{sp}.pt",
        repo_id=HF_DATA_REPO_GRID,   # noqa: F821
        repo_type="dataset",
        commit_message=f"GRID s{sp}: {len(frames_list)} clips",
    )
    shutil.rmtree(work, ignore_errors=True)
    return True


# ---------------------------------------------------------------------------
# 5. Driver — kick off the GRID preprocessing right now.
# ---------------------------------------------------------------------------
def run_grid_preprocessing(time_budget_min=None):
    if time_budget_min is None:
        try:
            time_budget_min = TIME_BUDGET_MIN   # noqa: F821
        except NameError:
            time_budget_min = 480   # 8 hours, leave room for the 9h Kaggle session
    target = [s for s in range(1, 35) if s != 21]   # s21 has no video in GRID
    done = existing_grid_speakers()
    pending = [s for s in target if s not in done]
    print(f"Done:    {sorted(done)}")
    print(f"Pending: {pending}")
    print(f"Time budget: {time_budget_min} min")
    t0 = time.time()
    for sp in pending:
        elapsed_min = (time.time() - t0) / 60
        if elapsed_min > time_budget_min - 30:
            print(f"Within 30 min of budget; stopping. Re-run to continue.")
            break
        ok = preprocess_grid_speaker(sp)
        elapsed = (time.time() - t0) / 60
        print(f"  [s{sp}] {'ok' if ok else 'FAILED'}  (cumulative {elapsed:.1f} min)")


print("kaggle_haar_fix loaded:")
print("  detect_face_bbox, text_to_phoneme_ids, preprocess_video,")
print("  parse_grid_filename, existing_grid_speakers, preprocess_grid_speaker,")
print("  run_grid_preprocessing")
print()
print("Next: call run_grid_preprocessing()")
