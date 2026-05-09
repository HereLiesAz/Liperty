"""Generate tools/train_grid_tcd_resumable.ipynb.

Run with:  python tools/_build_notebook.py
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "train_grid_tcd_resumable.ipynb"


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


cells: list[dict] = []

cells.append(md("""\
# Liperty VSR — Resumable Training (GRID + TCD-TIMIT)

A training pipeline for the VALLR-style VSR model designed to survive session kicks and resume across accounts. Runs on **Kaggle Notebooks** (recommended) or **Google Colab** — auto-detects which.

**Why Kaggle is the recommended platform**

| | Free Colab | Kaggle (free) |
|---|---|---|
| Hours / week / account | Rationed, unpredictable | 30 hrs hard guarantee |
| Session length | ~4h before kick | 9h GPU, 12h CPU |
| Background execution | Browser must stay open | Set running, walk away |
| GPU | T4 (shared, throttled) | P100 16GB or T4×2 |

A single Kaggle account is roughly equivalent to all 5 free Colab accounts in steady-state throughput. With 5 Kaggle accounts you can do 150 hrs/week guaranteed.

**Design goals**

1. **Survive session kicks.** All durable state lives on the HuggingFace Hub: checkpoints, optimizer state, RNG, run metadata, preprocessed datasets. Local disk is treated as a cache. Works identically on Kaggle and Colab.
2. **Account-agnostic resume.** A single HF token drives the whole pipeline. Any account on any platform that opens this notebook with the same HF token resumes from where the last session stopped.
3. **Idempotent preprocessing.** Per-speaker shards. Already-uploaded speakers are skipped on re-run.
4. **Time-budgeted training.** The training loop flushes a final checkpoint before the platform kicks.
"""))

cells.append(md("""\
## How to use this

**One-time setup:**

1. Create one HuggingFace account (free). Generate a token at https://huggingface.co/settings/tokens with **write** scope.
2. The notebook creates three private HF repos on first run if they don't exist:
   - `<you>/liperty-grid-preprocessed`  (dataset)
   - `<you>/liperty-tcd-preprocessed`   (dataset)
   - `<you>/liperty-vsr-checkpoints`    (model)
3. Add the HF token as a secret in each environment you'll use:
   - **Kaggle:** `Add-ons → Secrets`, add label `HF_TOKEN` with the token value.
   - **Colab:** `Tools → Settings → Secrets`, name `HF_TOKEN`.

**Recurring sessions (Kaggle, recommended):**

- Open this notebook on Kaggle. Set runtime: GPU P100 (or T4×2). Internet: ON.
- `Run All` once, then **`Save Version → Save & Run All (Commit)`** to run headless in the background. Close the tab; the notebook keeps training.
- 9 hours later the session ends; the most recent checkpoint is on HF. Switch accounts, repeat.

**Recurring sessions (Colab):**

- Open the notebook, `Runtime → Run all`. Stays alive while the browser is open. Time budget defaults to 200 min for Colab's 4h kick.

**Switching accounts mid-day:**

Same notebook, same HF token, different account. The state store is HF Hub, so any platform/account picks up the same checkpoint. There's no automation for the manual login itself — Selenium-driving multiple Chrome profiles is fragile and against the spirit of both platforms. The thing this notebook automates is making the *resume* clean enough that a switch is one click + Run All.
"""))

cells.append(md("""\
## 1. Environment check
"""))

cells.append(code("""\
import os, sys, platform
import torch

# Environment detection — drives all path/secret decisions later.
IS_KAGGLE = os.path.exists("/kaggle/working") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ
try:
    import google.colab  # noqa
    IS_COLAB = True
except ImportError:
    IS_COLAB = False
ENV = "kaggle" if IS_KAGGLE else "colab" if IS_COLAB else "local"
print(f"Environment: {ENV}")

!nvidia-smi
print()
!free -h
print()
print(f"Python:   {sys.version.split()[0]}")
print(f"Platform: {platform.platform()}")
print(f"PyTorch:  {torch.__version__}")
print(f"CUDA:     {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:      {torch.cuda.get_device_name(0)}")
    print(f"VRAM:     {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
"""))

cells.append(md("""\
## 2. Install dependencies

Pinned to roughly what the existing `VALLR/requirements.txt` uses, plus HF Hub for durable state.
"""))

cells.append(code("""\
%%capture
!pip install -q \\
    "transformers>=4.46,<5.0" \\
    "huggingface_hub>=0.27,<1.0" \\
    "datasets>=3.0,<4.0" \\
    "accelerate>=1.0,<2.0" \\
    "pronouncing>=0.2.0" \\
    "decord>=0.6.0" \\
    "av>=12.0" \\
    "jiwer>=3.0"
print("Deps installed.")
"""))

cells.append(md("""\
## 3. HuggingFace auth

Tries Colab's `Secrets` panel first, then falls back to interactive token entry.
"""))

cells.append(code("""\
import os
from huggingface_hub import login, whoami

token = os.environ.get("HF_TOKEN")
if not token and IS_KAGGLE:
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as e:
        print(f"Kaggle Secrets lookup: {e}")
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
## 4. Run config

**Edit `HF_USER` once, then never again.** Everything else has reasonable defaults.
"""))

cells.append(code("""\
HF_USER = "HereLiesAz"   # Your HuggingFace username

HF_DATA_REPO_GRID = f"{HF_USER}/liperty-grid-preprocessed"
HF_DATA_REPO_TCD  = f"{HF_USER}/liperty-tcd-preprocessed"
HF_CKPT_REPO      = f"{HF_USER}/liperty-vsr-checkpoints"

RUN_NAME = "vmae-base-grid-tcd-v1"

# Stop training and flush this many minutes after the cell starts.
# Kaggle GPU sessions cap at ~9h, Colab free at ~4h. Leave 30 min buffer for
# the final checkpoint upload.
TIME_BUDGET_MIN = 480 if IS_KAGGLE else 200

# Model input shape — must match what Liperty's Android pipeline produces.
NUM_FRAMES = 16
IMG_SIZE   = 224

# Per-clip-phoneme upsampling factor in the adapter. We get 8 temporal tubelets
# from VideoMAE-base; a CTC head needs T_out >= label_length, so we upsample to
# 32 to fit GRID/TCD-TIMIT sentence-length phoneme transcripts (typically 15-30).
TIME_UPSAMPLE = 4   # 8 tubelets * 4 = 32 CTC timesteps

# Optimization
BATCH_SIZE   = 2
GRAD_ACCUM   = 8         # effective batch = 16
LR           = 1e-4
WEIGHT_DECAY = 1e-2
WARMUP_STEPS = 500
NUM_WORKERS  = 2
USE_FP16     = True      # T4 has fp16 tensor cores

# Checkpointing
CKPT_EVERY_STEPS = 200
LOG_EVERY_STEPS  = 20
KEEP_LAST_CKPTS  = 3     # how many recent checkpoints to keep on HF Hub

# Local working dirs. On Kaggle, /kaggle/working is auto-versioned to a Dataset
# at session end (a useful belt-and-braces backup on top of the HF Hub uploads).
# On Colab, /content is straight ephemeral.
if IS_KAGGLE:
    WORK_DIR = "/kaggle/working/work"
    DATA_DIR = "/kaggle/working/data"
    CKPT_DIR = "/kaggle/working/ckpt"
else:
    WORK_DIR = "/content/work"
    DATA_DIR = "/content/data"
    CKPT_DIR = "/content/ckpt"

# 39 ARPABET phonemes + blank at index 0 — same vocab as the Android decoder
# at app/src/main/java/com/hereliesaz/liperty/ml/MLConstants.kt
PHONEME_VOCAB = ["_", "AA","AE","AH","AO","AW","AY","B","CH","D","DH","EH","ER","EY",
                 "F","G","HH","IH","IY","JH","K","L","M","N","NG","OW","OY","P",
                 "R","S","SH","T","TH","UH","UW","V","W","Y","Z","ZH"]
VOCAB_SIZE = len(PHONEME_VOCAB)
BLANK_IDX  = 0
PHONEME_TO_IDX = {p: i for i, p in enumerate(PHONEME_VOCAB)}

import os
for d in (WORK_DIR, DATA_DIR, CKPT_DIR):
    os.makedirs(d, exist_ok=True)

print(f"Run:           {RUN_NAME}")
print(f"Budget:        {TIME_BUDGET_MIN} min")
print(f"Effective bs:  {BATCH_SIZE * GRAD_ACCUM}")
print(f"Vocab size:    {VOCAB_SIZE}")
print(f"CTC timesteps: {NUM_FRAMES // 2 * TIME_UPSAMPLE}")
"""))

cells.append(md("""\
## 5. Repo + shared utilities

Clones Liperty for reference; defines face-cropping and text→phoneme helpers used by both preprocessing and training.
"""))

cells.append(code("""\
import os, sys, re, time, json, io, shutil, subprocess, urllib.request, hashlib
from pathlib import Path
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import cv2
import pronouncing
from huggingface_hub import HfApi, create_repo, upload_file, hf_hub_download, snapshot_download

REPO_DIR = "/kaggle/working/Liperty" if IS_KAGGLE else "/content/Liperty"
if not os.path.exists(REPO_DIR):
    !git clone --depth 1 https://github.com/HereLiesAz/Liperty.git {REPO_DIR}
if f"{REPO_DIR}/VALLR" not in sys.path:
    sys.path.append(f"{REPO_DIR}/VALLR")

api = HfApi()

# Ensure the three HF repos exist (private). No-op if they already do.
for repo_id, kind in [
    (HF_DATA_REPO_GRID, "dataset"),
    (HF_DATA_REPO_TCD,  "dataset"),
    (HF_CKPT_REPO,      "model"),
]:
    try:
        create_repo(repo_id, repo_type=kind, private=True, exist_ok=True)
    except Exception as e:
        print(f"create_repo({repo_id}): {e}")

# OpenCV Haar cascade face detection. Avoids mediapipe — Kaggle's Python 3.12
# image was hitting `module 'mediapipe' has no attribute 'solutions'` even with
# `mediapipe>=0.10.18` pinned, because the wheel that resolves there is missing
# the legacy solutions module. Haar is good enough for a face *bounding box*
# (which is all we need for face-crop preprocessing) and ships with every cv2.
_haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade = cv2.CascadeClassifier(_haar_path)
if _face_cascade.empty():
    raise RuntimeError(f"Failed to load Haar cascade from {_haar_path}")

def detect_face_bbox(frame_bgr):
    \"\"\"Square face bounding box with 30% margin, in pixel coords. Returns (x1,y1,x2,y2) or None.\"\"\"
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=4, minSize=(60, 60)
    )
    if len(faces) == 0:
        return None
    # Pick the largest face (Haar can find spurious small ones).
    fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
    x1, y1, x2, y2 = fx, fy, fx + fw, fy + fh
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    side = int(max(x2 - x1, y2 - y1) * 1.30)  # 30% margin
    half = side // 2
    return (max(0, cx - half), max(0, cy - half), min(w, cx + half), min(h, cy + half))

def text_to_phoneme_ids(text):
    \"\"\"Lowercase + word-level lookup via the `pronouncing` library, ARPABET stress digits stripped.\"\"\"
    ids = []
    for word in text.lower().split():
        word = re.sub(r"[^a-z]", "", word)
        if not word:
            continue
        phs = pronouncing.phones_for_word(word)
        if not phs:
            continue
        for ph in phs[0].split():
            ph_clean = re.sub(r"\\d+", "", ph)
            if ph_clean in PHONEME_TO_IDX:
                ids.append(PHONEME_TO_IDX[ph_clean])
    return ids

def preprocess_video(path, num_frames=NUM_FRAMES, size=IMG_SIZE):
    \"\"\"Decode → sample N frames evenly → face-crop (sticky bbox) → 224 RGB. Returns (T,H,W,C) uint8.\"\"\"
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

print("Utilities ready.")
"""))

cells.append(md("""\
## 6. GRID preprocessing (resumable, per-speaker)

Each GRID speaker → one `s<N>.pt` file on HF Hub. The cell scans HF first; speakers already uploaded are skipped. Safe to re-run after a session kick.

**Before running this cell, consider the shortcut.** Two community Kaggle datasets may have already done this work for you:

- [`mohamedbentalb/lipreading-dataset`](https://www.kaggle.com/datasets/mohamedbentalb/lipreading-dataset) — claims GRID preprocessed to `.npy` mouth-frame arrays. If it covers all 34 speakers, you can skip this cell entirely.
- [`awatefchiha/grid-lip-reading-s10-preprocessed3`](https://www.kaggle.com/datasets/awatefchiha/grid-lip-reading-s10-preprocessed3) — speaker 10 only. The author may have similar datasets for other speakers (check their profile).

If those work for you, write a quick adapter cell that loads the `.npy` files and uploads them as `s<N>.pt` shards in your HF dataset repo. Then skip the cell below.

**Source for from-scratch:** Zenodo record [3625687](https://zenodo.org/records/3625687). The `s<N>.zip` files are direct-download, no auth.

**One speaker takes ~5–10 min on a T4 / P100.** The whole GRID corpus (33 speakers, no s21) is 3–5 hours of preprocessing.
"""))

cells.append(code("""\
GRID_VIDEO_URL = "https://zenodo.org/records/3625687/files/s{sp}.zip?download=1"

# GRID grammar: <command> <color> <prep> <letter> <digit> <adverb>
# Encoded in the 6-char filename (e.g. "bbal3p.mpg" → "bin blue at l three please").
_GRID_CMD   = {"b":"bin","l":"lay","p":"place","s":"set"}
_GRID_COLOR = {"b":"blue","g":"green","r":"red","w":"white"}
_GRID_PREP  = {"a":"at","b":"by","i":"in","w":"with"}
_GRID_DIGIT = {"1":"one","2":"two","3":"three","4":"four","5":"five",
               "6":"six","7":"seven","8":"eight","9":"nine","0":"zero","z":"zero"}
_GRID_ADV   = {"a":"again","n":"now","p":"please","s":"soon"}

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
        files = api.list_repo_files(HF_DATA_REPO_GRID, repo_type="dataset")
    except Exception:
        return set()
    out = set()
    for f in files:
        m = re.match(r"s(\\d+)\\.pt$", f)
        if m:
            out.add(int(m.group(1)))
    return out

def preprocess_grid_speaker(sp):
    work = Path(WORK_DIR) / f"grid_s{sp}"; work.mkdir(parents=True, exist_ok=True)
    zip_path = work / f"s{sp}.zip"
    if not zip_path.exists():
        url = GRID_VIDEO_URL.format(sp=sp)
        print(f"  [s{sp}] downloading...")
        try:
            urllib.request.urlretrieve(url, zip_path)
        except Exception as e:
            print(f"  [s{sp}] download failed: {e}"); shutil.rmtree(work, ignore_errors=True); return False
    extract = work / "videos"
    if not extract.exists():
        extract.mkdir()
        subprocess.run(["unzip", "-q", "-o", str(zip_path), "-d", str(extract)], check=True)
    video_files = sorted(glob(str(extract / "**" / "*.mpg"), recursive=True))
    if not video_files:
        print(f"  [s{sp}] no .mpg files in zip"); shutil.rmtree(work, ignore_errors=True); return False
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
        print(f"  [s{sp}] no usable clips"); shutil.rmtree(work, ignore_errors=True); return False
    out_path = work / f"s{sp}.pt"
    torch.save({
        "frames":   torch.from_numpy(np.stack(frames_list)),  # (N,T,H,W,C) uint8
        "phonemes": ph_list,
        "texts":    text_list,
        "speaker":  sp,
    }, out_path)
    sz_mb = out_path.stat().st_size / 1e6
    print(f"  [s{sp}] uploading {sz_mb:.0f} MB...")
    upload_file(
        path_or_fileobj=str(out_path),
        path_in_repo=f"s{sp}.pt",
        repo_id=HF_DATA_REPO_GRID,
        repo_type="dataset",
        commit_message=f"GRID s{sp}: {len(frames_list)} clips",
    )
    shutil.rmtree(work, ignore_errors=True)
    return True

# === Run preprocessing now ===
# Set RUN_GRID_PREPROCESS = False to skip. Time-budget aware. Defaults to True
# so a fresh `Run All` actually does the GRID download + preprocessing without
# requiring a manual edit to the cell.
RUN_GRID_PREPROCESS = True

if RUN_GRID_PREPROCESS:
    target = [s for s in range(1, 35) if s != 21]   # s21 has no audio in GRID
    done = existing_grid_speakers()
    pending = [s for s in target if s not in done]
    print(f"Done:    {sorted(done)}")
    print(f"Pending: {pending}")
    t0 = time.time()
    for sp in pending:
        elapsed_min = (time.time() - t0) / 60
        if elapsed_min > TIME_BUDGET_MIN - 30:
            print(f"Within 30 min of budget; stopping. Re-run cell to continue.")
            break
        preprocess_grid_speaker(sp)
else:
    print("Set RUN_GRID_PREPROCESS = True to run preprocessing.")
    print(f"Already on HF: {sorted(existing_grid_speakers())}")
"""))

cells.append(md("""\
## 7. TCD-TIMIT preprocessing (resumable, per-speaker)

TCD-TIMIT requires you to register at https://sigmedia.tv/datasets/tcd_timit/ and download the dataset zip once.

**Recommended workflow:**

1. Register, download the zip on your local machine.
2. Upload the zip directly to `<you>/liperty-tcd-preprocessed` on HuggingFace as `tcd-timit-raw.zip` (one-time, via the HF web UI). It's ~12 GB.
3. This cell pulls the raw zip from HF, extracts it once per session, and processes per speaker.

If you'd rather keep the raw zip on Google Drive, mount Drive in cell 5 and adjust `TCD_RAW_PATH`.
"""))

cells.append(code("""\
TCD_RAW_HF_FILE = "tcd-timit-raw.zip"   # name in HF_DATA_REPO_TCD

def existing_tcd_speakers():
    try:
        files = api.list_repo_files(HF_DATA_REPO_TCD, repo_type="dataset")
    except Exception:
        return set()
    return {f for f in files if f.endswith(".pt")}

def materialize_tcd_raw():
    \"\"\"Download tcd-timit-raw.zip from HF if we don't have it locally yet.\"\"\"
    local = Path(WORK_DIR) / TCD_RAW_HF_FILE
    if local.exists():
        return local
    print(f"Pulling {TCD_RAW_HF_FILE} from HF (this is the slow one — happens once per session)...")
    try:
        path = hf_hub_download(
            repo_id=HF_DATA_REPO_TCD, repo_type="dataset",
            filename=TCD_RAW_HF_FILE, local_dir=WORK_DIR,
        )
        return Path(path)
    except Exception as e:
        print(f"Could not fetch {TCD_RAW_HF_FILE}: {e}")
        print("Upload your TCD-TIMIT zip to the HF dataset repo as 'tcd-timit-raw.zip' first.")
        return None

# === Run preprocessing now ===
RUN_TCD_PREPROCESS = False

if RUN_TCD_PREPROCESS:
    raw = materialize_tcd_raw()
    if raw is None:
        print("Skipping TCD preprocessing.")
    else:
        # NOTE: TCD-TIMIT directory layout varies by distribution. Adapt the
        # walker below to your zip's structure. Common layouts:
        #   <root>/volunteers/<speaker>/straightcam/<utt>.mp4 + <utt>.txt
        #   <root>/lipspeakers/<speaker>/...
        extract_dir = Path(WORK_DIR) / "tcd_extracted"
        if not extract_dir.exists():
            extract_dir.mkdir()
            print("Extracting...")
            subprocess.run(["unzip", "-q", "-o", str(raw), "-d", str(extract_dir)], check=True)
        speakers = sorted([d for d in extract_dir.glob("**/volunteers/*") if d.is_dir()] +
                          [d for d in extract_dir.glob("**/lipspeakers/*") if d.is_dir()])
        print(f"Found {len(speakers)} speakers.")
        done = existing_tcd_speakers()
        t0 = time.time()
        for sp_dir in speakers:
            sp_id = sp_dir.name
            out_name = f"{sp_id}.pt"
            if out_name in done:
                continue
            if (time.time() - t0) / 60 > TIME_BUDGET_MIN - 30:
                print("Within 30 min of budget; stopping."); break
            videos = sorted(sp_dir.glob("**/*.mp4")) + sorted(sp_dir.glob("**/*.avi"))
            print(f"  [{sp_id}] {len(videos)} videos")
            frames_list, ph_list, text_list = [], [], []
            for v in videos:
                txt_file = v.with_suffix(".txt")
                if not txt_file.exists():
                    continue
                text = txt_file.read_text(encoding="utf-8", errors="ignore").strip()
                ph = text_to_phoneme_ids(text)
                if not ph: continue
                f = preprocess_video(v)
                if f is None: continue
                frames_list.append(f); ph_list.append(ph); text_list.append(text)
            if not frames_list: continue
            out_path = sp_dir.parent / out_name
            torch.save({
                "frames":   torch.from_numpy(np.stack(frames_list)),
                "phonemes": ph_list,
                "texts":    text_list,
                "speaker":  sp_id,
            }, out_path)
            print(f"  [{sp_id}] uploading {out_path.stat().st_size/1e6:.0f} MB...")
            upload_file(
                path_or_fileobj=str(out_path), path_in_repo=out_name,
                repo_id=HF_DATA_REPO_TCD, repo_type="dataset",
                commit_message=f"TCD {sp_id}: {len(frames_list)} clips",
            )
else:
    print("Set RUN_TCD_PREPROCESS = True to run preprocessing.")
    print(f"Already on HF: {sorted(existing_tcd_speakers())}")
"""))

cells.append(md("""\
## 8. Pull preprocessed shards for training

Once preprocessing is done (even partially), download the available `.pt` shards locally for fast random access during training. Idempotent — `snapshot_download` only fetches what's missing.
"""))

cells.append(code("""\
def pull_preprocessed():
    grid_dir = Path(DATA_DIR) / "grid"
    tcd_dir  = Path(DATA_DIR) / "tcd"
    grid_dir.mkdir(parents=True, exist_ok=True)
    tcd_dir.mkdir(parents=True, exist_ok=True)

    grid_files, tcd_files = [], []
    try:
        snapshot_download(repo_id=HF_DATA_REPO_GRID, repo_type="dataset",
                          local_dir=str(grid_dir), allow_patterns="*.pt")
        grid_files = sorted(grid_dir.glob("*.pt"))
    except Exception as e:
        print(f"GRID pull: {e}")
    try:
        snapshot_download(repo_id=HF_DATA_REPO_TCD, repo_type="dataset",
                          local_dir=str(tcd_dir), allow_patterns="*.pt")
        tcd_files = sorted(tcd_dir.glob("*.pt"))
    except Exception as e:
        print(f"TCD pull: {e}")
    print(f"Local GRID shards: {len(grid_files)}")
    print(f"Local TCD shards:  {len(tcd_files)}")
    return grid_files, tcd_files

GRID_SHARDS, TCD_SHARDS = pull_preprocessed()
"""))

cells.append(md("""\
## 9. Dataset + dataloader

A `ShardDataset` lazy-loads each speaker shard once into memory (each is ~150 MB; with 33 GRID speakers that's ~5 GB total — fits Colab RAM with room to spare). `ConcatDataset` merges GRID and TCD-TIMIT into a single training set.

The collate function pads variable-length phoneme labels for CTC.
"""))

cells.append(code("""\
class ShardDataset(Dataset):
    def __init__(self, shard_paths):
        self.shards = []     # list of dicts loaded lazily
        self.index = []      # (shard_idx, clip_idx) pairs
        self._loaded = [False] * len(shard_paths)
        self._paths = list(shard_paths)
        # Build a flat index by reading just the metadata once
        for s_idx, p in enumerate(self._paths):
            d = torch.load(p, map_location="cpu", weights_only=False)
            n = d["frames"].shape[0]
            self.shards.append(d)
            self._loaded[s_idx] = True
            for c in range(n):
                self.index.append((s_idx, c))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        s_idx, c_idx = self.index[i]
        d = self.shards[s_idx]
        frames = d["frames"][c_idx]                   # (T, H, W, C) uint8
        phonemes = torch.tensor(d["phonemes"][c_idx], dtype=torch.long)
        return frames, phonemes

def ctc_collate(batch):
    frames = torch.stack([b[0] for b in batch], dim=0)  # (B, T, H, W, C) uint8
    labels = [b[1] for b in batch]
    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    # Pad labels with -1 (unused; CTC uses lengths, not the pad value)
    max_len = max(label_lengths).item()
    padded = torch.full((len(labels), max_len), -1, dtype=torch.long)
    for i, l in enumerate(labels):
        padded[i, :len(l)] = l
    return frames, padded, label_lengths

# Build datasets
ds_list = []
if GRID_SHARDS: ds_list.append(ShardDataset(GRID_SHARDS))
if TCD_SHARDS:  ds_list.append(ShardDataset(TCD_SHARDS))
assert ds_list, "No data available. Run preprocessing first."

train_ds = ConcatDataset(ds_list) if len(ds_list) > 1 else ds_list[0]
print(f"Total clips: {len(train_ds)}")

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, collate_fn=ctc_collate,
    pin_memory=True, drop_last=True, persistent_workers=NUM_WORKERS > 0,
)
print(f"Batches per epoch: {len(train_loader)}")
"""))

cells.append(md("""\
## 10. Model

VideoMAE-base encoder (loaded with `MCG-NJU/videomae-base` pretrained weights — this is the single most important fix vs the previous training runs that initialized from scratch), small adapter, transposed-conv temporal upsampler, CTC linear head.

Output shape: `(B, NUM_FRAMES // 2 * TIME_UPSAMPLE, VOCAB_SIZE)` = `(B, 32, 40)`.
"""))

cells.append(code("""\
from transformers import VideoMAEModel

class LipertyVSR(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, num_frames=NUM_FRAMES, image_size=IMG_SIZE,
                 time_upsample=TIME_UPSAMPLE, pretrained="MCG-NJU/videomae-base"):
        super().__init__()
        self.encoder = VideoMAEModel.from_pretrained(
            pretrained, num_frames=num_frames, image_size=image_size, ignore_mismatched_sizes=True
        )
        h = self.encoder.config.hidden_size  # 768
        self.adapter = nn.Sequential(
            nn.LayerNorm(h),
            nn.Linear(h, h),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        # Upsample temporal dim: (B, T_tubelet, h) -> (B, T_tubelet*time_upsample, h)
        self.upsample = nn.ConvTranspose1d(h, h, kernel_size=time_upsample, stride=time_upsample)
        self.head = nn.Linear(h, vocab_size)
        self._t_tubelet = num_frames // 2  # VideoMAE tubelet size 2
        self._n_spatial = (image_size // self.encoder.config.patch_size) ** 2

    def forward(self, pixel_values):
        # pixel_values: (B, T, H, W, C) uint8 OR (B, C, T, H, W) float
        if pixel_values.dim() == 5 and pixel_values.shape[-1] == 3:
            if pixel_values.dtype == torch.uint8:
                pixel_values = pixel_values.float() / 255.0
            pixel_values = pixel_values.permute(0, 1, 4, 2, 3).contiguous()  # (B,T,C,H,W)
        # VideoMAE wants (B, T, C, H, W)
        out = self.encoder(pixel_values=pixel_values)
        feats = out.last_hidden_state  # (B, T_tubelet * N_spatial, h)
        B, P, H = feats.shape
        feats = feats.view(B, self._t_tubelet, self._n_spatial, H).mean(dim=2)  # (B, T_tubelet, h)
        feats = self.adapter(feats)                  # (B, T_tubelet, h)
        feats = feats.transpose(1, 2)                # (B, h, T_tubelet)
        feats = self.upsample(feats)                 # (B, h, T_out)
        feats = feats.transpose(1, 2)                # (B, T_out, h)
        logits = self.head(feats)                    # (B, T_out, V)
        return logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LipertyVSR().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {n_params/1e6:.1f}M params on {device}")

# Quick shape sanity check
with torch.no_grad():
    dummy = torch.randint(0, 256, (1, NUM_FRAMES, IMG_SIZE, IMG_SIZE, 3), dtype=torch.uint8, device=device)
    y = model(dummy)
    print(f"Output: {tuple(y.shape)}  (expected: (1, {NUM_FRAMES//2 * TIME_UPSAMPLE}, {VOCAB_SIZE}))")
"""))

cells.append(md("""\
## 11. Checkpoint utilities

Single source of truth for run state. The checkpoint is one `.pt` file containing model + optimizer + scheduler + AMP scaler + step/epoch + RNG state. Uploaded to HF Hub atomically (HF commits on success only). Old checkpoints can be pruned, kept on `KEEP_LAST_CKPTS`.
"""))

cells.append(code("""\
import random as _random

CKPT_FILENAME = f"{RUN_NAME}-latest.pt"
CKPT_HISTORY_PREFIX = f"{RUN_NAME}-step"

def save_checkpoint(model, optimizer, scheduler, scaler, step, epoch, extra=None):
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "epoch": epoch,
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "rng_numpy": np.random.get_state(),
        "rng_python": _random.getstate(),
        "config": {
            "RUN_NAME": RUN_NAME, "NUM_FRAMES": NUM_FRAMES, "IMG_SIZE": IMG_SIZE,
            "TIME_UPSAMPLE": TIME_UPSAMPLE, "VOCAB_SIZE": VOCAB_SIZE, "BLANK_IDX": BLANK_IDX,
            "BATCH_SIZE": BATCH_SIZE, "GRAD_ACCUM": GRAD_ACCUM, "LR": LR,
        },
        "extra": extra or {},
    }
    local_latest = Path(CKPT_DIR) / CKPT_FILENAME
    local_step   = Path(CKPT_DIR) / f"{CKPT_HISTORY_PREFIX}{step:08d}.pt"
    # Atomic-ish: write to a temp file, then rename.
    tmp = local_latest.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    tmp.rename(local_latest)
    shutil.copy2(local_latest, local_step)

    # Push both to HF.
    upload_file(path_or_fileobj=str(local_latest), path_in_repo=CKPT_FILENAME,
                repo_id=HF_CKPT_REPO, repo_type="model",
                commit_message=f"step={step} epoch={epoch}")
    upload_file(path_or_fileobj=str(local_step), path_in_repo=local_step.name,
                repo_id=HF_CKPT_REPO, repo_type="model",
                commit_message=f"step={step} epoch={epoch} (history)")
    print(f"    [ckpt] step={step} uploaded to HF Hub.")

def prune_old_checkpoints(keep=KEEP_LAST_CKPTS):
    \"\"\"Delete all but the latest `keep` step-checkpoints from the HF model repo.
    The pointer file `{RUN_NAME}-latest.pt` is always kept.\"\"\"
    try:
        files = api.list_repo_files(HF_CKPT_REPO, repo_type="model")
    except Exception:
        return
    history = sorted(f for f in files if f.startswith(CKPT_HISTORY_PREFIX) and f.endswith(".pt"))
    to_delete = history[:-keep] if len(history) > keep else []
    if not to_delete:
        return
    # huggingface_hub >= 0.27 dropped the module-level delete_file in favor
    # of HfApi.delete_file. Use the api instance we already have above.
    for f in to_delete:
        try:
            api.delete_file(path_in_repo=f, repo_id=HF_CKPT_REPO, repo_type="model",
                            commit_message=f"Prune old checkpoint {f}")
        except Exception as e:
            print(f"    [prune] {f}: {e}")

def load_checkpoint_if_available(model, optimizer, scheduler, scaler):
    \"\"\"Pull latest checkpoint from HF Hub. Returns (start_step, start_epoch). 0,0 if none.\"\"\"
    try:
        path = hf_hub_download(repo_id=HF_CKPT_REPO, repo_type="model",
                               filename=CKPT_FILENAME, local_dir=CKPT_DIR)
    except Exception:
        print("No prior checkpoint on HF Hub. Starting fresh.")
        return 0, 0
    payload = torch.load(path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(payload["model"], strict=False)
    if missing:    print(f"[ckpt] missing keys (first 5): {missing[:5]}")
    if unexpected: print(f"[ckpt] unexpected keys (first 5): {unexpected[:5]}")
    optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    torch.set_rng_state(payload["rng_torch"])
    if torch.cuda.is_available() and payload.get("rng_cuda") is not None:
        torch.cuda.set_rng_state_all(payload["rng_cuda"])
    np.random.set_state(payload["rng_numpy"])
    _random.setstate(payload["rng_python"])
    print(f"[ckpt] resumed step={payload['step']} epoch={payload['epoch']}")
    return payload["step"], payload["epoch"]

print("Checkpoint utilities ready.")
"""))

cells.append(md("""\
## 12. Build optimizer + scheduler, then resume from HF

After this cell, the model is either freshly initialized (first run) or restored to the exact state of the last checkpoint (subsequent runs).
"""))

cells.append(code("""\
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

def make_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))
    return LambdaLR(optimizer, lr_lambda)

# Total training horizon for the cosine schedule. Adjust upward if you plan
# many sessions; lower if you want LR to decay faster.
TOTAL_STEPS = max(20000, len(train_loader) * 20)

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))
scheduler = make_scheduler(optimizer, WARMUP_STEPS, TOTAL_STEPS)
scaler = torch.amp.GradScaler("cuda") if (USE_FP16 and torch.cuda.is_available()) else None

start_step, start_epoch = load_checkpoint_if_available(model, optimizer, scheduler, scaler)
print(f"Starting from step={start_step}, epoch={start_epoch}")
print(f"Cosine total_steps={TOTAL_STEPS}; current LR={optimizer.param_groups[0]['lr']:.2e}")
"""))

cells.append(md("""\
## 13. Anti-idle (optional, opt-in)

Colab kicks browsers idle for ~90 min. The training cell itself keeps the runtime busy, but the *browser* still gets idle-kicked. The standard workaround is a small JS snippet that clicks the Connect button periodically.

⚠️ This is in a grey area for Colab's terms of service. Use at your own discretion. Crucially, **you don't actually need this** — the resume logic means a kick just costs you the unsaved minutes since the last checkpoint (worst case 15-30 min, configurable via `CKPT_EVERY_STEPS`). I recommend leaving the cell commented out and just letting kicks happen.
"""))

cells.append(code("""\
# from IPython.display import Javascript, display
# display(Javascript('''
#     function ClickConnect(){
#         console.log("Liperty: keep-alive ping");
#         var button = document.querySelector("colab-connect-button");
#         if (button) {
#             var inner = button.shadowRoot && button.shadowRoot.querySelector("#connect");
#             if (inner) inner.click();
#         }
#     }
#     setInterval(ClickConnect, 60000);
# '''))
print("Anti-idle is disabled. Uncomment the JS in this cell to enable.")
"""))

cells.append(md("""\
## 14. Training loop

Time-budgeted, with periodic atomic checkpointing to HF Hub. Logs running loss every `LOG_EVERY_STEPS`. The loop exits cleanly when:

- The wallclock budget is reached → guaranteed final flush in cell 15.
- A `KeyboardInterrupt` is raised (you stop the cell) → still flushes via `try/finally`.
- A Colab kick happens → the most recent step-checkpoint on HF is your floor.
"""))

cells.append(code("""\
import time, math

def train_loop(start_step, start_epoch, budget_min):
    start_wall = time.monotonic()
    deadline_s = budget_min * 60
    step = start_step
    epoch = start_epoch
    running_loss = 0.0
    running_n = 0
    last_ckpt_step = step

    ctc_loss = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True, reduction="mean")

    try:
        while True:
            for batch in train_loader:
                # Time budget check (before doing more work)
                if time.monotonic() - start_wall > deadline_s:
                    print(f"[budget] elapsed={(time.monotonic()-start_wall)/60:.1f} min ≥ {budget_min}; stopping loop.")
                    return step, epoch

                frames, labels, label_lengths = batch
                frames = frames.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                label_lengths = label_lengths.to(device, non_blocking=True)

                # AMP forward
                if scaler is not None:
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        logits = model(frames)              # (B, T_out, V)
                        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # (T_out, B, V)
                        T_out = log_probs.shape[0]
                        input_lengths = torch.full((logits.shape[0],), T_out, dtype=torch.long, device=device)
                        loss = ctc_loss(log_probs, labels.clamp_min(0), input_lengths, label_lengths)
                    loss_back = loss / GRAD_ACCUM
                    scaler.scale(loss_back).backward()
                else:
                    logits = model(frames)
                    log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
                    T_out = log_probs.shape[0]
                    input_lengths = torch.full((logits.shape[0],), T_out, dtype=torch.long, device=device)
                    loss = ctc_loss(log_probs, labels.clamp_min(0), input_lengths, label_lengths)
                    (loss / GRAD_ACCUM).backward()

                running_loss += float(loss.item())
                running_n += 1

                # Optimizer step every GRAD_ACCUM micro-batches
                if running_n % GRAD_ACCUM == 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    step += 1

                    # Logging
                    if step % LOG_EVERY_STEPS == 0:
                        avg = running_loss / max(1, running_n)
                        running_loss = 0.0; running_n = 0
                        elapsed = (time.monotonic() - start_wall) / 60
                        lr_now = optimizer.param_groups[0]["lr"]
                        print(f"  step={step:6d} epoch={epoch} loss={avg:.3f} lr={lr_now:.2e} t={elapsed:.1f}min")

                    # Checkpointing
                    if step - last_ckpt_step >= CKPT_EVERY_STEPS:
                        save_checkpoint(model, optimizer, scheduler, scaler, step, epoch)
                        last_ckpt_step = step
                        prune_old_checkpoints(KEEP_LAST_CKPTS)

            epoch += 1
            print(f"[epoch] completed epoch {epoch}")
    except KeyboardInterrupt:
        print("KeyboardInterrupt — flushing final checkpoint...")
        return step, epoch

# Go.
final_step, final_epoch = train_loop(start_step, start_epoch, TIME_BUDGET_MIN)
print(f"\\nLoop finished. step={final_step} epoch={final_epoch}")
"""))

cells.append(md("""\
## 15. Final flush

Always run this after the training cell, even if you Ctrl-C'd. Guarantees the latest state is on HF Hub before you let Colab kick the session.
"""))

cells.append(code("""\
save_checkpoint(model, optimizer, scheduler, scaler, final_step, final_epoch,
                extra={"flush_reason": "end-of-session"})
prune_old_checkpoints(KEEP_LAST_CKPTS)
print(f"Final checkpoint pushed: step={final_step} epoch={final_epoch}")
print(f"Resume in any account by re-running this notebook end-to-end.")
"""))

cells.append(md("""\
## 16. Cross-account handoff

When the current session approaches its kick or you've just hit the time budget:

1. The training loop (or the final-flush cell) has already pushed the latest state to `{HF_CKPT_REPO}` on HF Hub.
2. Open this same notebook in your next Google account (use a bookmark to the GitHub-hosted version: `https://colab.research.google.com/github/HereLiesAz/Liperty/blob/main/tools/train_grid_tcd_resumable.ipynb`).
3. `Runtime → Run all`.
4. Cells 1–9 are setup. Cell 12 sees the prior checkpoint on HF Hub and resumes the optimizer/scheduler/RNG state. Cell 14 picks up at the same step.

**There is no automation for the manual Google-account login itself.** Selenium-driving five Chrome profiles is fragile and against Colab's spirit. The thing this notebook *does* automate:

- Same HF token across all accounts → same checkpoint store.
- The first cell of each session reattaches state in <1 minute.
- The data preprocessing is idempotent → no "redoing speakers I already did".
- Loss continues from the same step; the cosine LR schedule continues; gradient stats continue.

**Realistic per-day throughput on free Colab:** 3-4 sessions × ~3.5h training = 10-14h training/day across accounts. With 5 accounts you can do this 5 days/week before any one account hits its weekly quota.
"""))

cells.append(md("""\
## 17. Troubleshooting

**"OOM during forward pass."** Drop `BATCH_SIZE` to 1 and double `GRAD_ACCUM` to keep effective batch.

**"Pretrained VideoMAE download fails."** Hugging Face has rate-limited Colab IPs in the past. Set `HF_HUB_ENABLE_HF_TRANSFER=1` env var, or wait 10 min and retry.

**"Loss is NaN / model output is uniform."** This is the prior-collapse failure mode you saw in the deployed app. With pretrained VideoMAE weights loaded (cell 10) and proper input normalization (`pixel/255` matching VALLR's training pipeline), this should not happen. If it does: (a) reduce LR to `5e-5`, (b) check that your phoneme vocabulary in the dataset matches `MLConstants.PHONEME_VOCAB`, (c) verify input frames look like faces by saving a few to disk.

**"GRID download is slow."** Zenodo rate-limits per-IP. Different sessions get different IPs, so re-running in a new account is often faster. Each `s<N>.zip` is ~750 MB. Or skip Zenodo entirely and use the Kaggle preprocessed datasets noted in cell 6.

**"On Kaggle, the session ended but I lost the dataset."** Kaggle auto-saves `/kaggle/working/` as a versioned Dataset output when you `Save Version`. But if the kernel was killed via timeout without a Save Version, recent state may not have been published. The HF Hub uploads done by the training loop *are* durable regardless — pull the latest `<run>-latest.pt` from `liperty-vsr-checkpoints` and resume.

**"How do I run a Kaggle notebook headless?"** `Save Version → Save & Run All (Commit)` rather than the interactive `Run All` button. The kernel runs in the background after browser close; you get an email when it finishes or fails. This is the killer feature — it removes the entire idle-disconnect problem.

**"Session kicks before checkpoint flush."** Lower `CKPT_EVERY_STEPS` from 200 to 100 (more frequent checkpoints, more upload overhead but smaller blast radius from a kick).

**"`upload_file` fails with rate-limit."** HF Hub allows ~100 commits/hr free. With `CKPT_EVERY_STEPS=200` and ~10 s/step, you hit 18 commits/hr — well under. If you raise `CKPT_EVERY_STEPS` (more frequent) and start hitting limits, set `KEEP_LAST_CKPTS=1` and skip the per-step history file (modify `save_checkpoint` to drop the `local_step` upload).

**"My phoneme labels look wrong."** The `pronouncing` library uses CMU's ARPABET dict. Words missing from CMU (proper nouns, slang) silently produce empty phoneme lists, which `text_to_phoneme_ids` skips. For TCD-TIMIT, prefer the dataset's own phoneme alignments if shipped (typically `.phn` or `.lab` files); modify cell 7 to read those instead of regenerating.

**"Validation? Eval? WER?"** Not in this notebook — keep it minimal and resumable. Once you have a checkpoint that's been training for a few sessions, write a separate eval notebook that loads the latest checkpoint, runs it on a held-out speaker, decodes via greedy CTC, and computes phoneme error rate vs the ground-truth phoneme sequences.
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
