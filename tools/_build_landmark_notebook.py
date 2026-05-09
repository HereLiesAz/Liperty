"""Generate tools/train_landmark_lrs3_resumable.ipynb.

Sibling to _build_notebook.py. Trains a small Transformer over per-frame
dlib 68-point landmarks (the e1lephant/lrs3-landmark Kaggle dataset format)
to produce 40-class ARPABET phoneme CTC output — the same vocab as the
pixel-based pipeline, so the deployed Android decoder needs no changes.

Run with:  python tools/_build_landmark_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "train_landmark_lrs3_resumable.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.splitlines(keepends=True)}


cells: list[dict] = []

cells.append(md("""\
# Liperty Landmark-VSR — Resumable Training (LRS3 + GRID + TCD)

Sibling to `train_grid_tcd_resumable.ipynb`. Trains a small Transformer over **dlib 68-point face landmarks** (not pixel video) to produce 40-class ARPABET phoneme CTC output.

**Why a landmark-only model:**

- Compatible with `e1lephant/lrs3-landmark` (MIT-licensed, ~152k LRS3 utterances as `.pkl`s of per-frame 68×2 float32 landmarks). Closest path to LRS3-tier data without the academic-license wait.
- ~10M params vs ~100M for the pixel VideoMAE path. Trains in hours not days.
- <50ms inference on a phone — slots into the deployed Android pipeline as a new `LipCoordNetEngine : ModelEngine`. The dual-input scaffolding for landmarks already exists in `VSRInference.kt`.
- Same 40-phoneme ARPABET vocab as the pixel model — no changes to `BeamSearchDecoder`/`HomopheneCorrector`/`LanguageModel`.

**Trainable today, scalable later.** Three data sources, mix and match:

1. **`grid_shards`** (default) — landmarks extracted from `liperty-grid-preprocessed` by `preprocess_landmarks_resumable.ipynb`. ~33k clips, multi-speaker GRID grammar. Transcripts from filenames. **Runnable today.**
2. **`tcd_shards`** — same, from `liperty-tcd-preprocessed`. ~6k clips, open vocabulary, 60+ speakers. **Runnable today.**
3. **`lrs3_landmark`** — `e1lephant/lrs3-landmark` on Kaggle. ~150k clips, but landmarks-only — transcripts must come from elsewhere (Oxford VGG academic access, YouTube auto-captions, or other). When transcripts arrive, add this to `DATA_SOURCES`.

The notebook concatenates whatever's enabled. Same model architecture and training loop regardless of source.
"""))

cells.append(md("""\
## How to use this

Identical pattern to the pixel notebook. The only setup wrinkle is the LRS3 landmark dataset — it's 8+ GB on Kaggle, so attach it as an input to your Kaggle notebook rather than re-downloading per session.

**Kaggle setup (recommended):**

1. Open this notebook on Kaggle.
2. Add input dataset: `e1lephant/lrs3-landmark` (right side panel → "+ Add Input").
3. Add input dataset: `mattymchen/lrs3-test` if available, or this notebook will pull it from HF Hub instead.
4. Set runtime: GPU P100, Internet ON.
5. Add HF token to Kaggle Secrets as `HF_TOKEN`.
6. `Save Version → Save & Run All (Commit)` for headless execution.

**Resume across accounts:** same HF token in Kaggle Secrets across all your Kaggle accounts → checkpoints continue from the same step.
"""))

cells.append(md("""\
## 1. Environment check
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

!nvidia-smi 2>&1 | head -20
print()
print(f"Python:   {sys.version.split()[0]}")
print(f"PyTorch:  {torch.__version__}")
print(f"CUDA:     {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:      {torch.cuda.get_device_name(0)}")
    print(f"VRAM:     {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
"""))

cells.append(md("""\
## 2. Install dependencies

Lighter than the pixel notebook — no MediaPipe, no decord, no OpenCV. Just torch, HuggingFace, pronouncing for ARPABET phoneme conversion, and pyarrow to read the test-split parquet.
"""))

cells.append(code("""\
%%capture
!pip install -q \\
    "huggingface_hub>=0.27,<1.0" \\
    "datasets>=3.0,<4.0" \\
    "pronouncing>=0.2.0" \\
    "pyarrow>=15.0" \\
    "youtube-transcript-api>=0.6.2" \\
    "jiwer>=3.0"
print("Deps installed.")
"""))

cells.append(md("""\
## 3. HuggingFace auth
"""))

cells.append(code("""\
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
"""))

cells.append(code("""\
# === EDIT ME ===
HF_USER = "CHANGE-ME"   # Your HuggingFace username
# ===============
assert HF_USER != "CHANGE-ME", "Set HF_USER to your HuggingFace username."

# Separate from the pixel-model checkpoint repo so the two architectures don't
# step on each other.
HF_CKPT_REPO = f"{HF_USER}/liperty-landmark-vsr-checkpoints"   # private model repo

RUN_NAME = "landmark-transformer-v1"

# Time budget. Kaggle GPU sessions cap at ~9h, Colab free at ~4h.
TIME_BUDGET_MIN = 480 if IS_KAGGLE else 200

# Data sources, comma-separated. The notebook concatenates whatever is enabled.
#   "grid_shards"  — landmarks extracted from liperty-grid-preprocessed
#                    via preprocess_landmarks_resumable.ipynb. Trainable today.
#   "tcd_shards"   — same, from TCD-TIMIT.
#   "lrs3_landmark" — e1lephant/lrs3-landmark on Kaggle. Requires transcripts
#                     (set LRS3_TRANSCRIPT_SOURCE below). Adds ~150k clips.
DATA_SOURCES = "grid_shards,tcd_shards"

# HF dataset repos for shard-format landmark sources (output of
# preprocess_landmarks_resumable.ipynb).
HF_LM_REPO_GRID = f"{HF_USER}/liperty-grid-landmarks"
HF_LM_REPO_TCD  = f"{HF_USER}/liperty-tcd-landmarks"

# Where the e1lephant/lrs3-landmark pkls live in the session.
# On Kaggle, attach the dataset as input and this is the path:
LRS3_LANDMARK_ROOT = "/kaggle/input/lrs3-landmark"

# For lrs3_landmark only — choose how to obtain transcripts:
#   "lrs3_official"    — local path to LRS3 .txt files (set LRS3_TXT_ROOT)
#   "youtube_captions" — youtube_transcript_api fallback (not recommended)
LRS3_TRANSCRIPT_SOURCE = "lrs3_official"
LRS3_TXT_ROOT          = "/kaggle/input/lrs3-text"

# Data quality
MAX_NONE_RATIO = 0.20   # reject clips where >20% of frames have no detected face
MIN_FRAMES     = 25
MAX_FRAMES     = 250    # truncate very long clips at training time

# Model
D_MODEL    = 192
N_HEADS    = 4
N_LAYERS   = 4
DROPOUT    = 0.1
NUM_LANDMARKS = 68
INPUT_DIM  = NUM_LANDMARKS * 2   # x,y per landmark
TIME_UPSAMPLE = 2   # CTC needs T_out >= label_length; up-rate the encoder output

# Optimization
BATCH_SIZE   = 16     # landmark sequences are tiny; this fits easily
GRAD_ACCUM   = 1
LR           = 3e-4
WEIGHT_DECAY = 1e-2
WARMUP_STEPS = 1000
NUM_WORKERS  = 2
USE_FP16     = True

# Checkpointing
CKPT_EVERY_STEPS = 500
LOG_EVERY_STEPS  = 50
KEEP_LAST_CKPTS  = 3

# Local working dirs
if IS_KAGGLE:
    WORK_DIR = "/kaggle/working/work"
    CKPT_DIR = "/kaggle/working/ckpt"
else:
    WORK_DIR = "/content/work"
    CKPT_DIR = "/content/ckpt"
for d in (WORK_DIR, CKPT_DIR):
    os.makedirs(d, exist_ok=True)

# Same vocab as the Android decoder (MLConstants.PHONEME_VOCAB)
PHONEME_VOCAB = ["_", "AA","AE","AH","AO","AW","AY","B","CH","D","DH","EH","ER","EY",
                 "F","G","HH","IH","IY","JH","K","L","M","N","NG","OW","OY","P",
                 "R","S","SH","T","TH","UH","UW","V","W","Y","Z","ZH"]
VOCAB_SIZE = len(PHONEME_VOCAB)
BLANK_IDX = 0
PHONEME_TO_IDX = {p: i for i, p in enumerate(PHONEME_VOCAB)}

print(f"Run:              {RUN_NAME}")
print(f"Budget:           {TIME_BUDGET_MIN} min")
print(f"Data sources:     {DATA_SOURCES}")
print(f"Effective bs:     {BATCH_SIZE * GRAD_ACCUM}")
print(f"Vocab size:       {VOCAB_SIZE}")
"""))

cells.append(md("""\
## 5. Optional: pull `mattymchen/lrs3-test` for held-out evaluation

677 MB Parquet with 1,321 LRS3 test-split rows including transcripts. Not used during training (its rows aren't keyed to e1lephant pkls), but useful as a held-out evaluation set if you later add an eval cell. Skip the cell if you don't need it.
"""))

cells.append(code("""\
from huggingface_hub import snapshot_download
from pathlib import Path
import re

MATTY_DIR = Path(WORK_DIR) / "mattymchen-lrs3-test"
if not (MATTY_DIR / "data").exists():
    snapshot_download(
        repo_id="mattymchen/lrs3-test",
        repo_type="dataset",
        local_dir=str(MATTY_DIR),
        allow_patterns="data/*.parquet",
    )
parquets = sorted(MATTY_DIR.glob("**/*.parquet"))
print(f"mattymchen test parquets: {len(parquets)}")
for p in parquets[:3]:
    print(f"  {p.name}: {p.stat().st_size / 1e6:.1f} MB")
"""))

cells.append(md("""\
## 6. Dataset class

`LandmarkDataset` walks a tree of `.pkl` files matching the e1lephant/lrs3-landmark layout (`<split>/<youtube_id>/<utt>.pkl`), pairs each clip with a transcript via a pluggable resolver, and yields:

- `landmarks: Tensor[T, 136]` — per-frame x,y of all 68 dlib points, normalized per-clip
- `phonemes: Tensor[L]` — ARPABET indices

Quality filtering: clips with too many `None` frames or out-of-range length are rejected at index time so training never sees them. `None` frames inside accepted clips are linearly interpolated.
"""))

cells.append(code("""\
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import pyarrow.parquet as pq
import pronouncing

# ---------- Per-clip normalization ----------
NOSE_TIP_IDX = 30
LEFT_EYE_OUTER, RIGHT_EYE_OUTER = 36, 45

def normalize_landmarks(arr: np.ndarray) -> np.ndarray:
    \"\"\"arr: (T, 68, 2) float32 with possible nan rows. Returns same shape, normalized
    so nose tip = origin and inter-ocular distance = 1, averaged over valid frames.\"\"\"
    valid = ~np.isnan(arr[:, 0, 0])
    if not valid.any():
        return arr
    nose = np.nanmean(arr[valid, NOSE_TIP_IDX, :], axis=0)
    eye_dist = np.linalg.norm(
        np.nanmean(arr[valid, LEFT_EYE_OUTER, :], axis=0)
        - np.nanmean(arr[valid, RIGHT_EYE_OUTER, :], axis=0)
    )
    eye_dist = max(eye_dist, 1e-3)
    out = (arr - nose[None, None, :]) / eye_dist
    return out

def load_pkl_landmarks(path):
    \"\"\"Return (T, 68, 2) float32 with nan for missing frames, plus None-ratio.\"\"\"
    with open(path, "rb") as f:
        seq = pickle.load(f)
    T = len(seq)
    arr = np.full((T, NUM_LANDMARKS, 2), np.nan, dtype=np.float32)
    none_count = 0
    for t, frame in enumerate(seq):
        if frame is None:
            none_count += 1
            continue
        arr[t] = frame
    none_ratio = none_count / max(T, 1)
    # Linear interpolate over short None gaps
    for j in range(NUM_LANDMARKS):
        for k in range(2):
            col = arr[:, j, k]
            mask = np.isnan(col)
            if mask.all() or not mask.any():
                continue
            xs = np.where(~mask)[0]
            ys = col[~mask]
            arr[mask, j, k] = np.interp(np.where(mask)[0], xs, ys)
    return arr, none_ratio

# ---------- Phoneme conversion ----------
def text_to_phoneme_ids(text: str) -> list[int]:
    ids = []
    for word in text.lower().split():
        word = re.sub(r"[^a-z']", "", word)
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

# ---------- Transcript resolvers ----------
class TranscriptResolver:
    def get(self, video_id: str, utt_id: str) -> str | None:
        raise NotImplementedError

class MattyMchenTestResolver(TranscriptResolver):
    \"\"\"Reads the HF parquet and indexes by row position. The mattymchen dataset
    doesn't preserve LRS3 video_id/utt_id, so this resolver is for SMOKE TESTING:
    we ignore the e1lephant pkls and synthesize an in-memory dataset directly
    from the parquet — landmarks computed on-the-fly via dlib would be needed,
    but for the smoke test we use the parquet's video frames and skip the pkl
    landmarks. See the SmokeTestDataset class below.\"\"\"
    def __init__(self, parquet_paths):
        rows = []
        for p in parquet_paths:
            t = pq.read_table(p)
            for row in zip(t.column("idx").to_pylist(), t.column("label").to_pylist()):
                rows.append({"idx": row[0], "label": row[1]})
        self.rows = rows
    def get(self, *_):
        return None  # not keyed by id; use SmokeTestDataset for smoke testing
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        return self.rows[i]

class LRS3OfficialResolver(TranscriptResolver):
    \"\"\"Reads <root>/<split>/<video_id>/<utt_id>.txt files in LRS3 layout. The
    .txt format starts with 'Text:  <transcript>' on the first line; we grab that.\"\"\"
    def __init__(self, root: str):
        self.root = Path(root)
    def get(self, split: str, video_id: str, utt_id: str) -> str | None:
        p = self.root / split / video_id / f"{utt_id}.txt"
        if not p.exists():
            return None
        try:
            line = p.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
        except Exception:
            return None
        if line.lower().startswith("text:"):
            line = line.split(":", 1)[1]
        return line.strip()

class YouTubeCaptionResolver(TranscriptResolver):
    \"\"\"Last-resort fallback. The original LRS3 utterance metadata (frame range
    within the YouTube video) is not preserved in e1lephant's pkls, so this
    can only return the WHOLE talk's auto-caption — useless for per-utterance
    CTC training. Included for completeness; do not enable for real training.\"\"\"
    def get(self, *_):
        return None

# ---------- Helper: convert a List[Optional[ndarray(68,2)]] to (T, 68, 2) with nan ----------
def list_to_landmark_array(landmarks_list):
    T = len(landmarks_list)
    arr = np.full((T, NUM_LANDMARKS, 2), np.nan, dtype=np.float32)
    none_count = 0
    for t, lm in enumerate(landmarks_list):
        if lm is None:
            none_count += 1
        else:
            arr[t] = lm
    none_ratio = none_count / max(T, 1)
    # Linear interpolate over None gaps
    for j in range(NUM_LANDMARKS):
        for k in range(2):
            col = arr[:, j, k]
            mask = np.isnan(col)
            if mask.all() or not mask.any():
                continue
            xs = np.where(~mask)[0]
            ys = col[~mask]
            arr[mask, j, k] = np.interp(np.where(mask)[0], xs, ys)
    return arr, none_ratio

# ---------- Shard-format dataset (GRID, TCD landmarks) ----------
class ShardLandmarkDataset(Dataset):
    \"\"\"Reads .pt shards in the format produced by preprocess_landmarks_resumable.ipynb:
        {'landmarks': [List[Optional[ndarray(68,2)]] per clip],
         'phonemes':  [list[int] per clip],
         'texts':     [str per clip],
         'speaker':   <id>}
    Each shard is loaded fully into memory once at construction time.
    \"\"\"
    def __init__(self, shard_paths):
        self.entries = []   # (landmarks_list, phonemes_tensor)
        for p in shard_paths:
            d = torch.load(p, map_location="cpu", weights_only=False)
            for lms, ph in zip(d["landmarks"], d["phonemes"]):
                if not ph:
                    continue
                self.entries.append((lms, torch.tensor(ph, dtype=torch.long)))
        print(f"  ShardLandmarkDataset: {len(self.entries)} clips across {len(shard_paths)} shards")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        lms, phonemes = self.entries[i]
        arr, none_ratio = list_to_landmark_array(lms)
        T = arr.shape[0]
        if none_ratio > MAX_NONE_RATIO or T < MIN_FRAMES:
            return None
        if T > MAX_FRAMES:
            arr = arr[:MAX_FRAMES]
        arr = normalize_landmarks(arr)
        landmarks = torch.from_numpy(arr.reshape(arr.shape[0], -1).astype(np.float32))
        if phonemes.numel() == 0:
            return None
        return landmarks, phonemes

# ---------- e1lephant directory-format dataset (LRS3) ----------
class LandmarkClipDataset(Dataset):
    def __init__(self, root: str, splits: list[str], resolver: TranscriptResolver,
                 max_none_ratio=MAX_NONE_RATIO, min_frames=MIN_FRAMES, max_frames=MAX_FRAMES):
        self.root = Path(root)
        self.resolver = resolver
        self.entries = []   # (split, video_id, utt_id, pkl_path, transcript)
        rejected_quality = rejected_text = 0
        for split in splits:
            split_dir = self.root / split
            if not split_dir.exists():
                continue
            for vid_dir in sorted(split_dir.iterdir()):
                if not vid_dir.is_dir():
                    continue
                for pkl in sorted(vid_dir.glob("*.pkl")):
                    utt_id = pkl.stem
                    text = resolver.get(split, vid_dir.name, utt_id)
                    if not text:
                        rejected_text += 1
                        continue
                    self.entries.append((split, vid_dir.name, utt_id, str(pkl), text))
        print(f"Found {len(self.entries)} clips with transcripts. "
              f"Rejected (no transcript): {rejected_text}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        split, vid, utt, path, text = self.entries[i]
        arr, none_ratio = load_pkl_landmarks(path)
        T = arr.shape[0]
        if none_ratio > MAX_NONE_RATIO or T < MIN_FRAMES:
            # Caller-driven retry: we don't filter at __init__ because
            # opening every pkl up front would be slow. Return None and let
            # collate skip.
            return None
        if T > MAX_FRAMES:
            arr = arr[:MAX_FRAMES]
        arr = normalize_landmarks(arr)
        landmarks = torch.from_numpy(arr.reshape(arr.shape[0], -1).astype(np.float32))
        phonemes = torch.tensor(text_to_phoneme_ids(text), dtype=torch.long)
        if phonemes.numel() == 0:
            return None
        return landmarks, phonemes

def landmark_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    lengths = torch.tensor([x[0].shape[0] for x in batch], dtype=torch.long)
    Tmax = lengths.max().item()
    L = torch.zeros(len(batch), Tmax, INPUT_DIM, dtype=torch.float32)
    for i, (l, _) in enumerate(batch):
        L[i, :l.shape[0]] = l
    label_lengths = torch.tensor([x[1].shape[0] for x in batch], dtype=torch.long)
    Lmax = label_lengths.max().item()
    Y = torch.full((len(batch), Lmax), -1, dtype=torch.long)
    for i, (_, p) in enumerate(batch):
        Y[i, :p.shape[0]] = p
    return L, lengths, Y, label_lengths

print("Dataset class defined.")
"""))

cells.append(md("""\
## 7. Build the training dataset

Concatenates whatever sources are listed in `DATA_SOURCES`. With the default `"grid_shards,tcd_shards"` you get a runnable dataset today — landmarks extracted by `preprocess_landmarks_resumable.ipynb` from your existing pixel preprocessed shards. Add `lrs3_landmark` once LRS3 transcripts are available.
"""))

cells.append(code("""\
from torch.utils.data import DataLoader, ConcatDataset
from huggingface_hub import snapshot_download

sources = [s.strip() for s in DATA_SOURCES.split(",") if s.strip()]
ds_list = []

# --- Shard-format sources (output of preprocess_landmarks_resumable.ipynb) ---
def _pull_shards(repo_id, subdir):
    local = Path(WORK_DIR) / subdir
    local.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(repo_id=repo_id, repo_type="dataset",
                          local_dir=str(local), allow_patterns="*.pt")
    except Exception as e:
        print(f"snapshot_download({repo_id}): {e}")
        return []
    return sorted(local.glob("*.pt"))

if "grid_shards" in sources:
    print("Pulling GRID landmark shards...")
    shards = _pull_shards(HF_LM_REPO_GRID, "grid-landmarks")
    print(f"  GRID shards: {len(shards)}")
    if shards:
        ds_list.append(ShardLandmarkDataset(shards))

if "tcd_shards" in sources:
    print("Pulling TCD landmark shards...")
    shards = _pull_shards(HF_LM_REPO_TCD, "tcd-landmarks")
    print(f"  TCD shards: {len(shards)}")
    if shards:
        ds_list.append(ShardLandmarkDataset(shards))

# --- e1lephant directory-format source (LRS3) ---
if "lrs3_landmark" in sources:
    if LRS3_TRANSCRIPT_SOURCE == "lrs3_official":
        resolver = LRS3OfficialResolver(LRS3_TXT_ROOT)
    elif LRS3_TRANSCRIPT_SOURCE == "youtube_captions":
        resolver = YouTubeCaptionResolver()
    else:
        raise ValueError(f"unknown LRS3_TRANSCRIPT_SOURCE: {LRS3_TRANSCRIPT_SOURCE}")
    print(f"Walking LRS3 landmarks at {LRS3_LANDMARK_ROOT} with resolver={LRS3_TRANSCRIPT_SOURCE}...")
    lrs3_ds = LandmarkClipDataset(
        root=LRS3_LANDMARK_ROOT,
        splits=["pretrain", "trainval"],
        resolver=resolver,
    )
    if len(lrs3_ds) > 0:
        ds_list.append(lrs3_ds)

if not ds_list:
    print("No data sources produced any clips. Check DATA_SOURCES and that the shards/transcripts exist.")
    train_ds = None
    train_loader = []
else:
    train_ds = ConcatDataset(ds_list) if len(ds_list) > 1 else ds_list[0]
    print(f"\\nTotal clips: {len(train_ds)}  ({len(ds_list)} source(s))")
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, collate_fn=landmark_collate,
        pin_memory=True, drop_last=True, persistent_workers=NUM_WORKERS > 0,
    )
    print(f"Batches per epoch: {len(train_loader)}")
"""))

cells.append(md("""\
## 8. Model — small Transformer over landmark sequences

`(B, T, 136)` → linear embed → 4-layer Transformer encoder → linear → CTC head over 40 phonemes. ~10M params. The `time_upsample` stage is needed because LRS3 utterances often have more phonemes than frames once short clips happen — CTC requires `T_out >= L_label`.
"""))

cells.append(code("""\
import torch.nn as nn

class LandmarkVSR(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, input_dim=INPUT_DIM,
                 d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
                 dropout=DROPOUT, time_upsample=TIME_UPSAMPLE, max_len=MAX_FRAMES * 2):
        super().__init__()
        self.embed = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.upsample = nn.ConvTranspose1d(d_model, d_model, kernel_size=time_upsample, stride=time_upsample)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x, lengths=None):
        # x: (B, T, INPUT_DIM)
        B, T, _ = x.shape
        h = self.embed(x) + self.pos_embed[:, :T]
        # Build padding mask if lengths provided
        mask = None
        if lengths is not None:
            mask = torch.arange(T, device=x.device)[None, :] >= lengths[:, None]
        h = self.encoder(h, src_key_padding_mask=mask)
        h = self.norm(h)
        h = h.transpose(1, 2)               # (B, d, T)
        h = self.upsample(h)                 # (B, d, T*K)
        h = h.transpose(1, 2)               # (B, T*K, d)
        return self.head(h)                  # (B, T*K, V)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LandmarkVSR().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {n_params/1e6:.2f}M params on {device}")

# Shape sanity check
with torch.no_grad():
    x = torch.randn(2, 100, INPUT_DIM, device=device)
    y = model(x)
    print(f"Forward: in={tuple(x.shape)} out={tuple(y.shape)}  "
          f"(expected T_out = {100 * TIME_UPSAMPLE})")
"""))

cells.append(md("""\
## 9. Checkpoint utilities

Same atomic-write-then-upload pattern as the pixel notebook. Different repo (`liperty-landmark-vsr-checkpoints`) so the two architectures don't collide.
"""))

cells.append(code("""\
import shutil, time, random as _random
from huggingface_hub import upload_file, hf_hub_download, create_repo, HfApi

api = HfApi()
try:
    create_repo(HF_CKPT_REPO, repo_type="model", private=True, exist_ok=True)
except Exception as e:
    print(f"create_repo: {e}")

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
        "config": {"RUN_NAME": RUN_NAME, "D_MODEL": D_MODEL, "N_HEADS": N_HEADS,
                   "N_LAYERS": N_LAYERS, "TIME_UPSAMPLE": TIME_UPSAMPLE,
                   "VOCAB_SIZE": VOCAB_SIZE, "INPUT_DIM": INPUT_DIM},
        "extra": extra or {},
    }
    local_latest = Path(CKPT_DIR) / CKPT_FILENAME
    local_step = Path(CKPT_DIR) / f"{CKPT_HISTORY_PREFIX}{step:08d}.pt"
    tmp = local_latest.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    tmp.rename(local_latest)
    shutil.copy2(local_latest, local_step)
    upload_file(path_or_fileobj=str(local_latest), path_in_repo=CKPT_FILENAME,
                repo_id=HF_CKPT_REPO, repo_type="model",
                commit_message=f"step={step} epoch={epoch}")
    upload_file(path_or_fileobj=str(local_step), path_in_repo=local_step.name,
                repo_id=HF_CKPT_REPO, repo_type="model",
                commit_message=f"step={step} epoch={epoch} (history)")
    print(f"    [ckpt] step={step} uploaded.")

def prune_old_checkpoints(keep=KEEP_LAST_CKPTS):
    try:
        files = api.list_repo_files(HF_CKPT_REPO, repo_type="model")
    except Exception:
        return
    history = sorted(f for f in files if f.startswith(CKPT_HISTORY_PREFIX) and f.endswith(".pt"))
    to_delete = history[:-keep] if len(history) > keep else []
    from huggingface_hub import delete_file
    for f in to_delete:
        try:
            delete_file(path_in_repo=f, repo_id=HF_CKPT_REPO, repo_type="model",
                        commit_message=f"Prune {f}")
        except Exception as e:
            print(f"    [prune] {f}: {e}")

def load_checkpoint_if_available(model, optimizer, scheduler, scaler):
    try:
        path = hf_hub_download(repo_id=HF_CKPT_REPO, repo_type="model",
                               filename=CKPT_FILENAME, local_dir=CKPT_DIR)
    except Exception:
        print("No prior checkpoint. Starting fresh.")
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
## 10. Optimizer + scheduler + resume
"""))

cells.append(code("""\
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

TOTAL_STEPS = max(40000, len(train_loader) * 50) if len(train_loader) > 0 else 40000

def make_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))
    return LambdaLR(optimizer, lr_lambda)

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))
scheduler = make_scheduler(optimizer, WARMUP_STEPS, TOTAL_STEPS)
scaler = torch.amp.GradScaler("cuda") if (USE_FP16 and torch.cuda.is_available()) else None

start_step, start_epoch = load_checkpoint_if_available(model, optimizer, scheduler, scaler)
print(f"Starting step={start_step} epoch={start_epoch}")
print(f"Cosine total_steps={TOTAL_STEPS}; current LR={optimizer.param_groups[0]['lr']:.2e}")
"""))

cells.append(md("""\
## 11. Training loop

Same time-budget pattern as the pixel notebook. CTC loss over the upsampled output.
"""))

cells.append(code("""\
import torch.nn.functional as F

def train_loop(start_step, start_epoch, budget_min):
    if len(train_loader) == 0:
        print("Empty dataloader. Set TRANSCRIPT_SOURCE and re-run cell 8.")
        return start_step, start_epoch
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
                if batch is None:
                    continue
                if time.monotonic() - start_wall > deadline_s:
                    print(f"[budget] elapsed={(time.monotonic()-start_wall)/60:.1f} min ≥ {budget_min}; stopping.")
                    return step, epoch
                L, lengths, Y, label_lengths = batch
                L = L.to(device, non_blocking=True)
                lengths = lengths.to(device, non_blocking=True)
                Y = Y.to(device, non_blocking=True)
                label_lengths = label_lengths.to(device, non_blocking=True)

                if scaler is not None:
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        logits = model(L, lengths)                       # (B, T_out, V)
                        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
                        T_out = log_probs.shape[0]
                        input_lengths = (lengths * TIME_UPSAMPLE).clamp_max(T_out)
                        loss = ctc_loss(log_probs, Y.clamp_min(0), input_lengths, label_lengths)
                    scaler.scale(loss / GRAD_ACCUM).backward()
                else:
                    logits = model(L, lengths)
                    log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
                    T_out = log_probs.shape[0]
                    input_lengths = (lengths * TIME_UPSAMPLE).clamp_max(T_out)
                    loss = ctc_loss(log_probs, Y.clamp_min(0), input_lengths, label_lengths)
                    (loss / GRAD_ACCUM).backward()

                running_loss += float(loss.item()); running_n += 1

                if running_n % GRAD_ACCUM == 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer); scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    step += 1

                    if step % LOG_EVERY_STEPS == 0:
                        avg = running_loss / max(1, running_n)
                        running_loss = 0.0; running_n = 0
                        elapsed = (time.monotonic() - start_wall) / 60
                        print(f"  step={step:6d} epoch={epoch} loss={avg:.3f} "
                              f"lr={optimizer.param_groups[0]['lr']:.2e} t={elapsed:.1f}min")

                    if step - last_ckpt_step >= CKPT_EVERY_STEPS:
                        save_checkpoint(model, optimizer, scheduler, scaler, step, epoch)
                        last_ckpt_step = step
                        prune_old_checkpoints(KEEP_LAST_CKPTS)

            epoch += 1
            print(f"[epoch] completed {epoch}")
    except KeyboardInterrupt:
        print("KeyboardInterrupt — flushing.")
        return step, epoch

final_step, final_epoch = train_loop(start_step, start_epoch, TIME_BUDGET_MIN)
print(f"\\nLoop finished. step={final_step} epoch={final_epoch}")
"""))

cells.append(md("""\
## 12. Final flush
"""))

cells.append(code("""\
save_checkpoint(model, optimizer, scheduler, scaler, final_step, final_epoch,
                extra={"flush_reason": "end-of-session"})
prune_old_checkpoints(KEEP_LAST_CKPTS)
print(f"Final checkpoint pushed: step={final_step} epoch={final_epoch}")
"""))

cells.append(md("""\
## 13. ONNX export (run after training converges)

When you're happy with a checkpoint, export to ONNX and bundle into `app/src/main/assets/landmark_vsr_model.onnx`. The Android side needs a new `LipCoordNetEngine : ModelEngine` that wraps `OrtSession` and reports `getInputLayout() = NTHWC`-ish (actually it's `(B, T, 136)` — a different shape regime; the engine's job is to translate from per-frame mediapipe landmarks to that input).
"""))

cells.append(code("""\
EXPORT_ONNX = False   # set True after a real training run
if EXPORT_ONNX:
    model.eval()
    dummy = torch.randn(1, 100, INPUT_DIM, device=device)
    dummy_lengths = torch.tensor([100], dtype=torch.long, device=device)
    onnx_path = Path(CKPT_DIR) / f"{RUN_NAME}.onnx"
    torch.onnx.export(
        model, (dummy, dummy_lengths), str(onnx_path),
        input_names=["landmarks", "lengths"],
        output_names=["logits"],
        dynamic_axes={"landmarks": {0: "batch", 1: "frames"},
                      "logits":    {0: "batch", 1: "out_frames"}},
        opset_version=14,
    )
    print(f"Exported: {onnx_path}  ({onnx_path.stat().st_size/1e6:.1f} MB)")
    upload_file(path_or_fileobj=str(onnx_path),
                path_in_repo=onnx_path.name,
                repo_id=HF_CKPT_REPO, repo_type="model",
                commit_message=f"ONNX export at step {final_step}")
else:
    print("ONNX export disabled. Set EXPORT_ONNX = True after training converges.")
"""))

cells.append(md("""\
## 14. Cross-account handoff

Same as the pixel notebook: identical HF token across all your Kaggle/Colab accounts means the next session resumes from the most recent checkpoint regardless of which account or platform.

When you switch accounts:
1. Open this notebook on the new platform.
2. Add the `HF_TOKEN` secret if not already there.
3. `Save Version → Save & Run All (Commit)` (Kaggle) or `Runtime → Run all` (Colab).

The first 3 cells set up the environment; cell 11 detects the prior checkpoint and resumes optimizer/scheduler/RNG state in <30 s.
"""))

cells.append(md("""\
## 15. Troubleshooting

**"Empty dataloader."** Most likely you haven't run `preprocess_landmarks_resumable.ipynb` yet, so no shards exist on HF. Run that first to populate `liperty-grid-landmarks` and `liperty-tcd-landmarks`. Alternatively, switch to `lrs3_landmark` once you have transcripts (Oxford VGG academic access for the .txt files, then set `LRS3_TXT_ROOT` and `LRS3_TRANSCRIPT_SOURCE = "lrs3_official"`).

**"Loss not decreasing."** With 10M params on landmarks, the model has limited capacity. If loss plateaus at high values:
- Increase `D_MODEL` to 256 and `N_LAYERS` to 6 (~25M params).
- Verify your transcripts are sane — print a few `(landmark_clip, phoneme_ids)` pairs.
- Drop `LR` to 1e-4.

**"Validation? Eval?"** Not in this notebook. Once you have a converged checkpoint, write a separate eval notebook that loads it, runs greedy CTC over a held-out split, and computes phoneme error rate.

**"My None-rejection rate is too aggressive."** Set `MAX_NONE_RATIO = 0.40` to keep more clips. Quality of those clips will be lower (more interpolation across larger gaps).

**"Checkpoint upload is slow."** A 10M-param model is ~40 MB. Each `CKPT_EVERY_STEPS=500` interval at ~5s/step = 40 min between uploads. If HF rate-limits you, drop the `local_step` (history) upload from `save_checkpoint` and only push `<run>-latest.pt`.

**"On Kaggle, the lrs3-landmark dataset doesn't show up."** Use the right-side panel (`+ Add Input → Search datasets → e1lephant/lrs3-landmark`). It mounts at `/kaggle/input/lrs3-landmark/` automatically.
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
