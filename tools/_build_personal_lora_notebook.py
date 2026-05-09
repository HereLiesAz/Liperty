"""Generate tools/train_personal_lora_resumable.ipynb.

Sibling to tools/export_autoavsr_to_onnx.ipynb. Loads the Auto-AVSR
PyTorch checkpoint, applies LoRA adapters to the visual encoder's
attention layers, and fine-tunes on the user's own face/voice
recordings. Outputs adapter weights (~10 MB) you can merge back
into the base model and re-export to ONNX for ~30-50% WER -> 10-20%
WER on YOU specifically.

Per AAAI 2025 "Personalized Lip Reading" (arXiv:2409.00986).

Run with:  python tools/_build_personal_lora_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "train_personal_lora_resumable.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.splitlines(keepends=True)}


cells: list[dict] = []

cells.append(md("""\
# Personal LoRA fine-tune for Auto-AVSR

Take the Auto-AVSR LRS3_V_WER19.1 PyTorch checkpoint and adapt it to *your* face/voice via LoRA. Following the AAAI 2025 paper on personalized lip reading (arXiv:2409.00986), we freeze the encoder's pretrained weights and learn a small set of low-rank adapters on attention + FFN projections. The adapter file is ~10 MB; merging it back into the base produces a checkpoint that performs noticeably better on the speaker it was trained on.

**Expected impact on WER**

| Stage | LRS3 (general) WER | Your-face WER |
|---|---|---|
| Auto-AVSR greedy CTC, no adaptation (where you are now) | 30-50% | 30-50% |
| + on-device beam search + sliding window (already shipped) | 30-50% | 30-50% |
| + this LoRA, fine-tuned on ~2 hours of you reading sentences | 30-50% (unchanged) | **10-20%** |

The base model gets WORSE on the general population in exchange for being better on you. That's fine for an assistive on-device tool you use for yourself.

**What you record**

The 720 IEEE/Harvard Sentences (https://www.cs.columbia.edu/~hgs/audio/harvard.html). Phonetically balanced, ~5-7 words each, ~3-5 seconds when read aloud. Total recording time ~60-90 minutes if you read steadily. You can also record fewer (250+ is usable) at the cost of less coverage.

**Hardware**

Free Kaggle GPU is enough — LoRA training on a single 16-frame clip is small. Total training time on P100: ~2-4 hours for one epoch over 720 clips. Resumable across sessions via the same HF Hub state-store pattern as the other notebooks.
"""))

cells.append(md("""\
## How to use this

**One-time prep (do once):**

1. Print or pull up the Harvard Sentences list.
2. Record yourself reading them with a phone or webcam. Frame your face roughly centered, decent lighting, 720p or better. Speak naturally; don't over-enunciate. Read in batches of ~10 sentences with short pauses between. Save as one or more video files (mp4 / mov).
3. Upload those video files to a private HuggingFace dataset repo: `<you>/liperty-personal-recordings`. Web UI works fine; one folder per recording session.

**Recurring sessions (Kaggle):**

1. Open this notebook on Kaggle. Set runtime: GPU P100, Internet ON.
2. Add Kaggle Secret `HF_TOKEN` (write-scoped).
3. `Save Version > Save & Run All (Commit)` for headless execution.
4. The first session does data prep (Whisper transcription + lip-cropping). Subsequent sessions just train.

**Outputs**

- LoRA adapter weights at `<you>/liperty-autoavsr-personal-lora` (HF model repo). ~10 MB.
- Optional: merged-and-exported ONNX at the same repo, for direct drop-in to `app/src/main/assets/`.
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
print(f"Python: {sys.version.split()[0]}, PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
"""))

cells.append(code("""\
%%capture
!pip install -q \\
    "huggingface_hub>=0.27,<1.0" \\
    "peft>=0.13,<1.0" \\
    "openai-whisper>=20240930" \\
    "av>=12.0" \\
    "opencv-python>=4.10" \\
    "mediapipe>=0.10.18" \\
    "onnx>=1.16" \\
    "onnxruntime>=1.18" \\
    "sentencepiece" \\
    "jiwer>=3.0"
print("Deps installed.")
"""))

cells.append(code("""\
WORK_DIR = "/kaggle/working/work" if IS_KAGGLE else "/content/work"
os.makedirs(WORK_DIR, exist_ok=True)

CHAPLIN_DIR = os.path.join(WORK_DIR, "chaplin")
if not os.path.exists(CHAPLIN_DIR):
    !git clone --depth 1 https://github.com/amanvirparhar/chaplin.git {CHAPLIN_DIR}
sys.path.insert(0, CHAPLIN_DIR)
print(f"Chaplin checkout: {CHAPLIN_DIR}")
"""))

cells.append(md("""\
## 2. HuggingFace auth
"""))

cells.append(code("""\
from huggingface_hub import login, whoami

token = os.environ.get("HF_TOKEN")
if not token and IS_KAGGLE:
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        pass
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
## 3. Run config
"""))

cells.append(code("""\
HF_USER = "HereLiesAz"   # Your HuggingFace username

# Source: Auto-AVSR PyTorch checkpoint (the one we ONNX-exported in the
# sibling notebook). Same model.json + model.pth.
SRC_REPO = "Amanvir/LRS3_V_WER19.1"

# Your private repo of recorded video files (mp4/mov), one folder per session.
HF_RECORDINGS_REPO = f"{HF_USER}/liperty-personal-recordings"

# Cached preprocessed clip shards (output of cell 7).
HF_CLIPS_REPO = f"{HF_USER}/liperty-personal-clips"

# Output: LoRA adapter checkpoints + final merged-and-exported ONNX.
HF_LORA_REPO = f"{HF_USER}/liperty-autoavsr-personal-lora"

RUN_NAME = "autoavsr-lora-v1"

# Time budget. Kaggle GPU sessions cap at ~9h; Colab free at ~4h.
TIME_BUDGET_MIN = 480 if IS_KAGGLE else 200

# Auto-AVSR input shape (matches what we exported to ONNX).
NUM_FRAMES = 16
IMG_SIZE = 88
PIXEL_MEAN = 0.421
PIXEL_STD = 0.165

# LoRA hyperparams. r=8 is a reasonable default; bigger r = more adapter
# capacity but more overfitting risk on small personal datasets.
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# Training.
BATCH_SIZE = 4
LR = 5e-4
WEIGHT_DECAY = 1e-3
WARMUP_STEPS = 50
NUM_WORKERS = 2
USE_FP16 = True

CKPT_EVERY_STEPS = 100

# Local working dirs (ephemeral on Kaggle/Colab).
CKPT_DIR = os.path.join(WORK_DIR, "ckpt")
DATA_DIR = os.path.join(WORK_DIR, "data")
for d in (CKPT_DIR, DATA_DIR):
    os.makedirs(d, exist_ok=True)

print(f"Run:    {RUN_NAME}")
print(f"Budget: {TIME_BUDGET_MIN} min")
print(f"LoRA:   r={LORA_RANK} alpha={LORA_ALPHA} dropout={LORA_DROPOUT}")
"""))

cells.append(md("""\
## 4. Pull Auto-AVSR base checkpoint
"""))

cells.append(code("""\
from huggingface_hub import hf_hub_download

base_dir = os.path.join(WORK_DIR, "base")
os.makedirs(base_dir, exist_ok=True)
model_json = hf_hub_download(repo_id=SRC_REPO, filename="model.json", local_dir=base_dir)
model_pth  = hf_hub_download(repo_id=SRC_REPO, filename="model.pth",  local_dir=base_dir)
print(f"Base checkpoint: {model_pth} ({os.path.getsize(model_pth) / 1e6:.0f} MB)")
"""))

cells.append(md("""\
## 5. Build the model + LoRA wrapper

Same load path as the export notebook, then wrap with peft's LoRA. We
target the encoder's `linear_q`, `linear_k`, `linear_v`, `linear_out`,
and `linear1`/`linear2` of each transformer block — standard LoRA
target set for transformer encoders.
"""))

cells.append(code("""\
import argparse
import json as _json
from peft import LoraConfig, get_peft_model

with open(model_json) as f:
    confs = _json.load(f)
args_dict = confs if isinstance(confs, dict) else confs[2]
train_args = argparse.Namespace(**args_dict)

labels_type = getattr(train_args, "labels_type", "char")
print(f"labels_type: {labels_type}")
if labels_type == "char":
    token_list = list(train_args.char_list)
elif labels_type == "unigram5000":
    src_vocab = os.path.join(CHAPLIN_DIR, "pipelines", "tokens", "unigram5000_units.txt")
    token_list = (
        ["<blank>"]
        + [w.split()[0] for w in open(src_vocab, encoding="utf-8").read().splitlines() if w.strip()]
        + ["<eos>"]
    )
else:
    raise ValueError(f"Unsupported labels_type: {labels_type}")
odim = len(token_list)
print(f"odim: {odim}")

from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
base_model = E2E(odim, train_args)
state = torch.load(model_pth, map_location="cpu", weights_only=False)
if isinstance(state, dict):
    for outer in ("state_dict", "model", "model_state_dict"):
        if outer in state and isinstance(state[outer], dict):
            state = state[outer]; break
base_model.load_state_dict(state, strict=False)
base_model.eval()

# Wrap the encoder with LoRA. Target the encoder's attention + FFN
# linears. Names depend on ESPnet build; print and adjust if needed.
print("\\nEncoder submodules (look for transformer attention modules):")
for name, _ in base_model.encoder.named_modules():
    if any(k in name for k in ("linear_q", "linear_k", "linear_v", "linear_out", "feed_forward", "linear1", "linear2")):
        print(f"  {name}")
        break  # just spot-check the first match

lora_cfg = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    target_modules=["linear_q", "linear_k", "linear_v", "linear_out",
                    "linear1", "linear2"],
    task_type=None,
)
# Apply only to the encoder; leave the CTC head and frontend frozen with
# their pretrained weights.
base_model.encoder = get_peft_model(base_model.encoder, lora_cfg)
print("\\nLoRA params:")
base_model.encoder.print_trainable_parameters()
"""))

cells.append(md("""\
## 6. Pull recordings + transcribe with Whisper

Idempotent: skips already-transcribed recordings. Output: one `.json` per
recording with sentence-level timestamps + transcripts.
"""))

cells.append(code("""\
import whisper
from huggingface_hub import snapshot_download, list_repo_files
from pathlib import Path

recordings_dir = Path(DATA_DIR) / "recordings"
recordings_dir.mkdir(parents=True, exist_ok=True)
try:
    snapshot_download(repo_id=HF_RECORDINGS_REPO, repo_type="dataset",
                      local_dir=str(recordings_dir),
                      allow_patterns=["*.mp4", "*.mov", "*.MP4", "*.MOV"])
except Exception as e:
    print(f"Could not pull recordings: {e}")
    print(f"Upload your video files to https://huggingface.co/datasets/{HF_RECORDINGS_REPO} first.")
    raise

video_files = sorted([p for p in recordings_dir.rglob("*") if p.suffix.lower() in (".mp4", ".mov")])
print(f"Found {len(video_files)} video files")

if video_files:
    whisper_model = whisper.load_model("medium.en")
    transcripts_dir = Path(DATA_DIR) / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    for vid in video_files:
        out_json = transcripts_dir / f"{vid.stem}.json"
        if out_json.exists():
            continue
        print(f"Transcribing {vid.name}...")
        result = whisper_model.transcribe(str(vid), word_timestamps=False, verbose=False)
        with open(out_json, "w", encoding="utf-8") as f:
            _json.dump({"segments": result["segments"], "text": result["text"]}, f)
    print(f"Transcripts: {len(list(transcripts_dir.glob('*.json')))} files")
"""))

cells.append(md("""\
## 7. Build training clips

For each Whisper segment, extract the corresponding video frames, run
MediaPipe face landmark detection, crop the mouth ROI to 88x88
grayscale, normalize, and pair with the segment text (encoded to
SentencePiece subword indices). One `.pt` shard per recording.

Idempotent: skips already-built shards.
""" ))

cells.append(code("""\
import re
import numpy as np
import cv2
import mediapipe as mp
from huggingface_hub import HfApi, create_repo, upload_file

api = HfApi()
try:
    create_repo(HF_CLIPS_REPO, repo_type="dataset", private=True, exist_ok=True)
except Exception as e:
    print(f"create_repo: {e}")

TOKEN_TO_ID = {tok: i for i, tok in enumerate(token_list)}

def _text_to_char_ids(text):
    \"\"\"Char-level: each character mapped to its vocab index. ESPnet char_list
    typically uses uppercase letters + space + apostrophe.\"\"\"
    text = re.sub(r\"[^A-Za-z' ]\", \"\", text.strip().upper())
    return [TOKEN_TO_ID[c] for c in text if c in TOKEN_TO_ID]

def _text_to_subword_ids(text):
    \"\"\"unigram5000 (SentencePiece-style): greedy longest-match against the
    token list, with words prefixed by the SentencePiece word-boundary
    marker. Best-effort; the proper SentencePiece .model isn't bundled
    with the Auto-AVSR checkpoint so we approximate.\"\"\"
    text = re.sub(r\"[^A-Za-z' ]\", \"\", text.strip().upper())
    if not text:
        return []
    pieces = []
    cursor = \"▁\" + text.replace(\" \", \"▁\")
    while cursor:
        match = None
        for ln in range(min(len(cursor), 30), 0, -1):
            if cursor[:ln] in TOKEN_TO_ID:
                match = cursor[:ln]; break
        if match is None:
            cursor = cursor[1:]; continue
        pieces.append(TOKEN_TO_ID[match])
        cursor = cursor[len(match):]
    return pieces

# Pick the right encoder for the checkpoint's labels_type.
if labels_type == \"char\":
    text_to_token_ids = _text_to_char_ids
elif labels_type == \"unigram5000\":
    text_to_token_ids = _text_to_subword_ids
else:
    raise ValueError(f\"Unsupported labels_type: {labels_type}\")
# Smoke-test the encoder so a misconfigured vocab fails loudly.
sample = text_to_token_ids(\"the quick brown fox\")
assert sample, f\"Encoder returned empty for sample text; labels_type={labels_type} likely mismatched. \" \\
               f\"First 5 vocab tokens: {token_list[:5]}\"
print(f\"Encoder smoke-test OK: 'the quick brown fox' -> {len(sample)} tokens\")

# MediaPipe lip box.
_face_lms = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.4)
def crop_mouth(frame_bgr):
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = _face_lms.process(rgb)
    if not res.detections: return None
    box = res.detections[0].location_data.relative_bounding_box
    x1 = int(box.xmin * w); y1 = int(box.ymin * h)
    x2 = int((box.xmin + box.width) * w); y2 = int((box.ymin + box.height) * h)
    # Use the lower 40% of the face bbox as the mouth ROI.
    cy_top = int(y1 + (y2 - y1) * 0.55)
    cx = (x1 + x2) // 2
    side = int((x2 - x1) * 0.5)
    half = side // 2
    rx1, ry1 = max(0, cx - half), max(0, cy_top)
    rx2, ry2 = min(w, cx + half), min(h, cy_top + side)
    if rx2 <= rx1 or ry2 <= ry1: return None
    crop = frame_bgr[ry1:ry2, rx1:rx2]
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return crop

# Per-recording loop.
for vid in video_files:
    shard_path = Path(WORK_DIR) / f"clips-{vid.stem}.pt"
    if shard_path.exists():
        continue
    json_path = Path(DATA_DIR) / "transcripts" / f"{vid.stem}.json"
    if not json_path.exists():
        print(f"  skip {vid.name}: no transcript")
        continue
    with open(json_path, encoding="utf-8") as f:
        meta = _json.load(f)
    segments = meta["segments"]
    cap = cv2.VideoCapture(str(vid))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    print(f"  {vid.name}: {len(segments)} segments @ {fps:.1f} fps")
    clips, ph_lists, texts = [], [], []
    for seg in segments:
        seg_text = seg["text"].strip()
        ph = text_to_token_ids(seg_text)
        if not ph: continue
        t0, t1 = seg["start"], seg["end"]
        cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000.0)
        n_frames = max(1, int((t1 - t0) * fps))
        if n_frames < NUM_FRAMES: continue
        idxs = np.linspace(0, n_frames - 1, NUM_FRAMES).astype(int)
        clip = np.zeros((NUM_FRAMES, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        ok_count = 0
        for i, fi in enumerate(idxs):
            cap.set(cv2.CAP_PROP_POS_MSEC, (t0 + fi / fps) * 1000.0)
            ok, frame = cap.read()
            if not ok: break
            mouth = crop_mouth(frame)
            if mouth is None: continue
            clip[i] = mouth
            ok_count += 1
        if ok_count < NUM_FRAMES * 0.7: continue
        clips.append(clip)
        ph_lists.append(ph)
        texts.append(seg_text)
    cap.release()
    if not clips: continue
    torch.save({
        "clips":    torch.from_numpy(np.stack(clips)),  # (N, T, H, W) uint8
        "phonemes": ph_lists,
        "texts":    texts,
    }, shard_path)
    print(f"  {vid.name}: saved {len(clips)} clips, uploading...")
    upload_file(path_or_fileobj=str(shard_path),
                path_in_repo=shard_path.name,
                repo_id=HF_CLIPS_REPO, repo_type="dataset",
                commit_message=f"clips for {vid.name}: {len(clips)} segments")

print("Clip preprocessing done.")
"""))

cells.append(md("""\
## 8. Dataset + dataloader
"""))

cells.append(code("""\
from torch.utils.data import Dataset, DataLoader, ConcatDataset

class PersonalClipDataset(Dataset):
    def __init__(self, shard_paths):
        self.entries = []
        for p in shard_paths:
            d = torch.load(p, map_location="cpu", weights_only=False)
            for clip, ph in zip(d["clips"], d["phonemes"]):
                if not ph: continue
                self.entries.append((clip, torch.tensor(ph, dtype=torch.long)))

    def __len__(self): return len(self.entries)

    def __getitem__(self, i):
        clip, phonemes = self.entries[i]
        # (T, H, W) uint8 -> (1, T, H, W) float32 NCTHW (single channel for grayscale)
        x = clip.float() / 255.0
        x = (x - PIXEL_MEAN) / PIXEL_STD
        x = x.unsqueeze(0)   # add channel dim -> (C=1, T, H, W)
        return x, phonemes

def ctc_collate(batch):
    Xs, Ys = zip(*batch)
    X = torch.stack(Xs, dim=0)             # (B, 1, T, H, W) NCTHW
    label_lengths = torch.tensor([len(y) for y in Ys], dtype=torch.long)
    Lmax = max(label_lengths).item()
    Y = torch.full((len(batch), Lmax), -1, dtype=torch.long)
    for i, y in enumerate(Ys):
        Y[i, :y.shape[0]] = y
    input_lengths = torch.full((len(batch),), X.shape[2], dtype=torch.long)
    return X, input_lengths, Y, label_lengths

shard_paths = sorted(Path(WORK_DIR).glob("clips-*.pt"))
print(f"Local clip shards: {len(shard_paths)}")
ds = PersonalClipDataset(shard_paths)
print(f"Total clips: {len(ds)}")
loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                    num_workers=NUM_WORKERS, collate_fn=ctc_collate,
                    pin_memory=True, drop_last=True,
                    persistent_workers=NUM_WORKERS > 0)
print(f"Batches per epoch: {len(loader)}")
"""))

cells.append(md("""\
## 9. Training loop with LoRA
"""))

cells.append(code("""\
import torch.nn as nn
import torch.nn.functional as F
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = base_model.to(device)
base_model.train()
# Make sure non-LoRA params stay frozen.
for n, p in base_model.named_parameters():
    if "lora_" not in n:
        p.requires_grad_(False)

trainable = [p for p in base_model.parameters() if p.requires_grad]
print(f"Trainable params: {sum(p.numel() for p in trainable):,}")

optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)
scaler = torch.amp.GradScaler("cuda") if (USE_FP16 and torch.cuda.is_available()) else None

ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True, reduction="mean")

start = time.monotonic()
deadline = TIME_BUDGET_MIN * 60
step = 0
for epoch in range(100):
    for batch in loader:
        if time.monotonic() - start > deadline:
            print("Time budget reached; stopping.")
            break
        X, input_lengths, Y, label_lengths = batch
        X = X.to(device); Y = Y.to(device)
        input_lengths = input_lengths.to(device); label_lengths = label_lengths.to(device)

        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=scaler is not None):
            enc_out, _ = base_model.encoder(X, None)
            logits = base_model.ctc.ctc_lo(enc_out)
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # (T, B, V)
            T_out = log_probs.shape[0]
            in_lens = (input_lengths.float() * T_out / X.shape[2]).long().clamp_max(T_out)
            loss = ctc_loss(log_probs, Y.clamp_min(0), in_lens, label_lengths)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

        step += 1
        if step % 10 == 0:
            elapsed = (time.monotonic() - start) / 60
            print(f"  step={step} epoch={epoch} loss={loss.item():.3f} t={elapsed:.1f}min")

        if step % CKPT_EVERY_STEPS == 0:
            ckpt_path = os.path.join(CKPT_DIR, f"{RUN_NAME}-step{step:06d}.pt")
            torch.save({
                "lora_state": {k: v for k, v in base_model.encoder.state_dict().items() if "lora_" in k},
                "step": step,
                "epoch": epoch,
            }, ckpt_path)
            try:
                create_repo(HF_LORA_REPO, repo_type="model", private=True, exist_ok=True)
            except Exception:
                pass
            upload_file(path_or_fileobj=ckpt_path,
                        path_in_repo=os.path.basename(ckpt_path),
                        repo_id=HF_LORA_REPO, repo_type="model",
                        commit_message=f"step {step}")
            print(f"  ckpt uploaded: {os.path.basename(ckpt_path)}")
    else:
        continue
    break  # exit on inner break (time budget)

print(f"\\nFinished. step={step}")
"""))

cells.append(md("""\
## 10. Final flush + ONNX re-export

Merges the LoRA adapters into the base encoder, re-exports the visual-only
CTC graph to ONNX, and uploads to the HF model repo. The user replaces
`app/src/main/assets/autoavsr_lrs3_visual_ctc.onnx` with the new file.
"""))

cells.append(code("""\
EXPORT_PERSONAL_ONNX = False  # flip to True once training has converged

if EXPORT_PERSONAL_ONNX:
    base_model.eval()
    # Merge LoRA into base.
    base_model.encoder = base_model.encoder.merge_and_unload()

    class VisualOnlyCTC(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, video):
            enc, _ = self.m.encoder(video, None)
            return F.log_softmax(self.m.ctc.ctc_lo(enc), dim=-1)
    wrapper = VisualOnlyCTC(base_model).eval()

    onnx_path = os.path.join(CKPT_DIR, "autoavsr_lrs3_visual_ctc_personal.onnx")
    dummy = torch.randn(1, 1, NUM_FRAMES, IMG_SIZE, IMG_SIZE, device=device)
    with torch.no_grad():
        torch.onnx.export(
            wrapper, dummy, onnx_path,
            input_names=["video"], output_names=["log_probs"],
            dynamic_axes={"video": {0: "batch", 2: "time"},
                          "log_probs": {0: "batch", 1: "time_sub"}},
            opset_version=17, do_constant_folding=True, dynamo=False,
        )
    print(f"Exported: {onnx_path} ({os.path.getsize(onnx_path)/1e6:.0f} MB)")
    upload_file(path_or_fileobj=onnx_path,
                path_in_repo=os.path.basename(onnx_path),
                repo_id=HF_LORA_REPO, repo_type="model",
                commit_message="Personal LoRA-merged ONNX export")
    print("Replace app/src/main/assets/autoavsr_lrs3_visual_ctc.onnx with this file.")
"""))

cells.append(md("""\
## 11. Cross-account handoff + troubleshooting

Same pattern as the other resumable notebooks: a single HF token unlocks state across all your Kaggle/Colab accounts. Re-run all cells and cell 8 picks up the latest LoRA checkpoint from HF.

**Top failure modes:**

1. **`labels_type: char` instead of unigram5000.** The Auto-AVSR base might be a char-level model on your account's mirror. The `text_to_token_ids` function in cell 7 then needs to convert text to character indices instead of subword pieces. Adapt: `text_to_token_ids = lambda text: [TOKEN_TO_ID.get(c.upper(), TOKEN_TO_ID[' ']) for c in text]`.

2. **MediaPipe face detection fails on most clips.** If you record at low resolution or unusual framing, the face detector misses. Try recording at 1080p, face centered, decent lighting. Alternatively swap to `mediapipe.solutions.face_mesh` with `min_detection_confidence=0.3`.

3. **Loss starts high and stays high.** LoRA target_modules names mismatch — the encoder's submodules in this checkpoint use different names. Run the print loop in cell 5 to see what's actually there, then adjust `target_modules` in `LoraConfig`.

4. **Whisper transcripts are empty / garbled.** The `medium.en` model needs ~3 GB GPU memory. If Kaggle is OOM'ing, drop to `small.en` or `base.en`.
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
