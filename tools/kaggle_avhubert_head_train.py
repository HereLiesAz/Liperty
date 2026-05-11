"""Train a phoneme CTC head on top of the AV-HuBERT base+vox+433h
fine-tuned visual encoder.

This is the V3 backend's "downstream task" stage. The encoder is
already public ONNX at HereLiesAz/liperty-avhubert-encoder (no
fairseq deps needed at training time!), and produces (T, 768)
features per clip. We train a tiny head (LayerNorm + Linear + LSTM +
CTC projection) to map those features to the same 40-symbol ARPABET
vocab Liperty's V1 baseline used.

End state: a second ONNX file (avhubert_v3_phoneme_head.onnx) the
Android app can chain after the encoder to get phoneme logits, then
existing SubwordCtcBeamDecoder.kt decodes to text.

Strategy details:
- Freeze the encoder (we don't have the VRAM/time to fine-tune it).
- Pre-compute features on the GRID shards we trained V1 against
  (HereLiesAz/liperty-grid-preprocessed, public). Cache to disk so
  feature extraction runs once, not every epoch.
- Train head with CTC loss. Effective batch ~32 fits on a T4.

Run from a fresh Kaggle notebook (GPU on, HF_TOKEN secret enabled):

    import urllib.request
    exec(urllib.request.urlopen(
        'https://raw.githubusercontent.com/HereLiesAz/Liperty/main/tools/kaggle_avhubert_head_train.py'
    ).read())
"""

import os
import sys
import subprocess
import time

# ---------------------------------------------------------------------------
# 0. Env detect
# ---------------------------------------------------------------------------
IS_KAGGLE = os.path.exists("/kaggle/working") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ
WORK_DIR = "/kaggle/working/work" if IS_KAGGLE else "/content/work"
os.makedirs(WORK_DIR, exist_ok=True)

print(f"Environment: {'kaggle' if IS_KAGGLE else 'other'}")
print(f"Python: {sys.version.split()[0]}")


# ---------------------------------------------------------------------------
# 1. Deps. onnxruntime-gpu + torch + huggingface_hub.
#    NO FAIRSEQ. NO OMEGACONF. NO TRANSFORMERS.
# ---------------------------------------------------------------------------
def _pip(*pkgs):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"pip install {pkgs} stderr (tail):\n{r.stderr[-800:]}")
        raise RuntimeError("pip install failed")

print("Installing deps...")
_pip("onnxruntime-gpu", "huggingface_hub>=0.27,<1.0", "jiwer")


# ---------------------------------------------------------------------------
# 2. HF auth + downloads
# ---------------------------------------------------------------------------
from huggingface_hub import login, hf_hub_download, snapshot_download, upload_file
token = os.environ.get("HF_TOKEN")
if not token and IS_KAGGLE:
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        pass
if token:
    login(token, add_to_git_credential=True)
print("HF auth done")

ENC_REPO = "HereLiesAz/liperty-avhubert-encoder"
ENC_FILE = "avhubert_base_vox_433h_visual_encoder.onnx"
GRID_REPO = "HereLiesAz/liperty-grid-preprocessed"
HEAD_REPO = "HereLiesAz/liperty-v3-phoneme-head"   # we'll create this

# 2a. Encoder ONNX
print(f"Pulling {ENC_FILE}...")
enc_path = hf_hub_download(ENC_REPO, ENC_FILE, local_dir=WORK_DIR)
print(f"  {enc_path}  ({os.path.getsize(enc_path) / 1e6:.0f} MB)")


# ---------------------------------------------------------------------------
# 3. Configuration
# ---------------------------------------------------------------------------
MAX_SPEAKERS = int(os.environ.get("LIPERTY_MAX_SPEAKERS", "6"))  # same cap as V1
NUM_FRAMES = 16                    # video clip length our shards have
CROP_SIZE  = 88                    # AV-HuBERT input size
PIXEL_MEAN = 0.421
PIXEL_STD  = 0.165

# Match Liperty's MLConstants.PHONEME_VOCAB exactly
PHONEME_VOCAB = ["_", "AA","AE","AH","AO","AW","AY","B","CH","D","DH","EH","ER","EY",
                 "F","G","HH","IH","IY","JH","K","L","M","N","NG","OW","OY","P",
                 "R","S","SH","T","TH","UH","UW","V","W","Y","Z","ZH"]
VOCAB_SIZE = len(PHONEME_VOCAB)
BLANK_IDX  = 0
PHONEME_TO_IDX = {p: i for i, p in enumerate(PHONEME_VOCAB)}

# Head architecture: kept small so it trains in ~1h on T4.
HEAD_DIM      = 512
HEAD_LAYERS   = 2
HEAD_DROPOUT  = 0.1

# Optimization
BATCH_SIZE   = 32
LR           = 3e-4
WEIGHT_DECAY = 1e-2
WARMUP_STEPS = 200
TOTAL_STEPS  = 5000   # ~30-60 min on T4
LOG_EVERY    = 50
CKPT_EVERY   = 500

CACHE_DIR = os.path.join(WORK_DIR, "feature_cache")
CKPT_DIR  = os.path.join(WORK_DIR, "head_ckpt")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 4. Pull GRID shards
# ---------------------------------------------------------------------------
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

print("Pulling GRID shards from HF...")
shard_dir = os.path.join(WORK_DIR, "grid_shards")
os.makedirs(shard_dir, exist_ok=True)
from huggingface_hub import HfApi
api = HfApi()
all_shards = sorted(f for f in api.list_repo_files(GRID_REPO, repo_type="dataset") if f.endswith(".pt"))
wanted = all_shards[:MAX_SPEAKERS] if MAX_SPEAKERS > 0 else all_shards
print(f"  pulling {len(wanted)} of {len(all_shards)} shards")
snapshot_download(repo_id=GRID_REPO, repo_type="dataset",
                  local_dir=shard_dir, allow_patterns=wanted)
shard_paths = sorted(f for f in os.listdir(shard_dir) if f.endswith(".pt"))
shard_paths = [os.path.join(shard_dir, f) for f in shard_paths if f in wanted]
print(f"  local shards: {len(shard_paths)}")


# ---------------------------------------------------------------------------
# 5. Frame -> 88x88 mouth ROI normalization
# ---------------------------------------------------------------------------
def frames_to_mouth_88(frames_uint8_rgb_224: np.ndarray) -> np.ndarray:
    """(T, 224, 224, 3) uint8 RGB -> (T, 1, 88, 88) float32 normalized,
    matching Liperty's MainActivity.AUTOAVSR_* preprocessing math."""
    T = frames_uint8_rgb_224.shape[0]
    out = np.empty((T, 1, CROP_SIZE, CROP_SIZE), dtype=np.float32)
    for t in range(T):
        bgr = cv2.cvtColor(frames_uint8_rgb_224[t], cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        # Mouth is centered ~2/3 down the face; take a 112x112 ROI there
        # then downsample to 88x88.
        cx = w // 2
        cy = int(h * 0.66)
        half = 56
        y1 = max(0, cy - half); y2 = min(h, cy + half)
        x1 = max(0, cx - half); x2 = min(w, cx + half)
        roi = gray[y1:y2, x1:x2]
        roi = cv2.resize(roi, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)
        out[t, 0] = (roi.astype(np.float32) / 255.0 - PIXEL_MEAN) / PIXEL_STD
    return out


# ---------------------------------------------------------------------------
# 6. Extract features via the AV-HuBERT ONNX (one-shot, cached)
# ---------------------------------------------------------------------------
import onnxruntime as ort

print("Setting up ONNX Runtime session...")
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
sess = ort.InferenceSession(enc_path, providers=providers)
print(f"  providers: {sess.get_providers()}")

def extract_features_batch(frames_88_batch: np.ndarray) -> np.ndarray:
    """frames: (B, T, 1, 88, 88) -> features: (B, T, 768).
    AV-HuBERT's ONNX expects (1, 1, T, 88, 88) per call (we exported
    with batch=1 dynamic_axes); loop the batch dim ourselves."""
    B = frames_88_batch.shape[0]
    feats = []
    for b in range(B):
        # frames_88_batch[b]: (T, 1, 88, 88) -> (1, 1, T, 88, 88) (B, C, T, H, W)
        x = frames_88_batch[b].transpose(1, 0, 2, 3)[None]  # (1, 1, T, 88, 88)
        y = sess.run(["features"], {"video": x})[0]   # (1, T, 768)
        feats.append(y[0])
    return np.stack(feats)   # (B, T, 768)


def build_or_load_feature_cache():
    """For each clip in each shard: preprocess frames -> 88x88 mouth ROI ->
    run ONNX -> save features. Re-running this script reuses the cache."""
    entries = []   # (cache_path, phonemes)
    for sp_path in shard_paths:
        sp_name = os.path.basename(sp_path)
        cache_path = os.path.join(CACHE_DIR, sp_name.replace(".pt", "_feats.pt"))
        if os.path.exists(cache_path):
            d = torch.load(cache_path, map_location="cpu", weights_only=False)
            for feat, ph in zip(d["features"], d["phonemes"]):
                entries.append((feat, torch.tensor(ph, dtype=torch.long)))
            print(f"  {sp_name}: loaded {len(d['features'])} cached clips")
            continue
        # Cache miss; build it.
        print(f"  {sp_name}: extracting features (one-time)...")
        d = torch.load(sp_path, map_location="cpu", weights_only=False)
        frames_all = d["frames"]   # (N, T, 224, 224, 3) uint8
        phonemes_all = d["phonemes"]
        n = frames_all.shape[0]
        feats_list = []
        ph_list = []
        t0 = time.time()
        for i in range(n):
            if not phonemes_all[i]:
                continue
            mouth_88 = frames_to_mouth_88(frames_all[i].numpy())  # (T, 1, 88, 88)
            # Run ONNX one clip at a time
            x = mouth_88.transpose(1, 0, 2, 3)[None]  # (1, 1, T, 88, 88)
            y = sess.run(["features"], {"video": x.astype(np.float32)})[0]   # (1, T, 768)
            feats_list.append(y[0])
            ph_list.append(phonemes_all[i])
            if (i + 1) % 200 == 0:
                el = time.time() - t0
                print(f"    {i+1}/{n}  ({el:.0f}s, {(i+1)/el:.1f} clips/s)")
        # Save shard's features
        torch.save({"features": [torch.from_numpy(f) for f in feats_list],
                    "phonemes": ph_list},
                   cache_path)
        for feat, ph in zip(feats_list, ph_list):
            entries.append((torch.from_numpy(feat), torch.tensor(ph, dtype=torch.long)))
        print(f"    {sp_name}: extracted {len(feats_list)} clips in {time.time()-t0:.0f}s, cached")
    return entries


print("Building feature cache (one-time pass over all shards)...")
all_entries = build_or_load_feature_cache()
print(f"Total training clips: {len(all_entries)}")


# ---------------------------------------------------------------------------
# 7. Dataset + DataLoader
# ---------------------------------------------------------------------------
class FeatureDataset(Dataset):
    def __init__(self, entries):
        self.entries = entries
    def __len__(self): return len(self.entries)
    def __getitem__(self, i):
        feat, ph = self.entries[i]
        return feat, ph

def ctc_collate(batch):
    feats = torch.stack([b[0] for b in batch], dim=0)   # (B, T, 768)
    label_lengths = torch.tensor([len(b[1]) for b in batch], dtype=torch.long)
    max_lab = label_lengths.max().item()
    labels = torch.full((len(batch), max_lab), -1, dtype=torch.long)
    for i, b in enumerate(batch):
        labels[i, :len(b[1])] = b[1]
    return feats, labels, label_lengths

train_loader = DataLoader(FeatureDataset(all_entries), batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=2, collate_fn=ctc_collate,
                          drop_last=True, pin_memory=True)
print(f"Batches per epoch: {len(train_loader)}")


# ---------------------------------------------------------------------------
# 8. Phoneme head: LayerNorm -> BiLSTM -> Linear(VOCAB_SIZE)
# ---------------------------------------------------------------------------
class PhonemeHead(nn.Module):
    def __init__(self, in_dim=768, hidden=HEAD_DIM, layers=HEAD_LAYERS,
                 vocab=VOCAB_SIZE, dropout=HEAD_DROPOUT):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.rnn = nn.LSTM(in_dim, hidden, num_layers=layers,
                           bidirectional=True, dropout=dropout, batch_first=True)
        self.proj = nn.Linear(2 * hidden, vocab)
    def forward(self, x):
        # x: (B, T, 768)
        x = self.norm(x)
        x, _ = self.rnn(x)
        return self.proj(x)   # (B, T, VOCAB_SIZE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
head = PhonemeHead().to(device)
n_params = sum(p.numel() for p in head.parameters())
print(f"Head: {n_params/1e6:.1f}M params on {device}")


# ---------------------------------------------------------------------------
# 9. Training loop
# ---------------------------------------------------------------------------
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

optimizer = AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))

def lr_lambda(step):
    if step < WARMUP_STEPS:
        return step / max(1, WARMUP_STEPS)
    progress = (step - WARMUP_STEPS) / max(1, TOTAL_STEPS - WARMUP_STEPS)
    return 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))

scheduler = LambdaLR(optimizer, lr_lambda)
ctc_loss = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True, reduction="mean")

print("Training...")
step = 0
running_loss = 0.0
running_n = 0
t0 = time.time()
head.train()
while step < TOTAL_STEPS:
    for feats, labels, label_lengths in train_loader:
        if step >= TOTAL_STEPS:
            break
        feats = feats.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        label_lengths = label_lengths.to(device, non_blocking=True)

        logits = head(feats)   # (B, T, V)
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)   # (T, B, V)
        T_out = log_probs.shape[0]
        input_lengths = torch.full((logits.shape[0],), T_out, dtype=torch.long, device=device)
        loss = ctc_loss(log_probs, labels.clamp_min(0), input_lengths, label_lengths)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        running_loss += float(loss.item())
        running_n += 1
        step += 1

        if step % LOG_EVERY == 0:
            avg = running_loss / max(1, running_n)
            running_loss = 0.0; running_n = 0
            el = (time.time() - t0) / 60
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  step={step:5d}  loss={avg:.3f}  lr={lr_now:.2e}  t={el:.1f}min")

        if step % CKPT_EVERY == 0:
            ckpt_path = os.path.join(CKPT_DIR, f"head_step{step}.pt")
            torch.save(head.state_dict(), ckpt_path)
            print(f"  [ckpt] saved {ckpt_path}")

print(f"Training done. Final ckpt: {ckpt_path if 'ckpt_path' in dir() else '(no ckpt)'}")


# ---------------------------------------------------------------------------
# 10. Export head to ONNX
# ---------------------------------------------------------------------------
head.eval()
ONNX_HEAD = os.path.join(WORK_DIR, "avhubert_v3_phoneme_head.onnx")
dummy_feat = torch.randn(1, 50, 768, device=device, dtype=torch.float32)
torch.onnx.export(
    head,
    dummy_feat,
    ONNX_HEAD,
    input_names=["features"],
    output_names=["phoneme_logits"],
    opset_version=17,
    do_constant_folding=True,
    dynamic_axes={"features": {1: "T"}, "phoneme_logits": {1: "T_out"}},
    training=torch.onnx.TrainingMode.EVAL,
)
print(f"Head ONNX: {ONNX_HEAD}  ({os.path.getsize(ONNX_HEAD) / 1e6:.1f} MB)")

# Parity check
with torch.no_grad():
    pt_out = head(dummy_feat).cpu().numpy()
sess_head = ort.InferenceSession(ONNX_HEAD, providers=["CPUExecutionProvider"])
ort_out = sess_head.run(["phoneme_logits"], {"features": dummy_feat.cpu().numpy()})[0]
diff = np.abs(pt_out - ort_out)
print(f"Parity max_diff={diff.max():.6f}  mean_diff={diff.mean():.6f}")


# ---------------------------------------------------------------------------
# 11. Upload head ONNX (to a SEPARATE repo from the encoder, since
#     this artifact is trained-from-Liperty-data and shouldn't be
#     conflated with Meta's encoder weights)
# ---------------------------------------------------------------------------
if token:
    from huggingface_hub import create_repo
    create_repo(HEAD_REPO, repo_type="model", private=False, exist_ok=True)
    upload_file(
        path_or_fileobj=ONNX_HEAD,
        path_in_repo="avhubert_v3_phoneme_head.onnx",
        repo_id=HEAD_REPO,
        repo_type="model",
        commit_message=f"V3 phoneme head trained on GRID ({MAX_SPEAKERS} speakers, "
                       f"{TOTAL_STEPS} steps, base+vox+433h encoder)",
    )
    print(f"Uploaded -> https://huggingface.co/{HEAD_REPO}/blob/main/avhubert_v3_phoneme_head.onnx")
else:
    print("No HF_TOKEN; head ONNX is at " + ONNX_HEAD)
print("AV-HuBERT V3 phoneme head training complete.")
