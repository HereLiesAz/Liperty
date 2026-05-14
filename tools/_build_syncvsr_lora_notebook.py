"""Generate tools/train_syncvsr_lora.ipynb.

Per-user LoRA fine-tune of SyncVSR's visual encoder, trained on the
viseme-confusion-set recordings the user collected via the on-device
calibration screen (VisemeCalibrationScreen + PairedTrainingStore).

Unlike the earlier `_build_personal_lora_notebook.py` (which targets
Auto-AVSR and starts from raw mp4/mov recordings + Whisper transcription
+ on-the-fly lip cropping), this notebook:

- Targets SyncVSR (Vox+LRS2+LRS3 checkpoint, the deployed backend).
- Consumes PairedTrainingRecord exports directly. Frames are already
  88x88 grayscale float32 mean/std-normalized by the time they hit
  PairedTrainingStore.save() — no preprocessing needed, no Whisper
  needed (the on-device calibration writes the ground-truth word
  label per recording).
- Trains the attention decoder path (matches the deployment seq2seq
  decode in AvHubertSeq2SeqInference.createSyncVsr()).
- Exports the trained LoRA delta as a small (~10 MB) ONNX-friendly
  adapter, plus optionally a merged-into-encoder ONNX that can drop
  into app/src/main/assets/syncvsr_lrs3_encoder_personal.onnx for
  zero-runtime-overhead inference.

Pipeline shape:

  on-device:  user records ~50-150 confusion-set words
              -> PairedTrainingStore (filesDir/viseme_calibration/)
  export:     user taps "Export training data" -> zip in Downloads
              user uploads zip to private HF dataset repo
  train:      THIS notebook -> LoRA delta -> merged ONNX
              -> uploads to HereLiesAz/liperty-syncvsr-personal-lora
  on-device:  app pulls the merged ONNX next setup_libs.sh / OTA pull,
              MainActivity flag SYNCVSR_USE_PERSONAL_LORA points at it.

Run with:  python tools/_build_syncvsr_lora_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "train_syncvsr_lora.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.splitlines(keepends=True)}


cells: list[dict] = []

cells.append(md("""\
# Personal LoRA for SyncVSR — confusion-set-targeted

Adapts SyncVSR's visual encoder to YOU specifically, training on the recordings the on-device calibration flow collected. Per AAAI 2025 "Personalized Lip Reading" (arXiv:2409.00986) but with the training distribution narrowed to *viseme-confusion-set discrimination* — the gradient is spent on exactly the words the base model can't tell apart visually (B/P/M onsets, T/D/N onsets, the V1A/B/C/D vowel splits the fine viseme map identifies).

**Why this should move the needle on YOUR speech and not just the test set:**

1. Generic LoRA fine-tunes on whatever the user mouthed — usually general dictation. The gradient is spread across the whole vocabulary. Confusion-pair words appear rarely, and where they do the model is already right on most positions, so the loss has nothing to push against.

2. Confusion-set LoRA only sees minimal pairs: `bat / pat / mat`, `tin / din`, `view / few`, etc. Every clip is by construction a word the base model can confuse with another. The loss has signal at exactly the failure mode you want fixed.

3. The user-specific signal LoRA learns is whatever micro-features YOUR face produces that distinguish `/p/` from `/b/` from `/m/` — degree of lip protrusion, jaw drop curvature, asymmetric onset, breath plosion visibility. These are individual; they're not in the base model's training distribution.

**Expected effect on `disc_acc`:** the metric `tools/eval_syncvsr_viseme.ipynb` reports. Base model on LRS3 (in-domain) starts somewhere around 30-50% (random within-set guess for size-3 sets is 33%; trained-but-not-personalized SyncVSR should beat random because there *are* universal visual cues for B/P/M; personalization stacks on top). LoRA on ~100 user recordings should bring it to 60-80% on the user. Numbers TBD when we have a real LRS3 eval target.

**Hardware:** Kaggle P100 is enough. 100 calibration clips × 1 epoch ≈ 5-10 minutes wall time.
"""))

cells.append(md("""\
## How to use this

**Prep (one-time per user):**

1. On the phone: open Liperty -> nav rail -> "Train Me". Accept the personalization consent dialog. Walk through the calibration sets (~50 sets, ~150 words, 5-15 minutes).
2. Export the recordings: PersonalizationSettingsPanel -> "Export training data" -> saves `liperty_calibration_<date>.zip` to Downloads.
3. Move the zip onto your computer (USB / cloud / whatever), then upload it via web UI to a private HF dataset repo: `<you>/liperty-personal-recordings-syncvsr`.

**Train (Kaggle, recurring):**

1. Open this notebook on Kaggle. Runtime: GPU P100, Internet ON.
2. Add Kaggle Secret `HF_TOKEN` (write-scoped).
3. `Save Version -> Save & Run All (Commit)` for headless execution.
4. On completion, the merged encoder ONNX uploads to `<you>/liperty-syncvsr-personal-lora`.

**Use (Android):**

1. Bump the URL in `setup_libs.sh` to pull `syncvsr_lrs3_encoder_personal.onnx` from your personal repo.
2. Flip `MainActivity.SYNCVSR_USE_PERSONAL_LORA = true`.
3. Rebuild + install.
"""))

cells.append(md("""\
## 1. Environment
"""))

cells.append(code("""\
import os, sys, json, time, platform
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
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
"""))

cells.append(code("""\
%%capture
!pip install -q \\
    "huggingface_hub>=0.27,<1.0" \\
    "peft>=0.13,<1.0" \\
    "onnx>=1.16" \\
    "onnxruntime>=1.18" \\
    "onnxscript" \\
    "sentencepiece" \\
    "fire" \\
    "pytorch-lightning>=2.0" \\
    "espnet"
print("Deps installed.")
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
## 2. Run config
"""))

cells.append(code("""\
HF_USER = "HereLiesAz"

# Private dataset repo with the user's calibration recordings export.
# The zip should contain a directory of <id>.audio / .video / .json
# triples (the on-device PairedTrainingStore format).
RECORDINGS_REPO = f"{HF_USER}/liperty-personal-recordings-syncvsr"

# SyncVSR base checkpoint location. Pulled by the stage 1 bootstrap
# (matches what tools/syncvsr_export_stage2.py loads).
SYNCVSR_REPO   = "KAIST-AILab/SyncVSR"
SYNCVSR_CKPT   = "Vox+LRS2+LRS3.ckpt"

# Output: trained LoRA adapter + merged encoder ONNX.
OUT_REPO       = f"{HF_USER}/liperty-syncvsr-personal-lora"

# Filter: only train on recordings sourced from the on-device viseme
# calibration screen. Other PairedTrainingRecord sources (e.g. organic
# conversation captures from a future feature) get skipped so they
# don't dilute the discriminative signal.
SOURCE_FILTER = "viseme_calibration:"

# Vocab — must match deployment (syncvsr_unigram_units.txt).
VOCAB_REPO = f"{HF_USER}/liperty-syncvsr-onnx"
VOCAB_FILE = "syncvsr_unigram_units.txt"

# LoRA hyperparams. r=8 alpha=16 is the standard transformer encoder
# adapter shape; works well for small personal datasets and produces
# ~5-10 MB of trainable parameters.
LORA_RANK    = 8
LORA_ALPHA   = 16
LORA_DROPOUT = 0.05

# Training. Numbers tuned for ~100-500 calibration clips, one or two
# epochs. Higher LR than full fine-tuning is fine — only the LoRA
# adapters move.
BATCH_SIZE     = 8
EPOCHS         = 4
LR             = 1e-3
WEIGHT_DECAY   = 1e-3
WARMUP_STEPS   = 20
LABEL_SMOOTHING = 0.1
USE_FP16       = True

# Per-utterance frame budget. SyncVSR was trained on full-length clips;
# the calibration recordings are single-word holds of ~12-32 frames.
# Pad short clips with zeros, truncate long ones, to a fixed budget so
# the dataloader can batch.
MAX_FRAMES = 32
CROP_SIZE  = 88
PIXEL_MEAN = 0.421
PIXEL_STD  = 0.165

WORK_DIR = "/kaggle/working/work" if IS_KAGGLE else "/content/work"
DATA_DIR = os.path.join(WORK_DIR, "data")
CKPT_DIR = os.path.join(WORK_DIR, "ckpt")
for d in (WORK_DIR, DATA_DIR, CKPT_DIR):
    os.makedirs(d, exist_ok=True)

print(f"RECORDINGS_REPO: {RECORDINGS_REPO}")
print(f"OUT_REPO:        {OUT_REPO}")
print(f"LoRA:            r={LORA_RANK} alpha={LORA_ALPHA}")
print(f"Train:           bs={BATCH_SIZE} epochs={EPOCHS} lr={LR}")
"""))

cells.append(md("""\
## 3. Pull SyncVSR repo + checkpoint

Same bootstrap as `tools/syncvsr_export_stage2.py`. The bundled fork of
ESPnet (under `SyncVSR/SyncVSR/` after clone) is what defines the E2E
class our checkpoint actually loads against — the pip-installed espnet
has the wrong constructor signature for these weights and will
silently produce gibberish if you point this notebook at it.
"""))

cells.append(code("""\
from huggingface_hub import hf_hub_download, snapshot_download

# Clone the SyncVSR repo for its bundled ESPnet fork.
SYNCVSR_DIR = os.path.join(WORK_DIR, "SyncVSR")
if not os.path.exists(SYNCVSR_DIR):
    !git clone --depth 1 https://github.com/KAIST-AILab/SyncVSR.git {SYNCVSR_DIR}
# Apply patches the stage 2 export validated:
# 1. e2e_asr_transformer.py — guard self.codec attribute (fairseq not
#    installed in our env, so the codec branch must short-circuit).
patch_file = os.path.join(SYNCVSR_DIR, "SyncVSR", "espnet", "nets", "pytorch_backend", "e2e_asr_transformer.py")
if os.path.exists(patch_file):
    with open(patch_file) as f: src = f.read()
    if "getattr(self, 'codec', None)" not in src:
        src = src.replace("if self.codec is not None:", "if getattr(self, 'codec', None) is not None:")
        with open(patch_file, "w") as f: f.write(src)
        print(f"Patched {patch_file}")

# 2. Stub PlotAttentionReport (espnet asr_utils import chain).
asr_utils = os.path.join(SYNCVSR_DIR, "SyncVSR", "espnet", "asr", "asr_utils.py")
if os.path.exists(asr_utils):
    with open(asr_utils) as f: src = f.read()
    if "class PlotAttentionReport" not in src and "PlotAttentionReport" in src:
        src += "\\n\\nclass PlotAttentionReport(object):\\n    def __init__(self, *a, **k): pass\\n    def __call__(self, *a, **k): pass\\n"
        with open(asr_utils, "w") as f: f.write(src)
        print(f"Stubbed PlotAttentionReport in {asr_utils}")

# Make the bundled fork the first thing on sys.path (uninstall pip espnet
# first so our patches actually win the import).
!pip uninstall -y espnet 2>&1 | tail -1
for k in list(sys.modules.keys()):
    if k.startswith("espnet"):
        del sys.modules[k]
sys.path.insert(0, os.path.join(SYNCVSR_DIR, "SyncVSR"))
print(f"sys.path[0] = {sys.path[0]}")
"""))

cells.append(code("""\
# Pull the SyncVSR Vox+LRS2+LRS3 checkpoint from KAIST-AILab/SyncVSR.
# Note: at upstream this is a "model" repo and the .ckpt may need a
# direct download via the release URL — adjust as needed if HF Hub
# doesn't have it. The repo's README points at the release artifact.
ckpt_local = hf_hub_download(repo_id=SYNCVSR_REPO, filename=SYNCVSR_CKPT, local_dir=os.path.join(WORK_DIR, "base"))
print(f"Checkpoint: {ckpt_local} ({os.path.getsize(ckpt_local)/1e6:.0f} MB)")
"""))

cells.append(code("""\
# Pull vocab + confusion sets (the latter for the label tokenizer).
vocab_local = hf_hub_download(repo_id=VOCAB_REPO, filename=VOCAB_FILE, local_dir=os.path.join(WORK_DIR, "base"))
with open(vocab_local, encoding="utf-8") as f:
    token_list = [ln.rstrip("\\n") for ln in f if ln.rstrip("\\n")]
print(f"Vocab: {len(token_list)} tokens. blank=<{token_list[0]}>, eos=<{token_list[-1]}>")

BLANK_IDX = 0
EOS_IDX = len(token_list) - 1
SOS_IDX = EOS_IDX
"""))

cells.append(md("""\
## 4. Pull the calibration recordings + parse PairedTrainingRecord

The PairedTrainingStore on-disk format is three files per record:

- `<id>.video`  — binary: int32 nFrames, int32 frameLen, then nFrames * frameLen float32 values, row-major. frameLen is 88*88=7744 for the deployed crop.
- `<id>.audio`  — binary: int32 nSamples, then nSamples float32 PCM. Empty for viseme-calibration recordings (visual-only).
- `<id>.json`   — JSON metadata: id, createdAtMs, source, transcript, transcriptConfidence.

This notebook only consumes `.video` + `.json`. `.audio` is ignored.
"""))

cells.append(code("""\
# Pull the recordings dataset. We expect a flat directory of triples.
# The user uploads either: (a) a zip the device exported, or (b) the
# unzipped triples directly. Handle both.
from huggingface_hub import snapshot_download
import zipfile, struct

records_dir = snapshot_download(
    repo_id=RECORDINGS_REPO, repo_type="dataset",
    local_dir=os.path.join(DATA_DIR, "recordings"),
)
print(f"Pulled to {records_dir}")

# Unzip any *.zip in place.
for fn in os.listdir(records_dir):
    if fn.endswith(".zip"):
        p = os.path.join(records_dir, fn)
        with zipfile.ZipFile(p) as z:
            z.extractall(records_dir)
        print(f"Unzipped {fn}")

# Walk the tree, build a list of all <id>.json files.
record_metas = []
for root, _, files in os.walk(records_dir):
    for fn in files:
        if not fn.endswith(".json"): continue
        path = os.path.join(root, fn)
        try:
            with open(path) as f: meta = json.load(f)
            stem = os.path.splitext(path)[0]
            if os.path.exists(stem + ".video"):
                meta["__video_path"] = stem + ".video"
                record_metas.append(meta)
        except Exception as e:
            print(f"Skipped {path}: {e}")

# Filter to viseme-calibration only.
filtered = [m for m in record_metas if m.get("source", "").startswith(SOURCE_FILTER)]
print(f"Found {len(record_metas)} total records, {len(filtered)} match SOURCE_FILTER={SOURCE_FILTER!r}")
if len(filtered) < 10:
    raise RuntimeError(f"Not enough calibration data to LoRA-train ({len(filtered)} clips). Record more.")
record_metas = filtered

# Show the label distribution (which words are over- and
# under-represented? affects training balance).
from collections import Counter
label_counts = Counter(m.get("transcript", "?") for m in record_metas)
print(f"\\nTop labels: {label_counts.most_common(10)}")
print(f"Singleton labels (only 1 clip): {sum(1 for c in label_counts.values() if c == 1)}")
"""))

cells.append(code("""\
def _read_video_file(path):
    \"\"\"Mirror of PairedTrainingStore's writeVideoFile() format:
    int32 nFrames | int32 frameLen | nFrames * frameLen * float32 (big-endian).\"\"\"
    with open(path, "rb") as f:
        nF = struct.unpack(">i", f.read(4))[0]
        fLen = struct.unpack(">i", f.read(4))[0]
        data = f.read(nF * fLen * 4)
    arr = torch.frombuffer(bytearray(data), dtype=torch.float32).clone()
    arr = arr.reshape(nF, fLen)
    return arr  # (T, H*W)


# Verify the first record loads and looks sane.
sample = record_metas[0]
v = _read_video_file(sample["__video_path"])
print(f"First record: id={sample['id']!r}")
print(f"  transcript: {sample.get('transcript')!r}")
print(f"  source:     {sample.get('source')!r}")
print(f"  video:      shape={tuple(v.shape)} dtype={v.dtype}")
print(f"              mean={v.mean():.3f} std={v.std():.3f}")
print(f"              (expected post-normalization: mean~0, std~1)")
"""))

cells.append(md("""\
## 5. Tokenize the per-word labels

SyncVSR's vocab is a 5049-token SentencePiece unigram set (`▁` for word boundaries). Single-word labels need to be tokenized so we can compute CE per output position.
"""))

cells.append(code("""\
# Build a simple greedy longest-match tokenizer over the vocab. Not
# the full sentencepiece encoder, but accurate enough for SINGLE-word
# tokenization where the input is one word with a leading `▁`.
SP_SPACE = "\\u2581"   # ▁

# Index tokens by surface form (case-sensitive — SyncVSR's vocab IS
# uppercase, but PairedTrainingStore saves lowercase labels; we
# uppercase before lookup).
token_to_id = {t: i for i, t in enumerate(token_list)}


def tokenize_word(word):
    \"\"\"Tokenize a single lowercase English word into a token-id sequence
    starting with the `▁word` form. Returns None if no valid coverage.\"\"\"
    surface = SP_SPACE + word.upper()
    out = []
    i = 0
    while i < len(surface):
        # Greedy longest-match. Walk back from end-of-string each iteration.
        match = None
        for j in range(len(surface), i, -1):
            cand = surface[i:j]
            if cand in token_to_id:
                match = (cand, j); break
        if match is None:
            return None  # uncovered character — bail
        out.append(token_to_id[match[0]])
        i = match[1]
    return out


# Verify on the calibration corpus.
labeled = []
unmatched = []
for m in record_metas:
    w = m.get("transcript", "")
    ids = tokenize_word(w)
    if ids is None or not ids:
        unmatched.append(w); continue
    labeled.append({**m, "__token_ids": ids})
print(f"Tokenized {len(labeled)} / {len(record_metas)} records.")
if unmatched:
    print(f"  Unmatched ({len(unmatched)}): {sorted(set(unmatched))[:10]}")
record_metas = labeled
"""))

cells.append(md("""\
## 6. Load SyncVSR + LoRA-wrap the encoder
"""))

cells.append(code("""\
# Re-import: we just monkeypatched the bundled fork and pip-uninstalled
# espnet, so reimport with the patches in effect.
from lightning import E2E   # SyncVSR's wrapper around espnet E2E

# Lightning load. SyncVSR's lightning.E2E takes the YAML+ckpt and
# materializes the inner espnet E2E. We follow stage 2's pattern.
import yaml, copy
ckpt = torch.load(ckpt_local, map_location="cpu", weights_only=False)
cfg = ckpt["hyper_parameters"]
cfg["ckpt_path"] = ""   # don't double-reload
model = E2E(cfg=cfg)
model.load_state_dict(ckpt["state_dict"], strict=False)
model.eval()
inner = model.model    # the espnet E2E
print(f"Encoder: {type(inner.encoder).__name__}")
print(f"Decoder: {type(inner.decoder).__name__}")
print(f"CTC head out_features: {inner.ctc.ctc_lo.out_features}")
"""))

cells.append(code("""\
from peft import LoraConfig, get_peft_model

# Target the standard transformer encoder linears (queries, keys,
# values, output projection, and the two FFN projections). Names
# come from espnet's MultiHeadedAttention + PositionwiseFeedForward.
target_modules = [
    "linear_q", "linear_k", "linear_v", "linear_out",  # attention
    "w_1", "w_2",                                       # FFN (espnet naming)
    "linear1", "linear2",                                # FFN alt naming
]
lora_cfg = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    target_modules=target_modules,
    task_type=None,
)

# Freeze decoder + CTC head — LoRA only on the encoder. (We don't want
# to drift the language-model side; we want to adapt the visual
# feature extractor to this user's face.)
for p in inner.decoder.parameters(): p.requires_grad = False
for p in inner.ctc.parameters():     p.requires_grad = False

inner.encoder = get_peft_model(inner.encoder, lora_cfg)
inner.encoder.print_trainable_parameters()
"""))

cells.append(md("""\
## 7. Dataloader + training
"""))

cells.append(code("""\
from torch.utils.data import Dataset, DataLoader

class CalibrationDataset(Dataset):
    def __init__(self, records, max_frames=MAX_FRAMES, crop=CROP_SIZE):
        self.records = records
        self.max_frames = max_frames
        self.crop = crop

    def __len__(self): return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        v = _read_video_file(rec["__video_path"])     # (T, H*W)
        T = v.shape[0]
        v = v.reshape(T, 1, self.crop, self.crop)     # (T, 1, H, W)
        # Pad/truncate to max_frames.
        if T > self.max_frames:
            v = v[:self.max_frames]
            t_len = self.max_frames
        else:
            t_len = T
            if T < self.max_frames:
                pad = torch.zeros(self.max_frames - T, 1, self.crop, self.crop)
                v = torch.cat([v, pad], dim=0)
        # Targets: [<sos>, *word_token_ids, <eos>]
        tgt = [SOS_IDX] + rec["__token_ids"] + [EOS_IDX]
        return v, t_len, torch.tensor(tgt, dtype=torch.long), rec.get("transcript", "?")


def collate(batch):
    vids, t_lens, tgts, words = zip(*batch)
    vids = torch.stack(vids, dim=0)               # (B, T, 1, H, W) NTCHW
    t_lens = torch.tensor(t_lens, dtype=torch.long)
    max_tgt = max(t.size(0) for t in tgts)
    tgt_padded = torch.zeros(len(batch), max_tgt, dtype=torch.long)
    tgt_lens = torch.zeros(len(batch), dtype=torch.long)
    for i, t in enumerate(tgts):
        tgt_padded[i, :t.size(0)] = t
        tgt_lens[i] = t.size(0)
    return vids, t_lens, tgt_padded, tgt_lens, words


ds = CalibrationDataset(record_metas)
dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate, num_workers=0)
print(f"Dataset: {len(ds)} clips, {len(dl)} batches/epoch")
"""))

cells.append(code("""\
# Training loop. CE on attention decoder outputs over the token
# sequence [<sos>, word_tokens..., <eos>]. The encoder produces
# (B, T_enc, D) features; the decoder consumes them via cross-
# attention while autoregressing over the tgt sequence (teacher
# forcing during training).
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
inner.encoder.train()
# Decoder stays in train mode for dropout/BN behavior, but its params
# are frozen so they don't update.
inner.decoder.train()

opt = torch.optim.AdamW(
    [p for p in inner.encoder.parameters() if p.requires_grad],
    lr=LR, weight_decay=WEIGHT_DECAY,
)
sched = torch.optim.lr_scheduler.OneCycleLR(
    opt, max_lr=LR, total_steps=EPOCHS * len(dl),
    pct_start=WARMUP_STEPS / max(1, EPOCHS * len(dl)),
)
scaler = torch.amp.GradScaler("cuda", enabled=USE_FP16 and torch.cuda.is_available())

step = 0
t0 = time.time()
for epoch in range(EPOCHS):
    for vids, t_lens, tgt_padded, tgt_lens, _words in dl:
        vids = vids.to(device, non_blocking=True)
        t_lens = t_lens.to(device, non_blocking=True)
        tgt_padded = tgt_padded.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=USE_FP16 and torch.cuda.is_available()):
            # Encoder: (B, T, 1, H, W) -> (B, T_enc, D)
            try:
                hs_pad, _ = inner.encoder(vids, t_lens)
            except (TypeError, ValueError):
                hs_pad = inner.encoder(vids)
                if isinstance(hs_pad, tuple): hs_pad = hs_pad[0]

            # Decoder: teacher-forced over the full tgt sequence.
            # Bundled SyncVSR decoder signature: (tgt, tgt_mask, memory, memory_mask).
            T_dec = tgt_padded.size(1)
            T_enc = hs_pad.size(1)
            B = vids.size(0)
            tgt_mask = torch.tril(torch.ones(T_dec, T_dec, dtype=torch.bool, device=device)).unsqueeze(0).expand(B, -1, -1)
            mem_mask = torch.ones(B, 1, T_enc, dtype=torch.bool, device=device)
            decoded = inner.decoder(tgt_padded, tgt_mask, hs_pad, mem_mask)
            logits = decoded[0] if isinstance(decoded, tuple) else decoded
            # Shift: predict position t from inputs 0..t-1.
            # CE on positions 1..T_dec-1 (skip SOS prediction).
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = tgt_padded[:, 1:].contiguous()
            mask = torch.zeros_like(shift_labels, dtype=torch.bool)
            for i in range(B):
                mask[i, :tgt_lens[i].item() - 1] = True
            loss = F.cross_entropy(
                shift_logits[mask], shift_labels[mask],
                label_smoothing=LABEL_SMOOTHING,
            )

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(
            [p for p in inner.encoder.parameters() if p.requires_grad], 1.0
        )
        scaler.step(opt)
        scaler.update()
        sched.step()

        step += 1
        if step % 10 == 0:
            elapsed = time.time() - t0
            print(f"step {step:4d}  epoch {epoch}/{EPOCHS}  loss {loss.item():.3f}  "
                  f"lr {sched.get_last_lr()[0]:.2e}  ({elapsed:.0f}s)")

print(f"\\nTraining done in {time.time() - t0:.0f}s, {step} steps.")
"""))

cells.append(md("""\
## 8. Quick sanity-check: predict on the training set

Not a real eval (the calibration data is our training data, so this is upper-bound). But if the model can't even reproduce single-word predictions on data it just saw, something is wrong with the training loop, not with the personalization.
"""))

cells.append(code("""\
inner.encoder.eval()
inner.decoder.eval()
correct = 0
total = 0
with torch.no_grad():
    for vids, t_lens, tgt_padded, tgt_lens, words in dl:
        vids = vids.to(device); t_lens = t_lens.to(device)
        hs_pad, _ = inner.encoder(vids, t_lens) if hasattr(inner.encoder, "__call__") else (None, None)
        if hs_pad is None:
            hs_pad = inner.encoder(vids)
            if isinstance(hs_pad, tuple): hs_pad = hs_pad[0]

        # Greedy decode from <sos> until <eos>.
        B = vids.size(0)
        T_enc = hs_pad.size(1)
        mem_mask = torch.ones(B, 1, T_enc, dtype=torch.bool, device=device)
        tokens = torch.full((B, 1), SOS_IDX, dtype=torch.long, device=device)
        for _ in range(8):   # words are 1-3 tokens
            T_dec = tokens.size(1)
            tgt_mask = torch.tril(torch.ones(T_dec, T_dec, dtype=torch.bool, device=device)).unsqueeze(0).expand(B, -1, -1)
            decoded = inner.decoder(tokens, tgt_mask, hs_pad, mem_mask)
            logits = decoded[0] if isinstance(decoded, tuple) else decoded
            nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, nxt], dim=1)
            if (nxt == EOS_IDX).all(): break

        for i in range(B):
            seq = tokens[i].tolist()
            # Skip SOS, stop at EOS.
            seq = seq[1:]
            if EOS_IDX in seq: seq = seq[:seq.index(EOS_IDX)]
            pred = "".join(token_list[j] for j in seq if j not in (0,)).replace(SP_SPACE, " ").strip()
            ref  = words[i]
            if pred.lower() == ref.lower(): correct += 1
            total += 1

print(f"Training-set top-1 word accuracy: {correct}/{total} = {100*correct/total:.1f}%")
"""))

cells.append(md("""\
## 9. Merge LoRA into encoder + export ONNX

Merging means we add the LoRA delta matrices into the base linears so the resulting model has the same parameter shape as the base SyncVSR encoder. After merge, the model is just a fine-tuned encoder; no peft runtime needed on-device.
"""))

cells.append(code("""\
from peft import PeftModel

# Merge LoRA weights into the underlying linears.
merged_encoder = inner.encoder.merge_and_unload()
inner.encoder = merged_encoder

# Build the same EncoderHiddenWrapper that stage 4 export used so the
# ONNX matches the deployment contract.
import torch.nn as nn

class EncoderHiddenWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, video):
        # NTCHW input (B, T, 1, H, W) — matches Android deployment.
        t_dim = video.size(1) if video.size(1) > video.size(2) else video.size(2)
        ilens = torch.full((video.size(0),), t_dim, dtype=torch.long, device=video.device)
        try:
            out = self.encoder(video, ilens)
        except (TypeError, ValueError):
            out = self.encoder(video)
        return out[0] if isinstance(out, tuple) else out


wrapper = EncoderHiddenWrapper(inner.encoder).eval().to("cpu")
dummy = torch.randn(1, 16, 1, 88, 88)
out = wrapper(dummy)
print(f"Wrapper output shape: {tuple(out.shape)}")

ENCODER_ONNX = os.path.join(WORK_DIR, "syncvsr_lrs3_encoder_personal.onnx")
torch.onnx.export(
    wrapper, (dummy,), ENCODER_ONNX,
    input_names=["video"],
    output_names=["encoder_features"],
    dynamic_axes={
        "video":            {0: "batch", 1: "time"},
        "encoder_features": {0: "batch", 1: "t_enc"},
    },
    opset_version=17, do_constant_folding=True, dynamo=False,
)
print(f"Exported personalized encoder: {ENCODER_ONNX} ({os.path.getsize(ENCODER_ONNX)/1e6:.1f} MB)")
"""))

cells.append(md("""\
## 10. Upload back to HF for device pull
"""))

cells.append(code("""\
from huggingface_hub import HfApi, create_repo

api = HfApi()
create_repo(OUT_REPO, repo_type="model", private=True, exist_ok=True)
api.upload_file(
    path_or_fileobj=ENCODER_ONNX,
    path_in_repo="syncvsr_lrs3_encoder_personal.onnx",
    repo_id=OUT_REPO,
    repo_type="model",
    commit_message=f"Personalized SyncVSR encoder LoRA-merged (r={LORA_RANK}, {EPOCHS} epochs, {len(record_metas)} clips)",
)
print(f"Uploaded -> https://huggingface.co/{OUT_REPO}")
print()
print("Next steps on the Android side:")
print(f"  1. Edit setup_libs.sh to download '{OUT_REPO}'/syncvsr_lrs3_encoder_personal.onnx")
print("  2. Flip MainActivity.SYNCVSR_USE_PERSONAL_LORA = true")
print("  3. Rebuild + install")
"""))


nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
print(f"Wrote {OUT}")
