"""In-session V1 phoneme training driver for Kaggle.

Cells 8-15 of train_grid_tcd_resumable.ipynb consolidated into a single
script that runs the full pipeline in the live kernel:

  - Pulls liperty-grid-preprocessed shards from HF Hub
  - Builds ShardDataset + ConcatDataset + DataLoader
  - Loads VideoMAE-base pretrained encoder + adapter + CTC head
  - Resumes from HF checkpoint if present, else starts fresh
  - Runs the time-budgeted training loop with periodic checkpointing
  - Final flush to HF on exit

Run from the Kaggle IPython console after kaggle_haar_fix.py has
already been exec'd (it sets up the per-shard preprocessing utilities
that this script's loader code doesn't need but kept for parity):

    import urllib.request
    exec(urllib.request.urlopen(
        'https://raw.githubusercontent.com/HereLiesAz/Liperty/main/tools/kaggle_grid_train.py'
    ).read())

Picks up these names from the kernel namespace (created by cells 4 + 5):
    Path, snapshot_download, hf_hub_download, upload_file, api,
    HfApi, HF_DATA_REPO_GRID, HF_DATA_REPO_TCD, HF_CKPT_REPO,
    DATA_DIR, CKPT_DIR, RUN_NAME, BATCH_SIZE, GRAD_ACCUM, LR,
    WEIGHT_DECAY, WARMUP_STEPS, NUM_WORKERS, USE_FP16, TIME_UPSAMPLE,
    NUM_FRAMES, IMG_SIZE, VOCAB_SIZE, BLANK_IDX, CKPT_EVERY_STEPS,
    LOG_EVERY_STEPS, KEEP_LAST_CKPTS, TIME_BUDGET_MIN,
    torch, nn, F, np, Dataset, DataLoader, ConcatDataset, shutil
"""

# ---------------------------------------------------------------------------
# Cell 8 — pull preprocessed shards
#
# Resource ceiling note: each preprocessed GRID shard at 224x224 uint8
# x 16 frames x ~1000 clips is ~2.4 GB. Kaggle's /kaggle/working is
# ~21 GB and the kernel container has ~33 GB RAM. Loading all 33
# speakers (~80 GB) blows both. MAX_SPEAKERS caps the subset; default
# 6 keeps total disk + RAM under ~15 GB which leaves headroom for
# checkpoints, the VideoMAE weights, and gradient buffers. Override
# via os.environ['LIPERTY_MAX_SPEAKERS'] before exec'ing this script
# (e.g. '0' for "all available", which is fine on a bigger box).
# ---------------------------------------------------------------------------
import os as _os
MAX_SPEAKERS = int(_os.environ.get("LIPERTY_MAX_SPEAKERS", "6"))


def pull_preprocessed():
    grid_dir = Path(DATA_DIR) / "grid"   # noqa: F821
    tcd_dir  = Path(DATA_DIR) / "tcd"
    grid_dir.mkdir(parents=True, exist_ok=True)
    tcd_dir.mkdir(parents=True, exist_ok=True)

    grid_files, tcd_files = [], []

    # If shards are already on disk (resumed session, manual upload, etc.),
    # use them as-is. Saves bandwidth and avoids re-fetching when disk is
    # near full. Override with LIPERTY_FORCE_DOWNLOAD=1 to force re-fetch.
    on_disk = sorted(grid_dir.glob("*.pt"))
    force = _os.environ.get("LIPERTY_FORCE_DOWNLOAD") == "1"
    if on_disk and not force:
        print(f"GRID: using {len(on_disk)} shard(s) already on disk; skipping download.")
        if MAX_SPEAKERS > 0:
            grid_files = on_disk[:MAX_SPEAKERS]
            print(f"      capped to first {len(grid_files)} (MAX_SPEAKERS={MAX_SPEAKERS}).")
        else:
            grid_files = on_disk
    else:
        # Decide which GRID shards to fetch.
        try:
            all_grid = sorted(api.list_repo_files(HF_DATA_REPO_GRID, repo_type="dataset"))   # noqa: F821
        except Exception as e:
            print(f"GRID list: {e}")
            all_grid = []
        grid_shards_remote = [f for f in all_grid if f.endswith(".pt")]
        if MAX_SPEAKERS > 0:
            wanted = grid_shards_remote[:MAX_SPEAKERS]
            print(f"GRID: subsetting {len(wanted)} of {len(grid_shards_remote)} shards "
                  f"(MAX_SPEAKERS={MAX_SPEAKERS}).")
        else:
            wanted = grid_shards_remote
            print(f"GRID: pulling all {len(wanted)} shards.")
        try:
            snapshot_download(repo_id=HF_DATA_REPO_GRID, repo_type="dataset",   # noqa: F821
                              local_dir=str(grid_dir), allow_patterns=wanted)
            grid_files = sorted(p for p in grid_dir.glob("*.pt") if p.name in wanted)
        except Exception as e:
            print(f"GRID pull: {e}")

    try:
        snapshot_download(repo_id=HF_DATA_REPO_TCD, repo_type="dataset",   # noqa: F821
                          local_dir=str(tcd_dir), allow_patterns="*.pt")
        tcd_files = sorted(tcd_dir.glob("*.pt"))
    except Exception as e:
        print(f"TCD pull: {e}")

    print(f"Local GRID shards: {len(grid_files)}")
    print(f"Local TCD shards:  {len(tcd_files)}")
    return grid_files, tcd_files


GRID_SHARDS, TCD_SHARDS = pull_preprocessed()


# ---------------------------------------------------------------------------
# Cell 9 — dataset + dataloader
# ---------------------------------------------------------------------------
class ShardDataset(Dataset):   # noqa: F821 — Dataset from kernel namespace
    def __init__(self, shard_paths):
        self.shards = []
        self.index = []
        self._loaded = [False] * len(shard_paths)
        self._paths = list(shard_paths)
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
        frames = d["frames"][c_idx]                                # (T, H, W, C) uint8
        phonemes = torch.tensor(d["phonemes"][c_idx], dtype=torch.long)
        return frames, phonemes


def ctc_collate(batch):
    frames = torch.stack([b[0] for b in batch], dim=0)             # (B, T, H, W, C) uint8
    labels = [b[1] for b in batch]
    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    max_len = max(label_lengths).item()
    padded = torch.full((len(labels), max_len), -1, dtype=torch.long)
    for i, l in enumerate(labels):
        padded[i, :len(l)] = l
    return frames, padded, label_lengths


ds_list = []
if GRID_SHARDS: ds_list.append(ShardDataset(GRID_SHARDS))
if TCD_SHARDS:  ds_list.append(ShardDataset(TCD_SHARDS))
assert ds_list, "No data available. Run preprocessing first."

train_ds = ConcatDataset(ds_list) if len(ds_list) > 1 else ds_list[0]   # noqa: F821
print(f"Total clips: {len(train_ds)}")

train_loader = DataLoader(   # noqa: F821
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, collate_fn=ctc_collate,
    pin_memory=True, drop_last=True, persistent_workers=NUM_WORKERS > 0,
)
print(f"Batches per epoch: {len(train_loader)}")


# ---------------------------------------------------------------------------
# Cell 10 — model
# ---------------------------------------------------------------------------
from transformers import VideoMAEModel


class LipertyVSR(nn.Module):   # noqa: F821
    def __init__(self, vocab_size=None, num_frames=None, image_size=None,
                 time_upsample=None, pretrained="MCG-NJU/videomae-base"):
        super().__init__()
        if vocab_size is None: vocab_size = VOCAB_SIZE   # noqa: F821
        if num_frames is None: num_frames = NUM_FRAMES
        if image_size is None: image_size = IMG_SIZE
        if time_upsample is None: time_upsample = TIME_UPSAMPLE
        self.encoder = VideoMAEModel.from_pretrained(
            pretrained, num_frames=num_frames, image_size=image_size, ignore_mismatched_sizes=True
        )
        h = self.encoder.config.hidden_size                         # 768
        self.adapter = nn.Sequential(
            nn.LayerNorm(h),
            nn.Linear(h, h),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.upsample = nn.ConvTranspose1d(h, h, kernel_size=time_upsample, stride=time_upsample)
        self.head = nn.Linear(h, vocab_size)
        self._t_tubelet = num_frames // 2
        self._n_spatial = (image_size // self.encoder.config.patch_size) ** 2

    def forward(self, pixel_values):
        if pixel_values.dim() == 5 and pixel_values.shape[-1] == 3:
            if pixel_values.dtype == torch.uint8:
                pixel_values = pixel_values.float() / 255.0
            pixel_values = pixel_values.permute(0, 1, 4, 2, 3).contiguous()
        out = self.encoder(pixel_values=pixel_values)
        feats = out.last_hidden_state
        B, P, H = feats.shape
        feats = feats.view(B, self._t_tubelet, self._n_spatial, H).mean(dim=2)
        feats = self.adapter(feats)
        feats = feats.transpose(1, 2)
        feats = self.upsample(feats)
        feats = feats.transpose(1, 2)
        logits = self.head(feats)
        return logits


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LipertyVSR().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {n_params/1e6:.1f}M params on {device}")


# ---------------------------------------------------------------------------
# Cell 11 — checkpoint utilities
# ---------------------------------------------------------------------------
import random as _random

CKPT_FILENAME = f"{RUN_NAME}-latest.pt"   # noqa: F821
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
    print(f"    [ckpt] step={step} uploaded to HF Hub.")


def prune_old_checkpoints(keep=None):
    if keep is None: keep = KEEP_LAST_CKPTS
    try:
        files = api.list_repo_files(HF_CKPT_REPO, repo_type="model")
    except Exception:
        return
    history = sorted(f for f in files if f.startswith(CKPT_HISTORY_PREFIX) and f.endswith(".pt"))
    to_delete = history[:-keep] if len(history) > keep else []
    if not to_delete:
        return
    for f in to_delete:
        try:
            api.delete_file(path_in_repo=f, repo_id=HF_CKPT_REPO, repo_type="model",
                            commit_message=f"Prune old checkpoint {f}")
        except Exception as e:
            print(f"    [prune] {f}: {e}")


def load_checkpoint_if_available(model, optimizer, scheduler, scaler):
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


# ---------------------------------------------------------------------------
# Cell 12 — optimizer + scheduler + resume
# ---------------------------------------------------------------------------
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def make_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))
    return LambdaLR(optimizer, lr_lambda)


TOTAL_STEPS = max(20000, len(train_loader) * 20)

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))
scheduler = make_scheduler(optimizer, WARMUP_STEPS, TOTAL_STEPS)
scaler = torch.amp.GradScaler("cuda") if (USE_FP16 and torch.cuda.is_available()) else None

start_step, start_epoch = load_checkpoint_if_available(model, optimizer, scheduler, scaler)
print(f"Starting from step={start_step}, epoch={start_epoch}")
print(f"Cosine total_steps={TOTAL_STEPS}; current LR={optimizer.param_groups[0]['lr']:.2e}")


# ---------------------------------------------------------------------------
# Cell 14 — training loop
# ---------------------------------------------------------------------------
import time


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
                if time.monotonic() - start_wall > deadline_s:
                    print(f"[budget] elapsed={(time.monotonic()-start_wall)/60:.1f} min "
                          f">= {budget_min}; stopping loop.")
                    return step, epoch

                frames, labels, label_lengths = batch
                frames = frames.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                label_lengths = label_lengths.to(device, non_blocking=True)

                if scaler is not None:
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        logits = model(frames)
                        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
                        T_out = log_probs.shape[0]
                        input_lengths = torch.full(
                            (logits.shape[0],), T_out, dtype=torch.long, device=device,
                        )
                        loss = ctc_loss(log_probs, labels.clamp_min(0), input_lengths, label_lengths)
                    loss_back = loss / GRAD_ACCUM
                    scaler.scale(loss_back).backward()
                else:
                    logits = model(frames)
                    log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
                    T_out = log_probs.shape[0]
                    input_lengths = torch.full(
                        (logits.shape[0],), T_out, dtype=torch.long, device=device,
                    )
                    loss = ctc_loss(log_probs, labels.clamp_min(0), input_lengths, label_lengths)
                    (loss / GRAD_ACCUM).backward()

                running_loss += float(loss.item())
                running_n += 1

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

                    if step % LOG_EVERY_STEPS == 0:
                        avg = running_loss / max(1, running_n)
                        running_loss = 0.0; running_n = 0
                        elapsed = (time.monotonic() - start_wall) / 60
                        lr_now = optimizer.param_groups[0]["lr"]
                        print(f"  step={step:6d} epoch={epoch} loss={avg:.3f} "
                              f"lr={lr_now:.2e} t={elapsed:.1f}min")

                    if step - last_ckpt_step >= CKPT_EVERY_STEPS:
                        save_checkpoint(model, optimizer, scheduler, scaler, step, epoch)
                        last_ckpt_step = step
                        prune_old_checkpoints(KEEP_LAST_CKPTS)

            epoch += 1
            print(f"[epoch] completed epoch {epoch}")
    except KeyboardInterrupt:
        print("KeyboardInterrupt - flushing final checkpoint...")
        return step, epoch


# ---------------------------------------------------------------------------
# Cell 15 — final flush wrapper
# ---------------------------------------------------------------------------
def run_training(budget_min=None):
    if budget_min is None: budget_min = TIME_BUDGET_MIN
    final_step, final_epoch = train_loop(start_step, start_epoch, budget_min)
    print(f"\nLoop finished. step={final_step} epoch={final_epoch}")
    save_checkpoint(model, optimizer, scheduler, scaler, final_step, final_epoch,
                    extra={"flush_reason": "end-of-session"})
    prune_old_checkpoints(KEEP_LAST_CKPTS)
    print(f"Final checkpoint pushed: step={final_step} epoch={final_epoch}")
    return final_step, final_epoch


print()
print("kaggle_grid_train loaded.")
print("Next: call run_training()  (uses TIME_BUDGET_MIN)")
print("  or: run_training(budget_min=240)  (4h subset)")
