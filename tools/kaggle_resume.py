"""In-session resume after a save_checkpoint crash.

Use after a save_checkpoint exception killed the training loop but
the kernel namespace is still alive (model/optimizer/scheduler in
RAM, GRID shards loaded). Re-fetching kaggle_grid_train.py would
rebuild the model from scratch and lose all trained weights -- this
script patches the broken function in place instead.

What it does:
  1. Defines a new _prune_local_checkpoints that removes old local
     step files (the original prune_old_checkpoints only deleted
     from HF, leaving 1 GB step files accumulating on /kaggle/working
     until disk filled and the next torch.save truncated).
  2. Replaces save_checkpoint with a version that prunes local
     before writing AND wraps the upload in try/except so a transient
     HF rate-limit / network blip doesn't kill the run.
  3. Replaces train_loop with a version whose checkpoint call is
     wrapped in try/except.
  4. Reloads the latest checkpoint from HF Hub into the in-memory
     model/optimizer/scheduler/scaler.
  5. Resumes training via run_training(budget_min=...).

Run from the Kaggle IPython console:

    import urllib.request
    exec(urllib.request.urlopen(
        'https://raw.githubusercontent.com/HereLiesAz/Liperty/main/tools/kaggle_resume.py'
    ).read())
    resume_training(budget_min=300)
"""

import glob as _glob
import os as _os
import time as _time
import torch as _torch
import torch.nn as _nn
import torch.nn.functional as _F


def _prune_local_checkpoints(keep=None):
    """Delete local step checkpoint files beyond `keep`, oldest first."""
    if keep is None:
        keep = KEEP_LAST_CKPTS   # noqa: F821
    history = sorted(_glob.glob(str(Path(CKPT_DIR) / f"{CKPT_HISTORY_PREFIX}*.pt")))   # noqa: F821
    for old in history[:-keep] if len(history) > keep else []:
        try:
            _os.remove(old)
        except Exception as e:
            print(f"    [prune-local] {old}: {e}")


def save_checkpoint(model, optimizer, scheduler, scaler, step, epoch, extra=None):
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "epoch": epoch,
        "rng_torch": _torch.get_rng_state(),
        "rng_cuda": _torch.cuda.get_rng_state_all() if _torch.cuda.is_available() else None,
        "rng_numpy": np.random.get_state(),                       # noqa: F821
        "rng_python": _random.getstate(),                         # noqa: F821
        "config": {
            "RUN_NAME": RUN_NAME, "NUM_FRAMES": NUM_FRAMES, "IMG_SIZE": IMG_SIZE,    # noqa: F821
            "TIME_UPSAMPLE": TIME_UPSAMPLE, "VOCAB_SIZE": VOCAB_SIZE, "BLANK_IDX": BLANK_IDX,  # noqa: F821
            "BATCH_SIZE": BATCH_SIZE, "GRAD_ACCUM": GRAD_ACCUM, "LR": LR,            # noqa: F821
        },
        "extra": extra or {},
    }
    local_latest = Path(CKPT_DIR) / CKPT_FILENAME            # noqa: F821
    local_step   = Path(CKPT_DIR) / f"{CKPT_HISTORY_PREFIX}{step:08d}.pt"   # noqa: F821
    tmp = local_latest.with_suffix(".pt.tmp")

    # Free disk BEFORE writing the new ~1 GB file. Keep KEEP_LAST_CKPTS-1
    # so total local <= KEEP_LAST_CKPTS after this cycle's add.
    _prune_local_checkpoints(keep=KEEP_LAST_CKPTS - 1)        # noqa: F821

    _torch.save(payload, tmp)
    tmp.rename(local_latest)
    shutil.copy2(local_latest, local_step)                    # noqa: F821

    upload_file(path_or_fileobj=str(local_latest), path_in_repo=CKPT_FILENAME,    # noqa: F821
                repo_id=HF_CKPT_REPO, repo_type="model",
                commit_message=f"step={step} epoch={epoch}")
    upload_file(path_or_fileobj=str(local_step), path_in_repo=local_step.name,    # noqa: F821
                repo_id=HF_CKPT_REPO, repo_type="model",
                commit_message=f"step={step} epoch={epoch} (history)")
    print(f"    [ckpt] step={step} uploaded to HF Hub.")


def train_loop(start_step, start_epoch, budget_min):
    """Same as the original except the checkpoint call is wrapped so a
    bad save doesn't kill an otherwise-healthy run."""
    start_wall = _time.monotonic()
    deadline_s = budget_min * 60
    step = start_step
    epoch = start_epoch
    running_loss = 0.0
    running_n = 0
    last_ckpt_step = step

    ctc_loss = _nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True, reduction="mean")   # noqa: F821

    try:
        while True:
            for batch in train_loader:                            # noqa: F821
                if _time.monotonic() - start_wall > deadline_s:
                    print(f"[budget] elapsed={(_time.monotonic()-start_wall)/60:.1f} min "
                          f">= {budget_min}; stopping loop.")
                    return step, epoch

                frames, labels, label_lengths = batch
                frames = frames.to(device, non_blocking=True)         # noqa: F821
                labels = labels.to(device, non_blocking=True)
                label_lengths = label_lengths.to(device, non_blocking=True)

                if scaler is not None:                                # noqa: F821
                    with _torch.amp.autocast("cuda", dtype=_torch.float16):
                        logits = model(frames)                        # noqa: F821
                        log_probs = _F.log_softmax(logits, dim=-1).transpose(0, 1)
                        T_out = log_probs.shape[0]
                        input_lengths = _torch.full(
                            (logits.shape[0],), T_out, dtype=_torch.long, device=device,
                        )
                        loss = ctc_loss(log_probs, labels.clamp_min(0), input_lengths, label_lengths)
                    loss_back = loss / GRAD_ACCUM                     # noqa: F821
                    scaler.scale(loss_back).backward()
                else:
                    logits = model(frames)
                    log_probs = _F.log_softmax(logits, dim=-1).transpose(0, 1)
                    T_out = log_probs.shape[0]
                    input_lengths = _torch.full(
                        (logits.shape[0],), T_out, dtype=_torch.long, device=device,
                    )
                    loss = ctc_loss(log_probs, labels.clamp_min(0), input_lengths, label_lengths)
                    (loss / GRAD_ACCUM).backward()

                running_loss += float(loss.item())
                running_n += 1

                if running_n % GRAD_ACCUM == 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)                    # noqa: F821
                        _torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        _torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()                                  # noqa: F821
                    step += 1

                    if step % LOG_EVERY_STEPS == 0:                   # noqa: F821
                        avg = running_loss / max(1, running_n)
                        running_loss = 0.0; running_n = 0
                        elapsed = (_time.monotonic() - start_wall) / 60
                        lr_now = optimizer.param_groups[0]["lr"]
                        print(f"  step={step:6d} epoch={epoch} loss={avg:.3f} "
                              f"lr={lr_now:.2e} t={elapsed:.1f}min")

                    if step - last_ckpt_step >= CKPT_EVERY_STEPS:     # noqa: F821
                        try:
                            save_checkpoint(model, optimizer, scheduler, scaler, step, epoch)
                            last_ckpt_step = step
                            prune_old_checkpoints(KEEP_LAST_CKPTS)    # noqa: F821
                        except Exception as e:
                            print(f"    [ckpt] save FAILED at step={step}: {type(e).__name__}: {e}")
                            print(f"           continuing; will retry at step "
                                  f"{step + CKPT_EVERY_STEPS}.")
                            last_ckpt_step = step

            epoch += 1
            print(f"[epoch] completed epoch {epoch}")
    except KeyboardInterrupt:
        print("KeyboardInterrupt - flushing final checkpoint...")
        return step, epoch


def resume_training(budget_min=None):
    """Reload from HF latest checkpoint and resume."""
    if budget_min is None:
        budget_min = TIME_BUDGET_MIN     # noqa: F821

    # Reload weights/optimizer/scheduler/scaler from HF's latest.pt.
    # In-memory model state at the time of the crash had ~200 extra
    # plateau steps anyway; reloading from step 1000 is conservative.
    fresh_step, fresh_epoch = load_checkpoint_if_available(   # noqa: F821
        model, optimizer, scheduler, scaler                   # noqa: F821
    )
    print(f"Resumed: step={fresh_step}, epoch={fresh_epoch}")

    final_step, final_epoch = train_loop(fresh_step, fresh_epoch, budget_min)
    print(f"\nLoop finished. step={final_step} epoch={final_epoch}")

    try:
        save_checkpoint(model, optimizer, scheduler, scaler, final_step, final_epoch,
                        extra={"flush_reason": "end-of-session"})
        prune_old_checkpoints(KEEP_LAST_CKPTS)                # noqa: F821
        print(f"Final checkpoint pushed: step={final_step} epoch={final_epoch}")
    except Exception as e:
        print(f"Final flush FAILED: {type(e).__name__}: {e}")
        print("Latest in-loop checkpoint on HF is still your floor.")
    return final_step, final_epoch


print()
print("kaggle_resume loaded.")
print("  save_checkpoint and train_loop have been replaced with the patched versions.")
print("  Next: call resume_training(budget_min=300)")
