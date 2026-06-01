# VSR Training & Evaluation Guide

How Liperty's Visual Speech Recognition models are trained, exported, and evaluated. All training runs **off-device** (Kaggle P100 / Colab); the Android app consumes the exported ONNX artifacts via `setup_libs.sh` (build time) or `ModelDownloadManager` (first launch).

The production backend is **SyncVSR** (visual-only, LRS3-trained), exported to ONNX. Notebooks live in `tools/` and are generated from `tools/_build_*.py` helpers — **edit the generator and regenerate** (`python tools/_build_*.py`), don't hand-edit the `.ipynb` (see CLAUDE.md pitfall on notebook corruption).

## Notebooks (resumable; Kaggle/Colab)

| Notebook | Purpose |
|---|---|
| `tools/export_syncvsr_to_onnx.ipynb` (+ `syncvsr_export_stage2.py`) | Export the SyncVSR encoder + CTC head to a self-contained ONNX, parity-check vs PyTorch, push to `HereLiesAz/liperty-syncvsr-onnx`. |
| `tools/train_syncvsr_lora.ipynb` | Per-user LoRA adaptation on top of a pretrained SyncVSR encoder. |
| `tools/train_landmark_lrs3_resumable.ipynb` | Landmark-only VSR training (uses `e1lephant/lrs3-landmark` shards). |
| `tools/train_grid_tcd_resumable.ipynb` | Pixel-baseline training (GRID + TCD-TIMIT). |
| `tools/eval_syncvsr_viseme.ipynb` | Offline WER/CER + viseme-confusion evaluation; mirrors deployment exactly (same ONNX, vocab, mean/std, decoders). |

## Dataset strategy

- **Pre-training / fine-tuning:** Oxford **LRS3 / LRS2** (sentence-level, open-vocabulary) — SyncVSR's training distribution. LRW for word-level density.
- **Evaluation:** must use an **in-domain, full-utterance** set (LRS3-matched). ⚠️ Do **not** evaluate on GRID 16-frame clips — that is out-of-distribution for SyncVSR (short clips + restricted-grammar vocab) and produced a 100% WER non-result (`EVAL_RESULTS_2026-05-13.md`). A real WER number is still pending an in-domain LRS3 test set (academic-access).

## Preprocessing (must match deployment)

Mouth ROI: MediaPipe landmarks → align/crop **88×88 grayscale** → per-channel normalize `(pixel/255 − 0.421) / 0.165`. These constants are shared by SyncVSR and Auto-AVSR and **must** match `MainActivity.AUTOAVSR_*` / `VSRInference` exactly, or deployment silently diverges from the trained model.

## Per-notebook Kaggle note

`HF_TOKEN` is **per-notebook** on Kaggle (not account-global) — toggle it ON in Add-ons → Secrets before running anything that pushes to / pulls from Hugging Face.
