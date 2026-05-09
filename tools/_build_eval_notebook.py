"""Generate tools/eval_autoavsr.ipynb.

A Colab/Kaggle notebook that:
1. Pulls the exported Auto-AVSR ONNX + token list from
   `HereLiesAz/liperty-autoavsr-onnx` (the output of
   tools/export_autoavsr_to_onnx.ipynb).
2. Pulls a held-out shard set from one of the preprocessed dataset
   repos (`HereLiesAz/liperty-grid-preprocessed`,
   `HereLiesAz/liperty-tcd-preprocessed`) — same format the training
   notebooks consume, so eval matches deployment exactly.
3. For each clip: converts (T,H,W,C uint8 RGB 224) -> (T,1,88,88
   float32 grayscale, mean/std normalised to Chaplin's
   transforms.VideoTransform pipeline).
4. Runs the ONNX encoder+CTC head greedily, plus an optional CTC
   beam-search decode (mirrors SubwordCtcBeamDecoder.kt on Android).
5. Computes per-utterance and corpus WER + CER.
6. Prints a few sample (predicted, reference) pairs for sanity.

This is the offline counterpart to running the Android app: same ONNX
file, same vocab, same preprocessing math, same greedy/beam decoder.
Whatever WER this notebook reports is the floor for what the on-device
pipeline will deliver before the optional LlmTextCleaner pass.

Run with:  python tools/_build_eval_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "eval_autoavsr.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.splitlines(keepends=True)}


cells: list[dict] = []

cells.append(md("""\
# Evaluate `Amanvir/LRS3_V_WER19.1` (exported Liperty ONNX) on held-out shards

Same ONNX file Liperty Android loads, same vocab, same preprocessing math, same decoders. Whatever WER this notebook reports is the floor for the on-device pipeline before the optional `LlmTextCleaner` pass.

**What's measured:**
- Greedy CTC decode (default Android path until 2nd-stage decoder lands)
- CTC beam search (mirrors `SubwordCtcBeamDecoder.kt`, beam width 8)

**What's NOT measured here:**
- Chaplin's full beam search with attention scorer + external LM (gives the headline 19.1% number; not exportable to ONNX)
- The on-device LLM cleanup pass (LlmTextCleaner with Gemma-2B-it). That's an opt-in user-side feature.

So expect numbers in the 30-50% WER ballpark for greedy CTC. The gap vs 19.1% is the price of fitting on a phone.
"""))

cells.append(md("""\
## 1. Setup
"""))

cells.append(code("""\
import os, sys
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
    "onnxruntime>=1.18" \\
    "numpy>=1.24" \\
    "opencv-python-headless" \\
    "jiwer>=3.0" \\
    "sentencepiece"
print("Deps installed.")
"""))

cells.append(code("""\
WORK_DIR = "/kaggle/working/work" if IS_KAGGLE else "/content/work"
os.makedirs(WORK_DIR, exist_ok=True)
print(f"Work dir: {WORK_DIR}")
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

Pick the model repo (output of the export notebook), the shard dataset to eval against, and how many speakers/clips to score. Smaller numbers iterate faster.
"""))

cells.append(code("""\
HF_USER = "HereLiesAz"

# ONNX + vocab repo (output of tools/export_autoavsr_to_onnx.ipynb)
MODEL_REPO = f"{HF_USER}/liperty-autoavsr-onnx"

# Shard dataset to evaluate against. The .pt shards have:
#   frames:   (N, T, H, W, C) uint8, RGB 224x224 face crops
#   texts:    list[str] ground-truth transcripts
#   phonemes: list[str] (unused here; we eval against `texts`)
#   speaker:  int
EVAL_DATA_REPO = f"{HF_USER}/liperty-grid-preprocessed"

# Speakers / shards to evaluate. None -> all available.
SHARDS = None              # e.g. ["s1.pt", "s5.pt"] for a quick subset
MAX_CLIPS_PER_SHARD = None # e.g. 100 for a smoke test

# Decoders
RUN_GREEDY = True
RUN_BEAM = True
BEAM_WIDTH = 8

# Print this many (pred, ref) pairs at the end
NUM_SAMPLE_PAIRS = 12
"""))

cells.append(md("""\
## 3. Pull the exported ONNX + vocab
"""))

cells.append(code("""\
from huggingface_hub import snapshot_download

model_dir = snapshot_download(repo_id=MODEL_REPO, local_dir=os.path.join(WORK_DIR, "model"))
print("Model dir contents:")
for f in sorted(os.listdir(model_dir)):
    sz = os.path.getsize(os.path.join(model_dir, f)) / 1e6
    print(f"  {f}  ({sz:.1f} MB)")

ONNX_PATH = os.path.join(model_dir, "model.onnx")
VOCAB_PATH = os.path.join(model_dir, "token_list.txt")
assert os.path.exists(ONNX_PATH), f"No model.onnx in {model_dir}"
assert os.path.exists(VOCAB_PATH), f"No token_list.txt in {model_dir}"
"""))

cells.append(code("""\
with open(VOCAB_PATH, encoding="utf-8") as f:
    token_list = [ln.rstrip("\\n") for ln in f if ln.rstrip("\\n")]
print(f"Vocab size: {len(token_list)}")
print(f"  first 10: {token_list[:10]}")
print(f"  last 5:   {token_list[-5:]}")

# Conventional ESPnet ordering: index 0 is <blank>, last is <eos>
BLANK_IDX = 0
EOS_IDX = len(token_list) - 1
print(f"Blank idx: {BLANK_IDX} ({token_list[BLANK_IDX]!r})")
print(f"EOS idx:   {EOS_IDX} ({token_list[EOS_IDX]!r})")
"""))

cells.append(md("""\
## 4. Pull the eval shards
"""))

cells.append(code("""\
from huggingface_hub import HfApi, hf_hub_download

api = HfApi()
all_files = api.list_repo_files(EVAL_DATA_REPO, repo_type="dataset")
shard_names = sorted(f for f in all_files if f.endswith(".pt"))
print(f"Available shards in {EVAL_DATA_REPO}: {len(shard_names)}")
if not shard_names:
    raise RuntimeError(f"No .pt shards in {EVAL_DATA_REPO}")

if SHARDS is None:
    pick = shard_names
else:
    pick = [s for s in shard_names if s in SHARDS]
print(f"Will evaluate on: {pick}")

shard_dir = os.path.join(WORK_DIR, "shards")
os.makedirs(shard_dir, exist_ok=True)
local_shards = []
for name in pick:
    p = hf_hub_download(EVAL_DATA_REPO, filename=name, repo_type="dataset", local_dir=shard_dir)
    local_shards.append(p)
    print(f"  {name}  ({os.path.getsize(p) / 1e6:.0f} MB)")
"""))

cells.append(md("""\
## 5. Preprocess: 224 RGB face crop -> 88x88 grayscale, normalised

Mirrors `pipelines/data/transforms.py:VideoTransform` from Chaplin and the on-device path in `ImageUtils.alignAndCropMouth` + `VSRInference`. Constants:

- Center-crop the face crop to extract the mouth region
- Resize to 88x88
- Convert to grayscale
- Normalize: mean=0.421, std=0.165

The training pipeline does roughly this; the deployed Android pipeline uses the exact same constants in `MainActivity.AUTOAVSR_PIXEL_MEAN/STD` and `AUTOAVSR_CROP_SIZE`.
"""))

cells.append(code("""\
import numpy as np
import cv2

CROP = 88
MEAN = 0.421
STD = 0.165

def preprocess_clip(frames_uint8_rgb_224):
    \"\"\"
    frames_uint8_rgb_224: (T, 224, 224, 3) uint8 RGB face crops
    returns: (T, 1, CROP, CROP) float32, mean/std-normalised grayscale
    \"\"\"
    T = frames_uint8_rgb_224.shape[0]
    out = np.empty((T, 1, CROP, CROP), dtype=np.float32)
    for t in range(T):
        bgr = cv2.cvtColor(frames_uint8_rgb_224[t], cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)  # (224, 224) uint8
        # Lower-half center-crop to the mouth — face crops are face-centered,
        # so the mouth is roughly in the lower-middle. Crop a 112x112 region
        # there and downsample to 88x88. Approximates the on-device
        # alignAndCropMouth pipeline well enough for offline eval.
        h, w = gray.shape
        cx = w // 2
        cy = int(h * 0.66)  # mouth is ~2/3 down the face
        half = 56
        y1 = max(0, cy - half); y2 = min(h, cy + half)
        x1 = max(0, cx - half); x2 = min(w, cx + half)
        roi = gray[y1:y2, x1:x2]
        roi = cv2.resize(roi, (CROP, CROP), interpolation=cv2.INTER_AREA)
        out[t, 0] = (roi.astype(np.float32) / 255.0 - MEAN) / STD
    return out

# Smoke test on one clip
import torch as _torch
sample_shard = _torch.load(local_shards[0], map_location="cpu", weights_only=False)
sample_frames = sample_shard["frames"][0].numpy()
print(f"Sample clip raw shape: {sample_frames.shape} dtype={sample_frames.dtype}")
sample_pp = preprocess_clip(sample_frames)
print(f"Sample clip pp shape:  {sample_pp.shape} dtype={sample_pp.dtype}")
print(f"  mean={sample_pp.mean():.3f}  std={sample_pp.std():.3f} (expect roughly 0/1)")
print(f"Sample text: {sample_shard['texts'][0]!r}")
"""))

cells.append(md("""\
## 6. ONNX inference + decoders
"""))

cells.append(code("""\
import onnxruntime as ort
import numpy as np

# Use CUDAExecutionProvider when available; otherwise CPU.
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if ENV != "local" else ["CPUExecutionProvider"]
sess = ort.InferenceSession(ONNX_PATH, providers=providers)
print("Inputs:")
for i in sess.get_inputs():  print(f"  {i.name}: {i.shape}  {i.type}")
print("Outputs:")
for o in sess.get_outputs(): print(f"  {o.name}: {o.shape}  {o.type}")

# Determine input layout: NCTHW -> (1, 1, T, 88, 88), or NTCHW -> (1, T, 1, 88, 88).
# The export notebook produces NCTHW.
inp_name = sess.get_inputs()[0].name
inp_shape = sess.get_inputs()[0].shape
out_name = sess.get_outputs()[0].name

def to_input(pp_TCHW):
    # pp_TCHW: (T, 1, 88, 88)
    T = pp_TCHW.shape[0]
    if len(inp_shape) == 5 and inp_shape[1] == 1:
        # NCTHW: (1, 1, T, 88, 88)
        return pp_TCHW.transpose(1, 0, 2, 3)[None]   # (1, 1, T, 88, 88)
    elif len(inp_shape) == 5 and inp_shape[2] == 1:
        # NTCHW: (1, T, 1, 88, 88)
        return pp_TCHW[None]
    else:
        raise RuntimeError(f"Unrecognised ONNX input shape: {inp_shape}")

# Smoke test
y = sess.run([out_name], {inp_name: to_input(sample_pp)})[0]
print(f"Output shape: {y.shape}  (B, T_out, V)")
print(f"  argmax sample: {y[0, :8].argmax(-1)}")
"""))

cells.append(code("""\
def greedy_ctc_decode(logits_TV, blank=BLANK_IDX):
    \"\"\"logits_TV: (T_out, V) numpy. Returns list[int] of token indices.\"\"\"
    ids = logits_TV.argmax(-1)
    out = []
    prev = -1
    for i in ids:
        if i != prev and i != blank:
            out.append(int(i))
        prev = int(i)
    return out

def beam_ctc_decode(logits_TV, beam_width=BEAM_WIDTH, blank=BLANK_IDX):
    \"\"\"CTC beam search with prefix-merging via logsumexp. Mirrors the on-device
    SubwordCtcBeamDecoder.kt: beams are (prefix, lp_blank, lp_nonblank).
    Returns list[int] of token indices.\"\"\"
    T, V = logits_TV.shape
    # Convert logits to log-probs
    logp = logits_TV - logits_TV.max(-1, keepdims=True)
    logp = logp - np.log(np.exp(logp).sum(-1, keepdims=True))

    NEG_INF = -1e30
    # beams: dict prefix(tuple) -> (lp_blank, lp_nonblank)
    beams = {(): (0.0, NEG_INF)}

    def lse(a, b):
        if a == NEG_INF: return b
        if b == NEG_INF: return a
        m = max(a, b)
        return m + np.log(np.exp(a - m) + np.exp(b - m))

    for t in range(T):
        next_beams = {}
        for prefix, (lpb, lpnb) in beams.items():
            # Extend by blank
            ent = next_beams.get(prefix, (NEG_INF, NEG_INF))
            new_lpb = lse(ent[0], lse(lpb, lpnb) + logp[t, blank])
            next_beams[prefix] = (new_lpb, ent[1])

            # Extend by each non-blank token in top-K (keep search tractable)
            topk = np.argpartition(-logp[t], beam_width)[:beam_width]
            for s in topk:
                if s == blank: continue
                pl = float(logp[t, s])
                if prefix and prefix[-1] == s:
                    # Repeat: only allowed via blank path
                    new_prefix = prefix + (int(s),)
                    ent_n = next_beams.get(new_prefix, (NEG_INF, NEG_INF))
                    next_beams[new_prefix] = (ent_n[0], lse(ent_n[1], lpb + pl))
                    # Same prefix repeats merge into nonblank
                    ent_s = next_beams.get(prefix, (NEG_INF, NEG_INF))
                    next_beams[prefix] = (ent_s[0], lse(ent_s[1], lpnb + pl))
                else:
                    new_prefix = prefix + (int(s),)
                    ent_n = next_beams.get(new_prefix, (NEG_INF, NEG_INF))
                    next_beams[new_prefix] = (ent_n[0], lse(ent_n[1], lse(lpb, lpnb) + pl))
        # Prune
        scored = sorted(next_beams.items(), key=lambda kv: -lse(kv[1][0], kv[1][1]))
        beams = dict(scored[:beam_width])

    best = max(beams.items(), key=lambda kv: lse(kv[1][0], kv[1][1]))
    return list(best[0])

def ids_to_text(ids):
    \"\"\"Strip <eos>/<blank>, join SentencePiece subwords on the boundary marker U+2581.\"\"\"
    pieces = []
    for i in ids:
        if i == BLANK_IDX or i == EOS_IDX: continue
        pieces.append(token_list[i])
    text = "".join(pieces).replace("▁", " ").strip()
    return text
"""))

cells.append(md("""\
## 7. Run eval
"""))

cells.append(code("""\
import time
from collections import defaultdict

results = []     # list of dicts: {ref, greedy, beam, shard, idx}
t0 = time.time()
n_done = 0

for shard_path in local_shards:
    shard = _torch.load(shard_path, map_location="cpu", weights_only=False)
    frames_all = shard["frames"]   # (N, T, H, W, C)
    texts_all = shard["texts"]
    n_clips = len(texts_all)
    if MAX_CLIPS_PER_SHARD is not None:
        n_clips = min(n_clips, MAX_CLIPS_PER_SHARD)

    sname = os.path.basename(shard_path)
    print(f"\\n[{sname}] {n_clips} clips")

    for i in range(n_clips):
        ref = texts_all[i]
        try:
            pp = preprocess_clip(frames_all[i].numpy())
            y = sess.run([out_name], {inp_name: to_input(pp)})[0][0]   # (T_out, V)

            row = {"shard": sname, "idx": i, "ref": ref}
            if RUN_GREEDY:
                ids = greedy_ctc_decode(y)
                row["greedy"] = ids_to_text(ids)
            if RUN_BEAM:
                ids = beam_ctc_decode(y, beam_width=BEAM_WIDTH)
                row["beam"] = ids_to_text(ids)
            results.append(row)
            n_done += 1
            if n_done % 25 == 0:
                el = time.time() - t0
                print(f"  {n_done} clips  ({el:.0f}s, {n_done/el:.1f} clips/s)")
        except Exception as e:
            print(f"  [{sname}#{i}] failed: {e!r}")

print(f"\\nDone: {len(results)} clips in {time.time()-t0:.0f}s")
"""))

cells.append(md("""\
## 8. Score
"""))

cells.append(code("""\
import jiwer

def score(field):
    refs = [r["ref"] for r in results if field in r]
    hyps = [r[field] for r in results if field in r]
    if not refs:
        print(f"({field}: no rows)")
        return
    wer = jiwer.wer(refs, hyps) * 100.0
    cer = jiwer.cer(refs, hyps) * 100.0
    print(f"  {field:8s}  WER={wer:6.2f}%   CER={cer:6.2f}%   N={len(refs)}")

print("Corpus-level scores:")
if RUN_GREEDY: score("greedy")
if RUN_BEAM:   score("beam")
"""))

cells.append(code("""\
import random
print(f"\\nSample (pred, ref) pairs ({NUM_SAMPLE_PAIRS} of {len(results)}):\\n")
sample = random.sample(results, k=min(NUM_SAMPLE_PAIRS, len(results)))
for r in sample:
    print(f"  REF:    {r['ref']}")
    if "greedy" in r: print(f"  GREEDY: {r['greedy']}")
    if "beam" in r:   print(f"  BEAM:   {r['beam']}")
    print(f"  ({r['shard']}#{r['idx']})\\n")
"""))

cells.append(md("""\
## 9. Save scored results

Optional: persist a CSV so you can diff against later runs (e.g. after the LLM cleanup pass lands or after fine-tuning).
"""))

cells.append(code("""\
import csv
out_csv = os.path.join(WORK_DIR, "eval_results.csv")
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    fields = ["shard", "idx", "ref"]
    if RUN_GREEDY: fields.append("greedy")
    if RUN_BEAM:   fields.append("beam")
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in results:
        w.writerow({k: r.get(k, "") for k in fields})
print(f"Wrote {out_csv}  ({len(results)} rows)")
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
