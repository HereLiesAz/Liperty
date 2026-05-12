"""Generate tools/eval_syncvsr_viseme.ipynb.

A Colab/Kaggle notebook that measures **viseme-confusion accuracy** for
the deployed SyncVSR ONNX models. This is the metric the per-user LoRA
loop is supposed to move, and the offline baseline you need before
spending compute on training.

Conventional WER doesn't move much when a lipreading model gets a
confusion-set word wrong — the LM context recovers the meaning anyway,
and the WER edit distance is dominated by other errors. The actual
failure mode is "model emits 'bat' when speaker said 'pat'." This
notebook scores exactly that:

  per ground-truth word that's a member of a viseme confusion set
  (built by tools/build_viseme_confusion_sets.py):
    - exact-match     : model emitted the right word
    - within-set swap : model emitted a different member of the same set
    - off-set miss    : model emitted something outside the set
                        (or skipped — alignment gap)

  discriminative accuracy = exact_match / (exact_match + within_set_swap)
  set recall              = (exact_match + within_set_swap) / total

Discriminative accuracy is the LoRA target: it measures how often the
model picks the right member of a phonetically-ambiguous neighbourhood.
A random guesser scores ~ 1 / mean_set_size (typically ~25%); a perfect
discriminator scores 100%.

Both CTC (encoder + CTC head) and seq2seq (encoder hidden states +
attention decoder, greedy autoregressive) decode paths are evaluated
so we can A/B the two backends on the same metric.

Run with:  python tools/_build_eval_syncvsr_viseme_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "eval_syncvsr_viseme.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.splitlines(keepends=True)}


cells: list[dict] = []

cells.append(md("""\
# SyncVSR viseme-confusion accuracy eval

Measures **discriminative accuracy** on viseme-confusion sets — the metric Liperty's per-user LoRA training loop is designed to move.

For every ground-truth word that's a member of a viseme confusion set (see `tools/build_viseme_confusion_sets.py`), classifies the model's aligned prediction as one of:

- `exact_match` — model emitted the right word
- `within_set_swap` — model emitted a different member of the same viseme set (the failure mode LoRA fixes)
- `off_set_miss` — model emitted something not even in the confusion set (a different failure mode — usually LM-context-recoverable)

Reports `discriminative_accuracy = exact_match / (exact_match + within_set_swap)` for both decode paths:

1. **CTC** — `syncvsr_lrs3_visual_ctc.onnx` + subword CTC greedy / beam decode (mirrors `SubwordCtcBeamDecoder.kt`).
2. **Seq2Seq** — `syncvsr_lrs3_encoder.onnx` + `syncvsr_lrs3_decoder.onnx` + greedy autoregressive decode (mirrors `AvHubertSeq2SeqInference.createSyncVsr()`).
"""))

cells.append(md("## 1. Setup"))

cells.append(code("""\
import os, sys, json
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
    "onnxruntime-gpu>=1.18; sys_platform == 'linux'" \\
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

The `BACKENDS` list controls which decode paths to evaluate. Drop `'seq2seq'` if you only have the CTC ONNX uploaded; drop `'ctc'` if you want a pure seq2seq pass.
"""))

cells.append(code("""\
HF_USER = "HereLiesAz"

# Model bundle on HF — output of tools/syncvsr_export_stage2.py +
# tools/syncvsr_export_stage3_decoder.py + tools/syncvsr_export_stage4_encoder.py.
MODEL_REPO = f"{HF_USER}/liperty-syncvsr-onnx"

# Eval shards: preprocessed LRS3 in the Liperty pipeline format. Same
# convention as the training/eval notebooks expect:
#   frames: (N, T, 224, 224, 3) uint8 RGB
#   texts:  list[str] ground-truth transcripts
EVAL_DATA_REPO = "HereLiesAz/liperty-lrs3-preprocessed"   # adjust to whatever
                                                          # eval shard repo you have

# Which decode paths to evaluate.
#   'ctc'     - syncvsr_lrs3_visual_ctc.onnx + greedy / beam CTC decode
#   'seq2seq' - syncvsr_lrs3_encoder.onnx + syncvsr_lrs3_decoder.onnx + greedy AR
BACKENDS = ["ctc", "seq2seq"]

# Limit eval. None -> use everything.
SHARDS = None                # e.g. ["test_00.pt"] for a smoke test
MAX_CLIPS_PER_SHARD = None   # e.g. 50 for fast iteration

# CTC beam width (mirrors SubwordCtcBeamDecoder.kt default = 8).
BEAM_WIDTH = 8

# Max decode steps for seq2seq (mirrors AvHubertSeq2SeqInference.createSyncVsr).
MAX_DECODE_STEPS = 64

# Print this many sample (ref, ctc, seq2seq) triples at the end.
NUM_SAMPLE_PAIRS = 12
"""))

cells.append(md("""\
## 3. Pull the exported ONNX bundle + the confusion-set index

`viseme_confusion_sets.json` is built offline by `tools/build_viseme_confusion_sets.py` and committed to `app/src/main/assets/`. We pull it from the Liperty repo directly since it's not in the HF model bundle.
"""))

cells.append(code("""\
from huggingface_hub import snapshot_download

model_dir = snapshot_download(repo_id=MODEL_REPO, local_dir=os.path.join(WORK_DIR, "model"))
print("Model dir contents:")
for f in sorted(os.listdir(model_dir)):
    sz = os.path.getsize(os.path.join(model_dir, f)) / 1e6
    print(f"  {f}  ({sz:.1f} MB)")

CTC_ONNX = os.path.join(model_dir, "syncvsr_lrs3_visual_ctc.onnx")
ENC_ONNX = os.path.join(model_dir, "syncvsr_lrs3_encoder.onnx")
DEC_ONNX = os.path.join(model_dir, "syncvsr_lrs3_decoder.onnx")
VOCAB_PATH = os.path.join(model_dir, "syncvsr_unigram_units.txt")

if "ctc" in BACKENDS:
    assert os.path.exists(CTC_ONNX), f"Missing {CTC_ONNX} on HF"
if "seq2seq" in BACKENDS:
    assert os.path.exists(ENC_ONNX), f"Missing {ENC_ONNX} on HF"
    assert os.path.exists(DEC_ONNX), f"Missing {DEC_ONNX} on HF"
assert os.path.exists(VOCAB_PATH), f"Missing {VOCAB_PATH} on HF"
"""))

cells.append(code("""\
with open(VOCAB_PATH, encoding="utf-8") as f:
    token_list = [ln.rstrip("\\n") for ln in f if ln.rstrip("\\n")]
print(f"Vocab size: {len(token_list)}")
print(f"  first 5: {token_list[:5]}")
print(f"  last 5:  {token_list[-5:]}")

# ESPnet/SyncVSR convention: index 0 is <blank>, last is <eos>;
# <eos> doubles as <sos> for the seq2seq decoder.
BLANK_IDX = 0
EOS_IDX = len(token_list) - 1
SOS_IDX = EOS_IDX
print(f"Blank: {BLANK_IDX} ({token_list[BLANK_IDX]!r})  EOS/SOS: {EOS_IDX} ({token_list[EOS_IDX]!r})")
"""))

cells.append(code("""\
# Pull viseme_confusion_sets.json directly from the Liperty repo's
# main branch. (Doesn't ship in the HF model bundle; it's a Liperty
# code asset.)
import urllib.request
CONFUSION_URL = (
    "https://raw.githubusercontent.com/HereLiesAz/Liperty/main/"
    "app/src/main/assets/viseme_confusion_sets.json"
)
print(f"Fetching {CONFUSION_URL}...")
with urllib.request.urlopen(CONFUSION_URL) as r:
    confusion = json.loads(r.read().decode("utf-8"))
print(f"  metadata: {confusion['metadata']}")
print(f"  total sets: {len(confusion['sets'])}")

# Build word -> set_id (the set's index in the sorted list). A word
# can in principle belong to multiple sets after dedup steps, but the
# generator picks one canonical word per viseme bucket, so the index
# is many-to-one.
word_to_set: dict[str, int] = {}
for set_id, s in enumerate(confusion["sets"]):
    for w in s["words"]:
        word_to_set[w.lower()] = set_id
print(f"  unique words across all sets: {len(word_to_set):,}")
"""))

cells.append(md("""\
## 4. Pull eval shards
"""))

cells.append(code("""\
from huggingface_hub import HfApi, hf_hub_download

api = HfApi()
try:
    all_files = api.list_repo_files(EVAL_DATA_REPO, repo_type="dataset")
except Exception as e:
    print(f"Could not list {EVAL_DATA_REPO}: {e}")
    print(f"Set EVAL_DATA_REPO at the top of the notebook to your preprocessed LRS3 (or other) shard repo.")
    raise

shard_names = sorted(f for f in all_files if f.endswith(".pt"))
print(f"Available shards in {EVAL_DATA_REPO}: {len(shard_names)}")
if not shard_names:
    raise RuntimeError(f"No .pt shards in {EVAL_DATA_REPO}")

if SHARDS is None:
    pick = shard_names
else:
    pick = [s for s in shard_names if s in SHARDS]
print(f"Will evaluate on: {pick[:5]}{' ...' if len(pick)>5 else ''}  ({len(pick)} shards)")

shard_dir = os.path.join(WORK_DIR, "shards")
os.makedirs(shard_dir, exist_ok=True)
local_shards = []
for name in pick:
    p = hf_hub_download(EVAL_DATA_REPO, filename=name, repo_type="dataset", local_dir=shard_dir)
    local_shards.append(p)
print(f"  {len(local_shards)} shards local")
"""))

cells.append(md("""\
## 5. Preprocessing

Identical to the deployment path (`MainActivity.AUTOAVSR_PIXEL_MEAN/STD`, `AUTOAVSR_CROP_SIZE`, Rec.601 luma). The whole point of running this offline is to mirror what the phone actually sees.
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
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)  # (224, 224) uint8 Rec.601
        # Lower-half center-crop to the mouth (~2/3 down a centered face).
        h, w = gray.shape
        cx = w // 2
        cy = int(h * 0.66)
        half = 56
        y1 = max(0, cy - half); y2 = min(h, cy + half)
        x1 = max(0, cx - half); x2 = min(w, cx + half)
        roi = gray[y1:y2, x1:x2]
        roi = cv2.resize(roi, (CROP, CROP), interpolation=cv2.INTER_AREA)
        out[t, 0] = (roi.astype(np.float32) / 255.0 - MEAN) / STD
    return out
"""))

cells.append(md("""\
## 6. Inference sessions
"""))

cells.append(code("""\
import onnxruntime as ort
import numpy as np

providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
print(f"ORT providers: {providers}")

sess_ctc = None
sess_enc = None
sess_dec = None
ctc_inp_name = ctc_out_name = None
enc_inp_name = enc_out_name = None
dec_inp_tok = dec_inp_feat = dec_out_logits = None
ctc_layout = enc_layout = None

def detect_layout(shape):
    # shape is (1, ?, ?, 88, 88). Channels=1 axis tells us layout.
    if len(shape) != 5: raise RuntimeError(f"unexpected shape {shape}")
    if shape[1] == 1: return "NCTHW"
    if shape[2] == 1: return "NTCHW"
    raise RuntimeError(f"can't infer layout from {shape}")

if "ctc" in BACKENDS:
    sess_ctc = ort.InferenceSession(CTC_ONNX, providers=providers)
    ctc_inp_name = sess_ctc.get_inputs()[0].name
    ctc_out_name = sess_ctc.get_outputs()[0].name
    ctc_layout = detect_layout(sess_ctc.get_inputs()[0].shape)
    print(f"CTC layout: {ctc_layout}")

if "seq2seq" in BACKENDS:
    sess_enc = ort.InferenceSession(ENC_ONNX, providers=providers)
    sess_dec = ort.InferenceSession(DEC_ONNX, providers=providers)
    enc_inp_name = sess_enc.get_inputs()[0].name
    enc_out_name = sess_enc.get_outputs()[0].name
    enc_layout = detect_layout(sess_enc.get_inputs()[0].shape)
    print(f"Encoder layout: {enc_layout}")
    inp_names = [i.name for i in sess_dec.get_inputs()]
    dec_inp_tok = next(n for n in inp_names if "tok" in n)
    dec_inp_feat = next(n for n in inp_names if "encoder" in n or "feat" in n)
    dec_out_logits = sess_dec.get_outputs()[0].name
    print(f"Decoder inputs: tok='{dec_inp_tok}' feat='{dec_inp_feat}' out='{dec_out_logits}'")


def to_input(pp_TCHW, layout):
    # pp_TCHW: (T, 1, 88, 88)
    if layout == "NCTHW":
        return pp_TCHW.transpose(1, 0, 2, 3)[None]   # (1, 1, T, 88, 88)
    elif layout == "NTCHW":
        return pp_TCHW[None]                          # (1, T, 1, 88, 88)
    raise RuntimeError(f"bad layout {layout}")
"""))

cells.append(md("""\
## 7. Decoders
"""))

cells.append(code("""\
def greedy_ctc_decode(logits_TV, blank=BLANK_IDX):
    ids = logits_TV.argmax(-1)
    out = []
    prev = -1
    for i in ids:
        if i != prev and i != blank:
            out.append(int(i))
        prev = int(i)
    return out


def beam_ctc_decode(logits_TV, beam_width=BEAM_WIDTH, blank=BLANK_IDX):
    T, V = logits_TV.shape
    logp = logits_TV - logits_TV.max(-1, keepdims=True)
    logp = logp - np.log(np.exp(logp).sum(-1, keepdims=True))
    NEG_INF = -1e30
    beams = {(): (0.0, NEG_INF)}

    def lse(a, b):
        if a == NEG_INF: return b
        if b == NEG_INF: return a
        m = max(a, b)
        return m + np.log(np.exp(a - m) + np.exp(b - m))

    for t in range(T):
        next_beams = {}
        for prefix, (lpb, lpnb) in beams.items():
            ent = next_beams.get(prefix, (NEG_INF, NEG_INF))
            new_lpb = lse(ent[0], lse(lpb, lpnb) + logp[t, blank])
            next_beams[prefix] = (new_lpb, ent[1])
            topk = np.argpartition(-logp[t], beam_width)[:beam_width]
            for s in topk:
                if s == blank: continue
                pl = float(logp[t, s])
                if prefix and prefix[-1] == s:
                    new_prefix = prefix + (int(s),)
                    ent_n = next_beams.get(new_prefix, (NEG_INF, NEG_INF))
                    next_beams[new_prefix] = (ent_n[0], lse(ent_n[1], lpb + pl))
                    ent_s = next_beams.get(prefix, (NEG_INF, NEG_INF))
                    next_beams[prefix] = (ent_s[0], lse(ent_s[1], lpnb + pl))
                else:
                    new_prefix = prefix + (int(s),)
                    ent_n = next_beams.get(new_prefix, (NEG_INF, NEG_INF))
                    next_beams[new_prefix] = (ent_n[0], lse(ent_n[1], lse(lpb, lpnb) + pl))
        scored = sorted(next_beams.items(), key=lambda kv: -lse(kv[1][0], kv[1][1]))
        beams = dict(scored[:beam_width])

    best = max(beams.items(), key=lambda kv: lse(kv[1][0], kv[1][1]))
    return list(best[0])


def seq2seq_greedy_decode(enc_feat, enc_shape, max_steps=MAX_DECODE_STEPS):
    \"\"\"Mirrors Seq2SeqGreedyDecoder.kt: start at SOS, append argmax each step,
    stop on EOS or budget. Re-runs the full decoder each step (no KV cache).\"\"\"
    tokens = np.array([[SOS_IDX]], dtype=np.int64)
    feat = np.ascontiguousarray(enc_feat).reshape(enc_shape).astype(np.float32)
    out = [SOS_IDX]
    for _ in range(max_steps):
        feeds = {dec_inp_tok: tokens, dec_inp_feat: feat}
        logits = sess_dec.run([dec_out_logits], feeds)[0]   # (1, T_dec, V)
        nxt = int(logits[0, -1].argmax())
        out.append(nxt)
        tokens = np.concatenate([tokens, [[nxt]]], axis=1)
        if nxt == EOS_IDX:
            break
    return out


SPECIALS = {"<blank>", "<unk>", "<eos>", "<sos>", "<sos/eos>", "<pad>"}


def ids_to_text(ids):
    pieces = []
    for i in ids:
        if i == BLANK_IDX or i == EOS_IDX:
            continue
        sym = token_list[i] if 0 <= i < len(token_list) else ""
        if sym in SPECIALS or not sym:
            continue
        pieces.append(sym)
    return "".join(pieces).replace("▁", " ").strip()
"""))

cells.append(md("""\
## 8. Run

For each clip we run the full encoder pass once, then dispatch to whichever decode paths are in `BACKENDS`. CTC is computed from the same encoder/CTC ONNX; seq2seq runs the autoregressive decode against encoder hidden states.
"""))

cells.append(code("""\
import time

results = []   # list of dicts: {ref, ctc_greedy?, ctc_beam?, seq2seq?, shard, idx}
t0 = time.time()
n_done = 0

for shard_path in local_shards:
    shard = torch.load(shard_path, map_location="cpu", weights_only=False)
    frames_all = shard["frames"]
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
            row = {"shard": sname, "idx": i, "ref": ref}

            if "ctc" in BACKENDS:
                y = sess_ctc.run([ctc_out_name], {ctc_inp_name: to_input(pp, ctc_layout)})[0][0]
                row["ctc_greedy"] = ids_to_text(greedy_ctc_decode(y))
                row["ctc_beam"]   = ids_to_text(beam_ctc_decode(y, beam_width=BEAM_WIDTH))

            if "seq2seq" in BACKENDS:
                feat_out = sess_enc.run([enc_out_name], {enc_inp_name: to_input(pp, enc_layout)})
                feat = feat_out[0]    # (1, T_enc, D)
                ids = seq2seq_greedy_decode(feat, feat.shape)
                row["seq2seq"] = ids_to_text(ids)

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
## 9. Score

For each clip:
- compute WER / CER for context
- align ref-word-by-ref-word against the hypothesis using a Levenshtein-based word alignment
- for each ref word that's a confusion-set member, classify the aligned hyp word

The alignment uses `jiwer.process_words` which exposes the underlying edit operations (`equal | substitute | insert | delete`). For our purposes:
- `equal` on a confusion word -> exact_match
- `substitute` on a confusion word -> check whether the substitute is in the same set
- `delete` on a confusion word -> off_set_miss
- `insert` ops don't involve a ref-side confusion word, so ignored
"""))

cells.append(code("""\
import jiwer

def classify_confusion(ref_text: str, hyp_text: str):
    \"\"\"Returns dict with counters:
      total            : confusion-set words in ref
      exact_match
      within_set_swap
      off_set_miss
    \"\"\"
    out = dict(total=0, exact_match=0, within_set_swap=0, off_set_miss=0)
    if not ref_text:
        return out
    aligned = jiwer.process_words([ref_text], [hyp_text])
    # aligned.alignments[0] is a list of jiwer Alignment objects.
    ref_words = ref_text.lower().split()
    hyp_words = hyp_text.lower().split()
    for chunk in aligned.alignments[0]:
        op = chunk.type
        if op == "equal":
            for k in range(chunk.ref_end_idx - chunk.ref_start_idx):
                w = ref_words[chunk.ref_start_idx + k]
                if w in word_to_set:
                    out["total"] += 1
                    out["exact_match"] += 1
        elif op == "substitute":
            # Pair ref[r..] with hyp[h..] one-to-one within the chunk.
            ref_len = chunk.ref_end_idx - chunk.ref_start_idx
            hyp_len = chunk.hyp_end_idx - chunk.hyp_start_idx
            pair_n = min(ref_len, hyp_len)
            for k in range(pair_n):
                rw = ref_words[chunk.ref_start_idx + k]
                hw = hyp_words[chunk.hyp_start_idx + k]
                if rw in word_to_set:
                    out["total"] += 1
                    if word_to_set.get(hw) == word_to_set[rw]:
                        out["within_set_swap"] += 1
                    else:
                        out["off_set_miss"] += 1
            # Surplus ref words in the chunk: counted as off-set misses
            for k in range(pair_n, ref_len):
                rw = ref_words[chunk.ref_start_idx + k]
                if rw in word_to_set:
                    out["total"] += 1
                    out["off_set_miss"] += 1
        elif op == "delete":
            for k in range(chunk.ref_end_idx - chunk.ref_start_idx):
                w = ref_words[chunk.ref_start_idx + k]
                if w in word_to_set:
                    out["total"] += 1
                    out["off_set_miss"] += 1
        # "insert" ops have no ref-side word — ignored.
    return out


def score_field(field: str):
    refs = [r["ref"] for r in results if field in r]
    hyps = [r[field] for r in results if field in r]
    if not refs:
        print(f"  ({field}: no rows)")
        return
    wer = jiwer.wer(refs, hyps) * 100.0
    cer = jiwer.cer(refs, hyps) * 100.0
    agg = dict(total=0, exact_match=0, within_set_swap=0, off_set_miss=0)
    for ref, hyp in zip(refs, hyps):
        c = classify_confusion(ref, hyp)
        for k in agg: agg[k] += c[k]
    in_set = agg["exact_match"] + agg["within_set_swap"]
    disc_acc = (agg["exact_match"] / in_set * 100.0) if in_set else float("nan")
    set_recall = (in_set / agg["total"] * 100.0) if agg["total"] else float("nan")
    print(f"  {field:14s}  WER={wer:6.2f}%  CER={cer:6.2f}%  "
          f"confusion: total={agg['total']:5d}  exact={agg['exact_match']:5d}  "
          f"swap={agg['within_set_swap']:5d}  miss={agg['off_set_miss']:5d}  "
          f"disc_acc={disc_acc:6.2f}%  set_recall={set_recall:6.2f}%")

print("Corpus-level scores:")
for field in ("ctc_greedy", "ctc_beam", "seq2seq"):
    if any(field in r for r in results):
        score_field(field)
"""))

cells.append(md("""\
## 10. Failure inspection

Print sample clips where each backend got the confusion word wrong (within-set swap). These are the LoRA training targets.
"""))

cells.append(code("""\
def collect_swap_examples(field: str, k: int = 8):
    out = []
    for r in results:
        if field not in r: continue
        c = classify_confusion(r["ref"], r[field])
        if c["within_set_swap"] > 0:
            out.append(r)
        if len(out) >= k: break
    return out

for field in ("ctc_beam", "seq2seq"):
    if not any(field in r for r in results):
        continue
    print(f"\\n== {field}: example within-set swaps (these are exactly what LoRA must fix) ==")
    for r in collect_swap_examples(field, k=NUM_SAMPLE_PAIRS):
        print(f"  REF:  {r['ref']}")
        print(f"  HYP:  {r[field]}")
        print(f"  ({r['shard']}#{r['idx']})\\n")
"""))

cells.append(md("""\
## 11. Save scored results
"""))

cells.append(code("""\
import csv
out_csv = os.path.join(WORK_DIR, "eval_syncvsr_viseme_results.csv")
fields = ["shard", "idx", "ref", "ctc_greedy", "ctc_beam", "seq2seq"]
with open(out_csv, "w", newline="", encoding="utf-8") as f:
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
