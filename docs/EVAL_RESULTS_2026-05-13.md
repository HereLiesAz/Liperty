# SyncVSR viseme-confusion eval — first numbers, 2026-05-13

First end-to-end run of `tools/eval_syncvsr_viseme.ipynb` on Kaggle
(GPU P100). Captures what the pipeline does and what it tells us
(spoiler: less than we want; the eval target was wrong).

## Run config

- Notebook: `tools/eval_syncvsr_viseme.ipynb` @ 9d3efb1 (with FP16
  CTC ONNX workaround for the corrupted FP32 upload — see
  "Followups" below).
- Model bundle: `HereLiesAz/liperty-syncvsr-onnx`
  - `syncvsr_lrs3_visual_ctc_fp16.onnx` (387.8 MB; argmax-identical
    to the FP32 ONNX in prior offline testing).
  - `syncvsr_lrs3_encoder.onnx` (759.2 MB).
  - `syncvsr_lrs3_decoder.onnx` (273.4 MB).
- Eval data: `HereLiesAz/liperty-grid-preprocessed`, shard `s1.pt`,
  first 50 clips. NCTHW-after-stride detected at runtime as NTCHW.
- Backends: `ctc_greedy`, `ctc_beam` (width 8), `seq2seq` (greedy
  autoregressive, max 64 decode steps).

## Numbers

```
ctc_greedy   WER=100.00% CER=99.24% confusion: total=113 exact=0 swap=0 miss=113 disc_acc=nan% set_recall=0.00%
ctc_beam     WER=100.00% CER=99.24% confusion: total=113 exact=0 swap=0 miss=113 disc_acc=nan% set_recall=0.00%
seq2seq      WER=100.33% CER=97.13% confusion: total=113 exact=0 swap=0 miss=113 disc_acc=nan% set_recall=0.00%
```

100% WER, 97-99% CER, zero in-set hits, zero exact matches, zero
within-set swaps. The discriminative-accuracy metric is `nan` because
the model never landed in the same viseme bucket as a GRID reference
word — `exact_match + within_set_swap == 0` for all three decoders.

## Why the numbers are zero (and what they actually mean)

The model isn't broken. The eval target is.

**1. Frame count mismatch.** The GRID shards have
`frames[0].shape == [16, 224, 224, 3]` — preprocessed for the legacy
VideoMAE 16-frame window. SyncVSR was trained on full LRS3 utterances
(typically 30-100 frames / 1-4 seconds at 25 fps). At 16 frames
(640 ms) the encoder receives a clip 2-5× shorter than anything it
saw during training. Whatever feature trajectory it produces is way
out of distribution.

**2. Vocabulary mismatch.** GRID transcripts are restricted-grammar
commands:
```
'bin blue at f two now'
'bin blue at f three soon'
'bin blue at f four please'
'bin blue at f five again'
'bin blue at l six now'
'bin blue at l seven soon'
```
SyncVSR's training distribution (Vox+LRS2+LRS3) is open-vocabulary
natural English. Most GRID words (`bin`, `lay`, `place`, `set`,
single-letter codes, digit names) aren't even in the top-30k
common-English cutoff that `viseme_confusion_sets.json` was built
against, so they don't appear in `word_to_set` — every GRID word
counted as `off_set_miss` by definition, never `exact_match` or
`within_set_swap`.

**3. The model IS doing something visual.** Sample predictions show
the model produces coherent natural-English fragments, not noise:

| ref                          | ctc_beam     | seq2seq          |
| ---------------------------- | ------------ | ---------------- |
| bin blue at f two now        | PROVE        | ALL PERFECT      |
| bin blue at f three soon     | B            | BEDROOM          |
| bin blue at f four please    | (empty)      | PEOPLE           |
| bin blue at f five again     | BUT HAVE     | BUT WHATEVER     |
| bin blue at l six now        | BRILLIANT    | MORE THAN        |
| bin blue at l seven soon     | PRESSURE     | PRESSURE         |

Both backends converging on "PRESSURE" for "bin blue at l seven soon"
in particular suggests the visual signal IS being mapped to phonetic
content — "seven soon" and "pressure" share /s/ + sibilant + alveolar
articulation patterns. But the output vocabulary is SyncVSR's
training language, not GRID's command grammar, so WER goes nowhere.

**4. Seq2seq vs CTC qualitative split.** CTC emits short, often empty
strings; the seq2seq attention decoder emits longer, more
grammatically-coherent natural English. Same encoder pass, different
decoder behavior. This matches expectation: the attention decoder
converges to the training-distribution language model implicitly.
The discriminative-accuracy comparison is moot on this data because
both score 0/0 — but the QUALITATIVE difference (seq2seq produces
sentence-fragments, CTC produces word-or-empty) is real and
consistent across the 50-clip sample.

## What this run did NOT measure

- Whether SyncVSR is accurate on its training distribution.
- Whether seq2seq beats CTC on in-domain data.
- Whether the viseme confusion sets capture the right discrimination
  axes.
- Latency / throughput (GPU P100 ran all 50 clips with both backends
  in ~10 seconds; on-device Pixel 5 CPU will be the latency bottleneck
  and wasn't tested here).

None of those questions can be answered against GRID. They need an
in-domain test set.

## What's next

In order of immediate utility:

1. **Build a preprocessed LRS3 shard repo on HF.** LRS3 has
   open-vocabulary natural-English speech matching SyncVSR's training
   distribution AND has full natural-length utterances. Preprocessing
   needs face crop → 224×224 RGB → variable-length (T, 224, 224, 3)
   uint8 tensor per clip, with transcripts. Mirrors what
   `liperty-grid-preprocessed` does but without the 16-frame
   truncation. Academic-access dataset, requires LRS3 download form.
2. **Sanity-check current model on full-utterance GRID.** Quick
   middle-ground: the raw GRID dataset has 3-second utterances. If
   any HF mirror has GRID with full clips (not 16-frame windows),
   we'd get a "domain shift but at least with realistic utterance
   length" baseline. Lower priority than 1.
3. **Pivot to qualitative on-device test.** With these results
   showing the model produces natural-English output (not noise),
   the fastest path to "is this useful?" is point a Pixel at a real
   speaker mouthing real natural-English phrases and see what comes
   out. The eval notebook's job (give a number) won't be doable
   until 1 lands; on-device qualitative is the substitute.
4. **Re-export the FP32 SyncVSR CTC ONNX as a self-contained file.**
   The current `syncvsr_lrs3_visual_ctc.onnx` on HF is 2 MB of proto
   with external-data references but the `.onnx.data` file got
   filtered out by `tools/syncvsr_export_stage2.py`'s
   `upload_folder(allow_patterns=["*.onnx", "*.txt", "*.json"])`.
   FP16 works around it but the FP32 should be valid for the same
   reason — useful for accuracy-vs-FP16 comparison if ever needed.
   Fix: rerun stage 2 with `save_as_external_data=False` on the
   `torch.onnx.export` call, or do an `onnx.save_model` merge of
   the existing .onnx + .onnx.data files.
