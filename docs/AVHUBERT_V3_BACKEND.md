# AV-HuBERT (V3) Backend — Research Plan

This document captures the in-progress research path for swapping
Liperty's deployed visual-speech encoder from **Auto-AVSR (V2, ESPnet
visual-only Conformer + CTC, 19.1% headline WER on LRS3)** to
**AV-HuBERT large (V3, Meta's audio-visual self-supervised encoder,
trained on LRS3 + VoxCeleb2)**.

**Status: research-only. Not deployed. Not on a release timeline.**
The Auto-AVSR backend stays the production path until V3 demonstrates
a concrete WER improvement on real Liperty input on real hardware.

---

## Why bother

- **Bigger pretraining corpus.** AV-HuBERT large was self-supervised on
  LRS3 (~433h) + VoxCeleb2 (~2400h, English subset). Auto-AVSR's
  encoder saw only LRS3.
- **Better downstream WER on paper.** [LMD-VSR (ICCV 2023)](
  https://github.com/JeongHun0716/lmd-vsr) reports **12.6% WER** on
  English LRS2 test using AV-HuBERT large + an LMDecoder, vs Auto-AVSR's
  19.1% headline.
- **Public weights.** Both `base_vox_iter5.pt` and `large_vox_iter5.pt`
  are downloadable from `dl.fbaipublicfiles.com/avhubert/...` — no
  academic registration required to obtain the encoder, even though
  re-training would need LRS3 access.

## Why this is a research project, not a 1-week swap

1. **Stack mismatch.** AV-HuBERT runs on `fairseq`, Meta's research
   framework. Exporting fairseq transformers to ONNX is fragile —
   custom layer norm placements, dynamic shapes, AMP-only ops. Auto-AVSR's
   ESPnet export was already painful; AV-HuBERT is harder.

2. **The 12.6% WER assumes the LMDecoder runs at inference.** That's a
   second fairseq transformer trained on LRS2+LRS3 transcripts. Without
   it, the encoder alone is closer to 25-30% WER, which is no improvement
   over Auto-AVSR. Both encoder *and* decoder need to ONNX-export and
   ship on-device for V3 to beat V2.

3. **Architecture incompatibility with the existing pipeline.**
   - Auto-AVSR: 88×88 grayscale, `pixelMean=0.421, pixelStd=0.165`
   - AV-HuBERT: 96×96 grayscale, mouth ROI cropped via the AV-HuBERT
     mean-face alignment (see [`avhubert/preparation/align_mouth.py`](
     https://github.com/JeongHun0716/lmd-vsr/tree/main/avhubert/preparation))
   - `MainActivity.AUTOAVSR_*` constants and the `FrameBuffer` crop
     pipeline both need to change for V3.

4. **No way to validate WER without LRS2/LRS3 test data.** Both are
   academically gated. Until the user has access, V3 is flying blind on
   whether the expected WER actually materializes for Liperty's
   deployment context.

5. **License obligation.** The AV-HuBERT weights are licensed
   non-commercial research only. Liperty is a research/accessibility
   project, which qualifies, but the LICENSE file must ship with any
   redistribution and Liperty's own LICENSE/about-screen must
   acknowledge Meta as the upstream author.

---

## Pieces already in hand

| Piece | Where | Status |
|---|---|---|
| `large_vox_iter5.pt` (3.91 GB) | `HereLiesAz/liperty-avhubert-encoder` on HF (public mirror) | Done |
| Original Meta URL | https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/clean-pretrain/large_vox_iter5.pt | n/a |
| AV-HuBERT license | bundled in the HF mirror as `LICENSE.txt` | Done |
| Reference mouth-alignment pipeline | https://github.com/JeongHun0716/lmd-vsr/tree/main/avhubert/preparation (`align_mouth.py` + mean face) | external |
| LMDecoder (LRS2+LRS3 trained, English, 12.6 WER) | https://www.dropbox.com/scl/fo/zxnycpjlffd18ok5bg7ob/AKwd8lxvbx_q_BECGnTI2Pc | not mirrored yet |
| LMD-VSR full English VSR (LRS2 only, 23.8 WER) | https://www.dropbox.com/scl/fo/60xihdj518w44ujnixp8p/AKhdf0TxhPL5MLjQLtX8zdc | not mirrored yet |

## Pieces NOT yet in hand

- Working ONNX export of the AV-HuBERT encoder
- Working ONNX export of the LMDecoder
- Adapted Liperty preprocessing pipeline that produces 96×96 mouth ROIs
  via the AV-HuBERT mean-face alignment
- Android `AvHubertModelEngine.kt` implementing `ModelEngine`
- Eval pipeline that scores the V3 backend on a held-out set

---

## Attempt log

### 2026-05: first ONNX export attempt blocked at fairseq install

Ran `tools/kaggle_avhubert_export.py` against a Kaggle T4 session
with torch 2.10.0+cu128. The script cloned `av_hubert`,
`pip install --editable`'d its vendored fairseq submodule (commit
`afc77bdf4bb51453ce76f1572ef2ee6ddcda8eeb`), and tried to
`import avhubert.hubert`. Hit:

```
File "/kaggle/working/work/av_hubert/avhubert/hubert.py", line 16, in <module>
    from fairseq import utils
ImportError: cannot import name 'utils' from 'fairseq' (unknown location)
```

Probing `fairseq.__file__` after the editable install returned
`None`. That means Python loaded `fairseq` as a **PEP 420 namespace
package**, not as a regular package — `__init__.py` was either not
written or not located on the path. The vendored fairseq commit was
tested against torch 1.10ish and its setup machinery doesn't quite
finish on torch 2.10.

The encoder-trace risk (transformer with dynamic control flow not
ONNX-able) is downstream of this — we never even reached it.

**Verdict:** the warm-kernel `fetch+exec` pattern that works for
Liperty's other Kaggle scripts doesn't work here because the
existing kernel's torch 2.10 is incompatible with the vendored
fairseq commit. Future attempts need a **fresh Kaggle kernel with
deliberate torch downgrade** before installing fairseq:

```bash
# Required first step in any V3 export attempt:
pip install -q torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118
# Then: clone av_hubert, pip install --editable its fairseq, retry.
```

The mirror at `HereLiesAz/liperty-avhubert-encoder` and the export
artifacts (`tools/_build_export_avhubert_notebook.py`,
`tools/export_avhubert_to_onnx.ipynb`,
`tools/kaggle_avhubert_export.py`) are durable — only the *running*
of them is blocked.

## Next steps (in rough order, not committed to a schedule)

1. **Open `tools/export_avhubert_to_onnx.ipynb` in a fresh Kaggle
   kernel** (not the long-lived Liperty training session). Edit cell 2
   to prepend `pip install -q torch==2.0.1 ...` before the fairseq
   editable install. Run All. If fairseq imports cleanly, proceed.
2. **Trace the encoder.** This is the *next* risky step; if fairseq's
   transformer doesn't trace cleanly, V3 stops here until someone
   finds a workaround (e.g. via torch.export.export with dynamic
   shapes, or via a manual model-class reimplementation in plain
   PyTorch).

3. **Validate the exported ONNX matches PyTorch on a sample.** Same
   parity-check pattern as the Auto-AVSR export — feed a dummy video
   tensor through both and confirm max abs diff < 1e-3.

4. **Build a 96×96 mouth-crop pipeline.** Either adapt
   `ImageUtils.alignAndCropMouth` to use AV-HuBERT's mean face, or run
   the AV-HuBERT preparation pipeline's `align_mouth.py` on Liperty's
   incoming frames.

5. **Score the encoder + a simple CTC head** on held-out GRID. This
   gives a lower bound on V3 quality. If the CTC-only result is no
   better than V2, the LMDecoder must be exported too — back to step 2
   for the decoder.

6. **Decide on V3 viability based on real numbers.** No production swap
   without (a) proven WER improvement on Liperty-style input and
   (b) all required pieces (encoder + decoder + preprocessing) cleanly
   exporting to ONNX and running within the existing app's RAM/latency
   budget.

---

## What to do right now

Nothing. The mirror exists, the docs exist, the next step (ONNX
export) is a research notebook that needs a 4+ hour focused session
on Kaggle to attempt. Don't merge the V3 backend into `MainActivity`,
`OnnxModelEngine`, or `setup_libs.sh` until the export is proven.

If you want a small thing to ship between now and then, the legacy
TFLite phoneme path can be marked deprecated (it's only retained as a
fallback). Or ship the eval-vs-Auto-AVSR notebook properly.

## Attribution

> AV-HuBERT is licensed under the AV-HuBERT license, Copyright (c)
> Meta Platforms, Inc. All Rights Reserved.

Source repository: https://github.com/facebookresearch/av_hubert
Paper: "Learning Audio-Visual Speech Representation by Masked
Multimodal Cluster Prediction" (Shi et al., ICLR 2022).
