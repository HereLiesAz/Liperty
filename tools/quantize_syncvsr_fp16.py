"""Convert syncvsr_lrs3_visual_ctc.onnx (FP32, 775 MB) to FP16.

Run locally:
    python tools/quantize_syncvsr_fp16.py

Inputs:
    app/src/main/assets/syncvsr_lrs3_visual_ctc.onnx  (FP32, 775 MB)

Outputs:
    /tmp/syncvsr_lrs3_visual_ctc_fp16.onnx            (FP16, ~390 MB)
    Parity numbers printed to stdout.

If the parity drift is tolerable (max diff < 0.5 on the encoder's
log-softmax output), the file is also copied into
app/src/main/assets/ so the Android build picks it up.

The goal is to halve inference latency on Pixel 5 (CPU XNNPACK
supports FP16 paths via packed weights) without hurting the CTC
decode. Keep input/output dtypes as FP32 so the Android side
doesn't need to convert.
"""
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import onnx
from onnxconverter_common import float16
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
FP32 = ROOT / "app" / "src" / "main" / "assets" / "syncvsr_lrs3_visual_ctc.onnx"
import tempfile
FP16 = Path(tempfile.gettempdir()) / "syncvsr_lrs3_visual_ctc_fp16.onnx"
DEST = ROOT / "app" / "src" / "main" / "assets" / "syncvsr_lrs3_visual_ctc_fp16.onnx"

assert FP32.exists(), f"Missing FP32 ONNX at {FP32}"
print(f"[quant] Loading FP32 from {FP32} ({FP32.stat().st_size/1e6:.1f} MB) ...")
t0 = time.time()
model_fp32 = onnx.load(str(FP32))
print(f"[quant]   loaded in {time.time()-t0:.1f}s")

print(f"[quant] Converting FP32 -> FP16 (keep_io_types=True so the Android")
print(f"[quant] side keeps writing/reading FP32 tensors -- the conversion")
print(f"[quant] inserts Cast nodes at the I/O boundary internally) ...")
t0 = time.time()
model_fp16 = float16.convert_float_to_float16(
    model_fp32,
    keep_io_types=True,
    disable_shape_infer=False,
)
print(f"[quant]   converted in {time.time()-t0:.1f}s")

print(f"[quant] Saving to {FP16} ...")
FP16.parent.mkdir(parents=True, exist_ok=True)
onnx.save(model_fp16, str(FP16))
print(f"[quant]   FP32 size: {FP32.stat().st_size/1e6:.1f} MB")
print(f"[quant]   FP16 size: {FP16.stat().st_size/1e6:.1f} MB")
print(f"[quant]   ratio: {FP16.stat().st_size/FP32.stat().st_size:.2%}")

# Parity check: run BOTH models on the same input, compare outputs.
print(f"[quant] Parity check (random dummy NTCHW input 1x16x1x88x88) ...")
np.random.seed(42)
x = np.random.randn(1, 16, 1, 88, 88).astype(np.float32)

# Time both for a rough speed indication on whatever this machine has.
def bench(path, label, runs=3):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    # Warmup
    out = sess.run(None, {inp_name: x})[0]
    times = []
    for _ in range(runs):
        t0 = time.time()
        out = sess.run(None, {inp_name: x})[0]
        times.append(time.time() - t0)
    print(f"[quant]   {label:8s}: shape={out.shape}, t/run={np.mean(times)*1000:.0f} ms")
    return out

out32 = bench(FP32, "FP32")
out16 = bench(FP16, "FP16")

diff = np.abs(out32 - out16)
print(f"[quant] PT-vs-FP16 max abs diff: {diff.max():.4f}, mean: {diff.mean():.6f}")

# CTC argmax stability: do the FP32 and FP16 models agree on which
# token wins at each timestep? That's what actually matters for
# greedy decode, not the exact log-prob magnitudes.
arg32 = out32.argmax(axis=-1)
arg16 = out16.argmax(axis=-1)
agree = (arg32 == arg16).mean()
print(f"[quant] argmax agreement: {agree:.2%}")

if diff.max() < 0.5 and agree > 0.95:
    print(f"[quant] Parity OK -- copying to {DEST}")
    shutil.copy(FP16, DEST)
    print(f"[quant] FP16 model ready for Android: {DEST}")
else:
    print(f"[quant] WARN: parity drift too large; not auto-copying.")
    print(f"[quant] Inspect {FP16} manually and decide.")
