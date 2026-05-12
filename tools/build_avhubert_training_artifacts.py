"""Generate ONNX Runtime On-Device Training artifacts for the AV-HuBERT
visual encoder + a small trainable adapter head.

This is Step 3 of Liperty's on-device personalization plan (see
docs/AVHUBERT_V3_BACKEND.md "tenth attempt entry" when written). The
artifacts produced here ship in the Android APK; the per-user fine-tune
runs entirely on the phone using these artifacts and the user's own
paired (lip-video, transcript) recordings.

The first version uses an MSE-loss placeholder against a dummy target
just to validate the pipeline. Once that runs on Android, we replace the
loss with CTC against a real classifier head. Both LoRA-style adapter
parameterization and direct trainable-Linear are supported here; the
PoC uses a trainable Linear projection to keep the parameter count
small (and to avoid pulling in `peft` deps inside the docker image).

Workflow:
  Off-device (this script, inside the v3-export docker):
    1. Load AV-HuBERT base encoder PyTorch state dict
    2. Add a small trainable adapter (rank-r Linear bottleneck on top of
       the (T, 768) features → (T, ADAPTER_OUT))
    3. Export the combined module to ONNX
    4. Call onnxruntime.training.artifacts.generate_artifacts(...) with
       only the adapter params marked trainable
    5. Output the 4 artifacts to /work/out/training_artifacts/

  On-device (Android, future work):
    1. Bundle the 4 artifacts as APK assets
    2. Open a TrainingSession (onnxruntime-training-android AAR)
    3. Loop user's paired recordings through TrainStep + OptimizerStep
    4. Save resulting checkpoint as the user's personal adapter weights

Run inside the existing liperty-v3-export Docker image (it has the
fairseq + AV-HuBERT deps already). Volume-mount tools/ at /scripts:

    docker run --rm \\
        -v "$PWD/tools:/scripts:ro" \\
        -v "$PWD/lm:/work/out" \\
        -v "$HOME/liperty-v3-ckpts:/ckpts:ro" \\
        --entrypoint python liperty-v3-export \\
        /scripts/build_avhubert_training_artifacts.py
"""

import os
import sys
import time

# ---------------------------------------------------------------------------
# Env / paths (mirrors the export script)
# ---------------------------------------------------------------------------
WORK_DIR     = os.environ.get("V3_WORK_DIR",     "/work")
AVHUBERT_DIR = os.environ.get("V3_AVHUBERT_DIR", os.path.join(WORK_DIR, "av_hubert"))
FAIRSEQ_DIR  = os.environ.get("V3_FAIRSEQ_DIR",  os.path.join(AVHUBERT_DIR, "fairseq"))
LOCAL_PT     = os.environ.get("V3_CKPT_PATH",    "/ckpts/base_vox_433h.pt")
ARTIFACTS_DIR = os.environ.get(
    "V3_TRAINING_ARTIFACTS_DIR",
    os.path.join(WORK_DIR, "out", "avhubert_training_artifacts"),
)
ADAPTER_DIM = int(os.environ.get("V3_ADAPTER_DIM", "64"))

os.environ.setdefault("WANDB_MODE", "disabled")

# Make av_hubert + fairseq importable (see kaggle_avhubert_export_conda.py).
if FAIRSEQ_DIR not in sys.path:
    sys.path.insert(0, FAIRSEQ_DIR)
AVHUBERT_PKG_DIR = os.path.join(AVHUBERT_DIR, "avhubert")
if AVHUBERT_PKG_DIR not in sys.path:
    sys.path.insert(0, AVHUBERT_PKG_DIR)


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn

# Pre-import avhubert submodules so register_model fires before checkpoint load
import hubert_pretraining  # noqa: F401
import hubert              # noqa: F401
import hubert_asr          # noqa: F401

from fairseq import checkpoint_utils


# ---------------------------------------------------------------------------
# Load AV-HuBERT and strip to the visual encoder
# ---------------------------------------------------------------------------
import tempfile
empty_data_dir = tempfile.mkdtemp(prefix="empty_avhubert_data_")
print(f"Loading {LOCAL_PT} via fairseq...")
t0 = time.time()
models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
    [LOCAL_PT],
    arg_overrides={"data": empty_data_dir, "label_dir": empty_data_dir, "tokenizer_bpe_model": None},
)
print(f"  loaded in {time.time()-t0:.0f}s")
model = models[0].eval()


# Same FusedLayerNorm swap as the inference exporter (apex's CUDA-fused
# LayerNorm has no ONNX equivalent and silently emits zeros).
try:
    from apex.normalization import FusedLayerNorm as _ApexFLN
    def _swap(m):
        n = 0
        for nm, ch in list(m.named_children()):
            if isinstance(ch, _ApexFLN):
                stk = nn.LayerNorm(ch.normalized_shape, eps=ch.eps,
                                   elementwise_affine=ch.elementwise_affine)
                if ch.elementwise_affine:
                    stk.weight.data.copy_(ch.weight.data)
                    stk.bias.data.copy_(ch.bias.data)
                stk.to(next(ch.parameters()).device)
                setattr(m, nm, stk); n += 1
            else:
                n += _swap(ch)
        return n
    print(f"Swapped {_swap(model)} FusedLayerNorm modules.")
except ImportError:
    print("apex not installed; assuming stock LayerNorm everywhere.")


# Locate the AVHubertModel (encoder) — same dance as the export script.
if hasattr(model, "feature_extractor_audio"):
    avhm = model
elif hasattr(model, "encoder") and hasattr(model.encoder, "w2v_model") \
        and hasattr(model.encoder.w2v_model, "feature_extractor_audio"):
    avhm = model.encoder.w2v_model
elif hasattr(model, "encoder") and hasattr(model.encoder, "feature_extractor_audio"):
    avhm = model.encoder
else:
    raise RuntimeError(f"Cannot locate AVHubertModel inside {type(model).__name__}")
print(f"AVHubertModel found at: {type(avhm).__name__}")


# ---------------------------------------------------------------------------
# Wrap: encoder (frozen) + adapter (trainable)
# ---------------------------------------------------------------------------
class TrainableEncoder(nn.Module):
    """AV-HuBERT visual encoder + a small trainable adapter head.

    PoC: the adapter is a bottleneck Linear (768 -> ADAPTER_DIM -> 768).
    The point is to verify ORT's training-artifact generation accepts the
    full graph; once that works, the adapter can be replaced with LoRA
    modules inserted into the transformer's attention layers.
    """
    AUDIO_FEAT_DIM = 104

    def __init__(self, avhubert_model, adapter_dim=ADAPTER_DIM):
        super().__init__()
        self.feature_extractor_audio = avhubert_model.feature_extractor_audio
        self.feature_extractor_video = avhubert_model.feature_extractor_video
        self.layer_norm = avhubert_model.layer_norm
        self.post_extract_proj = avhubert_model.post_extract_proj
        self.encoder = avhubert_model.encoder
        # Trainable adapter — the only thing that gets gradients on-device.
        self.adapter_down = nn.Linear(768, adapter_dim)
        self.adapter_up = nn.Linear(adapter_dim, 768)
        # Zero-init the up-projection so the adapter starts as a no-op
        # (Hu et al. 2021 — the LoRA initialization).
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        # Freeze everything except the adapter.
        for p in self.feature_extractor_audio.parameters(): p.requires_grad = False
        for p in self.feature_extractor_video.parameters(): p.requires_grad = False
        for p in self.layer_norm.parameters(): p.requires_grad = False
        if self.post_extract_proj is not None:
            for p in self.post_extract_proj.parameters(): p.requires_grad = False
        for p in self.encoder.parameters(): p.requires_grad = False

    def forward(self, video):
        # video: (B, 1, T, 88, 88) NCTHW float32
        B = video.shape[0]
        T = video.shape[2]
        zero_audio = torch.zeros(B, self.AUDIO_FEAT_DIM, T,
                                 dtype=video.dtype, device=video.device)
        feat_a = self.feature_extractor_audio(zero_audio)
        feat_v = self.feature_extractor_video(video)
        feats_cat = torch.cat([feat_a, feat_v], dim=1)
        feats_t = feats_cat.transpose(1, 2)
        feats_ln = self.layer_norm(feats_t)
        if self.post_extract_proj is not None:
            feats_proj = self.post_extract_proj(feats_ln)
        else:
            feats_proj = feats_ln
        x, _ = self.encoder(feats_proj, padding_mask=None, layer=None)
        # x: (B, T_out, 768)
        x = x + self.adapter_up(self.adapter_down(x))
        return x


wrapper = TrainableEncoder(avhm).eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wrapper = wrapper.to(device)


# ---------------------------------------------------------------------------
# Export the combined module to ONNX
# ---------------------------------------------------------------------------
ONNX_PATH = os.path.join(WORK_DIR, "out", "avhubert_trainable_encoder.onnx")
os.makedirs(os.path.dirname(ONNX_PATH), exist_ok=True)

T_DUMMY = 50
dummy = torch.randn(1, 1, T_DUMMY, 88, 88, dtype=torch.float32, device=device)
print(f"Exporting trainable encoder ONNX -> {ONNX_PATH}")
torch.onnx.export(
    wrapper,
    dummy,
    ONNX_PATH,
    input_names=["video"],
    output_names=["features"],
    opset_version=17,
    do_constant_folding=False,    # IMPORTANT: don't fold the adapter weights away
    training=torch.onnx.TrainingMode.TRAINING,  # keep gradient-relevant nodes
    dynamic_axes={"video": {0: "B", 2: "T"}, "features": {0: "B", 1: "T_out"}},
)
sz = os.path.getsize(ONNX_PATH) / 1e6
print(f"  ONNX size: {sz:.1f} MB")


# ---------------------------------------------------------------------------
# Generate ORT On-Device Training artifacts
# ---------------------------------------------------------------------------
print("Generating training artifacts via onnxruntime.training.artifacts.generate_artifacts...")
from onnxruntime.training import artifacts

# Identify trainable parameters by name. After ONNX export, parameter
# names match PyTorch module attribute paths joined with `.`.
trainable_param_names = [
    "adapter_down.weight", "adapter_down.bias",
    "adapter_up.weight",   "adapter_up.bias",
]

# Frozen list: every other initializer in the ONNX. We let ORT discover
# them from the model itself — passing frozen_params explicitly here
# would require enumerating all ~150 weight tensors in the encoder,
# brittle. Instead use the API's "everything not in requires_grad is
# frozen by default" behavior by passing requires_grad only.

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
artifacts.generate_artifacts(
    ONNX_PATH,
    requires_grad=trainable_param_names,
    loss=artifacts.LossType.MSELoss,
    optimizer=artifacts.OptimType.AdamW,
    artifact_directory=ARTIFACTS_DIR,
    nominal_checkpoint=True,
)
print(f"Artifacts written to {ARTIFACTS_DIR}:")
for f in sorted(os.listdir(ARTIFACTS_DIR)):
    p = os.path.join(ARTIFACTS_DIR, f)
    print(f"  {f}  ({os.path.getsize(p)/1e6:.1f} MB)")
print("Done. Next: bundle these into the Android APK and open a TrainingSession.")
