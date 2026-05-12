"""Generate ONNX Runtime On-Device Training artifacts for the AV-HuBERT
visual encoder, on Kaggle (no fairseq, no Docker required).

Sidesteps the fairseq dep stack entirely by working directly on the
already-published encoder ONNX from HereLiesAz/liperty-avhubert-encoder.
A small trainable Linear adapter is appended to the encoder's output
via ONNX graph surgery; the result is fed to
onnxruntime.training.artifacts.generate_artifacts(), producing the 4
files needed for on-device gradient descent on Android.

The PoC ran successfully on Kaggle on 2026-05-12. Artifacts live at
HereLiesAz/liperty-v3-training-artifacts on Hugging Face.

USAGE (paste-driven Kaggle cell):
  1. New notebook (Kaggle's default Python 3.12 image is fine — we
     bootstrap a Python 3.10 venv via uv to get onnxruntime-training).
  2. Ensure HF_TOKEN is set as a notebook secret (Add-ons -> Secrets).
  3. Ensure Internet is ON in the right-panel Settings.
  4. Paste the entire contents of this file into one cell. Run.

Total run time: ~5-8 minutes (uv venv creation: 30s; pip install: 60s;
encoder pull: 30s; graph surgery: 5s; generate_artifacts: 10s; HF upload: 2-3 min).

5 issues fixed during the PoC iteration that are now baked into this
script:
  1. onnxruntime-training has no PyPI wheels for Python 3.12 (Kaggle's
     default) -- create a Python 3.10 venv via uv and install there.
  2. uv venvs ship without pip -- use `uv pip install --python ...`.
  3. onnxruntime.training has an import-time dep on torch -- install it
     alongside the other deps.
  4. The published encoder ONNX exposes 5 diagnostic outputs (features,
     feat_a, feat_v, feats_ln, feats_proj) from V3 parity debugging; only
     'features' should feed downstream layers -- prune the rest and pass
     loss_input_names=['adapted_features'] to generate_artifacts.
  5. onnxblock writes intermediate temp files with
     save_as_external_data=True hardcoded; for a ~413 MB model it
     produces a temp.onnx + missing temp.onnx.data, failing
     check_model. Monkey-patch onnx.save_model to force
     save_as_external_data=False process-wide.

Output (on HF):
  - training_model.onnx        forward pass + loss + gradient subgraph
  - eval_model.onnx            forward pass + loss (no gradient nodes)
  - optimizer_model.onnx       AdamW step graph (~538 B, only 2 params)
  - checkpoint                 initial trainable parameter values
  - nominal_checkpoint         slim per-param state for memory-constrained sessions
"""

import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# 1. Bootstrap a Python 3.10 venv via uv (Kaggle's image is Python 3.12,
#    onnxruntime-training PyPI wheels stop at 3.11)
# ---------------------------------------------------------------------------
print("[+] Bootstrapping Python 3.10 venv via uv ...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "uv"])

VENV = "/kaggle/working/venv310"
PY = f"{VENV}/bin/python"
if not os.path.exists(PY):
    subprocess.check_call(["uv", "venv", "--python", "3.10", VENV])

subprocess.check_call([
    "uv", "pip", "install",
    "--python", PY,
    "onnx", "onnxruntime-training", "torch",
    "huggingface_hub>=0.27,<1.0", "numpy",
])


# ---------------------------------------------------------------------------
# 2. HF token: pull from Kaggle secrets into env so subprocess inherits
# ---------------------------------------------------------------------------
if "HF_TOKEN" not in os.environ:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")


# ---------------------------------------------------------------------------
# 3. Write the build script and run it inside the venv
# ---------------------------------------------------------------------------
BUILD_PATH = "/kaggle/working/build_training_artifacts.py"

script = r'''
import os, sys
print("python:", sys.version.split()[0])

from huggingface_hub import hf_hub_download, login, create_repo, upload_folder

SOURCE_REPO = "HereLiesAz/liperty-avhubert-encoder"
SOURCE_FILE = "avhubert_base_vox_433h_visual_encoder.onnx"
TARGET_REPO = "HereLiesAz/liperty-v3-training-artifacts"

WORK = "/kaggle/working/v3-train-artifacts"
os.makedirs(WORK, exist_ok=True)
ARTIFACT_DIR = os.path.join(WORK, "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
print("Pulling encoder ONNX ...")
encoder_onnx_path = hf_hub_download(
    repo_id=SOURCE_REPO, repo_type="model",
    filename=SOURCE_FILE, local_dir=WORK,
)
print(f"  {os.path.getsize(encoder_onnx_path) / 1e6:.1f} MB")

import onnx
import numpy as np
from onnx import helper, numpy_helper, TensorProto

# CRITICAL: monkey-patch onnx.save_model BEFORE importing
# onnxruntime.training so the patched version is in use when the
# library writes its intermediate temp files. onnxblock writes those
# with save_as_external_data=True hardcoded, which produces a
# temp.onnx + temp.onnx.data pair; if the .data file goes missing the
# check_model validator fails. Forcing inline saves process-wide is
# safe because our model is 413 MB -- well under the 2 GB protobuf limit.
_orig_save_model = onnx.save_model

def _patched_save_model(proto, path, *args, **kwargs):
    kwargs["save_as_external_data"] = False
    for k in ("all_tensors_to_one_file", "location", "size_threshold", "convert_attribute"):
        kwargs.pop(k, None)
    return _orig_save_model(proto, path, *args, **kwargs)

onnx.save_model = _patched_save_model
onnx.save = _patched_save_model

print("STEP A: Loading encoder ONNX ...")
model = onnx.load(encoder_onnx_path, load_external_data=True)
graph = model.graph

# Drop diagnostic outputs left over from V3 parity debugging. Keeping
# them would feed the MSE loss too many inputs.
for o in [o for o in graph.output if o.name != "features"]:
    graph.output.remove(o)
print(f"  pruned outputs: {[o.name for o in graph.output]}")

print("STEP B: Adding trainable Linear adapter ...")
# Hu et al. 2021 LoRA-style init: identity W + zero b means the adapter
# starts as a pass-through. The on-device training run is what teaches it.
W_init = np.eye(768, dtype=np.float32)
b_init = np.zeros(768, dtype=np.float32)
graph.initializer.append(numpy_helper.from_array(W_init, name="adapter.W"))
graph.initializer.append(numpy_helper.from_array(b_init, name="adapter.b"))

matmul_node = helper.make_node(
    "MatMul", inputs=["features", "adapter.W"], outputs=["__adapter_matmul"],
    name="adapter_matmul",
)
add_node = helper.make_node(
    "Add", inputs=["__adapter_matmul", "adapter.b"], outputs=["adapted_features"],
    name="adapter_add",
)
graph.node.extend([matmul_node, add_node])

old = [o for o in graph.output if o.name == "features"][0]
new_out = helper.make_tensor_value_info(
    "adapted_features", TensorProto.FLOAT,
    [d.dim_param if d.HasField("dim_param") else d.dim_value
     for d in old.type.tensor_type.shape.dim],
)
graph.output.remove(old)
graph.output.append(new_out)

PATCHED_ONNX = os.path.join(WORK, "avhubert_with_trainable_adapter.onnx")
if os.path.exists(PATCHED_ONNX):
    os.remove(PATCHED_ONNX)
for f in os.listdir(WORK):
    if f.endswith(".data"):
        os.remove(os.path.join(WORK, f))

print("STEP C: Saving patched ONNX inline ...")
onnx.save(model, PATCHED_ONNX)
print(f"  {os.path.getsize(PATCHED_ONNX) / 1e6:.1f} MB")

from onnxruntime.training import artifacts
print("STEP D: generate_artifacts ...")
artifacts.generate_artifacts(
    PATCHED_ONNX,
    requires_grad=["adapter.W", "adapter.b"],
    loss=artifacts.LossType.MSELoss,
    optimizer=artifacts.OptimType.AdamW,
    artifact_directory=ARTIFACT_DIR,
    nominal_checkpoint=True,
    loss_input_names=["adapted_features"],
)
for f in sorted(os.listdir(ARTIFACT_DIR)):
    p = os.path.join(ARTIFACT_DIR, f)
    if os.path.isfile(p):
        print(f"  {f}  ({os.path.getsize(p) / 1e6:.1f} MB)")

readme = (
    "# Liperty V3 On-Device Training Artifacts\n\n"
    "ORT On-Device Training artifacts for fine-tuning a Linear adapter\n"
    "on top of the frozen AV-HuBERT base+vox+433h visual encoder.\n\n"
    "## Files\n\n"
    "| File | Purpose |\n"
    "|---|---|\n"
    "| training_model.onnx  | Forward pass + loss + gradient subgraph. TrainStep input. |\n"
    "| eval_model.onnx      | Forward pass + loss (no gradient nodes). |\n"
    "| optimizer_model.onnx | AdamW update graph. OptimizerStep input. |\n"
    "| checkpoint           | Initial trainable parameter values. |\n"
    "| nominal_checkpoint   | Slim per-param state for memory-constrained sessions. |\n\n"
    "## Architecture\n\n"
    "    video (1, 1, T, 88, 88)\n"
    "      -> [FROZEN] AV-HuBERT base+vox+433h visual encoder\n"
    "      -> features (1, T_out, 768)\n"
    "      -> [TRAINABLE] Linear: features @ W + b\n"
    "      -> adapted_features (1, T_out, 768)\n"
    "      -> [LOSS] MSE against target (placeholder; on-device CTC against real labels)\n\n"
    "Only `adapter.W` (768x768 identity init) and `adapter.b` (768 zero init)\n"
    "carry gradients on-device -- ~590K trainable params against a 95M\n"
    "frozen encoder.\n\n"
    "Built by `tools/kaggle_build_training_artifacts.py` in the Liperty repo.\n"
    "See `docs/PERSONALIZATION.md` for the broader plan.\n"
)
with open(os.path.join(ARTIFACT_DIR, "README.md"), "w") as f:
    f.write(readme)

print(f"STEP E: Uploading to {TARGET_REPO} ...")
create_repo(TARGET_REPO, repo_type="model", private=False, exist_ok=True)
upload_folder(
    folder_path=ARTIFACT_DIR,
    repo_id=TARGET_REPO,
    repo_type="model",
    commit_message="Build training artifacts (PoC, Kaggle Python 3.10 venv)",
)
print(f"Uploaded -> https://huggingface.co/{TARGET_REPO}")
print("Done.")
'''

with open(BUILD_PATH, "w") as f:
    f.write(script)

print(f"[+] Running {BUILD_PATH} in venv ...")
r = subprocess.run(
    [PY, BUILD_PATH],
    capture_output=True, text=True,
    env={**os.environ},
)
print("=== STDOUT ===")
print(r.stdout)
if r.returncode != 0:
    print("=== STDERR ===")
    print(r.stderr)
    print("=== exit:", r.returncode)
    sys.exit(r.returncode)
