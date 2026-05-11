"""Bootstrap a conda env for the AV-HuBERT ONNX export.

Replaces the doomed-by-dep-rot in-process pip install attempts.
This script runs in the Kaggle kernel's Python (so it has access
to /usr/local/bin/mamba and can do subprocess calls), and it sets
up a separate Python 3.9 env at /kaggle/working/v3_env with the
exact 2022 dep stack that fairseq commit afc77bdf was tested
against.

Run from the Kaggle IPython console:

    import urllib.request
    exec(urllib.request.urlopen(
        'https://raw.githubusercontent.com/HereLiesAz/Liperty/main/tools/kaggle_avhubert_bootstrap_conda.py'
    ).read())
    # ... wait for completion ...
    # then run the export inside the env:
    import subprocess
    subprocess.run(
        ['/kaggle/working/v3_env/bin/python',
         '-c', 'import urllib.request; exec(urllib.request.urlopen('
               '"https://raw.githubusercontent.com/HereLiesAz/Liperty/main/tools/kaggle_avhubert_export_conda.py"'
               ').read())'],
        env={**os.environ, 'HF_TOKEN': HF_TOKEN_VALUE},
        check=True,
    )

Steps:
1. Verify mamba is available
2. Create env at /kaggle/working/v3_env with Python 3.9 (skipped if exists)
3. Install pinned deps via the env's pip
4. Clone av_hubert + init fairseq submodule (skipped if exists)
5. `pip install --editable` the fairseq inside the env
"""

import os
import sys
import subprocess
import shutil
import time

ENV_PREFIX = "/kaggle/working/v3_env"
WORK_DIR = "/kaggle/working/work"
AVHUBERT_DIR = os.path.join(WORK_DIR, "av_hubert")
FAIRSEQ_DIR = os.path.join(AVHUBERT_DIR, "fairseq")

MAMBA = "/usr/local/bin/mamba"
ENV_PYTHON = os.path.join(ENV_PREFIX, "bin", "python")
ENV_PIP = os.path.join(ENV_PREFIX, "bin", "pip")


def _run(cmd, check=True, capture=False):
    print(f"$ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    if capture:
        r = subprocess.run(cmd, shell=isinstance(cmd, str), check=False, capture_output=True, text=True)
        if check and r.returncode != 0:
            print(r.stdout[-1500:])
            print(r.stderr[-1500:])
            raise RuntimeError(f"command failed rc={r.returncode}")
        return r
    return subprocess.run(cmd, shell=isinstance(cmd, str), check=check)


# -------------------------------------------------------------------------
# 1. Verify mamba
# -------------------------------------------------------------------------
if not os.path.exists(MAMBA):
    raise RuntimeError(f"{MAMBA} not found. Kaggle's base image normally has it.")
print(f"mamba: {MAMBA}")


# -------------------------------------------------------------------------
# 2. Create env (skipped if already exists from background mamba create)
# -------------------------------------------------------------------------
if not os.path.exists(ENV_PYTHON):
    print(f"Creating env at {ENV_PREFIX}...")
    t0 = time.time()
    _run([MAMBA, "create", "-p", ENV_PREFIX, "-y", "-c", "conda-forge", "python=3.9"])
    print(f"  env created in {time.time()-t0:.0f}s")
else:
    print(f"env exists: {ENV_PYTHON}")


# -------------------------------------------------------------------------
# 3. Pinned deps (mirror lmd-vsr's documented stack)
# -------------------------------------------------------------------------
print("Installing pinned 2022 dep stack...")
t0 = time.time()
_run([ENV_PIP, "install", "-q", "--no-cache-dir",
      "torch==1.13.1+cu117", "torchvision==0.14.1+cu117", "torchaudio==0.13.1",
      "--extra-index-url", "https://download.pytorch.org/whl/cu117"])

_run([ENV_PIP, "install", "-q", "--no-cache-dir",
      "numpy==1.23.5",
      "scipy",
      "cython",
      "omegaconf==2.0.6",   # the version fairseq's vendored commit needs
      "hydra-core==1.0.7",  # same era as omegaconf 2.0
      "editdistance",
      "python_speech_features",
      "einops",
      "soundfile",
      "sentencepiece",
      "sacrebleu",
      "tqdm",
      "regex",
      "cffi",
      "huggingface_hub>=0.27,<1.0",
      "onnx>=1.16",
      "onnxruntime>=1.18"])
print(f"  pinned deps installed in {time.time()-t0:.0f}s")


# -------------------------------------------------------------------------
# 4. Clone av_hubert (if missing) + init submodule
# -------------------------------------------------------------------------
os.makedirs(WORK_DIR, exist_ok=True)
if not os.path.exists(AVHUBERT_DIR):
    print("Cloning av_hubert...")
    _run(["git", "clone", "--depth", "1", "https://github.com/facebookresearch/av_hubert.git", AVHUBERT_DIR])
    _run(f"cd {AVHUBERT_DIR} && git submodule init && git submodule update")
print(f"av_hubert: {AVHUBERT_DIR}")

if not os.path.exists(os.path.join(FAIRSEQ_DIR, "setup.py")):
    raise RuntimeError(f"fairseq submodule didn't populate at {FAIRSEQ_DIR}")


# -------------------------------------------------------------------------
# 5. pip install --editable fairseq into the env
# -------------------------------------------------------------------------
print("Installing fairseq (editable) into env...")
t0 = time.time()
_run([ENV_PIP, "install", "--editable", FAIRSEQ_DIR, "-q", "--no-deps"])
print(f"  fairseq editable install in {time.time()-t0:.0f}s")


# -------------------------------------------------------------------------
# 6. Smoke test: env's python can `import fairseq` properly
# -------------------------------------------------------------------------
print("Smoke-testing fairseq import in env...")
probe = (
    "import sys; sys.path.insert(0, '" + FAIRSEQ_DIR + "'); "
    "sys.path.insert(0, '" + AVHUBERT_DIR + "'); "
    "import fairseq; print('fairseq.__file__ =', fairseq.__file__); "
    "from omegaconf import II; print('II OK'); "
    "import avhubert; print('avhubert OK')"
)
r = _run([ENV_PYTHON, "-c", probe], capture=True, check=False)
print(r.stdout)
if r.returncode != 0:
    print("STDERR:")
    print(r.stderr[-2000:])
    raise RuntimeError("env smoke test FAILED; see stderr above")

print()
print("Bootstrap complete. Env: " + ENV_PREFIX)
print("Next: run the export via")
print(f"  subprocess.run(['{ENV_PYTHON}', '/path/to/kaggle_avhubert_export_conda.py'], ...)")
