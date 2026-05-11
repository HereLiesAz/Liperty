# AV-HuBERT visual encoder -> ONNX exporter
#
# After three failed Kaggle in-kernel attempts to install the 2022
# fairseq dep stack on a modern PyPI index, switching to a Docker
# container based on a 2022-vintage NVIDIA PyTorch image. The
# nvcr.io image freezes torch + numpy + cuda + everything at build
# time, so the dep graph composes cleanly without pip having to
# re-resolve against today's PyPI.
#
# Build:
#   docker build -t liperty-v3-export -f docker/v3-export/Dockerfile .
#
# Run (requires NVIDIA Container Toolkit + an NVIDIA GPU):
#   docker run --rm --gpus all \
#       -e HF_TOKEN=<your hf write token> \
#       -v /tmp/v3-out:/work/out \
#       liperty-v3-export
#
# Output:
#   - /work/out/avhubert_visual_encoder.onnx on the host (mounted)
#   - Same file pushed to HereLiesAz/liperty-avhubert-encoder on HF
#
# Base image rationale:
#   nvcr.io/nvidia/pytorch:22.12-py3 ships:
#     - Python 3.8.10
#     - PyTorch 1.14.0a0+44dac51 (close to the 1.13 AV-HuBERT was
#       tested against)
#     - CUDA 11.8
#     - cuDNN 8.7
#     - numpy 1.22.x (has np.float, pre-1.24 removal)
#   All frozen, no pip re-resolution at runtime needed.

FROM nvcr.io/nvidia/pytorch:22.12-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1

# System deps: git, ffmpeg (used by av_hubert's preprocessing but not
# the export itself; keeping it light), build-essential for any
# C extensions that need it.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        ffmpeg \
        ca-certificates \
        wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

# Clone av_hubert and init the vendored fairseq submodule. Pinning
# the av_hubert HEAD to a specific commit would be more reproducible
# but we want HEAD of main; if upstream breaks we'll catch it here.
RUN git clone --depth 1 https://github.com/facebookresearch/av_hubert.git \
    && cd av_hubert \
    && git submodule init \
    && git submodule update

# Pinned 2022 dep stack, matching what the vendored fairseq commit
# (afc77bdf) was tested against. The base image's torch/numpy/etc.
# already work; we only pin the non-base deps that fairseq's setup
# would otherwise pull at the wrong version.
#
# omegaconf 2.0.6 is what the lmd-vsr README recommends and what
# fairseq's setup.cfg constraints prefer. With Python 3.8 in this
# base image, it actually installs (the py3.9 constraint that bit
# the conda attempt doesn't apply here).
RUN pip install --no-cache-dir \
        "omegaconf==2.0.6" \
        "hydra-core==1.0.7" \
        "editdistance" \
        "python_speech_features" \
        "einops" \
        "soundfile" \
        "sentencepiece" \
        "sacrebleu" \
        "scipy" \
        "huggingface_hub>=0.27,<1.0" \
        "onnx>=1.16" \
        "onnxruntime>=1.18"

# DON'T pip install fairseq. The vendored commit's setup.py tries to
# Cython-build .pyx files (data_utils_fast etc.) and C++-compile a
# CUDA extension (libnat_cuda) that includes THC/THC.h — a header
# REMOVED in PyTorch 1.11+. The 22.12 base image has torch 1.14, so
# the install fails no matter what we do with extensions.
#
# Workaround: fairseq is fully importable via PYTHONPATH alone, and
# the Cython/CUDA extensions are only used for fast dataset loading
# and training-specific ops. AV-HuBERT inference (checkpoint load +
# forward pass + ONNX export) doesn't touch any of them.
#
# We tell fairseq to use pure-Python fallbacks via the documented
# env var, and rely on the ENV PYTHONPATH below to make `import
# fairseq` find the source tree directly.
ENV NO_FAIRSEQ_EXTENSIONS=1 \
    BUILD_NO_CUDA_EXTENSIONS=1 \
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Smoke test at build time. If this fails, the image build fails
# loudly rather than producing a broken image that errors out at
# `docker run`. We import explicitly the modules the export script
# touches; if any of them require the missing C/Cython extensions
# we'd find out HERE, not at run time.
RUN python -c "import sys; \
sys.path[:0] = ['/work/av_hubert/avhubert', '/work/av_hubert/fairseq']; \
import fairseq; assert fairseq.__file__, 'fairseq is a namespace package'; \
print('fairseq:', fairseq.__file__); \
from omegaconf import II; print('II OK'); \
import hubert_pretraining, hubert, hubert_asr; print('avhubert flat-imports OK'); \
from fairseq import checkpoint_utils; print('checkpoint_utils OK'); \
import torch; print('torch:', torch.__version__)"

# Copy the export script into the image. The script handles:
# - downloading the .pt from HereLiesAz/liperty-avhubert-encoder if not mounted
# - loading via fairseq.checkpoint_utils
# - video-only wrapper
# - torch.onnx.export with dynamic time axis
# - parity check vs the PyTorch path
# - upload back to HF
COPY tools/kaggle_avhubert_export_conda.py /work/export.py

# Mount /work/out at runtime to get the ONNX file back on the host
RUN mkdir -p /work/out

# When the container starts, run the export.
ENV V3_ENV_PREFIX=/usr \
    V3_WORK_DIR=/work \
    V3_AVHUBERT_DIR=/work/av_hubert \
    V3_FAIRSEQ_DIR=/work/av_hubert/fairseq \
    V3_CKPT_FILENAME=base_vox_433h.pt \
    V3_CKPT_PATH=/work/base_vox_433h.pt \
    V3_ONNX_OUT=/work/out/avhubert_base_vox_433h_visual_encoder.onnx \
    V3_HF_REPO=HereLiesAz/liperty-avhubert-encoder \
    PYTHONPATH=/work/av_hubert/avhubert:/work/av_hubert/fairseq \
    HUGGINGFACE_HUB_CACHE=/work/.hf

# NOTE: Default checkpoint is AV-HuBERT *base* (95M params, ~100 MB on
# disk, ~400 MB loaded) so the image runs on consumer GPUs with <=4 GB
# VRAM. To export AV-HuBERT *large* instead:
#   docker run --rm --gpus all \
#       -e HF_TOKEN=$HF_TOKEN \
#       -e V3_CKPT_FILENAME=large_vox_iter5.pt \
#       -e V3_CKPT_PATH=/work/large_vox_iter5.pt \
#       -v $(pwd)/out:/work/out \
#       liperty-v3-export
# But large needs ~8 GB VRAM during the trace; budget accordingly.

ENTRYPOINT ["python", "/work/export.py"]
