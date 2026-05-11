# Docker — AV-HuBERT V3 ONNX Exporter

The `avhubert_visual_encoder.onnx` artifact that Liperty's V3 backend
research needs, built from a 2022-vintage NVIDIA PyTorch base image
so the dep stack actually composes (see
[`docs/AVHUBERT_V3_BACKEND.md`](../../docs/AVHUBERT_V3_BACKEND.md) for the
three earlier in-Kaggle attempts that all failed at dep resolution).

## Requirements

- Docker
- An NVIDIA GPU + the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- An HF token with **write** access to `HereLiesAz/liperty-avhubert-encoder`
- ~25 GB free disk for the build (base image is ~15 GB; AV-HuBERT
  `.pt` pulled at runtime is 3.91 GB; ONNX output is ~3 GB)

You do NOT need to do this on Kaggle. Any Linux box with a recent NVIDIA
driver works. A T4, V100, A100, RTX 3090, etc. all qualify.

## Build

From the **Liperty repo root** (the Dockerfile copies `tools/...` into the
image so build context must include the repo):

```bash
docker build -t liperty-v3-export -f docker/v3-export/Dockerfile .
```

This step is the long one (~10-20 min). It does:

1. Pulls `nvcr.io/nvidia/pytorch:22.12-py3` (~15 GB, one-time)
2. `apt install git build-essential ffmpeg`
3. `git clone facebookresearch/av_hubert` + submodule init
4. `pip install` the pinned 2022 dep stack
5. `pip install --editable` the vendored fairseq
6. Smoke-tests that `import fairseq`, `from omegaconf import II`,
   and `import avhubert` all work — **fails the build** if any of
   them don't, so you don't get a broken image

## Run

```bash
docker run --rm --gpus all \
    -e HF_TOKEN=$HF_TOKEN \
    -v $(pwd)/out:/work/out \
    liperty-v3-export
```

This:

1. Pulls `large_vox_iter5.pt` (3.91 GB) from
   [`HereLiesAz/liperty-avhubert-encoder`](https://huggingface.co/HereLiesAz/liperty-avhubert-encoder)
   on first run (cached inside the container's `/work/.hf` after that).
2. Loads the model via `fairseq.checkpoint_utils`.
3. Wraps `model.extract_features` for video-only forward.
4. Calls `torch.onnx.export` with a dynamic time axis.
5. Parity-checks PyTorch vs onnxruntime on a dummy `(1, 1, 50, 88, 88)` tensor.
6. Writes `avhubert_visual_encoder.onnx` to `/work/out/` (bind-mounted
   so it lands at `./out/avhubert_visual_encoder.onnx` on the host).
7. Uploads the same file to `HereLiesAz/liperty-avhubert-encoder` on HF.

## If the trace fails

This is the actual research-risky step the docs have been pointing at.
fairseq's transformer encoder uses dynamic control flow for the
positional encoding mask path that may not trace cleanly. If
`torch.onnx.export` errors, the script prints the traceback and aborts.

Workarounds (from cheapest to most invasive):

1. Set `dynamo=True` in the `torch.onnx.export` call and retry. The
   torch 1.14 dynamo exporter handles some shapes the legacy tracer
   misses.
2. Pin the input `T` axis to a single static value (drop the
   `dynamic_axes` argument). Liperty's on-device path uses a fixed
   window of 50 frames anyway, so this might be acceptable.
3. Re-implement the encoder forward in plain PyTorch (no fairseq custom
   layers) and transfer the state dict. This is several days of work
   but produces a clean ONNX guaranteed to trace.

## Environment overrides

Every path the script touches is env-var configurable. Inside the
image they default to the values the Dockerfile sets, but you can
override at `docker run` time:

| Var | Default in image | What it controls |
|---|---|---|
| `V3_CKPT_PATH` | `/work/large_vox_iter5.pt` | Where to find / cache the AV-HuBERT `.pt` |
| `V3_ONNX_OUT` | `/work/out/avhubert_visual_encoder.onnx` | Where to write the ONNX (bind-mount this dir!) |
| `V3_HF_REPO` | `HereLiesAz/liperty-avhubert-encoder` | HF model repo to pull from / upload to |
| `HF_TOKEN` | (none — must be passed) | Required for the upload step |

## License

The image bundles AV-HuBERT under Meta's non-commercial research license.
The Dockerfile and run scripts in this directory are MIT under the
Liperty repo's license. See `LICENSE` at the repo root.
