#!/usr/bin/env bash
# Bring-up launcher for DeepSeek-V4-Flash (compressed-tensors: attention
# block-FP8 -> QPN8-blk, ffn NVFP4) on the 5-GPU box: TP=1 PP=5, RTX
# stages first. No serving gates yet -- this is the boot-debug path;
# measurement runs still go through bench.py against this server.
#
#   PORT (8021) MML (16384) GMU (0.92) EXTRA_ARGS
#   VLLM_PP_LAYER_PARTITION (11,11,7,7,7 -- 43 layers, VRAM-weighted
#   48/48/32/32/32)
set -euo pipefail
SNAP="${1:?usage: serve-deepseek-mini.sh <snapshot-dir>}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-8021}"
MML="${MML:-16384}"
GMU="${GMU:-0.92}"
export VLLM_PP_LAYER_PARTITION="${VLLM_PP_LAYER_PARTITION:-11,11,7,7,7}"
export VLLM_SKINNY_NVFP4_SRC="$REPO/kernels/skinny_kernels.cu"
export NCCL_P2P_DISABLE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,2,1,4,3}"
export CUDA_HOME="${CUDA_HOME:-$REPO/.cuda-nvcc-deb/usr/local/cuda-12.8}"
export PATH="$CUDA_HOME/bin:$PATH"
# the 1cat wheel registers no "vllm" dist metadata, so the console script
# refuses to start -- module launch works.
exec "$REPO/.venv-sm70/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$SNAP" \
  --served-model-name deepseek-v4-flash \
  --tensor-parallel-size 1 --pipeline-parallel-size 5 \
  --dtype half --max-model-len "$MML" \
  --kv-cache-dtype fp8 \
  --max-num-batched-tokens "${MNBT:-2048}" \
  --gpu-memory-utilization "$GMU" \
  --disable-custom-all-reduce \
  --port "$PORT" ${EXTRA_ARGS:-}
