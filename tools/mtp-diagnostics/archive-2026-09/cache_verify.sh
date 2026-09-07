#!/usr/bin/env bash
# Prueft nach dem Entfernen der Zwangsabschaltung:
#  1. wird der Compile-Cache jetzt ueberhaupt benutzt?
#  2. liefert ein Warmstart denselben Text wie der Kaltstart? (kein Drift)
# Fork-Defaults, KEIN VLLM_DISABLE_COMPILE_CACHE gesetzt.
set -uo pipefail
SP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPT=/home/mp/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4/snapshots/319f741cce68d7914884900c138a1fbb70a42f30
CACHE=$SP/verify_cache
PORT=8075
run_phase () {
  local label=$1 wipe=$2
  [ "$wipe" = "1" ] && rm -rf "$CACHE"
  mkdir -p "$CACHE"
  export PATH=/home/mp/vllm/venv/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
  export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_HOME=/home/mp/vllm/cuda TORCH_CUDA_ARCH_LIST=7.0
  export NCCL_P2P_DISABLE=1 VLLM_SM70_E5_CACHE=0 VLLM_SM70_NVFP4_TURBOMIND=1
  export VLLM_SM70_QUANT_BACKEND=auto VLLM_SKINNY_NVFP4=1 VLLM_SKINNY_QPN=1 VLLM_SKINNY_QPN2=1
  export VLLM_SKINNY_NVFP4_SRC=/home/mp/Projekte/v100-skinny/kernels/skinny_kernels.cu
  export TORCHINDUCTOR_CACHE_DIR=/home/mp/.cache/torchinductor VLLM_NO_USAGE_STATS=1
  export CUDA_VISIBLE_DEVICES=0 HOME=/home/mp VLLM_CACHE_ROOT="$CACHE"
  cd "$SP"
  /home/mp/vllm/venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$CKPT" --served-model-name verify --trust-remote-code --dtype float16 \
    --tensor-parallel-size 1 --gpu-memory-utilization 0.92 --block-size 16 \
    --max-model-len 8192 --max-num-seqs 1 --max-num-batched-tokens 2048 \
    --enable-prefix-caching --host 127.0.0.1 --port $PORT \
    --compilation-config '{"cudagraph_capture_sizes":[1,2]}' \
    > "$SP/verify_$label.log" 2>&1 &
  local pid=$!
  for i in $(seq 1 180); do
    sleep 5
    kill -0 $pid 2>/dev/null || { echo "$label: server tot"; return 1; }
    curl -s -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  done
  curl -s -m 300 "http://127.0.0.1:$PORT/v1/completions" -H "Content-Type: application/json" \
    -d '{"model":"verify","prompt":"Nenne die ersten zehn Primzahlen und erklaere kurz, warum 1 keine Primzahl ist.","temperature":0,"max_tokens":80,"seed":7}' \
    > "$SP/verify_$label.json" 2>&1
  kill -TERM $pid 2>/dev/null; sleep 12; kill -KILL $pid 2>/dev/null; sleep 5
}
echo "=== Kaltstart (Cache leer)"; run_phase cold 1
echo "=== Warmstart (gleicher Cache)"; run_phase warm 0
echo "FERTIG-VERIFY"
