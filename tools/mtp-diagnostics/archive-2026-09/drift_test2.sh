#!/usr/bin/env bash
# Real-Life-Test des Cache-Schluessel-Lochs.
#   Phase 1: QPN=0 mit LEEREM Cache   -> B_clean
#   Phase 2: QPN=1 mit LEEREM Cache   -> A_clean   (fuellt den Cache mit A)
#   Phase 3: QPN=0 mit Cache aus 2    -> B_cached  (ohne Fix: fremdes Artefakt)
#   Phase 4: wie 3, aber mit Fix      -> B_fixed
# Ausgewertet wird der Token-Text bei temperature 0.
set -uo pipefail
SP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPT=/home/mp/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4/snapshots/319f741cce68d7914884900c138a1fbb70a42f30
CACHE=$SP/drift2_cache
PORT=8071
run_phase () {  # $1=label  $2=QPN  $3=wipe(0/1)
  local label=$1 qpn=$2 wipe=$3
  [ "$wipe" = "1" ] && rm -rf "$CACHE"
  mkdir -p "$CACHE"
  export PATH=/home/mp/vllm/venv/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
  export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_HOME=/home/mp/vllm/cuda TORCH_CUDA_ARCH_LIST=7.0
  export NCCL_P2P_DISABLE=1 VLLM_SM70_E5_CACHE=0 VLLM_SM70_NVFP4_TURBOMIND=1
  export VLLM_SM70_QUANT_BACKEND=auto VLLM_SKINNY_QPN2=1
  export VLLM_SKINNY_NVFP4_SRC=/home/mp/Projekte/v100-skinny/kernels/skinny_kernels.cu
  export TORCHINDUCTOR_CACHE_DIR=/home/mp/.cache/torchinductor VLLM_NO_USAGE_STATS=1
  export CUDA_VISIBLE_DEVICES=0 HOME=/home/mp
  export VLLM_CACHE_ROOT="$CACHE"
  export VLLM_DISABLE_COMPILE_CACHE=0
  export VLLM_SKINNY_NVFP4="$qpn"; export VLLM_SKINNY_QPN="$qpn"
  cd "$SP"
  /home/mp/vllm/venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$CKPT" --served-model-name drift --trust-remote-code --dtype float16 \
    --tensor-parallel-size 1 --gpu-memory-utilization 0.92 --block-size 16 \
    --max-model-len 8192 --max-num-seqs 1 --max-num-batched-tokens 2048 \
    --enable-prefix-caching --host 127.0.0.1 --port $PORT \
    --compilation-config '{"cudagraph_capture_sizes":[1,2]}' \
    > "$SP/drift2_$label.log" 2>&1 &
  local pid=$!
  for i in $(seq 1 180); do
    sleep 5
    kill -0 $pid 2>/dev/null || { echo "$label: server tot"; return 1; }
    curl -s -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  done
  curl -s -m 300 "http://127.0.0.1:$PORT/v1/completions" -H "Content-Type: application/json" \
    -d '{"model":"drift","prompt":"Nenne die ersten zehn Primzahlen und erklaere kurz, warum 1 keine Primzahl ist.","temperature":0,"max_tokens":40,"seed":7,"logprobs":5}' \
    > "$SP/drift2_$label.json" 2>&1
  echo "$label cache-dirs: $(ls "$CACHE/torch_compile_cache" 2>/dev/null | tr '\n' ' ')"
  kill -TERM $pid 2>/dev/null; sleep 12; kill -KILL $pid 2>/dev/null; sleep 5
}
echo "=== Phase 1 (SKINNY=0 Marlin, leerer Cache)"; run_phase M_clean 0 1
echo "=== Phase 2 (SKINNY=1 eigene Kernel, leerer Cache)"; run_phase S_clean 1 1
echo "=== Phase 3 (SKINNY=0, Cache von Phase 2)";  run_phase M_cached 0 0
echo "FERTIG-TEIL1"
