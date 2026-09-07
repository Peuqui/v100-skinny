#!/usr/bin/env bash
# Produktionstest: EXAKT der llama-swap-Eintrag Qwen3.8-27B-NVFP4-vllm.
# Nichts getunt: TP2 auf GPU 0+2, GMU 0.98, 262144 Kontext, MTP k=3,
# vLLMs Standard-Cache-Root (~/.cache/vllm), Fork-Defaults.
# Zweimal booten: kalt, dann warm. Antwort und Cache-Verhalten vergleichen.
set -uo pipefail
SP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPT=/home/mp/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4/snapshots/319f741cce68d7914884900c138a1fbb70a42f30
PORT=8076
run_phase () {
  local label=$1
  export PATH=/home/mp/vllm/venv/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
  export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_HOME=/home/mp/vllm/cuda TORCH_CUDA_ARCH_LIST=7.0
  export NCCL_P2P_DISABLE=1 VLLM_SM70_E5_CACHE=0 VLLM_SM70_NVFP4_TURBOMIND=1
  export VLLM_SM70_QUANT_BACKEND=auto VLLM_SKINNY_NVFP4=1 VLLM_SKINNY_QPN=1 VLLM_SKINNY_QPN2=1
  export VLLM_SKINNY_NVFP4_SRC=/home/mp/Projekte/v100-skinny/kernels/skinny_kernels.cu
  export TORCHINDUCTOR_CACHE_DIR=/home/mp/.cache/torchinductor VLLM_NO_USAGE_STATS=1
  export CUDA_VISIBLE_DEVICES=0,2 VLLM_1CAT_ENABLE_SM70_MTP_DEFAULTS=1 HOME=/home/mp
  cd "$SP"
  /home/mp/vllm/venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$CKPT" --served-model-name Qwen3.8-27B-NVFP4-vllm --trust-remote-code \
    --dtype float16 --disable-custom-all-reduce --enable-auto-tool-choice \
    --tool-call-parser hermes --enable-prompt-tokens-details --enable-prefix-caching \
    --tensor-parallel-size 2 --pipeline-parallel-size 1 --gpu-memory-utilization 0.98 \
    --block-size 16 --max-model-len 262144 --max-num-seqs 4 --max-num-batched-tokens 2048 \
    --host 127.0.0.1 --port $PORT --language-model-only \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"greedy","use_local_argmax_reduction":true,"attention_backend":"FLASH_ATTN"}' \
    --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8]}' \
    > "$SP/prod_$label.log" 2>&1 &
  local pid=$!
  local t0=$(date +%s)
  for i in $(seq 1 240); do
    sleep 5
    kill -0 $pid 2>/dev/null || { echo "$label: server tot nach $(( $(date +%s)-t0 ))s"; return 1; }
    curl -s -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  done
  echo "$label: bereit nach $(( $(date +%s)-t0 ))s"
  curl -s -m 300 "http://127.0.0.1:$PORT/v1/completions" -H "Content-Type: application/json" \
    -d '{"model":"Qwen3.8-27B-NVFP4-vllm","prompt":"Nenne die ersten zehn Primzahlen und erklaere kurz, warum 1 keine Primzahl ist.","temperature":0,"max_tokens":80,"seed":7}' \
    > "$SP/prod_$label.json" 2>&1
  kill -TERM $pid 2>/dev/null; sleep 15; kill -KILL $pid 2>/dev/null; sleep 5
}
echo "=== Produktions-Kaltstart"; run_phase cold
echo "=== Produktions-Warmstart"; run_phase warm
echo "FERTIG-PROD"
