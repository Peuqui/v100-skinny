#!/usr/bin/env bash
# 27B mit MTP k=3, TP2 auf dem RTX-Paar, OHNE PP.
# Lauf A: V2-Runner erzwungen. Lauf B: Default (V1). Gleicher Prompt, greedy.
set -uo pipefail
W=/tmp/claude-1000/-home-mp-Projekte-AIfred-Intelligence/e75152f1-399f-4af7-baec-5de4bb58bb22/scratchpad/v1v2
CKPT=/home/mp/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4/snapshots/319f741cce68d7914884900c138a1fbb70a42f30
export PATH=/home/mp/vllm/venv/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_HOME=/home/mp/vllm/cuda TORCH_CUDA_ARCH_LIST=7.0
export NCCL_P2P_DISABLE=1 VLLM_SM70_E5_CACHE=0 VLLM_SM70_NVFP4_TURBOMIND=1
export VLLM_SM70_QUANT_BACKEND=auto VLLM_SKINNY_NVFP4=1 VLLM_SKINNY_QPN=1 VLLM_SKINNY_QPN2=1
export VLLM_SKINNY_NVFP4_SRC=/home/mp/Projekte/v100-skinny/kernels/skinny_kernels.cu
export TORCHINDUCTOR_CACHE_DIR=/home/mp/.cache/torchinductor VLLM_NO_USAGE_STATS=1
export VLLM_CACHE_ROOT=/home/mp/.cache/vllm-calibration
export CUDA_VISIBLE_DEVICES=0,2 HOME=/home/mp
export VLLM_1CAT_ENABLE_SM70_MTP_DEFAULTS=1

run_one () {
  local tag="$1"; shift
  echo "=== Lauf $tag ==="
  "$@" /home/mp/vllm/venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$CKPT" --served-model-name Qwen3.8-27B-NVFP4-vllm \
    --trust-remote-code --dtype float16 --disable-custom-all-reduce \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --enable-prompt-tokens-details --enable-prefix-caching \
    --tensor-parallel-size 2 --pipeline-parallel-size 1 \
    --gpu-memory-utilization 0.90 --block-size 16 --max-model-len 32768 \
    --max-num-seqs 4 --max-num-batched-tokens 2048 \
    --language-model-only --host 127.0.0.1 --port 8061 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"greedy","use_local_argmax_reduction":true,"attention_backend":"FLASH_ATTN"}' \
    --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8]}' \
    > "$W/boot_$tag.log" 2>&1 &
  local S=$!
  for i in $(seq 1 100); do
    sleep 5
    kill -0 $S 2>/dev/null || { echo "  server exited"; break; }
    grep -q "Application startup complete" "$W/boot_$tag.log" && { echo "  STARTUP nach $((i*5))s"; break; }
    grep -qi "Traceback (most recent call last)" "$W/boot_$tag.log" && { echo "  FEHLER"; break; }
  done
  if curl -s -m 10 http://127.0.0.1:8061/v1/models >/dev/null 2>&1; then
    curl -s -m 300 http://127.0.0.1:8061/v1/completions -H 'Content-Type: application/json' -d '{
      "model":"Qwen3.8-27B-NVFP4-vllm",
      "prompt":"Erkläre in genau fünf Sätzen, warum der Himmel blau ist.",
      "max_tokens":120,"temperature":0,"seed":1}' > "$W/gen_$tag.json"
    echo "  generiert: $(wc -c < $W/gen_$tag.json) Byte"
  else
    echo "  kein Server"
  fi
  for p in $(pgrep -f 'VLLM[:]:'); do kill -TERM $p 2>/dev/null; done
  kill -TERM $S 2>/dev/null; sleep 12
  for p in $(pgrep -f 'VLLM[:]:'); do kill -KILL $p 2>/dev/null; done
  kill -KILL $S 2>/dev/null; sleep 5
}

run_one v2forced env VLLM_USE_V2_MODEL_RUNNER=1
run_one v1default env
echo "ALLES FERTIG"
