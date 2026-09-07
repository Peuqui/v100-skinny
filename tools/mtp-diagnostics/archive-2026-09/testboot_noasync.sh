#!/usr/bin/env bash
# Testboot Flash-Next TP2 PP2 k=4 volle 262144 Token, mit Host-RAM-Wachhund.
set -uo pipefail
LOG=/tmp/claude-1000/-home-mp-Projekte-AIfred-Intelligence/aa7b05ef-c411-42a6-aa69-1abeee80dfca/scratchpad/testboot_noasync.log
CKPT=/home/mp/.cache/huggingface/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/7b719225242aacd3dbd3f9407468c2ee9a9d2594
cd "$(dirname "$LOG")"
export PATH=/home/mp/vllm/venv/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_HOME=/home/mp/vllm/cuda TORCH_CUDA_ARCH_LIST=7.0
export NCCL_P2P_DISABLE=1 VLLM_SM70_E5_CACHE=0 VLLM_SM70_NVFP4_TURBOMIND=1
export VLLM_SM70_QUANT_BACKEND=auto VLLM_SKINNY_NVFP4=1 VLLM_SKINNY_QPN=1 VLLM_SKINNY_QPN2=1
export VLLM_SKINNY_NVFP4_SRC=/home/mp/Projekte/v100-skinny/kernels/skinny_kernels.cu
export TORCHINDUCTOR_CACHE_DIR=/home/mp/.cache/torchinductor VLLM_NO_USAGE_STATS=1
export VLLM_CACHE_ROOT=/home/mp/.cache/vllm-calibration
export CUDA_VISIBLE_DEVICES=0,2,1,3 HOME=/home/mp
/home/mp/vllm/venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$CKPT" --served-model-name Qwen3.8-Flash-Next-NVFP4-vllm \
  --trust-remote-code --dtype float16 --disable-custom-all-reduce \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --enable-prompt-tokens-details --enable-prefix-caching \
  --tensor-parallel-size 2 --pipeline-parallel-size 2 \
  --gpu-memory-utilization 0.97 --block-size 16 --max-model-len 262144 \
  --max-num-seqs 4 --max-num-batched-tokens 2048 \
  --language-model-only --no-async-scheduling --host 127.0.0.1 --port 8061 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4,"draft_sample_method":"greedy","use_local_argmax_reduction":true,"attention_backend":"FLASH_ATTN_V100"}' \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,5,8]}' --distributed-timeout-seconds 3600 \
  > "$LOG" 2>&1 &
SERVER=$!
echo "server pid $SERVER"
TOTAL=$(awk '/MemTotal/{print $2}' /proc/meminfo)
FLOOR=$((TOTAL / 10))
for i in $(seq 1 240); do
  sleep 5
  kill -0 $SERVER 2>/dev/null || { echo "server exited"; break; }
  AVAIL=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  if [ "$AVAIL" -lt "$FLOOR" ]; then
    echo "WATCHDOG: available $((AVAIL/1024)) MB < floor $((FLOOR/1024)) MB -- killing boot"
    pkill -TERM -P $SERVER 2>/dev/null; kill -TERM $SERVER 2>/dev/null; sleep 10
    pkill -KILL -P $SERVER 2>/dev/null; kill -KILL $SERVER 2>/dev/null
    break
  fi
  if grep -q "Application startup complete" "$LOG" 2>/dev/null; then echo "STARTUP OK nach $((i*5))s"; break; fi
  if grep -q -i "User compiler error\|Traceback (most recent call last)" "$LOG" 2>/dev/null; then echo "FEHLER im Log nach $((i*5))s"; sleep 20; break; fi
done
echo "--- Ende, Server-PID $SERVER lebt: $(kill -0 $SERVER 2>/dev/null && echo ja || echo nein)"
