#!/usr/bin/env bash
# Flash-Next TP2 PP2 k=4 — Diagnoseboot: NCCL pro Rang + Flight-Recorder.
set -uo pipefail
DIR=/tmp/claude-1000/-home-mp-Projekte-AIfred-Intelligence/e75152f1-399f-4af7-baec-5de4bb58bb22/scratchpad
LOG=$DIR/diagboot_nccl.log
CKPT=/home/mp/.cache/huggingface/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/7b719225242aacd3dbd3f9407468c2ee9a9d2594
cd "$DIR"
rm -f nccl.*.log flight* diagboot_nccl.log

export PATH=/home/mp/vllm/venv/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_HOME=/home/mp/vllm/cuda TORCH_CUDA_ARCH_LIST=7.0
export NCCL_P2P_DISABLE=1 VLLM_SM70_E5_CACHE=0 VLLM_SM70_NVFP4_TURBOMIND=1
export VLLM_SM70_QUANT_BACKEND=auto VLLM_SKINNY_NVFP4=1 VLLM_SKINNY_QPN=1 VLLM_SKINNY_QPN2=1
export VLLM_SKINNY_NVFP4_SRC=/home/mp/Projekte/v100-skinny/kernels/skinny_kernels.cu
export TORCHINDUCTOR_CACHE_DIR=/home/mp/.cache/torchinductor VLLM_NO_USAGE_STATS=1
export VLLM_CACHE_ROOT=/home/mp/.cache/vllm-calibration
export CUDA_VISIBLE_DEVICES=0,2,1,3 HOME=/home/mp

# --- Diagnose ---
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,COLL,P2P
export NCCL_DEBUG_FILE=$DIR/nccl.%h.%p.log
export TORCH_NCCL_TRACE_BUFFER_SIZE=20000
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE=$DIR/flight

/home/mp/vllm/venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$CKPT" --served-model-name Qwen3.8-Flash-Next-NVFP4-vllm \
  --trust-remote-code --dtype float16 --disable-custom-all-reduce \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --enable-prompt-tokens-details --enable-prefix-caching \
  --tensor-parallel-size 2 --pipeline-parallel-size 2 \
  --gpu-memory-utilization 0.97 --block-size 16 --max-model-len 262144 \
  --max-num-seqs 4 --max-num-batched-tokens 2048 \
  --language-model-only --async-scheduling --host 127.0.0.1 --port 8060 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4,"draft_sample_method":"greedy","use_local_argmax_reduction":true,"attention_backend":"FLASH_ATTN_V100"}' \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,5,8]}' --distributed-timeout-seconds 1200 \
  > "$LOG" 2>&1 &
SERVER=$!
echo "server pid $SERVER"

TOTAL=$(awk '/MemTotal/{print $2}' /proc/meminfo)
FLOOR=$((TOTAL / 10))
STALL=0
PREV=""
for i in $(seq 1 420); do   # 420 * 10 s = 70 min
  sleep 10
  kill -0 $SERVER 2>/dev/null || { echo "[$((i*10))s] server exited"; break; }

  AVAIL=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  if [ "$AVAIL" -lt "$FLOOR" ]; then
    echo "[$((i*10))s] WATCHDOG Host-RAM: $((AVAIL/1024)) MB < $((FLOOR/1024)) MB -- Abbruch"
    pkill -TERM -P $SERVER 2>/dev/null; kill -TERM $SERVER 2>/dev/null; sleep 10
    pkill -KILL -P $SERVER 2>/dev/null; kill -KILL $SERVER 2>/dev/null
    break
  fi

  if grep -q "Application startup complete" "$LOG" 2>/dev/null; then
    echo "[$((i*10))s] STARTUP OK -- keine Verklemmung!"; break; fi
  if grep -q -i "User compiler error\|Traceback (most recent call last)" "$LOG" 2>/dev/null; then
    echo "[$((i*10))s] FEHLER im Log"; sleep 20; break; fi

  # Verklemmung erkennen: geschriebene Bytes aller Worker
  CUR=""
  for p in $(pgrep -f 'VLLM[:]:Worker' 2>/dev/null); do
    CUR="$CUR $(awk '/^wchar/{print $2}' /proc/$p/io 2>/dev/null)"
  done
  if [ -n "$CUR" ] && [ "$CUR" = "$PREV" ]; then
    STALL=$((STALL+1))
    [ $((STALL % 6)) -eq 0 ] && echo "[$((i*10))s] STILLSTAND seit $((STALL*10))s -- keine Bytes geschrieben"
  else
    [ -n "$CUR" ] && [ $STALL -gt 0 ] && echo "[$((i*10))s] laeuft wieder"
    STALL=0
  fi
  PREV="$CUR"

  # Watchdog-Dump von PyTorch abgefallen?
  if ls flight* >/dev/null 2>&1; then echo "[$((i*10))s] FLIGHT-RECORDER-DUMP da"; sleep 30; break; fi
  if grep -q "Watchdog caught collective operation timeout" "$LOG" 2>/dev/null; then
    echo "[$((i*10))s] WATCHDOG-TIMEOUT im Log"; sleep 60; break; fi
done
echo "--- Ende. Server lebt: $(kill -0 $SERVER 2>/dev/null && echo ja || echo nein)"
