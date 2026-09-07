#!/usr/bin/env bash
# Instrumentierter Lauf MIT echter Generierung; raeumt am Ende selbst auf.
set -uo pipefail
D=/tmp/claude-1000/-home-mp-Projekte-AIfred-Intelligence/e75152f1-399f-4af7-baec-5de4bb58bb22/scratchpad
LOG=$D/run2/boot.log
cd $D/run2
export PATH=/home/mp/vllm/venv/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_HOME=/home/mp/vllm/cuda TORCH_CUDA_ARCH_LIST=7.0
export NCCL_P2P_DISABLE=1 VLLM_SM70_E5_CACHE=0 VLLM_SM70_NVFP4_TURBOMIND=1
export VLLM_SM70_QUANT_BACKEND=auto VLLM_SKINNY_NVFP4=1 VLLM_SKINNY_QPN=1 VLLM_SKINNY_QPN2=1
export VLLM_SKINNY_NVFP4_SRC=/home/mp/Projekte/v100-skinny/kernels/skinny_kernels.cu
export TORCHINDUCTOR_CACHE_DIR=/home/mp/.cache/torchinductor VLLM_NO_USAGE_STATS=1
export VLLM_CACHE_ROOT=/home/mp/.cache/vllm-calibration
export CUDA_VISIBLE_DEVICES=0,2,1,3 HOME=/home/mp
export VLLM_SKINNY_PPDIAG=$D/run2/ppdiag VLLM_SKINNY_PPDIAG_CAP=400
CKPT=/home/mp/.cache/huggingface/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/7b719225242aacd3dbd3f9407468c2ee9a9d2594
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
for i in $(seq 1 120); do
  sleep 5
  kill -0 $SERVER 2>/dev/null || { echo "server exited"; exit 1; }
  grep -q "Application startup complete" "$LOG" && { echo "STARTUP nach $((i*5))s"; break; }
  grep -qi "Traceback (most recent call last)" "$LOG" && { echo "FEHLER"; break; }
done
if curl -s -m 10 http://127.0.0.1:8060/v1/models >/dev/null 2>&1; then
  echo "generiere..."
  curl -s -m 300 http://127.0.0.1:8060/v1/completions -H 'Content-Type: application/json' -d '{
    "model":"Qwen3.8-Flash-Next-NVFP4-vllm",
    "prompt":"Erkläre in genau fünf Sätzen, warum der Himmel blau ist.",
    "max_tokens":120, "temperature":0, "seed":1}' > $D/run2/gen.json
  echo "generiert: $(wc -c < $D/run2/gen.json) Byte"
fi
echo "raeume auf"
for p in $(pgrep -f 'VLLM[:]:'); do kill -TERM $p 2>/dev/null; done
kill -TERM $SERVER 2>/dev/null; sleep 12
for p in $(pgrep -f 'VLLM[:]:'); do kill -KILL $p 2>/dev/null; done
kill -KILL $SERVER 2>/dev/null
echo "fertig"
