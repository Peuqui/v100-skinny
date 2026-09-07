#!/usr/bin/env bash
# Prueft den lm_head-Teilen-Fix: A) 27B auf V2 erzwungen  B) Flash-Next PP2 k=4
set -uo pipefail
W=/tmp/claude-1000/-home-mp-Projekte-AIfred-Intelligence/e75152f1-399f-4af7-baec-5de4bb58bb22/scratchpad/fixtest
export PATH=/home/mp/vllm/venv/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_HOME=/home/mp/vllm/cuda TORCH_CUDA_ARCH_LIST=7.0
export NCCL_P2P_DISABLE=1 VLLM_SM70_E5_CACHE=0 VLLM_SM70_NVFP4_TURBOMIND=1
export VLLM_SM70_QUANT_BACKEND=auto VLLM_SKINNY_NVFP4=1 VLLM_SKINNY_QPN=1 VLLM_SKINNY_QPN2=1
export VLLM_SKINNY_NVFP4_SRC=/home/mp/Projekte/v100-skinny/kernels/skinny_kernels.cu
export TORCHINDUCTOR_CACHE_DIR=/home/mp/.cache/torchinductor VLLM_NO_USAGE_STATS=1
export VLLM_CACHE_ROOT=/home/mp/.cache/vllm-calibration HOME=/home/mp
PROMPT='Erkläre in genau fünf Sätzen, warum der Himmel blau ist.'

wait_and_gen () {
  local tag="$1" pid="$2" model="$3"
  for i in $(seq 1 140); do
    sleep 5
    kill -0 $pid 2>/dev/null || { echo "  $tag: server exited"; return 1; }
    grep -q "Application startup complete" "$W/boot_$tag.log" && { echo "  $tag: STARTUP nach $((i*5))s"; break; }
    grep -qi "Traceback (most recent call last)" "$W/boot_$tag.log" && { echo "  $tag: FEHLER"; return 1; }
  done
  curl -s -m 300 http://127.0.0.1:8062/v1/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$model\",\"prompt\":\"$PROMPT\",\"max_tokens\":120,\"temperature\":0,\"seed\":1}" \
    > "$W/gen_$tag.json"
  echo "  $tag: generiert $(wc -c < $W/gen_$tag.json) Byte"
}
cleanup () {
  for p in $(pgrep -f 'VLLM[:]:'); do kill -TERM $p 2>/dev/null; done
  sleep 12
  for p in $(pgrep -f 'VLLM[:]:'); do kill -KILL $p 2>/dev/null; done
  sleep 5
}

echo "=== A: 27B auf V2 erzwungen, TP2, kein PP, k=3 ==="
CUDA_VISIBLE_DEVICES=0,2 VLLM_USE_V2_MODEL_RUNNER=1 VLLM_1CAT_ENABLE_SM70_MTP_DEFAULTS=1 \
/home/mp/vllm/venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model /home/mp/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4/snapshots/319f741cce68d7914884900c138a1fbb70a42f30 \
  --served-model-name Qwen3.8-27B-NVFP4-vllm --trust-remote-code --dtype float16 \
  --disable-custom-all-reduce --enable-prompt-tokens-details --enable-prefix-caching \
  --tensor-parallel-size 2 --pipeline-parallel-size 1 --gpu-memory-utilization 0.90 \
  --block-size 16 --max-model-len 32768 --max-num-seqs 4 --max-num-batched-tokens 2048 \
  --language-model-only --host 127.0.0.1 --port 8062 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"greedy","use_local_argmax_reduction":true,"attention_backend":"FLASH_ATTN"}' \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8]}' > $W/boot_A27b.log 2>&1 &
wait_and_gen A27b $! Qwen3.8-27B-NVFP4-vllm
cleanup

echo "=== B: Flash-Next TP2xPP2 k=4 ==="
CUDA_VISIBLE_DEVICES=0,2,1,3 \
/home/mp/vllm/venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model /home/mp/.cache/huggingface/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/7b719225242aacd3dbd3f9407468c2ee9a9d2594 \
  --served-model-name Qwen3.8-Flash-Next-NVFP4-vllm --trust-remote-code --dtype float16 \
  --disable-custom-all-reduce --enable-prompt-tokens-details --enable-prefix-caching \
  --tensor-parallel-size 2 --pipeline-parallel-size 2 --gpu-memory-utilization 0.97 \
  --block-size 16 --max-model-len 262144 --max-num-seqs 4 --max-num-batched-tokens 2048 \
  --language-model-only --async-scheduling --host 127.0.0.1 --port 8062 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4,"draft_sample_method":"greedy","use_local_argmax_reduction":true,"attention_backend":"FLASH_ATTN_V100"}' \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,5,8]}' --distributed-timeout-seconds 1200 \
  > $W/boot_BflashNext.log 2>&1 &
wait_and_gen BflashNext $! Qwen3.8-Flash-Next-NVFP4-vllm
cleanup
echo "ALLES FERTIG"
