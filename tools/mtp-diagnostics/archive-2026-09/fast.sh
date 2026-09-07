#!/usr/bin/env bash
# Schnelle Iteration am 27B auf V2 (TP2, kein PP, 80-s-Boot).
set -uo pipefail
W=/tmp/claude-1000/-home-mp-Projekte-AIfred-Intelligence/e75152f1-399f-4af7-baec-5de4bb58bb22/scratchpad/fast
export PATH=/home/mp/vllm/venv/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_HOME=/home/mp/vllm/cuda TORCH_CUDA_ARCH_LIST=7.0
export NCCL_P2P_DISABLE=1 VLLM_SM70_E5_CACHE=0 VLLM_SM70_NVFP4_TURBOMIND=1
export VLLM_SM70_QUANT_BACKEND=auto VLLM_SKINNY_NVFP4=1 VLLM_SKINNY_QPN=1 VLLM_SKINNY_QPN2=1
export VLLM_SKINNY_NVFP4_SRC=/home/mp/Projekte/v100-skinny/kernels/skinny_kernels.cu
export TORCHINDUCTOR_CACHE_DIR=/home/mp/.cache/torchinductor VLLM_NO_USAGE_STATS=1
export VLLM_CACHE_ROOT=/home/mp/.cache/vllm-calibration HOME=/home/mp
export CUDA_VISIBLE_DEVICES=0,2 VLLM_USE_V2_MODEL_RUNNER=1 VLLM_1CAT_ENABLE_SM70_MTP_DEFAULTS=1
CKPT=/home/mp/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4/snapshots/319f741cce68d7914884900c138a1fbb70a42f30

one () {
  local tag="$1"; shift
  echo "=== $tag ==="
  # Vorpruefung: fremder Speicher auf den Zielkarten macht die Messung wertlos.
  local used
  used=$(nvidia-smi --id=0,2 --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc)
  if [ "${used:-0}" -gt 500 ]; then
    echo "  ABBRUCH: ${used} MiB auf GPU 0/2 belegt -- Messung waere wertlos"
    return 1
  fi
  /home/mp/vllm/venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$CKPT" --served-model-name m --trust-remote-code --dtype float16 \
    --disable-custom-all-reduce --enable-prompt-tokens-details --enable-prefix-caching \
    --tensor-parallel-size 2 --pipeline-parallel-size 1 --gpu-memory-utilization 0.90 \
    --block-size 16 --max-model-len 32768 --max-num-seqs 4 --max-num-batched-tokens 2048 \
    --language-model-only --host 127.0.0.1 --port 8063 "$@" \
    > "$W/boot_$tag.log" 2>&1 &
  local S=$!
  for i in $(seq 1 90); do
    sleep 5
    kill -0 $S 2>/dev/null || { echo "  server exited"; break; }
    grep -q "Application startup complete" "$W/boot_$tag.log" && { echo "  STARTUP ${i}0s"; break; }
    grep -qi "Traceback (most recent call last)" "$W/boot_$tag.log" && { echo "  FEHLER"; break; }
  done
  if curl -s -m 8 http://127.0.0.1:8063/v1/models >/dev/null 2>&1; then
    curl -s -m 200 http://127.0.0.1:8063/v1/completions -H 'Content-Type: application/json' \
      -d '{"model":"m","prompt":"Erkläre in genau fünf Sätzen, warum der Himmel blau ist.","max_tokens":80,"temperature":0,"seed":1}' \
      > "$W/gen_$tag.json"
    python3 -c "
import json;d=json.load(open('$W/gen_$tag.json'));t=d['choices'][0]['text']
print('  nicht-Leerraum:',sum(1 for x in t if not x.isspace()),'| ',repr(t[:110]))"
    grep -o "Avg Draft acceptance rate: [0-9.]*%" "$W/boot_$tag.log" | tail -1 | sed 's/^/  /'
  fi
  for p in $(pgrep -f 'VLLM[:]:'); do kill -TERM $p 2>/dev/null; done
  kill -TERM $S 2>/dev/null; sleep 10
  for p in $(pgrep -f 'VLLM[:]:'); do kill -KILL $p 2>/dev/null; done
  kill -KILL $S 2>/dev/null; sleep 4
}

one k0
one k3_localargmax --speculative-config '{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"greedy","use_local_argmax_reduction":true,"attention_backend":"FLASH_ATTN"}'
one k3_nolocalargmax --speculative-config '{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"greedy","use_local_argmax_reduction":false,"attention_backend":"FLASH_ATTN"}'
one k1_nolocalargmax --speculative-config '{"method":"mtp","num_speculative_tokens":1,"draft_sample_method":"greedy","use_local_argmax_reduction":false,"attention_backend":"FLASH_ATTN"}'
# Ohne die Fork-Vorgaben fuer SM70-MTP: die stammen aus der V1-Zeit.
unset VLLM_1CAT_ENABLE_SM70_MTP_DEFAULTS
one k3_no1catdefaults --speculative-config '{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"greedy","use_local_argmax_reduction":false,"attention_backend":"FLASH_ATTN"}'
echo "ALLES FERTIG"
