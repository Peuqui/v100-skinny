#!/usr/bin/env bash
# Parametrisierte MTP-V2-Sonde, 27B auf zwei RTX 8000 (GPU 0+2), k=3.
# Aufruf: probe.sh <name> [zusaetzliche server-args ...]
# Env-Overrides werden vom Aufrufer vorher exportiert (z.B. VLLM_ATTENTION_BACKEND).
set -uo pipefail
NAME=${1:?name fehlt}
shift || true

BASE=${WORKDIR:-$HOME/.cache/mtp-diagnostics}
W=$BASE/runs/$NAME
rm -rf "$W"; mkdir -p "$W"
# cwd MUSS neutral sein: im vLLM-Paketverzeichnis schattet vllm/tokenizers die
# echte tokenizers-Bibliothek (zirkulaerer Import in transformers).
cd "$BASE" || exit 1

export PATH=/home/mp/vllm/venv/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_HOME=/home/mp/vllm/cuda TORCH_CUDA_ARCH_LIST=7.0
export NCCL_P2P_DISABLE=1 VLLM_SM70_E5_CACHE=0
export VLLM_SM70_QUANT_BACKEND=${VLLM_SM70_QUANT_BACKEND:-auto}
export VLLM_SM70_NVFP4_TURBOMIND=${VLLM_SM70_NVFP4_TURBOMIND:-1}
export VLLM_SKINNY_NVFP4=${VLLM_SKINNY_NVFP4:-1}
export VLLM_SKINNY_QPN=${VLLM_SKINNY_QPN:-1}
export VLLM_SKINNY_QPN2=${VLLM_SKINNY_QPN2:-1}
export VLLM_SKINNY_NVFP4_SRC=/home/mp/Projekte/v100-skinny/kernels/skinny_kernels.cu
export TORCHINDUCTOR_CACHE_DIR=/home/mp/.cache/torchinductor VLLM_NO_USAGE_STATS=1
export VLLM_CACHE_ROOT=/home/mp/.cache/vllm-calibration HOME=/home/mp
export CUDA_VISIBLE_DEVICES=${DEVS:-0,2}
export VLLM_USE_V2_MODEL_RUNNER=${VLLM_USE_V2_MODEL_RUNNER:-1}
export VLLM_1CAT_ENABLE_SM70_MTP_DEFAULTS=${VLLM_1CAT_ENABLE_SM70_MTP_DEFAULTS:-1}
export VLLM_SKINNY_PPDIAG=$W/ppdiag VLLM_SKINNY_PPDIAG_CAP=${VLLM_SKINNY_PPDIAG_CAP:-60}

CKPT=/home/mp/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4/snapshots/319f741cce68d7914884900c138a1fbb70a42f30
K=${K:-3}
CGMODE=${CGMODE:-}
if [ -n "$CGMODE" ]; then
  COMPCFG='{"cudagraph_capture_sizes":[1,2,4,8],"cudagraph_mode":"'$CGMODE'"}'
else
  COMPCFG='{"cudagraph_capture_sizes":[1,2,4,8]}'
fi
SPEC=${SPEC:-'{"method":"mtp","num_speculative_tokens":'$K',"draft_sample_method":"greedy","use_local_argmax_reduction":false,"attention_backend":"FLASH_ATTN"}'}

# Genau ein Lauf gleichzeitig, und keine Fremdbelegung auf den Karten.
if pgrep -af 'api_server' | grep -v $$ | grep -q .; then
  echo "ABBRUCH: laeuft schon ein api_server"; pgrep -af 'api_server'; exit 1
fi
used=$(nvidia-smi --id=$CUDA_VISIBLE_DEVICES --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc)
[ "${used:-0}" -gt 500 ] && { echo "ABBRUCH: $used MiB auf GPU $CUDA_VISIBLE_DEVICES belegt"; exit 1; }

env | grep -E '^(VLLM|CUDA_VIS|TORCH_CUDA)' | sort > "$W/env.txt"
echo "server-args: $*" >> "$W/env.txt"

/home/mp/vllm/venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$CKPT" --served-model-name m --trust-remote-code --dtype float16 \
  --disable-custom-all-reduce --enable-prompt-tokens-details --enable-prefix-caching \
  --tensor-parallel-size ${TP:-2} --pipeline-parallel-size ${PP:-1} --gpu-memory-utilization 0.90 \
  --block-size 16 --max-model-len 32768 --max-num-seqs 4 --max-num-batched-tokens 2048 \
  --language-model-only --host 127.0.0.1 --port 8064 \
  --speculative-config "$SPEC" \
  --compilation-config "$COMPCFG" \
  "$@" > "$W/boot.log" 2>&1 &
S=$!

STATUS=timeout
for i in $(seq 1 120); do
  sleep 5
  kill -0 $S 2>/dev/null || { STATUS=exited; break; }
  grep -q "Application startup complete" "$W/boot.log" && { STATUS="up_$((i*5))s"; break; }
  grep -qi "Traceback (most recent call last)" "$W/boot.log" && { STATUS=traceback; break; }
done
echo "STATUS $STATUS"

if [ "${STATUS#up_}" != "$STATUS" ]; then
  curl -s -m 240 http://127.0.0.1:8064/v1/completions -H 'Content-Type: application/json' \
    -d '{"model":"m","prompt":"Erkläre in genau fünf Sätzen, warum der Himmel blau ist.","max_tokens":24,"temperature":0,"seed":1}' \
    > "$W/gen.json"
  echo "--- Ausgabe ---"; python3 -c "
import json,sys
try:
    d=json.load(open('$W/gen.json'))
    print(repr(d['choices'][0]['text']))
except Exception as e:
    print('gen-Fehler:', e, open('$W/gen.json').read()[:200])
"
fi

for p in $(pgrep -f 'VLLM[:]:'); do kill -TERM $p 2>/dev/null; done
kill -TERM $S 2>/dev/null; sleep 12
for p in $(pgrep -f 'VLLM[:]:'); do kill -KILL $p 2>/dev/null; done
kill -KILL $S 2>/dev/null
wait $S 2>/dev/null

echo "--- NaN-Marken (target.hidden_out) ---"
grep -h "target.hidden_out" "$W"/ppdiag.rank0.*.log 2>/dev/null | sed -E 's/.*(step=[0-9]+).*hs_nan=[^[]*\[([^]]*)\].*hs_shape0=([0-9]+).*/\1 nan=[\2] toks=\3/' | head -12
echo "FERTIG $NAME ($STATUS)"
