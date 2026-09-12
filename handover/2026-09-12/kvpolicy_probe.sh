#!/usr/bin/env bash
# E2E-Nachweis KV-Quant-Politik: reines Main + Patch, 27B-NVFP4 (Checkpoint kv_cache_quant_algo FP8), RTX-Paar,
# --kv-cache-dtype NICHT gesetzt (auto), TRITON_ATTN (FlashInfer stuerzt auf sm75, FA2 gibt es auf reinem Main nicht).
# Erwartung: Log "Ignoring the checkpoint's KV-cache quantization directive", kv_cache_dtype=auto/float16, Boot ok, Text.
set -uo pipefail
NAME=$1; TREE=$2
V=/home/mp/Projekte/vllm-research/v100-skinny/.venv-pr-turing
CKPT=/home/mp/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4/snapshots/319f741cce68d7914884900c138a1fbb70a42f30
W=$HOME/.cache/mtp-diagnostics/kvpolicy_$NAME; rm -rf "$W"; mkdir -p "$W"; cd $HOME/.cache/mtp-diagnostics
export CUDA_VISIBLE_DEVICES=${GPUS:-0,2} CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_HOME=/home/mp/vllm/cuda HOME=/home/mp
export PYTHONPATH=$TREE VLLM_CACHE_ROOT=/home/mp/.cache/vllm-calibration TORCHINDUCTOR_CACHE_DIR=/home/mp/.cache/torchinductor
export VLLM_SM70_NVFP4_TURBOMIND=1 VLLM_SM70_QUANT_BACKEND=auto NCCL_P2P_DISABLE=1
ATTN=(); SPEC_ATTN=""; if [ -n "${ATTN_BACKEND:-}" ]; then ATTN=(--attention-backend "$ATTN_BACKEND"); SPEC_ATTN=",\"attention_backend\":\"$ATTN_BACKEND\""; fi
if pgrep -f '^[^ ]*python[^ ]* -m vllm.entrypoints.openai.api_server' >/dev/null; then echo "ABBRUCH: api_server laeuft"; exit 1; fi
"$V/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$CKPT" --served-model-name m --trust-remote-code --dtype float16 \
  --disable-custom-all-reduce --no-enable-prefix-caching \
  --tensor-parallel-size 2 --pipeline-parallel-size 1 --gpu-memory-utilization 0.90 \
  --block-size 16 --max-model-len 32768 --max-num-seqs 4 --max-num-batched-tokens 2048 \
  --language-model-only --host 127.0.0.1 --port 8066 \
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":3,\"draft_sample_method\":\"greedy\"$SPEC_ATTN}" "${ATTN[@]}" \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8]}' > "$W/boot.log" 2>&1 &
S=$!; STATUS=timeout
for i in $(seq 1 180); do sleep 5; kill -0 $S 2>/dev/null || { STATUS=exited; break; }; grep -q "Application startup complete" "$W/boot.log" && { STATUS="up_$((i*5))s"; break; }; grep -qi "Traceback (most recent call last)" "$W/boot.log" && { STATUS=traceback; break; }; done
if [ "${STATUS#up_}" != "$STATUS" ]; then
"$V/bin/python" - "$W" <<'PY'
import hashlib, json, sys, time, urllib.request
W = sys.argv[1]
def chat(prompt, mt):
    b = json.dumps({"model": "m", "messages": [{"role": "user", "content": prompt}], "max_tokens": mt,
                    "temperature": 0, "seed": 1, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    t0 = time.perf_counter()
    with urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8066/v1/chat/completions", data=b, headers={"Content-Type": "application/json"}), timeout=1800) as r:
        d = json.load(r)
    return time.perf_counter() - t0, d
chat("Erklaere die Quantenphysik in 20 Saetzen.", 8)
rows = []
for i in range(3):
    t, d = chat("Erklaere die Quantenphysik in 20 Saetzen.", 200); rows.append(d["usage"]["completion_tokens"] / t)
    txt = d["choices"][0]["message"]["content"] or ""
rows.sort()
print("ERGEBNIS", json.dumps({"tok_s_median": round(rows[1], 2), "completion_tokens": d["usage"]["completion_tokens"], "sha": hashlib.sha256(txt.encode()).hexdigest()[:16], "head": txt[:120]}))
PY
fi
kill $S 2>/dev/null; sleep 3; kill -9 $S 2>/dev/null
echo "STATUS $STATUS ($NAME)"
echo "   politik: $(grep -a -o -m1 'Ignoring the checkpoint.s KV-cache quantization directive ([a-z0-9_]*)' "$W/boot.log")"
echo "   kv: $(grep -a -o -m1 'kv_cache_dtype=[a-z0-9_]*' "$W/boot.log") | fp8-log: $(grep -a -c 'data type to store kv cache' "$W/boot.log") | attn: $(grep -a -o -m1 'Using [A-Z_]* attention backend' "$W/boot.log")"
grep -a -m1 -E "Error:|Error\b" "$W/boot.log" | sed 's/.*\] //' | cut -c1-160
echo "FERTIG $NAME"
