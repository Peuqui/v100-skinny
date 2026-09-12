#!/usr/bin/env bash
# Paket C Messung: 27B auf dem RTX-Paar, Server wie speed_dflash.sh, dann (a) 400-Token-Kurzprobe,
# (b) 13k-Vorkontext: TTFT (max_tokens=1) und Decode (200 Token). Aufruf:
#   fa2_probe.sh <name> <attn: FLASH_ATTN|TRITON_ATTN> <kv: auto|float16>
set -uo pipefail
NAME=$1; ATTN=$2; KV=$3
REPO=/home/mp/Projekte/vllm-research/v100-skinny; V=$REPO/.venv-pr-turing
W=$HOME/.cache/mtp-diagnostics/fa2_$NAME; rm -rf "$W"; mkdir -p "$W"; cd $HOME/.cache/mtp-diagnostics
export PATH="$V/bin":/home/mp/vllm/cuda/bin:/usr/local/bin:/usr/bin:/bin
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_HOME=/home/mp/vllm/cuda TORCH_CUDA_ARCH_LIST=7.0
export NCCL_P2P_DISABLE=1 VLLM_SM70_NVFP4_TURBOMIND=1 VLLM_SM70_QUANT_BACKEND=auto
export TORCHINDUCTOR_CACHE_DIR=/home/mp/.cache/torchinductor VLLM_NO_USAGE_STATS=1
export VLLM_CACHE_ROOT=/home/mp/.cache/vllm-calibration HOME=/home/mp
export CUDA_VISIBLE_DEVICES=0,2 VLLM_USE_V2_MODEL_RUNNER=1 VLLM_1CAT_ENABLE_SM70_MTP_DEFAULTS=1
export VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH=1 VLLM_SM70_DFLASH2_QUANT_LM_HEAD=1
CKPT=/home/mp/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4/snapshots/319f741cce68d7914884900c138a1fbb70a42f30
if pgrep -f '^[^ ]*python[^ ]* -m vllm.entrypoints.openai.api_server' >/dev/null; then echo "ABBRUCH: api_server laeuft"; exit 1; fi
"$V/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$CKPT" --served-model-name m --trust-remote-code --dtype float16 \
  --disable-custom-all-reduce --no-enable-prefix-caching \
  --tensor-parallel-size 2 --pipeline-parallel-size 1 --gpu-memory-utilization 0.90 \
  --block-size 16 --max-model-len 32768 --max-num-seqs 4 --max-num-batched-tokens 2048 \
  --language-model-only --host 127.0.0.1 --port 8066 \
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":3,\"draft_sample_method\":\"greedy\",\"attention_backend\":\"$ATTN\"}" \
  --attention-backend "$ATTN" --kv-cache-dtype "$KV" \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8]}' > "$W/boot.log" 2>&1 &
S=$!; STATUS=timeout
for i in $(seq 1 180); do sleep 5; kill -0 $S 2>/dev/null || { STATUS=exited; break; }; grep -q "Application startup complete" "$W/boot.log" && { STATUS="up_$((i*5))s"; break; }; grep -qi "Traceback (most recent call last)" "$W/boot.log" && { STATUS=traceback; break; }; done
echo "STATUS $STATUS ($NAME: $ATTN, kv=$KV)"
if [ "${STATUS#up_}" != "$STATUS" ]; then
"$V/bin/python" - "$W" "$REPO/tools/mtp-diagnostics/vorkontext.txt" <<'PY'
import hashlib, json, sys, time, urllib.request
W, CTXF = sys.argv[1], sys.argv[2]
URL = "http://127.0.0.1:8066/v1/completions"
def ask(prompt, mt):
    b = json.dumps({"model": "m", "prompt": prompt, "max_tokens": mt, "temperature": 0, "seed": 1}).encode()
    t0 = time.perf_counter()
    with urllib.request.urlopen(urllib.request.Request(URL, data=b, headers={"Content-Type": "application/json"}), timeout=1800) as r:
        d = json.load(r)
    return time.perf_counter() - t0, d
ask("Erklaere die Quantenphysik in 20 Saetzen.", 8)
res = {}
rows = []
for i in range(5):
    t, d = ask("Erklaere die Quantenphysik in 20 Saetzen.", 400)
    rows.append(d["usage"]["completion_tokens"] / t); txt = d["choices"][0]["text"]
rows.sort(); res["kurz_tok_s_median"] = round(rows[2], 2); res["kurz_sha"] = hashlib.sha256(txt.encode()).hexdigest()[:16]
ctx = open(CTXF).read() + "\n\nAufgabe, unabhaengig vom Hintergrundmaterial oben: Erklaere die Quantenphysik in 20 Saetzen."
tt = []
for i in range(3):
    t, d = ask(ctx, 1); tt.append(t)
tt.sort(); res["ctx_prompt_tokens"] = d["usage"]["prompt_tokens"]; res["ttft13k_s_median"] = round(tt[1], 2)
dr = []
for i in range(3):
    t, d = ask(ctx, 200); dr.append(d["usage"]["completion_tokens"] / (t - tt[1]))
dr.sort(); res["decode13k_tok_s_median"] = round(dr[1], 2); res["ctx_sha"] = hashlib.sha256(d["choices"][0]["text"].encode()).hexdigest()[:16]
print("ERGEBNIS", json.dumps(res)); open(W + "/result.json", "w").write(json.dumps(res, indent=2))
PY
fi
echo "   attn: $(grep -a -o -m1 'Loaded FA2 library [^.]*\|Using [A-Z_]* attention backend' "$W/boot.log" | head -1) | kv: $(grep -a -o -m1 'kv_cache_dtype=[a-z0-9_]*' "$W/boot.log")"
grep -a -m1 -E "Error:|Error\b" "$W/boot.log" | sed 's/.*\] //' | cut -c1-160
for p in $(pgrep -f 'VLLM[:]:'); do kill -TERM $p 2>/dev/null; done; kill -TERM $S 2>/dev/null; sleep 12
for p in $(pgrep -f 'VLLM[:]:'); do kill -KILL $p 2>/dev/null; done; kill -KILL $S 2>/dev/null; wait $S 2>/dev/null
echo "FERTIG fa2_$NAME ($STATUS)"
