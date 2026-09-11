#!/usr/bin/env bash
# Verlustfreiheit der Spekulation pruefen: gleiche Frage, greedy, Text speichern.
#   qual_probe.sh <name> <fork|upstream> <k>        k=0 heisst ohne Spekulation
set -uo pipefail
NAME=${1:?name fehlt}; MODULE=${2:?fork|upstream fehlt}; K=${3:?k fehlt}
DEVS=${DEVS:-0,2}
TOOLDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE=${WORKDIR:-$HOME/.cache/mtp-diagnostics}
W=$BASE/qual_$NAME; rm -rf "$W"; mkdir -p "$W"; cd "$BASE" || exit 1

export PATH=/home/mp/vllm/venv/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_HOME=/home/mp/vllm/cuda TORCH_CUDA_ARCH_LIST=7.0
export NCCL_P2P_DISABLE=1 VLLM_SM70_NVFP4_TURBOMIND=1
export VLLM_SM70_QUANT_BACKEND=auto VLLM_SKINNY_NVFP4=1 VLLM_SKINNY_QPN=1 VLLM_SKINNY_QPN2=1
export VLLM_SKINNY_NVFP4_SRC=/home/mp/Projekte/v100-skinny/kernels/skinny_kernels.cu
export TORCHINDUCTOR_CACHE_DIR=/home/mp/.cache/torchinductor VLLM_NO_USAGE_STATS=1
export VLLM_CACHE_ROOT=/home/mp/.cache/vllm-calibration HOME=/home/mp
export CUDA_VISIBLE_DEVICES=$DEVS
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_1CAT_ENABLE_SM70_MTP_DEFAULTS=1
unset VLLM_SKINNY_PPDIAG
[ "$MODULE" = "upstream" ] && export AIFRED_FORCE_UPSTREAM_GDN=1
# Fix 2 (Baseline-Tor auf pre-Ampere) wirkt wie dieser Schalter; der Fork
# laeuft auf Turing-only produktiv ohne ihn (kein Auto-Setting).
[ "$MODULE" = "upstream" ] && export VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH=1

PORT=8066
CKPT=/home/mp/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4/snapshots/319f741cce68d7914884900c138a1fbb70a42f30
SPEC=()
[ "$K" != "0" ] && SPEC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$K,\"draft_sample_method\":\"greedy\"}")

if pgrep -af 'api_server' | grep -v $$ | grep -q .; then echo "ABBRUCH: api_server laeuft"; exit 1; fi
used=$(nvidia-smi --id=$DEVS --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc)
[ "${used:-0}" -gt 500 ] && { echo "ABBRUCH: $used MiB belegt"; exit 1; }

/home/mp/vllm/venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$CKPT" --served-model-name m --trust-remote-code --dtype float16 \
  --disable-custom-all-reduce --no-enable-prefix-caching \
  --tensor-parallel-size 2 --pipeline-parallel-size 1 --gpu-memory-utilization 0.90 \
  --block-size 16 --max-model-len 32768 --max-num-seqs 4 --max-num-batched-tokens 2048 \
  --language-model-only --host 127.0.0.1 --port $PORT "${SPEC[@]}" \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8]}' > "$W/boot.log" 2>&1 &
S=$!
STATUS=timeout
for i in $(seq 1 150); do
  sleep 5
  kill -0 $S 2>/dev/null || { STATUS=exited; break; }
  grep -q "Application startup complete" "$W/boot.log" && { STATUS="up_$((i*5))s"; break; }
  grep -qi "Traceback (most recent call last)" "$W/boot.log" && { STATUS=traceback; break; }
done
echo "STATUS $STATUS  ($NAME: $MODULE, k=$K)"

if [ "${STATUS#up_}" != "$STATUS" ]; then
  /home/mp/vllm/venv/bin/python - "$W" "$PORT" "$TOOLDIR" <<'PY'
import json, sys, time, urllib.request
W, PORT, SCR = sys.argv[1], sys.argv[2], sys.argv[3]
URL = f"http://127.0.0.1:{PORT}/v1/completions"
lines = open(f"{SCR}/vorkontext.txt").read().split("\n")
head, body = lines[0], lines[2:]
FRAGE = "Erklaere die Quantenphysik in 30 Saetzen."

def ask(prompt, mt):
    b = json.dumps({"model": "m", "prompt": prompt, "max_tokens": mt,
                    "temperature": 0, "seed": 1}).encode()
    r = urllib.request.Request(URL, data=b, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(r, timeout=1800) as resp:
        d = json.load(resp)
    return time.perf_counter() - t0, d

ask("Guten Tag.", 8)
res = {}
for n in (0, 5, 10, 20, 40, 60, 80, 110, 140, 170):
    ctx = "" if n == 0 else head + "\n\n" + "\n".join(body[:n]) + "\n\n"
    prompt = (ctx + ("Aufgabe, unabhaengig vom Hintergrundmaterial oben: " if n else "")
              + FRAGE + "\n\nAntwort:\n")
    t, d = ask(prompt, 260)
    txt = d["choices"][0]["text"]
    pt = d["usage"]["prompt_tokens"]
    res[str(n)] = {"absaetze": n, "prompt_tokens": pt, "s": round(t, 2), "text": txt}
    print(f"  {n:3d} Abs / {pt:6d} Tok: {txt[:90]!r}", flush=True)
    open(f"{W}/scan_{n:03d}.txt", "w").write(txt)
open(W + "/result.json", "w").write(json.dumps(res, indent=2))
PY
fi
for p in $(pgrep -f 'VLLM[:]:'); do kill -TERM $p 2>/dev/null; done
kill -TERM $S 2>/dev/null; sleep 12
for p in $(pgrep -f 'VLLM[:]:'); do kill -KILL $p 2>/dev/null; done
kill -KILL $S 2>/dev/null; wait $S 2>/dev/null
echo "FERTIG qual_$NAME ($STATUS)"
