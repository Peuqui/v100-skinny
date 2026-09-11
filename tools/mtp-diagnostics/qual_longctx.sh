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
import hashlib, json, sys, time, urllib.request
W, PORT, SCR = sys.argv[1], sys.argv[2], sys.argv[3]
URL = f"http://127.0.0.1:{PORT}/v1/completions"
CTX = open(f"{SCR}/vorkontext.txt").read()
FRAGEN = [
    ("q1", "Erklaere die Quantenphysik in 30 Saetzen."),
    ("q2", "Erklaere den Regenbogeneffekt in 30 Saetzen."),
    # ABSICHTLICHER Schreibfehler: "Kuanda" gibt es nicht. Wer ihn erklaert,
    # halluziniert; korrekt waere eine Rueckfrage. Standard seit 30.08., siehe
    # STAND.md "Qualitaetspruefung". NICHT auf "Coanda" zurueckaendern -- dann
    # ist es keine Fangfrage mehr, sondern ein real existierender Effekt.
    ("q3", "Erklaere den Kuanda-Effekt in 30 Saetzen."),
]

def ask(prompt, mt):
    body = json.dumps({"model": "m", "prompt": prompt, "max_tokens": mt,
                       "temperature": 0, "seed": 1}).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    return time.perf_counter() - t0, d

def spec():
    try:
        m = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=30).read().decode()
    except Exception:
        return None
    a = d_ = r_ = 0.0
    for line in m.splitlines():
        if line.startswith("#"):
            continue
        if line.startswith("vllm:spec_decode_num_accepted_tokens_total"):
            a = float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:spec_decode_num_draft_tokens_total"):
            d_ = float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:spec_decode_num_drafts_total"):
            r_ = float(line.rsplit(" ", 1)[1])
    return a, d_, r_

ask("Guten Tag.", 8)                                   # warmlaufen
res, prev = {}, spec()
for tag, frage in FRAGEN:
    prompt = (CTX + "\n\nAufgabe, unabhaengig vom Hintergrundmaterial oben: "
              + frage + "\n\nAntwort:\n")
    t, d = ask(prompt, 1200)
    txt = d["choices"][0]["text"]
    u = d["usage"]
    row = {"prompt_tokens": u["prompt_tokens"], "completion_tokens": u["completion_tokens"],
           "s": round(t, 3), "tok_per_s": round(u["completion_tokens"] / t, 2),
           "finish": d["choices"][0].get("finish_reason"),
           "sha256": hashlib.sha256(txt.encode()).hexdigest()[:16]}
    cur = spec()
    if prev and cur:
        da, dd, dr = (c - p for c, p in zip(cur, prev))
        if dd:
            row["acc_rate"] = round(da / dd, 4)
        if dr:
            row["acc_len"] = round(da / dr + 1, 3)
    prev = cur
    res[tag] = row
    open(f"{W}/text_{tag}.txt", "w").write(txt)
    print(f"  {tag}: {row}", flush=True)
open(W + "/result.json", "w").write(json.dumps(res, indent=2))
PY
fi
for p in $(pgrep -f 'VLLM[:]:'); do kill -TERM $p 2>/dev/null; done
kill -TERM $S 2>/dev/null; sleep 12
for p in $(pgrep -f 'VLLM[:]:'); do kill -KILL $p 2>/dev/null; done
kill -KILL $S 2>/dev/null; wait $S 2>/dev/null
echo "FERTIG qual_$NAME ($STATUS)"
