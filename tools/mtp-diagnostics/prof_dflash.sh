#!/usr/bin/env bash
# Wohin geht die DECODE-Zeit bei DFlash2? Kernel-Summen fuer ein Decode-Fenster.
#
#   prof_dflash.sh <name> [DEVS] [dflash|mtp]   DEVS default 0,2 (Turing-Paar)
#
# Beantwortet die offene Frage aus STAND.md Punkt 8: Turing faehrt DFlash2 mit
# 69,13 tok/s, erwartet waeren nach dem MTP-Verhaeltnis RTX/V100 (1,11) rund
# 82. Der Verdacht liegt auf dem Kandidaten-TopK: der QPN8-Rerank
# (_maybe_sm70_dflash2_qpn8_rerank) prueft hart auf (7, 0), also faellt Turing
# auf torch.topk ueber alle 248.320 Logits zurueck -- bei JEDEM Entwurfsschritt.
# Das ist bisher ein Verdacht aus dem Code. Dieses Skript misst ihn.
#
# Aufbau wie prof_prefill.sh (nsys launch/start/stop um das Fenster), nur liegt
# das Fenster hier auf dem Decode: kurzer Prompt, 400 Token Ausgabe.
set -uo pipefail
NAME=${1:?name fehlt}
DEVS=${2:-0,2}
MODE=${3:-dflash}

BASE=${WORKDIR:-$HOME/.cache/mtp-diagnostics}
W=$BASE/prof_$NAME
rm -rf "$W"; mkdir -p "$W"
cd "$BASE" || exit 1
SESSION="dflashprof_$$"

export PATH=/home/mp/vllm/venv/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_HOME=/home/mp/vllm/cuda TORCH_CUDA_ARCH_LIST=7.0
export NCCL_P2P_DISABLE=1 VLLM_SM70_E5_CACHE=0 VLLM_SM70_NVFP4_TURBOMIND=1
export VLLM_SM70_QUANT_BACKEND=auto VLLM_SKINNY_NVFP4=1 VLLM_SKINNY_QPN=1 VLLM_SKINNY_QPN2=1
export VLLM_SKINNY_NVFP4_SRC=/home/mp/Projekte/v100-skinny/kernels/skinny_kernels.cu
export TORCHINDUCTOR_CACHE_DIR=/home/mp/.cache/torchinductor VLLM_NO_USAGE_STATS=1
export VLLM_CACHE_ROOT=/home/mp/.cache/vllm-calibration HOME=/home/mp
export CUDA_VISIBLE_DEVICES=$DEVS
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_1CAT_ENABLE_SM70_MTP_DEFAULTS=1
export VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH=1
# Diagnose-Marken AUS: sie synchronisieren und wuerden das Profil verzerren.
unset VLLM_SKINNY_PPDIAG

CKPT=${CKPT:-/home/mp/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4/snapshots/319f741cce68d7914884900c138a1fbb70a42f30}
DRAFT=${DRAFT:-/home/mp/.cache/huggingface/hub/models--incoai--Qwen3.8-27B-DFlash2/snapshots/dedf8df68adfb1afeaf7b7480c0a0243108177b4}
case "$MODE" in
  dflash) SPEC="{\"method\":\"dflash\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":7,\"draft_sample_method\":\"greedy\"}" ;;
  mtp)    SPEC="{\"method\":\"mtp\",\"num_speculative_tokens\":3,\"draft_sample_method\":\"greedy\"}" ;;
  *)      echo "unbekannter MODE: $MODE"; exit 1 ;;
esac

if pgrep -af 'api_server' | grep -v $$ | grep -q .; then
  echo "ABBRUCH: laeuft schon ein api_server"; exit 1
fi
used=$(nvidia-smi --id=$DEVS --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc)
[ "${used:-0}" -gt 500 ] && { echo "ABBRUCH: $used MiB auf GPU $DEVS belegt"; exit 1; }

# --output gehoert an "nsys start", NICHT an "nsys launch" (nsys 2022.4.2
# lehnt es dort ab: "unrecognised option"). prof_prefill.sh hat denselben
# Fehler und ist auf dieser nsys-Version ebenfalls nicht lauffaehig.
nsys launch --session-new="$SESSION" --trace=cuda --cuda-graph-trace=node \
  --trace-fork-before-exec=true \
  /home/mp/vllm/venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$CKPT" --served-model-name m --trust-remote-code --dtype float16 \
  --disable-custom-all-reduce --no-enable-prefix-caching \
  --tensor-parallel-size 2 --pipeline-parallel-size 1 --gpu-memory-utilization 0.90 \
  --block-size 16 --max-model-len 32768 --max-num-seqs 4 --max-num-batched-tokens 2048 \
  --language-model-only --host 127.0.0.1 --port 8066 \
  --speculative-config "$SPEC" \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8]}' > "$W/boot.log" 2>&1 &
S=$!

STATUS=timeout
for i in $(seq 1 240); do
  sleep 5
  kill -0 $S 2>/dev/null || { STATUS=exited; break; }
  grep -q "Application startup complete" "$W/boot.log" && { STATUS="up_$((i*5))s"; break; }
  grep -qi "Traceback (most recent call last)" "$W/boot.log" && { STATUS=traceback; break; }
done
echo "STATUS $STATUS  ($NAME: $MODE, DEVS=$DEVS)"

if [ "${STATUS#up_}" != "$STATUS" ]; then
  /home/mp/vllm/venv/bin/python - "$W" "$SESSION" <<'PY'
import json, subprocess, sys, time, urllib.request
W, SESSION = sys.argv[1], sys.argv[2]
URL = "http://127.0.0.1:8066/v1/completions"
PROMPT = "Erklaere die Quantenphysik in 20 Saetzen."

def ask(prompt, max_tokens):
    body = json.dumps({"model": "m", "prompt": prompt, "max_tokens": max_tokens,
                       "temperature": 0, "seed": 1}).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    return time.perf_counter() - t0, d["usage"]

ask(PROMPT, 8)                                        # warmlaufen, nicht im Fenster
subprocess.run(["nsys", "start", f"--session={SESSION}",
                "--output", W + "/report", "--force-overwrite=true"], check=False)
t, u = ask(PROMPT, 400)                               # das Decode-Fenster
subprocess.run(["nsys", "stop", f"--session={SESSION}"], check=False)
res = {"completion_tokens": u["completion_tokens"], "decode_s": round(t, 3),
       "tok_per_s": round(u["completion_tokens"] / t, 2)}
print(json.dumps(res, indent=2))
open(W + "/decode.json", "w").write(json.dumps(res, indent=2))
PY
fi

for p in $(pgrep -f 'VLLM[:]:'); do kill -TERM $p 2>/dev/null; done
kill -TERM $S 2>/dev/null; sleep 15
for p in $(pgrep -f 'VLLM[:]:'); do kill -KILL $p 2>/dev/null; done
kill -KILL $S 2>/dev/null
wait $S 2>/dev/null

IMPORTER=/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter
for f in "$W"/*.qdstrm; do
  [ -f "$f" ] && [ -x "$IMPORTER" ] && "$IMPORTER" -i "$f" >/dev/null 2>&1
done
REP=$(ls -1 "$W"/*.nsys-rep 2>/dev/null | head -1)
if [ -n "$REP" ]; then
  nsys export --type sqlite --force-overwrite=true -o "$W/report.sqlite" "$REP" >/dev/null 2>&1
  echo "--- Kernel-Summen im Decode-Fenster ---"
  python3 /home/mp/Projekte/vllm-research/v100-skinny/tools/nsys_kernsum.py "$W/report.sqlite" 2>&1 | head -45
else
  echo "kein nsys-rep erzeugt; Inhalt:"; ls -la "$W" | head
fi
echo "FERTIG prof_$NAME ($STATUS)"
