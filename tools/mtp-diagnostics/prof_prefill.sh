#!/usr/bin/env bash
# Wohin geht die Prefill-Zeit? Kernel-Summen fuer EIN langes Prefill-Fenster.
#
#   prof_prefill.sh <name> [DEVS]      DEVS default 0,2 (Turing-Paar)
#
# Beantwortet: welchen Anteil haben die GDN-Kernel am Prefill? Davon haengt ab,
# ob ein FlashQLA-Port auf Turing ueberhaupt einen Hebel hat -- ein Kernel, der
# 5 % der Zeit stellt, kann auch verdoppelt nur 2,5 % bringen.
#
# Rezept nach docs/journal/TURING-COEXISTENCE-HANDOVER.md (2026-09-03): nsys launch/start/stop
# um das Messfenster, damit der Boot nicht mitprofiliert wird.
set -uo pipefail
NAME=${1:?name fehlt}
DEVS=${2:-0,2}

BASE=${WORKDIR:-$HOME/.cache/mtp-diagnostics}
W=$BASE/prof_$NAME
rm -rf "$W"; mkdir -p "$W"
cd "$BASE" || exit 1
SESSION="gdnprof_$$"

export PATH=/home/mp/vllm/venv/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_HOME=/home/mp/vllm/cuda TORCH_CUDA_ARCH_LIST=7.0
export NCCL_P2P_DISABLE=1 VLLM_SM70_NVFP4_TURBOMIND=1
export VLLM_SM70_QUANT_BACKEND=auto VLLM_SKINNY_NVFP4=1 VLLM_SKINNY_QPN=1 VLLM_SKINNY_QPN2=1
export VLLM_SKINNY_NVFP4_SRC=/home/mp/Projekte/v100-skinny/kernels/skinny_kernels.cu
export TORCHINDUCTOR_CACHE_DIR=/home/mp/.cache/torchinductor VLLM_NO_USAGE_STATS=1
export VLLM_CACHE_ROOT=/home/mp/.cache/vllm-calibration HOME=/home/mp
export CUDA_VISIBLE_DEVICES=$DEVS
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_1CAT_ENABLE_SM70_MTP_DEFAULTS=1
# Diagnose-Marken AUS: sie synchronisieren und wuerden das Profil verzerren.
unset VLLM_SKINNY_PPDIAG

CKPT=/home/mp/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4/snapshots/319f741cce68d7914884900c138a1fbb70a42f30

if pgrep -af 'api_server' | grep -v $$ | grep -q .; then
  echo "ABBRUCH: laeuft schon ein api_server"; exit 1
fi
used=$(nvidia-smi --id=$DEVS --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc)
[ "${used:-0}" -gt 500 ] && { echo "ABBRUCH: $used MiB auf GPU $DEVS belegt"; exit 1; }

# nsys-Sitzung anlegen, Aufzeichnung startet spaeter gezielt (--start-later)
nsys launch --session-new="$SESSION" --trace=cuda --cuda-graph-trace=node \
  --trace-fork-before-exec=true --output="$W/report" --force-overwrite=true \
  /home/mp/vllm/venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$CKPT" --served-model-name m --trust-remote-code --dtype float16 \
  --disable-custom-all-reduce --no-enable-prefix-caching \
  --tensor-parallel-size 2 --pipeline-parallel-size 1 --gpu-memory-utilization 0.90 \
  --block-size 16 --max-model-len 32768 --max-num-seqs 4 --max-num-batched-tokens 2048 \
  --language-model-only --host 127.0.0.1 --port 8066 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"greedy"}' \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8]}' > "$W/boot.log" 2>&1 &
S=$!

STATUS=timeout
for i in $(seq 1 150); do
  sleep 5
  kill -0 $S 2>/dev/null || { STATUS=exited; break; }
  grep -q "Application startup complete" "$W/boot.log" && { STATUS="up_$((i*5))s"; break; }
  grep -qi "Traceback (most recent call last)" "$W/boot.log" && { STATUS=traceback; break; }
done
echo "STATUS $STATUS"

if [ "${STATUS#up_}" != "$STATUS" ]; then
  /home/mp/vllm/venv/bin/python - "$W" "$SESSION" <<'PY'
import json, subprocess, sys, time, urllib.request
W, SESSION = sys.argv[1], sys.argv[2]
URL = "http://127.0.0.1:8066/v1/completions"

def ask(prompt, max_tokens):
    body = json.dumps({"model": "m", "prompt": prompt, "max_tokens": max_tokens,
                       "temperature": 0, "seed": 1}).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    return time.perf_counter() - t0, d["usage"]

para = ("Abschnitt {i}: Die Messung der Verarbeitungsgeschwindigkeit grosser "
        "Sprachmodelle haengt von vielen Faktoren ab, darunter die Belegung des "
        "Zwischenspeichers, die Bandbreite zwischen den Beschleunigern und die "
        "Auswahl der Rechenkerne fuer die einzelnen Schichten. ")
long_prompt = "".join(para.format(i=i) for i in range(260))

ask("Kurzer Aufwaermsatz.", 4)                       # warmlaufen, nicht im Fenster
subprocess.run(["nsys", "start", f"--session={SESSION}"], check=False)
t, u = ask(long_prompt, 1)                            # NUR der Prefill im Fenster
subprocess.run(["nsys", "stop", f"--session={SESSION}"], check=False)
res = {"prompt_tokens": u["prompt_tokens"], "prefill_s": round(t, 3),
       "prefill_tok_per_s": round(u["prompt_tokens"] / t, 1)}
print(json.dumps(res, indent=2))
open(W + "/prefill.json", "w").write(json.dumps(res, indent=2))
PY
fi

for p in $(pgrep -f 'VLLM[:]:'); do kill -TERM $p 2>/dev/null; done
kill -TERM $S 2>/dev/null; sleep 15
for p in $(pgrep -f 'VLLM[:]:'); do kill -KILL $p 2>/dev/null; done
kill -KILL $S 2>/dev/null
wait $S 2>/dev/null

# qdstrm -> nsys-rep -> sqlite -> Kernel-Summen
IMPORTER=/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter
for f in "$W"/*.qdstrm; do
  [ -f "$f" ] && [ -x "$IMPORTER" ] && "$IMPORTER" -i "$f" >/dev/null 2>&1
done
REP=$(ls -1 "$W"/*.nsys-rep 2>/dev/null | head -1)
if [ -n "$REP" ]; then
  nsys export --type sqlite --force-overwrite=true -o "$W/report.sqlite" "$REP" >/dev/null 2>&1
  echo "--- Kernel-Summen im Prefill-Fenster ---"
  python3 /home/mp/Projekte/vllm-research/v100-skinny/tools/nsys_kernsum.py "$W/report.sqlite" 2>&1 | head -45
else
  echo "kein nsys-rep erzeugt; Inhalt:"; ls -la "$W" | head
fi
echo "FERTIG prof_$NAME ($STATUS)"
