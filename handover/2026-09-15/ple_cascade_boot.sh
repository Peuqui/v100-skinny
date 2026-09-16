#!/usr/bin/env bash
# PLE-Kaskade Paket 1: Durchstich-Boot und Kontroll-Boot (Entwurf Abschnitt 10).
#
# Phase 1: Flash-Next-Eintrag aus llama-swap exakt nachgebaut, Port 8093, mit
#          genau zwei Abweichungen: CUDA_VISIBLE_DEVICES=0,2,1,3,4 und
#          VLLM_QWEN4EXP_PLE_STORE_DEVICE=4. Sonde, Log-Belege, Stopp.
# Phase 2: Kontroll-Boot ueber llama-swap mit UNVERAENDERTEM Eintrag (derselbe,
#          den AIfred geladen hatte), Sonde. Stellt die Produktion wieder her.
# Vergleich beider Sonden gegen ref_prod.json (Produktion vor der Aenderung).
#
# Aufruf: [CASCADE_ENV="K=V K=V"] ple_cascade_boot.sh OUTDIR [SWAP_MODEL] [REF_JSON] [SKIP_CONTROL=1]
# CASCADE_ENV: Kaskaden-Werte fuer Phase 1 (Paket 2: STORE_DEVICE, STORE_GIB, HOST_GIB).
set -uo pipefail
CASCADE_ENV=${CASCADE_ENV:-VLLM_QWEN4EXP_PLE_STORE_DEVICE=4}
OUT=$1
SWAP_MODEL=${2:-Qwen3.8-Flash-Next-180B-A4B-NVFP4-MTP-vllm-vlm-qwen3vl4b}
REF=${3:-ref_prod.json}
SKIP_CONTROL=${4:-0}
HERE=$(dirname "$(readlink -f "$0")")
PY=/home/mp/vllm/venv/bin/python
ENTRY=Qwen3.8-Flash-Next-180B-A4B-NVFP4-MTP-vllm
PORT=8093
SWAP=http://127.0.0.1:11435
STOP=/home/mp/Projekte/AIfred-Intelligence/scripts/vllm-swap-stop
mkdir -p "$OUT"; cd /tmp

log() { echo "[$(date +%H:%M:%S)] $*"; }
compute_gpus_free() {  # GPUs 0-3 (Rechenkarten); GPU 4 traegt VLM/TTS
  for _ in $(seq 1 60); do
    busy=$(nvidia-smi -i 0,1,2,3 --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1000' | wc -l)
    [ "$busy" -eq 0 ] && return 0
    sleep 5
  done
  return 1
}
snapshot() {  # $1=label
  { echo "== $1 $(date +%H:%M:%S)"; grep -E "MemAvailable|SwapFree|Shmem:" /proc/meminfo
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader; } >> "$OUT/memory.txt"
}

log "Phase 0: laufende Flash-Next-Varianten entladen"
for M in $(curl -s --max-time 5 $SWAP/running | $PY -c 'import json,sys; print(" ".join(r["model"] for r in json.load(sys.stdin)["running"] if "Flash-Next" in r["model"]))'); do
  log "  entlade $M"; curl -s -X POST --max-time 120 "$SWAP/api/models/unload/$M" >/dev/null
done
compute_gpus_free || { log "Rechenkarten nicht frei, Abbruch"; exit 1; }
snapshot "vor Phase 1"

log "Phase 1: Kaskaden-Boot aus Eintrag $ENTRY"
$PY - "$ENTRY" "$OUT/cascade.launch.sh" "$PORT" "$CASCADE_ENV" <<'PY'
import shlex, sys, yaml
name, out, port, cascade_env = sys.argv[1:5]
entry = yaml.safe_load(open("/home/mp/.config/llama-swap/config.yaml"))["models"][name]
cmd = " ".join(entry["cmd"].split()).replace("${PORT}", port)
overrides = {"CUDA_VISIBLE_DEVICES": "0,2,1,3,4"}
overrides.update(item.split("=", 1) for item in cascade_env.split())
env = dict(e.split("=", 1) for e in entry.get("env", []))
if env.get("CUDA_VISIBLE_DEVICES") != "0,2,1,3":
    sys.exit(f"Eintrag hat CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')}, erwartet 0,2,1,3")
env.update(overrides)
with open(out, "w") as f:
    f.write("#!/usr/bin/env bash\n")
    for k, v in env.items():
        f.write(f"export {k}={shlex.quote(v)}\n")
    f.write("exec " + cmd + "\n")
print("launch geschrieben, Abweichungen:", overrides)
PY
[ -s "$OUT/cascade.launch.sh" ] || { log "launch.sh fehlt, Abbruch"; exit 1; }
bash "$HERE/memsample.sh" "$OUT/cascade.mem.csv" &
SAMPLER=$!
setsid bash "$OUT/cascade.launch.sh" > "$OUT/cascade.boot.log" 2>&1 &
PID=$!; T0=$(date +%s); STATE=timeout
for _ in $(seq 1 480); do
  curl -sf -o /dev/null --max-time 2 http://127.0.0.1:$PORT/v1/models && { STATE="up_$(( $(date +%s) - T0 ))s"; break; }
  kill -0 $PID 2>/dev/null || { STATE=exited; break; }
  sleep 5
done
log "  Boot: $STATE"
kill "$SAMPLER" 2>/dev/null
snapshot "Phase 1 nach Boot ($STATE)"
if [ "${STATE#up_}" != "$STATE" ]; then
  $PY "$HERE/ple_probe.py" http://127.0.0.1:$PORT "$ENTRY" kaskade "$OUT/cascade.json" | tee "$OUT/cascade.probe.txt"
  $PY "$HERE/ple_logprobs.py" http://127.0.0.1:$PORT "$ENTRY" "$OUT/cascade_logprobs.json"
fi
{ echo "Tracebacks: $(grep -c Traceback "$OUT/cascade.boot.log")"
  grep -E "PleOffload|PLE cascade|remote placements|Worker ready|PLE table placement|PLE auto placement|PLE host share|host budget cut|store rows|Registrations complete|store device|overflow cascade|Error|error" "$OUT/cascade.boot.log" | cut -c1-600
} > "$OUT/cascade.evidence.txt"
cat "$OUT/cascade.evidence.txt"
log "  stoppe Kaskaden-Server (pid $PID)"
$STOP "$PID"
compute_gpus_free || { log "Rechenkarten nach Phase 1 nicht frei, Abbruch vor Phase 2"; exit 1; }

if [ "$SKIP_CONTROL" = 1 ]; then
  log "Phase 2 uebersprungen, Karten bleiben leer"
else
log "Phase 2: Kontroll-Boot ueber llama-swap, Eintrag $SWAP_MODEL unveraendert"
T0=$(date +%s)
curl -s --max-time 2400 "$SWAP/v1/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$SWAP_MODEL\",\"prompt\":\"ok\",\"max_tokens\":1}" > "$OUT/control.warmup.json"
log "  Kontroll-Boot bereit nach $(( $(date +%s) - T0 ))s"
snapshot "Phase 2 nach Boot"
$PY "$HERE/ple_probe.py" "$SWAP" "$SWAP_MODEL" kontrolle "$OUT/control.json" | tee "$OUT/control.probe.txt"

fi

log "Vergleich gegen Referenz $REF"
$PY - "$HERE/$REF" "$OUT/cascade.json" "$OUT/control.json" <<'PY'
import json, os, sys
ref = json.load(open(sys.argv[1]))
for path in sys.argv[2:]:
    if not os.path.exists(path):
        print(f"{os.path.basename(path)}: fehlt"); continue
    run = json.load(open(path))
    same = [a["sha"] == b["sha"] for a, b in zip(ref["runs"], run["runs"])]
    speed = [f'{b["tok_s"]:.1f}/{a["tok_s"]:.1f}' for a, b in zip(ref["runs"], run["runs"])]
    print(f'{run["label"]}: bitgleich {same} (alle: {all(same)}), tok/s neu/ref {speed}')
    for a, b in zip(ref["runs"], run["runs"]):
        if "text" in a and "text" in b and a["text"] != b["text"]:
            n = next((i for i, (x, y) in enumerate(zip(a["text"], b["text"])) if x != y), min(len(a["text"]), len(b["text"])))
            print(f'  p{a["prompt"]} weicht ab Zeichen {n} von {len(a["text"])} ab')
            print(f'    ref: {a["text"][max(0, n - 60):n + 40]!r}')
            print(f'    neu: {b["text"][max(0, n - 60):n + 40]!r}')
PY
log "FERTIG"
