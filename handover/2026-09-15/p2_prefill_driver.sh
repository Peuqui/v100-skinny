#!/usr/bin/env bash
# Paket 2: Prefill-Messung gegen die Kaskade (Startzeile aus dem Boot von p2_host2).
# Laeuft abgekoppelt; Ende markiert p2_prefill.DONE.
set -uo pipefail
HERE=$(dirname "$(readlink -f "$0")")
PY=/home/mp/vllm/venv/bin/python
SWAP=http://127.0.0.1:11435
STOP=/home/mp/Projekte/AIfred-Intelligence/scripts/vllm-swap-stop
ENTRY=Qwen3.8-Flash-Next-180B-A4B-NVFP4-MTP-vllm
PORT=8093
cd /tmp
rm -f "$HERE/p2_prefill.DONE"

echo "=== entlade Produktion $(date +%H:%M:%S)"
for M in $(curl -s --max-time 5 $SWAP/running | $PY -c 'import json,sys; print(" ".join(r["model"] for r in json.load(sys.stdin)["running"] if "Flash-Next" in r["model"]))'); do
  curl -s -X POST --max-time 180 "$SWAP/api/models/unload/$M" >/dev/null
done
for _ in $(seq 1 60); do
  busy=$(nvidia-smi -i 0,1,2,3 --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1000' | wc -l)
  [ "$busy" -eq 0 ] && break
  sleep 5
done

echo "=== Kaskaden-Boot $(date +%H:%M:%S)"
setsid bash "$HERE/p2_host2/cascade.launch.sh" > "$HERE/p2_prefill_cascade.boot.log" 2>&1 &
PID=$!
STATE=timeout
T0=$(date +%s)
for _ in $(seq 1 360); do
  curl -sf -o /dev/null --max-time 2 http://127.0.0.1:$PORT/v1/models && { STATE="up_$(( $(date +%s) - T0 ))s"; break; }
  kill -0 $PID 2>/dev/null || { STATE=exited; break; }
  sleep 5
done
echo "=== Boot: $STATE"
if [ "${STATE#up_}" != "$STATE" ]; then
  $PY "$HERE/ple_prefill.py" http://127.0.0.1:$PORT "$ENTRY" kaskade "$HERE/p2_prefill_cascade.json"
fi
echo "=== stoppe Kaskade $(date +%H:%M:%S)"
$STOP "$PID"
date > "$HERE/p2_prefill.DONE"
echo "FERTIG $(date +%H:%M:%S)"
