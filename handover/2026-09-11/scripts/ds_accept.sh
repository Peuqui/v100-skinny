#!/usr/bin/env bash
# DeepSeek-V4-Flash PP5 heterogen: Kohaerenzprobe zweimal im selben Serverprozess.
set -uo pipefail
VENV=${VENV:?VENV fehlt}
REPO=/home/mp/Projekte/vllm-research/v100-skinny
W=$HOME/.cache/mtp-diagnostics/ds_$1; rm -rf "$W"; mkdir -p "$W"
PORT=19998
for i in $(seq 1 60); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc)
  [ "${u:-0}" -le 1000 ] && break; sleep 5
done
cd "$REPO"
# Eigene Sitzung/Prozessgruppe: am Ende wird nur diese beendet.
setsid env VENV="$VENV" bash scripts/serve-deepseek-het-graphs.sh > "$W/boot.log" 2>&1 &
PID=$!
echo "Server pid $PID, pgid $(ps -o pgid= -p $PID | tr -d ' ')"
STATUS=timeout
for i in $(seq 1 720); do
  curl -sf -o /dev/null --max-time 2 "http://127.0.0.1:$PORT/v1/models" && { STATUS="up_$((i*5))s"; break; }
  kill -0 $PID 2>/dev/null || { STATUS=exited; break; }
  sleep 5
done
echo "STATUS $STATUS"
if [ "${STATUS#up_}" != "$STATUS" ]; then
  for r in 1 2; do
    echo "--- Lauf $r ---"
    "$VENV/bin/python" scripts/deepseek_coherence.py --url "http://127.0.0.1:$PORT" \
      --model dsv4-manual --label "$1_run$r" --out "$W/run$r.jsonl" 2>&1
  done
fi
kill -TERM -$PID 2>/dev/null; sleep 20; kill -KILL -$PID 2>/dev/null
echo "FERTIG ds_$1"
