#!/usr/bin/env bash
# Ein Zyklus der Kontextsuche: Boot (Probe-Skript, Server bleibt stehen),
# VRAM-Protokoll alle 5 s, ein Langprompt nahe max-model-len, dann die eigene
# Prozessgruppe beenden -- auch wenn der Test scheitert.
# Aufruf: EXTRA_ENV=... dsv4_ctx_cycle.sh <tag> <max_model_len> <blocks> <repeats>
set -uo pipefail
S=$(dirname "$0"); TAG=$1; MML=$2; BLK=$3; REP=$4
bash "$S/dsv4_ctx_probe.sh" "$S/ctx_$TAG" "$MML" "$BLK" 1 > "$S/ctx_$TAG.out" 2>&1
cat "$S/ctx_$TAG.out"
PGID=$(command grep -o "pgid [0-9]*" "$S/ctx_$TAG.out" | head -1 | awk '{print $2}')
if command grep -q "^== up_" "$S/ctx_$TAG.out"; then
  ( while true; do echo "$(date +%T) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd' ')"; sleep 5; done ) > "$S/ctx_${TAG}_vram.log" 2>&1 &
  LOGGER=$!
  ( cd /tmp && timeout 5400 /home/mp/vllm/venv/bin/python "$S/longctx_dsv4_long.py" http://127.0.0.1:8093 DeepSeek-V4-Flash-nvfp4-DSpark-vllm "$REP" "$S/ctx_${TAG}_long.json" ) > "$S/ctx_${TAG}_long.out" 2>&1
  echo "LANGTEST EXIT $?" >> "$S/ctx_${TAG}_long.out"
  kill "$LOGGER"
  cat "$S/ctx_${TAG}_long.out"
  awk '{for(i=2;i<=6;i++) if($i>m[i]) m[i]=$i} END{print "VRAM-Spitze GPU0..4:", m[2], m[3], m[4], m[5], m[6]}' "$S/ctx_${TAG}_vram.log"
  head -1 "$S/ctx_${TAG}_vram.log"
  command grep -c "OutOfMemoryError" "$S/ctx_$TAG/boot.log" | sed 's/^/OOM-Zeilen im Serverlog: /'
fi
[ -n "$PGID" ] && { kill -TERM -"$PGID" 2>/dev/null; sleep 20; kill -KILL -"$PGID" 2>/dev/null; }
nvidia-smi --query-gpu=memory.used --format=csv,noheader | paste -sd' '
echo ZYKLUS-ENDE
