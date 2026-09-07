#!/usr/bin/env bash
# Totmannschalter: beendet den Diagnose-Server nach TTL, falls ihn niemand
# mehr abraeumt (abgerissene Sitzung). Gibt die Karten frei, damit llama-swap
# wieder Modelle fuer AIfred laden kann.
TTL=${1:-2400}
sleep "$TTL"
if pgrep -f 'VLLM[:]:' >/dev/null 2>&1; then
  echo "$(date '+%F %T') Totmannschalter: beende verwaisten Diagnose-Server"
  for p in $(pgrep -f 'VLLM[:]:'); do kill -TERM $p 2>/dev/null; done
  pkill -TERM -f 'api_server.*port 806[0]' 2>/dev/null
  sleep 15
  for p in $(pgrep -f 'VLLM[:]:'); do kill -KILL $p 2>/dev/null; done
  pkill -KILL -f 'api_server.*port 806[0]' 2>/dev/null
  echo "$(date '+%F %T') beendet"
else
  echo "$(date '+%F %T') nichts zu tun, schon abgeraeumt"
fi
