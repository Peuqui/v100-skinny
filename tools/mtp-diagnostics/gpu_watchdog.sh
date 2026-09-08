#!/usr/bin/env bash
# Totmannschalter, der an einem PROZESS haengt statt an einer Uhr.
#
#   gpu_watchdog.sh <pid-der-messkette> [poll_s]
#
# Solange die Messkette lebt, tut er nichts -- egal wie lange sie braucht.
# Stirbt sie (Absturz im Skript, abgerissene Sitzung, Ctrl-C) und es steht
# noch ein Diagnose-Server, raeumt er ihn ab und gibt die Karten frei.
#
# Warum nicht "sleep TTL" wie in gpu_deadman.sh: ein fester Timer feuert
# entweder zu frueh (killt einen gesunden langen Boot -- Flash-Next braucht
# neun Minuten) oder zu spaet (die Karten bleiben blockiert). Der Tod des
# kontrollierenden Prozesses ist das richtige Signal, nicht eine Uhr.
#
# WICHTIG: nur per PID beenden, nie mit einem breiten "pkill -f <muster>".
# Das Muster matcht sonst die eigene Wrapper-Kommandozeile und bringt die
# aufrufende Shell um (am 08.09.2026 genau so passiert).
set -uo pipefail
PID=${1:?PID der Messkette fehlt}
POLL=${2:-15}

kill -0 "$PID" 2>/dev/null || { echo "$(date '+%F %T') PID $PID lebt nicht, nichts zu tun"; exit 0; }
echo "$(date '+%F %T') Wachhund an PID $PID (Poll ${POLL}s)"

while kill -0 "$PID" 2>/dev/null; do sleep "$POLL"; done
echo "$(date '+%F %T') Messkette $PID ist weg"

if pgrep -f 'VLLM[:]:' >/dev/null 2>&1; then
  echo "$(date '+%F %T') verwaister Diagnose-Server gefunden -- raeume ab"
  for p in $(pgrep -f 'VLLM[:]:'); do kill -TERM "$p" 2>/dev/null; done
  sleep 15
  for p in $(pgrep -f 'VLLM[:]:'); do kill -KILL "$p" 2>/dev/null; done
  echo "$(date '+%F %T') abgeraeumt"
else
  echo "$(date '+%F %T') nichts zu tun, die Kette hat selbst aufgeraeumt"
fi
