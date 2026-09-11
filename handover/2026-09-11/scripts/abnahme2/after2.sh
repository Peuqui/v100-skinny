#!/usr/bin/env bash
# Fortsetzung der Nacharbeit nach dem Sitzungsneustart 11.09. 18:19:
#   1. Chat-Sonde 3x gegen den noch laufenden Sondenserver (Port 8027, PID aus
#      .flash-next.pid), danach NUR dessen Prozessgruppe abbauen
#   2. 11a End-zu-End + CUDA-Tests
#   3. E5-A/B
#   4. Wheel-Bau und Wheel-Abnahme
set -uo pipefail
S=$(dirname "$0"); OUT=${1:?OUT fehlt}; mkdir -p "$OUT"
REPO=/home/mp/Projekte/vllm-research/v100-skinny
VENV=/home/mp/vllm/venv
PORT=8027
stamp() { echo "===== $(date '+%F %T') $*"; }

stamp "1) Chat-Sonde Flash-Next, 3 Laeufe im laufenden Server (Aufraeum-Stand)"
PID=$(cat "$REPO/.flash-next.pid" 2>/dev/null)
if curl -sf -o /dev/null --max-time 5 "http://127.0.0.1:$PORT/v1/models" && [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
  BOOT=$HOME/.cache/mtp-diagnostics/fnqc_acc2_chat_1/boot.log
  echo "   Server pid $PID pgid $(ps -o pgid= -p "$PID" | tr -d ' '), Boot-Log $BOOT"
  echo "   PARSER: $(grep -o 'reasoning_parser=[^,)]*\|tool_call_parser=[^,)]*' "$BOOT" | sort -u | tr '\n' ' ')"
  for r in 1 2 3; do
    stamp "1.$r"
    "$VENV/bin/python" "$S/chat_ask.py" "$HOME/.cache/mtp-diagnostics/fnqc_acc2_chat_$r" "$PORT" "$REPO/tools/mtp-diagnostics" 2400 2>&1 | grep "q[123]:\|Traceback\|Error"
  done
  stamp "Sondenserver abbauen (pgid $PID)"
  kill -TERM -"$PID" 2>/dev/null; sleep 20; kill -KILL -"$PID" 2>/dev/null
  echo "FERTIG fnqc_acc2_chat"
else
  echo "STATUS Sondenserver nicht erreichbar - Chat-Sonde uebersprungen"
fi

stamp "2) Paket 11a End-zu-End"
bash "$S/e2e_11a.sh" "$OUT/e2e11a" 2>&1

stamp "3) E5-A/B"
bash "$S/e5_ab.sh" "$OUT/e5" 2>&1

stamp "4) Wheel-Bau und Wheel-Abnahme"
bash "$S/wheel_test.sh" "$OUT/wheel" all 2>&1

stamp "NACHARBEIT-ENDE"
