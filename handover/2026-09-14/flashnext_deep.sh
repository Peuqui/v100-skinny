#!/usr/bin/env bash
# Flash-Next-Abnahme ueber den PRODUKTIONSPFAD: /v1/chat/completions mit dem
# Chat-Template des Checkpoints, enable_thinking=true (chat_template_kwargs wie
# AIfred), Reasoning-Parser qwen3 und Tool-Parser qwen3_coder wie im
# llama-swap-Eintrag. Denkblock landet im Feld `reasoning`, Antwort in `content`.
# Gegenstueck zu tools/mtp-diagnostics/flashnext_qual.sh (Rohtext-Sonde ohne
# Template, dort muss das Modell selbst entscheiden, ob es denkt).
#   flashnext_qual_chat.sh <name> <k>
# Server-Boot, Karten, Betriebspunkt und Abbau identisch zur Rohtext-Sonde.
set -uo pipefail
NAME=${1:?name fehlt}; K=${2:?k fehlt}
REPO=/home/mp/Projekte/vllm-research/v100-skinny
CKPT=${CKPT:-/home/mp/models/Qwen3.8-Flash-Next-180B-A4B-NVFP4-MTPQ}
VENV=${VENV:-/home/mp/vllm/venv}
W=$HOME/.cache/mtp-diagnostics/fndeep_$NAME; rm -rf "$W"; mkdir -p "$W"
PORT=8027
MAXTOK=${MAXTOK:-16000}  # Denkblock + 30 Saetze; bei MML 262144 kein Deckel mehr noetig (q3 brauchte 4.172 Token)

for i in $(seq 1 60); do
  u=$(nvidia-smi --id=0,1,2,3 --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc)
  [ "${u:-0}" -le 800 ] && break; sleep 5
done
if pgrep -af 'api_server' | grep -v $$ | grep -q .; then echo "ABBRUCH: api_server laeuft"; exit 1; fi

unset VLLM_SM70_QWEN_GDN_FULL_FORWARD
export AIFRED_FORCE_UPSTREAM_GDN=1
export AIFRED_STATE_FILE=$W/state.txt

cd $REPO
 CUDA_VISIBLE_DEVICES=0,2,1,3 \
TURBOMIND=1 QUANT_BACKEND=turbomind \
ENV_PREFIX="$VENV" \
TP=2 PP=2 K=$K GMU=0.95 MML=262144 PP_PARTITION=24,24 PLE_HOST_GIB=6 \
PORT=$PORT LOG=$W/boot.log BOOT_WAIT_S=2400 \
EXTRA_ARGS='--distributed-timeout-seconds 3600 --compilation-config {"cudagraph_capture_sizes":[1,2,4,5,8]} --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3' \
bash scripts/serve-qwen38-flash-next.sh "$CKPT" 2>&1 | tail -3
PID=$(cat $REPO/.flash-next.pid 2>/dev/null)

if curl -sf -o /dev/null --max-time 5 "http://127.0.0.1:$PORT/v1/models"; then
  "$VENV/bin/python" /home/mp/Projekte/vllm-research/v100-skinny/handover/2026-09-14/flashnext_deep_driver.py "$W" "$PORT" "$REPO/tools/mtp-diagnostics/vorkontext.txt" "$MAXTOK"
else
  echo "STATUS nicht_oben"; tail -5 $W/boot.log
fi

[ -n "${PID:-}" ] && kill -TERM -$PID 2>/dev/null
sleep 20
[ -n "${PID:-}" ] && kill -KILL -$PID 2>/dev/null
echo "   PARSER: $(grep -o 'reasoning_parser=[^,)]*\|tool_call_parser=[^,)]*\|Using reasoning parser[^\n]\{0,40\}' $W/boot.log | sort -u | tr '\n' ' ')"
echo "FERTIG fnqc_$NAME"
