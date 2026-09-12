#!/usr/bin/env bash
# Punkt 13: NCCL-Umgebungsschalter gegen den AllReduce (27B DFlash2, TP2).
# Je Variante fuenf Laeufe ueber speed_dflash.sh; Median, Annahmelaenge, SHA.
# Aufruf: nccl_sweep.sh <DEVS> <tag> <Variante ...>   Variante "base" = ohne Schalter
set -uo pipefail
DEVS=$1; TAG=$2; shift 2
D=/home/mp/.cache/huggingface/hub/models--maurienne-ai--Qwen3.8-27B-DFlash2-NVFP4-RTNcal/snapshots/bd7a934213c47a9e7ef69eef36bb3325f47fd1f1
cd /home/mp/Projekte/vllm-research/v100-skinny
for V in "$@"; do
  NAME="nccl_${TAG}_$(echo "$V" | tr '=,' '__' | tr -cd 'A-Za-z0-9_')"
  # speed_dflash.sh bricht ab, wenn noch ein api_server lebt oder die Karten
  # ueber 500 MiB belegt sind; nach dem Abbau der Vorgaenger-Variante ist das
  # fuer einige Sekunden der Fall (Lauf 11.09.: vier Varianten ohne Ergebnis).
  for i in $(seq 1 60); do
    u=$(nvidia-smi --id=$DEVS --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc)
    if [ "${u:-0}" -le 500 ] && ! pgrep -f '[a]pi_server' > /dev/null; then break; fi
    sleep 5
  done
  if [ "$V" = "base" ]; then
    DEVS=$DEVS DRAFT=$D bash tools/mtp-diagnostics/speed_dflash.sh "$NAME" fork dflash > "$HOME/.cache/mtp-diagnostics/sweep_$NAME.log" 2>&1
  else
    env $V DEVS=$DEVS DRAFT=$D bash tools/mtp-diagnostics/speed_dflash.sh "$NAME" fork dflash > "$HOME/.cache/mtp-diagnostics/sweep_$NAME.log" 2>&1
  fi
  R=$HOME/.cache/mtp-diagnostics/qual_$NAME
  /home/mp/vllm/venv/bin/python - "$V" "$R/result.json" <<'PY'
import json, statistics, sys
v, path = sys.argv[1:3]
try:
    d = json.load(open(path))
except (OSError, ValueError):
    print(f"{v:28s} KEIN ERGEBNIS"); raise SystemExit
acc = statistics.median(r.get("acc_len", 0) for r in d["rows"])
print(f"{v:28s} {d['median_tok_per_s']:6.2f} tok/s | Annahme {acc:.3f} | SHA {d['sha256']}")
PY
  command grep -q "Traceback\|NCCL WARN\|NCCL error" $R/boot.log 2>/dev/null && echo "   Warnung/Fehler im boot.log: $(command grep -m1 -o 'NCCL WARN.*\|NCCL error.*\|[A-Za-z]*Error:.*' $R/boot.log | cut -c1-120)"
done
echo SWEEP-ENDE
