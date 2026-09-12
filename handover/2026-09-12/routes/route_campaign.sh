#!/usr/bin/env bash
# Paket E, Schritt 1: Routenzaehler auf produktionsnahen Konfigurationen.
# Zaehlt Dispatch-Entscheidungen (route, M) des Skinny-Pfads (marlin.py) je
# Rang; unter CUDA-Graphs zaehlt das Capture, nicht die Replays.
set -uo pipefail
REPO=/home/mp/Projekte/vllm-research/v100-skinny; cd $REPO
OUT=$REPO/handover/2026-09-12/routes; D=/home/mp/.cache/huggingface/hub/models--maurienne-ai--Qwen3.8-27B-DFlash2-NVFP4-RTNcal/snapshots/bd7a934213c47a9e7ef69eef36bb3325f47fd1f1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
wait_free() { for i in $(seq 1 90); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc); [ "${u:-0}" -le 1000 ] && ! pgrep -f '[a]pi_server' >/dev/null && return; sleep 5; done; }
run() { # tag, cmd...
  local TAG=$1; shift; wait_free; echo "== $TAG"
  VLLM_SKINNY_ROUTE_COUNT_FILE=$OUT/routes_$TAG "$@" > $OUT/log_$TAG.txt 2>&1
  echo "   $(grep -o 'STATUS [a-z_0-9]*\|MEDIAN.*\|FERTIG.*' $OUT/log_$TAG.txt | head -2 | tr '\n' ' ')"
  for f in $OUT/routes_$TAG.*; do [ -f "$f" ] && echo "   $(basename $f): $(python3 -c "import json;d=json.load(open('$f'));print(', '.join(f'{k}={v}' for k,v in sorted(d.items())))")"; done
}
run 27b_mtp_rtx   env DEVS=0,2 bash tools/mtp-diagnostics/speed_dflash.sh route_27b_mtp_rtx fork mtp
run 27b_mtp_v100  env DEVS=1,3 bash tools/mtp-diagnostics/speed_dflash.sh route_27b_mtp_v100 fork mtp
run 27b_dflash_rtx env DEVS=0,2 DRAFT=$D bash tools/mtp-diagnostics/speed_dflash.sh route_27b_dflash_rtx fork dflash
run flashnext_tp2pp2 bash tools/mtp-diagnostics/flashnext_qual.sh route_fn 4
run deepseek_pp5  env VENV=/home/mp/vllm/venv bash handover/2026-09-11/scripts/ds_accept.sh route_ds
echo CAMPAIGN-ENDE
