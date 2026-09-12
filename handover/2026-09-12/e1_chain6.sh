#!/usr/bin/env bash
# E-1 E2E mit TRITON_ATTN: RTX (E-1) und V100 (TurboMind), je MTP k=3 und ohne Spekulation.
set -uo pipefail
REPO=/home/mp/Projekte/vllm-research/v100-skinny; cd $REPO; V=$REPO/.venv-pr-turing
summ() { R=$HOME/.cache/mtp-diagnostics/qual_$1; echo "$1: $(grep -o 'STATUS [a-z_0-9]*' $HOME/.cache/mtp-diagnostics/sweep_$1.log | head -1) $(grep -o 'MEDIAN.*' $HOME/.cache/mtp-diagnostics/sweep_$1.log) SHA $(python3 -c "import json;print(json.load(open('$R/result.json'))['sha256'])" 2>/dev/null || echo KEIN_ERGEBNIS) | attn: $(grep -a -o -m1 'Using [A-Z_]* attention backend' $R/boot.log) | pfade: $(grep -a -o 'SM7[05] ModelOpt [A-Za-z0-9 ]*path enabled' $R/boot.log | sort -u | tr '\n' ';')"; grep -a -m1 -E "Error:|Error\b" $R/boot.log | sed 's/.*\] //' | cut -c1-160; }
wait_free() { for i in $(seq 1 90); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc); [ "${u:-0}" -le 1000 ] && ! pgrep -f '[a]pi_server' >/dev/null && return; sleep 5; done; }
for P in "0,2 e1f_rtx_mtp mtp" "1,3 e1f_v100_mtp mtp" "0,2 e1f_rtx_0 0" "1,3 e1f_v100_0 0"; do
  set -- $P; wait_free; echo "== $2"
  env KV_DTYPE=float16 ATTN_BACKEND=TRITON_ATTN VENV=$V DEVS=$1 VLLM_SM70_QUANT_BACKEND=auto bash tools/mtp-diagnostics/speed_dflash.sh $2 upstream $3 > $HOME/.cache/mtp-diagnostics/sweep_$2.log 2>&1; summ $2
done
echo E1-ENDE
