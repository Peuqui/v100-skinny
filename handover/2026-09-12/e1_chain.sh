#!/usr/bin/env bash
# E-1 E2E: (wartet auf die Basismessung e1_base_rtx) -> E-1 auf RTX (QPN2+dense)
# -> V100 Referenz (TurboMind, reines main). Alle mit .venv-pr-turing, 27B MTP k=3.
set -uo pipefail
REPO=/home/mp/Projekte/vllm-research/v100-skinny; cd $REPO
V=$REPO/.venv-pr-turing
while pgrep -f "[s]peed_dflash.sh e1_base_rtx" >/dev/null; do sleep 10; done
summ() { R=$HOME/.cache/mtp-diagnostics/qual_$1; echo "$1: $(grep -o 'STATUS [a-z_0-9]*' $HOME/.cache/mtp-diagnostics/sweep_$1.log 2>/dev/null | head -1) $(grep -o 'MEDIAN.*' $HOME/.cache/mtp-diagnostics/sweep_$1.log 2>/dev/null) SHA $(python3 -c "import json;print(json.load(open('$R/result.json'))['sha256'])" 2>/dev/null || echo KEIN_ERGEBNIS) | pfad: $(grep -a -o -m1 'SM7[05] ModelOpt NVFP4 [A-Za-z0-9 ]*path enabled' $R/boot.log)"; }
wait_free() { for i in $(seq 1 90); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc); [ "${u:-0}" -le 1000 ] && ! pgrep -f '[a]pi_server' >/dev/null && return; sleep 5; done; }
cp $REPO/handover/2026-09-12/e1_baseline_rtx.out $HOME/.cache/mtp-diagnostics/sweep_e1_base_rtx.log 2>/dev/null
summ e1_base_rtx
wait_free; echo "== E-1 RTX"
env VENV=$V DEVS=0,2 VLLM_SM70_QUANT_BACKEND=auto bash tools/mtp-diagnostics/speed_dflash.sh e1_qpn2_rtx upstream mtp > $HOME/.cache/mtp-diagnostics/sweep_e1_qpn2_rtx.log 2>&1
summ e1_qpn2_rtx
wait_free; echo "== V100 Referenz (TurboMind)"
env VENV=$V DEVS=1,3 VLLM_SM70_QUANT_BACKEND=auto bash tools/mtp-diagnostics/speed_dflash.sh e1_tm_v100 upstream mtp > $HOME/.cache/mtp-diagnostics/sweep_e1_tm_v100.log 2>&1
summ e1_tm_v100
echo E1-ENDE
