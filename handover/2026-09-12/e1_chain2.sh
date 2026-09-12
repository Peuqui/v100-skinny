#!/usr/bin/env bash
# E-1 E2E: RTX (NVFP4 qpn2+dense, FP8 qpn8) -> V100 Referenz (TurboMind), .venv-pr-turing, 27B MTP k=3
set -uo pipefail
REPO=/home/mp/Projekte/vllm-research/v100-skinny; cd $REPO; V=$REPO/.venv-pr-turing
summ() { R=$HOME/.cache/mtp-diagnostics/qual_$1; echo "$1: $(grep -o 'STATUS [a-z_0-9]*' $HOME/.cache/mtp-diagnostics/sweep_$1.log | head -1) $(grep -o 'MEDIAN.*' $HOME/.cache/mtp-diagnostics/sweep_$1.log) SHA $(python3 -c "import json;print(json.load(open('$R/result.json'))['sha256'])" 2>/dev/null || echo KEIN_ERGEBNIS) | pfade: $(grep -a -o 'SM7[05] ModelOpt [A-Za-z0-9 ]*path enabled' $R/boot.log | sort -u | tr '\n' ';')"; grep -a -m1 -E "Error:|Error\b" $R/boot.log | sed 's/.*\] //' | cut -c1-160; }
wait_free() { for i in $(seq 1 90); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc); [ "${u:-0}" -le 1000 ] && ! pgrep -f '[a]pi_server' >/dev/null && return; sleep 5; done; }
wait_free; echo "== E-1 RTX"
env VENV=$V DEVS=0,2 VLLM_SM70_QUANT_BACKEND=auto bash tools/mtp-diagnostics/speed_dflash.sh e1_qpn_rtx upstream mtp > $HOME/.cache/mtp-diagnostics/sweep_e1_qpn_rtx.log 2>&1; summ e1_qpn_rtx
wait_free; echo "== V100 Referenz (TurboMind, main)"
env VENV=$V DEVS=1,3 VLLM_SM70_QUANT_BACKEND=auto bash tools/mtp-diagnostics/speed_dflash.sh e1_tm_v100 upstream mtp > $HOME/.cache/mtp-diagnostics/sweep_e1_tm_v100.log 2>&1; summ e1_tm_v100
echo E1-ENDE
