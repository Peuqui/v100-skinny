#!/usr/bin/env bash
# A/B auf der RTX: NVFP4 ueber Marlin sm75 (VLLM_SM70_NVFP4_TURBOMIND=0), FP8 weiter QPN8.
set -uo pipefail
REPO=/home/mp/Projekte/vllm-research/v100-skinny; cd $REPO; V=$REPO/.venv-pr-turing
while pgrep -f "^bash handover/2026-09-12/e1_chain6.sh" >/dev/null; do sleep 15; done
summ() { R=$HOME/.cache/mtp-diagnostics/qual_$1; echo "$1: $(grep -o 'STATUS [a-z_0-9]*' $HOME/.cache/mtp-diagnostics/sweep_$1.log | head -1) $(grep -o 'MEDIAN.*' $HOME/.cache/mtp-diagnostics/sweep_$1.log) SHA $(python3 -c "import json;print(json.load(open('$R/result.json'))['sha256'])" 2>/dev/null || echo KEIN_ERGEBNIS) | pfade: $(grep -a -o 'SM7[05] ModelOpt [A-Za-z0-9 ]*path enabled' $R/boot.log | sort -u | tr '\n' ';')"; grep -a -m1 -E "Error:|Error\b" $R/boot.log | sed 's/.*\] //' | cut -c1-160; }
wait_free() { for i in $(seq 1 90); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc); [ "${u:-0}" -le 1000 ] && ! pgrep -f '[a]pi_server' >/dev/null && return; sleep 5; done; }
wait_free; echo "== e1m_rtx_mtp (NVFP4 Marlin, FP8 QPN8)"
env KV_DTYPE=float16 ATTN_BACKEND=TRITON_ATTN VENV=$V DEVS=0,2 VLLM_SM70_QUANT_BACKEND=auto VLLM_SM70_NVFP4_TURBOMIND=0 bash tools/mtp-diagnostics/speed_dflash.sh e1m_rtx_mtp upstream mtp > $HOME/.cache/mtp-diagnostics/sweep_e1m_rtx_mtp.log 2>&1; summ e1m_rtx_mtp
echo AB-ENDE
