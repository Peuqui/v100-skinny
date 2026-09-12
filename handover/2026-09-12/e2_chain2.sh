#!/usr/bin/env bash
# E-2 auf DFlash2 (M=8 im Verify): Pack auto gegen 0 auf der RTX; Kontrolle MTP mit PACK=1 erzwungen (M=4, unter der Schwelle).
set -uo pipefail
REPO=/home/mp/Projekte/vllm-research/v100-skinny; cd $REPO; V=$REPO/.venv-pr-turing
while pgrep -f "^bash handover/2026-09-12/e2_chain.sh" >/dev/null; do sleep 15; done
summ() { R=$HOME/.cache/mtp-diagnostics/qual_$1; echo "$1: $(grep -o 'STATUS [a-z_0-9]*' $HOME/.cache/mtp-diagnostics/sweep_$1.log | head -1) $(grep -o 'MEDIAN.*' $HOME/.cache/mtp-diagnostics/sweep_$1.log) SHA $(python3 -c "import json;print(json.load(open('$R/result.json'))['sha256'])" 2>/dev/null || echo KEIN_ERGEBNIS) | pack: $(grep -a -c 'block-packed activations enabled' $R/boot.log)"; grep -a -m1 -E "Error:|Error\b" $R/boot.log | sed 's/.*\] //' | cut -c1-160; }
wait_free() { for i in $(seq 1 90); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc); [ "${u:-0}" -le 1000 ] && ! pgrep -f '[a]pi_server' >/dev/null && return; sleep 5; done; }
for P in "e2d_rtx_pack auto dflash" "e2d_rtx_nopack 0 dflash" "e2d_rtx_pack2 auto dflash" "e2m_rtx_force1 1 mtp" "e2m_rtx_nopack2 0 mtp"; do
  set -- $P; wait_free; echo "== $1 (PACK=$2, $3)"
  env VLLM_SM70_NVFP4_QPN2_PACK=$2 KV_DTYPE=float16 ATTN_BACKEND=TRITON_ATTN VENV=$V DEVS=0,2 VLLM_SM70_QUANT_BACKEND=auto bash tools/mtp-diagnostics/speed_dflash.sh $1 upstream $3 > $HOME/.cache/mtp-diagnostics/sweep_$1.log 2>&1; summ $1
done
echo E2B-ENDE
