#!/usr/bin/env bash
# Reproduzierbarkeit der SHA-Abweichung: PACK=0, Variable weg, PACK=1 (alle MTP k=3, RTX).
set -uo pipefail
REPO=/home/mp/Projekte/vllm-research/v100-skinny; cd $REPO; V=$REPO/.venv-pr-turing
summ() { R=$HOME/.cache/mtp-diagnostics/qual_$1; echo "$1: $(grep -o 'STATUS [a-z_0-9]*' $HOME/.cache/mtp-diagnostics/sweep_$1.log | head -1) $(grep -o 'MEDIAN.*' $HOME/.cache/mtp-diagnostics/sweep_$1.log) SHA $(python3 -c "import json;print(json.load(open('$R/result.json'))['sha256'])" 2>/dev/null || echo KEIN_ERGEBNIS) | pack: $(grep -a -c 'block-packed activations enabled' $R/boot.log)"; }
wait_free() { for i in $(seq 1 90); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc); [ "${u:-0}" -le 1000 ] && ! pgrep -f '[a]pi_server' >/dev/null && return; sleep 5; done; }
run() { wait_free; echo "== $1"; env "${@:2}" KV_DTYPE=float16 ATTN_BACKEND=TRITON_ATTN VENV=$V DEVS=0,2 VLLM_SM70_QUANT_BACKEND=auto bash tools/mtp-diagnostics/speed_dflash.sh $1 upstream mtp > $HOME/.cache/mtp-diagnostics/sweep_$1.log 2>&1; summ $1; }
run e2r_nopack VLLM_SM70_NVFP4_QPN2_PACK=0
run e2r_unset X_DUMMY=1
run e2r_force1 VLLM_SM70_NVFP4_QPN2_PACK=1
echo E2R-ENDE
