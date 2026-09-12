#!/usr/bin/env bash
set -uo pipefail
REPO=/home/mp/Projekte/vllm-research/v100-skinny; cd $REPO; V=$REPO/.venv-pr-turing
while pgrep -f "^bash handover/2026-09-12/e2_chain7.sh" >/dev/null; do sleep 15; done
summ() { R=$HOME/.cache/mtp-diagnostics/qual_$1; echo "$1: $(grep -o 'STATUS [a-z_0-9]*' $HOME/.cache/mtp-diagnostics/sweep_$1.log | head -1) $(grep -o 'MEDIAN.*' $HOME/.cache/mtp-diagnostics/sweep_$1.log) SHA $(python3 -c "import json;print(json.load(open('$R/result.json'))['sha256'])" 2>/dev/null || echo KEIN_ERGEBNIS) | pack: $(grep -a -c 'block-packed activations enabled' $R/boot.log)"; }
wait_free() { for i in $(seq 1 90); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc); [ "${u:-0}" -le 1000 ] && ! pgrep -f '[a]pi_server' >/dev/null && return 0; sleep 5; done; echo "ABBRUCH: Karten nicht frei"; exit 1; }
wait_free; echo "== e2p_k3_nopack (Nachholer)"
env MTP_K=3 VLLM_SM70_NVFP4_QPN2_PACK=0 KV_DTYPE=float16 ATTN_BACKEND=TRITON_ATTN VENV=$V DEVS=0,2 VLLM_SM70_QUANT_BACKEND=auto bash tools/mtp-diagnostics/speed_dflash.sh e2p_k3_nopack upstream mtp > $HOME/.cache/mtp-diagnostics/sweep_e2p_k3_nopack.log 2>&1; summ e2p_k3_nopack
echo E2P8-ENDE
