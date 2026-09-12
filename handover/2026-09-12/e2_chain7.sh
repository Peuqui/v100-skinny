#!/usr/bin/env bash
# Pack-A/B mit registriertem Schalter (gleicher Env-Hash fuer alle Arme): k=3 (M=4), k=4 (M=5), k=7 (M=8); je PACK=0 und auto.
set -uo pipefail
REPO=/home/mp/Projekte/vllm-research/v100-skinny; cd $REPO; V=$REPO/.venv-pr-turing
while pgrep -f "^bash handover/2026-09-12/e2_chain6.sh" >/dev/null; do sleep 15; done
summ() { R=$HOME/.cache/mtp-diagnostics/qual_$1; echo "$1: $(grep -o 'STATUS [a-z_0-9]*' $HOME/.cache/mtp-diagnostics/sweep_$1.log | head -1) $(grep -o 'MEDIAN.*' $HOME/.cache/mtp-diagnostics/sweep_$1.log) SHA $(python3 -c "import json;print(json.load(open('$R/result.json'))['sha256'])" 2>/dev/null || echo KEIN_ERGEBNIS) | pack: $(grep -a -c 'block-packed activations enabled' $R/boot.log) | unknown-env: $(grep -a -c 'Unknown vLLM environment variable detected: VLLM_SM70_NVFP4_QPN2_PACK' $R/boot.log)"; }
wait_free() { for i in $(seq 1 90); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc); [ "${u:-0}" -le 1000 ] && ! pgrep -f '[a]pi_server' >/dev/null && return; sleep 5; done; }
run() { wait_free; echo "== $1"; env "${@:2}" KV_DTYPE=float16 ATTN_BACKEND=TRITON_ATTN VENV=$V DEVS=0,2 VLLM_SM70_QUANT_BACKEND=auto bash tools/mtp-diagnostics/speed_dflash.sh $1 upstream mtp > $HOME/.cache/mtp-diagnostics/sweep_$1.log 2>&1; summ $1; }
for K in 3 4 7; do
  run e2p_k${K}_nopack MTP_K=$K VLLM_SM70_NVFP4_QPN2_PACK=0
  run e2p_k${K}_pack   MTP_K=$K VLLM_SM70_NVFP4_QPN2_PACK=auto
done
echo E2P-ENDE
