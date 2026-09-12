#!/usr/bin/env bash
# E-2 A/B: Block-Pack auto (Schwelle) gegen aus, RTX (E-1-Pfad) und V100 (TurboMind+qpn2? nein: V100 nimmt TurboMind fuer NVFP4!)
# -> V100 misst den Pack nur ueber DFlash2/compressed-tensors nicht; hier 27B MTP RTX pack auto/0, dazu V100 zur Kontrolle der Unveraendertheit.
set -uo pipefail
REPO=/home/mp/Projekte/vllm-research/v100-skinny; cd $REPO; V=$REPO/.venv-pr-turing
L=$REPO/handover/2026-09-12/build_e2.log
while ! grep -q "BUILD-EXIT" $L; do sleep 20; done; echo "build: $(grep BUILD-EXIT $L)"
grep -q "BUILD-EXIT 0" $L || { echo "BAU FEHLGESCHLAGEN"; exit 1; }
summ() { R=$HOME/.cache/mtp-diagnostics/qual_$1; echo "$1: $(grep -o 'STATUS [a-z_0-9]*' $HOME/.cache/mtp-diagnostics/sweep_$1.log | head -1) $(grep -o 'MEDIAN.*' $HOME/.cache/mtp-diagnostics/sweep_$1.log) SHA $(python3 -c "import json;print(json.load(open('$R/result.json'))['sha256'])" 2>/dev/null || echo KEIN_ERGEBNIS) | pack: $(grep -a -c 'block-packed activations enabled' $R/boot.log)"; grep -a -m1 -E "Error:|Error\b" $R/boot.log | sed 's/.*\] //' | cut -c1-160; }
wait_free() { for i in $(seq 1 90); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc); [ "${u:-0}" -le 1000 ] && ! pgrep -f '[a]pi_server' >/dev/null && return; sleep 5; done; }
for P in "0,2 e2_rtx_pack auto" "0,2 e2_rtx_nopack 0" "0,2 e2_rtx_pack2 auto"; do
  set -- $P; wait_free; echo "== $2 (PACK=$3)"
  env VLLM_SM70_NVFP4_QPN2_PACK=$3 KV_DTYPE=float16 ATTN_BACKEND=TRITON_ATTN VENV=$V DEVS=$1 VLLM_SM70_QUANT_BACKEND=auto bash tools/mtp-diagnostics/speed_dflash.sh $2 upstream mtp > $HOME/.cache/mtp-diagnostics/sweep_$2.log 2>&1; summ $2
done
echo E2-ENDE
