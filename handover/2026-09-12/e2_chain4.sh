#!/usr/bin/env bash
# Muenze oder Mechanismus: auto/0 mit gemeinsamem Inductor-Cache, dann auto/0 mit je frischem Cache-Ordner.
set -uo pipefail
REPO=/home/mp/Projekte/vllm-research/v100-skinny; cd $REPO; V=$REPO/.venv-pr-turing
summ() { R=$HOME/.cache/mtp-diagnostics/qual_$1; echo "$1: $(grep -o 'STATUS [a-z_0-9]*' $HOME/.cache/mtp-diagnostics/sweep_$1.log | head -1) $(grep -o 'MEDIAN.*' $HOME/.cache/mtp-diagnostics/sweep_$1.log) SHA $(python3 -c "import json;print(json.load(open('$R/result.json'))['sha256'])" 2>/dev/null || echo KEIN_ERGEBNIS) | pack: $(grep -a -c 'block-packed activations enabled' $R/boot.log)"; }
wait_free() { for i in $(seq 1 90); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc); [ "${u:-0}" -le 1000 ] && ! pgrep -f '[a]pi_server' >/dev/null && return; sleep 5; done; }
run() { wait_free; echo "== $1"; env "${@:2}" KV_DTYPE=float16 ATTN_BACKEND=TRITON_ATTN VENV=$V DEVS=0,2 VLLM_SM70_QUANT_BACKEND=auto bash tools/mtp-diagnostics/speed_dflash.sh $1 upstream mtp > $HOME/.cache/mtp-diagnostics/sweep_$1.log 2>&1; summ $1; }
run e2c_auto3 VLLM_SM70_NVFP4_QPN2_PACK=auto
run e2c_nopack3 VLLM_SM70_NVFP4_QPN2_PACK=0
rm -rf $HOME/.cache/torchinductor-fresh-a $HOME/.cache/torchinductor-fresh-b
run e2c_auto_fresh VLLM_SM70_NVFP4_QPN2_PACK=auto TORCHINDUCTOR_CACHE_DIR=$HOME/.cache/torchinductor-fresh-a
run e2c_nopack_fresh VLLM_SM70_NVFP4_QPN2_PACK=0 TORCHINDUCTOR_CACHE_DIR=$HOME/.cache/torchinductor-fresh-b
echo E2C-ENDE
