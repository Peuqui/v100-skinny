#!/usr/bin/env bash
# AOT-Theorie: "auto" mit frischem VLLM_CACHE_ROOT zweimal (Muenze je Compile?), dazu "0" frisch.
set -uo pipefail
REPO=/home/mp/Projekte/vllm-research/v100-skinny; cd $REPO; V=$REPO/.venv-pr-turing
while pgrep -f "^bash handover/2026-09-12/e2_chain4.sh" >/dev/null; do sleep 15; done
summ() { R=$HOME/.cache/mtp-diagnostics/qual_$1; echo "$1: $(grep -o 'STATUS [a-z_0-9]*' $HOME/.cache/mtp-diagnostics/sweep_$1.log | head -1) $(grep -o 'MEDIAN.*' $HOME/.cache/mtp-diagnostics/sweep_$1.log) SHA $(python3 -c "import json;print(json.load(open('$R/result.json'))['sha256'])" 2>/dev/null || echo KEIN_ERGEBNIS) | AOT-load: $(grep -a -c 'AOT compilation from path' $R/boot.log)"; }
wait_free() { for i in $(seq 1 90); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc); [ "${u:-0}" -le 1000 ] && ! pgrep -f '[a]pi_server' >/dev/null && return; sleep 5; done; }
run() { wait_free; echo "== $1"; env "${@:2}" KV_DTYPE=float16 ATTN_BACKEND=TRITON_ATTN VENV=$V DEVS=0,2 VLLM_SM70_QUANT_BACKEND=auto bash tools/mtp-diagnostics/speed_dflash.sh $1 upstream mtp > $HOME/.cache/mtp-diagnostics/sweep_$1.log 2>&1; summ $1; }
for i in 1 2 3; do rm -rf $HOME/.cache/vllm-fresh-root; run e2f_auto_fresh$i VLLM_SM70_NVFP4_QPN2_PACK=auto VLLM_CACHE_ROOT=$HOME/.cache/vllm-fresh-root; done
rm -rf $HOME/.cache/vllm-fresh-root; run e2f_nopack_fresh VLLM_SM70_NVFP4_QPN2_PACK=0 VLLM_CACHE_ROOT=$HOME/.cache/vllm-fresh-root
echo E2F-ENDE
