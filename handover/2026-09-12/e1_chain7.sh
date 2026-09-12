#!/usr/bin/env bash
set -uo pipefail
REPO=/home/mp/Projekte/vllm-research/v100-skinny; cd $REPO; V=$REPO/.venv-pr-turing
summ() { R=$HOME/.cache/mtp-diagnostics/qual_$1; echo "$1: $(grep -o 'STATUS [a-z_0-9]*' $HOME/.cache/mtp-diagnostics/sweep_$1.log | head -1) $(grep -o 'MEDIAN.*' $HOME/.cache/mtp-diagnostics/sweep_$1.log) SHA $(python3 -c "import json;print(json.load(open('$R/result.json'))['sha256'])" 2>/dev/null || echo KEIN_ERGEBNIS)"; grep -a -m1 -E "Error:|Error\b" $R/boot.log | sed 's/.*\] //' | cut -c1-160; }
wait_free() { for i in $(seq 1 90); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc); [ "${u:-0}" -le 1000 ] && ! pgrep -f '[a]pi_server' >/dev/null && return; sleep 5; done; }
wait_free; echo "== e1d_rtx_mtp (Dispatch-Op)"
env KV_DTYPE=float16 ATTN_BACKEND=TRITON_ATTN VENV=$V DEVS=0,2 VLLM_SM70_QUANT_BACKEND=auto bash tools/mtp-diagnostics/speed_dflash.sh e1d_rtx_mtp upstream mtp > $HOME/.cache/mtp-diagnostics/sweep_e1d_rtx_mtp.log 2>&1; summ e1d_rtx_mtp
wait_free; echo "== PROFIL e1prof2"
env CUDA_DEVICE_ORDER=PCI_BUS_ID KV_DTYPE=float16 ATTN_BACKEND=TRITON_ATTN VENV=$V VLLM_SM70_QUANT_BACKEND=auto bash tools/mtp-diagnostics/prof_dflash.sh e1prof2 0,2 mtp 2>&1 | grep -a -E "STATUS|GPU-Kernelzeit|^\s*[0-9.]+ ms" | head -18
echo CHAIN7-ENDE
