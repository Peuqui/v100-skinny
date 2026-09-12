#!/usr/bin/env bash
set -uo pipefail
REPO=/home/mp/Projekte/vllm-research/v100-skinny; cd $REPO; V=$REPO/.venv-pr-turing
while pgrep -f "^bash handover/2026-09-12/e1_ab_marlin.sh" >/dev/null; do sleep 15; done
for i in $(seq 1 90); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc); [ "${u:-0}" -le 1000 ] && ! pgrep -f '[a]pi_server' >/dev/null && break; sleep 5; done
echo "== PROFIL E-1 RTX (mtp, TRITON_ATTN, fp16 KV)"
env CUDA_DEVICE_ORDER=PCI_BUS_ID KV_DTYPE=float16 ATTN_BACKEND=TRITON_ATTN VENV=$V VLLM_SM70_QUANT_BACKEND=auto bash tools/mtp-diagnostics/prof_dflash.sh e1prof 0,2 mtp 2>&1 | tail -30
echo PROF-ENDE
