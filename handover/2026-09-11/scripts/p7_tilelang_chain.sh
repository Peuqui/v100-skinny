#!/usr/bin/env bash
# Punkt 7: TileLang-Pin 0.1.14 (venv .venv-sm70-tltest) — Kernel-Tests je Karte.
# Wichtig: CUDA_DEVICE_ORDER=PCI_BUS_ID, sonst ist Index 1 keine V100 (Falle 11.09.).
set -uo pipefail
cd /home/mp/Projekte/vllm-research/1Cat-vLLM-work
PY=/home/mp/Projekte/vllm-research/v100-skinny/.venv-sm70-tltest/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Wie speed_dflash.sh: ohne CUDA_HOME nimmt tilelang /usr/bin/nvcc (CUDA 12.0),
# das den System-g++ 13 ablehnt ("unsupported GNU version") — Falle 12.09.
export CUDA_HOME=/home/mp/vllm/cuda PATH=/home/mp/vllm/cuda/bin:$PATH
for P in "1 V100" "0 RTX8000"; do
  set -- $P; DEV=$1; TAG=$2
  for T in tests/kernels/test_mhc_kernels.py tests/kernels/test_mhc_sm70_fp16.py; do
    echo "== $TAG (Karte $DEV) $T"
    LOG=$HOME/.cache/mtp-diagnostics/p7_${TAG}_$(basename $T .py).log
    CUDA_VISIBLE_DEVICES=$DEV $PY -m pytest "$T" -q -p no:cacheprovider --tb=short > "$LOG" 2>&1
    tail -1 "$LOG"; grep -a "^FAILED" "$LOG" | head -5
  done
done
echo P7-ENDE
