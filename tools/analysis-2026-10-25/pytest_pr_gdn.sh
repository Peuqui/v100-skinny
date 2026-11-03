#!/bin/bash
# pytest in the GDN gate PR worktree with its own sources first on the path, on one card (PCI bus order).
# Usage: pytest_pr_gdn.sh <PCI index> <pytest arguments...>
CARD=$1
shift
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$CARD
cd /home/mp/Projekte/vllm-research/1Cat-vLLM-pr-gdn || exit 1
exec /home/mp/vllm/venv/bin/python -m pytest -p no:cacheprovider "$@"
