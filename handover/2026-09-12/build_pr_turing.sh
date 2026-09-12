#!/usr/bin/env bash
# Quellbau reines main (ae75fb9b) + lokale #601-Nachhilfe -> .venv-pr-turing (editable), Rezept reference_1cat_source_build_recipe
set -uo pipefail
cd /home/mp/Projekte/vllm-research/1Cat-vLLM-pr-turing-ops
env -u VLLM_FLASH_ATTN_SRC_DIR CPATH=/home/mp/Projekte/vllm-research/v100-skinny/.venv-pr-turing/lib/python3.12/site-packages/nvidia/cuda_cccl/include CUDA_HOME=/home/mp/vllm/cuda TORCH_CUDA_ARCH_LIST=7.0 MAX_JOBS=3 /home/mp/Projekte/vllm-research/v100-skinny/.venv-pr-turing/bin/python -m pip install -e . --no-build-isolation 2>&1
echo "BUILD-EXIT $?"
