#!/usr/bin/env bash
# Editable-Neubau work-main in der Produktions-venv, Rezept reference_1cat_source_build_recipe (nur sm_70, CCCL per CPATH).
cd /home/mp/Projekte/vllm-research/1Cat-vLLM-work
V=/home/mp/Projekte/vllm-research/v100-skinny/.venv-sm70-main
date; git log --oneline -1
env -u VLLM_FLASH_ATTN_SRC_DIR CPATH=$V/lib/python3.12/site-packages/nvidia/cuda_cccl/include CUDA_HOME=/home/mp/vllm/cuda TORCH_CUDA_ARCH_LIST=7.0 MAX_JOBS=4 $V/bin/python -m pip install -e . --no-build-isolation
echo "PIP-EXIT $?"; date
ls -la vllm/_C.abi3.so vllm/_moe_C.abi3.so vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so vllm/_sm70_sparse_attention_C*.so 2>&1 | awk '{print $6,$7,$8,$9}'
echo REBUILD-ENDE
