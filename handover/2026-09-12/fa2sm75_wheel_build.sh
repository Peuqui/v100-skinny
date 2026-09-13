#!/usr/bin/env bash
# Baubeweis Paket C: Turing-Wheel (TORCH_CUDA_ARCH_LIST=7.5) aus dem PR-Baum, ExternalProject holt Peuqui/flash-attention @ 43b9d29c.
cd /home/mp/Projekte/vllm-research/1Cat-vLLM-pr-fa2sm75
V=/home/mp/Projekte/vllm-research/v100-skinny/.venv-pr-fa2sm75
OUT=/home/mp/Projekte/vllm-research/v100-skinny/.wheels/fa2sm75; rm -rf $OUT; mkdir -p $OUT
date; git log --oneline -1
env -u VLLM_FLASH_ATTN_SRC_DIR CPATH=$V/lib/python3.12/site-packages/nvidia/cuda_cccl/include CUDA_HOME=/home/mp/vllm/cuda TORCH_CUDA_ARCH_LIST=7.5 MAX_JOBS=4 $V/bin/python -m pip wheel . --no-build-isolation --no-deps -w $OUT
echo "WHEEL-EXIT $?"; date
ls -la $OUT/*.whl 2>/dev/null | awk '{print $5, $9}'
for w in $OUT/*.whl; do unzip -l "$w" | grep -E "_vllm_fa2_C|_C\.abi3|_moe_C" ; done
echo WHEELBUILD-ENDE
