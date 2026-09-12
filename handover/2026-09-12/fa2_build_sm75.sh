#!/usr/bin/env bash
# C-0-Nachweis: eigenstaendiger Bau der sm75-FA2 aus dem Fork mit VLLM_FA2_OUTPUT_NAME, exakt die Argumente des geplanten ExternalProject.
set -uo pipefail
SRC=/home/mp/Projekte/vllm-research/flash-attention-sm75; B=$SRC/build-sm75-ext; rm -rf $B; mkdir -p $B
V=/home/mp/Projekte/vllm-research/v100-skinny/.venv-pr-turing
export CUDA_HOME=/home/mp/vllm/cuda PATH=/home/mp/vllm/cuda/bin:$PATH CPATH=$V/lib/python3.12/site-packages/nvidia/cuda_cccl/include
cd $B && cmake -G Ninja -S $SRC -B $B -DCMAKE_BUILD_TYPE=RelWithDebInfo -DPython_EXECUTABLE=$V/bin/python -DCMAKE_CUDA_COMPILER=/home/mp/vllm/cuda/bin/nvcc -DCUDA_ARCHS=7.5 -DVLLM_GPU_ARCHES=75-real -DFA2_ENABLED=ON -DFA3_ENABLED=OFF -DVLLM_FA2_OUTPUT_NAME=_vllm_fa2_C_sm75 2>&1 | tail -5
echo "CONFIGURE-EXIT ${PIPESTATUS[0]}"
ninja -C $B -j3 _vllm_fa2_C 2>&1 | grep -E "error|Error|FAILED|\[[0-9]+/[0-9]+\] Linking" | tail -5
echo "BUILD-EXIT ${PIPESTATUS[0]}"
ls -la $B/*.so $B/vllm_flash_attn/*.so 2>/dev/null | awk '{print $5, $9}'; find $B -name "_vllm_fa2_C_sm75*.so" | head -2
echo FA2BUILD-ENDE
