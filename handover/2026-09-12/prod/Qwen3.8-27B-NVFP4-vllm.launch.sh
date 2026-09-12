#!/usr/bin/env bash
export PATH=/home/mp/vllm/venv/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_HOME=/home/mp/vllm/cuda
export TORCH_CUDA_ARCH_LIST=7.0
export NCCL_P2P_DISABLE=1
export VLLM_SM70_NVFP4_TURBOMIND=1
export VLLM_SM70_QUANT_BACKEND=auto
export VLLM_SKINNY_NVFP4=1
export VLLM_SKINNY_QPN=1
export VLLM_SKINNY_QPN2=1
export VLLM_SKINNY_NVFP4_SRC=/home/mp/Projekte/v100-skinny/kernels/skinny_kernels.cu
export TORCHINDUCTOR_CACHE_DIR=/home/mp/.cache/torchinductor
export VLLM_NO_USAGE_STATS=1
export CUDA_VISIBLE_DEVICES=0,2
export VLLM_1CAT_ENABLE_SM70_MTP_DEFAULTS=1
export HOME=/home/mp
exec /home/mp/vllm/venv/bin/python -m vllm.entrypoints.openai.api_server --model /home/mp/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4/snapshots/319f741cce68d7914884900c138a1fbb70a42f30 --served-model-name Qwen3.8-27B-NVFP4-vllm --trust-remote-code --dtype float16 --disable-custom-all-reduce --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-prompt-tokens-details --enable-prefix-caching --tensor-parallel-size 2 --pipeline-parallel-size 1 --gpu-memory-utilization 0.98 --block-size 16 --max-model-len 262144 --max-num-seqs 4 --max-num-batched-tokens 2048 --host 127.0.0.1 --port 8093 --language-model-only --speculative-config '{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"greedy","use_local_argmax_reduction":true,"attention_backend":"FLASH_ATTN"}' --compilation-config '{"cudagraph_capture_sizes":[1,2,4,8]}'
