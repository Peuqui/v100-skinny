export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_HOME=/home/mp/Projekte/v100-skinny/.cuda-nvcc-deb/usr/local/cuda-12.8
export TORCH_CUDA_ARCH_LIST=7.0
export VLLM_SM70_E5_CACHE=0 VLLM_SM70_NVFP4_TURBOMIND=0 VLLM_SM70_QUANT_BACKEND=marlin
export VLLM_SKINNY_NVFP4=1 VLLM_SKINNY_QPN=1 VLLM_SKINNY_QPN2=1
export VLLM_SKINNY_NVFP4_SRC=/home/mp/Projekte/v100-skinny/kernels/skinny_kernels.cu
export VLLM_NO_USAGE_STATS=1
export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,2,1,4,3
export VLLM_PP_LAYER_PARTITION=11,11,7,7,7
exec /home/mp/Projekte/v100-skinny/.venv-sm70-130/bin/python \
  -m vllm.entrypoints.openai.api_server \
  --model /home/mp/models/DeepSeek-V4-Flash-nvfp4-DSpark \
  --served-model-name dsv4-manual --trust-remote-code --dtype half \
  --disable-custom-all-reduce --enable-prompt-tokens-details \
  --tensor-parallel-size 1 --pipeline-parallel-size 5 \
  --gpu-memory-utilization 0.92 --max-model-len 16384 \
  --max-num-seqs 4 --max-num-batched-tokens 2048 --kv-cache-dtype fp8 \
  --enforce-eager --host 127.0.0.1 --port 19998
