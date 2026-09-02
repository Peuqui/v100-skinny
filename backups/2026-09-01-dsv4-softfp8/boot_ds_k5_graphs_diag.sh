export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_HOME=/home/mp/Projekte/v100-skinny/.cuda-nvcc-deb/usr/local/cuda-12.8
export TORCH_CUDA_ARCH_LIST=7.0
export VLLM_SM70_E5_CACHE=0 VLLM_SM70_NVFP4_TURBOMIND=0 VLLM_SM70_QUANT_BACKEND=marlin
export VLLM_SKINNY_NVFP4=1 VLLM_SKINNY_QPN=1 VLLM_SKINNY_QPN2=1
export VLLM_SKINNY_NVFP4_SRC=/home/mp/Projekte/v100-skinny/kernels/skinny_kernels.cu
export VLLM_NO_USAGE_STATS=1
export VLLM_DISABLE_SHARED_EXPERTS_STREAM=1
export VLLM_DSPARK_DIAG=1
export VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64
export VLLM_SM70_ASYNC_CPU_TRACE=0
export VLLM_SM70_ASYNC_SCHEDULING_QUEUE_DEPTH=2
export VLLM_PP_SEAM_TRACE=0
export VLLM_DSPARK_DIAG=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1,4,3,2
export VLLM_PP_LAYER_PARTITION=11,8,8,8,8
exec /home/mp/Projekte/v100-skinny/.venv-sm70-130/bin/python \
  -m vllm.entrypoints.openai.api_server \
  --model /home/mp/models/DeepSeek-V4-Flash-nvfp4-DSpark \
  --served-model-name dsv4-manual --trust-remote-code --dtype half \
  --disable-custom-all-reduce --enable-prompt-tokens-details \
  --tensor-parallel-size 1 --pipeline-parallel-size 5 \
  --gpu-memory-utilization 0.95 --max-model-len 4096 \
  --max-num-seqs 1 --max-num-batched-tokens 64 --kv-cache-dtype fp8 --num-gpu-blocks-override 512 \
  --speculative-config '{"method": "dspark", "num_speculative_tokens": 5}' --compilation-config '{"cudagraph_capture_sizes":[6],"max_cudagraph_capture_size":6}' --host 127.0.0.1 --port 19998
