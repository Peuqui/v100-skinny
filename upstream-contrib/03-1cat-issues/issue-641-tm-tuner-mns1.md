### Your current environment

Our fork of 1Cat main 80c88e8d plus our open PRs, including #639 and #640, which this checkpoint needs to boot on pipeline parallelism.

<details>
<summary>The output of <code>python collect_env.py</code></summary>

```text
==============================
        System Info
==============================
OS                           : Ubuntu 24.04.5 LTS (x86_64)
GCC version                  : (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
Clang version                : Could not collect
CMake version                : version 3.28.3
Libc version                 : glibc-2.39

==============================
       PyTorch Info
==============================
PyTorch version              : 2.10.0+cu128
Is debug build               : False
CUDA used to build PyTorch   : 12.8
ROCM used to build PyTorch   : N/A
XPU used to build PyTorch    : N/A

==============================
      Python Environment
==============================
Python version               : 3.12.3 (main, Aug 31 2026, 10:18:26) [GCC 13.3.0] (64-bit runtime)
Python platform              : Linux-7.0.0-31-generic-x86_64-with-glibc2.39
    
==============================
       CUDA / GPU Info
==============================
Is CUDA available            : True
CUDA runtime version         : 12.0.140
CUDA_MODULE_LOADING set to   : 
GPU models and configuration : 
GPU 0: Quadro RTX 8000
GPU 1: Tesla V100-PCIE-32GB
GPU 2: Quadro RTX 8000
GPU 3: Tesla V100-PCIE-32GB
GPU 4: Tesla V100-PCIE-32GB

Nvidia driver version        : 580.173.02
cuDNN version                : Could not collect
HIP runtime version          : N/A
MIOpen runtime version       : N/A
Is XNNPACK available         : True

==============================
          CPU Info
==============================
Architektur:                             x86_64
CPU Operationsmodus:                     32-bit, 64-bit
Adressgrößen:                            48 bits physical, 48 bits virtual
Byte-Reihenfolge:                        Little Endian
CPU(s):                                  16
Liste der Online-CPU(s):                 0-15
Anbieterkennung:                         AuthenticAMD
Modellname:                              AMD Ryzen 7 7840HS w/ Radeon 780M Graphics
Prozessorfamilie:                        25
Modell:                                  116
Thread(s) pro Kern:                      2
Kern(e) pro Sockel:                      8
Sockel:                                  1
Stepping:                                1
Übertaktung:                             aktiviert
Skalierung der CPU(s):                   69%
Maximale Taktfrequenz der CPU:           5137,9038
Minimale Taktfrequenz der CPU:           419,4210
BogoMIPS:                                7585,94
Markierungen:                            fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl xtopology nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpuid_fault cpb cat_l3 cdp_l3 hw_pstate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local user_shstk avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd cppc arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold vgif x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq rdpid overflow_recov succor smca fsrm flush_l1d amd_lbr_pmc_freeze
Virtualisierung:                         AMD-V
L1d Cache:                               256 KiB (8 Instanzen)
L1i Cache:                               256 KiB (8 Instanzen)
L2 Cache:                                8 MiB (8 Instanzen)
L3 Cache:                                16 MiB (1 Instanz)
NUMA-Knoten:                             1
NUMA-Knoten0 CPU(s):                     0-15
Schwachstelle Gather data sampling:      Not affected
Schwachstelle Ghostwrite:                Not affected
Schwachstelle Indirect target selection: Not affected
Schwachstelle Itlb multihit:             Not affected
Schwachstelle L1tf:                      Not affected
Schwachstelle Mds:                       Not affected
Schwachstelle Meltdown:                  Not affected
Schwachstelle Mmio stale data:           Not affected
Schwachstelle Old microcode:             Not affected
Schwachstelle Reg file data sampling:    Not affected
Schwachstelle Retbleed:                  Not affected
Schwachstelle Spec rstack overflow:      Mitigation; Safe RET
Schwachstelle Spec store bypass:         Mitigation; Speculative Store Bypass disabled via prctl
Schwachstelle Spectre v1:                Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Schwachstelle Spectre v2:                Mitigation; Enhanced / Automatic IBRS; IBPB conditional; STIBP always-on; PBRSB-eIBRS Not affected; BHI Not affected
Schwachstelle Srbds:                     Not affected
Schwachstelle Tsa:                       Mitigation; Clear CPU buffers
Schwachstelle Tsx async abort:           Not affected
Schwachstelle Vmscape:                   Mitigation; IBPB before exit to userspace

==============================
Versions of relevant libraries
==============================
[pip3] flashinfer-python==0.6.11.post2
[pip3] numpy==2.3.5
[pip3] nvidia-cublas-cu12==12.8.4.1
[pip3] nvidia-cuda-cccl-cu12==12.9.27
[pip3] nvidia-cuda-cupti-cu12==12.8.90
[pip3] nvidia-cuda-nvcc-cu12==12.9.86
[pip3] nvidia-cuda-nvdisasm==13.3.73
[pip3] nvidia-cuda-nvrtc-cu12==12.8.93
[pip3] nvidia-cuda-runtime-cu12==12.8.90
[pip3] nvidia-cudnn-cu12==9.10.2.21
[pip3] nvidia-cudnn-frontend==1.18.0
[pip3] nvidia-cufft-cu12==11.3.3.83
[pip3] nvidia-cufile-cu12==1.13.1.3
[pip3] nvidia-curand-cu12==10.3.9.90
[pip3] nvidia-cusolver-cu12==11.7.3.90
[pip3] nvidia-cusparse-cu12==12.5.8.93
[pip3] nvidia-cusparselt-cu12==0.7.1
[pip3] nvidia-cutlass-dsl==4.7.0
[pip3] nvidia-cutlass-dsl-libs-base==4.7.0
[pip3] nvidia-cutlass-dsl-libs-core==4.7.0
[pip3] nvidia-cutlass-dsl-libs-cu12==4.7.0
[pip3] nvidia-ml-py==13.610.43
[pip3] nvidia-nccl-cu12==2.27.5
[pip3] nvidia-nvjitlink-cu12==12.8.93
[pip3] nvidia-nvshmem-cu12==3.4.5
[pip3] nvidia-nvtx-cu12==12.8.90
[pip3] pyzmq==27.2.0
[pip3] torch==2.10.0
[pip3] torch_c_dlpack_ext==0.1.5
[pip3] torchaudio==2.10.0
[pip3] torchvision==0.25.0
[pip3] transformers==5.16.1
[pip3] triton==3.6.0
[conda] Could not collect

==============================
         vLLM Info
==============================
ROCM Version                 : Could not collect
vLLM Version                 : 13b1.dev23+g1d3439f2b (git sha: 1d3439f2b)
vLLM Build Flags:
  CUDA Archs: Not Set; ROCm: Disabled; XPU: Disabled
GPU Topology:
  	[4mGPU0	GPU1	GPU2	GPU3	GPU4	CPU Affinity	NUMA Affinity	GPU NUMA ID[0m
GPU0	 X 	PHB	PHB	PHB	PHB	0-15	0		N/A
GPU1	PHB	 X 	PHB	PHB	PHB	0-15	0		N/A
GPU2	PHB	PHB	 X 	PHB	PHB	0-15	0		N/A
GPU3	PHB	PHB	PHB	 X 	PHB	0-15	0		N/A
GPU4	PHB	PHB	PHB	PHB	 X 	0-15	0		N/A

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks

==============================
     Environment Variables
==============================
VLLM_SM70_TP4_PUSH_ALLREDUCE_SMALL_MESSAGES=1
VLLM_SM70_TP4_PUSH_ALLREDUCE_CONCURRENCY=1
PYTORCH_NVML_BASED_CUDA_CHECK=1
TORCHINDUCTOR_COMPILE_THREADS=1
TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_mp

```

</details>

### 🐛 Describe the bug

With nvidia/Qwen3.8-Flash-Next-NVFP4 (revision fc694b54, MTP routed experts stored as ModelOpt FP8_PB_WO and loaded natively through MTPExpertFp8Config), the engine dies during the profiling run of the last pipeline stage when `--max-num-seqs 1` is set:

```text
(Worker_PP0_TP0) INFO [gpu_worker.py:647] Available KV cache memory: 6.14 GiB
(Worker_PP1_TP0) WARNING [backends.py:1048] Failed to read file <frozen os>
[TM][FATAL] kernels/gemm/tuner/measurer.cu(83): Check failed: status == cudaSuccess an illegal memory access was encountered
[TM][FATAL] kernels/gemm/tuner/measurer.cu(83): Check failed: status == cudaSuccess an illegal memory access was encountered
(EngineCore) ERROR [multiproc_executor.py:284] Worker proc VllmWorker-3 died unexpectedly, shutting down executor.
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {'EngineCore': 1}
```

Setup: 2x Quadro RTX 8000 as pipeline stage 0, 2x Tesla V100 32 GB as stage 1 (the drafter stage), TP2 x PP2, `VLLM_PP_LAYER_PARTITION=24,24`, `VLLM_QWEN4EXP_PLE_HOST_GIB=6`, fp16:

```text
python -m vllm.entrypoints.openai.api_server --model <nvidia/Qwen3.8-Flash-Next-NVFP4 snapshot> \
  --trust-remote-code --dtype float16 --disable-custom-all-reduce \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --enable-prompt-tokens-details --enable-prefix-caching --distributed-timeout-seconds 3600 \
  --tensor-parallel-size 2 --pipeline-parallel-size 2 --gpu-memory-utilization 0.95 \
  --block-size 16 --max-model-len 262144 --max-num-seqs 1 --max-num-batched-tokens 2048 \
  --language-model-only --async-scheduling \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4,"draft_sample_method":"greedy"}' \
  --compilation-config '{"cudagraph_capture_sizes":[1,2,4,5,8]}'
```

Bisection, one boot each, everything else as above:

- all runtime flags above: crash
- only `--enable-prefix-caching` added to a known-good launch (max-num-seqs 4): boots and serves
- only `--async-scheduling` added: boots and serves
- only `--max-num-seqs 1`: crash
- `--max-num-seqs 2`: boots (warm KV cache 646,993 tokens)
- `--max-num-seqs 4`: boots (warm KV cache 645,599 tokens), serves at 46 to 50 tok/s

The crash reproduced six times with `--max-num-seqs 1`, also with `VLLM_SM70_QUANT_BACKEND=turbomind` and `VLLM_1CAT_ENABLE_SM70_MTP_DEFAULTS=1`, and with `CUDA_LAUNCH_BLOCKING=1` the abort is still reported from the tuner itself, so it does not look like a deferred error from an earlier kernel. The RadixArk NVFP4 export of the same model, whose MTP block is NVFP4 instead of FP8, boots and serves with exactly the same flags including `--max-num-seqs 1`, which points at the FP8 MTP expert GEMM path.

What I could not check: TP4 without pipeline parallelism, the configuration fbdd1700 was validated on. The model does not fit our cards that way, so I cannot tell whether pipeline parallelism is part of the trigger.

Workaround: `--max-num-seqs 2` or higher; the extra sequence slots cost about 0.2 % of the KV cache here. We are not affected in practice and are not debugging this further, but single-request V100 deployments tend to set `--max-num-seqs 1`.

AI assistance (Claude) was used to bisect this and to write the report; the logs and numbers above are from our runs.
