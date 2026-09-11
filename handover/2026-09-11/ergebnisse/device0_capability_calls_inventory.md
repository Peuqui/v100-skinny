# Capability-Aufrufe ohne device_id in 1Cat main fe67339d (+#572), Stand 2026-09-11

| Klasse | Treffer |
|---|---|
| A modulebene (Import) | 85 |
| B gecachter Helfer | 5 |
| C Worker-Laufzeit (Modell/Layer/Backend) | 166 |
| D Hauptprozess (Config/Engine) | 7 |

Klasse A und B sind je Prozess eingefroren und laufen vor/neben der Geraetewahl — erste Kandidaten.
Klasse C laeuft im Worker und ist mit device_id=current_device_index() fixbar (Muster #576/11a).
Klasse D hat kein Worker-Geraet; dort gilt die any-participating-device-Semantik aus #514.


## A modulebene (Import)

| Datei:Zeile | Funktion | Klasse | Code |
|---|---|---|---|
| `vllm/compilation/passes/fusion/allreduce_rms_fusion.py:150` | `-` | `-` | `curr_device = current_platform.get_device_capability()` |
| `vllm/compilation/passes/fusion/sequence_parallelism.py:78` | `-` | `-` | `capability = current_platform.get_device_capability()` |
| `vllm/config/model.py:2028` | `-` | `-` | `device_capability = current_platform.get_device_capability()` |
| `vllm/engine/arg_utils.py:154` | `-` | `-` | `capability = current_platform.get_device_capability()` |
| `vllm/kernels/oink_ops.py:18` | `-` | `-` | `OINK_AVAILABLE = current_platform.has_device_capability(100) and hasattr(` |
| `vllm/model_executor/kernels/linear/__init__.py:417` | `-` | `-` | `_cc = current_platform.get_device_capability()` |
| `vllm/model_executor/kernels/linear/__init__.py:640` | `-` | `-` | `_cc = current_platform.get_device_capability()` |
| `vllm/model_executor/kernels/mhc/tilelang.py:19` | `-` | `-` | `capability = current_platform.get_device_capability()` |
| `vllm/model_executor/kernels/mhc/tilelang.py:45` | `-` | `-` | `capability = current_platform.get_device_capability()` |
| `vllm/model_executor/kernels/mhc/tilelang.py:469` | `-` | `-` | `current_platform.get_device_capability() if current_platform.is_cuda() else None` |
| `vllm/model_executor/kernels/mhc/tilelang.py:781` | `-` | `-` | `current_platform.get_device_capability() if current_platform.is_cuda() else None` |
| `vllm/model_executor/layers/attention/attention.py:172` | `-` | `-` | `and current_platform.has_device_capability(70)` |
| `vllm/model_executor/layers/attention/attention.py:173` | `-` | `-` | `and not current_platform.has_device_capability(75)` |
| `vllm/model_executor/layers/fla/ops/utils.py:153` | `-` | `-` | `or torch.cuda.get_device_capability()[0] >= 9` |
| `vllm/model_executor/layers/fused_moe/fused_moe.py:1314` | `-` | `-` | `and current_platform.has_device_capability(70)` |
| `vllm/model_executor/layers/fused_moe/fused_moe.py:1315` | `-` | `-` | `and not current_platform.has_device_capability(75)` |
| `vllm/model_executor/layers/fused_moe/moe_fused_mul_sum.py:73` | `-` | `-` | `is_sm90_plus = current_platform.has_device_capability(90)` |
| `vllm/model_executor/layers/fused_moe/moe_fused_mul_sum.py:74` | `-` | `-` | `is_sm80_before = not current_platform.has_device_capability(80)` |
| `vllm/model_executor/layers/fused_moe/moe_fused_mul_sum.py:76` | `-` | `-` | `if current_platform.has_device_capability(90):` |
| `vllm/model_executor/layers/fused_moe/oracle/fp8.py:90` | `-` | `-` | `and current_platform.is_device_capability(90)` |
| `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py:477` | `-` | `-` | `if current_platform.is_device_capability(90):` |
| `vllm/model_executor/layers/fused_moe/router/fused_topk_router.py:216` | `-` | `-` | `and current_platform.is_device_capability(70)` |
| `vllm/model_executor/layers/fused_moe/utils.py:413` | `-` | `-` | `use_gdc = current_platform.is_cuda() and current_platform.has_device_capability(90)` |
| `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:121` | `-` | `-` | `or not current_platform.is_device_capability(70)` |
| `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:1451` | `-` | `-` | `and current_platform.is_device_capability(70)` |
| `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:1738` | `-` | `-` | `if current_platform.is_device_capability(90):` |
| `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:1741` | `-` | `-` | `capability = current_platform.get_device_capability()` |
| `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:1822` | `-` | `-` | `if active_backend == "flashinfer" and current_platform.is_device_capability(90):` |
| `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:7419` | `-` | `-` | `if envs.VLLM_SM70_GDN_Z_CONTIGUOUS and current_platform.is_device_capability(` |
| `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:7517` | `-` | `-` | `if envs.VLLM_SM70_GDN_Z_CONTIGUOUS and current_platform.is_device_capability(70):` |
| `vllm/model_executor/layers/quantization/awq_marlin.py:84` | `-` | `-` | `and current_platform.has_device_capability(70)` |
| `vllm/model_executor/layers/quantization/awq_marlin.py:85` | `-` | `-` | `and not current_platform.has_device_capability(75)` |
| `vllm/model_executor/layers/quantization/utils/allspark_utils.py:22` | `-` | `-` | `capability_tuple = current_platform.get_device_capability()` |
| `vllm/model_executor/layers/quantization/utils/marlin_utils.py:55` | `-` | `-` | `capability_tuple = current_platform.get_device_capability()` |
| `vllm/model_executor/layers/quantization/utils/marlin_utils.py:120` | `-` | `-` | `capability_tuple = current_platform.get_device_capability()` |
| `vllm/model_executor/layers/quantization/utils/marlin_utils.py:476` | `-` | `-` | `device_capability = torch.cuda.get_device_capability(device)` |
| `vllm/model_executor/layers/quantization/utils/mxfp8_utils.py:94` | `-` | `-` | `if current_platform.has_device_capability(100):` |
| `vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py:57` | `-` | `-` | `) or (current_platform.is_cuda() and current_platform.has_device_capability(70))` |
| `vllm/model_executor/layers/vocab_parallel_embedding.py:479` | `-` | `-` | `if torch.cuda.get_device_capability(x.device) != (7, 0):` |
| `vllm/model_executor/layers/vocab_parallel_embedding.py:560` | `-` | `-` | `if torch.cuda.get_device_capability(x.device) != (7, 0):` |
| `vllm/model_executor/models/minimax_h3/encoder.py:455` | `-` | `-` | `if query.is_cuda and torch.cuda.get_device_capability(query.device) == (7, 0):` |
| `vllm/model_executor/models/pixtral.py:93` | `-` | `-` | `if current_platform.is_cuda() and current_platform.has_device_capability(100):` |
| `vllm/models/deepseek_v4/common/ops/cache_utils.py:203` | `-` | `-` | `current_platform.is_cuda() and current_platform.is_device_capability((7, 0))` |
| `vllm/models/deepseek_v4/common/ops/cache_utils.py:366` | `-` | `-` | `current_platform.is_cuda() and current_platform.is_device_capability((7, 0))` |
| `vllm/models/deepseek_v4/common/ops/cache_utils.py:386` | `-` | `-` | `current_platform.is_cuda() and current_platform.is_device_capability((7, 0))` |
| `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py:257` | `-` | `-` | `current_platform.is_cuda() and current_platform.is_device_capability((7, 0))` |
| `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py:391` | `-` | `-` | `if current_platform.is_cuda() and current_platform.is_device_capability((7, 0)):` |
| `vllm/models/deepseek_v4/nvidia/dspark.py:267` | `-` | `-` | `if current_platform.is_device_capability((7, 0)):` |
| `vllm/models/deepseek_v4/sm70/gemv.py:274` | `-` | `-` | `and current_platform.is_device_capability((7, 0))` |
| `vllm/models/deepseek_v4/sm70/indexer.py:791` | `-` | `-` | `and current_platform.is_device_capability((7, 0))` |
| `vllm/models/qwen4_exp/amd/ops/qsa.py:778` | `-` | `-` | `and current_platform.has_device_capability(90)` |
| `vllm/models/qwen4_exp/nvidia/ops/hc.py:362` | `-` | `-` | `sm70_decode = N == 1 and current_platform.is_device_capability(70)` |
| `vllm/models/qwen4_exp/nvidia/ops/qsa.py:1125` | `-` | `-` | `sm70_single_token = q.shape[0] == 1 and current_platform.is_device_capability(70)` |
| `vllm/models/qwen4_exp/nvidia/ops/qsa.py:1200` | `-` | `-` | `and current_platform.is_device_capability(70)` |
| `vllm/models/qwen4_exp/nvidia/ops/qsa.py:1477` | `-` | `-` | `and current_platform.has_device_capability(90)` |
| `vllm/models/qwen4_exp/nvidia/ops/qsa.py:1584` | `-` | `-` | `and current_platform.is_device_capability(70)` |
| `vllm/models/qwen4_exp/nvidia/ops/qsa.py:2204` | `-` | `-` | `not current_platform.has_device_capability(80),` |
| `vllm/models/qwen4_exp/nvidia/ops/qsa.py:2347` | `-` | `-` | `and current_platform.is_device_capability(70)` |
| `vllm/models/qwen4_exp/nvidia/sm70_fp16_gemv.py:387` | `-` | `-` | `capability_ok = current_platform.is_device_capability((7, 0))` |
| `vllm/models/qwen4_exp/nvidia/sm70_fp16_hc.py:433` | `-` | `-` | `or not current_platform.is_device_capability((7, 0))` |
| `vllm/v1/attention/backends/fa_utils.py:86` | `-` | `-` | `device_capability = current_platform.get_device_capability()` |
| `vllm/v1/attention/backends/flash_attn_v100.py:1662` | `-` | `-` | `device_capability = current_platform.get_device_capability(device_index)` |
| `vllm/v1/attention/backends/mla/prefill/selector.py:116` | `-` | `-` | `device_capability = current_platform.get_device_capability()` |
| `vllm/v1/attention/backends/triton_attn.py:68` | `-` | `-` | `return torch.cuda.get_device_capability(key.device)[0] < 8` |
| `vllm/v1/attention/ops/prefix_prefill.py:15` | `-` | `-` | `BASE_BLOCK = 128 if current_platform.has_device_capability(80) else 64` |
| `vllm/v1/attention/ops/prefix_prefill.py:19` | `-` | `-` | `IS_TURING = current_platform.get_device_capability() == (7, 5)` |
| `vllm/v1/attention/ops/triton_decode_attention.py:507` | `-` | `-` | `is_sm70=current_platform.is_device_capability(70),` |
| `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py:402` | `-` | `-` | `if torch.cuda.get_device_capability(key.device)[0] < 9:` |
| `vllm/v1/attention/ops/triton_unified_attention.py:917` | `-` | `-` | `if current_platform.is_device_capability(70):` |
| `vllm/v1/cudagraph_dispatcher.py:58` | `-` | `-` | `current_platform.is_cuda() and current_platform.is_device_capability((7, 0))` |
| `vllm/v1/cudagraph_dispatcher.py:111` | `-` | `-` | `current_platform.is_cuda() and current_platform.is_device_capability((7, 0))` |
| `vllm/v1/sample/ops/topk_topp_triton.py:34` | `-` | `-` | `or not current_platform.is_device_capability((7, 0))` |
| `vllm/v1/worker/gpu/spec_decode/dflash2/sparse_rejection.py:241` | `-` | `-` | `if torch.cuda.get_device_capability(sample_hidden_states.device) != (7, 0):` |
| `vllm/v1/worker/gpu_model_runner.py:643` | `-` | `-` | `and current_platform.is_device_capability(70)` |
| `vllm/v1/worker/gpu_model_runner.py:676` | `-` | `-` | `and current_platform.is_device_capability(70)` |
| `vllm/v1/worker/gpu_model_runner.py:687` | `-` | `-` | `and current_platform.is_device_capability(70)` |
| `vllm/v1/worker/gpu_model_runner.py:2071` | `-` | `-` | `and current_platform.is_device_capability(70)` |
| `vllm/v1/worker/gpu_model_runner.py:7056` | `-` | `-` | `if torch.cuda.get_device_capability(self.device) != (7, 0):` |
| `vllm/v1/worker/gpu_model_runner.py:7102` | `-` | `-` | `if torch.cuda.get_device_capability(self.device) != (7, 0):` |
| `vllm/v1/worker/gpu_model_runner.py:7742` | `-` | `-` | `if torch.cuda.get_device_capability(self.device) != (7, 0):` |
| `vllm/v1/worker/gpu_model_runner.py:7791` | `-` | `-` | `if torch.cuda.get_device_capability(self.device) != (7, 0):` |
| `vllm/v1/worker/gpu_model_runner.py:8258` | `-` | `-` | `and current_platform.is_device_capability(70)` |
| `vllm/v1/worker/gpu_model_runner.py:10796` | `-` | `-` | `and current_platform.is_device_capability(70)` |
| `vllm/v1/worker/gpu_model_runner.py:11000` | `-` | `-` | `and current_platform.is_device_capability(70)` |
| `vllm/v1/worker/gpu_model_runner.py:11905` | `-` | `-` | `and current_platform.is_device_capability(70)` |

## B gecachter Helfer

| Datei:Zeile | Funktion | Klasse | Code |
|---|---|---|---|
| `vllm/lora/ops/triton_ops/utils.py:327` | `supports_pdl` | `-` | `and current_platform.has_device_capability(90)` |
| `vllm/lora/ops/triton_ops/utils.py:335` | `supports_tma` | `-` | `return current_platform.is_cuda() and current_platform.has_device_capability(90)` |
| `vllm/model_executor/models/qwen3_dflash2.py:77` | `_flashinfer_topk` | `-` | `if not current_platform.has_device_capability(80):` |
| `vllm/utils/flashinfer.py:880` | `has_flashinfer_fp8_blockscale_gemm` | `-` | `and current_platform.is_device_capability(90)` |
| `vllm/utils/flashinfer.py:934` | `is_flashinfer_cudnn_fp8_prefill_attn_supported` | `-` | `if not current_platform.has_device_capability(90):` |

## C Worker-Laufzeit (Modell/Layer/Backend)

| Datei:Zeile | Funktion | Klasse | Code |
|---|---|---|---|
| `vllm/compilation/passes/fusion/allreduce_rms_fusion.py:1031` | `__init__` | `AllReduceFusionPass` | `and current_platform.is_device_capability(70)` |
| `vllm/compilation/passes/fusion/allreduce_rms_fusion.py:1187` | `register_patterns` | `AllReduceFusionPass` | `if current_platform.has_device_capability(100):` |
| `vllm/distributed/device_communicators/cuda_communicator.py:101` | `-` | `CudaCommunicator` | `and current_platform.is_device_capability(70)` |
| `vllm/distributed/device_communicators/custom_all_reduce.py:182` | `-` | `CustomAllreduce` | `device_capability = current_platform.get_device_capability()` |
| `vllm/distributed/device_communicators/symm_mem.py:57` | `-` | `SymmMemCommunicator` | `capability = current_platform.get_device_capability()` |
| `vllm/model_executor/kernels/linear/mixed_precision/cutlass.py:36` | `can_implement` | `CutlassW4A8LinearKernel` | `if not current_platform.is_device_capability(90):` |
| `vllm/model_executor/kernels/linear/mixed_precision/machete.py:35` | `can_implement` | `MacheteLinearKernel` | `if not current_platform.is_device_capability(90):` |
| `vllm/model_executor/kernels/linear/mxfp4/flashinfer.py:25` | `-` | `FlashInferMxFp4LinearKernel` | `if current_platform.has_device_capability(100) and has_flashinfer_cutedsl():` |
| `vllm/model_executor/kernels/linear/mxfp8/flashinfer.py:25` | `-` | `FlashInferCutlassMxfp8LinearKernel` | `if current_platform.has_device_capability(100):` |
| `vllm/model_executor/kernels/linear/mxfp8/marlin.py:18` | `-` | `MarlinMxfp8LinearKernel` | `if current_platform.is_cuda() and current_platform.has_device_capability(75):` |
| `vllm/model_executor/kernels/linear/nvfp4/flashinfer.py:36` | `-` | `FlashInferCutlassNvFp4LinearKernel` | `and current_platform.has_device_capability(100)` |
| `vllm/model_executor/kernels/linear/nvfp4/flashinfer.py:102` | `-` | `FlashInferTrtllmNvFp4LinearKernel` | `elif not current_platform.has_device_capability(100):` |
| `vllm/model_executor/kernels/linear/nvfp4/flashinfer.py:175` | `-` | `FlashInferCudnnNvFp4LinearKernel` | `elif not current_platform.has_device_capability(100):` |
| `vllm/model_executor/kernels/linear/nvfp4/flashinfer.py:240` | `-` | `FlashInferB12xNvFp4LinearKernel` | `if current_platform.has_device_capability(120) and has_flashinfer_b12x_gemm():` |
| `vllm/model_executor/kernels/linear/scaled_mm/cutlass.py:279` | `__init__` | `CutlassFp8BlockScaledMMKernel` | `self.is_hopper = current_platform.is_device_capability(90)` |
| `vllm/model_executor/layers/fla/ops/chunk_delta_h.py:27` | `_is_sm70` | `-` | `and torch.cuda.get_device_capability()[0] == 7` |
| `vllm/model_executor/layers/fla/ops/chunk_delta_h.py:28` | `_is_sm70` | `-` | `and torch.cuda.get_device_capability()[1] == 0` |
| `vllm/model_executor/layers/fla/ops/chunk_o.py:29` | `_is_sm70` | `-` | `and torch.cuda.get_device_capability()[0] == 7` |
| `vllm/model_executor/layers/fla/ops/chunk_o.py:30` | `_is_sm70` | `-` | `and torch.cuda.get_device_capability()[1] == 0` |
| `vllm/model_executor/layers/fla/ops/chunk_scaled_dot_kkt.py:25` | `_is_sm70` | `-` | `and torch.cuda.get_device_capability()[0] == 7` |
| `vllm/model_executor/layers/fla/ops/chunk_scaled_dot_kkt.py:26` | `_is_sm70` | `-` | `and torch.cuda.get_device_capability()[1] == 0` |
| `vllm/model_executor/layers/fla/ops/fused_recurrent.py:85` | `_is_sm70_device` | `-` | `major, minor = torch.cuda.get_device_capability(device_index)` |
| `vllm/model_executor/layers/fla/ops/kda.py:37` | `_is_sm70` | `-` | `and torch.cuda.get_device_capability()[0] == 7` |
| `vllm/model_executor/layers/fla/ops/kda.py:38` | `_is_sm70` | `-` | `and torch.cuda.get_device_capability()[1] == 0` |
| `vllm/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py:140` | `_supports_current_device` | `FlashInferExperts` | `p.is_device_capability(90)` |
| `vllm/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py:168` | `-` | `FlashInferExperts` | `and p.has_device_capability(90)` |
| `vllm/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py:177` | `-` | `FlashInferExperts` | `and p.is_device_capability(90)` |
| `vllm/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py:186` | `-` | `FlashInferExperts` | `and p.has_device_capability(100)` |
| `vllm/model_executor/layers/fused_moe/experts/fused_batched_moe.py:771` | `-` | `BatchedTritonExperts` | `p.is_cuda() and p.has_device_capability((8, 9))` |
| `vllm/model_executor/layers/fused_moe/experts/fused_humming_moe.py:165` | `_supports_current_device` | `HummingExpertsBase` | `return platform.is_cuda() and platform.has_device_capability((7, 5))` |
| `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py:44` | `_triton_kernel_moe_supports_current_device` | `-` | `cap = p.get_device_capability()` |
| `vllm/model_executor/layers/fused_moe/experts/marlin_moe.py:589` | `_supports_current_device` | `MarlinExpertsBase` | `return p.is_cuda() and p.has_device_capability((7, 0))` |
| `vllm/model_executor/layers/fused_moe/experts/triton_cutlass_moe.py:27` | `-` | `TritonOrCutlassExperts` | `self.is_sm100 = current_platform.has_device_capability(100)` |
| `vllm/model_executor/layers/fused_moe/experts/triton_moe.py:87` | `-` | `TritonExperts` | `and current_platform.has_device_capability((7, 5))` |
| `vllm/model_executor/layers/fused_moe/modular_kernel.py:1202` | `-` | `FusedMoEKernelModularImpl` | `capability = current_platform.get_device_capability()` |
| `vllm/model_executor/layers/fused_moe/modular_kernel.py:1228` | `-` | `FusedMoEKernelModularImpl` | `capability = current_platform.get_device_capability()` |
| `vllm/model_executor/layers/fused_moe/router/gate_linear.py:39` | `-` | `GateLinear` | `is_hopper_or_blackwell = current_platform.is_device_capability(` |
| `vllm/model_executor/layers/layernorm.py:208` | `_sm70_gemma_long_prefill_available` | `-` | `return current_platform.is_device_capability(70)` |
| `vllm/model_executor/layers/layernorm.py:612` | `-` | `GemmaRMSNorm` | `and current_platform.is_device_capability(70)` |
| `vllm/model_executor/layers/lightning_attn.py:407` | `forward` | `_attention` | `capability = torch.cuda.get_device_capability()` |
| `vllm/model_executor/layers/linear.py:439` | `process_weights_after_loading` | `UnquantizedLinearMethod` | `if torch.cuda.get_device_capability(layer.weight.device) != (7, 0):` |
| `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:715` | `_sm70_current_device_is_volta` | `-` | `return current_platform.is_device_capability(` |
| `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:2430` | `-` | `QwenGatedDeltaNetAttention` | `and current_platform.is_device_capability(70)` |
| `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:2439` | `-` | `QwenGatedDeltaNetAttention` | `current_platform.is_device_capability(70)` |
| `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:2458` | `-` | `QwenGatedDeltaNetAttention` | `and current_platform.is_device_capability(70)` |
| `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:2468` | `-` | `QwenGatedDeltaNetAttention` | `and current_platform.is_device_capability(70)` |
| `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:2473` | `-` | `QwenGatedDeltaNetAttention` | `and current_platform.is_device_capability(70)` |
| `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:2478` | `-` | `QwenGatedDeltaNetAttention` | `and current_platform.is_device_capability(70)` |
| `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:2485` | `-` | `QwenGatedDeltaNetAttention` | `and current_platform.is_device_capability(70)` |
| `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:2558` | `-` | `QwenGatedDeltaNetAttention` | `if current_platform.is_device_capability(70) and (` |
| `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:3818` | `-` | `QwenGatedDeltaNetAttention` | `capability = torch.cuda.get_device_capability(mixed_qkv.device)` |
| `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:4258` | `-` | `QwenGatedDeltaNetAttention` | `if envs.VLLM_SM70_GDN_Z_CONTIGUOUS and current_platform.is_device_capability(` |
| `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:4404` | `_warmup_sm70_causal_conv1d_real_state` | `QwenGatedDeltaNetAttention` | `if not current_platform.is_device_capability(70):` |
| `vllm/model_executor/layers/mhc.py:19` | `_use_sm70_fp16_mhc_fallback` | `-` | `capability = current_platform.get_device_capability()` |
| `vllm/model_executor/layers/quantization/awq.py:218` | `-` | `AWQConfig` | `and current_platform.has_device_capability(70)` |
| `vllm/model_executor/layers/quantization/awq.py:219` | `-` | `AWQConfig` | `and not current_platform.has_device_capability(75)` |
| `vllm/model_executor/layers/quantization/awq.py:252` | `-` | `AWQConfig` | `and current_platform.has_device_capability(70)` |
| `vllm/model_executor/layers/quantization/awq.py:253` | `-` | `AWQConfig` | `and not current_platform.has_device_capability(75)` |
| `vllm/model_executor/layers/quantization/awq.py:448` | `process_weights_after_loading` | `AWQLinearMethod` | `cap = torch.cuda.get_device_capability(layer.qweight.device)` |
| `vllm/model_executor/layers/quantization/awq.py:634` | `-` | `AWQLinearMethod` | `elif current_platform.is_cuda() and current_platform.is_device_capability(70):` |
| `vllm/model_executor/layers/quantization/awq_marlin.py:301` | `-` | `AWQMarlinConfig` | `and current_platform.has_device_capability(70)` |
| `vllm/model_executor/layers/quantization/awq_marlin.py:302` | `-` | `AWQMarlinConfig` | `and not current_platform.has_device_capability(75)` |
| `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py:339` | `-` | `CompressedTensorsConfig` | `capability_tuple = current_platform.get_device_capability()` |
| `vllm/model_executor/layers/quantization/fbgemm_fp8.py:55` | `__init__` | `FBGEMMFp8Config` | `self.use_marlin = not current_platform.has_device_capability(89)` |
| `vllm/model_executor/layers/quantization/fp8.py:540` | `get_min_capability` | `Fp8Config` | `and current_platform.has_device_capability(70)` |
| `vllm/model_executor/layers/quantization/fp8.py:541` | `get_min_capability` | `Fp8Config` | `and not current_platform.has_device_capability(75)` |
| `vllm/model_executor/layers/quantization/fp8.py:615` | `-` | `Fp8Config` | `and current_platform.has_device_capability(70)` |
| `vllm/model_executor/layers/quantization/fp8.py:616` | `-` | `Fp8Config` | `and not current_platform.has_device_capability(75)` |
| `vllm/model_executor/layers/quantization/fp8.py:625` | `-` | `Fp8Config` | `and current_platform.has_device_capability(70)` |
| `vllm/model_executor/layers/quantization/fp8.py:626` | `-` | `Fp8Config` | `and not current_platform.has_device_capability(75)` |
| `vllm/model_executor/layers/quantization/fp8.py:729` | `__init__` | `Fp8LinearMethod` | `and current_platform.has_device_capability(70)` |
| `vllm/model_executor/layers/quantization/fp8.py:730` | `__init__` | `Fp8LinearMethod` | `and not current_platform.has_device_capability(75)` |
| `vllm/model_executor/layers/quantization/fp8.py:1901` | `__init__` | `Fp8MoEMethod` | `and current_platform.has_device_capability(70)` |
| `vllm/model_executor/layers/quantization/fp8.py:1902` | `__init__` | `Fp8MoEMethod` | `and not current_platform.has_device_capability(75)` |
| `vllm/model_executor/layers/quantization/gguf.py:65` | `get_supported_act_dtypes` | `GGUFConfig` | `if current_platform.has_device_capability(100):` |
| `vllm/model_executor/layers/quantization/moe_wna16.py:67` | `-` | `MoeWNA16Config` | `capability_tuple = current_platform.get_device_capability()` |
| `vllm/model_executor/layers/quantization/moe_wna16.py:147` | `is_moe_wna16_compatible` | `MoeWNA16Config` | `capability_tuple = current_platform.get_device_capability()` |
| `vllm/model_executor/layers/quantization/quark/quark.py:231` | `_check_scheme_supported` | `QuarkConfig` | `capability_tuple = current_platform.get_device_capability()` |
| `vllm/model_executor/layers/quantization/quark/quark_moe.py:427` | `-` | `QuarkW8A8Fp8MoEMethod` | `not current_platform.has_device_capability(89)` |
| `vllm/model_executor/layers/quantization/sm70_online_qpn8.py:138` | `maybe_prepare_online_qpn8` | `-` | `or not current_platform.is_device_capability(70)` |
| `vllm/model_executor/layers/quantization/sm70_turbomind.py:65` | `is_exact_sm70_cuda` | `-` | `return torch.cuda.get_device_capability(tensor.device) == (7, 0)` |
| `vllm/model_executor/layers/quantization/sm70_turbomind.py:75` | `is_exact_sm70_cuda_platform` | `-` | `return current_platform.is_cuda() and current_platform.is_device_capability((7, 0))` |
| `vllm/model_executor/layers/quantization/torchao.py:119` | `_check_torchao_fp8_activation_capability` | `-` | `capability = current_platform.get_device_capability()` |
| `vllm/model_executor/layers/quantization/utils/flashinfer_fp4_moe.py:43` | `is_flashinfer_fp4_cutlass_moe_available` | `-` | `and current_platform.has_device_capability(100)` |
| `vllm/model_executor/layers/quantization/utils/flashinfer_utils.py:117` | `get_flashinfer_moe_backend` | `-` | `elif current_platform.is_device_capability(90):` |
| `vllm/model_executor/layers/quantization/utils/marlin_utils.py:439` | `maybe_warn_marlin_atomic_add` | `-` | `device_capability = torch.cuda.get_device_capability(device)` |
| `vllm/model_executor/layers/quantization/utils/marlin_utils.py:500` | `get_marlin_input_dtype` | `-` | `if not current_platform.is_device_capability(` |
| `vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py:29` | `is_fp4_marlin_supported` | `-` | `if current_platform.has_device_capability(75):` |
| `vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py:31` | `is_fp4_marlin_supported` | `-` | `return current_platform.is_device_capability((7, 0)) and ops.sm70_marlin_available()` |
| `vllm/model_executor/layers/quantization/utils/marlin_utils_fp8.py:25` | `is_fp8_marlin_supported` | `-` | `return current_platform.is_cuda() and current_platform.has_device_capability(70)` |
| `vllm/model_executor/layers/quantization/utils/mxfp4_utils.py:36` | `_swizzle_mxfp4` | `-` | `and current_platform.is_device_capability(90)` |
| `vllm/model_executor/layers/quantization/utils/mxfp4_utils.py:73` | `_swizzle_mxfp4` | `-` | `if current_platform.is_device_capability(90):` |
| `vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py:35` | `_supports_triton_nvfp4_emulation` | `-` | `and current_platform.has_device_capability(89)` |
| `vllm/model_executor/layers/quantization/utils/nvfp4_utils.py:59` | `cutlass_fp4_supported` | `-` | `capability_tuple = current_platform.get_device_capability()` |
| `vllm/model_executor/layers/quantization/utils/w8a8_utils.py:22` | `cutlass_fp8_supported` | `-` | `capability_tuple = current_platform.get_device_capability()` |
| `vllm/model_executor/layers/quantization/utils/w8a8_utils.py:32` | `cutlass_block_fp8_supported` | `-` | `capability_tuple = current_platform.get_device_capability()` |
| `vllm/model_executor/layers/quantization/utils/w8a8_utils.py:42` | `cutlass_group_gemm_supported` | `-` | `capability_tuple = current_platform.get_device_capability()` |
| `vllm/model_executor/layers/sm70_diffusion.py:117` | `supports_fused_scaled_add` | `-` | `and torch.cuda.get_device_capability(output.device) == (7, 0)` |
| `vllm/model_executor/layers/sparse_attn_indexer.py:41` | `_is_exact_sm70_cuda` | `-` | `return current_platform.is_cuda() and current_platform.is_device_capability((7, 0))` |
| `vllm/model_executor/layers/sparse_attn_indexer_kpool.py:57` | `_is_exact_sm70_cuda` | `-` | `return current_platform.is_cuda() and current_platform.is_device_capability((7, 0))` |
| `vllm/model_executor/layers/vocab_parallel_embedding.py:124` | `_is_sm70_lm_head_fastpath_eligible` | `-` | `if torch.cuda.get_device_capability(layer.weight.device) != (7, 0):` |
| `vllm/model_executor/layers/vocab_parallel_embedding.py:126` | `_is_sm70_lm_head_fastpath_eligible` | `-` | `f"capability={torch.cuda.get_device_capability(layer.weight.device)}"` |
| `vllm/model_executor/models/deepseek_v2.py:849` | `-` | `DeepSeekV2FusedQkvAProjLinear` | `current_platform.is_device_capability(90)` |
| `vllm/model_executor/models/gemma4_mm.py:979` | `__init__` | `-` | `and current_platform.is_device_capability(70)` |
| `vllm/model_executor/models/qwen2_moe.py:104` | `_sm70_force_shared_expert_silu_custom_op` | `-` | `return torch.cuda.get_device_capability() == (7, 0)` |
| `vllm/model_executor/models/qwen3_5.py:556` | `-` | `Qwen3_5DecoderLayer` | `and current_platform.is_device_capability(70)` |
| `vllm/model_executor/models/qwen3_dflash.py:650` | `_normalize_context_k` | `DFlashQwen3Model` | `and current_platform.is_device_capability(70)` |
| `vllm/model_executor/models/qwen3_dflash2.py:64` | `_use_sm70_bf16_emulation` | `-` | `and current_platform.is_device_capability(70)` |
| `vllm/model_executor/models/qwen3_dflash2.py:348` | `-` | `DFlash2Qwen3Model` | `and current_platform.is_device_capability(70)` |
| `vllm/model_executor/warmup/awq_sm70_warmup.py:1038` | `sm70_awq_warmup` | `-` | `if torch.cuda.get_device_capability(device) != (7, 0):` |
| `vllm/model_executor/warmup/kernel_warmup.py:76` | `kernel_warmup` | `-` | `elif has_flashinfer() and current_platform.has_device_capability(90):` |
| `vllm/models/deepseek_v4/attention.py:99` | `_is_exact_sm70_cuda` | `-` | `return current_platform.is_cuda() and current_platform.is_device_capability((7, 0))` |
| `vllm/models/deepseek_v4/attention.py:230` | `-` | `DeepseekV4MultiHeadLatentAttentionWrapper` | `cap = current_platform.get_device_capability()` |
| `vllm/models/deepseek_v4/compressor.py:51` | `_can_use_sm70_private_compressor_state` | `-` | `and current_platform.is_device_capability((7, 0))` |
| `vllm/models/deepseek_v4/compressor.py:472` | `-` | `DeepseekCompressor` | `if current_platform.is_cuda() and not current_platform.is_device_capability(` |
| `vllm/models/deepseek_v4/nvidia/dspark.py:142` | `-` | `DSparkDeepseekV4Model` | `2.0**-6 if current_platform.is_device_capability((7, 0)) else 1.0` |
| `vllm/models/deepseek_v4/nvidia/model.py:261` | `_check_runtime_supported` | `DeepseekV4MegaMoEExperts` | `if torch.cuda.get_device_capability(device)[0] != 10:` |
| `vllm/models/deepseek_v4/nvidia/model.py:1346` | `__init__` | `DeepseekV4ForCausalLM` | `and current_platform.is_device_capability((7, 0))` |
| `vllm/models/deepseek_v4/sm70/gemv.py:244` | `prepare_sm70_dsv4_fp13_gemv` | `-` | `or not current_platform.is_device_capability((7, 0))` |
| `vllm/models/glm5next/nvidia/attention.py:372` | `-` | `Indexer` | `current_platform.is_device_capability((7, 0))` |
| `vllm/models/glm5next/nvidia/kda.py:356` | `-` | `Glm5NextLinearAttention` | `and current_platform.get_device_capability() == (7, 0)` |
| `vllm/models/glm5next/nvidia/kda.py:363` | `-` | `Glm5NextLinearAttention` | `and current_platform.get_device_capability() == (7, 0)` |
| `vllm/models/glm5next/nvidia/kda.py:379` | `-` | `Glm5NextLinearAttention` | `and current_platform.get_device_capability() == (7, 0)` |
| `vllm/models/glm5next/nvidia/kda.py:384` | `-` | `Glm5NextLinearAttention` | `and current_platform.get_device_capability() == (7, 0)` |
| `vllm/models/glm5next/nvidia/ops/kpool_compress.py:28` | `_is_exact_sm70_cuda` | `-` | `return current_platform.is_cuda() and current_platform.is_device_capability((7, 0))` |
| `vllm/models/qwen4_exp/nvidia/low_latency_gemm.py:83` | `_is_sm103` | `-` | `return current_platform.is_device_capability((10, 3))` |
| `vllm/models/qwen4_exp/nvidia/ops/qsa.py:1213` | `_use_sm70_qsa_lexicographic_topk` | `-` | `return topk == 512 and current_platform.is_device_capability(70)` |
| `vllm/models/qwen4_exp/nvidia/ops/qsa.py:2064` | `_use_sm70_qsa_resolved_indices` | `-` | `current_platform.is_device_capability(70)` |
| `vllm/models/qwen4_exp/nvidia/ple_layer.py:519` | `_should_use_pinned_host_ple` | `-` | `capability = current_platform.get_device_capability(` |
| `vllm/models/qwen4_exp/nvidia/ple_layer.py:1115` | `-` | `Qwen4ExpNGramEmbedding` | `and current_platform.is_device_capability((7, 0))` |
| `vllm/models/qwen4_exp/nvidia/ple_layer.py:1676` | `-` | `Qwen4ExpPLELayer` | `current_platform.is_device_capability((7, 0))` |
| `vllm/models/qwen4_exp/nvidia/qsa.py:248` | `-` | `Qwen4ExpQSAAttention` | `if e4m3_cache and not current_platform.is_device_capability(70):` |
| `vllm/models/qwen4_exp/nvidia/sm70_fp16_gemv.py:122` | `_can_fuse_gdn_projection_split` | `-` | `and current_platform.is_device_capability(70)` |
| `vllm/v1/attention/backends/flash_attn_v100.py:3196` | `__init__` | `FlashAttnV100MetadataBuilder` | `and current_platform.is_device_capability(70)` |
| `vllm/v1/attention/backends/flash_attn_v100.py:4451` | `__init__` | `FlashAttnV100Impl` | `and current_platform.is_device_capability(70)` |
| `vllm/v1/attention/backends/flash_attn_v100.py:4596` | `__init__` | `FlashAttnV100Impl` | `and current_platform.is_device_capability(70)` |
| `vllm/v1/attention/backends/flashinfer.py:431` | `get_required_kv_cache_layout` | `FlashInferBackend` | `capability = current_platform.get_device_capability()` |
| `vllm/v1/attention/backends/mla/flashmla_sparse.py:674` | `-` | `FlashMLASparseMetadataBuilder` | `and current_platform.is_device_capability((7, 0))` |
| `vllm/v1/attention/backends/mla/prefill/flash_attn.py:31` | `_is_sm70_flash_v100_platform` | `-` | `capability = current_platform.get_device_capability()` |
| `vllm/v1/attention/backends/mla/prefill/flash_attn.py:146` | `-` | `FlashAttnPrefillBackend` | `device_capability = current_platform.get_device_capability()` |
| `vllm/v1/attention/backends/mla/sparse_swa.py:499` | `-` | `DeepseekSparseSWAMetadataBuilder` | `and current_platform.is_device_capability((7, 0))` |
| `vllm/v1/attention/backends/rocm_aiter_fa.py:768` | `supports_compute_capability` | `AiterFlashAttentionBackend` | `# DeviceCapability is currently created using torch.cuda.get_device_capability()` |
| `vllm/v1/attention/backends/turboquant_attn.py:97` | `_flash_attn_varlen_supported_on_device` | `-` | `device_capability = current_platform.get_device_capability()` |
| `vllm/v1/attention/ops/triton_prefill_attention.py:183` | `get_block_size` | `-` | `elif current_platform.is_cuda_alike() and current_platform.has_device_capability(` |
| `vllm/v1/attention/ops/triton_turboquant_decode.py:31` | `_use_fp8_e4b15` | `-` | `cap = torch.cuda.get_device_capability(device)` |
| `vllm/v1/sample/ops/topk_topp_sampler.py:59` | `__init__` | `TopKTopPSampler` | `capability = current_platform.get_device_capability()` |
| `vllm/v1/sample/sampler.py:320` | `-` | `Sampler` | `or torch.cuda.get_device_capability(logits.device) != (7, 0)` |
| `vllm/v1/sample/sampler.py:426` | `-` | `Sampler` | `or torch.cuda.get_device_capability(top_values.device) != (7, 0)` |
| `vllm/v1/spec_decode/dflash.py:65` | `_create_draft_vllm_config` | `DFlashProposer` | `and current_platform.is_device_capability(70)` |
| `vllm/v1/spec_decode/dflash.py:185` | `warmup_sm70_dflash_hotpath_kernels` | `DFlashProposer` | `if self.device.type != "cuda" or not current_platform.is_device_capability(70):` |
| `vllm/v1/spec_decode/llm_base_proposer.py:1005` | `warmup_sm70_mtp_hotpath_kernels` | `SpecDecodeBaseProposer` | `or not current_platform.is_device_capability(70)` |
| `vllm/v1/spec_decode/llm_base_proposer.py:1043` | `warmup_sm70_mtp_moe_kernels` | `SpecDecodeBaseProposer` | `or not current_platform.is_device_capability(70)` |
| `vllm/v1/worker/gpu/cudagraph_utils.py:66` | `_use_split_sm70_mtp_cudagraphs` | `-` | `and current_platform.is_device_capability((7, 0))` |
| `vllm/v1/worker/gpu/cudagraph_utils.py:336` | `-` | `CudaGraphManager` | `and current_platform.is_device_capability((7, 0))` |
| `vllm/v1/worker/gpu/cudagraph_utils.py:420` | `-` | `ModelCudaGraphManager` | `and current_platform.is_device_capability((7, 0))` |
| `vllm/v1/worker/gpu/model_runner.py:703` | `_warmup_sm70_aux_kernels` | `GPUModelRunner` | `or not current_platform.is_device_capability(70)` |
| `vllm/v1/worker/gpu/model_runner.py:1309` | `-` | `GPUModelRunner` | `and current_platform.is_device_capability(70)` |
| `vllm/v1/worker/gpu/model_states/mamba_hybrid.py:150` | `-` | `MambaHybridModelState` | `and current_platform.is_device_capability(70)` |
| `vllm/v1/worker/gpu/model_states/mamba_hybrid.py:162` | `-` | `MambaHybridModelState` | `and current_platform.is_device_capability(70)` |
| `vllm/v1/worker/gpu/spec_decode/dflash/utils.py:77` | `load_dflash_model` | `-` | `and current_platform.is_device_capability(70)` |
| `vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py:32` | `_requires_sm70_tail` | `-` | `and torch.cuda.get_device_capability(device) == (7, 0)` |
| `vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py:759` | `capture` | `DFlash2Speculator` | `or torch.cuda.get_device_capability(self.device) != (7, 0)` |
| `vllm/v1/worker/gpu/spec_decode/eagle/speculator.py:232` | `-` | `EagleSpeculator` | `or not current_platform.is_device_capability(70)` |
| `vllm/v1/worker/gpu_model_runner.py:2234` | `_warmup_sm70_aux_kernels` | `-` | `if not current_platform.is_device_capability(70):` |
| `vllm/v1/worker/gpu_worker.py:989` | `_use_sm70_static_pp_hidden_transfer` | `Worker` | `and current_platform.is_device_capability((7, 0))` |
| `vllm/vllm_flash_attn/flash_attn_interface.py:57` | `_is_fa2_supported` | `-` | `if not current_platform.has_device_capability(80):` |

## D Hauptprozess (Config/Engine)

| Datei:Zeile | Funktion | Klasse | Code |
|---|---|---|---|
| `vllm/config/compilation.py:211` | `default_fi_allreduce_fusion_max_size_mb` | `PassConfig` | `capability = current_platform.get_device_capability()` |
| `vllm/config/vllm.py:567` | `enable_allreduce_rms_fusion` | `-` | `and current_platform.is_device_capability(70)` |
| `vllm/config/vllm.py:577` | `enable_allreduce_rms_fusion` | `-` | `or current_platform.is_device_capability(90)` |
| `vllm/config/vllm.py:1176` | `-` | `VllmConfig` | `capability_tuple = current_platform.get_device_capability()` |
| `vllm/config/vllm.py:1715` | `__post_init__` | `VllmConfig` | `and current_platform.get_device_capability() == (7, 5)` |
| `vllm/config/vllm.py:2859` | `_set_cudagraph_sizes` | `VllmConfig` | `cap = current_platform.get_device_capability()` |
| `vllm/engine/arg_utils.py:1762` | `-` | `EngineArgs` | `cap = current_platform.get_device_capability()` |
