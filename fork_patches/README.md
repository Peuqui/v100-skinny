# Fork patches (1Cat-vLLM 1.2.2)

**These files are derivative works of [vLLM](https://github.com/vllm-project/vllm),
Copyright contributors to the vLLM project, licensed under Apache-2.0** — not
under this repository's MIT default. Each states its modifications in its
header as Apache-2.0 §4(b) requires. See [`../LICENSE-APACHE-2.0`](../LICENSE-APACHE-2.0)
and [`../NOTICE`](../NOTICE).

They are complete upstream sources carrying local edits, tracked here so the
changes are diffable and reversible outside the installed package. The engine
is installed as a wheel (no source checkout), so the patched files are copied
over the corresponding paths under the environment's `site-packages/`.

**Install them with [`../scripts/bootstrap-sm70.sh`](../scripts/bootstrap-sm70.sh)**,
which resolves the target paths, keeps a `.pre_bootstrap` backup of every file
it replaces, and verifies the kernel extension afterwards. Never hand-edit an
installed file: the tracked copy here is the reviewable source of truth.

| File | Install path (under `site-packages/`) | What we changed |
|---|---|---|
| `marlin.py` | `vllm/model_executor/kernels/linear/nvfp4/marlin.py` | The skinny NVFP4 dispatch: QPN2 geometry winners own decode M 1–8 including `lm_head`, per-shape (split, nacc) table, shape-aware route map; legacy QPN for M 9–16; Marlin above that. |
| `modelopt.py` | `vllm/model_executor/layers/quantization/modelopt.py` | The QPN8 FP8 W8A16 path (`mma.sync.m8n8k4`, incl. the MT=2 two-tile variant), lowers the ModelOpt minimum compute capability from SM89 to SM70, adds route/census logging. **This is what lets a published mixed FP8+NVFP4 checkpoint load on Volta at all.** |
| `torch_utils.py` | `vllm/utils/torch_utils.py` | KV-dtype policy: a checkpoint's `kv_cache_quant_algo` describes how its *weights* were made and is no longer honoured as a KV-cache directive below SM80. Without this the verbatim checkpoint silently booted an FP8 KV cache and lost the tensor-core decode route (+4.82 ms/round). |
| `attention.py` | `vllm/model_executor/layers/attention/attention.py` | The same policy on the compressed-tensors re-apply path. |
| `gpu_model_runner.py` | `vllm/v1/worker/gpu_model_runner.py` | Persistent-metadata speculative round, a per-phase GPU profiler, and NVTX phase brackets for per-kernel attribution. |
| `gdn_attn.py` | `vllm/v1/attention/backends/gdn_attn.py` | Chain-MTP GDN fast metadata build (−1.4 ms/step, byte-identical output). |
| `vllm_config.py` | `vllm/config/vllm.py` | The SM70 baseline env defaults (GDN decode FlashQLA, the GDN/FLA schedules, packed recurrent decode, the 0DOT3 compile graph) are decided by whether ANY visible device is SM70 instead of by device 0. This block runs once in the parent process and its `os.environ` defaults are inherited by every worker, so on a heterogeneous pipeline the V100 stage silently lost its entire tuning whenever device 0 was a Turing card — output stayed grammatical but degraded, and MTP acceptance collapsed. |
| `custom_all_reduce.py` | `vllm/distributed/device_communicators/custom_all_reduce.py` | All-reduce residency instrumentation, default off. Measurement tool, dormant in production. |
| `dsv4_cache_utils.py` | `vllm/models/deepseek_v4/common/ops/cache_utils.py` | Software-FP8/CuteDSL gating keyed on the absence of native FP8 units (< SM89) instead of exactly SM70, so the SM75 stages of a mixed V100+RTX8000 pipeline take the software path (three sites). |
| `dsv4_fused_compress_quant_cache.py` | `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py` | Same < SM89 gating for `USE_SOFTWARE_FP8`. |
| `dsv4_fused_indexer_q.py` | `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py` | Same < SM89 gating for the software indexer branch. |
| `dsv4_dspark.py` | `vllm/models/deepseek_v4/nvidia/dspark.py` | `DSparkDeepseekV4ForCausalLM` implements `SupportsPP` (interface only; the drafter runs whole on the last stage) so a dspark PP boot passes the startup gate. Same shape as the fork's own `DeepSeekV4MTP` addition. |
| `dsv4_amd_rocm.py` | `vllm/models/deepseek_v4/amd/rocm.py` | SWA ragged copy sized from the actual dense row width — DSpark drafting rows are wider than `window_size` and the old slice cut the copy short. |
| `multiproc_executor.py` | `vllm/v1/executor/multiproc_executor.py` | `VLLM_SM70_ASYNC_SCHEDULING_QUEUE_DEPTH` also caps the PP batch queue (depth pp_size deadlocked a 5-stage spec-decode boot). |
| `gpu_worker.py` | `vllm/v1/worker/gpu_worker.py` | Per-rank KV availability logging; `VLLM_PP_SEAM_TRACE` early-return diagnostics. |
| `parallel_state.py` | `vllm/distributed/parallel_state.py` | `VLLM_PP_SEAM_TRACE` diagnostics on the PP tensor-dict seam (metadata send/recv). |
| `spec_decode_dspark.py` | `vllm/v1/spec_decode/dspark.py` | `VLLM_DSPARK_DIAG` base-logits diagnostics in the draft sampler. |
| `flashmla_sparse.py`, `sm70_turbomind.py`, `dsv4_sm70_gemv.py` (+ edits in `deepseek_v4_attention.py`, `sparse_attn_indexer.py`, `sparse_swa.py`) | see paths | The DeepSeek 'exactly SM70' gates decide on the worker's device instead of device 0, so V100 stages of a mixed pipeline keep their SM70 paths; the grouped SM70 O-projection is selected only when `wo_a` is really TurboMind-prepared (with the QPN8-blk is_bmm route it is fp16-dequantised and the reference einsum applies). |
| `breakable_cudagraph.py` | `vllm/compilation/breakable_cudagraph.py` | `eager_break_during_capture(ignore_full_mode=True)`: a capture break that also fires under the FULL runtime mode -- the host-driven skinny NVFP4 MoE (`nvfp4_skinny_moe.py`, routes on `topk_ids.cpu()`) uses it so DeepSeek V4 captures CUDA graphs on pre-Ampere cards. Companion edits: pre-Ampere ranks feed the fp16 Triton indexer from the fused software producer (`deepseek_v4_attention.py`, `sparse_attn_indexer.py`); boot recipe `scripts/serve-deepseek-het-graphs.sh`. |

Not installed:

| File | Why |
|---|---|
| `sm70_native_round.py` | Original work (not derived from vLLM), offered under Apache-2.0 so it can combine with the engine. Experimental: built and validated byte-identical, but inert — the captured graph does not persist the drafter's recurrent state across rounds, so served drafts are rejected. |
| `llm_base_proposer.native_round.patch` | The proposer hook that would select the above. Reverted; kept as a diff for development in a proper source checkout. |
