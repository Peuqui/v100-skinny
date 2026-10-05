Titel: [Bugfix][SM70] Start block QPN8 on Volta: warm up the layout it holds, keep the TurboMind copy only when decode can use it

## Purpose

Two problems kept DeepSeek-V4-Flash (block FP8 linears) from starting with main's defaults on our rig (3x Tesla V100-PCIE-32GB + 2x Quadro RTX 8000, PP5 over all five cards, DSpark drafter, `--max-num-seqs 1`). They show up one after the other.

1. Memory. Since 63406aed6, block QPN8 keeps a second, TurboMind-packed copy of every block FP8 weight on Volta for rows beyond M=8. On the 8-layer V100 stages of DeepSeek-V4-Flash the start then ran out of memory while profiling CUDA graph memory (`Triton Error [CUDA]: out of memory` in `compress_norm_rope_store_triton`, PP1 and PP3): 5 of 5 starts, 3 of them with `sm70_fp8.enabled=false`. There was no way to hold one layout.

2. Warmup. Block QPN8 marks its layers `sm70_fp8_turbomind`, so the coordinated Volta dense warmup (`_iter_unique_fp8_dense_layers`, `_warmup_fp8_dense_layers`) treats them as TurboMind layers and reads `layer.sm70_fp8_k_ld`. Block QPN8 never sets it; its copy lives under `_sm70_block_fp8_turbomind_packed_weight` and `_qpn8_fallback_k_ld`. With the copy dropped by hand, the start failed in the warmup with `AttributeError: 'MergedColumnParallelLinear' object has no attribute 'sm70_fp8_k_ld'` (1 of 1). The copy does not help: a layer with the copy still has no `sm70_fp8_k_ld`, so any Volta start that gets past memory with block FP8 weights reaches this.

Changes:

- The warmup recognizes block QPN8 layers. With the Volta copy it warms that copy under the names it is stored with, and only for M > 8, the rows the copy serves. Without the copy the layer has nothing to tune and is skipped.
- New `kernel_config.sm70_fp8.block_qpn8_volta_turbomind_prefill: bool | None = None`. Auto keeps the copy only when decode can exceed 8 rows, `max_num_seqs * (num_speculative_tokens + 1) > 8`; below that the copy serves only prefill rows, where the dense FP16 prefill that Turing already uses keeps pace (table below). True or false overrides auto. When the copy is held, the startup message names the field, so an out-of-memory start points at the way out.

Why auto rather than a fixed default: the copy is worth its memory where decode batches are wider. On a V100 with the DeepSeek-V4-Flash block FP8 shapes the copy is about 4x faster for 9-64 rows and level at 256-512 rows. A single-request engine with 5 speculative tokens never decodes more than 6 rows, which QPN8 serves itself, so there the copy only costs memory. Engines that can decode more than 8 rows keep today's behavior.

Not a duplicate: no open PR touches the block QPN8 Volta copy or the dense warmup (gh pr list --state open --search "QPN8 TurboMind", "block QPN8", "qpn8 prefill", "warmup qpn8": none). #801 and 63406aed6 introduced the code changed here.

AI assistance was used for this change. I reviewed every line and ran the tests and measurements below.

## Test Plan

```bash
# CUDA_HOME pointing at the toolkit the engine uses (tilelang JIT in other tests needs it)
pytest tests/kernels/quantization/test_sm70_qpn8_block_fp8.py tests/quantization/test_block_fp8_qpn8_capabilities.py tests/quantization/test_sm70_warmup.py
pre-commit run --files vllm/config/kernel.py vllm/model_executor/kernels/linear/scaled_mm/qpn8_blk.py vllm/model_executor/warmup/awq_sm70_warmup.py tests/kernels/quantization/test_sm70_qpn8_block_fp8.py
python qpn8_prefill_bench.py   # V100, block FP8 shapes of DeepSeek-V4-Flash, M = 9..512, copy vs dense
```

New tests: one layout when the copy is off (M = 9, 64, 512 against an FP32 reference), the warmup yields and runs the copy only when it is held and only for M > 8, and the auto rule for 1x6, 8x1, 9x1, 2x6 rows plus both overrides.

## Acceleration and benchmark contract (required for performance changes)

Per call, summed over the six block FP8 shapes of a DeepSeek-V4-Flash layer (1536x4096, 32768x1024, 8192x1024, 4096x8192, 4096x4096, 4096x2048), Tesla V100-PCIE-32GB, ms, lower is better:

| rows (M) | TurboMind copy | dense FP16 prefill |
| ---: | ---: | ---: |
| 9 | 0.23 | 0.95 |
| 16 | 0.23 | 0.95 |
| 64 | 0.35 | 0.96 |
| 256 | 1.30 | 1.36 |
| 512 | 2.14 | 1.98 |

Relative error against FP32: 3.5e-4 for the copy, 3.5e-4 to 4.6e-4 dense, per shape.

End to end, DeepSeek-V4-Flash PP5 with DSpark (5 speculative tokens), `--max-num-seqs 1`, prefill chunks of 512, main's defaults and no kernel-config entry, so auto drops the copy: all 3 starts came up; 35.7k-token prompt prefill 15.3-15.6 s, decode 96-98 ms per step at 35.7k context and 75-82 ms at 26-1437 tokens; our production fork (single layout) 15.1-15.3 s and 84-91 ms. The decode difference is not from this change: the sparse indexer route under full graphs, addressed separately in #<PR B>. With `--max-num-seqs 4` (24 decode rows) auto keeps the copy and the start runs out of memory on these 32 GiB V100 stages; with the field set to false the engine serves four concurrent requests (aggregate 61.6-67.2 tok/s, production 55.7-58.7 tok/s, both with #<PR B>).

## Test Result

- Tesla V100-PCIE-32GB: 145 passed. Quadro RTX 8000: 135 passed, 10 skipped (the three Volta-only cases, by their stated condition).
- The 11 new cases on main f551e0afe without this change: 8 fail (no single-layout option, warmup AttributeError, auto rule), the 3 cases where the copy stays pass.
- pre-commit passes.
- Measurements on our build of the #837 head plus #856/#857 (cef0a2e4b, extensions built from source, sm_70 only); on current main the touched files are identical except vllm/config/kernel.py, where main has since added unrelated fields (QPN2 prefill threshold, auxiliary GEMV fusion, PLE result transport). Quality prompts (8) read by hand in both runs: equivalent to production, the unknown-person prompt correctly declined; greedy text differs from our production reference, as expected across builds.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
