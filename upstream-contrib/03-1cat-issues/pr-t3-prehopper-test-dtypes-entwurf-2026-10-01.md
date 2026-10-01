Title: [Test][SM70] Run the DeepSeek-V4 cache tests on pre-Ampere cards

## Purpose

Four DeepSeek-V4 kernel test modules fail on a Tesla V100 on current main,
none of them because of the code under test:

- `test_fused_kv_insert_indexer` calls
  `_fused_kv_compress_norm_rope_insert_indexer_attn` directly and misses its
  `USE_SOFTWARE_FP8` constexpr (`missing 1 required positional argument`). It
  now passes the value `compress_norm_rope_store_triton` passes. Its MXFP4
  cases pack through `cvt.rn.satfinite.e2m1x2.f32`, which ptxas accepts from
  sm_100 on, so they skip below that.
- `test_indexer_quant_cache_roundtrip` and
  `test_indexer_gather_accepts_upper_bound_output` feed bf16 keys to
  `indexer_k_quant_and_cache`. In `quant_utils.cuh` the bf16 -> fp8
  conversion is `assert(false)` below sm_80; release builds drop the assert,
  so the op writes no codes and the round trip is off by the full value
  (3.2 against a bound of 0.125). With fp16 keys the same op is within the
  bound. Pre-Ampere cards now use fp16, newer ones keep bf16.
- `test_fused_inv_rope_fp8_quant` compiles a Triton kernel with `fp8e4nv`,
  which Triton supports from sm_89 on. The module skips below that, worded
  like the existing fp8e4nv skips in tests/kernels/moe.
- `test_fused_deepseek_v4_qnorm_rope_kv_insert` only skipped when the op was
  not built. The op is built for sm_70 but raises "requires sm_80+" at run
  time, so the module also skips below sm_80.

Tests only; no code under vllm/ changes.

Why this is not a duplicate. `gh pr list --state open --search` for
"fp8e4nv", "USE_SOFTWARE_FP8", "test_compressor_kv_cache", "qnorm_rope_kv" and
"pre-Ampere tests" finds no PR that touches these test files; #726 fixes other
stale SM70 tests (scheduler, spec-token aliasing, QPN8, multihead) and does
not touch them.

AI assistance was used for this change. Every line was reviewed and the tests
below were run by the submitter.

## Test Plan

```bash
pytest tests/kernels/test_compressor_kv_cache.py
pytest tests/kernels/test_fused_inv_rope_fp8_quant.py
pytest tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py
pre-commit run --files <changed files>
```

## Test Result

On main d3046986, each file in its own process:

| | V100 main | V100 this PR | RTX 8000 main | RTX 8000 this PR |
|---|---|---|---|---|
| test_compressor_kv_cache.py | 21 failed, 17 passed | 32 passed, 6 skipped | 35 failed, 3 passed | 20 failed, 12 passed, 6 skipped |
| test_fused_inv_rope_fp8_quant.py | 47 failed, 3 errors | 50 skipped | 47 failed, 3 errors | 50 skipped |
| test_fused_deepseek_v4_qnorm_rope_kv_insert.py | 139 failed | 139 skipped | 139 failed | 139 skipped |

The six skips in the first file are the MXFP4 cases. The 20 failures left on
the Quadro RTX 8000 (sm_75) are all "type fp8e4nv not supported in this
architecture": the cache ops under test pick their software FP8 path on
exactly sm_70, so on Turing they compile fp8e4nv. That is a bug in the ops,
not in the tests; a follow-up PR fixes it (with it, 32 passed and 6 skipped
on both cards).

pre-commit clean.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

---
Status: gesendet 2026-10-01 nachts als https://github.com/1CatAI/1Cat-vLLM/pull/744 (Zweig pr-sm70-prehopper-test-dtypes, a4269536)
