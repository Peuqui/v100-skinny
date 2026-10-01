Title: [Bugfix][DeepSeek-V4] Decode FP8 in software on every card without FP8 units

## Purpose

The DeepSeek-V4 cache ops pick their software FP8 path on exactly SM70
(`is_device_capability((7, 0))`). Triton compiles `fp8e4nv` from sm_89 on, so
on Turing (sm_75) the K-cache quantize/insert, the dequantize/gather and the
fused compressor insert take the hardware path and fail to compile
("type fp8e4nv not supported in this architecture"). The CuteDSL gather gate
used the same exact-SM70 test.

`needs_software_fp8()` (no native FP8 before sm_89) now decides at all four
sites: `quantize_and_insert_k_cache`, `dequantize_and_gather_k_cache_triton`,
the CuteDSL gate in `dequantize_and_gather_k_cache`, and
`compress_norm_rope_store_triton`. On SM70 and on sm_89+ nothing changes.

Stacked on #744, whose test passes the kernel's `USE_SOFTWARE_FP8` the
same way the op does; this PR switches that line to the helper as well. #716
adds the same helper to `cache_utils.py` (byte-identical, same place); the
two branches merge cleanly, and whichever lands second can drop its copy.

Why this is not a duplicate. `gh pr list --state open --search` for
"USE_SOFTWARE_FP8", "needs_software_fp8", "software fp8" and "fp8e4nv" finds
only #716 (adds the helper for its own new ops, leaves these call sites on
exact SM70) and #604 (dense linears on Turing, unrelated files). No other open
PR touches `cache_utils.py` or `fused_compress_quant_cache.py`.

AI assistance was used for this change. Every line was reviewed and the tests
below were run by the submitter.

## Test Plan

```bash
pytest tests/kernels/test_compressor_kv_cache.py
pytest tests/kernels/test_deepseek_v4_sm70_qnorm_rope_kv_insert.py
pre-commit run --files <changed files>
pre-commit run mypy-3.10 --hook-stage manual --files <changed files>
```

## Test Result

On main d3046986 plus #744, each file in its own process:

| | Tesla V100 | Quadro RTX 8000 |
|---|---|---|
| test_compressor_kv_cache.py, before | 32 passed, 6 skipped | 20 failed, 12 passed, 6 skipped |
| test_compressor_kv_cache.py, this PR | 32 passed, 6 skipped | 32 passed, 6 skipped |
| test_deepseek_v4_sm70_qnorm_rope_kv_insert.py | 2 passed | 2 skipped |

The 20 failures before are all the fp8e4nv compile error; the six skips are
the MXFP4 cases (sm_100+). The same four call sites have run on the Turing
stages of a mixed 3x V100 + 2x Quadro RTX 8000 DeepSeek-V4-Flash pipeline in
our fork since September. pre-commit clean, mypy-3.10 passed.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

---
Status: gesendet 2026-10-01 nachts als https://github.com/1CatAI/1Cat-vLLM/pull/745 (Zweig pr-dsv4-software-fp8-prehopper, 6f96f5a4, gestapelt auf #744)
