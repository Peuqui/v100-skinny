Title: [Perf][SM70] DeepSeek-V4 sparse MLA as gather + batched matmul (opt-in)

## Purpose

Every head of a token attends to the same selected keys. This PR gathers those
keys once per token and hands the rest to cuBLAS: QK^T and PV as two batched
matmuls, the softmax in float32 with the attention sink as an extra column.
Two opt-in switches in `DeepseekV4SM70SparseImpl`, following the convention of
the split-K switches:

- `VLLM_SM70_DSV4_SPARSE_MLA_BMM` (decode): the gather reads the packed FP8
  paged cache by slot index and dequantizes on the way. Buffers come from the
  workspace manager with static shapes, so the call captures into CUDA graphs.
- `VLLM_SM70_DSV4_SPARSE_MLA_BMM_PREFILL` (prefill): replaces
  `sm70_sparse_attention_gathered` over the FP16 KV workspace. Its buffers join
  the KV workspace request so the manager does not alias them.

Measured on a Tesla V100, both kernels in CUDA graphs, same packed cache and
indices, 512 keys C4 top-k + 128 window (C4), 144 compressed + 128 window
(C128 at 18k), 485 + 128 (C128 at 62k); outputs agree to 3.1e-5.

Decode, per call (us), heads per rank x query tokens:

| case | heads x tokens | paged_fp8 (default) | split-K | split-K QK-D | BMM |
|---|---|---:|---:|---:|---:|
| SWA | 8 x 1 | 657 | 53 | 15 | 38 |
| SWA | 8 x 6 | 607 | 83 | 26 | 50 |
| SWA | 64 x 6 | 1037 | 368 | 136 | 53 |
| C4 | 8 x 1 | 2858 | 79 | 23 | 66 |
| C4 | 8 x 6 | 2988 | 242 | 91 | 131 |
| C4 | 16 x 6 | 3087 | 402 | 171 | 133 |
| C4 | 64 x 1 | 2922 | 269 | 108 | 65 |
| C4 | 64 x 6 | 4625 | 1566 | 628 | 146 |
| C128 at 18k | 8 x 6 | 1239 | 123 | 47 | 77 |
| C128 at 18k | 64 x 6 | 2025 | 790 | 315 | 88 |
| C128 at 62k | 8 x 6 | 2908 | 242 | 90 | 131 |
| C128 at 62k | 64 x 6 | 4505 | 1567 | 627 | 146 |

So for decode BMM is not a general replacement: at 8 heads per rank (TP8) the
split-K QK-D kernel is faster; from about 16 heads per rank with speculative
decoding BMM wins, up to 4.3x at 64 heads x 6 tokens (pipeline parallelism
without tensor parallelism, DSpark). Hence opt-in.

Prefill, per call (ms), key width x heads x query tokens:

| width | heads | tokens | gathered | BMM | speed-up |
|---:|---:|---:|---:|---:|---:|
| 256 | 8 | 8 | 0.301 | 0.173 | 1.74x |
| 256 | 8 | 512 | 2.084 | 1.567 | 1.33x |
| 256 | 64 | 512 | 12.053 | 1.967 | 6.13x |
| 640 | 8 | 8 | 0.656 | 0.171 | 3.83x |
| 640 | 8 | 512 | 4.634 | 3.337 | 1.39x |
| 640 | 16 | 128 | 2.328 | 0.881 | 2.64x |
| 640 | 64 | 512 | 30.060 | 4.448 | 6.76x |

Prefill BMM is faster at every size measured (1.3x to 7.2x, maximum difference
1.2e-4), including 8 heads per rank.

We run both routes in production on DeepSeek-V4-Flash with DSpark, pipeline
parallel over 3x V100 + 2x Quadro RTX 8000 (64 heads per stage, 6 query
tokens): decode step 90 ms at 18k and 62k context. In our fork they sit behind
a different sparse impl, so end-to-end numbers for this exact integration
into `DeepseekV4SM70SparseImpl` are not attached; DeepSeek-V4 on a V100-only
setup does not fit our three V100s.

`needs_software_fp8()` is added to `cache_utils` for the gather (native FP8
from sm89 on). The three existing SM70 checks in that file keep their exact
`(7, 0)` condition; moving them to the helper belongs to a later Turing change.

**Why this is not a duplicate.** No open PR touches
`vllm/models/deepseek_v4/sm70/sparse.py` or adds sparse attention ops under
`common/ops` (`gh pr diff --name-only` over all open PRs; `gh pr list --state
open --search` for "sparse mla", "batched matmul", "sparse attention decode",
"deepseek v4 sparse", "splitk").

AI assistance was used for this change. Every line was reviewed and the tests
below were run by the submitter.

## Test Plan

```bash
pytest tests/kernels/attention/test_dsv4_sm70_sparse_bmm.py \
  tests/kernels/attention/test_dsv4_sparse_prefill_bmm.py \
  tests/models/test_deepseek_v4_sm70_routes.py
```

## Test Result

On this branch (main 357d07bc), Tesla V100: 34 passed. The decode test compares
against `sm70_sparse_attention_paged_fp8` for SWA, C4 and C128, 8 and 64 heads,
1 and 6 tokens, with padded cache blocks, -1 slots and a NaN row past the
length, and captures the call into a CUDA graph. The prefill test compares
against a float64 reference. Two new route tests check the decode dispatch
and its workspace shapes, and that the prefill buffers are requested together
with the KV workspace. pre-commit and mypy-3.10 clean.
