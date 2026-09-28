Title: [Perf][SM70] Take the DeepSeek-V4 cuBLAS indexer decode under speculative decoding

## Purpose

The cuBLAS decode route in `sm70/indexer.py` (`VLLM_SM70_INDEXER_DECODE_CUBLAS`,
off by default) only accepts a single block-table row. The indexer metadata
builder flattens a uniform speculative decode with more than two verifier
tokens into one block-table row and one length per token, so with DSpark or
MTP (K > 1) the route never runs, even with the switch on, and the paged
Triton kernel scores every row.

- `sm70_indexer_decode_logits` takes `table_rows_per_request`; the caller in
  `sparse_attn_indexer` derives it from `per_req_decode_lens`. The route scores
  each request's rows over one shared paged gather, for one request or several
  with more than one row each.
- The route no longer requires a contiguous cache: the serving cache pads its
  blocks, the kernels read through `stride(0)`, and the per-block strides are
  still checked.
- The requirements are listed by name, and the chosen path is logged once per
  device ("cuBLAS route" or which requirement the call misses). This is how we
  found why the switch had no effect.

The default stays off. Under CUDA graphs the route always scores the full
graph width, so its cost does not grow with the context, while the paged
kernel's does; below roughly 18k tokens the paged kernel is faster.

**Why this is not a duplicate.** The route was added in #274; no open PR
touches `sm70/indexer.py` or the decode call in `sparse_attn_indexer.py`
(`gh pr list --state open --search` for "indexer cublas", "sm70 indexer
decode", "indexer spec decode", "sparse_attn_indexer").

AI assistance was used for this change. Every line was reviewed and the tests
below were run by the submitter.

## Test Plan

```bash
pytest tests/kernels/test_deepseek_v4_sm70_indexer.py tests/models/test_deepseek_v4_sm70_routes.py
# counter-check: sm70/indexer.py reverted to main, test kept
pytest tests/kernels/test_deepseek_v4_sm70_indexer.py -k cublas
```

End to end: DeepSeek-V4-Flash (DeepSeek's MXFP4 checkpoint) with DSpark (6
verifier tokens per step), pipeline parallel over 3x V100 + 2x Quadro RTX 8000,
greedy, decode step time with the switch on and off.

## Test Result

On this branch (main 357d07bc; extensions from a build of db292f9a, the
indexer is Triton and torch only), Tesla V100: 25 passed. With main's
`sm70/indexer.py` and the new test kept, the six cuBLAS cases fail.
pre-commit and mypy-3.10 clean.

The new test cases cover the native, flattened and two-request layouts, each
with padded and unpadded cache blocks, and assert that the cuBLAS route ran;
with main's `sm70/indexer.py` all six fail.

End to end, decode step (ms), greedy output identical with and without:

| context | switch off (paged kernel) | switch on (this PR) |
|---|---:|---:|
| short (26 tokens) | 81-85 | 90-94 |
| 18k | 91 | 90-91 |
| 62k | 116 | 90-91 |

Measured on our fork, where the same route also runs on the Turing stages
(this PR keeps it to Volta); the cuBLAS route is what differs between the
two columns.
