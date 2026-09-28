Title: [Bugfix][DeepSeek-V4] Size the ROCm SWA ragged copy from the actual row width

## Purpose

`DeepseekV4ROCMAiterSparseSWAMetadataBuilder` copies the ragged decode
indices into a persistent graph buffer. Both the buffer
(`max_num_batched_tokens * window_size`) and the copied slice
(`num_decode_tokens * window_size`) assume rows that are `window_size` wide.
`build_ragged_indices_from_dense` returns `num_rows * row_width` entries,
though, and with DSpark the base builder emits the non-causal rows at
`noncausal_index_width` (256 for the 128-token window). The slice is then
half as long as what is copied into it, and every DSpark decode step fails:

```
RuntimeError: The size of tensor a (512) must match the size of tensor b (1024) at non-singleton dimension 0
```

- The buffer is sized from `max(window_size, noncausal_index_width)`.
- The slice is sized from the dense row width.

Either change alone still fails, because the other one keeps the slice at
`num_rows * window_size`. The causal path is unchanged: there the row width
is `window_size`.

The builder is only selected on ROCm. We have no ROCm hardware; we found
this because our fork runs the same builder on CUDA (V100 / Quadro RTX 8000)
with DSpark, and reproduced it on main with the test below, which runs the
real builder on CUDA with the base builder stubbed.

**Why this is not a duplicate.** No open PR touches
`vllm/models/deepseek_v4/amd/rocm.py` (`gh pr diff --name-only` over all open
PRs), and `gh pr list --state open --search` / `gh issue list --state open
--search` for "swa ragged", "rocm swa", "ragged graph buffer",
"noncausal_index_width", "dspark rocm" and "decode_swa_ragged" find nothing
related.

AI assistance was used for this change. Every line was reviewed and the tests
below were run by the submitter.

## Test Plan

```bash
pytest tests/models/test_deepseek_v4_rocm_swa_ragged.py
# counter-check: rocm.py reverted to main, and each half of the change alone
```

## Test Result

On this branch (main 357d07bc), Tesla V100: 2 passed (causal and DSpark
rows; every schedulable token is a decode token, so the buffer is filled to
its size). With main's `rocm.py` the DSpark case fails with the error above;
with only the buffer change or only the slice change it fails the same way.
pre-commit and mypy-3.10 clean.
