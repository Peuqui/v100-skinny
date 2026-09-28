Title: [Test] Bring stale core and SM70 tests in line with current main

## Purpose

These tests fail on main 357d07bc for reasons in the tests, not in the code
they cover:

- `tests/v1/core/test_scheduler.py::test_abort_request_when_structured_output_fsm_cannot_advance`:
  the scheduler strips reasoning tokens through
  `structured_output_manager.trim_reasoning_for_advance` before the grammar
  sees them. The Mock returned a Mock, which the grammar rejected
  ("grammar rejected tokens <Mock ...>"). The test now passes the token ids
  through.
- `tests/v1/core/test_spec_token_aliasing.py` (two cases): the scheduler
  output carries `ddtree_payloads_by_req_id` and `scheduled_ddtree_payloads`
  now; the hand-built outputs lacked them.
- `tests/quantization/test_sm70_compressed_tensors_fastpaths.py::test_compressed_tensors_channel_fp8_qpn8_prepares_and_dispatches`:
  passes alone, fails after any test that builds an SM70 `VllmConfig`
  (for example `test_scheduler.py`). That config exports
  `VLLM_SM70_BATCH_GEMM_LAYOUTS=1` for the process, which sends the QPN8
  prepare into the batch-GEMM layout and the `fp8_sm70_prepare` the test
  stubs with `object()` ("'object' object is not callable"). The test covers
  the QPN8 prepare only, so it pins that layout off.
- `tests/kernels/attention/test_sm70_flash_v100_multihead.py`: the built-in
  long and grouped E4M3 cases call SM70-only kernels and fail on Turing (39
  cases, "E4M3 grouped FP32 supports SM70 only"). They are skipped off SM70
  now; the XQA cases keep running there.

`tests/v1/core/test_qwen4_exp_kv_cache.py` fails on main as well; #702 and
#696 both carry the fix, so it is left out here.

**Why this is not a duplicate.** Of the open PRs, none changes
`test_scheduler.py`, `test_spec_token_aliasing.py`,
`test_sm70_compressed_tensors_fastpaths.py` or
`test_sm70_flash_v100_multihead.py` (`gh pr diff --name-only` over all open
PRs). The branch merges cleanly with #623, which edits other lines of the
multihead test.

AI assistance was used for this change. Every line was reviewed and the tests
below were run by the submitter.

## Test Plan

```bash
pytest tests/v1/core/test_scheduler.py tests/v1/core/test_spec_token_aliasing.py \
  tests/quantization/test_sm70_compressed_tensors_fastpaths.py \
  tests/kernels/attention/test_sm70_flash_v100_multihead.py
# counter-check: the four test files reverted to main
```

## Test Result

On this branch, Tesla V100 (two visible for the PP case) and Quadro RTX 8000:
all pass, except on the V100 `test_no_spec_tokens_scheduled_for_prefill_chunks`,
which fails on main for a config reason fixed separately in #725. The
multihead file: 70 passed on the V100, 31 passed and 39 skipped on the RTX
8000 (39 failed there on main). With the test files reverted to main, the
cases listed above fail again. pre-commit and mypy-3.10 clean.
