Title: [Bugfix][SM70] Keep the SM70 cudagraph size cap inside max_num_batched_tokens

## Purpose

On a V100 the Flash-V100 graph setup in `VllmConfig.__post_init__` picks
its own capture sizes and sets `max_cudagraph_capture_size` to their
maximum. The generic sizing afterwards drops capture sizes above
`max_num_batched_tokens`. Since both values are now set, a cap above the
largest remaining size is treated as contradicting user input, and the
config fails. With speculative decoding (capture sizes up to 64) and
`max_num_batched_tokens=50`:

```
customized max_cudagraph_capture_size(=64) should be consistent with the max value of cudagraph_capture_sizes(=48)
```

The cap is now taken from the capture sizes that fit the token budget
(`_sm70_max_cudagraph_capture_size`), in both SM70 setups: the compile graph
and the decode graph without compile. Sizes above the budget were dropped
before as well, so the captured graphs do not change; only the failing
validation goes away. With the usual budgets (256 and more) nothing changes
at all.

**Why this is not a duplicate.** No open PR changes these lines;
`gh pr list --state open --search` for "max_cudagraph_capture_size",
"cudagraph capture size sm70" and "capture sizes max_num_batched_tokens"
finds nothing related. Several open PRs touch `vllm/config/vllm.py`
elsewhere; this branch merges cleanly with each of them (#646, #717, #702,
#696, #701, #629, #621 checked with git merge-tree).

AI assistance was used for this change. Every line was reviewed and the tests
below were run by the submitter.

## Test Plan

```bash
pytest tests/compile/test_config.py -k capture_cap
pytest tests/v1/core/test_scheduler.py -k no_spec_tokens_scheduled_for_prefill_chunks
# counter-check: vllm/config/vllm.py reverted to main, tests kept
```

## Test Result

On this branch (main 357d07bc), Tesla V100:
`test_sm70_speculative_capture_cap_fits_batched_tokens` passes for
`max_num_batched_tokens` 50 and 2048; with main's `vllm/config/vllm.py` the
50 case fails with the error above. The existing
`test_no_spec_tokens_scheduled_for_prefill_chunks` (budget 50) fails on main
on a V100 with the same error and passes here; on a Quadro RTX 8000 it
passes either way, the SM70 setup does not run there.

`tests/compile/test_config.py` on the V100: 13 failures on this branch
against 14 on main (`test_cudagraph_sizes_post_init` with a budget of 8 is
the one this also fixes); the remaining 13 fail identically on main and are
not touched here. pre-commit and mypy-3.10 clean.
