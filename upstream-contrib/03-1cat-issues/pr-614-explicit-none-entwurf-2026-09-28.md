Title: [Bugfix][SM70] Respect an explicit mode=NONE or cudagraph_mode=NONE in the Flash-V100 compile graph policy

## Purpose

Fixes #614. On a V100 the Flash-V100 0.0.3 compile graph policy treats only
`enforce_eager`, `TORCH_COMPILE_DISABLE` and `VLLM_USE_BREAKABLE_CUDAGRAPH`
as the user disabling the compile path (`sm70_compile_disabled_by_user`). A
compilation config with an explicit `mode=NONE` or `cudagraph_mode=NONE` is
overwritten with `mode=VLLM_COMPILE` and `cudagraph_mode=FULL_AND_PIECEWISE`,
with only an INFO line.

As #614 points out, "not set" and "set to NONE" are distinguishable at that
point: both fields stay `None` until `__post_init__` resolves them further
down, and the only paths before `sm70_compile_disabled_by_user` that set them
to `NONE` are the three switches it already lists. An explicit `NONE` there
therefore comes from the user, and now counts as disabling the policy. The
warning names it as a reason. With nothing set, the policy applies as before.

This keeps to the case #614 describes. Other explicit settings (for example
`cudagraph_mode=PIECEWISE`, `FULL_DECODE_ONLY` or `mode=STOCK_TORCH_COMPILE`)
are still replaced by the policy; whether it should step aside for any
explicit value is a design choice I left to you.

**Why this is not a duplicate.** No open PR changes
`sm70_compile_disabled_by_user` or references #614 (`gh pr list --state open
--search` for "614", "cudagraph_mode NONE", "explicit mode none", "0.0.3
compile graph"). #629 touches `vllm/config/vllm.py` elsewhere and currently
conflicts with main on its own. The branch merges cleanly with #725, which
adds another SM70 test to the same test file.

AI assistance was used for this change. Every line was reviewed and the tests
below were run by the submitter.

## Test Plan

```bash
pytest tests/compile/test_config.py -k respects_explicit_none
# counter-check: vllm/config/vllm.py reverted to main, test kept
# and the upstream cases that expect NONE, each in its own process:
pytest "tests/compile/test_config.py::test_use_cudagraphs[NONE-0]"
pytest tests/compile/test_config.py::test_no_compilation
```

## Test Result

On this branch (main 357d07bc), Tesla V100: the new test passes for
`cudagraph_mode=NONE`, for `mode=NONE` with `cudagraph_mode=FULL` (the
configuration from #614), and for an empty config, where the policy still
yields `VLLM_COMPILE` / `FULL_AND_PIECEWISE`. With main's `vllm/config/vllm.py`
the two explicit cases fail and the empty one passes.

Of the `tests/compile/test_config.py` cases that fail on a V100 on main (13
when each runs in its own process), four pass with this change:
`test_use_cudagraphs[NONE-0]`, `test_no_compilation` and the two
`test_cudagraph_sizes_post_init` cases with `cudagraph_mode=NONE`. The others
set a different explicit mode or capture sizes (see above).
pre-commit and mypy-3.10 clean.
