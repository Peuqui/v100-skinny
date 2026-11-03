## Purpose

test_sm70_v2_mtp_profile_reports_from_last_pp_stage in tests/v1/worker/test_gpu_model_runner_v2.py fails on main (6db9c816c) in two of its three cases, pp2-last-stage-tp1 and pp2-first-stage, with "assert True is False". 85c71e9bf ([Perf] Reduce host transfers in Flash-Next decode verification) made an enabled runner_v2 step profiler report from every rank: create_step_profiler in vllm/v1/spec_decode/profiling.py passes report_rank=(lambda: True) for runner_v2 so that TP stragglers stay visible, and test_v2_reports_nonzero_tp_rank_timings in tests/v1/worker/test_step_profiler.py covers that. The stage gate is part of the enabled condition (is_last_pp_rank), which test_sm70_v2_mtp_profile_gate covers (_profile(last=False) is disabled). The #414 regression test still expected no report from TP rank 1 and from a first stage.

This PR changes only that test. It now asserts that an enabled V2 profiler reports whatever the global first rank, the PP rank seen by parallel_state and the TP rank are, and it is renamed to test_sm70_v2_mtp_profile_reports_on_every_rank. No runtime code changes.

Not a duplicate. On 2026-11-03 I searched the issues and PRs of this repository, open and closed, for the test name, and the open PRs for "profile report TP rank". No open PR changes vllm/v1/spec_decode/profiling.py, and the open PRs that change tests/v1/worker/test_gpu_model_runner_v2.py (#1279, #1280, #1282, #1286, #1347, #1348, all mine) do not change this test. One more test fails on main, for a different reason: test_abort_request_when_structured_output_fsm_cannot_advance in tests/v1/core/test_async_scheduler.py raises AttributeError: 'AsyncScheduler' object has no attribute 'mixed_prefill_budget'. #1260 adds the missing scheduler.mixed_prefill_budget = Mock() to that test, so this PR leaves it alone.

AI assistance was used for this change. I reviewed every line and ran the tests below.

## Test Plan

On a Tesla V100 (SM70) and on a Quadro RTX 8000 (SM75), each as device 0 (CUDA_DEVICE_ORDER=PCI_BUS_ID), on main 6db9c816c and on this branch:
python -m pytest tests/v1/worker/test_gpu_model_runner_v2.py
On this branch also:
python -m pytest tests/v1/worker/test_gpu_model_runner_v2.py tests/v1/worker/test_step_profiler.py
pre-commit run --files tests/v1/worker/test_gpu_model_runner_v2.py

## Path coverage

N/A, test-only change: no runtime path is touched.

## Acceleration and benchmark contract (required for performance changes)

N/A, not a performance change.

## Test Result

tests/v1/worker/test_gpu_model_runner_v2.py: on main 2 failed (pp2-last-stage-tp1, pp2-first-stage), 5 passed on both cards; on this branch 7 passed on both cards. Together with tests/v1/worker/test_step_profiler.py: 20 passed on both cards. pre-commit on the changed file: every hook passed or had no files to check.

🤖 Generated with Claude Code (https://claude.com/claude-code)

https://claude.ai/code/session_0172gtUjD6mwMTtmzDaC7unX
