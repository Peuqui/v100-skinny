## Purpose

Two test fixes, one commit each. No runtime code changes.

1. test_sm70_v2_mtp_profile_reports_from_last_pp_stage in tests/v1/worker/test_gpu_model_runner_v2.py fails on main (6db9c816c) in two of its three cases, pp2-last-stage-tp1 and pp2-first-stage, with "assert True is False". 85c71e9bf ([Perf] Reduce host transfers in Flash-Next decode verification) made an enabled runner_v2 step profiler report from every rank: create_step_profiler in vllm/v1/spec_decode/profiling.py passes report_rank=(lambda: True) for runner_v2 so that TP stragglers stay visible, and test_v2_reports_nonzero_tp_rank_timings in tests/v1/worker/test_step_profiler.py covers that. The stage gate is part of the enabled condition (is_last_pp_rank), which test_sm70_v2_mtp_profile_gate covers (_profile(last=False) is disabled). The #414 regression test still expected no report from TP rank 1 and from a first stage. It now asserts that an enabled V2 profiler reports whatever the global first rank, the PP rank seen by parallel_state and the TP rank are, and it is renamed to test_sm70_v2_mtp_profile_reports_on_every_rank.

2. Two scheduler tests build a pipeline_parallel_size=2 config for scheduler bookkeeping only and fail on main when one GPU is visible. test_async_scheduling_pp_allows_rescheduling_with_output_placeholders in tests/v1/core/test_scheduler.py fails the world size check in ParallelConfig. test_v2_pp_spec_decode_respects_pipeline_cadence in tests/v1/core/test_async_scheduler.py fakes device_count() = 2; the runtime policy defaults then probe the capability of every participating device, local device 1 included, and device_id_to_physical_device_id raises IndexError because CUDA_VISIBLE_DEVICES lists one GPU. That IndexError is expected behavior (test_nvml_platform_resolves_worker_uuid_order in tests/video/test_h3_gpu_visibility.py asserts it), so the fix stays in the tests. A new helper, pretend_gpu_count in tests/v1/core/utils.py, patches device_count and is_device_capability on the platform instance, like _patch_visible_devices in tests/config/test_sm70_gates_any_visible_device.py, and answers every capability probe like the first GPU. Both tests use it.

Not a duplicate. On 2026-11-03 and 2026-11-04 I searched the issues and PRs of this repository, open and closed, for the three test names, and the open PRs for "profile report TP rank" and "one visible GPU". No open PR changes vllm/v1/spec_decode/profiling.py or tests/v1/core/utils.py. The open PRs that change tests/v1/worker/test_gpu_model_runner_v2.py (#1279, #1280, #1282, #1286, #1347, #1348, all mine) do not change the profile test. The open PRs that change tests/v1/core/test_async_scheduler.py or tests/v1/core/test_scheduler.py (#1260, #1276, #1304) change other tests there; my #1276 lists the two scheduler failures in its description but does not fix them. One more test in these files fails on main for a different reason: test_abort_request_when_structured_output_fsm_cannot_advance raises AttributeError: 'AsyncScheduler' object has no attribute 'mixed_prefill_budget'. #1260 adds the missing scheduler.mixed_prefill_budget = Mock() to that test, so this PR leaves it alone. #1260 also adds five tests in tests/v1/core that fake the GPU count the same way (device_count() = 7); they could use pretend_gpu_count as well.

AI assistance was used for this change. I reviewed every line and ran the tests below.

## Test Plan

On a Tesla V100 (SM70) and on a Quadro RTX 8000 (SM75), each as the only visible device (CUDA_DEVICE_ORDER=PCI_BUS_ID, CUDA_VISIBLE_DEVICES=<one GPU>), on main 6db9c816c and on this branch:
python -m pytest tests/v1/worker/test_gpu_model_runner_v2.py
python -m pytest tests/v1/core/test_async_scheduler.py tests/v1/core/test_scheduler.py
On this branch also:
python -m pytest tests/v1/worker/test_gpu_model_runner_v2.py tests/v1/worker/test_step_profiler.py
python -m pytest tests/v1/core/test_async_scheduler.py::test_v2_pp_spec_decode_respects_pipeline_cadence tests/v1/core/test_scheduler.py::test_async_scheduling_pp_allows_rescheduling_with_output_placeholders, with CUDA_VISIBLE_DEVICES set to the RTX 8000, to the V100, to both, empty (no GPU), and unset (five GPUs)
pre-commit run --files on the changed files

## Path coverage

N/A, test-only change: no runtime path is touched.

## Acceleration and benchmark contract (required for performance changes)

N/A, not a performance change.

## Test Result

tests/v1/worker/test_gpu_model_runner_v2.py: on main 2 failed (pp2-last-stage-tp1, pp2-first-stage), 5 passed on both cards; on this branch 7 passed on both cards. Together with tests/v1/worker/test_step_profiler.py: 20 passed on both cards.

tests/v1/core/test_async_scheduler.py and tests/v1/core/test_scheduler.py: on main 3 failed (the two scheduler tests above and test_abort_request_when_structured_output_fsm_cannot_advance), 118 passed on both cards; on this branch 1 failed (test_abort_request_when_structured_output_fsm_cannot_advance, see #1260), 120 passed on both cards. The two scheduler tests alone: 2 passed with each of the five CUDA_VISIBLE_DEVICES settings.

pre-commit on the changed files: every hook passed or had no files to check.

🤖 Generated with Claude Code (https://claude.com/claude-code)

https://claude.ai/code/session_0172gtUjD6mwMTtmzDaC7unX
