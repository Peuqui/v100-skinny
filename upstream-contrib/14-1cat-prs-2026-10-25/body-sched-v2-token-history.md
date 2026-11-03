## Purpose

With the V2 model runner and pipeline parallelism a request is scheduled only every pp_size-th step (AsyncScheduler sets next_decode_eligible_step = current_step + pp_size). Scheduler._make_cached_request_data therefore never finds the request in prev_step_scheduled_req_ids and copies its complete token list into CachedRequestData.all_token_ids on every decode step. That list is serialized to every worker.

The V2 runner never reads it: update_requests uses only req_ids, num_computed_tokens and new_block_ids, the tokens stay in req_states on the GPU, and a resumed request is sent as a new request (the use_v2_model_runner branch in schedule()). The only reader of CachedRequestData.all_token_ids in the tree is the V1 runner (vllm/v1/worker/gpu_model_runner.py, resumed_token_ids = req_data.all_token_ids[req_id]), which rebuilds a request that fell out of its persistent batch.

This PR skips the copy when the V2 runner is in use. The V1 runner still gets the list.

The cost grows with the context. Engine core trace (VLLM_SM70_ASYNC_CPU_TRACE=1 with VLLM_SM70_ASYNC_CPU_TRACE_EVERY=1, one line per loop iteration) on DeepSeek-V4-Flash with PP=5, median over the decode steps of one request, in ms for schedule() and for the submit of execute_model: short context 0.15 and 0.10; about 178k tokens 0.69 and 2.42; about 287k tokens 1.09 and 4.09. With this change: about 178k tokens 0.25 and 0.10; about 287k tokens 0.31 and 0.10.

Not a duplicate: I searched the open PRs of this repository for all_token_ids and for scheduler token history under pipeline parallelism (only #1217 appears, which does not touch scheduler.py). Of the 70 open PRs, four change vllm/v1/core/sched/scheduler.py or tests/v1/core/test_async_scheduler.py (#1260, #943, #903, #241); none of them changes _make_cached_request_data or its all_token_ids copy. The same keyword searches in the open PRs of vllm-project/vllm show nothing related.

AI assistance was used for this change. I reviewed every line and ran the tests and measurements below.

## Test Plan

python -m pytest tests/v1/core/test_async_scheduler.py tests/v1/core/test_scheduler.py tests/v1/core/test_engine_core_structured_drafts.py -q
pre-commit run --from-ref origin/main --to-ref HEAD

New test: test_cached_request_history_is_only_sent_to_the_runner_that_reads_it. A request that skipped a step is passed to _make_cached_request_data; with the V2 runner its history must not be in all_token_ids, with the V1 runner it must.

## Path coverage

| Path / provider | Format | Key conditions (shape, TP/EP, graph, capability) | Expected selection / fallback | Evidence (source / static / executed / measured) | Unverified scope |
| --- | --- | --- | --- | --- | --- |
| Scheduler._make_cached_request_data, V2 model runner | all (scheduler only) | async scheduling, request not scheduled in the previous step (every decode step under PP) | no token history in CachedRequestData.all_token_ids | source: only gpu_model_runner.py reads the field; executed: unit test; measured: DeepSeek-V4-Flash and GLM-5.3-Flash with PP=5 | PP=1 and TP-only not measured; end-to-end runs used a fork build, not 6a143197e |
| Scheduler._make_cached_request_data, V1 model runner | all | same | unchanged: history is sent | executed: unit test | no end-to-end run with the V1 runner |

## Acceleration and benchmark contract (required for performance changes)

- Default enabled or opt-in: default enabled, no switch. It takes effect with the V2 model runner (the benchmark set VLLM_USE_V2_MODEL_RUNNER=1) for a request that is skipped between two steps, which under pipeline parallelism is every decode step.
- Required CLI options and environment switches: none for the change. Benchmark entry: --pipeline-parallel-size 5 --kv-cache-dtype fp8 --max-num-seqs 1 --max-num-batched-tokens 1024, speculative method dspark with 5 speculative tokens, VLLM_USE_V2_MODEL_RUNNER=1; the trace runs also set VLLM_SM70_ASYNC_CPU_TRACE=1 and VLLM_SM70_ASYNC_CPU_TRACE_EVERY=1.
- KV cache dtype used for the benchmark: fp8 (DeepSeek-V4-Flash), fp8_e4m3 (GLM-5.3-Flash).
- Wheel SHA or source commit: fork build, reference 1addd22db, changed f4faec141 (1addd22db plus this commit), tag verified-2026-10-25-dsv4-sched in github.com/Peuqui/1Cat-vLLM; GLM reference tag verified-2026-10-25-glm53-meta, changed tag verified-2026-10-25-glm53-sched. Not measured on 6a143197e.
- PYTHONPATH, source overlays, or external native libraries used: the fork source tree and a patched Triton 3.6 build (sm_70/sm_75 tensor cores) on PYTHONPATH; native libraries built from the fork, identical in the reference and the changed run. None of this is part of this PR.
- User-entry route-hit, speed, and output-quality evidence: OpenAI API server entry, A/B/A/B below; the eight greedy answers of the changed runs are identical to those of the reference runs.
- Existing kernel/backend registry and capability-rejection reasons: not touched (scheduler only).
- Per-engine configuration and deprecated-variable compatibility: no new or changed setting.
- New/changed environment metadata and generated reference check: none.
- Route snapshot command and intentional changed rows: none (no route changes).
- SM70 registered variables / unregistered reads / model-parameter restrictions, before -> after: no change.

## Test Result

DeepSeek-V4-Flash, PP=5 on 3 Tesla V100 + 2 Quadro RTX 8000, A/B/A/B through the same server entry with identical compile caches, ms per engine step (lower is better), two reference runs then two changed runs:
- 32,668-token prompt: reference 60.1 and 60.8; changed 60.3 and 60.5.
- 178,480-token prompt: reference 79.1 and 79.6; changed 72.5 and 73.0.
- 287,039-token prompt: reference 85.0 and 86.5; changed 72.9 and 73.0.
- Prefill of an 8,921-token prompt is 2.28 s to 2.29 s and short decode 54.74 to 54.76 tok/s in all four runs.

GLM-5.3-Flash, PP=5 with DFlash2, one run each (not A/B/A/B): at 516,196 tokens 114.5 ms per step before and 96.7 ms after; prefill of the same prompt 251.93 s before and 251.21 s after.

Tests, with this PR: 124 passed, 1 failed. The failure is test_async_scheduler.py::test_abort_request_when_structured_output_fsm_cannot_advance (AttributeError: 'AsyncScheduler' object has no attribute 'mixed_prefill_budget'). It fails identically on origin/main 6a143197e (checked with scheduler.py and the test file at that commit); the fixture lacks the attribute since 1cabc7ce9, and #1260 adds the missing line to it. With that one line added locally, all 125 passed.

The new test without the scheduler change: the V2 case fails (the history is in all_token_ids), the V1 case passes. With the change both pass.

pre-commit run --from-ref origin/main --to-ref HEAD: ruff check, ruff format, typos, mypy-3.10 and the other hooks that apply pass. check-layering fails with "vllm/model_executor/layers/fused_moe/runner/moe_runner.py: platform coupling 37 -> 39"; this PR does not touch that file, its last change on main is b32072ac9 (#1254). The CI job pre-commit runs on all files and fails on this PR with the same hooks as on main (ruff-check, mypy-3.10 to mypy-3.13, check-layering; the latest main run, #1259, is run 40009610580), for findings in files this PR does not touch, for example vllm/model_executor/layers/quantization/gguf_turbomind_moe.py:609 (undefined name tp_slice_packed) and vllm/v1/kv_cache_interface.py:156-158 (mypy attr-defined).

🤖 Generated with Claude Code (https://claude.com/claude-code)

https://claude.ai/code/session_011k7FsBAcZ8Cw91wisiZxMf
