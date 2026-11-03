## Purpose

Under pipeline parallelism, Worker.execute_model waited for every pending PP send of the previous step before it ran the next one. On NCCL, handle.wait() makes the compute stream wait for the transfer, so the next prefill chunk could not start before the previous chunk's hidden states had reached the next stage. In an nsys profile of a DeepSeek-V4-Flash PP5 prefill (three Tesla V100 and two Quadro RTX 8000, 1020-token chunks, 150k-token prompt) every stage spent 4.1-5.2 s of 35.5 s in NCCL kernels and none of that time beside compute kernels.

An eager step (more tokens than the largest captured CUDA graph) sends fresh tensors that the send itself holds and that no later forward overwrites. Its send now overlaps the next step and is awaited only before the next send (_wait_pp_sends). Graph-sized steps, whose output buffers the next replay overwrites, and the SM70 static B1 hidden-state transfer keep waiting before the next step, as before (_pp_send_overlaps). Nothing changes without pipeline parallelism.

Not a duplicate: I searched the open PRs for "pp send", _pp_send_work, "PP send overlap" and "pipeline send"; the hits match other words only. Open PRs that change vllm/v1/worker/gpu_worker.py: #1260, #1031 and #1243 (stacked on #1031) add further static send paths in the block where this PR decides the overlap, so they conflict with it there in text. A send into a persistent buffer of the next stage must keep waiting; the rule here gives that to every graph-sized step, and if such a path can also serve an eager step, it should pass as static_transfer when the two are merged. #903 and #1245 also change the file and merge without a new conflict.

AI assistance was used for this change. I reviewed every line and ran the tests and measurements below.

## Test Plan

Native build of main with this branch in place (TORCH_CUDA_ARCH_LIST=7.0), then one pytest process per file:
python -m pytest tests/v1/worker/test_gpu_worker_static_pp.py tests/v1/worker/test_gpu_worker_allocator.py tests/v1/worker/test_gpu_worker_memory_profile.py tests/v1/worker/test_gpu_worker_profile_failure.py

New tests in tests/v1/worker/test_gpu_worker_static_pp.py: only an eager step's send overlaps (a step of 9 tokens with a largest graph of 8 overlaps, 8 tokens or the static transfer do not), and _wait_pp_sends waits for every handle and clears the list.

pre-commit run --from-ref origin/main --to-ref HEAD; pre-commit run mypy-3.10 --hook-stage manual --all-files

Measurement: A/B/A/B on a DeepSeek-V4 cut of the checkpoint (8 layers, PP2 over a Tesla V100 and a Quadro RTX 8000, 1024-token chunks), one uncounted boot first, then old, new, old, new; prefill of an 8k, a 32k and a 150k-token prompt; greedy tokens after a short and a 4k-token prompt compared between all runs.

## Path coverage

| Path / provider | Format | Key conditions (shape, TP/EP, graph, capability) | Expected selection / fallback | Evidence (source / static / executed / measured) | Unverified scope |
| --- | --- | --- | --- | --- | --- |
| Worker.execute_model, PP send of a non-last stage | any | step with more tokens than max_cudagraph_capture_size, no SM70 static B1 transfer | the send overlaps the next step, awaited before the next send | executed: test_gpu_worker_static_pp.py; measured: mini-checkpoint A/B, fork PP5 production since 2026-10-23 | non-NCCL PP backends |
| same | any | graph-sized step or static B1 transfer | wait before the next step (unchanged) | executed: test_gpu_worker_static_pp.py | none |

## Acceleration and benchmark contract (required for performance changes)

- Default enabled or opt-in: enabled (pipeline parallelism only).
- Required CLI options and environment switches: none.
- KV cache dtype used for the benchmark: fp8 (the DeepSeek-V4 cut, as in our deployment).
- Wheel SHA or source commit: fork build of github.com/Peuqui/1Cat-vLLM (fork-main 993f69f5d with and without this commit, which our production has carried since 2026-10-23); this branch is the same change on main 6aea01fc6 (the tests ran with it on 9a33cee2a; the three later main commits #1290-#1292 touch neither the changed files nor these tests).
- PYTHONPATH, source overlays, or external native libraries used (write "none" if absent): the fork tree on PYTHONPATH in both arms of the measurement; none for the tests.
- User-entry route-hit, speed, and output-quality evidence: see Test Result.
- Existing kernel/backend registry and capability-rejection reasons: not touched.
- Per-engine configuration and deprecated-variable compatibility: no setting added.
- New/changed environment metadata and generated reference check: none.
- Route snapshot command and intentional changed rows (write "none" for refactors): none.
- SM70 registered variables / unregistered reads / model-parameter restrictions, before -> after: unchanged.

## Test Result

Prefill time in s, lower is better (old a / old b -> new a / new b):
- 8k tokens: 0.839 / 0.837 -> 0.817 / 0.818 (-2.4 %)
- 32k tokens: 3.154 / 3.150 -> 3.028 / 3.030 (-3.9 %)
- 150k tokens: 15.585 / 15.607 -> 15.439 / 15.467 (-0.9 %)

Greedy tokens after a short and a 4k-token prompt are identical in all runs; the logprobs after the 4k-token prompt differ between the arms by at most 0.011, less than between two boots of the same build (0.015 and 0.016). Most of the NCCL time in the PP5 profile is therefore waiting on the neighbouring stage, not serialized transfer.

Tests on this branch, Quadro RTX 8000 and Tesla V100: test_gpu_worker_static_pp.py 6 passed (two of them new), test_gpu_worker_allocator.py 10 passed, test_gpu_worker_profile_failure.py 4 passed; test_gpu_worker_memory_profile.py 7 failed and 2 passed on both cards, the same test IDs on main 9a33cee2a with the same build (fakes without speculative_config, and a 32 MiB tolerance on a 64 MiB difference).

pre-commit run --from-ref origin/main --to-ref HEAD: every hook passes except check-layering, which reports vllm/model_executor/layers/fused_moe/runner/moe_runner.py (platform coupling 37 -> 39), as on main 9a33cee2a; this PR does not touch that file. pre-commit run mypy-3.10 --hook-stage manual --all-files: the only errors are three arg-type errors in vllm/v1/attention/backends/flash_v100/runtime.py, which mypy reports on main as well in our environment and 1Cat's CI does not report; this PR does not touch that file.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
