## Purpose

Fixes #739: its first test was fixed on main by 91701504c, this fixes the second. `_init_offload_distributed` (vllm/v1/ple_offload/worker.py) builds the VllmConfig for the PLE offload process's isolated Gloo world as `VllmConfig()` and forces DP1/TP1/PP1 only afterwards. `ParallelConfig` takes the DP variables the process inherits from its GPU worker (`VLLM_DP_SIZE`, `VLLM_DP_RANK`, `VLLM_DP_RANK_LOCAL`), and `VllmConfig.__post_init__` derives the participating devices from them (`apply_runtime_policy_defaults` -> `_participating_cuda_device_ids`). With an inherited local DP rank of 1 and one visible device, that lookup asks for device 1 and fails before DP1 is set:

```
vllm/v1/ple_offload/worker.py:374: in _init_offload_distributed
vllm/config/vllm.py:1425: in __post_init__
vllm/platforms/runtime_defaults.py:259: in apply_runtime_policy_defaults
vllm/platforms/runtime_defaults.py:79: in _any_participating_device_is_capability
...
vllm/platforms/cuda.py:700: IndexError
E   IndexError: list index out of range
```

This PR builds the `ParallelConfig` first, reduces it to DP1/TP1/PP1 with the same assignments as before, and passes it to `VllmConfig(parallel_config=...)`, so `__post_init__` already sees DP1. From the source only, not run: outside the test this would hit the offload process of a DP rank whose GPU worker sees only its own devices (`set_device_control_env_var` restricts the visible devices per DP rank), if it inherits the DP variables as the test assumes.

The regression test, `test_offload_distributed_sets_config_only_for_model_parallel`, set the DP variables with `monkeypatch.setattr(envs, ...)`. `VLLM_DP_RANK_LOCAL` defaults to the current `VLLM_DP_RANK`, and the test patched `VLLM_DP_RANK` first, so monkeypatch recorded 1 as the old value of `VLLM_DP_RANK_LOCAL` and restored it as a module attribute. Every later test in the process then ran as local DP rank 1; with one visible device, `test_offload_world_drops_the_pipeline_layer_partition` failed with the same IndexError (it passes alone). The test now sets the three variables through `set_lazy_env`, which exists for exactly this shadowing.

Not a duplicate: gh pr list --repo 1CatAI/1Cat-vLLM --state all --search with "_init_offload_distributed", "PLE offload DP", "VLLM_DP_RANK_LOCAL", "offload DP1", "test_offload_distributed_sets_config_only_for_model_parallel", "ple_offload worker IndexError" and "739 in:body" finds no PR that changes this construction or this test (#1005 mentions #739 only as not covered). #1214 (ours) repairs three other stale fakes in the same test file and merges with this PR without conflicts. git merge-tree against all 124 open PR heads adds one conflict: #1023 (draft) also edits the `from vllm.config import ...` line of worker.py.

AI assistance was used for this change. I reviewed every line and ran the tests below.

## Test Plan

```bash
CUDA_VISIBLE_DEVICES=<one GPU> pytest tests/v1/worker/test_ple_offload_worker.py
CUDA_VISIBLE_DEVICES=<two GPUs> pytest tests/v1/worker/test_ple_offload_worker.py
pre-commit run --from-ref origin/main --to-ref HEAD
pre-commit run mypy-3.10 --hook-stage manual --files vllm/v1/ple_offload/worker.py tests/v1/worker/test_ple_offload_worker.py
```

## Path coverage

| Path / provider | Format | Key conditions (shape, TP/EP, graph, capability) | Expected selection / fallback | Evidence (source / static / executed / measured) | Unverified scope |
| --- | --- | --- | --- | --- | --- |
| PLE offload process world setup | any PLE table | process inherits DP variables of its GPU worker, one or more visible devices | VllmConfig built from a DP1/TP1/PP1 ParallelConfig | executed: unit test on a Tesla V100 and a Quadro RTX 8000, one and two visible devices | DP > 1 serving with PLE offload (no such setup here) |
| PLE offload process world setup | any PLE table | no inherited DP variables | unchanged: the same DP1/TP1/PP1 values as before | executed: same test file | |

## Acceleration and benchmark contract (required for performance changes)

Not a performance change. Python only; no new switches, fields or environment variables.

## Test Result

- tests/v1/worker/test_ple_offload_worker.py on main 6f3557c97: one visible device (a Tesla V100, and separately a Quadro RTX 8000) 5 failed, 50 passed, 1 skipped; two visible Quadro RTX 8000 3 failed, 52 passed, 1 skipped.
- The same file with this PR: 3 failed, 52 passed, 1 skipped with one V100, with one RTX 8000 and with two V100. The three remaining failures (`test_ple_offload_uses_event_per_inflight_mrv2_batch`, `test_ple_offload_requires_ple_layers[ple_layer_ids0-True]`, `test_ple_offload_runner_groups_registrations_by_dp_rank`) are the stale fakes #1214 repairs; with the test part of #1214 applied as well: 55 passed, 1 skipped with one V100 and with one RTX 8000.
- Both parts are needed (the regression test and `test_offload_world_drops_the_pipeline_layer_partition`, one visible V100): with only the runtime change, the second test fails after the first (1 failed, 1 passed); with only the test change, the regression test fails with the IndexError above (1 failed, 1 passed).
- pre-commit run --from-ref origin/main --to-ref HEAD and mypy-3.10: all hooks pass.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01YaG8VjXn5eN3Aq7eZrA3dp
