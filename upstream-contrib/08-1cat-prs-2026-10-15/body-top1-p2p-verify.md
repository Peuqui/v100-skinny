## Purpose

On SM70, `CudaCommunicator` provisions `CustomAllreduce` IPC resources for the compact greedy top1 collective (and the other SM70 compact collectives) even with `--disable-custom-all-reduce`; the code comment says this is deliberate. The only guard left is the P2P check in `CustomAllreduce.__init__`, and with the default `VLLM_SKIP_P2P_CHECK=1` that check trusts the driver's `cudaDeviceCanAccessPeer` report. On hosts where the driver reports peer access that vLLM's own IPC write test rejects, the compact collective never completes and the engine wedges, although the user disabled custom all-reduce.

We hit this with two Tesla V100-PCIE-32GB on a PCIe host (`nvidia-smi topo -m`: PHB; NCCL's P2P transport also hangs there, so we run `NCCL_P2P_DISABLE=1`). For this pair `torch.cuda.can_device_access_peer` returns True in both directions, while vLLM's own IPC write test `can_actually_p2p([0, 1], [1, 0])` returns `[False, False]`.

This PR passes `verify_p2p=not use_custom_allreduce` from `CudaCommunicator` to `CustomAllreduce`, and `_can_p2p(..., verify=True)` then runs the real IPC write test (`gpu_p2p_access_check`, cached per `CUDA_VISIBLE_DEVICES` under `VLLM_CACHE_ROOT`) instead of trusting the driver report. When the test fails, `CustomAllreduce` stays disabled as for any other P2P failure, and `custom_top1_argmax` returns None, so `LogitsProcessor` takes its existing all-gather. With custom all-reduce disabled, the compact collectives thus follow the same real test that `VLLM_SKIP_P2P_CHECK=0` selects; engines that keep custom all-reduce enabled and ROCm are unchanged.

Not a duplicate: gh search issues (issues and PRs, any state) for "VLLM_SKIP_P2P_CHECK", "P2P check custom allreduce", "disable-custom-all-reduce top1", "top1 argmax hang", "P2P PCIe hang", "can_device_access_peer", "gpu_p2p_access_check" finds nothing; among the open PRs, #1208 concerns IPC export of graph inputs and #1216 is a separate host-memory all-reduce backend.

AI assistance was used for this change. I reviewed every line and ran the tests and measurements below.

## Test Plan

```bash
pytest tests/distributed/test_custom_all_reduce_dispatch.py
pre-commit run --from-ref origin/main --to-ref HEAD
```

The new test sets `VLLM_SKIP_P2P_CHECK=1`, a driver report of True and a failing real test, and checks that `_can_p2p` follows the driver without `verify` and the real test with it.

Serving on our fork (main 6f3557c97 plus our fork's commits and this change; custom_all_reduce.py is identical to main, and cuda_communicator.py differs only by the host-memory all-reduce of #1216, outside the provisioning block): Qwen3.8-27B NVFP4 + MTP (k=3), TP2 on the two V100-PCIE, `--disable-custom-all-reduce`, `NCCL_P2P_DISABLE=1`, greedy reference prompts recorded on a Quadro RTX 8000 pair, then 29k-token prompts, short prose, code and a 1.9k-token edit task.

## Path coverage

| Path / provider | Format | Key conditions (shape, TP/EP, graph, capability) | Expected selection / fallback | Evidence (source / static / executed / measured) | Unverified scope |
| --- | --- | --- | --- | --- | --- |
| SM70 compact collectives (top1 and related) | any | custom all-reduce disabled, P2P reported by the driver but real IPC test fails | CustomAllreduce disabled, top1 falls back to all-gather | executed + measured: V100-PCIE pair serving, unit test | none |
| Same | any | custom all-reduce disabled, real IPC test passes | provisioned, as with `VLLM_SKIP_P2P_CHECK=0` | static: only the check changes | NVLink hosts not run here |
| Custom all-reduce enabled | any | user did not disable custom all-reduce | driver report as before (`VLLM_SKIP_P2P_CHECK`) | static + unit test (verify=False) | none |

## Test Result

- Before, V1 runner (the default for this model): the engine starts, the first greedy request never completes, the log repeats `No available shared memory broadcast block found in 60 seconds`, and after 600 s `TimeoutError: RPC call to sample_tokens timed out` ends the engine. py-spy shows both workers in `_count_contiguous_spec_tokens` (from `sample_tokens` → `_update_states_after_model_execute`) with both GPUs at 100 % utilization, unchanged 30 s later. The hang stays with every TP all-reduce on PyNCCL. With `VLLM_USE_V2_MODEL_RUNNER=1` the engine already wedges during startup with the same repeating log line and never reports healthy. `VLLM_SM70_TOP1_CUSTOM_AR=0` removes the hang in both runners.
- With this PR and no override: the real P2P test runs, both ranks log `Custom allreduce is disabled because your platform lacks GPU P2P capability or P2P test failed`, no hang, greedy output identical to the RTX 8000 references for 3 of 3 prompts.
- Throughput with this PR (one boot) vs `VLLM_SM70_TOP1_CUSTOM_AR=0` (two boots), V1 runner, tok/s: prose 55.0-55.1 vs 53.6-55.3, code 85.3 vs 83.8-85.5, edit 84.2-84.3 vs 81.6-84.2; time to first token for the 29k-token prompts 38.2-38.5 s vs 38.2-38.3 s.
- tests/distributed/test_custom_all_reduce_dispatch.py on main 6f3557c97 with this change: 45 passed with a Tesla V100 and with a Quadro RTX 8000 as device 0.
- pre-commit run --from-ref origin/main --to-ref HEAD: all hooks pass.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01Ly7LM3m9fa6DRJPYQjnt95
