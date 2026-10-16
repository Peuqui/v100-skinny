## Purpose

On SM70, `CudaCommunicator` provisions `CustomAllreduce` IPC resources for the compact greedy top1 collective (and the other SM70 compact collectives) even with `--disable-custom-all-reduce`; the code comment says this is deliberate. The only guard left is the P2P check in `CustomAllreduce.__init__`, and with the default `VLLM_SKIP_P2P_CHECK=1` that check trusts the driver's `cudaDeviceCanAccessPeer` report. On hosts where the driver reports peer access that vLLM's own IPC write test rejects, the compact collective never completes and the engine wedges, although the user disabled custom all-reduce.

We hit this with two Tesla V100-PCIE-32GB on a PCIe host (`nvidia-smi topo -m`: PHB; NCCL's P2P transport also hangs there, so we run `NCCL_P2P_DISABLE=1`). For this pair `torch.cuda.can_device_access_peer` returns True in both directions, while vLLM's IPC write test `can_actually_p2p([0, 1], [1, 0])` returns `[False, False]`; our Quadro RTX 8000 pair shows the same.

`CudaCommunicator` passes `verify_p2p=not use_custom_allreduce` to `CustomAllreduce`. With it, the check runs the real CUDA IPC write test from this rank's device to the devices of its group peers and agrees on the result within the group. When any rank fails, `CustomAllreduce` stays disabled as for any other P2P failure, and `custom_top1_argmax` returns None, so `LogitsProcessor` takes its existing all-gather. Engines that keep custom all-reduce enabled, and ROCm, are unchanged. The launch of the test processes moves into `run_p2p_access_test`, which `gpu_p2p_access_check` now shares.

The first version of this PR (first commit) routed the forced check through `gpu_p2p_access_check`, which builds its cache on world local rank 0 and then waits on a world-group barrier. With pipeline parallelism over mixed cards only the SM70 TP group provisions the compact collectives and reaches that barrier, so Qwen3.8-Flash-Next TP2xPP2 (Quadro RTX 8000 pair on stage 0, Tesla V100 pair on stage 1) hung right after communicator setup. It also passed the group rank as a device index. The second commit replaces that with the group-local check above; the PR is ready for review again.

Not a duplicate: gh search issues (issues and PRs, any state) for "VLLM_SKIP_P2P_CHECK", "P2P check custom allreduce", "disable-custom-all-reduce top1", "top1 argmax hang", "P2P PCIe hang", "can_device_access_peer", "gpu_p2p_access_check" finds nothing besides this PR; among the open PRs, #1208 concerns IPC export of graph inputs and #1216 is a separate host-memory all-reduce backend.

AI assistance was used for this change. I reviewed every line and ran the tests and measurements below.

## Test Plan

```bash
pytest tests/distributed/test_custom_all_reduce_dispatch.py
pre-commit run --from-ref origin/main --to-ref HEAD
pre-commit run mypy-3.10 --hook-stage manual --from-ref origin/main --to-ref HEAD
```

The new tests check that the forced check tests the group's devices rather than its rank numbers, agrees within the group without reaching `gpu_p2p_access_check`, and fails on every rank when one rank's test fails.

Serving on our fork (main 6f3557c97 plus our commits plus this PR; custom_all_reduce.py and all_reduce_utils.py are identical to main plus this PR, cuda_communicator.py differs only by the host-memory all-reduce of #1216, outside the provisioning block), `--disable-custom-all-reduce`, `NCCL_P2P_DISABLE=1`:
- Qwen3.8-27B NVFP4 + MTP (k=3), TP2 on the two V100-PCIE, V1 runner, no `VLLM_SM70_TOP1_CUSTOM_AR` override.
- Qwen3.8-Flash-Next-180B NVFP4 + MTP (k=4), TP2xPP2: Quadro RTX 8000 pair on stage 0, Tesla V100 pair on stage 1.
Per boot: three greedy reference prompts, four 29k-token prompts, short prose, code and a 1.9k-token edit task, eight quality prompts.

## Path coverage

| Path / provider | Format | Key conditions (shape, TP/EP, graph, capability) | Expected selection / fallback | Evidence (source / static / executed / measured) | Unverified scope |
| --- | --- | --- | --- | --- | --- |
| SM70 compact collectives (top1 and related) | any | custom all-reduce disabled, P2P reported by the driver but real IPC test fails | CustomAllreduce disabled on every rank of the group, top1 falls back to all-gather | executed + measured: V100-PCIE pair serving, unit tests | none |
| Same, pipeline over mixed cards | any | only the SM70 TP group provisions the collectives | that group checks and agrees alone; other stages are not involved | executed + measured: Flash-Next TP2xPP2 serving, unit test | more than two stages |
| Same | any | custom all-reduce disabled, real IPC test passes | provisioned, as with `VLLM_SKIP_P2P_CHECK=0` | static: only the check changes | NVLink hosts not run here |
| Custom all-reduce enabled | any | user did not disable custom all-reduce | driver report as before (`VLLM_SKIP_P2P_CHECK`) | static: unforced path unchanged | none |

## Test Result

- Before, V1 runner (the default for this model): the engine starts, the first greedy request never completes, the log repeats `No available shared memory broadcast block found in 60 seconds`, and after 600 s `TimeoutError: RPC call to sample_tokens timed out` ends the engine. py-spy shows both workers in `_count_contiguous_spec_tokens` (from `sample_tokens` → `_update_states_after_model_execute`) with both GPUs at 100 % utilization, unchanged 30 s later. The hang stays with every TP all-reduce on PyNCCL. With `VLLM_USE_V2_MODEL_RUNNER=1` the engine already wedges during startup with the same repeating log line and never reports healthy. `VLLM_SM70_TOP1_CUSTOM_AR=0` removes the hang in both runners.
- With this PR, 27B on the V100 pair: both ranks log `Custom allreduce is disabled because your platform lacks GPU P2P capability or P2P test failed`, no hang, greedy output identical to the Quadro RTX 8000 references for 3 of 3 prompts. Throughput (one boot) prose 55.3, code 85.8, edit 84.4-84.5 tok/s; time to first token for the 29k-token prompts 38.2-38.3 s. The first version measured 55.0-55.1, 85.3 and 84.2-84.3 tok/s, `VLLM_SM70_TOP1_CUSTOM_AR=0` (two boots) 53.6-55.3, 83.8-85.5 and 81.6-84.2 tok/s.
- With this PR, Flash-Next TP2xPP2 (the case the first version hung on): the engine starts, the two V100 ranks (stage 1) log the same warning, the Quadro RTX 8000 ranks (stage 0) never enter the check. Greedy output identical to our reference for 3 of 3 prompts; 29k prefill 18.7-18.8 s, edit 109.4-109.5 tok/s, against 18.7 s and 109.3-109.7 tok/s for the same entry without this PR.
- The quality prompts of both boots, read by hand, are correct.
- tests/distributed/test_custom_all_reduce_dispatch.py on main 6f3557c97 with this PR: 46 passed with a Tesla V100 and with a Quadro RTX 8000 as device 0.
- pre-commit run --from-ref origin/main --to-ref HEAD and mypy-3.10: all hooks pass.
