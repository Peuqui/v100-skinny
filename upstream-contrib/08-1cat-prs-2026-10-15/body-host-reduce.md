## Purpose

Two-rank tensor-parallel groups whose GPUs cannot reach each other peer to peer (PCIe hosts with P2P disabled or unavailable, KVM guests, mixed boards) fall back to NCCL for every all-reduce, and the custom all-reduce needs P2P/IPC. For the small FP16 all-reduces of single-stream decode, NCCL's latency dominates.

This PR adds a two-rank FP16 all-reduce through pinned host memory that both ranks map. One kernel does the exchange: each block writes its chunk to a host slot, fences, raises a per-block flag, waits for the peer's flag and adds the peer's chunk. The sum is one FP16 rounding and commutative, so both ranks produce the same bits as NCCL's two-rank FP16 sum. There is no CPU work and no stream synchronization, so it is captured in CUDA graphs; slots are double buffered by call parity.

It is placed after the SM70 ring in `CudaCommunicator.all_reduce` and takes FP16 payloads of two-rank groups without peer access up to `kernel_config.host_reduce.max_bytes` (256 KiB); larger payloads and every other case stay on the existing path. `kernel_config.host_reduce.enabled=false` turns it off (field outside the compile hash).

The exchange kernel is adapted from `kernels/skinny_ar.cu` of mzen17/v100-skinny-unify (MIT; the notice is kept in `csrc/host_reduce.cu`), turned into handle-based channels so one process can own more than one two-rank group. Credit: @mzen17 (Co-authored-by on the commit).

Not a duplicate: gh pr list --state all --search "host memory all-reduce", "pinned host", "all-reduce P2P", "skinny_ar", "two-rank" finds no PR for a host-memory collective; #1208 concerns IPC registration of the P2P custom all-reduce and #963 the TP4 push all-reduce.

AI assistance was used for this change. I reviewed every line and ran the tests and measurements below.

## Test Plan

```bash
pytest tests/distributed/test_host_reduce.py
pre-commit run --from-ref origin/main --to-ref HEAD
```

`test_host_reduce_matches_nccl` spawns two ranks with `NCCL_P2P_DISABLE=1` and compares the host all-reduce bit for bit with NCCL.

Serving: Qwen3.8-27B NVFP4 with MTP (k=3), TP2 on two Quadro RTX 8000 that only share a PCIe host bridge (`nvidia-smi topo`: PHB), `NCCL_P2P_DISABLE=1`, `--disable-custom-all-reduce`, CUDA graphs. Same build and compile artifact, the field off (`--kernel-config '{"host_reduce":{"enabled":false}}'`) and on, alternating boots (off, off, on, off, on). Per boot: four 29k-token prompts, then greedy short prose, code and a 1.9k-token edit task, plus three greedy reference prompts.

## Path coverage

| Path / provider | Format | Key conditions (shape, TP/EP, graph, capability) | Expected selection / fallback | Evidence (source / static / executed / measured) | Unverified scope |
| --- | --- | --- | --- | --- | --- |
| Host all-reduce | FP16 | TP2, no peer access, payload <= 256 KiB, eager and CUDA graph | host kernel | executed + measured: test on RTX 8000 pair, V100 pair, mixed V100 + RTX 8000; 27B TP2 serving | groups with P2P (not selected by design) |
| Existing all-reduce chain | any | larger payload, non-FP16, world size != 2, peer access available, field off | unchanged (SM70 ring / custom / NCCL) | static: guard order in cuda_communicator.py; executed: field-off boots | none |

## Acceleration and benchmark contract (required for performance changes)

- Default enabled or opt-in: enabled where admitted (two ranks, no peer access, FP16, <= 256 KiB).
- Required CLI options and environment switches: none; `kernel_config.host_reduce.enabled=false` disables it.
- KV cache dtype used for the benchmark: FP16 (model default for this entry).
- Wheel SHA or source commit: our fork (main 6f3557c97 + our commits), production commit baaa5a687; this PR is the same change on main.
- PYTHONPATH, source overlays, or external native libraries used: none for the production entry; native build of the fork tree.
- User-entry route-hit, speed, and output-quality evidence: see Test Result.
- Existing kernel/backend registry and capability-rejection reasons: admission recorded with the communicator's existing collective selection.
- Per-engine configuration and deprecated-variable compatibility: KernelConfig field only, no environment variable.
- New/changed environment metadata and generated reference check: architecture reference regenerated.
- Route snapshot command and intentional changed rows: none.
- SM70 registered variables / unregistered reads / model-parameter restrictions, before -> after: unchanged.

## Test Result

- tests/distributed/test_host_reduce.py: 1 passed on two Quadro RTX 8000, on two Tesla V100 and on a V100 + RTX 8000 pair (`NCCL_P2P_DISABLE=1`).
- pre-commit run --from-ref origin/main --to-ref HEAD: all hooks pass.
- Qwen3.8-27B NVFP4 TP2, greedy decode step (ms) / throughput (tok/s), off (3 boots) -> on (2 boots): prose 37 / 66.3-66.4 -> 33 / 73.7-73.8, code 37 / 101.9-102.6 -> 33 / 113.8-114.0, edit 39 / 102.8-102.9 -> 35 / 113.8-114.0, i.e. about +11 % decode throughput. Decode step at 29k context 56-57 -> 53 ms. Cold 29k-token prefill unchanged (41.3-41.8 s both).
- Greedy output identical to the reference for 3 of 3 prompts in every boot, with the field off and on.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01Ly7LM3m9fa6DRJPYQjnt95
