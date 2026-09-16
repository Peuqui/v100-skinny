# PR-Entwurf 1Cat: PLE-Überlaufkaskade (Feature-Paket) — GESENDET 16.09.2026 als #646 (Go Peuqui)

https://github.com/1CatAI/1Cat-vLLM/pull/646 — Branch auf fork gepusht, Kaskaden-Commit f511e4fb.

Stand 16.09.2026 abends. Worktree `1Cat-vLLM-pr-plecascade`, Branch
`qwen4exp-ple-tier-cascade-pr` auf origin/main 02c87ab8:
- 22d1900b = Kopie von #622 (Abhängigkeit, fällt nach dessen Merge beim Rebase weg)
- Kaskade als ein Commit (mit Design-Dokument), lokal committet, NICHT gepusht
Fork-Stand: `1Cat-vLLM-work` Branch `qwen4exp-ple-tier-cascade` (70d2df4b) plus
uncommittete Angleichungen (Kommentar, Docstring, torch.accelerator, Partition-Fix
mit Test); PLE-Dateien im Fork = PR-Dateien bis auf den #640-Teil (per diff geprüft).
Senden erst nach Zeilen-Review und Go von Peuqui: push auf `fork`, dann
`gh pr create --repo 1CatAI/1Cat-vLLM --base main --head Peuqui:qwen4exp-ple-tier-cascade-pr`.

Titel: [Feature][Qwen4Exp] PLE overflow cascade: device, pinned host, spare GPU, disk

---

## Purpose

Depends on #622 (code base). Running it with `nvidia/Qwen3.8-Flash-Next-NVFP4`
also needs #640, and with MTP under pipeline parallelism #639; see
Dependencies below.

Qwen4Exp's PLE table is large: 47.7 GiB of FP8 rows for Qwen3.8-Flash-Next,
23.8 GiB per tensor-parallel rank at TP2. Today a pre-Ampere deployment has two
places for it. The pinned-host split keeps as much as fits in device memory and
pins the rest in host memory, every rank its own share. The PLE offload worker
holds the whole table in host memory, or maps it from disk with
`VLLM_PLE_DISK_OFFLOAD`. A host with little RAM loses either way: on our rig
(30 GiB RAM) the split pins 12 GiB and leaves about 2.5 GiB available after
startup, a table held in the worker needs more memory than the host has, and
the disk lane reads every row from the checkpoint.

This PR adds an overflow cascade. Each rank fills four tiers, fastest first:

1. device memory of the compute card, up to its measured budget (the same
   computation as the existing auto placement: utilization budget minus
   allocated memory, the KV cache for `max_model_len` and a reserve),
2. a pinned host share,
3. a store card, i.e. the memory of a GPU that does not compute this stage,
   served by the PLE offload worker,
4. the mapped checkpoint on disk, read by the offload worker.

What it gives a deployment:

- Host memory back. On the rig below, available memory after startup goes from
  about 2.5 GiB to 11.8 GiB (2 GiB host share per rank) or 16.0 GiB (no host
  share), with the rest of the table on a spare V100.
- Identical output. Greedy text is identical to the pre-change path in 4 of 4
  probes in every configuration, including the one that reads rows from disk.
- A small price. Decode is 1.7 to 2.5 % slower than the pre-change path;
  prefill is unchanged.
- Larger tables on fewer cards. The disk tier also works without a store card
  (`VLLM_QWEN4EXP_PLE_DISK=1` alone), so a box without a spare GPU can go
  device, host, disk; end to end we measured it only together with a store
  card. Rows are never dropped: a table that does not fit the configured tiers
  fails at startup with a message that names the knobs.
- A fitting table pins nothing. With the cascade on, the host share is used
  only for what the device cannot hold.

Nothing changes without the new variables: the pinned-host split, the
whole-table worker, the disk lane and the hybrid lane keep their behaviour.

## Configuration

| Variable | Meaning |
|---|---|
| `VLLM_QWEN4EXP_PLE_STORE_DEVICE` | Visible CUDA index of the store card. Setting it starts the cascade. |
| `VLLM_QWEN4EXP_PLE_STORE_GIB` | Store budget in GiB, in total, shared equally by the TP ranks. Required with the device. |
| `VLLM_QWEN4EXP_PLE_DISK=1` | Let the remainder be read from the checkpoint. Also starts the cascade on its own (no store card). |
| `VLLM_QWEN4EXP_PLE_HOST_GIB` | Unchanged meaning: pinned host share per rank. |

Example, TP2 with a spare fifth card:

```
VLLM_QWEN4EXP_PLE_HOST_GIB=2 VLLM_QWEN4EXP_PLE_STORE_DEVICE=4 VLLM_QWEN4EXP_PLE_STORE_GIB=6 vllm serve ...
```

The cascade cannot be combined with `VLLM_SM70_QWEN38_HYBRID_PLE` or
`VLLM_PLE_DISK_OFFLOAD`; `VllmConfig` refuses that, and a model without PLE
layers, before any worker starts.

## How it works

- Placement (`common/ple.py`): `plan_ple_placement` returns a four-tier
  `PLEPlacement`. With the cascade, device rows come first up to the measured
  budget, then host, store and disk. Without it the host share comes first and
  the device holds the rest, as before.
- Compute ranks (`nvidia/ple_layer.py`): `Qwen4ExpNGramEmbedding` keeps its
  resident tables under the offload contract (`offload_keeps_local_tables()`,
  a new hook on `PleOffloadLayer`). It gathers device and host rows as today,
  waits for the worker with `ple_offload_wait` inside the CUDA graph, and
  merges the worker's rows by rank-local id before the tensor-parallel
  all-reduce. The worker sends raw FP8 bytes, dequantized with the same op and
  scale as the resident gathers, which is why outputs stay identical.
- Registration: each rank sends a `PLERemotePlacement` (vocabulary range,
  resident rows, store rows) in `PleOffloadRegistration.remote_placements`.
- Worker: the ranks' store and disk segments are disjoint in the global id
  space, so one output buffer serves every rank and each rank takes only its
  own slots. Store segments are copied from the mapped shards to the store card
  once at registration, as raw bytes and without an anonymous host copy; a step
  does `index_select` on that card. The disk tier reuses the disk lane's mmap
  reader, now factored out as `_gather_mapped_rows` and used by both.
- Host share: an explicit `VLLM_QWEN4EXP_PLE_HOST_GIB` is checked once in
  `EngineArgs.create_engine_config`, share times TP size against available
  memory minus `VLLM_QWEN4EXP_PLE_HOST_RESERVE_GIB`. It used to be checked in
  each rank while the sibling ranks were already pinning, so a sibling's share
  could count against a rank's own. The budget and reserve helpers moved to
  `common/ple.py` so that config and ranks use the same code. A derived host
  budget (no `VLLM_QWEN4EXP_PLE_HOST_GIB`) is still capped per rank as before.
- Pipeline parallelism, two fixes the cascade needs: only ranks that own a
  `PleOffloadLayer` build a connector (later stages used to fail in
  `_setup_layers`), and the worker drops an inherited `VLLM_PP_LAYER_PARTITION`,
  which made `get_pp_indices()` refuse its single-stage world while the meta
  model was built.
- `docs/design/qwen4exp_ple_tier_cascade.md` (new): tiers, configuration,
  placement, per-step flow, validation and limits.
- `tests/utils.py`: `set_lazy_env`. `monkeypatch.setattr(envs, ...)` leaves a
  real module attribute behind that hides `envs.__getattr__` from later tests;
  the new tests set the variables through the environment instead.

## Dependencies

- #622: required. The cascade extends the gather that #622 changes, so this
  branch carries #622's commit as its first commit. Please review the second
  commit; the first one disappears on rebase once #622 is merged.
- #640: required for `nvidia/Qwen3.8-Flash-Next-NVFP4`. Without it that
  checkpoint cannot take the pinned-host path on pre-Ampere cards, which the
  cascade builds on:
  `NotImplementedError: Qwen4Exp pinned-host PLE requires FP8 checkpoint storage`.
  Checkpoints that set `ple_embedding_dtype` do not need it.
- #639: required for MTP with that checkpoint under pipeline parallelism, which
  is the configuration measured below. The cascade itself does not depend on it.

## Test Plan

All on this branch, compiled extensions from our fork build, Python 3.12,
torch 2.10.0+cu128, `CUDA_VISIBLE_DEVICES` = one free Tesla V100 plus a second
visible card (some tests set DP=2 and query device 1),
`HF_HUB_OFFLINE=1`:

```
pytest tests/models/qwen4_exp/test_ple.py
pytest tests/v1/worker/test_ple_offload_worker.py
pytest tests/compile/test_sm70_decode_graph.py
pytest tests/v1/worker/test_release_cleanup.py
pytest tests/v1/executor/test_executor.py
pytest tests/models/qwen4_exp/test_weight_loading.py
pytest tests/compile/passes/test_functionalization.py

pre-commit run --files <the 15 changed files>
pre-commit run mypy-3.10 --hook-stage manual --files <the 14 changed Python files>
```

Counter-check for the partition fix: `vllm/v1/ple_offload/worker.py` without
the `VLLM_PP_LAYER_PARTITION` line, new test kept.

End to end, on our fork, whose PLE files equal this branch except for #640's
hunk: 2x Quadro RTX 8000 (first pipeline stage, TP2, holds the table), 2x Tesla
V100 (second stage, TP2), 1x Tesla V100 as store card, all PCIe Gen3 x4 (the
store card behind a USB4 tunnel), 30 GiB host RAM, checkpoint on an NVMe SSD in
a USB enclosure. `nvidia/Qwen3.8-Flash-Next-NVFP4`, MTP k=4, TP2 x PP2, async
scheduling, `--max-model-len 262144`, `--gpu-memory-utilization 0.95`. Probe:
three prompts plus the first one again, greedy, `ignore_eos`, 260 tokens each,
SHA-256 of the text against a reference taken on the pre-change path.
Available memory is `MemAvailable` after startup.

## Test Result

Unit tests, this branch against its parent (main 02c87ab8 + #622):

| File | parent | this branch |
|---|---|---|
| `tests/models/qwen4_exp/test_ple.py` | 60 passed | 85 passed |
| `tests/v1/worker/test_ple_offload_worker.py` | 27 passed | 31 passed |
| `tests/compile/test_sm70_decode_graph.py` | 23 passed | 24 passed |
| `tests/v1/worker/test_release_cleanup.py` | 4 passed | 4 passed |
| `tests/v1/executor/test_executor.py` | 11 passed | 11 passed |
| `tests/models/qwen4_exp/test_weight_loading.py` | 27 passed, 1 failed | 27 passed, 1 failed |
| `tests/compile/passes/test_functionalization.py` | 8 passed, 7 failed | 8 passed, 7 failed |

The failures are the same on the parent: `test_qsa_e4m3_loader_requires_all_24_scales`
(DID NOT RAISE) and the bfloat16 cases of `test_fix_functionalization`, which a
V100 cannot compile ("BF16 is not supported"). pre-commit: all hooks passed;
mypy-3.10: passed. Counter-check: the new partition test fails, the other 30 pass.

End to end:

| Configuration | pinned host | store card | disk | available after start | text vs reference | decode tok/s |
|---|---|---|---|---|---|---|
| before: `HOST_GIB=6`, no cascade | 12 GiB | – | – | 2.7 GiB | reference | 60.5 / 56.5 / 66.8 / 60.2 |
| cascade: `HOST_GIB=2`, store | 4 GiB | 4.3 GiB | – | 11.8 GiB | 4/4 identical | 59.0 / 54.9 / 65.0 / 58.8 |
| cascade: `HOST_GIB=0`, store | 0 | 8.3 GiB | – | 16.0 GiB | 4/4 identical | 59.1 / 55.1 / 65.3 / 58.8 |
| cascade: `HOST_GIB=2`, store 1 GiB, disk | 4 GiB | 1.0 GiB | 3.3 GiB | not recorded | 4/4 identical | 59.1 / 54.9 / 64.8 / 59.5 |
| before, control boot | 12 GiB | – | – | 2.5 GiB | 4/4 identical | 60.1 / 56.0 / 65.9 / 60.6 |

- Decode against the reference run: -2.5 % (`HOST_GIB=2`), -2.3 % (`HOST_GIB=0`),
  -2.3 % (with disk); against the control boot -2.0, -1.8 and -1.7 %. The
  first disk-tier probe right after a boot ran 3.9 % below the reference, a
  second one 2.3 %.
- Measured device budget per rank: 19.60 and 19.79 GiB of the 23.84 GiB
  shard; the rest goes to the outer tiers.
- Prefill with identical prompts on both sides (13k and 39k tokens of
  synthetic word lists, which makes the absolute numbers higher than with real
  text), first request after boot excluded: 1,092 to 1,100 tok/s before, 1,096
  to 1,101 tok/s with the store tier, 1,096 to 1,100 tok/s with the disk tier.
- Store load at registration: 4.3 GiB in 84.6 s, 8.3 GiB in 157.6 s, 1.0 GiB
  in 25.8 s.
- Disk tier with a cold page cache (`drop_caches` before the boot, three chat
  requests): the worker read the 1.0 GiB of store rows as 261,830 single-page
  major faults and 10 to 56 MiB of disk-tier rows per request, so the rows did
  come from the disk. The store load took 25.8 s, as in warm boots (25.1 to
  25.8 s), and decode forward steps per second (tok/s divided by the MTP
  acceptance length) were within 0.5 % of the same three requests in an
  earlier session whose page cache had not been dropped.
- The worker holds 184 and 212 MiB on the two first-stage cards (nvidia-smi), the
  footprint that #530 describes for the whole-table worker. Our runs did not
  run out of device memory; the measured device budget keeps its reserve free
  (default 8 % of the card, at most 4 GiB), but it does not account for the
  worker's memory, and this PR does not change #530.

## Not a duplicate

`gh pr list --state open --search` for "PLE offload", "PLE host",
"per-layer embedding", "ple_embedding", "PLE disk", "PLE_HOST_GIB",
"pinned host PLE" and "offload worker" returns only our own #622 and #640 (both
dependencies above) and unrelated work (#576, #598, #637). No open PR references
#530 or #479 in its body. Related issues: #479 (ours, Qwen4Exp under pipeline
parallelism; the two PP fixes above continue it) and #530 (see the last point
of the results).

AI assistance (Claude) was used to design, implement and measure this change
and to write this description. I reviewed every changed line and ran the tests
and the end-to-end measurements on the hardware named above.

---
<details>
<summary> Essential Elements of an Effective PR Description Checklist </summary>

- [x] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [x] The test plan, such as providing test command.
- [x] The test results, such as pasting the results comparison before and after, or e2e results
- [x] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model.
</details>
