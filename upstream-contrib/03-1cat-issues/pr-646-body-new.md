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
(30 GiB RAM) the split pins 12 GiB and leaves about 2.7 GiB available after
startup, a table held in the worker needs more memory than the host has, and
the disk lane reads every row from the checkpoint.

This PR adds an overflow cascade. Each rank fills three tiers, fastest first:

1. device memory of the compute card, up to its measured budget (the same
   computation as the existing auto placement: utilization budget minus
   allocated memory, the KV cache for `max_model_len` and a reserve),
2. a pinned host share,
3. the checkpoint on disk, mapped and read in place by the PLE offload worker;
   nothing is copied.

What it gives a deployment:

- Host memory back. At TP2 x PP2 on the rig below, available memory goes from
  2.7 GiB (12 GiB pinned) to 16 GiB with no host share, or 11.5 GiB with 2 GiB
  per rank.
- Topologies that did not fit. Under PP4 the first stage holds 18.9 GiB of the
  table and the other 28.8 GiB have nowhere to go on a 30 GiB host; with the
  disk tier it serves, with 17.8 GiB available.
- Identical output. Greedy text is identical to the path without the cascade
  in 3 of 3 probes in every configuration measured.
- A small price: on a cold page cache the disk tier reads 35 to 120 MiB per
  request at TP2 x PP2. Against a 2 GiB host share per rank, reading those rows
  from disk cost about 1 % of prefill and decode.
- Rows are never dropped: a table that does not fit the configured tiers fails
  at startup with a message that names the knobs.
- A fitting table pins nothing. With the cascade on, the host share is used
  only for what the device cannot hold.

Nothing changes without the new switches: without `VLLM_QWEN4EXP_PLE_DISK`
and `VLLM_PLE_DISK_RELEASE_PAGES` the pinned-host split, the whole-table
worker, the disk lane and the hybrid lane keep their behaviour.

An earlier version of this PR also had a store tier on a spare GPU. Measured
against the disk tier it saved 0.3 to 2.5 s of prefill on a cold cache and no
decode time, and loading it after every stage had set up its KV cache needed a
hook in the general worker startup. It is gone; the configuration and the
worker are simpler.

## Configuration

| Variable | Meaning |
|---|---|
| `VLLM_QWEN4EXP_PLE_DISK=1` | Starts the cascade: rows beyond device and host are read from the checkpoint. |
| `VLLM_QWEN4EXP_PLE_HOST_GIB` | Unchanged meaning: pinned host share per rank. 0 skips the host tier. |
| `VLLM_PLE_DISK_RELEASE_PAGES=1` | Unmap the checkpoint pages the worker read after every disk gather, in the cascade and in the whole-table disk lane. Recommended on hosts with little RAM; off (default) keeps them mapped. |

Examples:

```
# PP4: the first stage holds what fits, the rest is read from disk
VLLM_QWEN4EXP_PLE_HOST_GIB=0 VLLM_QWEN4EXP_PLE_DISK=1 VLLM_PLE_DISK_RELEASE_PAGES=1 vllm serve ... --pipeline-parallel-size 4
# TP2 x PP2 with 2 GiB pinned per rank
VLLM_QWEN4EXP_PLE_HOST_GIB=2 VLLM_QWEN4EXP_PLE_DISK=1 vllm serve ... --tensor-parallel-size 2 --pipeline-parallel-size 2
```

The cascade cannot be combined with `VLLM_SM70_QWEN38_HYBRID_PLE` or
`VLLM_PLE_DISK_OFFLOAD`; `VllmConfig` refuses that, and a model without PLE
layers, before any worker starts.

## How it works

- Placement (`common/ple.py`): `plan_ple_placement` returns a three-tier
  `PLEPlacement` (device, host and disk rows). With the cascade, device rows
  come first up to the measured budget, then host, then disk. Without it the
  host share comes first and the device holds the rest, as before.
- Compute ranks (`nvidia/ple_layer.py`): `Qwen4ExpNGramEmbedding` keeps its
  resident tables under the offload contract (`offload_keeps_local_tables()`,
  a new hook on `PleOffloadLayer`). It gathers device and host rows as today,
  waits for the worker with `ple_offload_wait` inside the CUDA graph, and
  merges the worker's rows by rank-local id before the tensor-parallel
  all-reduce. The worker sends raw FP8 bytes, dequantized with the same op and
  scale as the resident gathers, which is why outputs stay identical.
- Registration: each rank sends a `PLERemotePlacement` (vocabulary range,
  resident rows) in `PleOffloadRegistration.remote_placements`.
- Worker: the ranks' disk segments are disjoint in the global id space, so one
  output buffer serves every rank and each rank takes only its own slots. The
  disk tier reuses the disk lane's mmap reader, factored out as
  `_gather_mapped_rows` and used by both.
- With `VLLM_PLE_DISK_RELEASE_PAGES=1` the worker unmaps the checkpoint pages
  it read after each gather (`madvise(MADV_DONTNEED)` on the private,
  file-backed shard mappings; the pages stay in the page cache). Mapped pages
  are ones the kernel keeps: without this the worker's mapped pages grew by
  about 150 MiB per request on our 30 GiB host while the kernel swapped other
  processes out. It is a switch, off by default, because it also reaches the
  whole-table disk lane, and a host with RAM to spare is better off keeping
  the pages mapped: re-reading an unmapped page costs a fault even when it is
  still in the page cache. Only shards the loader
  recorded as file-backed are released; on anonymous memory `MADV_DONTNEED`
  discards data. The existing test
  `test_ngram_embedding_retains_and_gathers_disk_shards` stubbed that check
  with anonymous tensors and now loads a real safetensors file.
- `copy_ple_embedding_shard_` copies with `copy_` directly instead of
  `source.to(...)` first: the caching allocator kept the staging block (382 MiB
  for one 0.37 GiB checkpoint shard, measured in isolation) reserved on the
  card that holds the device tier.
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
  branch carries #622's commit as its first commit. Please review the commits
  after it; the first one disappears on rebase once #622 is merged.
- #640: required for `nvidia/Qwen3.8-Flash-Next-NVFP4`. Without it that
  checkpoint cannot take the pinned-host path on pre-Ampere cards, which the
  cascade builds on:
  `NotImplementedError: Qwen4Exp pinned-host PLE requires FP8 checkpoint storage`.
  Checkpoints that set `ple_embedding_dtype` do not need it.
- #639: required for MTP with that checkpoint under pipeline parallelism, which
  is the configuration measured below. The cascade itself does not depend on it.

## Test Plan

All on this branch, compiled extensions from our fork build, Python 3.12,
torch 2.10.0+cu128, `CUDA_VISIBLE_DEVICES` = two Tesla V100 (some tests set
DP=2 and query device 1):

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

The memory tests were checked against broken code:
`test_shard_copy_to_the_device_keeps_no_staging_memory` fails with
`source.to(...)` restored (20 MiB reserved against 10 MiB);
`test_disk_gather_unmaps_the_pages_it_read` fails when the release is forced
off, and `test_disk_gather_keeps_its_pages_mapped_by_default` fails when it is
forced on.

End to end, on our fork, whose PLE files equal this branch except for #640's
hunk and the connector change of #654: 2x Quadro RTX 8000 + 2x Tesla V100, all
PCIe Gen3 x4 without peer-to-peer, 30 GiB host RAM, checkpoint on an NVMe SSD
in a USB enclosure. `nvidia/Qwen3.8-Flash-Next-NVFP4`, MTP k=4, async
scheduling, `--max-model-len 262144`, `--gpu-memory-utilization 0.95`.

- Greedy probe: three prompts, 200, 142 and 200 tokens, temperature 0,
  SHA-256 of the text against a reference taken at TP2 x PP2 without the
  cascade (`VLLM_QWEN4EXP_PLE_HOST_GIB=3`).
- Timing: twelve different real texts of 8k to 17k tokens (slices of our own
  engineering notes), page cache of the checkpoint evicted before the first
  one, first request after boot discarded. Word-list prompts share most
  n-grams and hit the page cache far more often than real text, so they
  understate the disk tier.

## Test Result

Unit tests, this branch against its parent (main 02c87ab8 + #622):

| File | parent | this branch |
|---|---|---|
| `tests/models/qwen4_exp/test_ple.py` | 60 passed | 80 passed |
| `tests/v1/worker/test_ple_offload_worker.py` | 27 passed | 31 passed |
| `tests/compile/test_sm70_decode_graph.py` | 23 passed | 24 passed |
| `tests/v1/worker/test_release_cleanup.py` | 4 passed | 4 passed |
| `tests/v1/executor/test_executor.py` | 11 passed | 11 passed |
| `tests/models/qwen4_exp/test_weight_loading.py` | 27 passed, 1 failed | 27 passed, 1 failed |
| `tests/compile/passes/test_functionalization.py` | 8 passed, 7 failed | 8 passed, 7 failed |

The failures are the same eight tests on both sides:
`test_qsa_e4m3_loader_requires_all_24_scales` (DID NOT RAISE) and the
bfloat16 cases of `test_fix_functionalization`, which a V100 cannot compile.
pre-commit: all hooks passed; mypy-3.10: passed.

TP2 x PP2 (RTX 8000 pair first stage, V100 pair second, 24 + 24 layers):

| Configuration | pinned host | disk | available | prefill, sum of 12 | decode tok/s, mean (range) | swap-out per request | text vs reference |
|---|---|---|---|---|---|---|---|
| before: `HOST_GIB=6`, no cascade | 12 GiB | – | 2.7–3.3 GiB | 124.8 s | 29.4 (17.7–63.3) | 0–306 MiB | 3/3 identical |
| cascade: `HOST_GIB=0` | 0 | 8.3 GiB | 16 GiB | 118.8 s | 49.0 (44.2–65.1) | 0 | 3/3 identical |
| cascade: `HOST_GIB=2` | 4 GiB | 4.3 GiB | 11.5 GiB | 117.3 s | 49.6 (43.2–63.2) | 0 | 3/3 identical |

- The "before" row ran with our usual services on the host, as in production.
  With 2.7 GiB left the kernel swapped during most requests, and that is what
  halves its decode; it is not a property of the pinned-host gather. The first
  version of this description measured the same path with short prompts, also
  at 2.7 GiB available, and saw no slowdown; the 15k-token real texts here
  need more memory per request.
- Measured device budget per rank: 19.60 and 19.79 GiB of the 23.84 GiB
  shard.
- The disk tier read 35 to 120 MiB per request with no host share and 18 to
  62 MiB with 2 GiB per rank.

PP4 (TP1 x PP4, 12 layers per stage), same twelve texts, cold page cache.
Stage 0 holds 18.9 GiB of the table; without the disk tier the other 28.8 GiB
would have to be pinned, more than the host has:

| Configuration | pinned host | disk | available | prefill, sum of 12 | decode tok/s, mean (range) | swap-out |
|---|---|---|---|---|---|---|
| cascade: `HOST_GIB=0` | 0 | 28.8 GiB | 17.6–17.8 GiB | 105.1 s | 40.8 (35.6–50.9) | 0 |
| cascade: `HOST_GIB=3` | 3 GiB | 25.8 GiB | 13.2–13.6 GiB | 102.9 s | 41.1 (34.9–45.4) | 0 |

- `HOST_GIB=0` with `VLLM_PLE_DISK_RELEASE_PAGES=1` at PP4 is what we now run
  in production: needles 4/4 at
  24,488 and 101,605 tokens, the worker's mapped file pages at 113 MB after
  loading and after all requests.
- Releasing the pages, PP4 with a 12 GiB host share and 16.8 GiB on disk,
  3 GiB available, twelve texts: worker RssFile 112 to 1,939 MiB and 2.1 GiB
  swapped out without it; 105 to 109 MiB flat and 96 MiB swapped out with it,
  prefill and decode unchanged. All cascade rows above ran with the release
  on.
- In the first version of this PR the worker held 184 and 212 MiB on the two
  first-stage cards (nvidia-smi), the footprint that #530 describes for the
  whole-table worker; the worker's device side is unchanged since. The measured
  device budget keeps its reserve free (default 8 % of the card, at most
  4 GiB), but it does not account for the worker's memory, and this PR does
  not change #530.

## Not a duplicate

`gh pr list --state open --search` for "PLE offload", "PLE host",
"per-layer embedding", "ple_embedding", "PLE disk", "PLE_HOST_GIB",
"pinned host PLE" and "offload worker" returns only our own #622, #640 and #654
and unrelated work. #684 (draft) speeds up short gathers in the disk lane and
touches the same files (`ple_layer.py`, `ple_offload/worker.py`, `envs.py`,
the PLE tests). It is not a duplicate, but the two meet in
`_disk_embedding_lookup`, which this PR turns into the shared
`_gather_mapped_rows`; if #684 lands first, I will rebase this PR onto it and
carry its short-gather path into the shared reader. Its fast path relies on the
pages staying mapped, which is why the release here is off by default. Other
than this PR and #654, no open PR references #530 or #479 in its body. Related
issues: #479 (ours, Qwen4Exp under pipeline parallelism; the two PP fixes above
continue it) and #530 (see the last point of the results).

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

🤖 Generated with [Claude Code](https://claude.com/claude-code)
