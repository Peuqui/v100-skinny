Title: [Feature][Qwen4Exp] Allow the PLE CPU offload under pipeline parallelism

## Purpose

`Worker._validate_ple_offload_config` refuses `VLLM_PLE_CPU_OFFLOAD` with
pipeline parallelism (`PP=N` in the unsupported list). Stacked on #646: there a
rank that holds no `PleOffloadLayer` does not create a connector, so the
offload worker still receives exactly one registration per (dp, tp) rank, and
`check_ple_layers_on_first_pp_rank` already refuses a layout that puts PLE
layers anywhere but the first stage. With that in place the PP entry can go.

We run Qwen3.8-Flash-Next this way in production: pipeline parallel over four
cards (2x Quadro RTX 8000 + 2x V100), MTP k=4; the PLE rows that do not fit on
the first stage are served by the offload worker from disk (#646 cascade,
host budget 0).

Until #646 lands this branch contains its commit; please review only the last
commit here.

**Why this is not a duplicate.** No open PR removes the PP restriction
(`gh pr list --state open --search` for "PLE offload pipeline",
"PLE_CPU_OFFLOAD", "ple offload PP"). #702 relaxes the DCP entry in the same
function; the two changes touch neighbouring lines, whichever lands second
needs a trivial rebase there.

AI assistance was used for this change. Every line was reviewed and the tests
below were run by the submitter.

## Test Plan

```bash
pytest tests/v1/worker/test_ple_offload_worker.py
# counter-check: gpu_worker.py reverted to the #646 state, test kept
pytest tests/v1/worker/test_ple_offload_worker.py -k capability_not_model_identity
```

## Test Result

`test_ple_offload_uses_capability_not_model_identity` now runs with
`pipeline_parallel_size` 1 and 4 for each architecture: 6 passed. With the
previous `gpu_worker.py` the three PP=4 cases fail with "VLLM_PLE_CPU_OFFLOAD
does not support the requested configuration. Unsupported settings: PP=4".

Whole file on two Tesla V100: 36 passed, 1 skipped. With a single visible GPU
`test_offload_distributed_sets_config_only_for_model_parallel` fails with an
IndexError on this branch and on #646 alike; it needs two devices.
pre-commit clean.
