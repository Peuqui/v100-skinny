GESENDET 06.10.2026 als #1004 — Branch pr-ple-row-gather-policy (aa8ff8f9e auf 1Cat main bc01f2899)
Titel: [Bugfix][Qwen4Exp] Read the PLE row-gather policy when the layer is built
---
## Purpose

Since #885, `Qwen4ExpNGramEmbedding.load_weights` reads `kernel_config.ple_disk_row_gather` through `get_current_vllm_config()` and writes the row-reader admission into `kernel_config.ple_disk_row_readers`. `get_current_vllm_config()` asserts outside a `set_current_vllm_config()` context. The engine loads inside one, but direct loads do not, and ten cases in `tests/models/qwen4_exp/test_ple.py` fail on main with that assertion.

This PR

- reads the policy when the layer is built, next to `ple_disk_release_pages`, which `__init__` already reads through `get_current_vllm_config_or_none()`, and records the admission only when an engine config exists (same pattern as #857 for the SM70 sparse policy);
- makes the native row gather (`csrc/ple_disk_rows.cpp`) report an out-of-range row id the way the Python lane and the tests do: `TORCH_CHECK_INDEX` (an `IndexError` in Python) with the Python lane's message and the offending id, instead of a `RuntimeError` with "row ID". Callers catching `IndexError` missed it on the native lane;
- adds the two new attributes to the test helper that builds the layer without `__init__`.

In the engine nothing changes: the layer is built inside the config context, so it reads the same `KernelConfig` object as before.

Not a duplicate: no open PR touches `ple_disk_row_gather` or `ple_disk_rows.cpp` (gh pr list --state open --search "ple_disk_row_gather", "ple_disk_rows": none; "row id out of range" only finds #828, QSA prefill attention, unrelated). #903 also edits `ple_layer.py`, in other functions (fused n-gram ids, PLE MTP guard); this branch merges cleanly with it.

AI assistance was used for this change. I reviewed every line and ran the tests below.

## Test Plan

```bash
pytest tests/models/qwen4_exp/test_ple.py tests/v1/worker/test_ple_offload_worker.py
pre-commit run --files csrc/ple_disk_rows.cpp tests/models/qwen4_exp/test_ple.py vllm/models/qwen4_exp/nvidia/ple_layer.py
pre-commit run mypy-3.10 --hook-stage manual --files tests/models/qwen4_exp/test_ple.py vllm/models/qwen4_exp/nvidia/ple_layer.py
```

## Acceleration and benchmark contract (required for performance changes)

Not a performance change. The admission decision is the same; only where the policy is read from moves, and the native error type changes.

## Test Result

- On main bc01f2899: 10 of the cases in `test_ple.py` fail (`AssertionError: Current vLLM config is not set`). With the Python part alone, 7 of them pass and the 3 cases of `test_ngram_embedding_disk_gather_rejects_invalid_ids` then reach the native lane and fail on the exception type (checked on 17310de95; none of the three files changed since); with the native part all pass.
- Main bc01f2899 built from source (sm_70 only) with this PR: Tesla V100-PCIE-32GB `test_ple.py` 98 passed, `test_ple_offload_worker.py` 55 passed and 1 skipped; Quadro RTX 8000 95 passed and 3 skipped (the existing exact-SM70 cases), 55 passed and 1 skipped.
- pre-commit and mypy (manual stage) pass.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
