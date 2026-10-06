ENTWURF (nicht gesendet) — Branch pr-ple-row-gather-policy (23e8b97bb auf 1Cat main 17310de95)
Titel: [Bugfix][Qwen4Exp] Read the PLE row-gather policy when the layer is built

## Purpose

Since #885, `Qwen4ExpNGramEmbedding.load_weights` reads `kernel_config.ple_disk_row_gather` through `get_current_vllm_config()` and writes the row-reader admission into `kernel_config.ple_disk_row_readers`. `get_current_vllm_config()` asserts outside a `set_current_vllm_config()` context. The engine loads inside one, but direct loads do not, and ten cases in `tests/models/qwen4_exp/test_ple.py` fail on main with that assertion.

This PR

- reads the policy when the layer is built, next to `ple_disk_release_pages`, which `__init__` already reads through `get_current_vllm_config_or_none()`, and records the admission only when an engine config exists (same pattern as #857 for the SM70 sparse policy);
- makes the native row gather (`csrc/ple_disk_rows.cpp`) report an out-of-range row id the way the Python lane and the tests do: `TORCH_CHECK_INDEX` (an `IndexError` in Python) with the Python lane's message and the offending id, instead of a `RuntimeError` with "row ID". Callers catching `IndexError` missed it on the native lane;
- adds the two new fields to the test helper that builds the layer without `__init__`.

Not a duplicate: no open PR touches `ple_disk_row_gather` or `ple_disk_rows.cpp` (gh pr list --state open --search "ple_disk_row_gather", "ple_disk_rows": none; "row id out of range" only finds #828, QSA prefill attention, unrelated).

AI assistance was used for this change. I reviewed every line and ran the tests below.

## Test Plan

```bash
pytest tests/models/qwen4_exp/test_ple.py
pre-commit run --files csrc/ple_disk_rows.cpp tests/models/qwen4_exp/test_ple.py vllm/models/qwen4_exp/nvidia/ple_layer.py
```

## Test Result

- On main 17310de95: 10 of the cases fail (`AssertionError: Current vLLM config is not set`). With the Python part alone, 7 of them pass and 3 (`test_ngram_embedding_disk_gather_rejects_invalid_ids`) then reach the native lane and fail on the exception type; with the native part all pass.
- Main 17310de95 built from source (sm_70 only) with this PR: Tesla V100-PCIE-32GB 98 passed; Quadro RTX 8000 95 passed, 3 skipped (the existing exact-SM70 cases). Same on our fork with the PLE disk-tier additions: 114 passed / 111 passed, 3 skipped.
- pre-commit passes.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
