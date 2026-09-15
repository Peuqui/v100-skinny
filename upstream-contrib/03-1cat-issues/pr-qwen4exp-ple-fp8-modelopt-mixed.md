# PR-Entwurf: [Bugfix][Qwen4Exp] Detect the FP8 PLE table from ModelOpt mixed-precision configs

Status: ENTWURF 14.09.2026, nicht gesendet. Worktree `1Cat-vLLM-pr-plemixed`,
Branch `qwen4exp-ple-fp8-modelopt-mixed` auf origin/main 02c87ab8, gestaged,
nicht committet, nicht gepusht. Fork-Stand: work-main 914333d6.

Titel: [Bugfix][Qwen4Exp] Detect the FP8 PLE table from ModelOpt mixed-precision configs

## Purpose

`nvidia/Qwen3.8-Flash-Next-NVFP4` stores its PLE table exactly like the other
Flash-Next NVFP4 exports: 129 FP8 E4M3 shards under
`model.language_model.layers.1.ple.ple_embedding.ngram_embedding` with one
`weight_scale`. It declares that table in the ModelOpt mixed-precision config
(`quantized_layers[...ngram_embedding] = {"quant_algo": "FP8"}`) and does not
set `ple_embedding_dtype` in `text_config`.

`_get_ple_embedding_quant_method` only recognises FP8 storage through
`ple_embedding_dtype == "float8_e4m3fn"` or an `Fp8Config`. For this checkpoint
it returns `None`, and on pre-Ampere cards, where the pinned-host PLE placement
is taken, the engine refuses to start:

    NotImplementedError: Qwen4Exp pinned-host PLE requires FP8 checkpoint storage

The change adds a `ModelOptMixedPrecisionConfig` branch: if the table is not
excluded and its per-layer algorithm resolves to `FP8`, the existing
`Qwen4ExpPLEFp8EmbeddingMethod` is selected. The lookup goes through the
config's own `is_layer_excluded` and `_resolve_quant_algo`, which already map
between the checkpoint's `model.language_model.` and vLLM's `language_model.model.`
prefixes and are used the same way for MTP experts in `mtp_fp8_experts.py`.
Checkpoints that set `ple_embedding_dtype` or use `Fp8Config` take the same path
as before.

Not a duplicate: searched open and closed PRs and issues for "PLE FP8",
"ple_embedding_dtype", "ModelOptMixedPrecision PLE", "pinned-host PLE",
"Flash-Next nvidia". #553 concerns the MTP experts of the same checkpoint and
does not touch the PLE table.

## Test Plan

    .venv/bin/python -m pytest tests/models/qwen4_exp/test_ple.py
    pre-commit run --files vllm/models/qwen4_exp/nvidia/ple_layer.py tests/models/qwen4_exp/test_ple.py
    pre-commit run mypy-3.10 --hook-stage manual --files vllm/models/qwen4_exp/nvidia/ple_layer.py tests/models/qwen4_exp/test_ple.py

End to end with `nvidia/Qwen3.8-Flash-Next-NVFP4` revision fc694b54 on 2x Quadro
RTX 8000 + 2x Tesla V100, TP2 x PP2, pinned-host PLE with a 6 GiB host budget.

## Test Result

- Tests: 59 passed, 3 skipped. The new test builds the config from a ModelOpt
  mixed-precision dict and checks both prefix spellings: FP8 table selects the
  FP8 embedding method, an unlisted table and an excluded table do not. Without
  the fix both parametrized cases fail.
- pre-commit and mypy-3.10 pass.
- Before: startup fails on the PLE stage with the NotImplementedError above.
- After: the checkpoint loads straight from its Hugging Face snapshot, without
  editing its config, and serves; the same run is described in the companion PR
  for FP8 MTP experts under pipeline parallelism, which this checkpoint needs as
  well for MTP on that topology.

These end-to-end runs were made on our fork (1Cat main 80c88e8d plus our open
PRs).

AI assistance (Claude) was used for this change and this description. I reviewed
every changed line and ran the tests and measurements above.
