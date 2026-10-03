# PR-Entwurf: DSv4 SM70 sparse policy at construction (03.10.2026)

Branch `pr-dsv4-sm70-sparse-policy-at-init` (Fork), Basis 1Cat main 248a525cc, Commit 9817bec2d.
Go von Peuqui für Fehlerbehebungen und PRs: 03.10. mittags. Vor dem Push: Revert-Prüfung gegen
frisches main, Duplikatsuche wiederholen, Platzhalter aus Messdateien füllen.

Titel: [Bugfix][DeepSeek-V4] Read the SM70 sparse policy when the layers are built

---

## Purpose

The sparse batched-matmul admission (`_bmm_blocker` in `vllm/models/deepseek_v4/sm70/sparse.py`) and the SM70 indexer decode (`sm70_indexer_decode_logits` in `vllm/models/deepseek_v4/sm70/indexer.py`), both from #816, read `KernelConfig.sm70_sparse` through `get_current_vllm_config()` inside the forward pass. The worker sets that context while it builds, loads and captures the model (`load_model`, `initialize_from_config`, `profile_cudagraph_memory`), but not for the memory profile in `determine_available_memory` or for `execute_model`. On DeepSeek-V4-Flash at PP5 on 3x V100 + 2x Quadro RTX 8000, every start on main stopped in `profile_run` on a V100 stage:

```
AssertionError: Current vLLM config is not set. This typically means get_current_vllm_config() was called outside of a set_current_vllm_config() context, ...
  File "vllm/models/deepseek_v4/sm70/sparse.py", line 59, in _bmm_blocker
    policy = get_current_vllm_config().kernel_config.sm70_sparse
```

An eager decode reaches the same kind of lookup in the indexer. The existing tests call these functions inside `set_current_vllm_config()` (autouse fixtures in test_deepseek_v4_sm70_routes.py and test_deepseek_v4_sm70_indexer.py), so they did not see it.

This PR keeps the per-engine policy with modules that are built while the config is set, the same way #838 keeps the auxiliary GEMV policy: `DeepseekV4MLAAttention` and the indexer K caches (`DeepseekV4IndexerCache`, and `DeepseekV32IndexerCache`, which the GLM-5.3 kpool cache extends) store `kernel_config.sm70_sparse`. `_bmm_blocker` takes the layer's policy, and the two indexer ops pass the setting of the cache registered under their layer name (`no_compile_layers[k_cache_prefix]`). Nothing changes when the context is set.

Not a duplicate: no open PR touches these lookups (gh pr list / gh search for "sm70_sparse", "_bmm_blocker", "get_current_vllm_config forward", "Current vLLM config is not set"). #838 and #847 changed the same attention file for the auxiliary GEMV and left these two lookups as they were.

AI assistance was used for this change. I reviewed every line and ran the tests below.

## Test Plan

```bash
pytest tests/models/test_deepseek_v4_sm70_sparse_policy.py   # new, runs without a config context
pytest tests/models/test_deepseek_v4_sm70_routes.py
pytest tests/kernels/test_deepseek_v4_sm70_indexer.py
pytest tests/kernels/attention/test_dsv4_sm70_sparse_bmm.py tests/kernels/attention/test_dsv4_sparse_prefill_bmm.py
pytest tests/models/glm5next/test_sm70_sparse.py
pre-commit run --files <changed files>
pre-commit run mypy-3.10 --hook-stage manual --files <changed files>
```

## Acceleration and benchmark contract (required for performance changes)

Not a performance change. The admission decisions are the same; only where the policy is read from moves.

## Test Result

The new test fails on main (3 of 3: `_bmm_blocker` raises the assertion above, and the caches have no policy) and passes with this change. On a Tesla V100 all six files pass (96 passed); on a Quadro RTX 8000 78 passed and 18 skipped, all by their stated Volta-only conditions. <ohne GPU nachholen: routes + neuer Test mit CUDA_VISIBLE_DEVICES= (mit sichtbaren GPUs 32 passed)> pre-commit and mypy-3.10 pass.

End to end on 3x V100 + 2x Quadro RTX 8000 (main 035be3644 with this change; the touched code is identical on current main): DeepSeek-V4-Flash at PP5 with the DSpark drafter <Start, Greedy, Tempo aus ab_837b>; without the change the same entry failed to start twice with the assertion above (ab_837 run, 13:21 and 13:33).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
