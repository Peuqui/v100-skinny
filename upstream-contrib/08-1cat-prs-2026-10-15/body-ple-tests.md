## Purpose

Four tests in tests/models/qwen4_exp/test_ple.py fail on main 6f3557c97 (they already failed on be3722bc0):

- `test_ngram_embedding_disk_offload_allocates_only_meta_table` and `test_ngram_gpu_offload_retains_only_fp8_global_scale` set `ple_module.envs.VLLM_PLE_DISK_OFFLOAD` / `VLLM_PLE_CPU_OFFLOAD` as module attributes. The n-gram embedding reads both flags through `ple_policy()`, which resolves them from the environment (`resolve_legacy_fields` calls the env registry reader), so the attribute never reaches it: the first test sees `_disk_offload` False; the second skips the CPU-offload branch of `load_weights` that keeps only the weight scale, and the regular loader then fails on the missing `layer_multipliers` buffer. The tests now set the environment variables with `monkeypatch.setenv`.
- `test_ngram_cpu_offload_padding_does_not_overwrite_real_tokens` and `test_ngram_fp8_cpu_offload_preserves_quantized_output` build the module with `__new__` and fail with `AttributeError: 'Qwen4ExpNGramEmbedding' object has no attribute 'layer_name'`, which `forward_impl` uses for its diagnostic labels since 19cbb4089. The tests now set `layer_name`.

No runtime code changes.

Not a duplicate: the open PRs that edit test_ple.py (#1199, #1094, #1023, #1020) do not touch these four tests or the offload flags in them; gh pr list --state all --search "test_ple" finds only the closed #793 and #646.

AI assistance was used for this change. I reviewed every line and ran the tests below.

## Test Plan

```bash
pytest tests/models/qwen4_exp/test_ple.py
pre-commit run --from-ref origin/main --to-ref HEAD
```

## Path coverage

N/A, test-only change: no runtime path is selected differently.

## Test Result

- On a Tesla V100: 104 passed (main: 4 failed, 100 passed). On a Quadro RTX 8000: 101 passed, 3 skipped (main: 4 failed, 97 passed, 3 skipped).
- pre-commit run --from-ref origin/main --to-ref HEAD: all hooks pass.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01Ly7LM3m9fa6DRJPYQjnt95
