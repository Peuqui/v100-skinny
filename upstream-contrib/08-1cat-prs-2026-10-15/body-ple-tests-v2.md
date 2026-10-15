## Purpose

Four tests in tests/models/qwen4_exp/test_ple.py fail on main 6f3557c97 (they already failed on be3722bc0):

- `test_ngram_embedding_disk_offload_allocates_only_meta_table` and `test_ngram_gpu_offload_retains_only_fp8_global_scale` set `ple_module.envs.VLLM_PLE_DISK_OFFLOAD` / `VLLM_PLE_CPU_OFFLOAD` as module attributes. The n-gram embedding reads both flags through `ple_policy()`, which resolves them from the environment (`resolve_legacy_fields` calls the env registry reader), so the attribute never reaches it: the first test sees `_disk_offload` False; the second skips the CPU-offload branch of `load_weights` that keeps only the weight scale, and the regular loader then fails on the missing `layer_multipliers` buffer. The tests now set the environment variables with `monkeypatch.setenv`.
- `test_ngram_cpu_offload_padding_does_not_overwrite_real_tokens` and `test_ngram_fp8_cpu_offload_preserves_quantized_output` build the module with `__new__` and fail with `AttributeError: 'Qwen4ExpNGramEmbedding' object has no attribute 'layer_name'`, which `forward_impl` uses for its diagnostic labels since 19cbb4089. The tests now set `layer_name`.

A second commit repairs the remaining stale fakes in the same PLE offload area and the SM70 decode-graph test (20 more failures on main 6f3557c97):

- `tests/compile/test_sm70_decode_graph.py::test_shared_capture_context_reaches_target_and_draft` (8 cases): `CudaGraphManager` now captures through a shared `capture_context` and passes `stream=` to `torch.cuda.graph` (629063e7d); the hand-built manager sets the context and the graph fake accepts keywords.
- `tests/kernels/ple/test_host_result.py::test_mtp_mapped_result_transport_admission` (4) and `tests/models/qwen4_exp/test_gguf_pinned_policy.py` (5): the hybrid PLE flag is read through `ple_policy`, not a module-level `envs` import, so `monkeypatch.setattr(module.envs, ...)` raised AttributeError; the fakes now provide the policy or clear the environment variable.
- `tests/v1/worker/test_ple_offload_worker.py` (3): `VLLM_PLE_CPU_OFFLOAD` is set in the environment that `ple_policy` reads, the runner fake gets a `kernel_config` for the fast-lane check, and the connector fake its `fast_lane` attribute.

No runtime code changes.

Not a duplicate: the open PRs that edit these files (#1199 and #1094, #1023, #1020 for test_ple.py / test_ple_offload_worker.py) do not touch the repaired tests or fakes; gh pr list --state all --search "test_ple", "test_sm70_decode_graph", "test_host_result", "capture_context", "stale fakes" finds no other PR for them.

AI assistance was used for this change. I reviewed every line and ran the tests below.

## Test Plan

```bash
pytest tests/models/qwen4_exp/test_ple.py tests/v1/worker/test_ple_offload_worker.py tests/models/qwen4_exp/test_gguf_pinned_policy.py tests/compile/test_sm70_decode_graph.py tests/kernels/ple/test_host_result.py
pre-commit run --from-ref origin/main --to-ref HEAD
```

## Path coverage

N/A, test-only change: no runtime path is selected differently.

## Test Result

- test_ple.py alone: on a Tesla V100 104 passed (main: 4 failed, 100 passed); on a Quadro RTX 8000 101 passed, 3 skipped (main: 4 failed, 97 passed, 3 skipped).
- The four other files, all GPUs visible, Quadro RTX 8000 or Tesla V100 as device 0: 125 passed each (main: 20 failed, 105 passed each).
- All five files together: 229 passed with a V100 as device 0, 226 passed and 3 skipped with an RTX 8000 as device 0.
- pre-commit run --from-ref origin/main --to-ref HEAD: all hooks pass.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01Ly7LM3m9fa6DRJPYQjnt95
