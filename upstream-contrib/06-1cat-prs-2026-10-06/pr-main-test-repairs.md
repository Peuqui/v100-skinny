GESENDET 06.10.2026 als #1005 — Branch pr-main-test-repairs (5537bda8f auf 1Cat main bc01f2899)
Titel: [Test] Repair twelve SM70 test cases broken by recent main changes
---
## Purpose

Twelve test cases fail on main bc01f2899 on Tesla V100 and Quadro RTX 8000 alike. In each case a test stand-in or expectation predates a change on main; the code under test behaves as intended. This PR only touches tests.

- `tests/v1/worker/test_sm70_long_attention_graphs.py::test_runner_does_not_dispatch_short_prefill_as_tail` (6 cases): the hand-built `GPUModelRunner` has no `device`, which `execute_model` reads since the mixed-prefill timer (1cabc7ce9). The stand-in now runs on the CPU device, which skips the timer.
- `tests/v1/attention/test_gdn_metadata_builder.py` (1): `FakeLayer` lacks `_can_use_sm70_gdn_preprocess`, consulted since the SM70 verifier preprocessing (deadbde95). The case covers the existing convolution path, so the stand-in declines.
- `tests/quantization/test_sm70_modelopt_mixed_nvfp4.py` (2): the automatic MoE route asks the skinny MoE admission first, which records into the engine's kernel config (356306b73) and asserts without a config context. The TurboMind-required case now states its precondition (skinny MoE not admitted); the TurboMind case runs inside a config context.
- `tests/v1/core/test_scheduler.py::test_schedule_partial_requests` (1): the test is about the encoder budget; the mixed-prefill limit (512 tokens for a prefill next to a running decode, 1cabc7ce9) now caps the second request at 512 instead of 700. The limit is turned off for this test, as f48eb63d9 did for another hand-built scheduler.
- `tests/quantization/test_sm70_fp8_kernel_selection.py::test_unused_fp8_policy_preserves_nvfp4_fingerprint` (1): the recorded hash is stale; updated. A bisect from the commit that recorded it (0a10b96e0) finds a096d6d28 first, and e03af7c79 (`sm70_gguf.lut4_expert_dp4a`) changed it again. Noted in passing, not changed here: several kernel-config fields enter `KernelConfig.compute_hash()` also for engines that never consult them (`sm70_rmsnorm_gated_exact` while None, `sm70_gguf`, `sm70_ring`, `ple_pinned_decode`, `ple_result_transport`, `hc_ll_shard`), unlike `sm70_awq`/`sm70_fp8`, which are ignored while unresolved, so this hash moves with every new field there. Whether they should stay out of the compile identity is your call.
- `tests/v1/worker/test_release_cleanup.py::test_sm70_workspace_cleanup_releases_all_tensor_owners` (1): the SM70 FP8 workspace caches moved from `layers/quantization/fp8.py` to `kernels/linear/scaled_mm/sm70_fp8.py` (333217c91), which `_clear_loaded_gpu_workspaces` already clears; the test still filled them on `fp8` (`AttributeError`). It now fills and checks them in their new module.

The other ten failing cases on main (`tests/models/qwen4_exp/test_ple.py`) are a code issue and have their own PR (#1004).

Not a duplicate: no open PR touches these tests (gh pr list --state open --search "test_sm70_long_attention_graphs", "test_gdn_metadata_builder", "test_sm70_modelopt_mixed_nvfp4", "test_schedule_partial_requests", "nvfp4_fingerprint", "test_release_cleanup": none). My issue #739 lists two other failing tests; they are not part of this PR.

AI assistance was used for this change. I reviewed every line and ran the tests below.

## Test Plan

```bash
pytest tests/v1/worker/test_sm70_long_attention_graphs.py tests/v1/attention/test_gdn_metadata_builder.py tests/quantization/test_sm70_modelopt_mixed_nvfp4.py tests/v1/core/test_scheduler.py tests/quantization/test_sm70_fp8_kernel_selection.py tests/v1/worker/test_release_cleanup.py
pre-commit run --files <the six test files>
pre-commit run mypy-3.10 --hook-stage manual --files <the six test files>
```

## Acceleration and benchmark contract (required for performance changes)

Not a performance change; tests only.

## Test Result

- Main bc01f2899 built from source (sm_70 only) with this PR, Tesla V100-PCIE-32GB: 44, 116, 70, 107, 15 and 5 passed (files in the order of the test plan). Quadro RTX 8000: 44 passed; 94 passed, 22 skipped; 64 passed, 6 skipped; 107, 15 and 5 passed (the skips are the existing SM70-only cases).
- On main without this PR the twelve cases fail as listed.
- pre-commit and mypy (manual stage) pass.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
