GESENDET 06.10.2026 als #1006 — Branch pr-sm70-fp8-small-shape-tuning (eefb1ae50 auf 1Cat main bc01f2899)
Titel: [SM70] Make FP8 small-shape runtime tuning a KernelConfig field, off by default
---
## Purpose

The native FP8 selectors (`select_fp8_dense_dispatch_policy` and `select_fp8_moe_dispatch_policy` in `csrc/sm70_turbomind/ops/awq_sm70_gemm.cu`) return `DispatchPolicy::kMeasure` for every new small shape (dense M <= 16 unless `VLLM_SM70_FP8_DENSE_TUNE_MAX_M` says otherwise, MoE up to its token limit) while `VLLM_SM70_FP8_TUNE_SMALL_SHAPES` is unset, i.e. by default. They then time the kernel candidates once and keep the fastest. On DeepSeek-V4-Flash two things follow:

- A shape the SM70 warmup does not cover stalls its first request while the candidates are timed. With py-spy on the engine, the slow request spends 493 samples (about 2.5 s at 200 Hz) in `_C::fp8_gemm_sm70_out` called from the grouped FP8 output projection (`sm70_fp8.py`, the per-group loop); the repeated request spends none. Of five new short prompts, three took 2.27-2.94 s to the first token on their first request (two boots); with tuning off the same requests took 0.39-0.57 s.
- The candidates are close enough that timing noise picks the winner, so the choice changes from boot to boot. With it the summation order changes, and greedy output differed between boots of the same build (two of three greedy prompts changed between two boots).

Your migration notes already record the second point: items 285-290 and 396 in `docs/design/sm70_v100_migration_control.md` keep the dynamic FP8 selector out of quality baselines because of full-model token drift, and pin `VLLM_SM70_FP8_TUNE_SMALL_SHAPES=0` for them. The same notes measured the price of fixed dispatch on Qwen3.6-27B-FP8 TP2: 42.22 tok/s instead of 44.41 tok/s steady decode (-4.9 %). On DeepSeek-V4-Flash PP5 I measured no comparable loss (see below). This PR makes the deterministic choice the default and keeps tuning as a per-engine opt-in.

This PR

- adds `kernel_config.sm70_fp8.small_shape_tuning` (default off). `VLLM_SM70_FP8_TUNE_SMALL_SHAPES` stays as a deprecated alias (registration metadata updated, category `deprecated`); explicit configuration wins, and resolution does not write the process environment;
- adds the native op `sm70_set_fp8_small_shape_tuning(bool)`, which takes precedence over the environment variable in `fp8_tune_small_shapes_enabled()`. Without a call the native unset default stays on, so standalone benchmarks keep their behavior;
- has each GPU worker hand the setting to the native selector before loading the model. The worker applies it even when the FP8 policy is not resolved for the engine's quantization: DeepSeek-V4's `deepseek_v4_fp8` reaches the same selector without resolving `Sm70Fp8Config`;
- lets the SM70 warmup read the same setting instead of the environment for its LUT-cache decision and its coordinated tuning, so an engine with tuning off no longer skips the LUT import as if it measured.

AWQ, MXFP4 and NVFP4 small-shape tuning are untouched.

Not a duplicate: no open PR touches the FP8 small-shape tuning (gh pr list --state open --search "TUNE_SMALL_SHAPES", "kMeasure", "fp8_gemm_sm70_out": none; "small shape tuning" finds #903, MTP and segmented decode, which does not touch these selectors, the warmup or the kernel config; this branch merges cleanly with it).

AI assistance was used for this change. I reviewed every line and ran the tests and measurements below.

## Test Plan

```bash
pytest tests/quantization/test_sm70_fp8_kernel_selection.py tests/quantization/test_sm70_warmup.py tests/tools/test_check_env_metadata.py
pytest tests/v1/worker/test_gpu_worker_memory_profile.py tests/v1/worker/test_gpu_worker_static_pp.py tests/v1/worker/test_gpu_worker_profile_failure.py tests/v1/worker/test_gpu_worker_allocator.py
.venv/bin/python -m tools.generate_env_reference --check
pre-commit run --files <the ten changed files>
pre-commit run mypy-3.10 --hook-stage manual --files <the ten changed files>
```

`test_small_shape_tuning_policy` covers default, alias on/off and explicit-over-alias, both resolved and unresolved, and that the environment is not written; `test_small_shape_tuning_native_setter` calls the op on an SM70 build; the warmup tests now pass the engine setting, and the LUT-cache case covers both settings.

## Acceleration and benchmark contract (required for performance changes)

- Default enabled or opt-in: fixed dispatch by default; runtime tuning opt-in through `kernel_config.sm70_fp8.small_shape_tuning=true` (or the deprecated variable).
- Required CLI options and environment switches: none.
- KV cache dtype used for the benchmark: fp8.
- Wheel SHA or source commit: our fork at 6576724ad (1Cat main fb52756f4 plus our open PRs), native build from source, sm_70 only; tuning on = unset variable, off = `VLLM_SM70_FP8_TUNE_SMALL_SHAPES=0`, same build. Boot stability rechecked with this field at its default on 65a3bf176.
- PYTHONPATH, source overlays, or external native libraries used: PYTHONPATH on the fork checkout, otherwise none.
- User-entry route-hit, speed, and output-quality evidence: DeepSeek-V4-Flash at PP5 on 3x Tesla V100-PCIE-32GB + 2x Quadro RTX 8000, DSpark with 5 speculative tokens, `--max-num-seqs 1`, three boots per arm (two series on the same day, each interleaved with an unchanged reference build), the first long request after a boot not counted (lower is better): prefill of a 35,758-token prompt on 15.5 s, off 15.5-15.6 s; decode step at that context on 73-75 ms, off 74-76 ms; short prompts on 69-72 ms, off 70-73 ms; 1.9k-token edit prompt on 76-77 ms, off 78 ms. Time to first token, first request of a new prompt: on 2.27-2.94 s for three of five prompts, off 0.39-0.57 s; repeated requests 0.4-0.5 s in both. Greedy, three prompts, 200 tokens, temperature 0: on, the three boots gave two different outputs for two of the prompts; off, identical across five boots (three with the variable set to 0, two with this field at its default). Eight quality prompts read by hand: all correct in both arms; three answers are identical, five differ only in wording and length.
- Existing kernel/backend registry and capability-rejection reasons: unchanged; only the dispatch policy inside the existing TurboMind FP8 selectors changes.
- Per-engine configuration and deprecated-variable compatibility: per-engine field; the native selector state is per process and set by each worker before loading, so two engines in one process with different settings would share the last one. The variable keeps working as an alias with a deprecation warning when the FP8 policy resolves.
- New/changed environment metadata and generated reference check: `VLLM_SM70_FP8_TUNE_SMALL_SHAPES` moved to `deprecated` with its replacement; `tools.generate_env_reference --check` and `check_env_metadata.py` pass (the variable is not user-visible, so the generated reference is unchanged).
- Route snapshot command and intentional changed rows: none; the routes stay, only `kMeasure` becomes `kDefault` for FP8 small shapes.
- SM70 registered variables / unregistered reads / model-parameter restrictions, before -> after: unchanged count; no new variable, no unregistered read.

## Test Result

- On main bc01f2899 with this PR (extensions built from source, sm_70 only), on Tesla V100-PCIE-32GB and on Quadro RTX 8000 alike: `test_sm70_fp8_kernel_selection.py` 19 passed and 1 failed, `test_sm70_warmup.py` 26 passed, `test_check_env_metadata.py` 14 passed, the four `test_gpu_worker_*` files above 23 passed. The one failing case, `test_unused_fp8_policy_preserves_nvfp4_fingerprint`, fails identically on main (stale hash, repaired in #1005).
- `tools.generate_env_reference --check`, pre-commit and mypy (manual stage) pass.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
