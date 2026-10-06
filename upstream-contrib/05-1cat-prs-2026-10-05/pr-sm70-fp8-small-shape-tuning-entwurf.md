ENTWURF (nicht gesendet) — Branch pr-sm70-fp8-small-shape-tuning (4588654de auf 1Cat main 82392a1c1)
Titel: [SM70] Make FP8 small-shape runtime tuning a KernelConfig field, off by default

## Purpose

The native FP8 selector (`select_dense_dispatch_policy_impl` in `csrc/sm70_turbomind/ops/awq_sm70_gemm.cu`) returns `DispatchPolicy::kMeasure` for every new small shape (M <= 16) while `VLLM_SM70_FP8_TUNE_SMALL_SHAPES` is unset, i.e. by default. It then times the kernel candidates once and keeps the fastest. On DeepSeek-V4-Flash two things follow:

- The first request whose shape is new stalls while the candidates are timed. With py-spy on the engine, the slow request spends 493 samples (about 2.5 s at 200 Hz) in `_C::fp8_gemm_sm70_out` called from the grouped FP8 output projection (`sm70_fp8.py`, the per-group loop); the repeated request spends none. Short prompts of a new length took 2.3-2.9 s to the first token instead of 0.4-0.6 s.
- The candidates are close enough that timing noise picks the winner, so the choice changes from boot to boot. With it the split-K and summation order change, and greedy output differed between boots of the same build (two of three greedy prompts changed between two boots).

With tuning off the selector takes `kDefault`, the existing heuristic choice. Decode speed did not change (see below), the first-request stall is gone and greedy output is identical across boots.

This PR

- adds `kernel_config.sm70_fp8.small_shape_tuning` (default off); `VLLM_SM70_FP8_TUNE_SMALL_SHAPES` stays as a deprecated alias, explicit configuration wins, and resolution does not write the process environment;
- adds the native op `sm70_set_fp8_small_shape_tuning(bool)`, which takes precedence over the environment variable in `fp8_tune_small_shapes_enabled()`;
- has each GPU worker hand the setting to the native selector before loading the model. The worker applies it even when the FP8 policy is not resolved for the engine's quantization: DeepSeek-V4's `deepseek_v4_fp8` reaches the same selector without resolving `Sm70Fp8Config`.

AWQ, MXFP4 and NVFP4 small-shape tuning are untouched; I have not measured whether they show the same effect.

Not a duplicate: no open PR touches the FP8 small-shape tuning (gh pr list --state open --search "TUNE_SMALL_SHAPES", "small shape tuning", "kMeasure", "fp8_gemm_sm70_out": none).

AI assistance was used for this change. I reviewed every line and ran the tests and measurements below.

## Test Plan

```bash
pytest tests/quantization/test_sm70_fp8_kernel_selection.py
pre-commit run --files <changed files>
pre-commit run mypy-3.10 --hook-stage manual --files <changed Python files>
```

`test_small_shape_tuning_policy` covers default, alias on/off and explicit-over-alias, both resolved and unresolved, and that the environment is not written; `test_small_shape_tuning_native_setter` calls the op on an SM70 build.

## Acceleration and benchmark contract (required for performance changes)

DeepSeek-V4-Flash at PP5 on 3x Tesla V100-PCIE-32GB + 2x Quadro RTX 8000, DSpark with 5 speculative tokens, `--max-num-seqs 1`, FP8 KV cache. Same build, tuning on (current default) against off; two boots per arm after a cold one; the first long request after a boot not counted.

Time to first token, short prompts, first request of each new length (lower is better): on 2.0-2.9 s, off 0.39-0.57 s; repeated requests 0.4-0.5 s in both.

Decode ms per step (lower is better): 35.7k context on 74-75, off 74-76; short prompts on 69-72, off 70-73; 1.9k edit prompt on 77, off 78. Prefill of 35.7k tokens 15.5 s in both arms.

Greedy (three prompts, 200 tokens, temperature 0): on, two of three prompts changed between boots; off, identical across five boots.

## Test Result

- `tests/quantization/test_sm70_fp8_kernel_selection.py`: all new cases pass; `test_unused_fp8_policy_preserves_nvfp4_fingerprint` fails identically on main (stale hash).
- pre-commit and mypy (manual stage) pass.
- Native build from source, sm_70 only; quality prompts (8) read by hand: equivalent to the tuning-on build.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
