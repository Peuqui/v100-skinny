SENDEFASSUNG 11.10.2026 abends. Ersatz für #1006. Branch pr-sm70-fp8-tune-default-off (Worktree 1Cat-vLLM-bisect) = 739083e52 auf
1Cat main be3722bc0 (ein Commit: Standard + neuer Test). Die Testreparatur ist NICHT dabei: #903 (areslp, offen) repariert dieselben
Warmup-Tests (warmup.envs und native_policy-Fakes) → wäre Duplikat. Sicherungen: pr-fp8-default-backup-e9732e9e0,
pr-fp8-with-testrepair-23ee7c9fa. Quellen der Zahlen: fp8tune_chain_2026-10-11.log (main + #1183 = cff4c875c, je zwei Boots Tuning
an/aus per Env), fp8default_pr_2026-10-11.log (e9732e9e0 + #1183 = 51f39d89f, zwei Boots ohne Env; Laufzeitcode = 739083e52), Testläufe
auf 8d04e48f1 (Code gleich 739083e52, nur Commit-Nachricht anders); 1Cat docs/design/sm70_v100_migration_control.md (Punkt 285 Z. 23713,
Punkt 396 Z. 31877, Z. 35762).
Titel: [SM70] Default the FP8 small-shape runtime tuning to off
---
## Purpose

With `fp8_tune_small_shapes` unset, the native FP8 dense and MoE selectors time the kernel candidates for each new small shape and keep the fastest. This PR makes fixed dispatch the default again in the three places that define it: the native policy reader (csrc/sm70_turbomind/ops/gemm_runtime.cpp), the SM70 warmup (vllm/model_executor/warmup/awq_sm70_warmup.py) and the env registry (vllm/envs.py). Tuning stays available through `kernel_config.sm70_fp8.native.fp8_tune_small_shapes` and its alias `VLLM_SM70_FP8_TUNE_SMALL_SHAPES=1`. A new test, `test_fp8_small_shape_tuning_defaults_to_fixed_dispatch`, pins the unset default: the env registry reads it as off, the warmup does not disable the LUT cache for FP8 dense layers, and it runs the plain instead of the coordinated FP8 warmup.

Why, measured on DeepSeek-V4-Flash PP5 (3x Tesla V100 + 2x Quadro RTX 8000, DSpark, greedy), main be3722bc0 with #1183 applied (without #1183 DSpark accepts almost no drafts there), one build, two boots per setting:
- With tuning on, four of the six distinct short prompts took 2.27-4.54 s to the first token on their first request, in both boots. With tuning off, every request took 0.41-0.57 s. Prompts sent a second time took 0.45-0.49 s in both settings.
- Timing noise picks the winner, and the winner sets the summation order. Compared with a greedy reference recorded on our fork with fixed dispatch, one tuning-on boot produced different text for 2 of 3 prompts and the other matched all 3. Both tuning-off boots matched all 3.
- Decode of the three 200-token greedy prompts: the first ran at 45.6-45.8 tok/s with tuning on and 44.5-44.8 tok/s with it off, the second and third at 30.4-38.9 vs 50.3-51.7 tok/s.

The migration notes already treat fixed dispatch as the quality-safe setting: item 285 records that the source default was fixed dispatch because earlier FP8 MoE exactness work found dynamic tuning could produce nonzero Type-A diffs, and item 396 pins `VLLM_SM70_FP8_TUNE_SMALL_SHAPES=0` as the quality-safe dense selector.

Trade-off: the notes also record Qwen3.6-27B-FP8 TP2 at 42.22 tok/s fixed vs 44.41 tok/s dynamic with an identical token hash. Users who want that can still enable tuning per engine; this PR only changes what an unset value means, so runs are reproducible from boot to boot by default.

This replaces #1006, which made the same default change together with a KernelConfig field and a native binding for it. Main has since added the typed field itself and moved the native reader into gemm_runtime.cpp, so only the default is left.

Not a duplicate: gh pr list --state all --search "FP8_TUNE_SMALL_SHAPES", "fp8_tune_small_shapes", "FP8 tuning default", "small-shape tuning": the only PR changing this default is #1006, which this replaces. #903 edits tests/quantization/test_sm70_warmup.py and vllm/envs.py but does not change this default; the two branches merge without conflicts in those files.

AI assistance was used for this change. I reviewed every line and ran the tests and measurements below.

## Test Plan

```bash
pytest tests/quantization/test_sm70_warmup.py tests/quantization/test_sm70_fp8_kernel_selection.py tests/config/test_sm70_provider_lifecycle.py tests/tools/test_check_env_metadata.py
python -m tools.generate_env_reference --check
pre-commit run --from-ref origin/main --to-ref HEAD
```

DeepSeek-V4-Flash PP5 with DSpark, two boots per setting; per boot the first-token time of eight short requests (six distinct prompts, two of them sent twice) and three 200-token greedy prompts compared with the stored reference.

## Test Result

- On a Tesla V100 and a Quadro RTX 8000, each: test_sm70_warmup 12 passed, 14 failed (main: 11 passed, 14 failed; the new test fails without this change); test_sm70_fp8_kernel_selection 14 passed, 1 failed (same on main); test_sm70_provider_lifecycle 20 passed; test_check_env_metadata 21 passed.
- The 14 warmup failures are the same cases on main and this branch: four still call `warmup.envs.disable_envs_cache()`, which no longer exists, and ten replace native ops with fakes that do not accept the `native_policy` argument the warmup passes once the native policy ABI is built. #903 already rewrites both groups (the `warmup.envs` calls and the native-op fakes), so this PR leaves them alone. test_legacy_dispatch_snapshot compares route hashes with tests/quantization/data/sm70_fp8_routes.json and fails the same way on main.
- generate_env_reference --check: 998 environment registrations documented. pre-commit passes.
- This default change on top of #1183, without any env, two boots: first token 0.41-0.58 s for all requests, greedy identical to the reference for 3 of 3 prompts in both boots, decode 44.2-44.5 tok/s for the first greedy prompt and 50.2-51.8 tok/s for the second and third.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
