## Purpose

Two changes for the V100 stages of DeepSeek-V4, in production in our fork since 2026-10-24:

1. One grouped GEMM for wo_a (58bc13ca7). On the V100 the wo_a projection (8 local groups of 1024 x 4096 block FP8 per layer at TP1) ran one TurboMind GEMM per group, each with a tile sized for large M: 195 us per layer at the six tokens a DSpark verification step sends. The one-launch grouped path (fp8_moe_gemm_sm70_per_expert_dispatch_out) existed only for two groups of that shape and one token. It now covers every grouped BMM with the native ops up to SM70_FP8_GROUPED_DECODE_MAX_TOKENS (16) tokens per group; the layer keeps one row of group offsets per token count, and the warmup launches each of its decode sizes up to 16. Switch back with kernel_config sm70_fp8.native.fp8_grouped_bmm_decode=false.
2. Measure the warmup in the engine's policy and freeze the plan after it (a72495d1d), with the new switch kernel_config sm70_runtime.freeze_tuning_after_warmup (off by default). With sm70_fp8.native.fp8_tune_small_shapes (on by default on main), TurboMind measures its tile per FP8 shape: dense shapes up to M 64, grouped shapes up to 128 tokens.
   - The warmup did not measure anything that serving reused: sm70_awq_warmup captured its native policy from the current vLLM config, which is not set when the worker runs compile_or_warm_up_model, so the warmup GEMMs dispatched in the default policy partition (TM_GEMM_TRACE: policy=default, measured=0), one the layers never dispatch in. It now runs inside set_current_vllm_config(worker.vllm_config): 14 shapes measured per DeepSeek-V4 V100 stage, against 0 before. This applies to every SM70 TurboMind warmup and also makes kernel_config gemm_lut_path and fp8_coordinated_tuning reach the warmup, which before silently used the defaults; it matters only when kernel_config sets native fields.
   - Every shape the warmup does not cover is measured on the first request that brings it. On DeepSeek-V4 the last prefill chunk has n mod 1020 tokens, so about one request in 16 met a new size: +1.4 .. 3.7 s once per size and boot (6 of 24 prompt lengths around 35.7k tokens), and the kernel choice depended on the request order. With the switch on, the device is marked final after the warmup through the new native op sm70_gemm_freeze_tuning: dense and MoE dispatch then return kReuse within the tuning limits (the next larger measured plan, else the heuristic) and never kMeasure. The freeze applies to every tuned TurboMind family on the device (FP16, AWQ, FP8, MXFP4, NVFP4).
   - Without the switch (main's default), the grouped path behaves like the dense shapes today: grouped sizes outside the warmup set (3, 5, 7, 9-15 tokens per group) are measured on their first use. In one single-boot probe without the freeze, a 35,711-token prompt (11 tokens per group in its last chunk) took 12.8 s instead of 7.0 s once.
3. Tests (ad925b180): 14 tests in tests/quantization/test_sm70_warmup.py still assumed the old convention (fakes without the native_policy keyword, exact keyword comparison, warmup.envs which no longer exists) and failed on main before this work; they now accept the keyword (one asserts that it is passed) and set the tuning through the captured config. New tests check that the warmup runs inside the worker's config and that the plan is frozen only with the switch on. gguf_grouped_prefill_gemm_sm70_out is registered with a native_policy argument but was missing from CONFIGURED_OPERATORS, which test_packaged_native_policy_schemas_parse_before_build checks; it is listed now, together with the new sm70_gemm_freeze_tuning.

Not a duplicate: I searched the open PRs for wo_a, "grouped GEMM", freeze_tuning, fp8_tune_small_shapes and sm70_awq_warmup, and compared the changed files with the complete file lists of all open PRs.
- #1187 (mine) defaults the FP8 small-shape tuning to off. It decides whether the grouped launch is tuned by default, and it adds a test right after test_fp8_explicit_lut_reuse_allows_dynamic_cache_import, which this PR edits, so the second to land needs a small rebase. It touches neither the grouped path nor the warmup policy.
- #903 (a larger SM70 MTP integration on an older main) also adapts test_sm70_warmup.py to the native_policy keyword; the two test repairs overlap. It does not touch sm70_awq_warmup, the config around the warmup or the freeze; its sm70_runtime.py change is in another class.
- #1167 edits gemm_runtime.cpp in a separate hunk of the same TurboMindRuntime struct.
- #828 (grouped page4 route for QSA DCP prefill) also edits _warmup_fp8_dense_layers in awq_sm70_warmup.py (warmup of the block-QPN8 TurboMind copy), in another part of the function; the second to land may need a small rebase. It does not touch the warmup's config context, the grouped decode sizes or the freeze.
- #822, #890, #897, #953, #1092, #1102, #1125, #1235 and #1243 share only the operator lists in csrc/ops.h, csrc/torch_bindings.cpp and vllm/_sm70_ops.py. #903, #1167, #1216, #1217, #1218, #1227, #1235, #1245, #1260 and #1271 also regenerate docs/design/architecture/runtime_reference.md.

AI assistance was used for this change. I reviewed every line and ran the tests and measurements below.

## Test Plan

Native build of this branch (in-place), then:
python -m pytest tests/quantization/test_sm70_warmup.py tests/quantization/test_sm70_native_policy.py tests/quantization/test_sm70_native_cache.py tests/quantization/test_sm70_fp8_prefill_exact_dense.py tests/config/test_runtime_default_ownership.py tests/config/test_unused_kernel_policy_hash.py, one process per file, on a Tesla V100.
pre-commit run --from-ref origin/main --to-ref HEAD; pre-commit run mypy-3.10 --hook-stage manual --from-ref origin/main --to-ref HEAD

## Path coverage

| Path / provider | Format | Key conditions (shape, TP/EP, graph, capability) | Expected selection / fallback | Evidence (source / static / executed / measured) | Unverified scope |
| --- | --- | --- | --- | --- | --- |
| grouped BMM (DeepSeek-V4 wo_a) on SM70 | block FP8 | up to 16 tokens per group | one grouped launch (fp8_moe_gemm_sm70_per_expert_dispatch_out) | executed: test_sm70_warmup.py; measured: CUDA graphs, model A/B | DeepSeek-V4 at TP2/TP4 (4 or 2 groups per rank) not run |
| same | block FP8 | more than 16 tokens per group, or fp8_grouped_bmm_decode=false | per-group loop as before | executed: test_sm70_warmup.py | none |
| freeze_tuning_after_warmup=true | every tuned TurboMind family | after warmup | kReuse within the tuning limits, no measuring in serving | measured: journal "tuning frozen after warmup: 14 measured shapes final" per V100 stage, 24 prompt lengths; executed: the Python wrapper only (no native unit test) | families other than FP8 dense and MoE not measured |
| same, switch off (default) | any | any | measuring on first use as before | source; executed: tests | none |
| sm70_awq_warmup | n/a | worker warmup, kernel_config native fields set | runs in the worker's config and policy partition | measured: TM_GEMM_TRACE policy=measure, measured=1 during warmup (policy=default, measured=0 before) | other models' TurboMind warmups not measured |

## Acceleration and benchmark contract (required for performance changes)

- Default enabled or opt-in: the grouped wo_a path is on by default for SM70; freeze_tuning_after_warmup is opt-in (default off).
- Required CLI options and environment switches: reference arm --kernel-config '{"triton_sm7x_tensor_cores":true}' (the small-shape tuning off, the default of our fork with #1187); changed arm --kernel-config '{"triton_sm7x_tensor_cores":true,"sm70_fp8":{"native":{"fp8_tune_small_shapes":true}},"sm70_runtime":{"freeze_tuning_after_warmup":true}}'. The A/B therefore compares (old code, tuning off) with (grouped path, tuning on, freeze); the comparison against main's default (tuning on, freeze off) was not run. Entry: DeepSeek-V4-Flash PP=5 on three Tesla V100 and two Quadro RTX 8000, DSpark with 5 speculative tokens, --kv-cache-dtype fp8, --max-num-batched-tokens 1024, VLLM_USE_V2_MODEL_RUNNER=1; all other options identical in both arms.
- KV cache dtype used for the benchmark: fp8.
- Wheel SHA or source commit: fork builds (github.com/Peuqui/1Cat-vLLM): reference f21d1c564, changed 1addd22db (the reference plus four fork commits whose changes form the first two commits here, tag verified-2026-10-24-tuning-freeze). The commits of this branch are rebased onto main 6a143197e; the branch was built and unit-tested, not booted with the full model.
- PYTHONPATH, source overlays, or external native libraries used: each arm's fork tree and the same patched Triton 3.6 build on PYTHONPATH; the Triton build is not part of this PR.
- User-entry route-hit, speed, and output-quality evidence: see Test Result.
- Existing kernel/backend registry and capability-rejection reasons: grouped path selection stays in the SM70 FP8 linear kernel; no registry change.
- Per-engine configuration and deprecated-variable compatibility: new KernelConfig field sm70_runtime.freeze_tuning_after_warmup (default False); no environment variable.
- New/changed environment metadata and generated reference check: docs/design/architecture/runtime_reference.md regenerated with python -m tools.generate_architecture_reference --write (the new field moves its line anchors); --check passes; tools.generate_env_reference --check unchanged.
- Route snapshot command and intentional changed rows: none.
- SM70 registered variables / unregistered reads / model-parameter restrictions, before -> after: no change.

## Test Result

Kernel (V100, DeepSeek-V4 layer 12 weights, CUDA graph, us per layer, eight GEMMs / one grouped call, lower is better): 1 token 204.3 / 56.0, 6 tokens 195.4 / 125.4; with fp8_tune_small_shapes 6 tokens 161.3 / 50.5, 8 tokens 156.5 / 54.7, 16 tokens 184.4 / 56.9. Error against FP32 on the dequantized weights: RMS 2.1e-4 in every case.

DeepSeek-V4-Flash, A/B/A/B through the server entry (reference a/b -> changed a/b, one boot each after an uncounted cold boot; ms and s lower is better, tok/s higher is better):
- Decode step: 35.7k-token prompt 66.9/67.2 -> 63.0/63.2 ms; prose 59/59 -> 55/56; code 62/62 -> 58/58; edit 67/67 -> 63/63; at 178k context 81.3/82.2 -> 78.9/78.5; at 287k 88.3/88.6 -> 84.4/84.6.
- Prefill: 35.7k 7.0 s in all arms, 178k 41.07/41.12 -> 41.13/41.13 s, 287k 73.60/73.56 -> 73.58/73.62 s. 24 prompt lengths (seeds 6200-6223) after one boot of the changed build: all 7.0-7.1 s; before the warmup fix 6 of 24 took +1.4 .. 3.7 s on their first request.
- Memory peak per V100 stage -36 / -16 / -36 MiB, RTX unchanged.
- The eight quality answers of the changed arm were read and are all correct; with the tuning on, greedy texts differ from the reference at single positions (another split-K choice in the tuned grouped GEMM).
Split of the decode gain (single boots, decode step in ms, prose / code / edit / 178k / 287k): reference 59 / 62 / 67 / 82.2 / 88.4; grouped path alone with the tuning off 57 / 60 / 65 / 79.8 / 85.9 (greedy text, token count and acceptance identical to the reference in two boots); with the tuning on and the freeze 55 / 58.5 / 63 / 78.0 / 84.4.
Draft acceptance (one boot per arm, sampling temperature 1.0, top_k 40, top_p 1.0, min_p 0.1; rounds 6-17 of each request, paired per request): prose 2.77 -> 2.78, english 2.90 -> 2.87, code 4.59 -> 4.60, edit 5.67 -> 5.68 tokens per round (all |t| <= 1.1); tok/s +4.0 .. +6.2 %.

Tests on a Tesla V100, native in-place build of this branch (ad925b180), one process per file:
- tests/quantization/test_sm70_warmup.py: 29 passed
- tests/quantization/test_sm70_native_policy.py: 91 passed
- tests/quantization/test_sm70_native_cache.py: 3 passed
- tests/quantization/test_sm70_fp8_prefill_exact_dense.py: 24 passed
- tests/config/test_runtime_default_ownership.py: 31 passed
- tests/config/test_unused_kernel_policy_hash.py: 6 passed
The same card on main 6a143197e (in-place build; the tree also carried the Python-only #1272, which touches neither file): test_sm70_warmup.py 14 failed, 11 passed; test_sm70_native_policy.py 1 failed (test_packaged_native_policy_schemas_parse_before_build), 90 passed.
pre-commit run --from-ref origin/main --to-ref HEAD: every hook passes except check-layering, which reports vllm/model_executor/layers/fused_moe/runner/moe_runner.py (platform coupling 37 -> 39); this PR does not touch that file and main 6a143197e reports the same. pre-commit run mypy-3.10 --hook-stage manual: passed. The pre-commit CI job of this PR fails exactly like main's own run at 6a143197e (run 40009610580): mypy in vllm/model_executor/layers/quantization/gguf_turbomind_moe.py:609 and vllm/v1/kv_cache_interface.py:156-158, and the check-layering count above; none of these files is touched here.

🤖 Generated with Claude Code (https://claude.com/claude-code)

https://claude.ai/code/session_0172gtUiymppDzziDqKfy78Q
