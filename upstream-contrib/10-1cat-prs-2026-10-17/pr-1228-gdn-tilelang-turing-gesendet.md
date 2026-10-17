## Purpose

On Turing the GDN prefill runs the Triton/FLA chunk chain, because `tilelang_fused_chunk_gdr_fwd_sm70` needs 86,016 B of dynamic shared memory per block and sm_75 allows at most 65,536 B (`shared_memory_per_block_optin` of a Quadro RTX 8000). Volta runs the fused TileLang kernel. For Qwen3.8-27B TP2 on two Quadro RTX 8000 and the RTX 8000 stages of Qwen3.8-Flash-Next, the Triton chain is the slow part of the GDN prefill.

This PR adds an sm_75 variant of the fused kernel that fits:

1. The transposed copy of k is dropped; the last GEMM reads `k_shared` with `transpose_A`.
2. The P tile reuses the A buffer.
3. A barrier at the end of the chunk loop protects the reused tiles before the next chunk overwrites them.

The sm70 kernel source is left unchanged (its compiled kernels stay valid in the TileLang cache). `fused_gdr_fwd` picks the variant from the shared memory limit of the device it runs on, and the GDN prefill selector now admits sm75 for fp16 models; other dtypes keep the Triton/FLA chain as before.

Not a duplicate: gh pr list --repo 1CatAI/1Cat-vLLM --state all --search "TileLang Turing", "FlashQLA", "GDN Turing", "sm75 GDN" and "fused_gdr_fwd" find no PR that runs the FlashQLA kernel on Turing; #1163 (merged) shares the GDN operand preparation and is part of main. No open PR touches the files of this PR, and it adds no merge conflict to any open PR (git merge-tree against each PR head).

AI assistance was used for this change. I reviewed every line and ran the tests and measurements below.

## Test Plan

```bash
pytest tests/kernels/test_flashqla_turing_fused_fwd.py tests/config/test_flashqla_policy_lifecycle.py tests/kernels/core/test_sm70_flashqla_runtime_compiler.py tests/kernels/mamba/test_flashqla_bound_policy.py tests/kernels/mamba/test_sm70_flashqla_extension.py tests/kernels/test_gdn_prefill_stages.py tests/kernels/test_gdn_execution_stages.py tests/kernels/test_sm70_gdn_mixed_spec_verify.py tests/model_executor/layers/test_gdn_execution_plan.py tests/model_executor/layers/test_gdn_prefill_backend_resolve.py tests/model_executor/layers/test_gdn_stage_lifecycle.py
pre-commit run --from-ref origin/main --to-ref HEAD
pre-commit run mypy-3.10 --hook-stage manual --from-ref origin/main --to-ref HEAD
```

New test (`tests/kernels/test_flashqla_turing_fused_fwd.py`): the sm_75 kernel against the Triton/FLA chain for 8/24 and 16/48 key/value heads and one, two and three sequences (lengths 2048; 17+2031; 63+1+1984), output and final state within 2e-3. It skips on devices that fit the sm70 kernel. `test_gdn_prefill_backend_resolve.py` now expects TileLang on Turing for fp16.

Kernel benchmark on a Quadro RTX 8000 (fp16, head dim 128): `chunk_gated_delta_rule_fwd` of this tree (the Triton/FLA chain, as main runs it on Turing) against `chunk_gated_delta_rule_fwd_sm70_tilelang` (the sm_75 variant), median of 20 calls.

Serving on our fork (main 6f3557c97 plus our commits). Arms: fix (fork commit ee261cc74 = this change and #1229 on top of the base) against base (fork commit 9a41a8e67), boots fix cold, base, fix, base, fix; the cold boot and the first long request after each boot are not counted. Per boot: three greedy reference prompts, four ~29k-token prompts, short greedy prompts, eight quality prompts and a long-context probe (184,134 and 249,950 tokens from repository documents, greedy, 400 output tokens). Qwen3.8-27B has no QSA layers, so its numbers show this PR alone; on Qwen3.8-Flash-Next both PRs act on the RTX 8000 stages.

## Path coverage

| Path / provider | Format | Key conditions (shape, TP/EP, graph, capability) | Expected selection / fallback | Evidence (source / static / executed / measured) | Unverified scope |
| --- | --- | --- | --- | --- | --- |
| GDN prefill, FlashQLA fused TileLang kernel | fp16 model, head dim 128 | sm_75 (opt-in shared memory below 86,016 B) | sm_75 variant; devices with at least 86,016 B keep the sm70 kernel; non-fp16 models keep Triton/FLA | executed + measured (kernel benchmark and serving on Quadro RTX 8000); unit tests on RTX 8000 and V100 | other head dims, bf16 (kept on Triton/FLA) |

## Acceleration and benchmark contract (required for performance changes)

- Default enabled or opt-in: default on sm_75 for fp16 models.
- Required CLI options and environment switches: none.
- KV cache dtype used for the benchmark: auto (fp16).
- Wheel SHA or source commit: fork commit ee261cc74 against 9a41a8e67 (both main 6f3557c97 plus our commits).
- PYTHONPATH, source overlays, or external native libraries used: fix arm PYTHONPATH on the fork tree with its own compile cache; same native build in both arms (this PR changes Python and the TileLang source only).
- User-entry route-hit, speed, and output-quality evidence: see Test Result.
- Existing kernel/backend registry and capability-rejection reasons: the GDN prefill selector no longer rejects sm75 fp16 for FlashQLA.
- Per-engine configuration and deprecated-variable compatibility: no new fields.
- New/changed environment metadata and generated reference check: none.
- Route snapshot command and intentional changed rows: none.
- SM70 registered variables / unregistered reads / model-parameter restrictions, before -> after: unchanged.

## Test Result

- On a Quadro RTX 8000 (main 6f3557c97 plus this PR): 78 passed, 56 skipped. Main without this PR: 71 passed, 56 skipped (the difference is the six cases of the new test and one more case in test_gdn_prefill_backend_resolve.py). On a Tesla V100: 128 passed, 1 skipped against 127 passed on main (the new test module skips on devices that fit the sm70 kernel). pre-commit and mypy-3.10 pass.

- Kernel benchmark, ms per call, Triton/FLA chain as on main -> sm_75 TileLang kernel: 8/24 key/value heads, 2,048 tokens 9.688 -> 1.143; 17+2,031 tokens 9.959 -> 1.116; 16/48 heads, 1,500+548 tokens 22.077 -> 1.472; 63+1+1,984 tokens 21.684 -> 1.605; 8,192 tokens 82.423 -> 5.175. Maximum relative difference of output and final state 1.02e-3.

- Qwen3.8-27B TP2 on two Quadro RTX 8000 (this PR alone), time to first token in s, fix (two boots) against base (two boots): ~29k tokens 35.3-35.4 against 38.5-38.6, 184,134 tokens 358.6 against 378.7-378.8, 249,950 tokens 565.1-565.2 against 592.6. Decode unchanged (~29k 38-39 ms, 184,134 tokens 55.3 ms, 249,950 tokens 63.2 ms per step in both arms). The three greedy reference prompts are identical in all five boots and the eight quality prompts are word-identical to the base.
- Qwen3.8-Flash-Next-180B-A4B (with #1229): TP2xPP2 (RTX 8000 pair and V100 pair) ~29k tokens 14.8 s against 17.6-17.7 s, 184,134 tokens 126.3-129.2 s against 178.9-194.1 s, 249,950 tokens 157.0-162.4 s against 253.3-275.7 s; PP4 ~29k tokens 10.2-10.6 s against 11.6 s (its long-context times vary with the page cache of the PLE disk tier in both arms). Decode step time in the long-context probe (TP2xPP2): 54.1-55.4 ms against 53.4-53.6 ms at 184,134 tokens, 58.6-60.5 ms against 58.0-59.0 ms at 249,950 tokens. A separate decode A/B with 20 sampled code answers per boot (two boots per arm, TP2xPP2) gave 77.3 and 76.9 tok/s against 77.0 and 76.9 tok/s, 51.0 ms median step time in all four boots, acceptance length 3.93 against 3.91.
- Flash-Next outputs: greedy text differs between boots in both arms, as it does on main with fresh compile caches. A hallucination probe (a non-existent person, temperature 1.0, 20 draws per arm) was handled equally often correctly with and without the two PRs (PP4 13 and 14 of 20, TP2xPP2 18 and 19 of 20).

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01GJxVXkjSHhfq2D71DoYSK7
