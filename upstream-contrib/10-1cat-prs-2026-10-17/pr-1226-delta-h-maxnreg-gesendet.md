## Purpose

On Turing (sm_75) ptxas gives `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`, the state recurrence of the FLA/Triton GDN prefill chain, only 64 registers per thread in all twelve configs of its autotune list (BV 32/64, 2/4 warps, 2-4 stages), with 2,928 to 8,688 bytes of stack per thread. Triton passes `maxnreg` on as `.maxnreg`, which lifts the cap: the same configs compiled with `maxnreg=255` get 255 registers and 1,768 to 5,064 bytes of stack (cuobjdump --dump-resource-usage of the cached sm_75 cubins of both variants).

This PR adds a `maxnreg=255` twin next to every delta_h config and lets the autotuner keep the faster one on each device. The engine-owned tuner (`create_chunk_kernels`) now reuses the module's config list instead of a copy, so both stay in sync. The results are bit-identical: maxnreg only changes register allocation. The kkt and chunk_o kernels keep their lists.

On Volta the GDN prefill of these models runs the fused FlashQLA kernel, so V100 serving does not reach this kernel; the unit tests below also run on a V100.

Not a duplicate: gh pr list --repo 1CatAI/1Cat-vLLM --state all --search "delta_h", "maxnreg", "register cap", "ptxas" and "chunk_gated_delta_rule_fwd" find no PR that touches the delta_h configs or register limits. The open PRs touching chunk_delta_h.py (#1020, #1023) change other parts. This PR adds no merge conflict to any open PR (git merge-tree against the head of each open PR that touches the same files; the ones that conflict do so with main already).

AI assistance was used for this change. I reviewed every line and ran the tests and measurements below.

## Test Plan

```bash
pytest tests/kernels/test_fla_delta_h_maxnreg.py tests/config/test_fla_schedule_lifecycle.py tests/kernels/test_fla_schedule_owner.py tests/kernels/test_gdn_execution_stages.py tests/kernels/test_gdn_prefill_stages.py tests/model_executor/test_fla_context_free_import.py
pre-commit run --from-ref origin/main --to-ref HEAD
pre-commit run mypy-3.10 --hook-stage manual --from-ref origin/main --to-ref HEAD
```

New tests (`tests/kernels/test_fla_delta_h_maxnreg.py`): every plain config has a maxnreg=255 twin, and for four configs the twin's outputs and final state are bit-identical to the plain config.

Kernel benchmark on a Quadro RTX 8000, Qwen3.8-27B TP2 shapes (8 key heads, 24 value heads, K=V=128, one 2048-token prefill step): autotune over the twelve plain configs (as on main) against the full list (this PR), median of 20 calls after warm-up.

Serving on our fork (main 6f3557c97 plus our commits). Arms: fix (fork commit 9e81ffeb0: this change, #1219 and a test repair on top of the base) against base (fork commit df0941355), boots fix cold, base, fix, base, fix; the cold boot and the first long request after each boot are not counted. Per boot: four ~29k-token prompts, short greedy prompts, eight quality prompts and a long-context probe (184,133 and 249,950 tokens from repository documents, greedy, 400 output tokens).

## Path coverage

| Path / provider | Format | Key conditions (shape, TP/EP, graph, capability) | Expected selection / fallback | Evidence (source / static / executed / measured) | Unverified scope |
| --- | --- | --- | --- | --- | --- |
| FLA chunk delta_h (Triton GDN prefill chain) | fp16 activations, fp32 state | sm_75, K=V=128, BT=64, Qwen3.8-27B and Qwen3.8-Flash-Next GDN shapes | the autotuner picks the faster of each config and its maxnreg=255 twin | executed + measured (kernel benchmark and serving on Quadro RTX 8000); unit tests on V100 and RTX 8000 | other GPU families (the list grows; the tuner picks per device) |

## Acceleration and benchmark contract (required for performance changes)

- Default enabled or opt-in: default; additional autotune candidates only.
- Required CLI options and environment switches: none.
- KV cache dtype used for the benchmark: auto (fp16).
- Wheel SHA or source commit: fork commit 9e81ffeb0 against df0941355 (both main 6f3557c97 plus our commits).
- PYTHONPATH, source overlays, or external native libraries used: fix arm PYTHONPATH on the fork tree with its own compile cache; same native build in both arms (this PR changes Python only).
- User-entry route-hit, speed, and output-quality evidence: see Test Result.
- Existing kernel/backend registry and capability-rejection reasons: unchanged.
- Per-engine configuration and deprecated-variable compatibility: no new fields.
- New/changed environment metadata and generated reference check: none.
- Route snapshot command and intentional changed rows: none.
- SM70 registered variables / unregistered reads / model-parameter restrictions, before -> after: unchanged.

## Test Result

- On a Quadro RTX 8000 and on a Tesla V100 (main 6f3557c97 plus this PR): 11 passed, 33 skipped and 44 passed. Main without this PR: 6 passed, 33 skipped and 39 passed (the difference is the five new tests). pre-commit and mypy-3.10 pass.

- Kernel benchmark: 6.91 ms per call when autotuned over the twelve plain configs against 3.01 ms over the full list; outputs and final state bit-identical.

- Prefill time to first token in s, fix (two boots) against base (two boots): Qwen3.8-27B TP2 on two Quadro RTX 8000: ~29k tokens 38.5-38.6 against 41.2-41.3, 184,133 tokens 378.8 against 395.6-395.7, 249,950 tokens 592.7 against 615.5-615.6. Qwen3.8-Flash-Next-180B-A4B PP4 (two RTX 8000 and two V100 stages): ~29k tokens 11.6-11.9 against 12.5-12.6, 249,950 tokens 139.9-140.1 against 152.4-154.9 (184,133 tokens varies with the page cache of the PLE disk tier in both arms: 93.7-100.1 against 110.3-176.0). Flash-Next TP2xPP2 (RTX 8000 pair and V100 pair): ~29k tokens 17.6-17.7 against 18.6-18.7, 184,133 tokens 187.1-194.5 against 199.5-200.6, 249,950 tokens 253.4-262.6 against 291.4-294.8.
- Decode step time and acceptance length unchanged in all three entries (e.g. Qwen3.8-27B at 249,950 tokens 182.4-182.5 ms per step in both arms).
- Outputs: the kernel results are bit-identical (unit test above). Qwen3.8-27B greedy and quality answers were word-identical in all five boots. Flash-Next greedy text differs between boots in both arms, as it does on main with fresh compile caches, so it was not used as a criterion.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01GJxVXkjSHhfq2D71DoYSK7
