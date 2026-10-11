ENTWURF 11.10.2026 abends — noch NICHT gesendet. Branch pr-dspark-policy-regressions (Worktree 1Cat-vLLM-bisect) auf 1Cat main be3722bc0:
e53ca2a69 Bugfix, 9a9a2205b + cff4c875c Perf (Inhalt bytegleich mit c2b8e2e45, Bugfix-Nachricht präzisiert).
Quellen der Zahlen: pr_probe_dspark_2026-10-11.log (Branch-Messung, ein Bau, drei Stände), pr_text_probe_2026-10-11.log (zweite
Boots main/PR + Texte probe_texts/), bisect_dspark_2026-10-11.log, specprobe_envAB/envC_2026-10-11.log, ab_fmnext_2026-10-11.log
(fmn3ds = Fork mit Commit 1), ab_fmnext_ds5_2026-10-11.log (fmn5ds = Fork mit allen drei), nsys ds-decode-fmn/-fix/-trace,
pyspy_pyspy_1/2.raw. Vor dem Senden: main-SHA, offene PRs, Duplikatsuche neu abfragen; Branch auf fork pushen; Live-Text prüfen.
Titel: [Bugfix][DeepSeek-V4] Restore DSpark decode on SM70 after the per-engine policy migration
---
## Purpose

Since the per-engine policy migration (#1137 and the following binding commits), DeepSeek-V4-Flash with DSpark on SM70 accepts almost no drafts and spends extra host time per decode step. This PR has three commits, one per cause; all three are Python-only.

1. `eager_break_during_capture` (vllm/compilation/breakable_cudagraph.py) decided at decoration whether to wrap, via `is_breakable_cudagraph_enabled()`. Decoration runs at import. Before #1137 that read `VLLM_USE_BREAKABLE_CUDAGRAPH`, which the SM70/DeepSeek V4 runtime defaults had already written into the process environment. Since #1137 those defaults go into the engine's graph policy, and at import no engine policy exists yet: `capture_execution_policy` falls back to `_standalone_policy`, which reads "off". The MLA attention, sparse indexer and SM70 FP8 MoE ops therefore stay unwrapped, while `cudagraph_dispatcher` and `gpu_model_runner` read the engine policy at run time ("on") and capture breakable graphs, so the ops are recorded inside the graph segments instead of running eagerly between them. The decorator now always wraps; the wrapper already calls the op directly when no `BreakableCUDAGraphCapture` is active, and the runner opens one only when the engine policy enables breakable capture.
2. `_dflash_ddtree_target_forward_profile_scope` reads two diagnostic switches (NVTX range, cudaProfilerApi step) on every speculative decode step. It opens before `set_forward_context`, where neither a forward context nor a current VllmConfig exists, so `capture_runtime_trace()` fell back to `_standalone_runtime_trace()` and built and resolved a new `RuntimeTraceConfig` twice per step. The runner now passes its own `observability_config.runtime_trace`, the owner the forward context lends.
3. DSpark precomputes the drafter's context KV outside the forward context, and `sm70_qnorm_rope_kv_fp8_insert` looked up `sparse_policy()` on every call, which there builds and resolves a standalone `Sm70SparseConfig` for every drafter layer on every step. The drafter now keeps its engine's sparse policy and passes it; other callers keep the lookup.

How the causes were found, all on DeepSeek-V4-Flash PP5 (3x Tesla V100 + 2x Quadro RTX 8000, DSpark, 5 speculative tokens, greedy):
- A first-parent bisect between 2164365ab and 16628e2f0 with DSpark acceptance as the criterion: ad19d7ad1 is the last good commit (mean acceptance length 2.94-3.07, 39.2-40.9 tok/s), 51c0f9720 (merge of #1137) the first bad one (1.01-1.06, 14.3-14.4 tok/s).
- The workers of our previous build (based on 2164365ab) carry 15 `VLLM_*` variables that the runtime defaults exported and that main 16628e2f0 no longer sets in the process environment. Setting them on main 16628e2f0 (its Python tree with the native libraries of our fork build) in two groups (4 GEMM variables, 11 others), only the group containing `VLLM_USE_BREAKABLE_CUDAGRAPH` restored acceptance; that variable alone restored it (2.97-3.09, 38.2-39.6 tok/s).
- With acceptance restored, on our fork (main be3722bc0 plus patches outside this path) compared with our previous build, nsys showed identical kernel times on all five GPUs but longer idle gaps between kernels; py-spy on the first pipeline rank put 8.8 % of its CPU samples into the `RuntimeTraceConfig` construction of cause 2, and nsys on the last rank showed about 120 us more idle time before each drafter `_sm70_qnorm_rope_kernel` (three per step), which is cause 3.

Not a duplicate: gh pr list --state all --search "breakable", "eager_break_during_capture", "VLLM_USE_BREAKABLE_CUDAGRAPH", "breakable cudagraph", "DSpark acceptance": no open PR on this; #727 (closed) concerned mode=NONE in the Flash-V100 graph policy.

AI assistance was used for this change. I reviewed every line and ran the tests and measurements below.

## Test Plan

```bash
pytest tests/v1/cudagraph/test_breakable_cudagraph.py tests/v1/worker/test_spec_target_profile_scope.py tests/kernels/test_deepseek_v4_sm70_qnorm_rope_kv_insert.py
pytest tests/kernels/attention/test_dflash2_context_pipeline.py tests/v1/spec_decode/test_dspark.py tests/config/test_placement_policy_lifecycle.py
pre-commit run --from-ref origin/main --to-ref HEAD
pre-commit run mypy-3.10 --hook-stage manual --files <the seven changed files>
```

New cases: `test_decorator_applied_before_policy_still_breaks` decorates an op while the standalone policy reads "off" and checks that it still breaks inside a capture and replays correctly; `test_spec_target_profile_scope.py` checks that the scope follows the runner's trace and never builds a standalone one; `test_sm70_qnorm_kv_insert_uses_the_passed_engine_policy` checks that a passed policy decides the path without another lookup. Each new case fails without its commit.

DeepSeek-V4-Flash PP5 with DSpark: one native build of this branch serves all states (the changes are Python-only); per state one boot, the same 300-token greedy prompt twice, acceptance from the SpecDecoding metrics log.

## Acceleration and benchmark contract (required for performance changes)

Commit 1 restores DSpark acceptance; commits 2 and 3 remove host work per decode step. No new fast path and no change to the kernels that run.

## Test Result

- Tests above on a Tesla V100: 13, 4, 4, 5, 15 and 21 passed. Without the respective commit the new cases fail (1 of 13; 4 of 4; 2 of 4).
- pre-commit and mypy (manual stage) pass.
- DeepSeek-V4-Flash PP5, this branch, one build:
  - main be3722bc0: mean acceptance length 1.01-1.07, 13.9-14.1 tok/s (two boots)
  - main + commit 1: 2.95-3.07, 37.9-40.2 tok/s
  - main + all three commits: 2.95-3.19, 39.0-42.6 tok/s (two boots)
- Commits 2 and 3 on our fork (main be3722bc0 plus patches outside this path), A/B/A/B against our previous build (main 2164365ab plus our patches), decode tok/s:
  - with commit 1 only: prose 41.4-42.0 vs 42.5-43.0, code 61.3-64.2 vs 65.5-65.8, edit 70.8-71.2 vs 72.0-72.5
  - with all three: prose 42.5-42.7 vs 42.9-43.0, code 65.0-65.5 vs 65.3-65.6, edit 72.4-72.8 vs 72.3-73.1
  On that fork, where FP8 small-shape tuning is off, greedy output and eight quality answers were character-identical to the previous build in every run. On main, two boots of the same build gave different greedy texts, both with and without this change. The two texts I compared word by word (main and this branch, second boot each) diverge after about 140 words of the 300-token answer, and both read correctly. This matches the boot-to-boot variation #1006 describes for the timing-based FP8 small-shape tuning, which is on by default; I did not isolate it on this tree.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
