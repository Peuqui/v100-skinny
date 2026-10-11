ENTWURF 11.10.2026 — noch NICHT gesendet. Branch pr-breakable-eager-break-policy (Fix-Commit auf aktuellem 1Cat main, nach dem A/B anlegen)
Offen vor dem Senden: PR-Branch nur mit dem Fix auf frischem main bauen, Annahmeprobe darauf (Zahlen unten ersetzen), Test + pre-commit + mypy auf dem Branch, main-SHA und Duplikatsuche unmittelbar vor dem Senden neu abfragen.
Titel: [Bugfix] Let eager breaks follow the engine's graph policy, not import time
---
## Purpose

Since #1137 DeepSeek-V4-Flash with DSpark accepts almost no drafts on SM70. On PP5 (3x Tesla V100 + 2x Quadro RTX 8000, DSpark, 5 speculative tokens, greedy) the mean acceptance length dropped from about 3.0 to 1.00-1.04 and a 300-token decode from 38-40 to 13.9 tok/s. On the same build the generated text is the same with and without breakable capture, so only throughput shows it.

Cause: `eager_break_during_capture` (vllm/compilation/breakable_cudagraph.py) decides at decoration whether to wrap, via `is_breakable_cudagraph_enabled()`. Decoration runs at import. Before #1137 that read `VLLM_USE_BREAKABLE_CUDAGRAPH`, which the SM70/DeepSeek V4 runtime defaults had already written into the process environment. Since #1137 the defaults go into the engine's graph policy instead (`vllm/platforms/runtime_defaults.py`, `vllm/model_executor/models/runtime_defaults.py`), and at import no engine policy exists yet: `capture_execution_policy` falls back to `_standalone_policy`, which reads "off". So the MLA attention, sparse indexer and SM70 FP8 MoE ops stay unwrapped, while `cudagraph_dispatcher` and `gpu_model_runner` read the engine policy at run time ("on") and capture with breakable graphs. The ops are then recorded inside the graph segments instead of running eagerly between them.

Evidence:
- First-parent bisect between 2164365ab (good) and 16628e2f0 (bad), DSpark acceptance as the criterion: ad19d7ad1 good (mean acceptance length 2.94-3.07), 51c0f9720 (merge of #1137) bad (1.01-1.06).
- Worker environments of a good and a bad build differ in 14 SM70 variables the old code exported. Setting them on main 16628e2f0 in two groups: only the group with `VLLM_USE_BREAKABLE_CUDAGRAPH` restores acceptance; `VLLM_USE_BREAKABLE_CUDAGRAPH=1` alone restores it (2.97-3.09, 38.2-39.6 tok/s).

This PR makes the decorator always wrap. The wrapper already calls the op directly when no `BreakableCUDAGraphCapture` is active, and the runner opens one only when the engine's policy enables breakable capture, so the decision now follows the engine policy. Engines without breakable capture pay one wrapper call and a thread-local lookup per decorated op call outside graphs.

Not a duplicate: gh pr list --state all --search "breakable", "eager_break_during_capture", "VLLM_USE_BREAKABLE_CUDAGRAPH", "breakable cudagraph": no open PR; #727 (closed) concerned mode=NONE in the Flash-V100 graph policy.

AI assistance was used for this change. I reviewed every line and ran the tests and measurements below.

## Test Plan

```bash
pytest tests/v1/cudagraph/test_breakable_cudagraph.py
pre-commit run --files vllm/compilation/breakable_cudagraph.py tests/v1/cudagraph/test_breakable_cudagraph.py
pre-commit run mypy-3.10 --hook-stage manual --files vllm/compilation/breakable_cudagraph.py
```

`test_decorator_applied_before_policy_still_breaks` decorates an op while the standalone policy reads "off" and checks that it still breaks inside a capture and replays correctly; it fails without the change.

DeepSeek-V4-Flash PP5 with DSpark (5 draft tokens), the same 300-token greedy prompt twice per boot, acceptance from the SpecDecoding metrics log.

## Acceleration and benchmark contract (required for performance changes)

Restores the DSpark acceptance of the state before #1137; no new fast path.

## Test Result

- tests/v1/cudagraph/test_breakable_cudagraph.py: 13 passed with the change; without it the new case fails (1 failed, 12 passed). Tesla V100.
- DeepSeek-V4-Flash PP5, DSpark: main 16628e2f0 mean acceptance length 1.00-1.04, 13.9 tok/s; [PR-Branch: ZAHLEN EINSETZEN]; for reference, ad19d7ad1 (last good first-parent commit before #1137) 2.94-3.07, 39.2-40.9 tok/s; 51c0f9720 (#1137) 1.01-1.06, 14.3-14.4 tok/s. On the same build the output text is the same with and without the change.
- pre-commit and mypy (manual stage) pass.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
