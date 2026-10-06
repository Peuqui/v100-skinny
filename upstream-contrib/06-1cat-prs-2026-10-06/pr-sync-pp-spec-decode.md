ENTWURF (ZURÜCKGEHALTEN: Flash-Next PP4 sync liefert kaputte erste Token, siehe Gedächtnis) — Branch pr-sync-pp-spec-decode (3c0cdb4fe auf 1Cat main bc01f2899)
Titel: [Bugfix][PP][Spec Decode] Make sync scheduling work with speculative decoding under pipeline parallelism
---
## Purpose

With pipeline parallelism and speculative decoding, `--no-async-scheduling` cannot serve a single request. Every PP-specific spec-decode path in the model runner (`_pp_broadcast_prev_sampled_token_ids`, `_pp_broadcast_draft_token_ids`, `_pp_receive_prev_sampled_token_ids_to_input_batch`) runs only under async scheduling, and the sync path has two gaps:

1. The batch queue keeps several steps in flight. In sync mode `post_step()` hands a step's drafts to the scheduler right after the step was enqueued, before its sampled token is applied. The next `schedule()` then computes `num_tokens + len(drafts) - num_computed_tokens` = `len(drafts)` new tokens for a request whose previous step is still in flight, and sends its drafts without the token they follow. The worker fails `assert num_scheduled_tokens[req_idx] >= draft_len + 1` in `_prepare_inputs`; the other ranks then wait in gloo `recv` until the 30-minute timeout.
2. The scheduler sends non-last ranks only the tokens from `num_computed_tokens` on (`scheduler.py`, the `use_pp and not async_scheduling` branch). The drafts accepted in the previous step come before them, and the first-stage rank had only fed them as drafts, so its token positions shift by the number of accepted drafts and the output degrades into repeated fragments after a few tokens.

This PR

- keeps a request with drafts out of a sync schedule while it has tokens in flight (`num_in_flight_tokens > 0`). Requests without drafts are unaffected, so prefill chunks of one request still pipeline across the stages;
- lets non-last ranks restore the accepted drafts: they are a prefix of the drafts that rank fed in its previous step, which it now keeps per request.

Async scheduling is unchanged.

Not a duplicate: no open PR touches sync scheduling with PP and speculative decoding (gh pr list --state open --search "pipeline parallel spec decode sync": none; "num_in_flight_tokens" and "no-async-scheduling" find #956, Mamba align-mode blocks under async scheduling, which only reads `num_in_flight_tokens` in its description, and #629, C32 serving; both unrelated, and this branch merges cleanly with #956 and #903, which also edits scheduler.py). #852 reports sync faster than async at TP4 x PP2 without speculative decoding; this PR is about the speculative case, which did not run at all.

AI assistance was used for this change. I reviewed every line and ran the tests and measurements below.

## Test Plan

```bash
pytest tests/v1/core/test_scheduler.py
pre-commit run --files tests/v1/core/test_scheduler.py vllm/v1/core/sched/scheduler.py vllm/v1/worker/gpu_model_runner.py
pre-commit run mypy-3.10 --hook-stage manual --files tests/v1/core/test_scheduler.py vllm/v1/core/sched/scheduler.py vllm/v1/worker/gpu_model_runner.py
```

`test_sync_pp_spec_decode_waits_for_in_flight_step` drives the scheduler through the failing sequence (prefill, a plain decode step in flight, its drafts arriving, schedule again).

## Acceleration and benchmark contract (required for performance changes)

Not a performance change: the sync path with PP and speculative decoding did not run before. Numbers for the #852 question are under Test Result.

## Test Result

- The new test passes; on main without the change it fails exactly as the server did: the request is scheduled with its 3 drafts alone (`{'0': 3}`).
- `tests/v1/core/test_scheduler.py` on main bc01f2899 with this PR: 107 passed and 1 failed on Tesla V100-PCIE-32GB and on Quadro RTX 8000 (Python change; extensions built from source, sm_70 only). The one case that fails, `test_schedule_partial_requests`, fails identically on main; #1005 repairs it.
- pre-commit and mypy (manual stage) pass.
- DeepSeek-V4-Flash at PP5 on 3x Tesla V100-PCIE-32GB + 2x Quadro RTX 8000, DSpark with 5 speculative tokens, `--no-async-scheduling`: without the change the first request fails as above (two boots); with the first part only it no longer fails but the output degrades after a few tokens; with both parts all three greedy prompts and all eight quality prompts give the same text as async scheduling on the same build (quality prompts also read by hand). FLASHNEXT_SYNC_RESULT
- For #852-style questions on this rig (DeepSeek-V4-Flash PP5, sync vs async, same build, three boots each, lower is better): prefill of a 35.7k-token prompt 61.1-61.3 s vs 15.5-15.6 s, decode step at 35.7k context 83-85 ms vs 74-76 ms, short prompts 72-75 ms vs 70-73 ms; time to first token of short prompts 0.2 s vs 0.4 s. So async stays the better default here; this PR only makes the sync path usable.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
