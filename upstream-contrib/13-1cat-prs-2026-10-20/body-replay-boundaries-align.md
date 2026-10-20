## Purpose

`KVCacheCoordinator.get_replay_boundaries` returns the raw positions `(num_tokens - 1, num_tokens)` when no KV cache group has the EAGLE last-block drop, and block-aligned positions otherwise. Its two consumers treat the result differently: `reachable_block_mask` aligns each boundary before marking blocks, but `Scheduler._mamba_block_aligned_split` uses the values directly as chunk stops (`start < stop < end`). With Mamba `align` prefix caching and sparse retention (`mamba_state_retention_interval` not None), every prompt of a model without EAGLE/MTP that takes the sparse replay boundaries (the default tail checkpoint; the dense boundaries under contention bypass them) therefore ends its prefill with an extra step: up to `num_tokens - 1`, then the last token alone (unless `num_tokens - 1` is itself a block end). The state materialized at the unaligned position `num_tokens - 1` can never be cached (Mamba states are cached at block ends), so the split costs one forward step per request and changes the chunking of the recurrent prefill without any benefit.

Example, 816-token Mamba blocks, 50-token prompt: the chunk ends were [49, 50]; with this change [50]. A 1662-token prompt: [1632, 1661, 1662] before, [1632, 1662] now.

This PR aligns both positions of the branch without an EAGLE drop, so the scheduler and the cache mask read the same block-aligned boundaries. The EAGLE branch is unchanged. `reachable_block_mask` already aligned the values, so the retained blocks are the same as before.

Found while porting vllm#61589 (lookahead block hashes): with lookahead hashes the EAGLE/MTP groups no longer take the drop, so the non-EAGLE branch became active for MTP models, and one of the three greedy reference outputs on Qwen3.8-Flash-Next changed in all six lookahead boots (three PP4, three TP2xPP2), although every prompt was far shorter than one cache block. One PP4 boot of the same build and compile cache with the lookahead hashes disabled reproduced all three references; with this fix, the lookahead build reproduced them in all four following boots (three PP4, one TP2xPP2).

Not a duplicate: #617 introduced the replay boundaries for cache retention, where reachable_block_mask aligns them; #754 also made them chunk stops in _mamba_block_aligned_split, which is where the raw non-EAGLE values split the prompt; #787 kept them for single requests. None of the 125 open PRs changes get_replay_boundaries or the chunk stops (checked the diffs of the seven that touch the scheduler, the coordinator or the Mamba manager), and no PR or issue mentions get_replay_boundaries or mamba_state_retention_interval.

AI assistance was used for this change. I reviewed every line and ran the tests below.

## Test Plan

```bash
pytest tests/v1/core/test_mamba_align_chunk_split.py tests/v1/core/test_mamba_sparse_retention.py
CUDA_VISIBLE_DEVICES=<two GPUs> pytest tests/v1/core
pre-commit run --from-ref origin/main --to-ref HEAD
pre-commit run mypy-3.10 --hook-stage manual --files vllm/v1/core/kv_cache_coordinator.py tests/v1/core/test_mamba_align_chunk_split.py
```

## Path coverage

| Path / provider | Format | Key conditions (shape, TP/EP, graph, capability) | Expected selection / fallback | Evidence (source / static / executed / measured) | Unverified scope |
| --- | --- | --- | --- | --- | --- |
| Mamba align prefill split, no EAGLE drop | hybrid GDN/Mamba models | prefix caching, sparse retention, `num_tokens - 1` not a block end | last chunk ends at the prompt end, no extra step; same retained blocks | executed: unit tests | end-to-end timing of models without MTP not measured |
| Mamba align prefill split, EAGLE drop | MTP/EAGLE models | same | unchanged | executed: unit tests | |

## Acceleration and benchmark contract (required for performance changes)

Not a performance change: it removes one prefill step per affected prompt; end-to-end timing was not measured. Python only; no new switches, fields or environment variables.

## Test Result

- `tests/v1/core/test_mamba_align_chunk_split.py` and `tests/v1/core/test_mamba_sparse_retention.py` with this PR: 168 passed.
- The first file with main's `kv_cache_coordinator.py`: 3 failed, 34 passed. The failures are the three new cases without the EAGLE drop (lengths 50, 815, 1662), whose chunk ends contain `num_tokens - 1`; the three EAGLE cases pass on main as well.
- `tests/v1/core` with two visible Tesla V100: on main 6f3557c97 and with this PR the run stops at the same test, `test_scheduler_e2e.py::test_concurrent_partial_prefill`. Its EngineCore starts and then waits without progress (stopped after 6 and 8 minutes). Up to that test: main 614 passed, 1 failed; this PR 620 passed (the same 614 plus the six new cases), 1 failed. The failure in both is `test_async_scheduler.py::test_abort_request_when_structured_output_fsm_cannot_advance` (AttributeError: 'AsyncScheduler' object has no attribute 'mixed_prefill_budget'). With this PR, `test_scheduler_e2e.py` alone: 2 passed; the 31 tests after it (four files): 31 passed.
- pre-commit run --from-ref origin/main --to-ref HEAD and mypy-3.10: all hooks pass.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01YaG8VjXn5eN3Aq7eZrA3dp
