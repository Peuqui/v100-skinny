# [Bug]: PP + async scheduling + spec decode: output_token_ids trim only runs on the last PP rank, desynchronizing the ranks until NCCL deadlocks

## Summary

Verified present in current `main` (checked 2026-08-28: the `elif` sits at
line 1425 of `vllm/v1/worker/gpu_model_runner.py`) and in v0.27.1, where we
hit and debugged it. `_update_states()` contains:

```python
if not is_last_rank:
    ...
    # optimistic extend of req_state.output_token_ids
    # (assumes all k spec tokens accepted)
    ...
elif num_output_tokens < len(req_state.output_token_ids):
    # Some output tokens were discarded due to a sync-KV-load
    # failure, or output_token_ids was inflated by the optimistic
    # extend above (async spec decode). Align the cached state.
    del req_state.output_token_ids[num_output_tokens:]
```

The trim branch is an `elif`, so it only ever executes on the **last** PP
rank. But the optimistic extend it is meant to correct runs on the
**non-last** ranks. With pipeline parallelism, async scheduling and
speculative decoding active at the same time, the non-last ranks accumulate
an unbounded overshoot in `req_state.output_token_ids`.

## Failure chain (observed)

1. Non-last rank extends output_token_ids optimistically each round
   (all k drafts assumed accepted), never trimmed back to num_output_tokens.
2. The inflated length feeds the discard/chunked-prefill bookkeeping;
   after enough rounds the discard mask flips on the non-last rank only.
3. The PP broadcast guard (`_is_all_reqs_chunked_prefill()`) now disagrees
   between ranks: the last rank keeps sending sampled-token broadcasts,
   the first rank skips the receives.
4. NCCL streams wedge after ~15 rounds; the engine appears deadlocked
   (`RPC call to sample_tokens timed out`).

Diagnosed by logging per-rank send/recv counters: last rank at send #10-15
while rank 0 reports "recv SKIP (chunked)".

## Fix

Make the trim an independent `if` so it runs on all ranks:

```python
if num_output_tokens < len(req_state.output_token_ids):
    del req_state.output_token_ids[num_output_tokens:]
    ...
```

The condition is false whenever there is nothing to trim, so behavior on
the last rank is unchanged. We have been running this fix under sustained
PP=2 + MTP k=7 load (Qwen3.5-family hybrid model, 4096-token generations,
chunked prefill) with exact send==recv counter symmetry afterwards.

## Environment where reproduced

- vLLM 0.27.1 code path (fork based on it; the affected function is
  upstream code)
- 2 PP stages, TP=2 per stage, --async-scheduling, MTP speculative
  decoding k=4..7
- Hardware: 5 external GPUs (2x Quadro RTX 8000 + 3x Tesla V100) on a
  GEM10 mini-PC via OCuLink/USB4, all PCIe Gen3 x4, NCCL_P2P_DISABLE=1
  (broken root-port P2P). None of this is required to hit the bug — it
  only extends how long the desync takes to wedge.
- Model: Qwen3.5/qwen4_exp-family hybrid (GDN linear attention + full
  attention); nothing in the failure chain is model-specific — any
  PP + async + spec workload should hit it once generations run long
  enough for the overshoot to flip the discard mask.

## Notes for maintainers

Happy to turn this into a PR with the one-line change if the analysis is
confirmed. The bug is invisible in the common configurations (no PP, or
spec decoding without async scheduling), which likely explains why it
survived: the trim comment even mentions "async spec decode" but the
control flow prevents it from ever running where the inflation happens.
