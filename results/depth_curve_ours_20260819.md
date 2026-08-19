# Our depth-cost curve — the drafter, not the verifier, prices depth

2026-08-19, 4x V100-SXM2-16GB, TP4, MML 16,384 (the fast plateau, so the
partition tax cannot leak in), GMU 0.88, greedy, thinking off, `prose`.
Two reps per arm; reps agree to 0.03 ms.

`prose` is deliberately the lowest-acceptance cell, so a round-time change is
close to the pure cost of depth rather than cost-minus-benefit. This is the
companion to `depth_cost_20260819.md`, which measured the same curve on
ninfer.

## The curve

| k | round | **verify** G0G1 | **drafter** G2G4 | tau | tok/round | tok/s |
|--:|------:|------:|------:|----:|----:|------:|
| 1 | 18.58 | 15.18 | 0.95 | 0.73 | 1.73 | 93.0 |
| 3 | 20.34 | 16.01 | 2.55 | 1.33 | 2.33 | **114.4** |
| 7 | 24.94 | 17.48 | 5.85 | 1.50 | 2.50 | 100.1 |

(k=7 row from the MML 16,384 arms of `mml_ladder` / `partition_fix_timing`,
same configuration. k=5 was not reached — the run was stopped deliberately.)

## The two slopes are separate, linear, and very different

- **verify: +0.383 ms per draft row** (0.415 then 0.368 across the two
  intervals). Verifying 8 rows costs **1.155x** verifying 2 — squarely
  inside the 1.05..1.16x band predicted from the `m8n8k4` tile issuing for
  8 rows whether or not all 8 carry work. **The tile-quantization claim is
  confirmed.**
- **drafter: +0.817 ms per step** (0.80 then 0.825). Strictly linear, with
  no tile effect at all: the proposer runs k *sequential* q=1 forward passes
  (`step3p5.py:390`, `for token_index in range(self.num_speculative_tokens - 1)`),
  each ending in a full `compute_logits` over the 248,320-token vocabulary.

**The drafter is 2.1x more expensive per token than the verify it feeds.**

## What this does to the headline claim

Round-level, ours is **1.07 ms per draft token**; ninfer's is **1.32**.
Least-squares fits:

    ours    round ~= 17.35 + 1.07k  ms
    ninfer  round ~= 13.28 + 1.32k  ms

So depth is **1.25x cheaper for us, not 3-8x**. The README's "we can
speculate 7 deep because it's free for us, not them" is true *of the
verifier* and not true of the round. The two lines cross at **k ~= 16.3** --
past ninfer's hard cap of 5, and near our k=15.

## Matched depth on prose: we lose

| k=3 vs draft=3 | round | tau | tok/round | tok/s |
|---|---:|---:|---:|---:|
| ours | 20.34 | 1.33 | 2.33 | 114.4 |
| ninfer | 17.39 | 1.37 | 2.37 | **136.3** |

**1.19x to them.** Tokens per round are essentially identical (2.33 vs 2.37)
and so is tau, so the drafters are comparable in quality — the entire gap is
engine round speed. This is the matched-depth row, and it is an honest loss.

## Our production default is wrong for low-acceptance work

Our own prose optimum is **k=3 (114.4)**, not the production k=7 (100.1) --
we give up **14%** on prose-like workloads. That mirrors ninfer's own
3-beats-5 result on the same cell. Depth belongs per-request, not in a boot
flag, on both engines.

## The drafter is the target

At k=7 the drafter is 5.85 ms of a 24.94 ms round (23%); at k=15 it projects
to ~12.4 ms while verify grows only via the tile. Every route to exploiting
depth -- the one axis where ninfer is structurally capped -- runs through
making a drafter step cheaper.

Composition of the 0.817 ms step, partly estimated:

| term | ms | basis |
|---|---:|---|
| lm_head `compute_logits` (248,320 x 5,120, vocab-parallel) | ~0.21 | 0.16 GiB/rank at ~750 GB/s |
| one MTP layer forward at q=1 | ~0.08 | bandwidth estimate |
| remainder (many tiny q=1 launches, norms, projection, argmax) | ~0.5 | residual |

**Not a lever: softmax.** With `use_local_argmax_reduction` (which we run)
the proposer path is `logits.argmax(dim=-1)` with no softmax
(`step3p5.py:264-269`). The known double-softmax cost is in the *target*
sampler, not the drafter.

### Ranked levers

1. **Graph the drafter chain with device-side metadata — 1.5-2.5 ms/round.**
   Already scoped in the residual ledger as outstanding. At q=1 every op is
   latency-bound, so k separate launch sequences is largely overhead. No
   accuracy cost. Obvious first move.
2. **Tree drafting — ~3.3 ms/round, and the move our cost shape argues for.**
   Verify costs +0.38 ms/row with 8 rows free per tile; drafting costs
   +0.82 ms/step. Today we spend 7 sequential steps to fill 7 nearly-free
   rows. A branching tree fills ~8 rows from ~3 steps: drafter 5.85 -> ~2.5
   ms at unchanged verify cost. Requires tree attention masks in the
   verifier and new acceptance logic — real work, not a flag.
3. **Draft-vocab shortlist on lm_head — up to ~1.5 ms/round at k=7.** Cuts
   the largest per-step tensor ~8x. Costs acceptance. Known-blocked: the
   fork's static-draft-vocab path requires probabilistic rejection while the
   flagship requires greedy, so it needs one-hot rejection taught. ninfer
   ships `--lm-head-draft`, which appears to attack the same term.

### Do first

Point the existing nsys per-phase kernel attribution at the **G2G4 window**.
It has only ever been run on verify. It would say directly whether 0.817 ms
is lm_head, layer compute, or launch gap — which decides between levers 1
and 3 instead of guessing. The estimate table above is the hypothesis it
should test.
