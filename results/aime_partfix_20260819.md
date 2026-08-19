# AIME f01 re-run with the decode-partition fix

2026-08-19, 4x V100-SXM2-16GB, TP4, MML 98,304, K=7, medium reasoning effort,
`VLLM_FLASH_V100_DECODE_PARTITION_SIZE=256`. Five fixed seeds, ninfer's
published sampling (temp 0.6, top-p 0.95, top-k 20), ninfer's metric
definitions.

Every AIME figure we held before today was measured at MML 98,304 *without*
the pin, i.e. on the slow plateau of `mml_ladder_20260819.md`.

## Result

| arm | completion tokens | tok/s | acceptance | tok/round | correct |
|---|---:|---:|---:|---:|---:|
| **real** — presence 1.0 | 1,512.6 ± 44.5 | **219.1 ± 5.9** | 69.9% ± 2.2 | 5.89 ± 0.15 | **5/5** |
| diagnostic — presence 0.0 | 1,649.8 ± 231.2 | 224.8 ± 5.3 | 70.8% ± 2.1 | 5.95 ± 0.15 | 5/5 |

Prior banked figure, same fixture and seeds, no pin: **206.8 ± 4.0**.

**+5.9% from one environment variable, with the answer unchanged (277) on
every seed.**

## On the two arms

`presence_penalty 1.0` is ninfer's published sampling and is therefore the
comparison number. `presence 0.0` is a **diagnostic only** and must not be
quoted as a result: it changes the sampler away from the protocol being
reproduced.

The diagnostic is nominally faster (224.8 vs 219.1) but the informative
difference is dispersion, not speed — completion length goes from
**1,512.6 ± 44.5 to 1,649.8 ± 231.2, a 5x increase in variance**. That is
the termination effect from
`presence_penalty_termination_20260819.md` appearing as instability in
reasoning length rather than as wrong answers: EOS is the one token a
presence penalty never penalises, so its relative logit climbs with trace
length and truncates traces at a length that depends on the trajectory.
Both arms answered 277 on all five seeds, so on this fixture it costs
consistency, not correctness.

## Comparison status

Against ninfer's 189.5 ± 4.3 this is **1.156x**, up from 1.09x. But that
ratio is *at their published `--draft-tokens 3` only*, and their published
depth is not their optimum (`depth_cost_20260819.md`). Their AIME acceptance
measures ~78% at depth 3 — far above prose's 45.6% — so depth 5 is expected
to help them here. The matched-depth and best-vs-best rows are pending that
sweep and this ratio should not be published before it lands.

## Caveat

f01 runs ~1,500 tokens, well inside the range where partition 256 is the
right pin. f15 and f30 run to tens of thousands of tokens and have **not**
been re-tested; partition 1024 is the default above 32,768 presumably
because it wins there. `DECODE_PARTITION=` is an override for exactly this
reason.

---

# Best-vs-best on AIME f01: parity, not a win

ninfer's AIME depth swept on our 5090, five seeds, their published sampling,
thinking on, ctx 32,768:

| ninfer depth | completion | tok/s | acceptance | tok/round | correct |
|---|---:|---:|---:|---:|---:|
| 3 (published) | 1,463.6 ± 237.7 | 189.6 ± 4.3 | 76.5% ± 2.4 | 3.30 ± 0.07 | 5/5 |
| **5 (their ceiling)** | 1,403.0 ± 253.2 | **214.7 ± 9.2** | 65.4% ± 3.7 | 4.27 ± 0.18 | 5/5 |

Depth 5 is worth **+13.2%** to them here: tokens/round rises 3.30 -> 4.27
(+29%), comfortably clearing the +14.3% round cost. AIME acceptance at depth
3 is 76.5%, far above prose's 45.6%, which is exactly the condition under
which depth pays.

The depth-3 row reproduces the independently banked 189.5 ± 4.3 from a
different boot on a different day.

## The three classes

| class | ours | theirs | ratio |
|---|---:|---:|---:|
| as first banked (their 3, our unpinned) | 206.8 ± 4.0 | 189.6 ± 4.3 | 1.09x |
| their published 3, our partition pin | 219.1 ± 5.9 | 189.6 ± 4.3 | 1.156x |
| **best-vs-best (their 5, our k=7)** | **219.1 ± 5.9** | **214.7 ± 9.2** | **1.02x** |

**At each engine's best configuration this is parity.** The intervals
overlap heavily; 1.02x is not a defensible margin at n=5.

## Mechanism, unchanged

We take **5.89 tokens per 26.9 ms round**; they take **4.27 per 19.9 ms**.
We are 38% slower per round and they are 28% shorter per round, and it
cancels. Every measurement today supports the same two-sided picture: their
round is cheap and steeply depth-priced (`13.28 + 1.32k` ms, hard-capped at
k=5); ours is expensive and nearly depth-flat.

## What may be claimed

- **Supported:** 4x V100 (2017, SM70, no FP8 or FP4 tensor cores) matches a
  single RTX 5090 on long-form reasoning at each engine's best setting, both
  5/5 correct on AIME 2026 f01.
- **Not supported:** any "beats a 5090" headline. The earlier 1.09x compared
  against ninfer's published depth 3, which costs them 13.2% on this
  workload.
- **Still open:** matched-depth (both engines at the same k) needs our own
  k-sweep, running now. And the checkpoints remain unmatched -- see the
  update in `headtohead_5090_20260819.md`.

---

# Seconds to answer — the metric we had and did not publish

Added 2026-08-19 after the v1.1 launch review identified this as the
strongest attack available against the launch:

> "You chose the metric you win. On seconds-to-a-correct-answer the single
> 5090 beats your four V100s, and your own head-to-head plan says you must
> report it. You didn't."

The criticism is well founded on process. `docs/v11_headtohead_plan.md`
mandates time-to-answer "because they point opposite ways when one side
writes a longer answer". Every own-arm table carries it; **no cross-engine
table did**. And `benchmarks/ninfer_repro.py` has banked `decode_seconds`
per seed, for both engines, the whole time. The number existed and was not
printed, in the direction that flattered us. The table below is computed
from those already-committed rows — no re-run was needed, which is the
point.

## Result, AIME f01, n=5, decode only

| arm | seconds to answer | completion tokens | correct |
|---|---:|---:|---:|
| ours, k=7, medium | **6.90 ± 0.30** | 1,513 ± 44 | 5/5 |
| ninfer, draft 3 (their published) | 7.73 ± 1.40 | 1,464 ± 238 | 5/5 |
| ninfer, draft 5 (their best) | **6.56 ± 1.34** | 1,403 ± 253 | 5/5 |

**Against their best depth their point estimate is 5.0% ahead of ours.**
Publish that. It does not overturn the conclusion already stated — the arms
are a tie and we do not claim a win — but a launch that reports tok/s and
withholds seconds-to-answer is choosing its metric, and this corpus has
published unfavourable results before precisely so the favourable ones are
believable.

Two things the row also shows, which are ours:

1. **Against their published configuration we are 10.7% faster to the
   answer** (6.90 vs 7.73 s). Their depth-5 tuning is what closes it.
2. **Our latency is 4.5x more consistent: SD 0.30 s against 1.34-1.40 s.**
   The 0.34 s gap at depth 5 sits well inside their own spread, so on this
   metric too the honest verdict is "indistinguishable, point estimate
   theirs" -- but the *predictability* difference is real and is not noise:
   their completion length varies 18% seed to seed, ours 3%.

## What this row still does not include

**Prefill.** This is decode-only, because that is what both harnesses
measure. `README.md` concedes prefill is theirs by roughly 4x (native FP4
W4A4 flops). A true end-to-end seconds-to-answer would therefore favour
them by *more* than 5.0%, not less. Stating the decode-only number without
this sentence would repeat the original error in a smaller way.

`benchmarks/v11_suite.py` and `benchmarks/ninfer_repro.py` now stamp `t0`
and bank `ttft_ms` and `wall_s`, so the next cross-engine run reports the
complete figure rather than the half we could reconstruct.
