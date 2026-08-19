# v1.1 shipped configuration: direct-execution proof, three-depth table,
# and the effort x sampling grid

2026-08-19, 4x V100-SXM2-16GB, TP4. Production cell-C boot with the decode
partition pinned (`VLLM_FLASH_V100_DECODE_PARTITION_SIZE=256`), MML 32,768,
GMU 0.88, MNS 1, MT=2 on, greedy unless stated.

**Metric convention.** Throughput is `tok_s` = `(gen - (tau + 1)) / decode_s`
-- ours, subtracting a whole speculative round. ninfer's convention is
`(gen - 1) / decode_s` and reads higher by exactly tau/gen -- 0.4-0.7% on
these cells, more at k=15 where tau is larger. The launch table and every
number here use OURS. `v11_suite.py` now prints both columns labelled,
because it previously printed ninfer's under a bare `tok/s` header while
writing ours to the JSON, which caused exactly one mis-quote during this
work (doc2json k=15 stated as 403.5 when the table convention gives 392.1).

## 1. Direct execution, demonstrated

The launcher used to hard-fail unless a 1.9 GB shard from
`Qwen3.6-27B-NVFP4` -- a different, older model -- was on disk. It was
vestigial twice over: the code resolves lm_head from the SERVED checkpoint
(`marlin.py _lmhead_resolve_native_path`, which never borrows a foreign
checkpoint), and this checkpoint's lm_head is already NVFP4 in-checkpoint so
it never enters that path at all.

Both boots below ran with `~/models/Qwen3.6-27B-NVFP4` **moved off disk**:

    k=7    route map: M=1  N=62080 K=5120 -> qpn2
           route map: M=8  N=62080 K=5120 -> qpn2
    k=15   route map: M=16 N=62080 K=5120 -> qpn        (legacy qpn owns M 9-16)
           route map: M=1  N=62080 K=5120 -> qpn2
    both   repack/fallback lines: 0
           Ignoring the checkpoint's kv_cache quantization directive (fp8_e4m3)
           kv_cache_dtype=auto

`N=62080` is 248,320/4, the TP4 vocab shard: **lm_head is served from the
checkpoint's own 4-bit codes**, at every M, with zero repacks.

**Volta executes the published ModelOpt NVFP4 checkpoint directly** -- no
repack, no requantization, nothing from any other model on disk. The one
deviation is deliberate and is a KV-cache *policy*, not a weight change: the
checkpoint's `kv_cache_quant_algo` describes how its WEIGHTS were made, and
honouring it below SM80 drops the tensor-core decode route for +4.82
ms/round. Phrase it as **weights verbatim, KV policy ours**.

**Scope.** ModelOpt format. compressed-tensors NVFP4 (llm-compressor,
unsloth) has no SM70 path -- QPN8 is wired into the ModelOpt path only.

## 2. Launch table, three depths

| cell | k=3 | tau | rnd | k=7 | tau | rnd | k=15 | tau | rnd | best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--|
| doc2json | 195.1 | 3.00 | 20.42 | 309.5 | 6.83 | 25.13 | **392.1** | 13.88 | 37.87 | k=15 |
| log2json | 195.6 | 3.00 | 20.45 | 307.1 | 6.81 | 25.39 | **391.4** | 13.94 | 38.10 | k=15 |
| json | 184.9 | 2.77 | 20.36 | 268.6 | 5.73 | 25.08 | **270.3** | 8.93 | 36.79 | k=15* |
| mergesort | 175.3 | 2.56 | 20.32 | **226.9** | 4.71 | 25.03 | 197.7 | 6.27 | 36.62 | k=7 |
| code | 173.1 | 2.52 | 20.33 | **197.1** | 3.93 | 24.98 | 155.0 | 4.67 | 36.50 | k=7 |
| math | 161.6 | 2.30 | 20.43 | **187.0** | 3.75 | 25.20 | 147.0 | 4.46 | 36.85 | k=7 |
| prose | **113.9** | 1.32 | 20.33 | 101.7 | 1.55 | 24.98 | 68.5 | 1.49 | 36.38 | k=3 |

*json's k=15 lead over k=7 is 0.6% -- a tie, not a win.

**The optimum is ordered by acceptance, monotonically**: extraction wants
k=15, code/math want k=7, prose wants k=3. Depth is a workload property,
not an engine setting, on our stack exactly as on ninfer's.

**doc2json and log2json pin at tau = 3.00 at k=3** -- every draft accepted,
the hard ceiling of depth 3. That is the same ceiling that pins ninfer's
extraction cells at 4.00 tokens/round. The difference is not acceptance: it
is that we can spend depth to break the ceiling and their engine caps at
`--draft-tokens 5`.

**Against the previously published table** (measured before the partition
pin): log2json +10.3%, doc2json +6.1%, json +5.4%, prose +5.1%, code +4.9%,
math +1.4%. Every round now sits on the ~25 ms plateau where the old table
ranged 25.6-28.0.

## 3. Reasoning effort x sampling, crossed, at k=3

All six arms are per-request, so one boot serves them. Cells span the
acceptance range.

| cell | mode | sampling | tok/s | tau | ms/rnd | gen |
|---|---|---|---:|---:|---:|---:|
| math | xhigh | greedy | 162.7 | 2.30 | 20.22 | 363 |
| math | xhigh | sampled | 162.8 | 2.49 | 21.26 | 392 |
| math | medium | greedy | 169.5 | 2.43 | 20.22 | 659 |
| math | medium | sampled | 153.6 | 2.29 | 21.43 | 715 |
| math | nothink | greedy | 178.1 | 2.61 | 20.26 | 885 |
| math | nothink | sampled | 160.1 | 2.42 | 21.30 | 652 |
| json | xhigh | greedy | 161.6 | 2.28 | 20.29 | 1024 |
| json | xhigh | sampled | 148.6 | 2.20 | 21.51 | 1024 |
| json | medium | greedy | 175.3 | 2.56 | 20.27 | 1024 |
| json | medium | sampled | 163.1 | 2.51 | 21.48 | 1024 |
| json | nothink | greedy | 185.7 | 2.77 | 20.26 | 1024 |
| json | nothink | sampled | 176.4 | 2.80 | 21.48 | 1024 |
| prose | xhigh | greedy | 106.2 | 1.15 | 20.27 | 1024 |
| prose | xhigh | sampled | 98.1 | 1.11 | 21.49 | 1024 |
| prose | medium | greedy | 119.3 | 1.42 | 20.28 | 1024 |
| prose | medium | sampled | 108.9 | 1.34 | 21.50 | 1024 |
| prose | nothink | greedy | 114.2 | 1.32 | 20.26 | 1024 |
| prose | nothink | sampled | 102.1 | 1.19 | 21.46 | 1024 |

### The sampler tax is a flat per-round cost

**+1.17 ms/round, sd 0.08, n=9** (range 1.03-1.22). It does not vary with
reasoning mode, and it does not vary with acceptance -- tau spans 1.11 to
2.80 across those nine pairs. Being flat in absolute terms, it is a larger
FRACTION of a shorter round: **6.0% at k=3 (~20.3 ms) against ~4.7% for the
same cost at k=7 (~25.0 ms)**.

### Reasoning effort does not change round time

Greedy rounds: 20.22 / 20.22 / 20.26 / 20.26 / 20.27 / 20.27 / 20.28 /
20.29 / 20.33. Effort changes what is generated and how predictable it is,
never how fast a round turns. It is a quality/length knob, not a speed one.

### `xhigh` is the template default and it is the slowest arm

Cap-matched comparison only (both arms hit the 1024 cap, i.e. identical
work):

| cell | greedy | sampled |
|---|---:|---:|
| json, xhigh vs medium | **-7.8%** | **-8.9%** |
| prose, xhigh vs medium | **-10.9%** | **-9.9%** |

**math is excluded**: its two arms produced 363 vs 659 tokens, so they did
different amounts of work and the comparison is confounded -- which is why
that row's sign flips between greedy (-4.0%) and sampled (+6.0%). Do not
quote it.

### Reconciling the "~30%" figure

Three numbers for the cost of `xhigh` now exist across `results/`:

| source | figure | what it compares |
|---|---|---|
| `realworld_and_sweeps_20260819.md` heading | ~30% | rate, vs a no-instruction control |
| `v11_launch_20260819.md:118` | 19.6% | already corrects the above, under the ninfer profile |
| this document | 7.8-10.9% | rate vs `medium`, cap-matched, k=3 |

They are not necessarily contradictory -- different baselines, different k,
and rate versus time-to-answer -- but **v1.1 must not ship all three
unlabelled**. The ~30% heading in `realworld_and_sweeps` is the one already
known to be wrong on magnitude and should carry the retraction its own body
text acknowledges.

### Thinking helps prose and hurts structured output

- **structured**: thinking costs json 5.6% (medium) to 13.0% (xhigh)
  against nothink, and costs acceptance too (tau 2.56 / 2.28 vs 2.77) --
  reasoning prose is less predictable than templated JSON.
- **prose**: `medium` thinking BEATS nothink, 119.3 vs 114.2 (+4.5%, tau
  1.42 vs 1.32). Planning text drafts more easily than creative narrative.

This supports the existing "disable thinking on structured and extraction
endpoints" guidance, and adds that the same advice is wrong for prose.

## Reproduce

    scripts/v11_direct_exec_proof.sh     # proof + k=7/k=15 table
    scripts/v11_table_k3.sh              # k=3 arm + the crossed matrix
