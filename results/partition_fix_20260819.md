# Decode-partition fix: 98K context capability at 16K-class round time

2026-08-19, 4x V100-SXM2-16GB, TP4. Follows `mml_ladder_20260819.md`.

## The fix

    VLLM_FLASH_V100_DECODE_PARTITION_SIZE=256

Untraced, three cells, K=7, GMU 0.88, greedy, thinking off:

| arm | MML | mergesort | json | log2json | G0G1 | G2G4 | wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 16,384 | 25.12 | 25.18 | 25.43 | 17.48 | 5.85 | 24.94 |
| B | 98,304 | 25.80 | 26.44 | 27.98 | 18.29 | 5.85 | 25.76 |
| **C** | **98,304 + part 256** | **25.04** | **25.09** | **25.36** | **17.38** | 5.86 | **24.85** |

Arm C matches or beats the 16,384 plateau on every cell while keeping full
98,304 capability. Best cell: **log2json 27.98 -> 25.36 ms, -9.4%.**

Arms A and B reproduce the ladder (25.13/25.18/25.44 and 25.84/26.52/28.07)
to within 0.1 ms, so the instrument is sound.

## Where the tax lives — phase-localised, not inferred

| arm | MML | G0G1 verify | G1G2 | G2G4 drafter | bubble | wall |
|---|---:|---:|---:|---:|---:|---:|
| A | 16,384 | 17.48 | 0.41 | 5.85 | 1.19 | 24.94 |
| B | 98,304 | **18.29** | 0.41 | **5.85** | 1.19 | 25.76 |

**The whole +0.82 ms wall delta appears in verify (+0.81).** The drafter --
seven sequential q=1 decode calls per round -- is identical to the last digit
under a 6x change in `max_model_len`, as are G1G2 and bubble.

That is the decisive evidence for the q>1 small-q path being the locus: the
q=1 path derives its partition size from the ACTUAL sequence length (trace:
`max_seq_hint=22`, `1000`, `1001` ...), so below 32,768 it picks 256 at any
MML. Only the verify call carries an MML-derived hint.

## Mechanism

`flash_attn_v100.py:2332`, in the small-q metadata builder:

    raw_seq_capacity = int(block_table.shape[1]) * int(self.block_size)
    ...
    flash_metadata.smallq_decode_workspace_seq_capacity_hint = raw_seq_capacity

`block_table.shape[1] * block_size` **is** `max_model_len`. The default
partition selector is:

    def _select_default_decode_partition_size(max_seq_len_hint):
        if max_seq_len_hint is None: return _DEFAULT_DECODE_PARTITION_SIZE
        if max(1, int(max_seq_len_hint)) >= 32768: return 1024
        return _DEFAULT_DECODE_PARTITION_SIZE      # 256

Valid sizes are (256, 512, 1024). The escape hatch that would fix this --
`active_num_partitions`, "letting kernels skip inactive partitions for the
current runtime sequence length" -- is populated only for q=1 decode. MTP
verification arrives as q>1 prefix prefill, so it is `None` every round.

The intended fix is **half-built in the tree**: `build_for_cudagraph_capture`
computes `workspace_seq_capacity_cap` and would pass a bucketed partition
hint, but only via `_mtp_context_bucket_partition_size_hint()`, which reads
`VLLM_SM70_MTP_CONTEXT_BUCKET_PARTITION_SIZE` and returns `None` when unset.
The capping half ships enabled; the partition half ships dormant.

## Still unexplained: the magnitude

The source explains a *step*, and the ladder's two-plateau shape confirms it.
It does **not** explain why the delta scales with live context at ~1.15
us/token. A 256 -> 1024 switch predicts deltas in ratio 0.73 / 1.24 / 1.24
against measured 0.73 / 1.33 / 2.64 -- log2json is 2x off. Modelling it as
`ctx/partitions` under-saturation gives 473 / 322 / 497 against a measured
473 / 880 / 1727. And the partition count *decreases*, so this is a
parallelism collapse, not the extra pass the linearity suggests.

**Locus confirmed, trigger confirmed, cost model not.** Do not publish a
mechanism for the linearity.

## Caveat before this becomes unconditional

Partition 1024 is presumably the default above 32,768 because it wins at
genuinely long context. Pinning 256 is right for our sub-32K range and has
**not** been re-tested at 47K+ actual context, where the AIME f15/f30
fixtures live. `prodrun.sh` and `ninfer_repro_run.sh` therefore take
`DECODE_PARTITION=` as an override rather than hardcoding. The principled
fix remains bucketing by actual length, i.e. completing the dormant path.

## Method note — the witness cost 6.7 ms/round

`VLLM_FLASH_V100_TRACE_DECODE_ACTIVE=1` logs from inside the decode path:
8,696 INFO lines and 3.0 MB of log in one arm, against 208 KB untraced. Its
MML 16,384 arm measured **31.84 ms where the untraced ladder measures
25.13**. The overhead lands almost entirely in the drafter (G2G4 5.85 ->
10.34), because the drafter issues k=7 q=1 calls per round while verify is a
single q>1 call the trace never fires on.

Use the trace for geometry only. Never for timing.
