# QP-N m8n8k4 — VERDICT: WIN (2026-08-10)

The resurrection experiment succeeded. The Volta-native four-quadpair
form of `mma.sync.m8n8k4` — QPs on N, A-stationary registers, direct
NVFP4→B-register decode — beats both incumbents across the M≤8 band and
reaches the streaming floor at M=8. First tensor-core path on this stack
to hit SIMT-class effective bandwidth. Data: `qpn_race_20260810.csv`;
kernel `qpn_race.cu`; prepack in `qpn_bench.py`. Master tables untouched.

## The race (µs, 5-shape totals; per-shape rows in CSV)

| M | simt | wmma (padded) | qpn | qpn vs best incumbent |
|---|---|---|---|---|
| 5 | 187.9 | 251.4 | **130.7** | **1.44×** (wins 4/5; loses only N=2048 to simt) |
| 8 | 253.0 | 251.8 | **129.6** | **1.94×** (wins 5/5) |
| 11 | — | 256.8 | 333.4 (naive hybrid) | loses — see below |
| 16 | — | 264.7 | **253.8** (two 8-row tiles) | 1.04× (wins 4/5) |

Effective bandwidth at M=8 on (5120,8704): 25.1 MB / 38.8 µs ≈
**647 GB/s — the SIMT M=1 flagship number, now at M=8 with tensor-core
MACs**. M=8 costs the same as M=5 costs the same as (in the flat region)
M=1 SIMT: the M-scaling penalty of the verify band is gone up to 8 rows.

Kernel: **56 registers, zero spills, one barrier** (the cross-warp
K-reduce at output), 9 blocks/SM occupancy headroom — the grid (N/32),
not the kernel, is the residency limiter.

## Why this one won where v1/B_ring died

- **QPs split N, not K**: one warp instruction = 4 independent 8×8×4
  MMAs sharing the same 8×4 activation tile → activation traffic per
  weight byte drops 4× (B_ring's killing mechanism, inverted).
- **No smem at all in the main loop**: A comes straight from global
  (x is KB-scale, L1/L2-resident; QP-sibling lanes hit the same line),
  B streams global→registers via the prepack. Zero main-loop barriers —
  the register gate's latency-exposure disease has nothing to bite.
- **Direct decode**: nibble pre-interleave in the prepack makes
  `dequant8_tm`'s (i, i+4) output exactly the adjacent-k B-fragment
  register pair — v1's 8 pack instructions per window are simply gone.
- **Scale economy**: one fp8 group scale register serves exactly its
  group's 4 mmas (k=4 × 4 = the group-16 granularity).

## Caveats / honest edges

- **M=11 naive hybrid loses** (333.4 vs 256.8): the qpn(8)+simt(3)
  stitching pays a full simt launch plus a row copy. The real M 9–16
  route is two qpn A-tiles (M=16 row: 253.8 — pad 11→16), which is
  parity-to-+4% vs padded WMMA. The M 9–16 band is a wash; the prize is
  M≤8.
- **(5120,2048) at M=5** still goes to simt (grid 64 CTAs = 0.8 waves —
  the known N-starved shape).
- The three CSV `ok=0` rows are the **simt incumbent** at 1.005–1.08e-3
  vs the 1e-3 gate — the fp16-accumulate path held to a tighter gate
  than its own 1e-2 design spec; not a qpn failure. **qpn passes 1e-3 on
  every cell including outlier activations.**

## What this is worth (adoption path, not done here)

The M 2–8 band is: spec-verify at k≤7, and — more valuably — **plain
decode at 2–8 concurrent streams, exactly the crossover zone vs
TurboMind AWQ** (ab_report.md: crossover 4–8 streams). A 1.4–1.9× GEMM
cut there moves the crossover right. Adoption needs:
1. Loader-side prepack (the B-fragment nibble-interleaved layout) built
   once at weight load — converter/loader work, deliberately out of this
   bench's scope.
2. Dispatch: qpn for M 3–8 (seam vs simt to be re-measured at M 1–4),
   env-gated (`VLLM_SKINNY_QPN=1` pattern), WMMA unchanged above 16.
3. The standing rules: correctness ladder + end-to-end losslessness diff
   before any adopted serving number.

## Scope discipline

Old questions stayed closed: no ring, no bank swizzling, no staging-depth
re-litigation, batch band M17–64 remains conceded (twin_race_notes.md).
This experiment only reopened — and won — the native-form M≤8 tile.
