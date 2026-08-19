# The +6.05 ms/round mixed regression, closed by measurement (2026-08-18)

**It was never one effect, and the larger of the two has nothing to do with
QPN8.** Serving the RadixArk checkpoint verbatim also inherits its
`kv_cache_quant_algo: FP8` directive, which on SM70 drops decode attention
off the tensor-core XQA path onto a scalar paged kernel. That accounts for
+4.82 ms of the +6.05; the QPN8 weight tax accounts for +1.08.

## 1. The two variables, separated (2x2, unprofiled)

`[e5p]` medians, n=300 steady rounds/rank, TP0 quoted; the four ranks agree
to within 0.06 ms on every cell. Same prose prompt, temperature 0, k=7,
capture sizes (8,16), MNS=1.

| cell | weights | KV cache | attention route | wall | G0→G1 | G1→G2 | G2→G4 | bubble |
|---|---|---|---|---:|---:|---:|---:|---:|
| **A** | all-NVFP4, QPN2 ×256 | fp16 | xqa_tc | 24.50 | 17.05 | 0.40 | 5.85 | 1.23 |
| **B** | all-NVFP4, QPN2 ×256 | fp8_e4m3 | scalar_paged | 28.85 | 20.42 | 0.41 | 6.90 | 1.18 |
| **C** | verbatim, QPN8 ×128 | fp16 | xqa_tc | 25.58 | 18.18 | 0.41 | **5.84** | 1.20 |
| **D** | verbatim, QPN8 ×128 | fp8_e4m3 | scalar_paged | 30.40 | 21.87 | 0.40 | 6.91 | 1.23 |

A and D reproduce the campaign's reference arms (24.48/17.06/0.41/5.85 and
30.53/21.95/0.41/6.94) to within 0.08 ms.

**Main effects**

| | wall | verify | drafter |
|---|---:|---:|---:|
| KV dtype fp16→fp8 (at QPN2) | +4.35 | +3.37 | +1.05 |
| KV dtype fp16→fp8 (at QPN8) | +4.82 | +3.69 | +1.07 |
| weights QPN2→QPN8 (at fp16 KV) | +1.08 | +1.13 | **−0.01** |
| weights QPN2→QPN8 (at fp8 KV) | +1.55 | +1.45 | **+0.01** |

**The drafter column reads 5.85 / 6.90 / 5.84 / 6.91.** It moves only with
the KV dtype and is completely insensitive to the weights — QPN8 contributes
0.00 ms to G2→G4, exactly as the checkpoint census said it must (every
`mtp.*` tensor is BF16 and identically shaped in both checkpoints).

Cell C ran at `gpu_memory_utilization=0.88` because the verbatim weights plus
an fp16 KV cache do not fit at 0.93. Control: cell A re-measured at 0.88 gives
wall 24.50 / G0G1 17.08 / G2G4 5.85 — identical to 0.93, so GMU is neutral
and the factorial is clean.

## 2. Where the KV directive comes from

| | `hf_quant_config.json` | `config.json` | resolved |
|---|---|---|---|
| verbatim RadixArk | `kv_cache_quant_algo: "FP8"` | `kv_cache_scheme {num_bits 8, type float}` | `kv_cache_dtype=fp8_e4m3` |
| requantised CTfull | *(absent)* | `kv_cache_scheme: null` | `kv_cache_dtype=auto` → fp16 |

vLLM honours the checkpoint's scheme whenever `--kv-cache-dtype` is left at
`auto` (`attention.py:263-269`: "Honor it only when the user did not
explicitly pick a kv_cache_dtype"). The boot then logs
`FLASH_ATTN_V100 FP8 KV cache decode path active (kv_cache_dtype=fp8_e4m3,
route=scalar_paged)` instead of the tensor-core route.

Passing `--kv-cache-dtype auto` does **not** undo it (auto *is* the case the
checkpoint overrides), and FLASH_ATTN_V100 rejects an explicit `float16`
(`RuntimeError: Unsupported kv_cache_dtype: float16`). Cell C therefore
needed a config-only checkpoint variant — weights symlinked, the two KV keys
deleted (`scripts/make_fp16kv_variant.sh`).

This is a V100 story, not a checkpoint defect: SM70 has no FP8 hardware, so
an FP8 KV cache has to be unpacked in software and the tensor-core decode
kernel is unavailable.

## 3. Per-kernel-family breakout (nsys, node-level graph tracing)

NVTX brackets on the existing G0..G4 markers (`VLLM_SM70_E5_NVTX=1`) let every
GPU kernel be attributed to verify / rejection / drafter through
`kernel.correlationId → runtime call → enclosing NVTX range`. Under graph
replay one launch carries all its node kernels, so a whole replay lands in
exactly one phase. FP4: 487 rounds, 1.17 M kernels on device 0, 97.7%
attributed. Mixed: 189 rounds.

Raw profiled ms/round, both arms under identical profiler settings:

| verify category | FP4 | mixed | delta | FP4 µs/call | mixed µs/call |
|---|---:|---:|---:|---:|---:|
| protected linears | 2.025 | 2.953 | **+0.928** | 15.85 | 23.07 |
| common trunk linears | 4.976 | 4.986 | +0.010 | 22.26 | 22.26 |
| attention | 2.374 | 12.294 | **+9.919** | 74.33 | 384.17 |
| GDN/state | 1.570 | 1.574 | +0.005 | 10.92 | 10.93 |
| collectives | 2.483 | 2.517 | +0.034 | 19.14 | 19.36 |
| norms/elementwise | 2.603 | 2.614 | +0.012 | 3.62 | 3.63 |
| copies/materializations | 2.560 | 2.744 | +0.184 | 4.01 | 4.29 |

| drafter category | FP4 | mixed | delta | FP4 µs/call | mixed µs/call |
|---|---:|---:|---:|---:|---:|
| lm_head | 1.536 | 1.482 | −0.055 | 219.56 | 221.12 |
| drafter linears | 2.292 | 2.152 | −0.140 | 36.38 | 36.10 |
| state/attention | 0.426 | 1.350 | **+0.924** | 28.40 | 94.68 |
| sampling | 0.069 | 0.066 | −0.003 | 4.62 | 4.61 |
| collectives | 0.739 | 0.682 | −0.057 | 21.12 | 20.50 |
| norms/elementwise | 0.534 | 0.515 | −0.019 | 6.51 | 6.58 |
| copies/other | 0.295 | 0.359 | +0.064 | 3.21 | 4.07 |

Only two rows move in either phase. Every other category matches to within
0.035 ms and its per-call cost is identical to the second decimal — including
`lm_head` (219.56 vs 221.12 µs) and `drafter linears` (36.38 vs 36.10 µs),
which settles the question of whether the two arms' lm_head provenance
(native NVFP4 codes vs packed-from-BF16) mattered: it does not.

Two measurement notes. The profiled attention figure is inflated — the mixed
arm's profiled verify totals 29.68 ms against its own unprofiled 21.87 — so
the +9.92 is **not** quoted as the real cost; the unprofiled 2×2 supplies
that (+3.69). Everywhere else the profiler adds a per-launch constant
(FP4: +1.531 ms over 2015.4 launches = 0.76 µs each), which is common-mode
because the two arms launch the same number of kernels (2015.4 vs 2019.0).

Layer identification is exact, not apportioned: the skinny GEMM family encodes
N in its grid (grid = N/32), and the one signature shared by two shapes
(gridX=160, out_proj K=1536 and mlp down K=4352) splits at graph-node
granularity into exactly 64 nodes at 10.58 µs and 64 at 24.94 µs.

## 4. Fixed-state pollution test — pollution refuted

`scripts/drafter_pollution.py`: one CUDA-graph-captured drafter workload
(7 draft steps, 3136 MiB/rank of weight reads), identical buffers and
addresses, replayed under four preceding conditions, round-robin, n=40.

```
condition                     probe ms      min      max   vs cold
A cold                           4.241    4.229    4.253    +0.000
B after FP4 verify               4.287    4.275    4.301    +0.047
C after mixed verify             4.287    4.270    4.302    +0.046
D after 0.8GB stream             4.244    4.235    4.260    +0.004

C-B = -0.001 ms   D-B = -0.043 ms   B-A = +0.047 ms
```

The two verify passes are the real thing: 256 GEMMs each, 3262 MiB/rank (FP4,
QPN2) vs 4015 MiB/rank (mixed, QPN8 on the protected 128) — a +752 MiB/rank
surplus that matches the checkpoint census to 1 MiB.

**C ≈ B (−0.001 ms).** Streaming an extra 752 MiB/rank immediately before an
identical drafter changes it by nothing. D ≈ A says even a dedicated 0.8 GB
streaming read leaves no residue. The only effect present at all is
B − A = +0.047 ms, i.e. *any* preceding GEMM work costs the same trivial
amount in both arms.

By the pre-registered decision tree this is the `C ≈ B` branch: the drafter
delta is not a cache/TLB/memory-hierarchy effect. The 2×2 says what it is
instead — the KV-cache dtype, +1.05 ms of it, in the drafter's own attention.

## 5. What to do about it

`--kv-cache-dtype` cannot express "fp16" to this attention backend, so the fix
is either the config-only variant used for cell C, or teaching the SM70
backend to accept `float16` explicitly. Either way the verbatim-serving result
should be quoted as **25.58 ms/round (cell C)**, not 30.40: serving the
RadixArk weights verbatim costs **+1.08 ms/round (+4.4%)** over the all-FP4
requant, which is close to the +0.81 ms the isolated graph-replay A/B
predicted for QPN8 and to the +0.68 ms cold-stream figure. The remaining
+4.82 ms was an unintended second variable.

## 6. Closed accounting

FP4 column: the profiled composition with the profiler's per-launch constant
removed (0.760 µs × launches for verify, 0.134 µs for the drafter), which
makes it sum to that arm's own unprofiled total. Deltas: the two unprofiled
2×2 main effects (protected linears ← A→C, attention ← C→D), less the
small category deltas that the profiled breakout measures directly and that
are common-mode enough to read off it.

```
VERIFY
category                       FP4    mixed    delta
protected linears             1.93     3.00    +1.07
common trunk linears          4.81     4.82    +0.01
attention                     2.35     5.86    +3.51
GDN/state                     1.46     1.47    +0.00
collectives                   2.38     2.42    +0.03
norms/elementwise             2.06     2.07    +0.01
copies/materializations       2.07     2.26    +0.18
TOTAL                        17.06    21.88    +4.82
reference                    17.06    21.95    +4.89   unexplained +0.07

DRAFTER
category                       FP4    mixed    delta
lm_head                       1.54     1.54    +0.00
drafter linears               2.28     2.28    +0.00
state/attention               0.42     1.43    +1.01
sampling                      0.07     0.07    +0.00
collectives                   0.73     0.73    +0.00
norms/elementwise             0.52     0.52    +0.00
copies/other                  0.28     0.35    +0.06
TOTAL                         5.85     6.92    +1.07
reference                     5.85     6.94    +1.09   unexplained +0.02

REJECTION (G1->G2)            0.41     0.41    +0.00
```

`5.96` of the `6.05 ms` wall delta is accounted for by two directly measured
terms — the inherited FP8 KV-cache directive (+4.82) and the QPN8 weight tax
(+1.08). The `0.09 ms` remainder is inside the boot-to-boot spread between
these cells and the campaign's reference arms (which they reproduce to
≤0.08 ms), and no mechanism is claimed for it.

## 7. Hypotheses this retires

- **"~1.8 ms of QPN8 integration overhead."** There is none. At matched KV
  dtype the whole verbatim-vs-requant cost is +1.13 ms of verify, of which
  the protected GEMMs are +1.07 — consistent with the isolated graph-replay
  A/B (+0.81) and the cold-stream figure (+0.68) once in-situ launch context
  is included.
- **"Something pollutes the drafter."** Refuted twice: cell C (QPN8 weights,
  fp16 KV) has a 5.84 ms drafter against the FP4 arm's 5.85, and the
  fixed-state replay puts C−B at −0.001 ms.
- **"The two arms' lm_head provenance matters."** Native NVFP4 codes vs
  packed-from-BF16: 219.56 vs 221.12 µs/call, 7 calls/round, delta −0.055 ms.
- **"Boot variance."** Every cell here is reproducible to ≤0.08 ms and the
  factorial is additive to 0.09 ms.

## 8. Loader fix — the published checkpoint now serves correctly as-is

The checkpoint variant was a workaround; the bug was architectural. A
checkpoint's `kv_cache_quant_algo` / `kv_cache_scheme` describes how its
**weights** were produced, and vLLM was reading it as standing permission to
also run the **KV cache** quantized whenever `--kv-cache-dtype` was left at
`auto`. On a part with no FP8 hardware that is never the right trade.

`fork_patches/torch_utils.py` adds `checkpoint_kv_quant_allowed()`, and both
application points now consult it:

- `resolve_kv_cache_dtype_string()` (the ModelOpt path, `utils/torch_utils.py`)
- the compressed-tensors re-apply in `attention.py:263`

Below SM80 the directive is ignored and the KV cache stays unquantized. An
explicit `--kv-cache-dtype` never reaches either site, so a user who asks for
a quantized KV cache still gets one; `VLLM_SM70_ALLOW_CKPT_KV_QUANT=1`
restores the old behaviour.

Verified on the **unmodified published checkpoint** (`Qwen3.8-27B-NVFP4-radixark`,
GMU 0.88):

```
kv_cache_dtype=auto                      (was fp8_e4m3)
route=qpn                                (tensor-core; was scalar_paged)
"Ignoring the checkpoint's kv_cache quantization directive (fp8_e4m3)"
wall 25.68  G0G1 18.23  G1G2 0.41  G2G4 5.86  bubble 1.20
```

against cell C's config-variant result of 25.58 / 18.18 / 0.41 / 5.84 — the
same number within boot spread. **`scripts/make_fp16kv_variant.sh` is retired**;
it is kept only as the reproduction of how cell C was first reached.

Minor follow-up, not on the critical path: the quant config still carries
`kv_cache_quant_method="FP8"`, so per-layer `k_scale`/`v_scale` parameters are
still allocated (156,992 KV tokens vs the variant's 169,219, ~7% of capacity).
Timing is unaffected (18.23 vs 18.18 ms verify).

## 9. The remaining QPN8 tax is the FP8 bytes, not tuning

Both levers on the +1.07 ms protected-linears delta were measured and are null.

**Single-op vs two-op dispatch** (arm C, fp16 KV, GMU 0.88, boot-flag flip via
`VLLM_SM70_QPN8_TWOOP`):

| dispatch | wall | G0→G1 | G2→G4 |
|---|---:|---:|---:|
| single-op (deployed) | 25.64 | 18.19 | 5.85 |
| two-op (the pre-refactor form) | 25.62 | 18.17 | 5.86 |

−0.02 ms, inside the ±0.06 rank spread. The single-op refactor was adopted on
the 27.98 ms boot that later proved to be an outlier; the comparison had never
been made cleanly. Dispatch shape is not a lever.

**Graph-mode geometry retune** (`benchmarks/qpn8_graph_sweep.py`, M=8, 300
iters, all six (split,nacc) per shape, eager and graph):

| shape | deployed | graph best | Δ |
|---|---|---|---:|
| gdn in_proj_qkvz K5120 N4096 | (16,2) 28.11 µs | (16,2) 28.27 | +0.6% |
| out_proj K1536 N5120 | (8,2) 12.29 µs | (8,2) 12.27 | −0.1% |
| attn qkv K5120 N3584 | (16,2) 25.27 µs | (16,2) 25.29 | +0.1% |

The deployed table is already the graph-mode optimum on every shape, and QPN8
shows **none** of QPN2's eager-vs-graph divergence (<1.5% here vs QPN2's known
27%). Total available saving: **0.007 ms**.

**Why there is no headroom.** Per-rank, per round, over the 128 protected layers:

| | bytes | time | effective |
|---|---:|---:|---:|
| QPN2 (arm A) | 1014 MiB | 1.727 ms | 587 GB/s |
| QPN8 (arm C) | 1804 MiB | 2.540 ms | 710 GB/s |

QPN8 already moves 1.78× the bytes at 1.21× the bandwidth. Closing the
+0.813 ms would need ~1250 GB/s against a measured copy ceiling of ~825 GB/s,
so **the tax is irreducible at this storage format** — it is the price of the
FP8 weights, not an implementation defect.

What is left is small and specific: `out_proj` (K=1536) runs at 641 GB/s
against the 825 GB/s ceiling, worth ~0.107 ms if it could be closed — it is
the short-K, launch-and-tail-bound shape — plus the ~0.26 ms by which the
in-situ protected cost (≈3.00 ms) exceeds the isolated graph-replay figure
(2.54 ms). Roughly 0.36 ms of the 1.07 is addressable; the rest is bytes.

## 10. Production baseline

`scripts/prodrun.sh` boots the published checkpoint verbatim: fp16 KV (the
loader fix declines the FP8 directive), GMU 0.88, k=7, thinking on by default,
usage proxy on :8010. Boot witness must read `kv_cache_dtype=auto` and
`route=qpn`.

Verified live against the A/B arm-C figures on the four cells whose identity
check passed: prose 26.18 vs 26.20, code 26.17 vs 26.20, csv 26.17 vs 26.18,
extract 28.01 vs 28.05 ms/round — identical to 0.03 ms.

Caveat on the boot-time `[e5p]` line: it samples the FIRST 300 qualifying
rounds, so a boot whose opening requests are short or whose context is growing
reads high (29.2 ms on this boot, with G1→G2 at 1.75 vs the steady 0.41). The
per-cell decode-only figures are the quotable ones; the boot-window `[e5p]` is
not, unless the drive is a single long steady generation as in the A/B runs.

## 11. Prefill dispatch boundary was in the wrong place (fixed)

`m8n8k4` gives 8 rows per tile, so M>8 needs several. Production chunked only
to M=16 and handed everything above to a transient fp16 reconstruct. Measured,
that boundary is far too low — chunking wins to M≈112:

| M | chunked | reconstruct | (gdn in_proj, µs) |
|---:|---:|---:|---|
| 32 | 113.1 | 349.4 | chunked 3.1× |
| 64 | 223.5 | 378.2 | chunked 1.7× |
| 96 | 336.4 | 418.3 | chunked 1.24× |
| 128 | 452.8 | 418.4 | reconstruct |

Per-shape crossings are 104–116, so the boundary is set to **96**
(`VLLM_SM70_QPN8_CHUNK_MAX`, =16 restores the old behaviour). The newly
exercised ragged tails (M=17→8+8+1, M=20→8+8+4) validate at the same
2.75e-4 as the native path across 30 cases
(`benchmarks/qpn8_chunk_validate.py`).

This window is **short prompts** — the live route census shows real prefill
chunks at M=20 and M=103. End-to-end TTFT, median of 5, same boot script:

| prompt tokens | CHUNK_MAX=16 | CHUNK_MAX=96 | Δ |
|---:|---:|---:|---:|
| 15 | 143.1 ms | 140.2 | −2% (control, ≤16 both) |
| 47 | 275.8 | **196.4** | **−28.8%** |
| 83 | 276.5 | **203.8** | **−26.3%** |
| 200 | 278.9 | 271.4 | −3% (control, >96 both) |
| 560 | 351.4 | 345.2 | −2% (control) |

Both controls flat, the win confined to the window that moved — ~75 ms off
interactive TTFT.

## 12. The kernel is DRAM-bound — which is what MT=2 needs

| shape | pure fp16 read | copy (r+w) | qpn8 M=8 |
|---|---:|---:|---:|
| gdn in_proj | 31.44 µs / 667 GB/s | 797 GB/s | **28.29 µs / 741 GB/s** |
| out_proj | 17.11 / 460 | 752 | **12.36 / 636** |
| attn qkv | 28.50 / 644 | 792 | **25.56 / 718** |

The GEMM is **faster than a pure streaming read of the same bytes** and sits
at ~90% of achievable read bandwidth. So the weight stream, not the MMA issue,
sets the time — and a second tile that adds no weight traffic should ride
along nearly free. From the measured per-row slope (0.056–0.083 µs/row),
MT=2 at M=16 projects to ≈28.9 µs against chunked's 57.3 — **1.98×** — worth
~3.6 ms/round over the 128 protected layers and ~6.5 ms across all 256 GEMM
sites, which would take a k=15 round from 43.8 ms to ≈37 ms.

Contrary evidence, stated: an L2-resident probe put the second tile at +101%
(i.e. no win). That probe is unfaithful — fitting 6 MB of L2 forces K≈1024
against the real K=5120, which lands in a launch/epilogue-bound regime. The
DRAM measurement is on the real shapes with no regime change. What neither
probe can settle is whether a second tile's MMA issue hides inside the ~10%
slack; only the kernel answers that.

## 13. MT=2 built and measured — 1.65× on the k≤15 band

`skinny_fp8_qpn8_mt2<SPLITK, NACC, FASTDEC>` keeps two accumulator sets and
issues both m8n8k4 row-tiles against a single B fragment, so M=9..16 pays one
weight pass instead of chunking's two. Shared staging doubles to
`SPLITK*512` floats — 32 KB at SPLITK=16, inside the 48 KB static limit,
which is why SPLITK=32 is not instantiated.

**Correct on the first run**: rel_err 2.72–2.83e-4 across M ∈ {1,4,8,9,12,15,16}
on all three protected shapes — indistinguishable from the native path's
2.75e-4.

| shape | native M=8 | chunked M=16 | **MT=2 M=16** | MT=2 cfg | speedup |
|---|---:|---:|---:|---|---:|
| gdn in_proj_qkvz | 37.70 µs | 56.83 | **34.59** | (16, 1, fast) | 1.64× |
| out_proj | 13.90 | 25.39 | **14.49** | (8, 1, fast) | 1.75× |
| attn qkv | 35.07 | 51.83 | **34.05** | (16, 1, fast) | 1.52× |

128 protected layers at M=16: **5.182 → 3.132 ms, saving 2.050 ms/round.**

On two of three shapes MT=2 at M=16 is faster than the native kernel at M=8 —
sixteen rows for less than the price of eight, which is what "the weight
stream is the whole cost" means when you stop paying it twice.

This vindicates the DRAM probe over the L2-resident probe: the second tile
was never the expense, the second weight pass was.

### The fast decoder had never worked (found by deploying it)

The first MT=2 deployment picked its geometry off a **timing** sweep and
selected `nacc=3` — the fast decoder — which the correctness pass had never
covered. The server booted, served, and reported **98.8% acceptance uniform
across all 15 positions** (τ=15.82) against the control's normal decay
(0.711, 0.387, 0.209, … 0.000, τ=2.48). The verifier's logits were NaN, so
every draft "matched".

`fp8x8_to_half2x4_fast` had **two** bugs and had never produced a correct
answer in either kernel:

1. `EM = 0x7F807F80` masks fp16 bits 7..14, but e4m3's exp+mantissa is seven
   bits and lands in 7..13. Bit 14 caught the **sign**, which sits in the fp16
   exponent — every negative weight overflowed to inf and accumulated to NaN.
   Correct mask: `0x3F803F80`.
2. The consumer expects the scalar decoder's pairing,
   `out[i] = (q.x byte i, q.y byte i)` — the (i, i+4) interleave that cancels
   the prepack's korder. The fast path paired sequentially within each word
   (`(x0,x1),(x2,x3),(y0,y1),(y2,y3)`), decoding every value correctly and
   still producing a garbage GEMM (rel_err 1.22).

Production was never exposed: the native table selects `nacc=2`, so the path
was unreachable. Both fixed; all 12 kernel × config combinations now score
2.72–2.77e-4, matching the scalar path. **`benchmarks/qpn8_mt2_eval.py` now
asserts the numerical correctness of whatever config it recommends**, so a
timing sweep can no longer crown a NaN.

With a correct fast decoder the optimum is (split=16, nacc=1, fast) on all
three shapes: **5.179 → 3.111 ms, 1.66×, saving 2.068 ms/round.**

### End-to-end at k=15

Verify runs at M=16 there, so the path is actually taken (k=7 is M=8 and
unaffected — production is unchanged by this).

| k=15 | wall | G0→G1 | G1→G2 | G2→G4 |
|---|---:|---:|---:|---:|
| chunked | 39.78 | 25.07 | 0.64 | 12.73 |
| **MT=2** | **37.13** | **22.50** | 0.64 | 12.74 |

**Verify −2.57 ms, wall −2.65 ms**, slightly better than the 2.07 ms
projected offline. Per-position acceptance matches the control
(0.699/0.441/0.238/… vs 0.689/0.440/0.245/…), confirming the numerics in
situ.

k=15 is still not competitive with k=7 *on prose*, where acceptance is poor
(τ≈1.57 → 2.57 tokens/round): MT=2 narrows the gap without flipping it on
that workload. The remaining prize is the trunk — 128 NVFP4 MLP sites still
chunk at M=16, worth roughly another 3.7 ms if QPN2 gets the same treatment.

Remaining for the full k=15 prize: the trunk MLP (128 more call sites) is
NVFP4 and still chunks. Porting MT=2 to the QPN2 kernel would take its M=16
cost from ~9.6 to ~5.9 ms, so full MT=2 is worth ≈5.8 ms off a k=15 verify.

## 14. k=15 + MT=2 throughput — a big win, on exactly two cells

Six-cell suite at k=15 with MT=2 against the k=7 production numbers:

| cell | k=7 tok/s | k=15+MT=2 tok/s | Δ | τ 7 → 15 | round 7 → 15 |
|---|---:|---:|---:|---|---|
| **extract** | 285.2 | **392.8** | **+37.7%** | 7.00 → **15.00** | 28.05 → 40.73 |
| json | 255.3 | 260.3 | +2.0% | 5.73 → 8.93 | 26.39 → 38.19 |
| code | 187.9 | 149.3 | −20.5% | 3.93 → 4.67 | 26.20 → 37.90 |
| math | 184.5 | 145.2 | −21.3% | 3.75 → 4.46 | 25.55 → 37.32 |
| csv | 148.8 | 106.7 | −28.3% | 2.89 → 3.03 | 26.17 → 37.79 |
| prose | 97.0 | 67.1 | −30.8% | 1.55 → 1.54 | 26.20 → 37.77 |

**392.8 tok/s on extract at τ=15.00** — the k=15 ceiling, every draft token
accepted. MT=2 contributes +6.5% of that (without it the round is 43.4 ms and
extract lands ≈369).

The k=15 round is *longer*, not shorter: 37–41 ms vs 26 ms. So depth has to
earn back the round, and the crossover is exact:

> k=15 wins iff tokens/round improves by more than round₁₅/round₇ = **1.45×**

extract 2.00 ✓ · json 1.48 ✓ · code 1.15 ✗ · math 1.15 ✗ · csv 1.04 ✗ ·
prose 1.00 ✗

This makes `num_speculative_tokens` a **per-request** lever, not a global
default: extraction and structured output want depth, prose and reasoning do
not. Porting MT=2 to the NVFP4 trunk (still chunking at M=16, ~3.7 ms) would
take the k=15 round to ≈34 ms and the crossover to ≈1.30×, pushing extract
past 430 without obviously rescuing code or math at 1.15.
