# v100-skinny

**Blackwell gets NVFP4 support in silicon. Volta gets it from software.**

v100-skinny runs Qwen3.8-27B's **published mixed NVFP4/FP8 weight
representation unchanged** on four Tesla V100-SXM2-16GB. In the release's
main decode and k≤7 verification regime, NVFP4 regions run through **QPN2**
(W4A16) and FP8 regions through **QPN8** (W8A16), both hand-written around
Volta's `mma.sync.m8n8k4`. No FP8→FP4 requantization. No persistent FP16
weight copy. FP16 activations and FP16 KV cache. OpenAI-compatible serving —
a usable Qwen3.8 endpoint, not a kernel demo.

The GPUs cost about **A$600 used**. *(That is the accelerators only — not the
server, not the rest of the build.)*

[Quick start](#quick-start) · [Results](#result) · [How QPN works](#how-qpn-works) ·
[Reproduction](#reproduction) · [Limitations](#hardware-cost-power-limitations)

---

## Result

**AIME 2026 Problem 1, five seeds, both engines run in our lab through one
harness, each at its own best measured native-MTP depth.** The RTX 5090 was
running **NInfer**, a specialist engine built for maximum performance on this
exact model family rather than a weak generic baseline. Sampling is NInfer's
published profile applied identically to both: temperature 0.6, top-p 0.95,
top-k 20, presence penalty 1.0, thinking on, `reasoning_effort` medium, seeds
1001–5005.

| | 4× V100-SXM2-16GB (2017) | 1× RTX 5090 (2025) |
|---|---:|---:|
| Engine | v100-skinny on 1Cat-vLLM, k=7 | NInfer, `--draft-tokens 5` |
| Decode | **219.1 ± 5.9 tok/s** | **214.7 ± 9.2 tok/s** |
| Tokens committed / round | 5.89 | 4.27 |
| Round latency | 26.9 ms | 19.9 ms |
| Time to correct answer | 6.90 ± 0.30 s | **6.56 ± 1.34 s** |
| Completion tokens | 1,513 ± 44 | 1,403 ± 253 |
| Correct answer (277) | **5/5** | **5/5** |
| Published quantization | RadixArk | NInfer artifact, Unsloth-derived |

This is a **same-lab system comparison on the same Qwen3.8 base model**, not a
same-weight causal engine A/B: the two engines use different published NVFP4
artifacts. NInfer depth 5 is its maximum exposed depth and its best measured
setting on this fixture; v100-skinny's best measured setting is k=7.

**This is parity — 1.02×, intervals overlapping.** We do not claim a win, and
the result is not universal: their prefill is roughly 4× ours. The decode
result is an exact cancellation — **our round is 35% longer (26.9 vs 19.9 ms)
and we commit 38% more tokens per round (5.89 vs 4.27); 1.38 / 1.35 = 1.02.**
QPN makes the wider k=7 verification batch cheap enough for the model's native
MTP depth to pay.

**Why depth is the lever.** `mma.sync.m8n8k4` issues an 8-row tile whether or
not all 8 rows carry work, and QPN2 holds **71% of read roofline at M=8** —
the k=7 verification width. On the measured stack, verification costs
**+0.383 ms per extra draft row** against **+0.817 ms per sequential drafter
step**. The 5090 completes a round much faster; Volta compensates by extracting
more useful tokens from each round.

---

## What v1.1 changes

**The published mixed weight allocation is now directly executable on Volta.**
ModelOpt's
native mixed FP8/NVFP4 path was gated above Volta — the FP8 regions had no
SM70 path at all. **v1.0 created the first practical bridge:** it down-converted
those protected FP8 layers to FP4 and served the resulting derivative through
the new W4A16 stack. **v1.1 adds QPN8, lowers the mixed-path gate to SM70, and
retires the conversion that v1.0 had made necessary.** The boot gate verifies
the QPN2/QPN8 route census and zero reference fallbacks on every start.
Across four ranks, all **512/512 protected FP8 module instances** are eligible
for QPN8, the checkpoint's own `lm_head` routes to QPN2, and reference fallback
calls remain zero.

Also in v1.1:

- **QPN8 (W8A16 FP8)** — the FP8 half of the checkpoint, 718.6 GB/s at M=1,
  82% of read roofline. Packed QPN8 routes cover M≤96; above that, the
  v100-skinny dispatch uses a transient prefill reconstruction rather than a
  persistent FP16 representation. Its MT=2 two-tile dispatch ships default-on
  for M 9–16. **MT=2 exists for FP8 only** — the NVFP4 M 9–16 band still
  runs the first-generation QPN kernel; see the gap noted under kernel
  benchmarks.
- **FP16-KV policy.** The checkpoint declares FP8 KV cache, which is sensible
  on hardware with an efficient FP8-attention path. On SM70, honouring that
  directive silently forced scalar paged attention. The measured cost was
  **+4.82 ms/round**, so v1.1 deliberately uses FP16 KV.
- **GDN speculative-state contract.** Boolean-mask indexing on the target
  verify path cost 21 device syncs and ~70 copies per step; the fast path
  removes them with byte-identical output. (Landed pre-v1.1 and measured on
  the previous model generation — the mechanism carries forward, the step
  times do not, so they are not quoted here.)
- **Greedy MTP drafting.** The SM70 default was `probabilistic` — the drafter
  *sampled* its proposals. Switching to greedy with local-argmax reduction is
  worth **10–25 acceptance points** on sampled serving.
- **Decode-partition fix.** `--max-model-len ≥ 32768` taxed every speculative
  round for capacity it never used. Pinning the partition size recovers
  **0.7–2.6 ms/round** and took AIME from 206.8 to 219.1 tok/s.
- **Graph-mode tuning and zero-fallback route validation** — every decode and
  verify call accounted to a QPN route, no silent Marlin fallbacks.

---

## End-to-end benchmarks

**Serving configuration.** TP4, GMU 0.88, FP16 KV, decode partition pinned,
MT=2 on, MTP with greedy drafting and local-argmax reduction,
`--max-num-seqs 1` (single stream). NUMA-pinned to socket 0, application
clocks pinned.

The domain table below was run at `--max-model-len 32768`; the AIME rows were
run at 98,304. Before the decode-partition fix that difference changed short-
context round latency. On the shipping path, declared MML is timing-neutral;
the two values are retained here as measurement provenance, not as performance
tuning. The same AIME configuration reads 206.8 tok/s without the pin and
219.1 tok/s with it.

**Sampling — the domain table.** Fully greedy and fully specified; no field is
left for the server to fill from a model card:

```
temperature 0.0 · top_p 1.0 · top_k disabled
presence_penalty 0.0 · frequency_penalty 0.0 · seed 1001
thinking off (except the math cell, which is thinking-on)
```

**Sampling — the AIME rows.** ninfer's published profile, applied identically
to both engines:

```
temperature 0.6 · top_p 0.95 · top_k 20
presence_penalty 1.0 · frequency_penalty 0.0
thinking on · reasoning_effort medium
five fixed seeds: 1001, 2002, 3003, 4004, 5005
```

`reasoning_effort` is set explicitly rather than defaulted: the chat template
defaults to `xhigh`, which injects a system message and costs 4.0–11.0% at
matched output length, while `medium` injects none and matches the prompt
ninfer's converter produces.

**Metric.** Throughput is `(generated − (τ+1)) / decode_seconds` — we subtract
a whole speculative round. The convention that subtracts a single token reads
higher by exactly τ/generated: 0.4–0.7% on the cells below, more at k=15 where
τ is larger. Every figure here uses ours, the more conservative one.

| Cell | k=3 | τ | k=7 | τ | k=15 | τ | best |
|---|---:|---:|---:|---:|---:|---:|:--|
| doc2json (invoice → JSON) | 195.1 | 3.00 | 309.5 | 6.83 | **392.1** | 13.88 | k=15 |
| log2json (logs → records) | 195.6 | 3.00 | 307.1 | 6.81 | **391.4** | 13.94 | k=15 |
| json (generate records) | 184.9 | 2.77 | 268.6 | 5.73 | **270.3** | 8.93 | k=15 (tie) |
| mergesort | 175.3 | 2.56 | **226.9** | 4.71 | 197.7 | 6.27 | k=7 |
| code (red-black tree) | 173.1 | 2.52 | **197.1** | 3.93 | 155.0 | 4.67 | k=7 |
| math (thinking on) | 161.6 | 2.30 | **187.0** | 3.75 | 147.0 | 4.46 | k=7 |
| prose | **113.9** | 1.32 | 101.7 | 1.55 | 68.5 | 1.49 | k=3 |

Round latency is nearly constant within each depth — **20.3–20.5 ms at k=3,
25.0–25.4 at k=7, 36.4–38.1 at k=15** — so cells differ almost entirely in how
many drafted tokens survive verification. (The k=15 band is the loosest: the
two extraction cells sit ~1.5 ms above prose there, which is the M=16 NVFP4
kernel gap below showing up end-to-end.)

**The ~392 tok/s figures are workload-specific, not headline throughput.**
Extraction is the best case for speculation: at k=3 both extraction cells pin
at τ = 3.00 — every draft accepted, the ceiling of that depth — and depth is
what breaks the ceiling. Free-form prose runs at 113.9 and *prefers k=3*. The
optimal depth is a property of the workload, not of the engine, which is why
every column names its depth. Depth is a boot-time setting in v1.1.

**Recommended release profiles** (concrete commands in
[Quick start](#depth-profiles)): k=7 for general short/medium-context work,
k=3 for long-context serving (65k measured), and k=15 only for highly
predictable structured/extraction workloads where its wider verification
actually pays.

### Declared context is free

`--max-model-len` used to cost real time: the decode partition geometry is
derived from it, so declaring a large window taxed every speculative round
even at short live context. With the pin, that is gone across the whole
supported range — eleven boots, 4,096 to 262,144, a **64× range of declared
context**, ms/round:

| MML | mergesort | json | log2json |
|---:|---:|---:|---:|
| 4,096 | 25.14 | 25.18 | 25.40 |
| 8,192 | 25.10 | 25.17 | 25.42 |
| 16,384 | 25.06 | 25.10 | 25.36 |
| 32,768 | 25.13 | 25.23 | 25.44 |
| 65,536 | 25.05 | 25.10 | 25.36 |
| 98,304 | 25.13 | 25.17 | 25.47 |
| 131,072 | 25.20 | 25.22 | 25.51 |
| 163,840 | 25.14 | 25.19 | 25.48 |
| 196,608 | 25.17 | 25.14 | 25.51 |
| 244,608 | 25.08 | 25.13 | 25.53 |
| 262,144 | 25.15 | 25.19 | 25.61 |

**Flat to within 0.25 ms**, with τ identical in every arm (4.71 / 5.73 / 6.81),
so all eleven did byte-identical work. The only hint of a trend is log2json,
the longest cell, drifting ~0.2 ms across the full 64× range — under 1%, and
absent from the other two. Unpinned at the same 32,768 the figures
are **25.84 / 26.50 / 28.03** — the pin recovers up to 2.59 ms (−9.2% on
log2json), and the penalty it removes grows with live context.

There is no "fastest" `--max-model-len`. **Declare the window you need within
the KV capacity reported at boot**; 244,608 is reliable on both measured memory
profiles, while 262,144 is marginal
([`results/mml_speed_20260819.csv`](results/mml_speed_20260819.csv)).

**Declaring a large window is free; filling it is not.** Every arm above
generates at short *live* context — what is flat is the cost of having
declared a large window, not the cost of having filled one. Decode does slow
as the cache fills: at k=7, ~127 tok/s at short context against ~55 at 65k.
Ordinary conversation stays in the fast regime, and long documents want the
k=3 profile ([Depth profiles](#depth-profiles)), which is ~40% faster than
k=7 at 65k. The measured curve is in
[Limitations](#hardware-cost-power-limitations).

**Seconds to answer**, AIME f01, decode only, n=5 — published including where
it goes against us:

| | seconds | completion tokens |
|---|---:|---:|
| ours, k=7 | 6.90 ± 0.30 | 1,513 ± 44 |
| NInfer, draft 5 | **6.56 ± 1.34** | 1,403 ± 253 |
| NInfer, draft 3 (their published default) | 7.73 ± 1.40 | 1,464 ± 238 |

Their point estimate is 4.9% ahead at their best depth; we are 10.7% ahead of
their published configuration; our latency variance is 4.5× tighter. This
excludes prefill, which is theirs by ~4×, so a full end-to-end figure would
favour them by more than 5%.

---

## Quick start

Requires 4× SM70 GPUs, CUDA 12.8+, and the
[1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) 1.2.2 wheel. The bootstrap
pins **`tilelang==0.1.10` and `apache-tvm-ffi==0.1.10`** together. Earlier
tilelang does not build on Volta and fails later inside GDN attention where it
looks like a kernel bug; and tilelang does not pin its own `apache-tvm-ffi`,
so a machine built today otherwise resolves a newer one that **aborts on
import** (`tvm::ffi::Error: TypeAttr __ffi_repr__ is already registered`).

pip will print a red dependency-conflict block during this step, because the
1Cat-vLLM 1.2.2 wheel declares the 0.1.9 versions of both. **That is expected
and the bootstrap says so** — the install is correct, and the bootstrap now
fails loudly if `import tilelang` does not work afterwards.

```bash
# Reproducible path: no overrides. The wheel URL and its SHA256 are pinned in
# the script, and the digest is verified before installation.
bash scripts/bootstrap-sm70.sh

# The checkpoint is pinned to an immutable revision. Any other revision is a
# different set of weights and the published numbers do not describe it.
hf download RadixArk/Qwen3.8-27B-NVFP4 \
  --revision 554ebba9b5f1b79dc11246341960360e6ef05ef4 \
  --local-dir ./Qwen3.8-27B-NVFP4

bash scripts/serve-qwen38-native.sh ./Qwen3.8-27B-NVFP4
```

Supplying your own `VLLM_WHEEL` requires `VLLM_WHEEL_SHA256` with it — the
bootstrap fails closed on an unverified wheel rather than installing it. The
serve script accepts any directory containing a `config.json`, which is a
convenience for local experiments; **it is not the reproducible path**, and
results from an unpinned checkpoint are not comparable to the ones here.

### Depth profiles

Speculation depth is a **boot-time** setting in v1.1 (per-request selection is
v1.2 work). `K` and `MML` are environment overrides the launcher already
honours, so each profile below is a real command rather than a suggestion —
and the boot gate verifies that the depth you asked for is the depth the
engine served, so a profile that fails to take exits non-zero instead of
quietly serving k=7:

```bash
CKPT=/path/to/Qwen3.8-27B-NVFP4

# general, short/medium context — the default, and what the benchmarks use
bash scripts/serve-qwen38-native.sh $CKPT

# long context — k=3 measures 76.3 tok/s at ~65k against k=7's 54.7 (+39%),
# and beats speculation-off (65.5) too. Declaring the larger window is free.
K=3 MML=196608 bash scripts/serve-qwen38-native.sh $CKPT

# structured extraction, short context only — k=15 wins on high-acceptance
# cells (doc2json 392.1 tok/s) and loses badly on prose. Do not use it as a
# general default, and do not use it at long context.
K=15 bash scripts/serve-qwen38-native.sh $CKPT
```

**Pick k by context length, not by taste.** The drafter runs k sequential
steps per round and each one reads the whole KV cache, so a step costs ~0.817
ms at short context but ~3.65 ms at 65k — while the tokens it wins barely
improve (1.55 accepted/round at k=3 against 1.63 at k=7). Deep speculation
stops paying as context grows, which is why the long-context profile is
shallower rather than off.

`bootstrap-sm70.sh` installs the pinned engine, deploys the fork patches with
backups, and builds the kernels. It refuses a wheel that is not 1Cat-vLLM
1.2.2 — every published number here was measured on it — and verifies
`VLLM_WHEEL_SHA256`. Overriding the wheel without a digest is refused.

`serve-qwen38-native.sh` starts an OpenAI-compatible server on port 8000,
**bound to 127.0.0.1**. The server has no authentication, so reaching it from
another machine should be an SSH tunnel:

```bash
ssh -N -L 8000:127.0.0.1:8000 you@your-box
```

Binding a public interface requires saying so explicitly
(`HOST=0.0.0.0 I_UNDERSTAND_THIS_IS_UNAUTHENTICATED=1 …`), and even then it
belongs behind a firewall — vLLM's own security guidance is that its API keys
do not protect every endpoint. It aborts before boot if the GPUs are
occupied, then **refuses to report success unless the release gates pass**:
served depth equals requested `k`; `lm_head` is served from the checkpoint's own codes; no
`lm_head` repack fallback occurs; the resolved KV storage is FP16; the fast
XQA/tensor-core decode-attention route is observed with zero `scalar_paged`
calls; and the expected QPN2/QPN8 decode routes are present. A mis-booted
server exits non-zero rather than quietly producing unquotable numbers.

---

## How QPN works

The naive approach to low-bit weights on old hardware is: dequantize a tile to
FP16 in memory, then call a GEMM. That spends HBM bandwidth on the format you
were trying to avoid.

The QPN decode and verification paths do not do that. **Packed low-bit weights
stay compressed through HBM and are decoded only at the point of consumption**,
directly into the FP16 register operands `mma.sync.m8n8k4` requires. On those
paths, the weight stream exists in FP16 only in registers for the instant it
is consumed. Large-M FP8 prefill above the packed-kernel range is a separate,
transient reconstruction path described below.

The mapping is the hard part. `m8n8k4` is the one FP16 tensor-core op Volta
exposes and it is notoriously difficult to feed. QPN puts **all four quadpairs
on the N dimension, sharing one activation tile**: a single warp instruction
issues four independent 8×8×4 MMAs against the same 8×4 activation fragment,
so activation traffic per weight byte drops 4×. Weights are pre-permuted at
load time into fragment order — nibble-interleaved so the decoder's output
register pair *is* the B-fragment the instruction wants, with no shuffle in
the inner loop. No shared memory in the main loop, one barrier total, 56
registers, zero spills
([`docs/qpn_race_notes.md`](docs/qpn_race_notes.md)).

**QPN2** is the NVFP4 (e2m1 + FP8 group-16 scale) instantiation. **QPN8**
generalizes the same execution architecture to FP8 e4m3 — one contiguous byte
stream, a simpler decoder, and the same quadpair-on-N geometry. **MT=2** issues
two 8-row tiles against a single weight pass, which is what makes the FP8
M 9–16 band viable for k=15. It exists for QPN8 only: there is no
`gemm_qpn2_mt2`, so NVFP4 at M 9–16 falls back to the first-generation
`gemm_qpn`.

One lesson worth recording: the first correct-looking version overflowed FP16
on real activation outliers. Only real-weight, real-activation tests exposed
it. The fix folds the global scale into the group scales in-kernel with a
per-16-element FP32 flush.

---

## Native mixed-precision checkpoint path

The checkpoint is served as published, which requires the loader to do three
things it previously would not:

1. **Dispatch per region, and the two regions have different reach.**

   | M | NVFP4 regions (MLP trunk) | FP8 regions (attention projections) |
   |---|---|---|
   | 1–8 | `qpn2` | `qpn8` |
   | 9–16 | `qpn` | `qpn8-mt2` (two 8-row tiles, one weight pass) |
   | 17–96 | Marlin | `qpn8-chunked` |
   | > 96 | Marlin | `qpn8-prefill-reconstruct` |

   **FP8 stays on packed QPN8 kernels through M=96; above that it remains under
   v100-skinny dispatch but uses transient reconstruction. NVFP4 hands off at
   M=17.** The asymmetry is deliberate on one side and a consequence on the
   other. For FP8, chunking was measured to beat transient reconstruction
   all the way to M≈112 — by 3.1× on all three protected shapes — so the
   boundary sits at 96. Above it, weights are reconstructed transiently from
   the packed codes and never persisted.

   For NVFP4 the WMMA path that nominally covers 17–64 requires the
   checkpoint-native code stash, and production drops that stash
   (`VLLM_SKINNY_DROP_CT=1`) once the QPN prepack exists, to reclaim its
   memory. So M ≥ 17 falls to Marlin — a memory decision with a routing
   consequence, not a designed handoff at 17. Decode and speculative verify
   never enter that band.

   Every route is logged and counted, and the census is checked for fallbacks
   after every boot.
   The FP8 regions are **the attention projections of every layer, two per
   layer, 128 modules per rank**: 48 GDN layers contributing
   `linear_attn.in_proj_qkvz` and `linear_attn.out_proj`, and 16
   full-attention layers contributing `self_attn.qkv_proj` and
   `self_attn.o_proj`. They are not 128 transformer layers — the model has 64.
   Everything else, the MLP trunk and `lm_head`, is NVFP4.

2. **Lower the capability gate.** `ModelOptMixedPrecisionConfig` — the config
   that governs this checkpoint — declares a minimum compute capability of
   **89**, and `ModelOptFp8Config` the same; a V100 reports 70. QPN8 is what
   makes SM70 admissible, and the prepack is verified invertible and
   byte-identical against the source weights per shape before the original is
   freed.
3. **Override the KV directive for SM70.** The checkpoint requests FP8 KV
   cache. On this Volta backend that selects scalar paged attention rather
   than the tensor-core XQA route, costing 4.82 ms per round — more than four
   times the measured cost of preserving the FP8 weight regions themselves.
   v1.1 therefore resolves the KV cache to FP16 on SM70.

`lm_head` is served by QPN2 straight from the checkpoint's own 4-bit codes;
there is no separate FP16 head and nothing is borrowed from another
checkpoint.

**What is actually resident.** NVFP4 layers hold two packed 4-bit copies — the
QPN prepack in fragment order for M≤16 and the Marlin repack that serves
prefill; the checkpoint-native staging copy is freed once the prepack exists
(`VLLM_SKINNY_DROP_CT=1`). FP8 layers hold one persistent packed 8-bit copy,
with the source tensor freed after the prepack is verified invertible. **There
is no persistent FP16 weight representation.** Decode and verification expand
weights only on-chip; M>96 FP8 prefill may materialize a transient FP16
workspace for the duration of that call.

---

## Kernel benchmarks

Both arms measured over one set of weights, one process, one sitting, on the
real per-rank shapes. Ceilings measured in the same harness: **read-only
879 GB/s, copy 822 GB/s.** Percentages use the **read** ceiling — these GEMMs
stream weights in and write a tiny M×N result, while a memcpy touches DRAM
twice, so a copy-rate denominator overstates them.

**NVFP4 trunk** (six per-rank projections; `lm_head` reported separately):

| M | route | v100-skinny | % of read ceiling | 1Cat-vLLM Marlin backport | ratio |
|--:|---|---:|---:|---:|---:|
| 1 | qpn2 | **679.5 GB/s** | **77%** | 145.6 GB/s | **4.67×** |
| 4 | qpn2 | 676.6 | 77% | 143.2 | 4.72× |
| 8 | qpn2 | 619.8 | 71% | 139.8 | 4.43× |
| 16 | qpn *(no MT=2)* | 301.9 | **34%** | 138.5 | 2.18× |

**The M=16 row is a known gap, not a Volta limit.** NVFP4 at M 9–16 runs
`gemm_qpn`, the first-generation kernel — there is no `gemm_qpn2_mt2`, so it
gets neither the QPN2 geometry nor MT=2. The FP8 side at the same M does get
MT=2, and the difference is stark:

| at M=16 | route | GB/s | % of read ceiling |
|---|---|---:|---:|
| NVFP4 | `qpn` | 301.9 | 34% |
| FP8 | `qpn8-mt2` | 558.5 | **64%** |

**1.85× apart on the same card at the same M**, purely because one has two
tiles per weight pass and the other does not. This is the largest identified
kernel gap in the stack and it is scoped work, not a research question:
porting MT=2 to QPN2 is estimated at ~3.7 ms/round at k=15, where verification
width *is* M=16. It is v1.2 work and is not in v1.1.

**FP8 attention projections:**

| M | route | v100-skinny | % of read ceiling |
|--:|---|---:|---:|
| 1–4 | qpn8 | **718.6–720.5 GB/s** | **82%** |
| 8 | qpn8 | 712.4 | 81% |
| 16 | qpn8-mt2 | 558.5 | 64% |

`lm_head` (5120 × 62,080) is excluded from the trunk aggregate — at ~179 MB
packed it would dominate any byte-weighted mean, and it is evaluated once per
token while the trunk projections are evaluated once per layer, 64 layers
deep. On its own it is the fastest shape at
**842.9 GB/s, 96% of read ceiling**, but only 2.14× the fallback, because
large-N is where Marlin is least weak.

The comparison is against the **1Cat-vLLM SM70 Marlin backport** — the
fallback QPN replaces. It is not a comparison against upstream vLLM: upstream
declares a minimum compute capability of 75 for both of its NVFP4 entry points
and raises `ValueError: … Minimum capability: 75. Current capability: 70.` at
engine construction on a V100. There is no upstream number on this hardware in
either direction.

---

## Correctness and quality

- **AIME 2026 f01, five seeds, both engines: 5/5 correct**, scored against
  independently derived ground truth (277).
- **Matched-depth acceptance diagnostic.** A separate k=3 diagnostic using the
  FP16-head control path, seeded ×3 on NInfer's verbatim fixtures, measured
  **f01 83.4 ± 0.6% for v100-skinny against NInfer's published 80.8 ± 1.8%**.
  This is not the shipping QPN2 `lm_head` path; it is retained only as a
  matched-depth acceptance control and should not be read against any k=7 or
  k=15 figure elsewhere in this document.
- **Prepack invertibility.** "Weights unchanged" means the stored FP4/FP8 code
  values and scales are preserved without arithmetic requantization; the loader
  only permutes their addresses into fragment order. The QPN8 permutation is
  inverted and asserted byte-identical against the source weights, per shape,
  before the original is freed — the packed buffer is the only persistent copy,
  so a silent corruption would be unrecoverable.
- **Kernel numerics** validated against a dequantized reference at 2.75e-4
  across 30 cases ([`results/mixed_regression_closed_20260818.md`](results/mixed_regression_closed_20260818.md)).
- **Byte-identical output** under greedy decoding across the GDN fast-path
  change and the MML sweep — the `--max-model-len` ladder produced identical
  token counts and identical τ in all eleven arms, which is what makes its
  timing deltas interpretable.

---

## Hardware, cost, power, limitations

Four Tesla V100-SXM2-16GB, TP4, NUMA-pinned to socket 0, application clocks
pinned. The GPUs are roughly **A$600 of used silicon** — that is the
accelerators alone and not the cost of the machine around them. V100-SXM2
cards are rated for 300 W each; this is not a low-power configuration, and the
case for it is capability and price, not efficiency.

**Known limitations:**

- **The full 262,144 window is reachable but marginal.** "Available KV cache
  memory" is bimodal across boots of an identical configuration — 4.68 GiB or
  4.34 GiB, unrelated to `--max-model-len`. A 262,144-token sequence needs
  4.63 GiB, so it clears a high boot by 0.05 GiB and is refused on a low one,
  where vLLM estimates 244,608 instead. **244,608 is the largest window that
  boots on both profiles**; 262,144 boots on the higher-memory profile and does
  not tax short-context round latency, but expect an occasional startup
  refusal. Any of this is possible only because Qwen3.8 is
  hybrid — 48 of its 64 layers are linear-attention and hold recurrent state
  rather than a KV cache, so only 16 layers pay per-token KV
  ([`results/mml_ceiling_20260819.csv`](results/mml_ceiling_20260819.csv)).
- **Decode slows with live context, and fixed k=7 becomes the wrong depth.**
  Declared context is free (above); live context is a different axis.
  Measured with [llama-benchy](https://github.com/eugr/llama-benchy), k=7
  falls from **127.4 tok/s at ~0.5k to 54.7 at ~65k**. Plain decode falls only
  **86.3 → 65.5**, but the correct conclusion is not "disable speculation":
  at ~65k, **k=3 reaches 76.3 tok/s**, beating both. The extra k=7 drafter
  steps each traverse the long-context state (~3.65 ms/step at 65k versus
  0.817 ms shallow), while accepted tokens barely increase (1.55 per round at
  k=3 versus 1.63 at k=7). This is a **depth-economics problem, not a QPN
  weight-kernel regression**. v1.1 selects depth at boot: use k=7 for short
  and medium contexts and the k=3 profile for long-context serving. Automatic
  per-request depth selection is deferred to v1.2. The decode-partition pin
  remains equal or better than the unpinned selector at every tested depth
  ≥65k ([`results/ctx_depth_20260819.md`](results/ctx_depth_20260819.md)).
- **Prefill is roughly 4× slower than a 5090.** Much of that gap is
  architectural: decode is the bandwidth-bound GEMV regime, where a
  weight-stream kernel at 77–82% of roofline is most of the battle, while
  prefill is the compute-bound large-M GEMM regime and Volta has no FP4 or FP8
  Tensor Cores. The large-M software path also retains headroom documented
  below, so v1.1 does not claim the entire 4× gap is irreducible. The release
  claims are about single-user decode.
- **Domain benchmark cells are n=1**; the AIME results are n=5.
- **Checkpoints are not matched in the head-to-head** — we serve RadixArk,
  NInfer's artifact is built from an unsloth NVFP4 revision that no longer
  exists upstream and cannot be reproduced.

### Known headroom

Not caveats — these do not qualify any figure above. They are measured gaps
between what the stack does and what it could do, and the numbers reported
here would *improve* if they were closed.

- **NVFP4 has no MT=2 at M 9–16.** 301.9 GB/s against FP8's 558.5 at the same
  M, because `gemm_qpn2_mt2` does not exist. M=16 is the k=15 verification
  width, so this is the band k=15 runs in: porting MT=2 to QPN2 is estimated
  at ~3.7 ms/round at k=15. The k=7 results are untouched — they verify at
  M=8. Largest identified gap in the stack.
- **NVFP4 above M = 16 runs on Marlin.** The WMMA path covering 17–64 needs
  the checkpoint-native code stash, which production frees rather than hold
  two packed copies of every weight. Prefill band only; the V100 WMMA plateau
  there is structural anyway. FP8 is unaffected — `qpn8-chunked` covers 17–96.
- **The FP8 chunked path re-reads the weight stream once per 8 rows**, so
  useful bandwidth at M=32 is about a quarter of the M ≤ 8 figure — time is
  linear in M (113.1 µs at M=32, 223.5 at M=64, i.e. 4 and 8
  chunks). Prefill band, not decode. Extending MT=2 up the M range is the same lever as the first item.

---

## Reproduction

Every number in this document is traceable to a committed file in
[`results/`](results/), which is the canonical source; this README restates
rather than redefines.

| claim | file | harness |
|---|---|---|
| kernel bandwidth, both arms | [`results/kernel_matched_20260819.csv`](results/kernel_matched_20260819.csv) | [`benchmarks/kernel_matched_bench.py`](benchmarks/kernel_matched_bench.py) |
| launch table, three depths | [`results/v11_shipped_table_20260819.md`](results/v11_shipped_table_20260819.md) | [`benchmarks/v11_suite.py`](benchmarks/v11_suite.py) |
| AIME + seconds to answer | [`results/aime_partfix_20260819.md`](results/aime_partfix_20260819.md) | [`benchmarks/ninfer_repro.py`](benchmarks/ninfer_repro.py) |
| head-to-head vs NInfer | [`results/headtohead_5090_20260819.md`](results/headtohead_5090_20260819.md) | [`benchmarks/v11_suite.py`](benchmarks/v11_suite.py) |
| speculation depth economics | [`results/depth_curve_ours_20260819.md`](results/depth_curve_ours_20260819.md) | — |
| decode-partition fix | [`results/partition_fix_20260819.md`](results/partition_fix_20260819.md) | [`benchmarks/v11_suite.py`](benchmarks/v11_suite.py) |
| decode vs live context | [`results/ctx_depth_20260819.md`](results/ctx_depth_20260819.md) | [llama-benchy](https://github.com/eugr/llama-benchy) |
| FP16-KV policy (+4.82 ms/round) | [`results/mixed_regression_closed_20260818.md`](results/mixed_regression_closed_20260818.md) | [`benchmarks/v11_suite.py`](benchmarks/v11_suite.py) |

The harnesses above are portable and take their paths from the environment.
The per-experiment boot drivers that orchestrated these runs are
machine-specific and are not shipped; `serve-qwen38-native.sh` is the
supported way to bring up the configuration they used.

Both engines are driven through the same harness and the same metric
definitions. Sampling fields are always sent explicitly — neither engine is
left to fill defaults from its own model card — and streaming reads both
`reasoning` and `reasoning_content`, since reading only one places
time-to-first-token after the entire reasoning phase.

---

## Credits and lineage

```
vLLM  ->  1Cat-vLLM (SM70)  ->  v100-skinny
```

**[1Cat AI](https://github.com/1CatAI/1Cat-vLLM)** made modern vLLM run on
Volta. Their distribution also ships `flash_attn_v100`, a native
FlashAttention for SM70 — 15 MB of compiled `sm_70` device code, with paged
KV, split-KV prefill, XQA and WMMA decode paths and quantized-KV decode.
Upstream FlashAttention requires *"Ampere, Ada, or Hopper"*; Turing is served
only by a separate third-party partial port; Volta is a generation below that
and is not listed at all. **Every attention number here depends on their work,
and there would be no stack to build on without it.**

**[vLLM](https://github.com/vllm-project/vllm)** is the engine both forks
derive from.

**v100-skinny** contributes the QPN execution architecture — QPN2, QPN8 and
MT=2 — the native mixed-checkpoint loader and dispatch path that makes the
published representation executable on SM70, and the serving fixes above. The
files in [`fork_patches/`](fork_patches/) are 1Cat-vLLM's, carrying our edits,
and remain Apache-2.0.

**License:** MIT for the kernels, harnesses, scripts, docs and results;
Apache-2.0 for [`fork_patches/`](fork_patches/). See
[`LICENSE`](LICENSE), [`LICENSE-APACHE-2.0`](LICENSE-APACHE-2.0) and
[`NOTICE`](NOTICE).

---

## Citation

```bibtex
@software{vertzyas_v100skinny_2026,
  author  = {Vertzyas, Dennis},
  title   = {v100-skinny: native mixed NVFP4/FP8 LLM inference on NVIDIA Volta},
  version = {1.1},
  year    = {2026},
  url     = {https://github.com/dnv2003/v100-skinny}
}
```
