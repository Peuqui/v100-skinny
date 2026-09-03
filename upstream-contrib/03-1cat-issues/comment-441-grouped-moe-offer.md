# Entwurf: Kommentar in #441 — Angebot grouped NVFP4-MoE-Decode-Kernel (pre-Ampere)

> Status: ENTWURF, wartet auf Freigabe Peuqui. Ziel: Interesse abfragen,
> KEIN PR — Portierung in deren csrc/sm70_turbomind wird als Folge-PR
> angeboten, falls die Maintainer Interesse signalisieren.
> Zahlen-Quellen: scripts/nvfp4_skinny_moe_qpn_test.py (Standalone, echte
> DSv4-Layer-5-Gewichte, V100 + RTX 8000), Boot 64/65 (Kohärenz + nsys).

---

Hi @yangzhuxinyzx — following up on the pre-Ampere work from #469: we have
built a grouped NVFP4 MoE decode kernel for pre-Ampere cards in our local
fork and are wondering whether upstream would be interested in a port.

What it does:

- One launch per weight matrix and layer for the whole decode/verify batch
  (tokens <= 8). Routing is device-side (argsort + scatter_add + cumsum),
  so there is no host sync and the kernel captures into CUDA graphs
  without an eager break.
- Compute is mma.m8n8k4 (HMMA.884) on fragment-order prepacked weights
  with fp32 accumulation. The same SASS runs on sm70 and sm75 (binary
  compatible), and the measured best launch configs are identical on
  V100 and RTX 8000, so there is no per-arch code path.
- The weight prepack is a byte-equal permutation applied in place at
  load time — no extra VRAM, parameter shapes unchanged.
- It is shape-generic rather than contract-gated: the only requirements
  are N % 32 == 0 and K % 64 == 0 per expert matrix, any expert count and
  top-k. We currently serve DeepSeek-V4-Flash (64 experts/layer, top-k 6,
  4096 hidden, 2048 intermediate) with it, which the current
  nvfp4_sm70_moe.py contract list does not cover.

Numbers on real DeepSeek-V4-Flash expert weights (decode/verify point,
6 tokens, ~29 active experts): w13+w2 kernel time 0.58 ms/layer vs
0.87 ms for our previous SIMT grouped kernel — about 1.35x of the
weight-traffic floor. Full layer (routing + both GEMMs + activation +
combine) 1.40-1.46x faster on V100 and 1.27-1.36x on RTX 8000. In
5-GPU pipeline-parallel serving (2x RTX 8000 + 3x V100, DSpark K=5)
this took a decode step from ~144 ms to ~113 ms; greedy outputs stayed
bit-identical to our reference at the usual quota.

To check the shape-generic claim on a second, very different geometry we
also ran it at the Qwen3.8 Flash Next TP1 expert shapes (512 experts,
top-k 10, w13 1280x2560, w2 2560x640) with synthetic NVFP4 bytes
(timing is weight-traffic-bound, so synthetic bytes are representative;
correctness anchored against the checkpoint-layout SIMT kernel on the
same bytes): w13+w2 3.2-3.4x faster at 1 token and 2.5-2.9x at 6-8
tokens vs our SIMT grouped kernel, identical numerics on V100 and
RTX 8000. One honest limitation: each expert matrix needs K % 64 == 0,
so the FN TP4 W2 shard (K=160) would need padding or a relaxed tail —
TP1/TP2 shapes are fine.

We also benchmarked directly against your compact grouped route (the
1.5.0 wheel's nvfp4_moe_dense_stage_sm70_out with one-row-per-group
compact offsets, exactly as your decode path stages it; identical
pre-built routing on both sides; both routes checked against an fp32
dequant reference — both land in the fp16 rounding class). On real
weights, V100:

- Qwen3.6-35B-A3B (your smallest contract, E256/topk8, tiny experts):
  1.32x at T=1, 1.05-1.13x at T=2-8.
- DeepSeek-V4-Flash (E64/topk6, large experts, real expert sharing):
  1.22x at T=1, 1.27x at T=6 (the speculative verify point), 1.38x
  at T=8.

Two design points drive this: the launch grid runs over
slot-count-many compact groups (a static bound, so it still captures
into CUDA graphs) instead of all experts, and each weight read serves
up to 8 routed rows instead of one — one-row-per-group re-reads a
shared expert's weights per slot, which is exactly the
speculative-verify regime. An earlier version of our kernel gridded
over all experts and lost the tiny-expert T=1 lane to your route
(0.67x); the compact grid flipped that, so we can confirm the empty
blocks were the entire gap. On RTX 8000 the TurboMind op currently
aborts ("No feasible kernel found ... sm75_f16_e2m1k16..."), while
this kernel runs on sm70 and sm75 with the same launch configs.

The kernel currently lives in our fork as a torch cpp_extension next to
the skinny GEMM family, not in csrc/sm70_turbomind. If this is something
you would take, we would port it to your ops structure and benchmark it
against the compact grouped route on main (per AGENTS.md, with tests).
Happy to share the standalone test/bench script in the meantime.

---

## Nicht behauptet (bewusst weggelassen)
- Keine tok/s-Werbezahlen über die Step-Zeit hinaus (Akzeptanz-Lotterie).
- Kein Vergleich der Prefill-Pfade (deren indexed/fused-Routen ungetestet).

## Beleg-Quellen
- A/B: tools/moe_ab_1cat.py + benchmarks/moe-ab-1cat-2026-09-03.txt
  (1.5.0-Wheel in .venv-sm70-150, echte Qwen3.6- und DSv4-Bytes).
- Eigenzahlen: scripts/nvfp4_skinny_moe_qpn_test.py (DSv4 real),
  scripts/nvfp4_skinny_moe_qpn_synth_fn.py (FN-Geometrie synthetisch).
