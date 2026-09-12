# PR-Entwurf E-2: Block-gepackte Aktivierungen für die QPN2-Kernel

Branch `sm70-qpn2-block-pack` im Worktree `1Cat-vLLM-pr-turing-ops` (auf
E-1 aufgesetzt, für den PR auf main zu rebasen; nur `nvfp4_qpn2_sm70.cu` und
ein neuer Test). Stand 12.09. mittags: Code fertig, clang-format grün, Bau
und Messung laufen. NICHT committet, NICHT eröffnet.

## Commit-Message

```
[Perf][SM70] Block-pack the activations of the NVFP4 QPN2 kernels

The m8n8k4 A fragment hands lane L the activation row (L & 3) + 4 * (L >> 4),
so a warp reads eight rows per 16-k group; from the row-major activations
those rows lie k * 2 bytes apart and every group costs eight cache lines.
A tiny pack kernel lays the activations out as [k / 16][rows][16] so the
same eight rows are one contiguous block, and the GEMM and gated GEMM read
that layout through a template parameter; values, order and the fp32
accumulation are unchanged. The pack pays from M=5 on Turing (64 KB L1)
and from M=8 on Volta (128 KB L1) and is skipped below that;
VLLM_SM70_NVFP4_QPN2_PACK=0/1 forces either side for A/B runs.

Co-authored-by: Claude Fable 5.1 <noreply@anthropic.com>
Signed-off-by: Peuqui <peuqui@github.com>
```

## PR-Body (Kern)

Purpose: Belege aus dem Fork (STAND Punkt 8, 09.09.): Kernel-Trunk-Summe bei
M=8 1,45× auf der RTX 8000 (sm75-Bau) / 1,39× (sm70-Bau), 1,04× auf der
V100; DFlash2 27B RTX 69,22 → 72,72 tok/s, SHA unverändert; Ursache war die
Zeilenstreuung (L1-Wavefronts 300k → 1,58 Mio. auf 64 KB L1), nicht die
MMA-Form (m8n8k4 = m16n8k8 auf Turing, gemessen).

Test Plan: (1) pre-commit clang-format; (2) neuer GPU-Test
`tests/kernels/quantization/test_nvfp4_qpn2_block_pack.py`: GEMM und gated
GEMM bitgleich mit und ohne Pack für M 1..32 (V100 und RTX); (3) E2E 27B
MTP RTX (E-1-Pfad) Pack auto gegen Pack 0, SHA und tok/s; (4) Kernel-A/B
je Form (Ergebnisse eintragen).

Not a duplicate: #561 (shared QPN2 layout) ändert die Gewichtsseite, nicht
die Aktivierungsseite; Prepack unverändert (geprüft 12.09.).


## Nachtrag 12.09. nachmittags: Messmethodik

- Kernel-Test `test_nvfp4_qpn2_block_pack.py`: **58/58 bitgleich auf V100 und
  RTX 8000** (GEMM und gated GEMM, M 1..32 inkl. Zwei-Kachel-Bereich, alle
  Split-K/Ketten-Kombinationen, Formen 3584/5120, 8704/5120, 5120/1536,
  5120/4352, 62080/5120; Schalterzustände 0/1/auto/weg).
- **E2E-SHA ist unter der 0DOT3-Config KEIN Kriterium:** jeder frische
  Compile wählt Combo-Kernel per Zeitmessung (Münze), der AOT-Pfad friert die
  Wahl pro Env-Hash im Inductor-Cache ein. Belegt mit 12 Boots (siehe
  `paket-e-plan.md`). Deshalb: Schalter `VLLM_SM70_NVFP4_QPN2_PACK` in
  `envs.py` registriert (str, "auto") und in `compile_factors` ignoriert →
  beide A/B-Arme gleicher Hash, gleiche gecachte Kernelwahl, nur der Pack
  unterscheidet sich.
- DFlash2 auf main+Turing gesperrt (Drafter braucht nicht-kausale Attention,
  `FLASH_ATTN_V100` prüft exakt SM70) → Tempo-Beleg über MTP k=4 (M=5) und
  k=7 (M=8) auf der RTX; k=3 (M=4) als Kontrolle unter der Schwelle.
- Im PR-Text zusätzlich der Fork-Beleg DFlash2 RTX 69,22 → 72,72 tok/s.


## Ergebnisse 12.09. nachmittags (RTX 8000, 27B MTP, gleicher Env-Hash, Triton-Attn, fp16-KV, 5 × 400 Token)

| k (Verify-M) | ohne Pack | mit Pack (auto) | Δ | SHA / Annahme |
|---|---:|---:|---:|---|
| 3 (M=4) | 68,8 | 68,9 (ungepackt, unter Schwelle) | — | `cb2d4b…` / 2,920 beide |
| 4 (M=5) | 69,7 | 69,4 | −0,4 % | `cb2d4b…` / 3,053 beide |
| 5 (M=6) | 71,4 | 71,6 | +0,3 % | `cb2d4b…` / 3,390 beide |
| 6 (M=7) | 66,5 | **67,8** | **+1,9 %** | `cb2d4b…` / 3,419 beide |
| 7 (M=8) | 64,4 | **66,7** | **+3,7 %** | `cb2d4b…` / 3,571 beide |

**Schwelle Turing deshalb 7 (nicht 5 aus dem Mikrobenchmark); Volta 8
(Fork-Messung, hier nicht E2E messbar, weil modelopt auf Volta TurboMind
nimmt).** Spannen je Lauf ≤ 0,5 tok/s. Innerhalb jedes Paares gleiche SHA
und Annahmelänge (gleicher Env-Hash, registrierter Schalter).
Fork-Referenz DFlash2 (M=8): 69,22 → 72,72 (+5,1 %). PR-Worktree
`1Cat-vLLM-pr-blockpack` (Branch `sm70-qpn2-block-pack-pr` auf main).
