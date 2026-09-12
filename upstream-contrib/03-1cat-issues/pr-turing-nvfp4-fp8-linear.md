# PR-Entwurf E-1: ModelOpt NVFP4 und FP8 Linears auf Turing über die SM70-QPN-Kernel

Worktree `1Cat-vLLM-pr-turing-ops`, Branch `sm70-ops-on-turing` auf
`origin/main` (ae75fb9b). Stand 12.09.: Code fertig, pre-commit + mypy grün,
CPU-Test 13, GPU-Test 6 (V100), E2E läuft. NICHT committet, NICHT eröffnet.
`setup.py` trägt die lokale #601-Nachhilfe — NICHT in den PR.

## Commit-Message

```
[Feature][SM75] Run ModelOpt NVFP4 and FP8 linears on Turing through the SM70 QPN kernels

The compiled SM70 ops carry the QPN2 (NVFP4) and QPN8 (FP8) decode kernels
and the QPN8 dense prefill, none of which depend on the TurboMind GEMM
registry, but every gate admitted exact SM70 only and the registry itself
is Arch<700, 750>. A Turing worker therefore took Marlin for NVFP4 and,
for the mixed ModelOpt checkpoints, did not load at all (minimum
capability 89; with the gate lowered the Marlin FP8 path fails on
orig_dtype).

Turing now prepares NVFP4 linears as the QPN2 prepack only and runs the
QPN2 kernels for M <= 32 and a Triton dequantization into a transient
fp16 buffer plus cuBLAS above that; FP8 per-tensor linears become QPN8
codes with channel scales and run the existing QPN8 dispatch (QPN8
kernels for small M, dequantization plus cuBLAS for prefill). Gates read
the worker's own device. Volta keeps its TurboMind path unchanged.

Co-authored-by: Claude Fable 5.1 <noreply@anthropic.com>
Signed-off-by: Peuqui <peuqui@github.com>
```

## PR-Body

## Purpose

On a Turing (sm_75) worker the ModelOpt NVFP4 linear method falls back to
Marlin and a `modelopt_mixed` checkpoint (NVFP4 MLP, per-tensor FP8 GDN
projections, e.g. RadixArk/Qwen3.8-27B-NVFP4) does not load at all: the
mixed config requires capability 89, and with that gate lowered the generic
FP8 fallback fails in `prepare_fp8_layer_for_marlin` (`orig_dtype`).

The compiled SM70 ops already contain what Turing needs: the QPN2 decode
kernels (`nvfp4_qpn2_sm70.cu`) and the QPN8 decode kernels plus the QPN8
dense prefill (`fp8_qpn8_sm70.cu`, dequantization into a workspace and
`at::mm`). Neither depends on the TurboMind GEMM registry, whose kernels
are registered for exact SM70 (`Sm70` is `Arch<700, 750>` in `arch.h`).
The sm_70 cubins run on Turing by binary compatibility. What kept Turing
out were the gates: `is_exact_sm70_cuda` / `is_exact_sm70_cuda_platform`
admit `(7, 0)` only, and the NVFP4 prepare always builds the TurboMind
weight, whose apply would call into the empty registry.

This change adds a Turing route next to the Volta one:

- `sm70_turbomind.py`: `is_turing_cuda`, `is_turing_cuda_platform`,
  `is_pre_ampere_cuda_platform` (all judged on the worker's own device),
  `should_prepare_turing_qpn2`; `prepare_nvfp4_qpn2_dense_linear` builds
  the QPN2 prepack as the only resident layout (`op_kind` `nvfp4_qpn2_dense`,
  launch configuration from a small table plus a heuristic);
  `apply_prepared_linear` runs `nvfp4_qpn2_gemm_sm70_out` for M <= 32 and
  `nvfp4_qpn2_dense_linear` above; `prepare_fp8_qpn8_dense_linear` /
  `apply_prepared_fp8_qpn8_linear` broadcast the per-tensor scale to channel
  scales for `fp8_qpn8_prepare_sm70` and call `fp8_qpn8_dispatch_sm70_out`
  with a cached fp16 `[K, N]` workspace.
- `utils/nvfp4_qpn2_dequant.py`: Triton dequantization of the QPN2 prepack
  (derived from v100-skinny, MIT, like the kernels themselves) as a
  registered custom op, plus a pure-torch reference.
- `modelopt.py`: the NVFP4 prepare hook takes the Turing route when the
  Volta route does not apply; `ModelOptFp8LinearMethod` takes the QPN8
  route on Turing; the mixed config's minimum capability uses the
  pre-Ampere platform check.

Why dequantize for prefill instead of a second packed layout: Marlin's FP4
GEMM reaches about 27 TFLOPS on Turing, cuBLAS fp16 does better, and a
second resident layout would double the weight memory (about 10 GiB per
layout on the 27B). This is the same shape as the existing QPN4 / QPN8
prefill ops.

Not in this change: AWQ, MXFP4, GPTQ and uint4 (no Turing measurement),
the MoE methods, attention. Two Turing issues sit outside the linear path
and were worked around in the tests: the FlashInfer backend selected by
default on Turing fails in `BatchPrefillWithPagedKVCache` (invalid
argument), so the tests use `--attention-backend TRITON_ATTN`; and the
checkpoint's FP8 KV cache makes Inductor emit an `fp8e4nv` cast that
Triton rejects on sm_75, so the tests run `--kv-cache-dtype float16`.

## Test Plan

1. `pre-commit run --files <changed files>` and the `mypy-3.10` hook.
2. New CPU test `tests/quantization/test_sm70_turbomind_turing_gates.py`
   (routes per capability, backend switch, launch configuration, padding).
3. New GPU test `tests/kernels/quantization/test_nvfp4_qpn2_dequant.py`:
   random NVFP4 codes and e4m3 scales -> `nvfp4_qpn2_prepare_sm70` ->
   Triton dequantization, compared bit-exactly with the direct checkpoint
   dequantization and the pure-torch inverse; and the dense linear against
   an fp16 matmul. Run on a Tesla V100 and a Quadro RTX 8000.
4. End to end, RadixArk/Qwen3.8-27B-NVFP4 (`modelopt_mixed`), TP2, greedy,
   5 x 400 tokens, `--attention-backend TRITON_ATTN --kv-cache-dtype
   float16`, with MTP k=3 and without speculation: 2x Quadro RTX 8000 with
   this change (the checkpoint does not load there on `main`) against
   2x Tesla V100 on the unchanged Volta path.

## Test Result

1. All applicable hooks passed; mypy-3.10 passed.
2. 13 passed.
3. 11 passed on the Tesla V100 and 11 passed on the Quadro RTX 8000
   (dequantization bit-exact against the checkpoint dequantization and the
   pure-torch inverse for five shapes; dense linear bit-exact against the
   fp16 matmul; the run-time dispatch agrees with cuBLAS to fp16 rounding
   on both sides of the M=32 threshold).
   `tests/quantization/test_sm70_modelopt_mixed_nvfp4.py`: 64 passed,
   6 skipped (the minimum-capability test now covers Volta and Turing);
   `tests/quantization/test_sm70_nvfp4_qpn2.py`, `test_sm70_modelopt_nvfp4.py`,
   `test_sm70_turbomind_adapter.py`, `test_sm70_glm53_modelopt_nvfp4.py`,
   `test_nvfp4_mxfp4_gating.py`, `test_sm70_mxfp4_moe.py`: all passed.
4. RadixArk/Qwen3.8-27B-NVFP4, TP2, MTP k=3, greedy, 400 tokens, five
   runs each, `--attention-backend TRITON_ATTN --kv-cache-dtype float16`:

   | | decode tok/s (median) | acceptance length | text SHA-256 (400 tokens) |
   |---|---:|---:|---|
   | 2x Tesla V100, `main`, TurboMind path | 63.4 | 3.000 | `38848c08a44405ae` |
   | 2x Quadro RTX 8000, this change, QPN2 + QPN8 | 71.0 | 3.000 | `38848c08a44405ae` |
   | 2x Quadro RTX 8000, `main` | does not load (minimum capability 89) | | |

   Same text on both card generations. Decode profile on the RTX 8000
   (one rank, nsys, 5.19 s of kernel time in the window): `nvfp4_qpn2_sm70`
   31 %, NCCL AllReduce 21 %, `fp8_qpn8_sm70` 18 %, Triton attention 6.5 %,
   `_nvfp4_qpn2_dequant_kernel` absent from the decode steps (it runs on
   the prefill only).

   A first version branched on M in Python; torch.compile traced that
   branch at the warm-up M and kept the dense path in the decode graph
   (17 tok/s, the dequantization at 64 % of the window). The split now
   happens inside the registered op, the same shape as the C++ dispatchers.

## Not a duplicate

Checked on 2026-09-12 against `1CatAI/1Cat-vLLM`: `gh pr list --state open
--search` for "turing", "sm75", "qpn2", "qpn8", "pre-ampere", "RTX 8000",
`gh issue list --search` for "turing", "sm75", "RTX 8000", and
`gh pr diff --name-only` over every open PR for `sm70_turbomind.py`,
`quantization/modelopt.py`, `nvfp4_qpn2*`, `fp8_qpn8*`. Related, not
duplicates: #576 (mine) makes `is_exact_sm70_cuda_platform` answer for
the worker's own device; this change adds the Turing and pre-Ampere
platform helpers next to it and touches different lines. #561 adds a
second, shared QPN2 layout for DFlash2 and keeps the current prepack as
the control layout; the dequantization here follows the current
`nvfp4_qpn2_prepack_codes_kernel` layout and its test compares against the
prepack op bit for bit, so a layout change would fail loudly rather than
silently. #441 (mine) is the discussion thread for pre-Ampere tuning.
#255 is the Volta side of the capability gate (min 75 vs 70), not Turing.

AI assistance (Claude) was used to trace the gates and the registry limit,
port the dequantization and run the measurements; I reviewed every line and
ran the tests above.
