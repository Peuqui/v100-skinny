## Purpose

Turing (SM75) runs the SM70 kernel family through binary compatibility with sm_70 code, and the hand kernels use Volta's mma.m8n8k4, which Turing executes at half its tensor rate. This PR gives the family native Turing paths where they pay off, keeps every Volta path as it is, and bounds one grid for both. Eleven commits in three groups:

1. Block-FP8 QPN8 linears on Turing (77006c020, 4fc21d426, c4b72f197) and the occupancy rule for the M <= 8 split-K (f5486ee76, 69dfe6906).
   - A new m16n8k8 GEMM decodes and block-scales the packed QPN8 codes once per CTA while staging them in shared memory (bit-identical to the dense reconstruction), reads both operands through ldmatrix, rasterises row tiles in groups and splits K across CTAs with a fixed-order FP32 reduction. No second weight layout.
   - Turing runs M = 9..512 through it. M <= 8 keeps the packed m8n8k4 GEMM, which timed in CUDA graphs was faster at decode on 10 of 13 DeepSeek-V4 and Qwen3.8-27B shapes. Above M = 512 the dense prefill stays.
   - Split-K for the M <= 8 GEMM: on Turing (1024 threads per SM) a split-16 CTA fills half an SM, so N = 5120 spills a second, almost empty wave; split 8 fits it into one wave. Volta keeps its table (split 8 loses there on spills). A split-32 rule for small grids was measured and dropped again (no step-time gain, see the commit message).
2. Grouped skinny MoE (f2d894f86, f852ca00e, c27eeae9f, f178ad557).
   - The grid spans min(routing slots, experts) rows instead of one row per slot; a batch never has more groups than experts. Bit-identical.
   - From SM75 on, expert weights are packed at load in the m16n8k8 fragment order of the card (a pure permutation, same size, no second copy; each layer lives on one card), and a new kernel uses the weights as the 16-row A operand and the expert rows as the B operand. Volta keeps m8n8k4. The decoded weights equal the m8n8k4 path bit for bit; only the tensor core's summation order differs.
   - Volta takes over the tile structure found on Turing: two 32-column tiles per CTA sharing each staged activation fragment, eight split-K warps (16 or 10 only where 8 does not divide K).
   - mma_fragments_sm70.cuh holds the e2m1 decoder, the m16n8k8 MMA and ldmatrix once, replacing copies in nvfp4_qpn4_sm70.cu and fp8_qpn8_mma_sm75.cu.
   - The skewed-routing case (a few experts with hundreds of rows, others with a handful) is now part of the dispatch test.
3. Build (369cccdc1, d0156d51e): the SM70 family (TurboMind ops, ring collectives, sampler fragment, the H3/QSA/router/exact-reduce/sparse-attention extensions) builds for 7.0 and 7.5 through one variable, SM70_FAMILY_ARCHS. TORCH_CUDA_ARCH_LIST stays 7.0 on purpose: listing 7.5 there would turn the Marlin build mixed and drop the SM70 Marlin kernels. The SM75 branches in the TurboMind core headers only enable ldmatrix/m16n8k8 helpers that the sm70_884 kernels do not call. The architecture list is passed unquoted, because cmake_parse_arguments keeps a quoted "70;75" as one element (nvcc then received compute_70;75 and the H3 extension failed).

Not a duplicate: open PRs that touch the same files are #1030 and #1031 (skinny_moe_qpn_sm70.cu: further MXFP4 split-K values and geometries for V100), which add different split-K choices and do not change fragment order, grid or tiles. Both already conflict with main (GitHub: CONFLICTING, last updated 2026-10-07 and 2026-10-09); git merge-tree also reports content conflicts between them and this branch (3 files for #1030, 6 for #1031), so whichever lands second needs a rebase. Searches for m16n8k8, "QPN8 Turing", "Turing tensor", sm_75 and SM75 find only our own PRs, none of which changes these kernels.

AI assistance was used for this change. I reviewed every line and ran the tests and measurements below.

## Test Plan

Native build of this branch (in-place, MAX_JOBS=2, TORCH_CUDA_ARCH_LIST=7.0): _C carries 72 sm_70 and 55 sm_75 cubins.
python -m pytest tests/kernels/quantization/test_sm70_qpn8_block_fp8.py tests/quantization/test_turing_qpn_kernel_selection.py tests/kernels/moe/test_skinny_sm70_moe.py, one process per file, on a Quadro RTX 8000 and on a Tesla V100.
pre-commit run --from-ref origin/main --to-ref HEAD; pre-commit run mypy-3.10 --hook-stage manual --from-ref origin/main --to-ref HEAD

## Path coverage

| Path / provider | Format | Key conditions (shape, TP/EP, graph, capability) | Expected selection / fallback | Evidence (source / static / executed / measured) | Unverified scope |
| --- | --- | --- | --- | --- | --- |
| QPN8Fp8BlockScaledMMLinearKernel on SM75 | block FP8 (QPN8 codes) | M 9..512 | new m16n8k8 GEMM | executed: test_sm70_qpn8_block_fp8.py on RTX 8000; measured: CUDA graphs and model A/B (below) | other SM7.5 cards than the RTX 8000 |
| same on SM75 | block FP8 | M <= 8; M > 512 | packed m8n8k4 GEMM; dense prefill | executed: test_turing_decode_keeps_the_packed_gemm (bit-identical) | none |
| same on SM70 | block FP8 | all M | unchanged routes; split-K table unchanged | executed: same file on V100 | none |
| M <= 8 split-K | QPN8 FP8 block and per-tensor | Turing, grid spills a second wave | split 8 instead of 16 | executed: test_turing_qpn_kernel_selection.py; measured: CUDA graphs, 27B A/B | none |
| skinny grouped MoE, SM75 | NVFP4, MXFP4, CSF scales | prefill and decode | m16n8k8 fragment order packed at load | executed: test_skinny_sm70_moe.py on RTX 8000 (both orders); measured: CUDA graphs | SM8x+ would also pick the m16n8k8 order; only SM75 measured |
| skinny grouped MoE, SM70 | same | prefill and decode | m8n8k4 order, new tiles and split | executed: same file on V100; measured: CUDA graphs | none |
| build | n/a | TORCH_CUDA_ARCH_LIST=7.0 | SM70 family carries sm_70 and sm_75 code; Marlin unchanged | static: cuobjdump; executed: tests above | builds with other arch lists |

## Acceleration and benchmark contract (required for performance changes)

- Default enabled or opt-in: default enabled on the matching capability (per device, never device 0); no switch.
- Required CLI options and environment switches: none for the change. Model runs below used the fork's production entries: DeepSeek-V4-Flash PP=5 with DSpark (5 speculative tokens), Qwen3.8-27B NVFP4 TP=2 on two RTX 8000 with MTP (3 speculative tokens), VLLM_USE_V2_MODEL_RUNNER=1.
- KV cache dtype used for the benchmark: fp8 (DeepSeek-V4-Flash), auto (Qwen3.8-27B).
- Wheel SHA or source commit: kernel microbenchmarks against the respective previous kernel source; model A/B runs on fork builds (github.com/Peuqui/1Cat-vLLM), commits named below. The commits of this branch are rebased onto main 6a143197e; the branch itself was built and unit-tested, not booted with a full model.
- PYTHONPATH, source overlays, or external native libraries used: model runs used the fork source tree and a patched Triton 3.6 build on PYTHONPATH (identical in both arms); the kernels here do not use Triton.
- User-entry route-hit, speed, and output-quality evidence: see Test Result.
- Existing kernel/backend registry and capability-rejection reasons: selection stays in QPN8Fp8BlockScaledMMLinearKernel and the skinny MoE dispatch, keyed on the capability of the layer's own device.
- Per-engine configuration and deprecated-variable compatibility: no setting added or changed.
- New/changed environment metadata and generated reference check: none.
- Route snapshot command and intentional changed rows: none.
- SM70 registered variables / unregistered reads / model-parameter restrictions, before -> after: no change.

## Test Result

Kernel microbenchmarks (CUDA graphs, time per call, lower is better):
- QPN8 block FP8 on a Quadro RTX 8000, Qwen3.8-27B TP2 projections: M 9..64 0.16-0.29x the previous time (whole-matrix reconstruction + cuBLAS); BM128 tiles at M 128..512 a further 7-12 % faster after reading the block scale with the codes (bitwise equal in 840 tile/split configurations).
- M <= 8 split-K on Turing: o_proj/out_proj (K 3072) 0.78-0.90 and down_proj (K 8704) 0.73-0.87 of the split-16 time.
- Skinny MoE grid bound, both cards: DeepSeek-V4 512 tokens (E256 top-6) RTX 8000 w13 7070 -> 6506 us, w2 3546 -> 3250; V100 w13 5524 -> 5004, w2 3004 -> 2442; decode unchanged; bit-identical.
- Skinny MoE m16n8k8 on a Quadro RTX 8000 (same checkpoint weights in both orders): DeepSeek-V4 512 tokens w13 6555 -> 4303 us, w2 3271 -> 2238; GLM-5.3 512 tokens w13 8566 -> 5316, w2 4205 -> 2760; decode (4-6 tokens) -1.3 % .. +1.0 %; the m16n8k8 order stays within 6.3e-4 of the output maximum and matches the fp32 reference as closely as the m8n8k4 order.
- Skinny MoE tiles on a V100: DeepSeek-V4 512 tokens w13 4979 -> 4374 us, w2 2441 -> 2395; GLM-5.3 512 w13 6878 -> 5954, w2 3567 -> 3218; decode 0.80-0.97x.
- sm_75 build: grouped skinny MoE 1-4 % faster on the RTX 8000 (one decode shape 10 %), bit-identical, unchanged on V100.

Model A/B/A/B on fork builds (3 Tesla V100 + 2 Quadro RTX 8000):
- Qwen3.8-27B FP8, TP=2 on the two RTX 8000, QPN8 Turing GEMM (77006c020 against the previous production build): decode 56.3-56.9 -> 56.9-57.4 tok/s (prose), 82.6-84.5 -> 83.2-85.1 (code); at 184k context 67.1 -> 66.8-66.9 ms per step; prefill unchanged; greedy output and eight quality answers byte-identical in all counted boots.
- DeepSeek-V4-Flash PP=5 with the M <= 8 routing of c4b72f197: decode step unchanged (62/65/70 ms on three prompts), prefill at 150k 113.40 -> 112.71 s.
- Qwen3.8-27B NVFP4 with the Turing split-K rule (f5486ee76): decode +1.4-1.8 % tok/s (e.g. 74.2-74.5 -> 75.6-75.7 prose), texts byte-identical in all boots.
- DeepSeek-V4-Flash with the MoE grid bound (fork build 79e07bfef, whose only change over the fork build e1cb50fd9 is the grid-bound commit; its arms compared with the e1cb50fd9 arms of the previous series on the same entry): prefill at 33k / 178k / 287k tokens 4.8 / 3.8 / 3.5 % faster; greedy output and the eight quality answers byte-identical to the e1cb50fd9 boots.

Unit tests on this branch (built in place):
- Quadro RTX 8000: test_skinny_sm70_moe.py 103 passed, 33 skipped; test_sm70_qpn8_block_fp8.py 113 passed, 10 skipped; test_turing_qpn_kernel_selection.py 33 passed.
- Tesla V100: test_skinny_sm70_moe.py 89 passed, 47 skipped; test_turing_qpn_kernel_selection.py 33 passed; test_sm70_qpn8_block_fp8.py 95 passed, 27 skipped, 1 failed: test_volta_medium_prefill_keeps_original_turbomind_output, which compares two TurboMind calls bit for bit. It is intermittent with the default VLLM_SM70_FP8_TUNE_SMALL_SHAPES=1 (two separate runs of the test alone: 2 failed, then 1 failed of its 2 cases) because the runtime tuning can pick different tiles between the two calls; with VLLM_SM70_FP8_TUNE_SMALL_SHAPES=0 it passed three runs out of three. This PR does not change the TurboMind path the test compares; #1187 proposes that default to be off.

PRECOMMIT_PLACEHOLDER

🤖 Generated with Claude Code (https://claude.com/claude-code)

https://claude.ai/code/session_011k7FsBAcZ8Cw91wisiZxMf
