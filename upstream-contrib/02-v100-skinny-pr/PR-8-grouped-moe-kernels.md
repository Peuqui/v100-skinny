# PR an dnv2003/v100-skinny: gebündelte NVFP4-MoE-Kernel

ERÖFFNET 20.09.2026 als https://github.com/dnv2003/v100-skinny/pull/8 (Go Peuqui: „Machen wir bitte").
Worktree: vllm-research/v100-skinny-pr-moe, Branch grouped-moe-kernels auf origin/main 5b589c0, Commit a8e1596.
Inhalt: nur der MoE-Abschnitt aus kernels/skinny_kernels.cu (moe_simt + moe_qpn, 386 Zeilen, zwei pybind-Einträge)
und scripts/moe_qpn_test.py (eigenständig, synthetische Experten). ALL OK auf V100 und RTX 8000.
Die Frage nach dem Kachel-Regime steht im PR-Text; der getrennte Issue-Entwurf (issue-grouped-moe-decode-regime.md) ist damit
ÜBERHOLT und wird nicht gepostet. Bezug: dort offenes Issue #4 (Qwen3.6 35B NVFP4, ein MoE-Modell).

## PR-Text

Two kernels for NVFP4 mixture-of-experts layers, built from the pieces this repository already has. Nothing existing is touched: one new section in kernels/skinny_kernels.cu, two pybind entries, one test script.

## What they do

Today an NVFP4 MoE layer has to call `gemm_qpn` once per active expert, from Python, with a host sync to learn the routing. Both kernels here run all experts of a layer in one launch with device-side routing, so there is no host sync and the launch captures into CUDA graphs.

- `moe_qpn` is the tensor-core path. It keeps your QPN fragment order per expert (`_qpn_prepack` applied to each expert, so the prepacked experts are a byte permutation of the checkpoint) and your `MMA_8N8K4` inner loop. Routing arrives compact: the (token, top-k) slots sorted by expert, one grid row per group of equal experts, and padding groups exit before they touch a weight, so the launch does not scale with the expert count (with 512 experts and one token a per-expert grid was mostly empty blocks). The activation rows come through the slot indirection, token-major for w13 and slot-major for w2. An expert with more than 8 slots takes one pass per 8 rows, each re-reading its tile.
- `moe_simt` is the SIMT variant of the same routing on the checkpoint layout, for up to 8 tokens.

## Numbers

scripts/moe_qpn_test.py, synthetic experts (64 experts, top-6, w13 2048x2048, w2 2048x1024), ms per matrix, Tesla V100 / Quadro RTX 8000 (sm70 code on both):

| tokens | per-expert loop over gemm_qpn | moe_simt | moe_qpn |
|---|---|---|---|
| w13, 1 | 0.48 / 0.48 | 0.064 / 0.070 | 0.028 / 0.032 |
| w13, 6 | 2.02 / 2.04 | 0.198 / 0.238 | 0.113 / 0.138 |
| w13, 8 | 2.54 / 2.31 | 0.226 / 0.277 | 0.132 / 0.161 |
| w13, 128 | 4.81 / 4.36 | n/a | 0.539 / 0.715 |
| w2, 6 | 2.02 / 1.95 | 0.143 / 0.143 | 0.064 / 0.065 |
| w2, 128 | 4.34 / 4.35 | n/a | 0.345 / 0.369 |

The script checks moe_qpn against the loop (same decoder, max abs diff 1e-4 to 1e-3 at values around 1.5) and both kernels against an fp32 dequantised reference, including experts with more than 8 rows. It prints ALL OK on both cards.

In a real deployment (DeepSeek-V4-Flash, 256 experts, top-6, experts 4096x4096 and 4096x2048, pipeline over two RTX 8000 and three V100, 5 draft tokens) moe_qpn took a decode step from 144 to 113 ms when it replaced moe_simt in September, and using it for prefill chunks as well, instead of the per-expert loop, took a cold 22k-token prefill from 46 to 19 s. With the rest of the stack tuned since, a decode step is 81 ms and the MoE kernel is about half of it (0.75 of 1.4 to 1.5 ms per layer).

## A question, since you know this instruction set best

At 6 verifier tokens an expert gets one or two of the eight rows of the tile, so `m8n8k4` runs mostly on zero activations. The memory-bandwidth floor for the expert bytes of such a step (about 29 experts x 12 MB per layer) is roughly 0.39 ms on V100 and 0.52 ms on RTX 8000; the kernel is 1.5 to 2x above that. SIMT is not the answer: it loses to the tensor-core path even at one row per expert, by about 2x in the table above and by 1.35x on the real DeepSeek experts. Do you see a shape that fits 1 to 2 activation rows better, or a way to let rows of different experts share an instruction although their B operands differ? The quadpair shares one A tile across four N-slices, which is the opposite of what this regime needs.

## Scope

This is deliberately small, unlike my earlier #7. The vLLM side that drives these kernels (expert prepack at load time, the routing tensors, the MoE backend class) lives in our vLLM fork and is not part of this PR; if you want it as a fork_patch I can add it in a follow-up.

AI assistance (Claude) was used for the kernels, the test and this text. I ran the test on both card generations and the deployment numbers on my own hardware.
