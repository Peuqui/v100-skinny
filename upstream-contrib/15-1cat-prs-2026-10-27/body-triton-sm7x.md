## Purpose

Stock Triton lowers every tl.dot to FMA before SM80, so the SM70-family routes run cuBLAS or hand-written kernels on Volta and Turing. We built a Triton 3.6 with the sm_70 (mma.m8n8k4, through linear layouts) and sm_75 (mma.m16n8k8) lowerings: github.com/Peuqui/triton, branch sm7x-tensor-cores (23 commits on v3.6.0), proposed upstream in triton-lang/triton#TRITON_ISSUE. With it, Triton kernels on tensor cores beat several of the DeepSeek-V4 prefill routes on both card types. This PR adds one opt-in setting for such a Triton and the routes that use it. With the setting off (the default) nothing changes; with the setting on and a stock Triton the engine stops at startup and says why, instead of running FMA kernels.

Ten commits, stacked on #DSV4_PR (which stacks on #TP_PR and #TURING_PR); only the last ten commits are new here:

1. The setting (d354f36e3, ef5816176): kernel_config.triton_sm7x_tensor_cores (default false, part of the compilation hash). KernelConfig.use_triton_sm7x_tensor_cores(device_index) is the one place routes ask: it answers for the device's own architecture and, on first use per architecture, compiles a 16x16 fp16 tl.dot without launching it and requires mma.sync in the PTX (vllm/triton_utils/mma_probe.py, generic).
2. DeepSeek-V4 indexer prefill scores in one Triton kernel for Volta and Turing (26062dddc): heads of BLOCK_T tokens stacked into one MMA operand reused by every key tile, e4m3 keys widened bit-exactly to fp16, products rounded to fp16 as the cuBLAS route stores them, relu/head weights/head sum in registers. Tiles chosen per card from a 216-tile sweep. On Turing it is faster up to 19.2k keys and 2-7 % slower beyond (the fused native scorer of #DSV4_PR stays the reference there); Volta gains 28-39 %.
3. Skinny MoE top-k reduction in one pass (d6457136f): the batched _mtp_weighted_reduce kernel the FP4 stages already use (bit-identical to moe_unpermute) replaces a five-kernel Torch chain. This commit does not depend on the setting.
4. mHC pre-stage projection on tensor cores for prefill (a229958f2): the fp32 weight split into fp16 hi + lo * 2^-11 makes the products exact; every 16-wide K block starts from zero and block sums add round-to-nearest in two levels, 32 K splits summed in fp64. Error against fp64 equal or lower than today's route. Decode (<= 16 tokens) keeps its kernels.
5. DeepSeek-V4 sparse MLA prefill matmuls that read the selected rows through the indices (4474655a6), instead of gathering a [T, W, 512] workspace for two cuBLAS BMMs; workspace per pass 110 -> 30 MiB at width 640. Bit-identical on V100; on the RTX 8000 0.97-1.25 % of the fp16 outputs differ by one step, error against fp64 equal.
6. Volta skinny MoE on a Triton grouped GEMM for long chunks (4e83ea6a9, ecd2c0953, 435b8a563): with more than 16 rows per expert (1024-token chunks give about 24 at top-6 of 256) a grouped GEMM that decodes each FP4 tile once per 32- or 64-row block beats the hand kernel, which decodes it per 8-row block. Programs map to row blocks of one group in three row tiles (16/32/64) and find their block inside the kernel. 512-token chunks stay on the hand kernel.
7. One sparse prefill pass per chunk on Turing (a65db609a): passes and tiles per architecture; Turing (48 GiB cards) attends a whole chunk in one pass, Volta keeps 128-token passes for the memory of its 32 GiB stages.

Not a duplicate: I searched the open PRs for Triton, "tensor core", mma, Volta, Turing, mHC, indexer and "skinny MoE", and compared the changed files with those of every open PR. SEARCH_PLACEHOLDER

AI assistance was used for this change. I reviewed every line and ran the tests and measurements below.

## Test Plan

Native build of this branch (in-place).
python -m pytest TESTFILES_PLACEHOLDER, one process per file, on a Quadro RTX 8000 and on a Tesla V100, each with stock Triton 3.6 and with the sm7x Triton build.
pre-commit run --from-ref origin/main --to-ref HEAD; pre-commit run mypy-3.10 --hook-stage manual --from-ref origin/main --to-ref HEAD

## Path coverage

| Path / provider | Format | Key conditions (shape, TP/EP, graph, capability) | Expected selection / fallback | Evidence (source / static / executed / measured) | Unverified scope |
| --- | --- | --- | --- | --- | --- |
| setting and probe | n/a | triton_sm7x_tensor_cores true/false; stock vs sm7x Triton | false: no change; true + MMA in PTX: routes below; true + FMA: startup error | executed: test_triton_sm7x_tensor_cores / test_kernel_config_triton_sm7x with both Tritons (no GPU needed) | other Triton versions |
| DSv4 indexer prefill scores | e4m3 keys, 32/64 heads | setting on, SM70 or SM75 | Triton scorer; setting off: cuBLAS + epilogue (Volta), fused native scorer (Turing) | executed: test_deepseek_v4_sm70_indexer.py on both cards, both Tritons; measured: CUDA graphs | other head counts |
| skinny MoE top-k reduction | any skinny QPN | always | _mtp_weighted_reduce | executed: test_skinny_sm70_moe.py, test_sm70_modelopt_mixed_nvfp4.py on both cards | none |
| mHC pre-stage projection | fp32 weights, fp16 x | setting on, more than 16 tokens | tensor-core projection; decode and setting off unchanged | executed: test_mhc_prenorm_tensor_cores.py on both cards (sm7x Triton), staging/post tests with stock Triton | none |
| DSv4 sparse MLA prefill | FP8 KV | setting on | index-reading scores/PV GEMMs; off: gather + cuBLAS BMMs | executed: test_dsv4_sparse_prefill_bmm.py (both routes against fp64), test_deepseek_v4_sm70_routes.py | fewer than 16 heads padded, not measured in a model |
| Volta skinny MoE, long chunks | E8M0-scaled FP4 in M8N8K4 order | setting on, SM70, more than 16 rows per expert | Triton grouped GEMM; otherwise the hand kernel | executed: test_skinny_moe_triton.py (V100, sm7x Triton) against the hand kernel and fp32 | Turing (keeps its m16n8k8 hand kernel) |

## Acceleration and benchmark contract (required for performance changes)

- Default enabled or opt-in: opt-in (kernel_config.triton_sm7x_tensor_cores, default false), except commit 3, which is always on.
- Required CLI options and environment switches: --kernel-config '{"triton_sm7x_tensor_cores":true}' and the sm7x Triton build first on PYTHONPATH. Measured entry: DeepSeek-V4-Flash, PP=5 over three Tesla V100 and two Quadro RTX 8000, DSpark with 5 speculative tokens, VLLM_USE_V2_MODEL_RUNNER=1; new arm also --max-num-batched-tokens 1024 (old arm 512), --num-gpu-blocks-override 3500 and VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64.
- KV cache dtype used for the benchmark: fp8.
- Wheel SHA or source commit: fork builds (github.com/Peuqui/1Cat-vLLM): reference 79e07bfef, changed a1ac0c89d. The changed build also contains commits of #TURING_PR (Turing m16n8k8 MoE, Volta MoE tiles), #TP_PR (Turing prefill attention, not used by DeepSeek-V4) and #DSV4_PR (Turing indexer scorer, logits per sub-chunk, KV gather), and one commit not offered yet (PP send of an eager step overlapped with the next step). The commits here are rebased onto #DSV4_PR and main 6a143197e; the branch itself was built and unit-tested, not booted with a full model.
- PYTHONPATH, source overlays, or external native libraries used: the sm7x Triton build (github.com/Peuqui/triton, sm7x-tensor-cores) on PYTHONPATH in the changed arm.
- User-entry route-hit, speed, and output-quality evidence: see Test Result; a cold boot checked that the new Triton kernels in the compile cache contain mma.sync.
- Existing kernel/backend registry and capability-rejection reasons: every route asks KernelConfig.use_triton_sm7x_tensor_cores for its own device when the layer is built; the probe rejects a Triton that lowers tl.dot to FMA with an error naming the compiler path.
- Per-engine configuration and deprecated-variable compatibility: one new KernelConfig field; no environment variable.
- New/changed environment metadata and generated reference check: REFERENCE_PLACEHOLDER
- Route snapshot command and intentional changed rows: none.
- SM70 registered variables / unregistered reads / model-parameter restrictions, before -> after: no change.

## Test Result

Model A/B/A/B, DeepSeek-V4-Flash PP=5, one boot per arm, the first boot of the changed build not counted (fork builds above; the changed arm combines this PR with parts of the stacked PRs and 1024-token chunks, see the contract). Prefill time in s, lower is better (old a / old b -> new a / new b):
- 35.7k-token prompts (four per arm): 12.4-12.6 / 12.4-12.6 -> 7.0 / 7.0
- 178k tokens: 79.16 / 79.14 -> 41.08 / 41.06 (-48 %)
- 287k tokens: 144.74 / 144.64 -> 73.57 / 73.55 (-49 %)
Decode step in ms (lower is better): 178k 83.5 / 85.2 -> 81.8 / 82.5, 287k 90.0 / 90.3 -> 88.7 / 88.1; 35.7k prompts 68-70 -> 66-68 ms. Peak memory per card: RTX 8000 +340 MiB (47,625 MiB), V100 -20 to -60 MiB. The eight quality answers of both changed boots read by hand: all correct, byte-identical between the two changed boots.

Kernel microbenchmarks (CUDA graphs, lower is better):
- Indexer prefill scores, sum over 4.8k/19.2k/38.4k/57.6k/76.8k keys, ms, today's route -> Triton: V100 64 heads 46.94 -> 28.87, 32 heads 23.96 -> 17.18; RTX 8000 64 heads 23.97 -> 24.21, 32 heads 12.53 -> 13.12.
- Skinny MoE top-k reduction, us per call (tokens 1 / 64 / 508 / 2048): RTX 8000 13.3 / 67.6 / 469.7 / 1873.6 -> 1.4 / 6.8 / 54.2 / 217.8; V100 11.8 / 52.1 / 352.2 / 1388.8 -> 1.5 / 5.8 / 44.5 / 198.7.
- mHC projection, us per call (tokens 508 / 2048): RTX 8000 (TileLang) 428.6 / 877.4 -> 78.0 / 263.8; V100 (SM70 Triton) 467.5 / 1988.5 -> 64.1 / 201.8.
- Sparse prefill, whole 508-token chunk, width 640, random indices, ms: RTX 8000 3.064 -> 1.733, V100 2.452 -> 1.352.
- Volta skinny MoE, DeepSeek-V4-Flash layer 5, ms per layer W13 / W2: 1020 tokens 7.502 / 4.080 -> 5.967 / 3.466, 2040 tokens 13.780 / 7.464 -> 9.302 / 5.080; 512 tokens stay on the hand kernel (4.218 / 2.295 against 4.555 / 2.647 on Triton).
- One sparse pass per chunk on Turing, 1020 tokens, 64 heads, shared indices, ms: 640 slots 2.448 -> 2.194, 1152 slots 4.209 -> 3.859; bit-identical.

TESTS_PLACEHOLDER

🤖 Generated with Claude Code (https://claude.com/claude-code)

https://claude.ai/code/session_011k7FsBAcZ8Cw91wisiZxMf
