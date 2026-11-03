## Purpose

Serve NVIDIA's GLM-5.3-Flash-FP4-Local checkpoint (ModelOpt MIXED_PRECISION: NVFP4 routed experts with csf_32x2_v1 compressed block scales, MXFP8 KDA and dense projections, block-FP8 MLA projections) on SM70 and SM75, without converting the checkpoint. Unpacked to E4M3 the expert scales add 8.1 GiB and the model no longer fits five 32/48 GiB cards, so the scales stay compressed on the GPU, and the 8-bit projections stay 8-bit in QPN8 instead of being widened to fp16 at load. Our deployment: PP=5 over three Tesla V100 and two Quadro RTX 8000 with DFlash2, 524,288 tokens of context (docs/design/sm70_glm53_pp5_dflash2_mixed.md, added here).

30 commits above #TRITON_PR (which stacks on #DSV4_PR, #TP_PR and #TURING_PR): the first eight are #LOADER_PR and #MLA_PR (index shards in subfolders, direct-I/O dtypes, full-precision LM head with ModelOpt; MLA workspace and profile reservations), which the boot needs; the other 22 are new here. Groups of the new commits:

1. Loading and correctness (2530d2e3f, 7303dc246, 0cd2e83ea, a8758f247, 8d9b55c90): csf_32x2_v1 decoder (byte-equal to the reference encoder) and an expander that fills a shared QPN scale raster with the routed experts before each grouped skinny call (fixed launch shape, CUDA-graph safe; experts opt in through the CsfScaleConsumer protocol); ModelOpt MIXED_PRECISION detection of weight_scale_encoding and a CSF loader for the expert scales; block scales held next to their weights across AutoWeightsLoader calls (before, a scale that preceded its FP8 weight in the shards was silently dropped, the MLA projections stayed zero and the output was garbled); csf32 parts routed by their checkpoint name; the indexer query built in fp16 on Turing like the indexer op; Turing admitted on the FP16 sparse-MLA, KPool, indexer, mHC and KDA paths, each gate reading the capability of the worker's own device. Three GLM/ModelOpt test files that failed on main as well follow the current code.
2. Sparse MLA prefill with FP8 KV on cuBLAS blocks (8a21179f0): the Triton kernel (tl.dot as FMA before SM80, 78-87 % of the prefill compute) is replaced by the decode's gather/dequantize + cuBLAS GEMM route in blocks of 16 tokens.
3. 8-bit projections in QPN8 (acb0fe649, 23515aaae, f78b43972, 9a4eafd7b): a third QPN8 scale mode [K/32, N] for MXFP8 (prepare, dequantization bit-identical to load-time unpacking, GEMV for M <= 8; E8M0 exponent range widened to 95..134, exact in fp16); MXFP8 KDA q/k/v and dense MLPs kept as MXFP8 linears (the KDA layer keeps its quant config and splits in_proj into a quantized qkv and a BF16 b/f_a/g_a part); a Turing m16n8k8 path for the row-group scales at M 9..512; MLA q_b_proj, kv_b_proj and o_proj kept as block FP8 when the checkpoint names them.
4. CSF expansion speed (092923862, 9c7c2a21b, 1c8a6d502, 526b82adb, b7a781d85): grid bounded by the expert count (same bound as the skinny MoE grid), more programs for decode, a CUDA kernel (plain CUDA, four raster positions per thread, exception records written in the same pass) called through a _custom_ops wrapper.
5. Five-bit scale planes (1d64434cc, 0c3296413, 5af9b1485, 328e6f230): at load the csf scales are re-encoded as base + 4-bit nibble + 1 high bit per scale in the raster's positions (one block in six has a csf exception record, which a kernel cannot look up in its loop, but the scales of a row span at most 31 in all but the last layers). Decode and prefill read the planes in place, the shared raster is freed; projections whose rows span more than five bits (layers 36 and 40-44) keep their csf planes.
6. Memory and deployment (ac75fad2e, 127f9d31c, fa31143cc): skinny MoE routing groups built by one Triton program up to 128 slots (19 small torch kernels before; the torch version stays the reference); the embedding table can live in pinned host memory through the existing UVA offloader (--cpu-offload-gb 1.4 --cpu-offload-params embed_tokens), which frees 1.18 GiB on the first stage and on the DFlash replica of the last stage for the KV pool; the PP5 + DFlash2 layout document.

Not a duplicate: I searched the open PRs for GLM, glm5, GLM-5.3, csf, ModelOpt, MXFP8, KDA and "sparse MLA", and compared the changed files with those of every open PR. SEARCH_PLACEHOLDER

AI assistance was used for this change. I reviewed every line and ran the tests and measurements below.

## Test Plan

Native build of this branch (in-place).
python -m pytest TESTFILES_PLACEHOLDER, one process per file, on a Quadro RTX 8000 and on a Tesla V100.
pre-commit run --from-ref origin/main --to-ref HEAD; pre-commit run mypy-3.10 --hook-stage manual --from-ref origin/main --to-ref HEAD

## Path coverage

| Path / provider | Format | Key conditions (shape, TP/EP, graph, capability) | Expected selection / fallback | Evidence (source / static / executed / measured) | Unverified scope |
| --- | --- | --- | --- | --- | --- |
| ModelOpt MIXED_PRECISION loader | NVFP4 experts with csf_32x2_v1 scales, MXFP8, block FP8 | weight_scale_encoding present | CSF loader for expert scales (TP1 per expert, PP free) | executed: test_csf32_scales.py, test_modelopt.py, test_mxfp8_weight_pairs.py; measured: full boots | expert TP > 1 (rejected with an error) |
| skinny SM70 MoE with compressed scales | NVFP4 + csf / five-bit planes | SM70 and SM75 | five-bit planes in place; projections spanning more than 31 keep csf + raster expansion | executed: test_skinny_sm70_moe.py, test_s5_scales.py, class test csf vs planes (bit-identical); measured: model A/B | other checkpoints with csf scales |
| QPN8 MXFP8 linear | MXFP8 [K/32, N] scales | M <= 8 GEMV (both cards); M 9..512 m16n8k8 on SM75; dequant + cuBLAS above | as listed | executed: test_sm70_qpn8_mxfp8_linear.py on V100 and RTX 8000 | none |
| GLM sparse MLA prefill | FP8 KV | SM70 family | gather/dequantize + cuBLAS in 16-token blocks | executed: test_glm53_sm70_fp8_kv_prefill_blocks_match_reference | none |
| embedding in host memory | BF16 table | opt-in through --cpu-offload-params embed_tokens | UVA offloader | executed: smoke test on both cards (bit-identical, also in a CUDA graph) | none |

## Acceleration and benchmark contract (required for performance changes)

- Default enabled or opt-in: enabled for this checkpoint format on SM70/SM75 (per device); the embedding offload is opt-in through the existing --cpu-offload-* options.
- Required CLI options and environment switches: deployment as in docs/design/sm70_glm53_pp5_dflash2_mixed.md (PP=5, DFlash2 with 7 speculative tokens, --max-model-len 524288, --cpu-offload-gb 1.4 --cpu-offload-params embed_tokens, 512-token chunks, VLLM_USE_V2_MODEL_RUNNER=1).
- KV cache dtype used for the benchmark: fp8_e4m3.
- Wheel SHA or source commit: fork builds (github.com/Peuqui/1Cat-vLLM), each A/B against the build before the measured commits. The commits are rebased onto #TRITON_PR and main 6a143197e; the branch itself was built and unit-tested, not booted with the full model.
- PYTHONPATH, source overlays, or external native libraries used: none for the GLM A/B series below (stock Triton 3.6).
- User-entry route-hit, speed, and output-quality evidence: see Test Result.
- Existing kernel/backend registry and capability-rejection reasons: the checkpoint format selects the CSF loader; experts without CsfScaleConsumer are rejected with an error naming the requirement.
- Per-engine configuration and deprecated-variable compatibility: no setting added.
- New/changed environment metadata and generated reference check: REFERENCE_PLACEHOLDER
- Route snapshot command and intentional changed rows: none.
- SM70 registered variables / unregistered reads / model-parameter restrictions, before -> after: no change.

## Test Result

Model A/B/A/B series on GLM-5.3-Flash-FP4-Local, PP=5 over three Tesla V100 and two Quadro RTX 8000, prompt of 8,687 tokens, 192 greedy decode tokens unless noted (tok/s higher is better, s lower is better):
- MXFP8 projections in QPN8 (group 3, first part), without draft model: decode 17.38 -> 19.96 tok/s (+14.8 %), prefill 4.23 -> 4.34 s, KV pool 41,506 -> 119,057 tokens.
- MLA block FP8 (9a4eafd7b), without draft model: decode 19.96 -> 23.58 tok/s, prefill 4.34 s unchanged, KV pool 119,057 -> 139,810 tokens.
- DFlash2 deployment against the same build without draft model: GSM8K-60 59/60 in both, 53.35 against 23.26 tok/s (x2.29); prefill +6.6 %; KV pool 667,683 against 1,167,414 tokens (the draft model and its state live on the last stage).
- CUDA CSF kernel, DFlash2: decode +4.1 %, prefill -2.8 %; texts, answers and acceptance identical.
- Five-bit planes, DFlash2: prefill 4.49 -> 4.30 s (-4.2 %), decode 33.48 -> 35.36 tok/s (+5.6 %), after 14k tokens 45.30 -> 48.35 tok/s (+6.7 %); eight quality texts byte-identical, GSM8K-60 59/60, acceptance length 5.5109; KV pool 667,683 -> 645,277 tokens (-3.4 %).
Long context in the deployment: a 516,196-token prompt prefills at about 1,890 tok/s.

Kernel microbenchmarks (CUDA graphs, lower is better):
- CSF expansion, layer 13, w13 + w2, ms per expansion, Triton -> CUDA with exceptions in the same pass: V100 1 / 8 / 512 tokens 0.065 / 0.417 / 1.759 -> 0.047 / 0.247 / 1.074.
- Five-bit planes, layer 10, decode of 8 tokens, ms per layer W13 + W2: RTX 8000 1.512 (csf expansion + raster) -> 1.203, V100 1.166 -> 1.034; bit-identical.
- MXFP8 row-group GEMM on the RTX 8000, o_proj M 512: 0.561 ms against 0.627 (unpack + cuBLAS).

TESTS_PLACEHOLDER

🤖 Generated with Claude Code (https://claude.com/claude-code)

https://claude.ai/code/session_011k7FsBAcZ8Cw91wisiZxMf
