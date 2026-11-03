## Purpose

Long DeepSeek-V4-Flash prefills on SM70/SM75 pipeline stages spent most of their non-MoE time in memory-bound glue around the indexer and the sparse MLA attention. The per-stage nsys profile of a 150k-token prefill (PP=5 over three Tesla V100 and two Quadro RTX 8000) showed the indexer's relu/head-sum epilogue at 14.0 and 17.6 s of the two RTX stages, the pipeline's slowest, and six elementwise torch ops taking 820 of 1434 us of each sparse attention pass. Six commits, stacked on #TP_PR (shared build and binding lines, the m16n8k8 fragment header of #TURING_PR) and on #1222 (cuBLAS decode indexer above 131,072 tokens, same indexer file); only the last six commits are new here:

1. Vectorized indexer prefill epilogue (a08dbcee9): BLOCK_K 512 over 2 warps reads eight fp16 keys (16 bytes) per thread instead of two, which reaches the cards' plain read rate of the score tile. The head sum stays fp32; its reduction order changes in the last bits (up to 1.5e-5 on unit-scale scores), so the top-k can differ only at near ties; prompts up to index_topk * 4 tokens select every key and are unaffected.
2. Fused sparse MLA softmax (58bf56833, tests 36e84abe8): the gather writes zeros for unused slots and never reads their indices, and one Triton kernel reads the fp32 scores once and writes fp16 probabilities (scale, mask, sink column, fp32 softmax with libdevice exp, the same expf torch's softmax uses). The fp32 logits and probabilities buffers go away (140 -> 110 MiB per 128-token pass at 640 keys). 99.2-99.8 % of the outputs are bit-identical, the rest differ by at most 3e-5 (reduction order); the error against float64 is unchanged.
3. Fused lightning-indexer prefill scores on Turing tensor cores (1cb3d8eeb): indexer_prefill_scores_sm75 keeps one token's queries for 32 heads as m16n8k8 A fragments in registers, stages e4m3 key tiles widened to fp16 in shared memory, rounds scores to fp16 as the cuBLAS route stores them, and keeps relu, head weights and the head sum in registers; only the [tokens, keys] result reaches memory. Turing only (capability of the layer's own device); Volta keeps cuBLAS plus the epilogue of commit 1.
4. Indexer logits freed per prefill sub-chunk (5bb23f18c): the loop kept the previous sub-chunk's scores alive until the next scorer had allocated its own, so the peak under VLLM_SPARSE_INDEXER_MAX_LOGITS_MB was two sub-chunks while the profile run reserves one.
5. Compressed KV gather with 2048 two-warp workers instead of 128 per request (f17e11c49), same kernel and arithmetic, bit-identical output.

Not a duplicate: I searched the open PRs for indexer, "sparse MLA", softmax, epilogue, DeepSeek-V4, DSv4 and "lightning indexer", and compared the changed files with those of every open PR. SEARCH_PLACEHOLDER

AI assistance was used for this change. I reviewed every line and ran the tests and measurements below.

## Test Plan

Native build of this branch (in-place).
python -m pytest TESTFILES_PLACEHOLDER, one process per file, on a Quadro RTX 8000 and on a Tesla V100.
pre-commit run --from-ref origin/main --to-ref HEAD; pre-commit run mypy-3.10 --hook-stage manual --from-ref origin/main --to-ref HEAD

## Path coverage

| Path / provider | Format | Key conditions (shape, TP/EP, graph, capability) | Expected selection / fallback | Evidence (source / static / executed / measured) | Unverified scope |
| --- | --- | --- | --- | --- | --- |
| SM70-family indexer prefill epilogue | fp16 scores | cuBLAS prefill scoring | BLOCK_K 512, 2 warps | executed: test_deepseek_v4_sm70_indexer.py on RTX 8000 and V100; measured: CUDA graphs, model A/B | none |
| sparse MLA bmm attention, prefill and decode | FP8 KV (DeepSeek-V4) | SM70 family | gather with zeroed slots + fused Triton softmax | executed: test_dsv4_sparse_prefill_bmm.py, test_dsv4_sm70_sparse_bmm.py (decode reference Volta-only), test_deepseek_v4_sm70_routes.py; measured: CUDA graphs, model A/B | none |
| indexer prefill scores on SM75 | e4m3 keys, 32 or 64 heads | Turing | indexer_prefill_scores_sm75; Volta keeps cuBLAS + epilogue | executed: TESTS_SM75_PLACEHOLDER; measured: microbenchmark (max error against fp64 equal to the cuBLAS route) | SM8x+ (not selected) |
| indexer logits per sub-chunk | any | VLLM_SPARSE_INDEXER_MAX_LOGITS_MB below one chunk's scores | previous sub-chunk freed before the next is scored | executed: mini checkpoint runs with 1, 2 and 5 sub-chunks per step | none |
| compressed KV gather | FP8 KV (DeepSeek-V4) | prefill | 2048 workers, bit-identical | executed: test_deepseek_v4_sm70_qnorm_rope_kv_insert.py on RTX 8000 and V100; bench compares every grid bitwise | none |

## Acceleration and benchmark contract (required for performance changes)

- Default enabled or opt-in: default enabled on the matching capability (per device); no switch.
- Required CLI options and environment switches: none for the change. Measured entry: DeepSeek-V4-Flash, PP=5 over three Tesla V100 and two Quadro RTX 8000, DSpark with 5 speculative tokens, VLLM_USE_V2_MODEL_RUNNER=1.
- KV cache dtype used for the benchmark: fp8.
- Wheel SHA or source commit: fork builds (github.com/Peuqui/1Cat-vLLM); old arm = fork production at that time, new arm = the same plus the measured commits. The commits are rebased onto #TP_PR and main 6a143197e; the branch itself was built and unit-tested, not booted with a full model.
- PYTHONPATH, source overlays, or external native libraries used: none in the two A/B series below (stock Triton 3.6).
- User-entry route-hit, speed, and output-quality evidence: see Test Result.
- Existing kernel/backend registry and capability-rejection reasons: selection stays in the DeepSeek-V4 SM70 indexer and sparse attention routes, keyed on the capability of the layer's own device.
- Per-engine configuration and deprecated-variable compatibility: no setting added or changed.
- New/changed environment metadata and generated reference check: none.
- Route snapshot command and intentional changed rows: none.
- SM70 registered variables / unregistered reads / model-parameter restrictions, before -> after: no change.

## Test Result

Model A/B/A/B on DeepSeek-V4-Flash PP=5, one boot per arm, the first boot after a compile change not counted. Prefill time in s, lower is better (old a / old b -> new a / new b):
- Epilogue alone (commit 1): 32.7k tokens 14.66 / 14.66 -> 14.16 / 14.15 (-3.4 %); 178k 112.79 / 112.76 -> 93.94 / 93.96 (-16.7 %); 287k 222.35 / 222.50 -> 173.07 / 173.14 (-22.2 %). Decode step unchanged (87-88 ms at 178k, 93-94 ms at 287k). Greedy output and the eight quality answers byte-identical in all five boots.
- Epilogue and fused softmax (commits 1-2): 32.7k 14.67 / 14.66 -> 12.94 / 12.94 (-11.8 %); 178k 112.71 / 112.70 -> 82.20 / 82.14 (-27.1 %); 287k 222.26 / 222.19 -> 149.73 / 149.72 (-32.6 %). Decode step 1-4 ms shorter (178k 86.9 / 86.8 -> 84.9 / 84.5 ms, 287k 93.6 / 93.8 -> 90.4 / 90.8 ms). Texts byte-identical within each arm and different between the arms (reduction order); all read by hand and equivalent.
Commits 3 and 5 were measured in the model only together with the tensor-core Triton routes of a later PR of this series; their kernel numbers:

Kernel microbenchmarks (CUDA graphs, lower is better):
- Indexer epilogue, [508 tokens x 64 heads, 1024 keys] fp16 tile (66.6 MB): RTX 8000 632 -> 118.6 us (105 -> 562 GB/s), V100 362 -> 85.1 us (184 -> 782 GB/s).
- Sparse attention pass, us per call: prefill T128 H64 W640 RTX 8000 1438.3 -> 755.7, V100 1178.7 -> 632.0; W2176 RTX 8000 4666.8 -> 2367.7, V100 3760.3 -> 1983.5; decode T6 H64 (128 + 512 keys) V100 146.5 -> 96.5, SWA only (128 keys) 55.1 -> 36.6.
- Turing indexer scores, 512 tokens x 64 heads, ms per call: 1024 keys 0.326 -> 0.151, 4096 keys 1.270 -> 0.586, 18750 keys 5.551 -> 2.305, 37500 keys 11.018 -> 4.518 (57-70 TFLOPS); max error against fp64 equal to the cuBLAS route in all shapes.
- Compressed KV gather, us per call (rows 2048 / 18750 / 37500): RTX 8000 47.2 -> 7.2 / 532.3 -> 107.3 / 1061.9 -> 215.6; V100 45.8 -> 7.7 / 616.4 -> 87.4 / 1229.3 -> 167.7.
- Logits per sub-chunk (DeepSeek-V4 8-layer mini checkpoint, PP=2, 280k-token prompt, V100 stage, allocated peak MiB): budget 160 before 16759 (160 + 113), after 16646; prefill time with budget 512 against 64 unchanged (280k 32.849 / 32.758 vs 32.888 / 32.871 s).

TESTS_PLACEHOLDER

🤖 Generated with Claude Code (https://claude.com/claude-code)

https://claude.ai/code/session_011k7FsBAcZ8Cw91wisiZxMf
