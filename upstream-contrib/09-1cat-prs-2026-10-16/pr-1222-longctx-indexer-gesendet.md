## Purpose

DeepSeek-V4 decode on SM70/SM75 slows down sharply once the context passes 131,072 tokens. On our PP5 rig (three Tesla V100, two Quadro RTX 8000) the decode step went from 80 to 154 ms at the moment a request crossed 131,072 tokens and grew to 250 ms at 283k. V1 and MRV2 behave the same (178,479 tokens: 180.7-180.9 ms on V1, 174.8-175.1 ms on MRV2).

An nsys capture of the same decode at 124,029 and at 140,029 tokens shows the cause. Below the threshold the indexer scores with the cuBLAS GEMM of its decode route (volta_s884gemm, about 0.11 ms per call on a V100). Above it, `_paged_index_logits_relu_kernel` runs instead: 1.27 ms per call on a V100 and 6.9 ms on a Quadro RTX 8000, about 73 ms per decode step in total. Two gates keep the cuBLAS route off there:

1. The automatic compressed-index buckets end at 64 times the short bound, 131,072 tokens for DeepSeek-V4. Longer contexts replay the full-length graph, where `_decode_cublas_blocker` keeps the indexer on the paged kernel by design.
2. The cuBLAS route needs the gathered keys and one [rows * heads, keys] FP32 score tile within the indexer workspace budget (`indexer_prefill_tile_mb`). For six verifier rows of 64 heads the default 192 MiB admits about 112,000 compressed keys; the 64 MiB a PP5 deployment uses to keep its V100 prefill stages clear of OOM admits 32,768 keys (131,072 tokens). A diagnostic boot confirmed this gate for the 262,144 and 307,200 buckets. Raising the budget is no way out: the same field bounds the prefill score tile, and 512 MiB ran a V100 stage out of memory in the prefill.

This PR closes both gates, in two commits:

1. Score the cuBLAS decode indexer in key tiles. The gathered keys stay whole and the FP32 scores are computed in key tiles that fit next to them, so the route only needs the keys plus one tile within the budget. A key bound whose scores fit keeps its single tile, so the route and its result are unchanged wherever it was admitted before.
2. End the automatic compressed-index buckets at the model length: a last bucket at max_model_len keeps contexts above the largest multiple on the cuBLAS route. It applies wherever the SM70 graph plan is read, the V1 dispatcher and the MRV2 graph manager (#1221).

With the default budget, the second commit alone restores the route up to a 449k-token bound; the first is needed for smaller budgets and for longer models (a 1M-token bound means 262,144 keys).

A third commit repairs three tests in tests/v1/cudagraph/test_cudagraph_dispatch.py, the file the second commit extends, which fail on main: their MagicMock VllmConfig had no observability config, so the decode event tracer of CUDAGraphWrapper compared the elapsed time against a MagicMock threshold ("'<' not supported between instances of 'float' and 'MagicMock'"). The mock now carries a default ObservabilityConfig, as other config tests do.

Not a duplicate: gh pr list --repo 1CatAI/1Cat-vLLM --state all --search "indexer decode", "cuBLAS indexer", "workspace budget", "graph_contract", "DeepSeek-V4 long context" and "131072" find no PR that changes this route or the bucket list; #874 introduced the buckets, #714 (closed) proposed the cuBLAS decode route. Open PRs touching the same files (#1190, #1192) change other parts (codec ids in the indexer kernels, tests) and merge without conflicts.

AI assistance was used for this change. I reviewed every line and ran the tests and measurements below.

## Test Plan

```bash
pytest tests/kernels/test_deepseek_v4_sm70_indexer.py tests/v1/cudagraph/test_cudagraph_dispatch.py tests/models/test_deepseek_v4_sm70_routes.py tests/v1/worker/test_sm70_long_attention_graphs.py
pre-commit run --from-ref origin/main --to-ref HEAD
pre-commit run mypy-3.10 --hook-stage manual --from-ref origin/main --to-ref HEAD
```

New tests: `test_decode_cublas_score_tiles_match_one_tile` (a 3 MiB budget splits 6,144 keys into six tiles; logits within FP32 rounding and identical top-512 keys per row against one tile, also captured and replayed as a CUDA graph; a budget without room for one tile falls back to the paged kernel) and `test_dsv4_decode_context_buckets_end_at_the_model_length`. Two dispatcher tests now expect the model-length bucket.

Serving on our fork (main 6f3557c97 plus our commits, MRV2 with the DSpark speculator of #1221): DeepSeek-V4-Flash MXFP4/FP8 with DSpark (num_speculative_tokens 5), PP5 over three Tesla V100 and two Quadro RTX 8000, FP8 KV cache, max_num_seqs 1, `VLLM_SM70_INDEXER_PREFILL_TILE_MB=64`. Arms: fix (this PR on the fork, PYTHONPATH on the fork tree, own compile cache) against the same entry without it; boots fix cold, fix, base, fix, plus two earlier base boots with the same probe. Per boot: one warm-up request (32,668 tokens), then 178,479 and 287,077 tokens from repository documents and sources, greedy, 400 output tokens each. Step time = decode time / engine decode steps (vllm:spec_decode_num_drafts_total), independent of the acceptance length.

## Path coverage

| Path / provider | Format | Key conditions (shape, TP/EP, graph, capability) | Expected selection / fallback | Evidence (source / static / executed / measured) | Unverified scope |
| --- | --- | --- | --- | --- | --- |
| SM70 indexer decode, cuBLAS route | FP8 index cache (128 values + FP32 scale) | 6 rows x 64 heads, 64 MiB budget, bucket 307,200 (76,800 keys, three tiles of up to 30,720 keys), FULL graph | cuBLAS route in key tiles; paged kernel when not even one tile fits | executed + measured (serving), unit tests on V100 and RTX 8000 | other row and head counts beyond the unit tests |
| Automatic compressed-index buckets | DeepSeek-V4 graph plan | max_model_len 307,200, SM70 family | buckets 2048 ... 131,072 plus 307,200; override field unchanged | executed + measured on MRV2; dispatcher unit tests | V1 serving with the new bucket (same plan, not measured) |

## Acceleration and benchmark contract (required for performance changes)

- Default enabled or opt-in: default; both commits follow the existing graph plan and budget.
- Required CLI options and environment switches: none beyond the DeepSeek-V4 entry; the measurement used VLLM_USE_V2_MODEL_RUNNER=1 and VLLM_SM70_INDEXER_PREFILL_TILE_MB=64.
- KV cache dtype used for the benchmark: fp8 (fp8_ds_mla).
- Wheel SHA or source commit: fork commit df0941355 (this PR on main 6f3557c97 plus our commits); base arm 62b127eab.
- PYTHONPATH, source overlays, or external native libraries used: fix arm PYTHONPATH on the fork tree with its own compile cache; native build of the fork tree in both arms (this PR changes Python only).
- User-entry route-hit, speed, and output-quality evidence: see Test Result.
- Existing kernel/backend registry and capability-rejection reasons: unchanged; the budget requirement now reads "the shared key gather and one FP32 score tile within the indexer workspace budget".
- Per-engine configuration and deprecated-variable compatibility: no new fields; indexer_prefill_tile_mb and compilation_config.runtime.dsv4_context_buckets keep their meaning.
- New/changed environment metadata and generated reference check: none.
- Route snapshot command and intentional changed rows: none.
- SM70 registered variables / unregistered reads / model-parameter restrictions, before -> after: unchanged.

## Test Result

- On a Tesla V100 and on a Quadro RTX 8000 (main 6f3557c97 plus this PR): 151 passed. Main without this PR: 144 passed, 3 failed (tests/v1/cudagraph/test_cudagraph_dispatch.py: TestCUDAGraphWrapper::test_capture_and_replay, test_capture_replay_bypass_logic, test_nested_wrappers, repaired by the third commit). pre-commit and mypy-3.10 pass.
- Decode step in ms (fix: cold / a / b; base: three boots): 32,668 tokens 67.9 / 68.2 / 68.1 against 67.6 / 68.0 / 67.6; 178,479 tokens 87.3 / 86.7 / 87.2 against 175.1 / 174.8 / 175.4; 287,077 tokens 93.1 / 93.1 / 93.5 against 242.4 / 241.4 / 241.9. Output speed at 178,479 tokens 38.4-43.1 against 19.1-20.1 tok/s, at 287,077 tokens 33.2-34.3 against 12.6-13.1 tok/s.
- Answers at all three lengths read by hand in both arms: on topic and equivalent. Greedy text at these lengths is not word-identical across boots in either arm (base against base diverges after 0 to 334 characters), so the arms were compared by content.
- Short context unchanged: in the production control after the switch, the three greedy reference prompts are identical to the reference, prose / code / edit 48.3-48.4 / 73.5-73.6 / 80.7-81.0 tok/s, and the eight quality prompts are correct.
- The DeepSeek-V4 coding entry (num_speculative_tokens 7) with the fix: 97.1 ms per step at 178,479 tokens and 103.6 ms at 287,077 tokens.
