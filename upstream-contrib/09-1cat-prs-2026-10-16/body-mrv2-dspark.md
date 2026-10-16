## Purpose

DeepSeek-V4 with its DSpark drafter runs only on Model Runner V1 in this fork: MRV2 rejects the speculative method 'dspark'. This PR adds it, in three commits.

1. A DSpark speculator for MRV2, ported from vllm-project/vllm's MRV2 DSparkSpeculator (vllm#46995 by @benchislett, speculator as of 2026-09-30) onto this fork's V2 DFlash speculator. The anchor is the first query position (SAMPLE_FROM_ANCHOR), the sequential Markov sampling runs inside the drafter's FULL CUDA graph, the DeepSeek-V4 context slots are passed per cache layer, the staging buffers are sized to the drafter's hidden size instead of the hc_mult-wide DeepSeek-V4 MTP residual, and the drafter keeps its own embedding copy under pipeline parallelism, the same contract as the V1 proposer. MRV2 admits 'dspark', reads dspark_target_layer_ids and dspark_noise_token_id, and feeds the drafter the target hidden state unless the target returns an override (DeepSeek-V4 MTP only); the two inline call sites of that override now share one helper that also accepts a missing override. Credit: Co-authored-by Benjamin Chislett on the commit. Not ported: the speculators-format 1+N layout, the reduced draft vocabulary scatter, the top-k Markov projection, the confidence head and confidence-scheduled verification (rejected on MRV2 with a pointer to V1), acceptance estimation and watermarking.
2. Compressed-index decode graphs at context buckets for MRV2. MRV2 captured the DeepSeek-V4 decode graph at the maximum model length, where the SM70 indexer stays on its paged kernel; a 35.7k-token decode step took 91-92 ms. The graph manager now captures variants at the compressed-index buckets of the SM70 graph plan, the plan the V1 dispatcher already reads, builds the capture metadata at the bucket, and replays the smallest captured bucket that covers the live context. As in the V1 dispatcher, the variants exist for one request with one verification block.
3. Tests for both.

The layering baseline records the one SM70 graph-plan import of the MRV2 graph manager and the new speculator's references to the DFlash base it extends, as it does for vllm/v1/spec_decode/dspark.py.

Not a duplicate: gh pr list --repo 1CatAI/1Cat-vLLM --state all --search "dspark", "MRV2", "model runner v2", "context bucket" and "compressed index graph" find no PR that adds DSpark or compressed-index buckets to MRV2; #874 added the buckets to the V1 dispatcher. Open PRs that touch the same files (#1182, #1184, #1197, #1215, #1217) change other things and merge without conflicts.

AI assistance was used for this change. I reviewed every line and ran the tests and measurements below.

## Test Plan

```bash
pytest tests/v1/worker/test_mrv2_dspark_context_buckets.py tests/v1/worker/test_sm70_long_attention_graphs.py tests/v1/spec_decode/test_dspark.py tests/v1/spec_decode/test_dflash_mrv2_config.py tests/v1/worker/test_gpu_model_runner_v2.py tests/v1/worker/test_gpu_model_runner_v2_greedy.py tests/v1/worker/test_gpu_model_runner_v2_prefill_dispatch.py tests/v1/worker/test_gpu_cudagraph_memory_reserve.py
pre-commit run --from-ref origin/main --to-ref HEAD
pre-commit run mypy-3.10 --hook-stage manual --from-ref origin/main --to-ref HEAD
```

Serving on our fork (main 6f3557c97 plus our commits): DeepSeek-V4-Flash MXFP4/FP8 with DSpark (num_speculative_tokens 5), PP5 over three Tesla V100 and two Quadro RTX 8000, FP8 KV cache, max_num_seqs 1. Arms: the same entry on V1 and with VLLM_USE_V2_MODEL_RUNNER=1 (cudagraph_capture_sizes [5, 6]); boots V2 cold, V1, V2, V1, V2. Per boot: three greedy reference prompts, four 35.7k-token prompts, short prose, code and a 1.9k-token edit task at production sampling, eight quality prompts.

## Path coverage

| Path / provider | Format | Key conditions (shape, TP/EP, graph, capability) | Expected selection / fallback | Evidence (source / static / executed / measured) | Unverified scope |
| --- | --- | --- | --- | --- | --- |
| MRV2 DSparkSpeculator | DeepSeek-V4 DSpark drafter, FP8 KV | method dspark, PP5, FULL graphs [5, 6], greedy drafts, SM70 and SM75 stages | DSpark speculator; confidence-scheduled verification rejected with a pointer to V1 | executed + measured (serving), unit tests | probabilistic drafting, TP > 1, data parallel |
| MRV2 compressed-index decode graphs | DeepSeek-V4 sparse MLA | one request, verification block of 6, buckets 2048 to 131072 from the graph plan | smallest covering bucket graph; full-length graph above the largest bucket | executed + measured (35.7k), unit tests | batches of several requests (no variants, as in V1) |

## Acceleration and benchmark contract (required for performance changes)

- Default enabled or opt-in: MRV2 stays opt-in (VLLM_USE_V2_MODEL_RUNNER=1). On MRV2, DSpark follows the speculative config and the buckets follow the existing graph plan (auto-enabled on the SM70 family, overridable through compilation_config.runtime.dsv4_context_buckets).
- Required CLI options and environment switches: VLLM_USE_V2_MODEL_RUNNER=1, --speculative-config '{"method":"dspark","num_speculative_tokens":5}', --compilation-config '{"cudagraph_capture_sizes":[5,6],"max_cudagraph_capture_size":6}'.
- KV cache dtype used for the benchmark: fp8 (fp8_ds_mla).
- Wheel SHA or source commit: V2 arm fork commit 5c04f68c9, whose DSpark and bucket code equals this PR (it also carried a P2P check that was withdrawn afterwards and only acts with TP > 1); V1 arm fork commit baaa5a687. Both are main 6f3557c97 plus our commits.
- PYTHONPATH, source overlays, or external native libraries used: the V2 arm ran the production entry with PYTHONPATH on the fork tree and its own compile cache; native build of the fork tree in both arms.
- User-entry route-hit, speed, and output-quality evidence: see Test Result.
- Existing kernel/backend registry and capability-rejection reasons: unchanged; the indexer's existing cuBLAS admission decides per captured bucket.
- Per-engine configuration and deprecated-variable compatibility: no new fields or environment variables.
- New/changed environment metadata and generated reference check: none.
- Route snapshot command and intentional changed rows: none.
- SM70 registered variables / unregistered reads / model-parameter restrictions, before -> after: unchanged.

## Test Result

- On a Tesla V100 and on a Quadro RTX 8000 (main 6f3557c97 plus this PR): 130 passed, 14 failed. The same 14 fail identically on main 6f3557c97 without this PR (125 passed, 14 failed): tests/v1/spec_decode/test_dflash_mrv2_config.py (4), tests/v1/worker/test_gpu_model_runner_v2_greedy.py (9) and tests/v1/worker/test_gpu_model_runner_v2_prefill_dispatch.py::test_execute_rejects_prefill_before_dp_graph_dispatch. pre-commit and mypy-3.10 pass.
- Serving, V1 (2 boots) -> V2 (3 boots): prose 42.9-43.0 -> 48.2-48.3 tok/s (+12 %), code 65.7-65.8 -> 73.5-73.6 tok/s (+12 %), edit 72.6-72.8 -> 80.5-81.0 tok/s (+11 %); decode step 70 / 73 / 78 -> 63 / 65 / 70 ms; at 35.7k context 75-76 -> 69-72 ms per step; 35.7k prefill 15.3-15.6 -> 15.2-15.3 s. Acceptance per round: prose 3.02 and code 4.78 in both arms, edit 5.66-5.68 (V1) and 5.64-5.68 (V2).
- The three greedy reference prompts are identical in all five boots. The eight quality prompts give the same text in both arms and are correct (read by hand).
- One V2 boot without the context buckets: 91-92 ms per step at 35.7k context.
- Above 131,072 tokens of context the indexer leaves its cuBLAS route on V1 and V2 alike; that is a separate change, sent as its own PR.
