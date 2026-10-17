## Purpose

The QSA indexer of Qwen3.8-Flash-Next has a cuBLAS scorer (FP16 `torch.mm` with FP32 accumulation and a Triton epilogue) and an exact lexicographic top-k (the native `_C_qsa_sm70` op). Both are gated on compute capability 7.0, so Turing stages score with the paged Triton kernel `_qsa_mqa_paged_kernel` and select with the persistent top-k. In an nsys capture of a 184k-token Flash-Next PP4 prefill (two Quadro RTX 8000 and two Tesla V100 stages), `_qsa_mqa_paged_kernel` took 20.0 s and 30.1 s on the two RTX 8000 stages and never ran on the V100 stages.

This PR gates the single-request cuBLAS route, the multi-request cuBLAS route and the exact top-k on the SM70 family, as the DeepSeek-V4 indexer does. Turing stages thereby also get the deterministic tie rule of the exact selector (#394), which V100 stages already use. The native top-k op is built for sm_70 and runs unchanged on sm_75. The exact-index test now also runs on Turing.

Not a duplicate: gh pr list --repo 1CatAI/1Cat-vLLM --state all --search "QSA indexer Turing", "lexicographic topk", "qsa_indexer_cublas", "is_device_capability_family QSA" and "Turing QSA" find no PR that changes these gates; #394 (merged) introduced the exact selector because `persistent_topk` can pick a different block set when scores tie at the cutoff, #1171 (merged) reduced its work for long contexts, #469 (merged) widened the QSA launch profile to pre-Ampere. The open PRs touching vllm/models/qwen4_exp/nvidia/ops/qsa.py (#523, #828, #903, #1023, #1167, #1176, #1192, #1194, #1204-#1206, #1210, #1211) do not change these three gates, and this PR adds no merge conflict to any open PR (git merge-tree against each PR head; the ones that conflict do so with main already).

AI assistance was used for this change. I reviewed every line and ran the tests and measurements below.

## Test Plan

```bash
pytest tests/config/test_sm70_provider_lifecycle.py tests/kernels/test_sm70_qsa_shared_key.py tests/kernels/test_top_k_per_row.py tests/models/qwen4_exp/test_qsa_batch_topk.py tests/models/qwen4_exp/test_qsa_indexer_request_split.py tests/models/qwen4_exp/test_qsa_ops.py tests/models/qwen4_exp/test_qsa_reference.py tests/models/qwen4_exp/test_qsa_resolved_dispatch.py
pre-commit run --from-ref origin/main --to-ref HEAD
pre-commit run mypy-3.10 --hook-stage manual --from-ref origin/main --to-ref HEAD
```

Kernel benchmark on a Quadro RTX 8000: `qsa_select_paged_tokens` for the last 1,616-row prefill chunk of a single request (4 indexer heads, head dim 128, compress ratio 4, top-k 2,048, contiguous cache pages), with the three gates as on main (paged Triton scorer, persistent top-k) against this PR (cuBLAS scorer, exact top-k), median of 20 calls, and the share of rows whose selected token set is identical.

Serving on our fork (main 6f3557c97 plus our commits): the same A/B as #PR_C (fork commit ee261cc74 = this change and #PR_C on top of fork commit 9a41a8e67), Qwen3.8-Flash-Next-180B-A4B NVFP4 with MTP in PP4 and TP2xPP2 over two Quadro RTX 8000 and two Tesla V100, boots fix cold, base, fix, base, fix; the cold boot and the first long request after each boot are not counted.

## Path coverage

| Path / provider | Format | Key conditions (shape, TP/EP, graph, capability) | Expected selection / fallback | Evidence (source / static / executed / measured) | Unverified scope |
| --- | --- | --- | --- | --- | --- |
| QSA indexer scoring, cuBLAS route | FP16 index cache | SM70 family (sm_70, sm_75), single request at or above `qsa_cublas_min_rows`, or the multi-request split | cuBLAS scorer; short and generic batches keep the paged Triton kernel as before | executed + measured (kernel benchmark, serving on Quadro RTX 8000); unit tests on V100 and RTX 8000 | other SM7x parts (sm_72) |
| QSA indexer top-k | int32 indices | SM70 family, block top-k 512 | exact lexicographic top-k (native op, sm_70 build) | executed (exact-index test on RTX 8000 and V100), serving | as above |

## Acceleration and benchmark contract (required for performance changes)

- Default enabled or opt-in: default on the SM70 family (now including Turing).
- Required CLI options and environment switches: none.
- KV cache dtype used for the benchmark: auto (fp16).
- Wheel SHA or source commit: fork commit ee261cc74 against 9a41a8e67 (both main 6f3557c97 plus our commits).
- PYTHONPATH, source overlays, or external native libraries used: fix arm PYTHONPATH on the fork tree with its own compile cache; same native build in both arms (this PR changes Python only).
- User-entry route-hit, speed, and output-quality evidence: see Test Result.
- Existing kernel/backend registry and capability-rejection reasons: the three QSA indexer gates accept the SM70 family instead of capability 7.0 alone.
- Per-engine configuration and deprecated-variable compatibility: no new fields.
- New/changed environment metadata and generated reference check: none.
- Route snapshot command and intentional changed rows: none.
- SM70 registered variables / unregistered reads / model-parameter restrictions, before -> after: unchanged.

## Test Result

- On a Quadro RTX 8000 (main 6f3557c97 plus this PR): 234 passed, 38 skipped, 5 failed. Main without this PR: 224 passed, 48 skipped, 5 failed; ten tests that skip on Turing on main now run there. On a Tesla V100: 272 passed, 5 failed with and without this PR.
- The five failures are the same on both GPUs with and without this PR: tests/kernels/test_top_k_per_row.py::test_persistent_topk_stress ("CUDA top-k contains values smaller than non-top-k" for a row of 106,861 scores with k 2,048; `persistent_topk` is the kernel Turing used for QSA before this PR, where it selects 512 blocks, a case this test does not cover) and four cases in tests/models/qwen4_exp/test_qsa_reference.py whose metadata-builder fakes lack is_dummy_batch, reads_sharded_group or input_size. pre-commit and mypy-3.10 pass.

- Kernel benchmark, ms per call for the last 1,616-row chunk, gates as on main -> this PR: 29,000 tokens 80.94 -> 2.28, 184,000 tokens 505.38 -> 10.86; the selected token sets are identical in all rows. This is the most expensive chunk of each prompt; the serving numbers below cover whole prefills.

- Serving, measured together with #PR_C (both act on the RTX 8000 stages), time to first token, fix (two boots) against base (two boots): TP2xPP2 ~29k tokens 14.8 s against 17.6-17.7 s, 184,134 tokens 126.3-129.2 s against 178.9-194.1 s, 249,950 tokens 157.0-162.4 s against 253.3-275.7 s; PP4 ~29k tokens 10.2-10.6 s against 11.6 s (its long-context times vary with the page cache of the PLE disk tier in both arms). A separate decode A/B with 20 sampled code answers per boot (TP2xPP2, two boots per arm) gave 77.3 and 76.9 tok/s against 77.0 and 76.9 tok/s, 51.0 ms median step time in all four boots, acceptance length 3.93 against 3.91.
- Outputs: greedy text differs between boots in both arms, as it does on main with fresh compile caches. A hallucination probe (a non-existent person, temperature 1.0, 20 draws per arm) was handled equally often correctly with and without the two PRs (PP4 13 and 14 of 20, TP2xPP2 18 and 19 of 20).
