## Purpose

Two changes for long DeepSeek-V4 contexts on Volta and Turing. This PR is stacked on #1282: it needs that PR's Triton prefill index scorer (_prefill_index_scores_kernel), the triton_tensor_cores argument of the indexer entries and _TC_TOKENS_PER_PASS, and it sits on #1282's commit that keeps the Triton prefill index scores in FP32. Its own diff is the last two commits (git log e9147089c..bbb5f46b0).

First, score the decode index keys in one kernel. With context buckets (#874, #1222) the decode indexer scores a one-request verifier batch through cuBLAS: the keys are gathered and widened to fp16, one [rows x heads, keys] GEMM writes FP32 scores to memory, and the relu, head-weight and head-sum epilogue reads them back. That is the part of a decode round that grows with the context: in an nsys trace on PP=5 at 282,241 tokens, the score GEMMs and the epilogue took about 9.6 ms more per decode round than after a short prompt, summed over the five stages. With kernel_config.triton_sm7x_tensor_cores the prefill scorer of #1282 does all of it in one pass over the FP8 keys. The gather kernel gets a DEQUANT_FP16 switch and copies the FP8 bytes once per request, and every row and head of the request is scored together. The per-head products stay FP32, as in the cuBLAS decode route: the keys enter without their scale, so one product can pass fp16's 65504, which GLM-5.3's index queries reach. The fused route takes the cuBLAS route's admission (_decode_cublas_blocker) and passes its graph-tail test; without the setting the cuBLAS route is unchanged. Its route log line is process-scoped, as #1293 makes the other decode route lines, so every pipeline stage reports its route.

Second, halve the sparse prefill workspace. The scores and probabilities of the tensor-core sparse prefill route are sized for the widest layer, L/128 plus the window slots, so with a long --max-model-len they set the workspace of every pipeline stage. 512 instead of 1,024 tokens per pass halves them. Once they are smaller, the eager prefill KV workspace of the cr=4 layers would become the largest user: it reserves PREFILL_CHUNK_SIZE (8) rows of about L/4 tokens each, but a chunk holds whole requests, and an engine with max_num_seqs below 8 never fills more rows than it has sequences. The SM70 sparse impl now reserves min(8, max_num_seqs) rows and plans its prefill chunks with the same number, which gives the same chunks, since a step never holds more prefills than max_num_seqs.

Neither change reaches DeepSeek-V4.1: its attention_ops.py calls attend_gathered_keys with its own tokens_per_pass and uses neither the SM70 indexer entries, the tensor-core prefill route nor this impl. The GLM-5 next kpool indexer passes the new argument; in our GLM-5.3-Flash runs its decode indexer takes the paged Triton kernel ("the cuBLAS route needs an int32 block table that covers the key bound"), so the fused scorer is not reached there.

Not a duplicate: I searched the open PRs of this repository with gh pr list --state open --search for indexer decode, decode indexer, sm70 indexer, fused scorer, index scores, sparse prefill, sparse prefill workspace, TOKENS_PER_PASS, prefill chunk, DeepSeek-V4 SM70, deepseek_v4 workspace, triton_sm7x_tensor_cores, sm70_indexer_decode_logits, _prefill_chunk_requests and max_num_seqs workspace. Apart from my own #1280, #1282 and #1293, which this builds on or complements, none changes the decode index scorer or the sparse prefill pass. #1328 changes the precision of the cuBLAS prefill score tile, a route this PR does not touch. My merged #1275 bounds the MLA chunked-prefill workspace and the indexer prefill buffer by max_num_seqs (mla_attention.py, mla/indexer.py); the eager prefill KV workspace of the DeepSeek-V4 SM70 sparse impl that the second change bounds the same way is a separate allocation in vllm/models/deepseek_v4/sm70/sparse.py. None of the open PRs from other authors that these searches return changes a file of this PR.

AI assistance (Claude) was used for this change. I reviewed every changed line and ran the tests and measurements below.

## Test Plan

python -m pytest tests/kernels/test_deepseek_v4_sm70_indexer.py tests/kernels/attention/test_dsv4_sparse_prefill_bmm.py tests/models/test_deepseek_v4_sm70_routes.py tests/models/test_deepseek_v4_sm70_sparse_policy.py tests/models/glm5next/test_sm70_sparse.py, on a Tesla V100 and on a Quadro RTX 8000, each with a Triton build that lowers tl.dot to tensor cores on sm_70/sm_75 and with stock Triton 3.6. With stock Triton the fused-scorer cases and the multi-pass test skip by design (requires_tensor_core_triton).
pre-commit run --from-ref 4d57473db --to-ref HEAD (main below the whole stack, so the range covers #1280 and #1282 as well)
pre-commit run check-layering --all-files
pre-commit run mypy-3.10 --all-files --hook-stage manual
DeepSeek-V4-Flash through the OpenAI API server, A/B series below, and a boot of this branch.

## Path coverage

| Path / provider | Format | Key conditions (shape, TP/EP, graph, capability) | Expected selection / fallback | Evidence (source / static / executed / measured) | Unverified scope |
| --- | --- | --- | --- | --- | --- |
| DeepSeek-V4 decode indexer, sm70_indexer_decode_logits from SparseAttnIndexer | FP8 index cache, relu scores, 64 index heads, head dim 128 | SM70 and SM75; triton_sm7x_tensor_cores on; sm70_sparse.indexer_decode_cublas on; cuBLAS route admitted (context-bucket graph or eager, key bound at least indexer_decode_min_keys) | fused multi-row scorer, log "SM70 indexer decode: fused multi-row scorer" | measured: model A/B on PP=5 with V100 and RTX 8000 stages; executed: graph-tail test with 3, 8 and 13 rows in native, flattened and two-request layouts | TP > 1; cards other than V100 and RTX 8000 |
| same entry, setting off or route blocked | same | triton_sm7x_tensor_cores off, or a cuBLAS blocker (full graph without context bucket, key bound below the minimum, block table) | cuBLAS route or paged Triton kernel, unchanged | executed: the existing decode tests with triton_tensor_cores=False; measured: reference arms | none |
| GLM-5 next decode indexer, sparse_attn_indexer_kpool | FP8 index cache | exact SM70 on this base | passes triton_tensor_cores; same admission as above | source; GLM-5.3-Flash route log: paged Triton kernel, cuBLAS blocker on the block table | a GLM model that admits the cuBLAS decode route |
| DeepSeek-V4 tensor-core sparse prefill, sparse_attn_prefill_tensor_cores | fp16 KV rows, 64 heads, head dim 512 | SM70 and SM75; triton_sm7x_tensor_cores on | 512 tokens per pass instead of 1,024; scores and probabilities sized for 512 | measured: model A/B; executed: test_tensor_core_route_spans_several_passes (two full passes and a partial one against the BMM route) | as above |
| DeepSeek-V4 SM70 sparse impl, eager prefill KV workspace and chunk plan | fp16 | SM70 and SM75, any prefill route; max_num_seqs | min(8, max_num_seqs) rows and chunks of that many requests | executed: unit test for 1, 3 and 64 sequences; measured with max_num_seqs 1 | model runs with max_num_seqs from 2 to 7 |
| BMM sparse prefill, sparse decode BMM, DeepSeek-V4.1 attend_gathered_keys | - | - | unchanged (MAX_TOKENS_PER_PASS 128, V4.1 keeps its own tokens_per_pass) | source | none |

## Acceleration and benchmark contract (required for performance changes)

- Default enabled or opt-in: the fused decode scorer and 512 tokens per pass are opt-in through kernel_config.triton_sm7x_tensor_cores (#1282, default off); the scorer also needs the cuBLAS decode route (sm70_sparse.indexer_decode_cublas, default on) to be admitted for the call. The workspace row bound min(8, max_num_seqs) is default enabled in the SM70-family DeepSeek-V4 sparse impl.
- Required CLI options and environment switches: --kernel-config '{"triton_sm7x_tensor_cores":true}' and a Triton that lowers tl.dot to tensor cores on sm_70/sm_75 (otherwise the engine stops, #1282). Benchmark entry, same in both arms: --pipeline-parallel-size 5 --kv-cache-dtype fp8 --max-num-seqs 1 --max-num-batched-tokens 1024 --max-model-len 343000 --num-gpu-blocks-override 3600, speculative method dspark with 5 speculative tokens, CUDA graph capture sizes 5 and 6, kernel config also with sm70_fp8.native.fp8_tune_small_shapes and sm70_runtime.freeze_tuning_after_warmup, VLLM_USE_V2_MODEL_RUNNER=1, VLLM_PP_LAYER_PARTITION=10,8,8,8,9, VLLM_SM70_INDEXER_PREFILL_TILE_MB=64, VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64.
- KV cache dtype used for the benchmark: fp8.
- Wheel SHA or source commit: fork builds in github.com/Peuqui/1Cat-vLLM. Scorer series: reference tag verified-2026-10-28-prod-v5 (89ce1e78b), changed 60ee88f91 = that tag plus the scorer commit (branch dsv4-decode-fused-scorer; the same decode arithmetic as this PR's first commit, which differs only in the log scope and in dropping a prefill rounding switch that the FP32 fix in #1282 makes unnecessary). Pass series: reference f1b54cf46 (our build with the commit of #1344), changed = that commit plus this PR's second commit, committed as 772cbe803 (branch dsv4-prefill-pass512). This branch: the head of #1282 (e9147089c, with its FP32 score fix) plus the two commits; the stack merges cleanly into main 6db9c816c.
- PYTHONPATH, source overlays, or external native libraries used: the fork tree and a patched Triton 3.6 build (tensor cores on sm_70/sm_75) on PYTHONPATH in both arms; the native _C library identical in both arms of each series.
- User-entry route-hit, speed, and output-quality evidence: OpenAI API server entry. While capturing the context-bucket decode graphs, every changed run logged "SM70 indexer decode: fused multi-row scorer on cuda:0" and every reference run "SM70 indexer decode: cuBLAS route on cuda:0" (stage 0 only; the line was rank-local then). Speed and output below.
- Existing kernel/backend registry and capability-rejection reasons: no registry change. The scorer uses the cuBLAS route's admission; KernelConfig.use_triton_sm7x_tensor_cores checks the capability of the layer's own device (#1282).
- Per-engine configuration and deprecated-variable compatibility: no new setting. The DeepSeek-V4 attention layer keeps scheduler_config.max_num_seqs from when it is built, as it keeps max_num_batched_tokens and max_model_len.
- New/changed environment metadata and generated reference check: none.
- Route snapshot command and intentional changed rows: with the setting on, a decode indexer call that the cuBLAS route admits logs "fused multi-row scorer" instead of "cuBLAS route"; no other route changes.
- SM70 registered variables / unregistered reads / model-parameter restrictions, before -> after: no change.

## Test Result

DeepSeek-V4-Flash, PP=5 on 3 Tesla V100 + 2 Quadro RTX 8000 (stage 0 and stage 4 on the RTX 8000), runs T T P T P through the same server entry, counted T#2 + T#4 against P#3 + P#5; the first run after a build does not count. First token in seconds and decode round in ms, lower is better; the tok/s of a round follow the acceptance of the run, so the ms per round are the number to compare.

Fused decode scorer, changed = reference plus the scorer commit:
Decode round at 32,668 tokens: 57.9 / 57.7 ms changed, 57.6 / 58.0 ms reference.
Decode round at 178,480 tokens: 64.5 / 64.6 ms changed, 72.1 / 72.2 ms reference.
Decode round at 287,039 tokens: 64.4 / 64.1 ms changed, 72.7 / 72.5 ms reference.
Decode round at 336,216 tokens: 65.0 / 65.0 ms (55.3 / 49.1 tok/s) changed, 73.0 / 73.0 ms (44.8 / 44.1 tok/s) reference.
That is 7.6 to 8.4 ms less per round (10.5 to 11.5 %) from 178,480 tokens on; the growth from 32,668 to 336,216 tokens falls from about 15 ms to about 7 ms.
Unchanged: prefill of an 8,921-token prompt 2.30 / 2.31 s (3,878 / 3,854 tok/s) changed, 2.30 / 2.30 s (3,886 / 3,882 tok/s) reference; short decode 56.63 / 56.60 tok/s changed, 56.63 / 56.62 tok/s reference (17.7 ms per token); first token at 336,216 tokens 92.17 / 92.41 s changed, 91.88 / 91.71 s reference (the scorer does not run in prefill).
Memory peak per card (MiB, lower is better): Tesla V100 32,488 / 32,486 in all counted runs, RTX 8000 last stage 47,789 in all, RTX 8000 first stage 39,765 / 39,681 changed, 39,681 / 39,885 reference. KV pool 349,900 tokens in all runs.
Output: GSM8K-30 29/30 in all five runs, every answer identical across them (12,445 tokens, acceptance length 4.118). These zero-shot requests with at most 1,024 answer tokens stay in the 2,048-token context bucket, whose 512 compressed keys are below the scorer's 1,024-key bound, so they show that the rest of the path is unchanged. Eight fixed questions byte-identical to our production reference. The long-context answers from 178,480 to 336,216 tokens were read by hand: in both builds they are coherent summaries of the documents in the prompt. At these lengths greedy text also differs between two boots of the same build on this setup, so it is not compared byte for byte; at kernel level the graph-tail test checks the FP32 top-k set.
@@KERNEL@@
A second series a few hours later, our combined build with this commit against the build before it (that build also carries a prefill-overlap fix for tensor parallelism and a newer Turing FA2 library): decode round 64.4 / 64.4 against 74.1 / 72.5 ms at 178,480 tokens, 64.5 / 64.2 against 72.9 / 73.2 ms at 287,039 and 64.7 / 65.3 against 75.1 / 73.6 ms at 336,216. Another process loaded onto the first RTX 8000 during two of those runs, so their short-prompt numbers do not count.

512 tokens per pass, changed = reference plus the second commit:
First token at 178,480 tokens: 41.20 / 41.20 s changed, 41.11 / 41.06 s reference.
First token at 287,039 tokens: 73.53 / 73.51 s changed, 73.39 / 73.38 s reference.
First token at 336,216 tokens: 89.90 / 89.97 s changed, 89.72 / 89.69 s reference (+0.3 %).
Prefill of an 8,921-token prompt: 2.29 / 2.31 s (3,898 / 3,869 tok/s) changed, 2.29 / 2.28 s (3,903 / 3,914 tok/s) reference.
Decode round at 178,480 tokens: 63.7 / 64.2 ms changed, 64.4 / 64.1 ms reference; at 287,039: 64.1 / 63.8 against 64.4 / 64.1 ms; at 336,216: 64.9 / 64.9 ms (55.9 / 50.8 tok/s) against 64.9 / 64.3 ms (48.8 / 52.5 tok/s). Short decode 55.91 / 55.89 tok/s in both (17.9 ms per token).
Memory peak per card (MiB, lower is better): Tesla V100 30,432 changed, 30,992 reference (-560); RTX 8000 last stage 46,701 changed, 47,241 reference (-540); RTX 8000 first stage 39,885 in both. Available KV cache memory per worker 8.92 / 1.84 / 1.84 / 1.84 / 0.95 GiB changed, 8.38 / 1.29 / 1.29 / 1.29 / 0.41 GiB reference; --num-gpu-blocks-override held the pool at 349,900 tokens in both.
Output: GSM8K-30 29/30 in all five runs, every answer identical across them (12,285 tokens, acceptance length 4.103).
With this change and #1344, --max-model-len 750,000 (pool 760,402 tokens) and 960,000 (pool 971,476 tokens) booted, and prompts of 735,446 and 934,854 tokens ran without error (first token 266.72 s and 388.53 s; memory peak at 960,000: 31,864 MiB per Tesla V100, 48,193 MiB on the last RTX 8000). The committed change has run in our production build since; its control run at --max-model-len 500,000 reproduced the eight fixed answers byte for byte and ran a 486,316-token prompt.

Tests on this branch: @@TESTS@@

Boot of this branch: @@BOOT@@

Static checks on this branch: pre-commit run --from-ref 4d57473db --to-ref HEAD passes every hook except check-layering, which reports only vllm/model_executor/layers/fused_moe/runner/moe_runner.py (platform coupling 37 -> 39), a file this PR does not touch; check-layering --all-files reports the same, exactly as main 4d57473db does. pre-commit run mypy-3.10 --all-files --hook-stage manual: no issues apart from three in vllm/v1/attention/backends/flash_v100/runtime.py that come from a local flash_attn_v100 checkout and that main 4d57473db shows the same way.

🤖 Generated with Claude Code (https://claude.com/claude-code)

https://claude.ai/code/session_0172gtUkBtN8FSrthHLCZe8W


===== INTERN, NICHT SENDEN: Fundstellen (Peuqui) =====

Titelvorschlag: [Perf][SM70][SM75][DeepSeek-V4] Long context: score decode index keys in one fused kernel, halve the sparse prefill workspace

Purpose
- 9.6 ms je Runde (Score-GEMMs + Epilog, 282 241 Token, PP=5): STAND.md Z. 138 (Nachtrag 28.10. 18:30), Rohdaten ~/.cache/nsys-runs/dsv4-longdecode-1028/. Die dort genannten weiteren 1,6 ms (_dequant_gather_rows_kernel) NICHT zitiert: der Kern gehört zu sparse_decode_bmm.py (Sparse-MLA-Decode), nicht zur Indexer-Schlüsselkopie; STAND.md Z. 138 benennt ihn falsch.
- GLM-Decode nimmt den paged-Triton-Kern: ~/.cache/bench-scripts/ab_v5b_glm_a/journal-*.txt ("SM70 indexer decode: paged Triton kernel on cuda:0, the cuBLAS route needs an int32 block table that covers the key bound."), Fork-Build mit Scorer.
- Duplikatsuche: Skript-Ausgabe /tmp/claude-1000/-home-mp-Projekte-vllm-research/0197c9c1-5674-4590-b355-e9e32ee28cb0/scratchpad/dsv4ld/dup_search.log (02.11., main 6db9c816c). Vor dem Senden neu abfragen.

Contract
- Startparameter der Messarme: ~/.cache/prod-switch-2026-09-29/config.yaml.vor-prod-v5b-2026-10-31 Z. 35–58 (P-Arm Scorer-Reihe), ~/.cache/bench-scripts/fscore_entry_dsv4_2026-10-31.yaml (T-Arm), gleiche Flags.
- Route-Treffer: ~/.cache/bench-scripts/ab_fscore_dsv4_a/journal-{1,2,4}-T.txt je 1x "fused multi-row scorer on cuda:0", journal-{3,5}-P.txt je 1x "cuBLAS route on cuda:0".
- _C 44bb1007516a: Kopfzeilen ab_fscore_dsv4_2026-10-31a.log Z. 1, ab_pass512_dsv4_2026-11-01a.log Z. 2.
- Saubere Mergebarkeit: git merge-tree --write-tree origin/main HEAD, Exit 0 (main 6db9c816c).
- Umgebungsvariablen in vllm/envs.py registriert (geprüft im PR-Baum).

Scorer-Reihe: ~/.cache/bench-scripts/ab_fscore_dsv4_2026-10-31a.log (T = 60ee88f91, P = prod-v5)
- 32 668: T2 Z. 51 (57.9), T4 Z. 97 (57.7), P3 Z. 74 (57.6), P5 Z. 120 (58.0)
- 178 480: Z. 52 (64.5), Z. 98 (64.6), Z. 75 (72.1), Z. 121 (72.2)
- 287 039: Z. 53 (64.4), Z. 99 (64.1), Z. 76 (72.7), Z. 122 (72.5)
- 336 216: Z. 54 (65.0, 55.3 tok/s, ttft 92.17), Z. 100 (65.0, 49.1, 92.41), Z. 77 (73.0, 44.8, 91.88), Z. 123 (73.0, 44.1, 91.71)
- 7,6–8,4 ms / 10,5–11,5 %: aus den Mittelwerten je Länge gerechnet (178k 64,55 gegen 72,15; 287k 64,25 gegen 72,60; 336k 65,0 gegen 73,0). Wachstum 32k→336k: T 57,8→65,0 (+7,2), P 57,8→73,0 (+15,2).
- Prefill/Decode kurz: Z. 47 (T2 2.30 s 3878, 56.63), Z. 93 (T4 2.31 s 3854, 56.60), Z. 70 (P3 2.30 s 3886, 56.63), Z. 116 (P5 2.30 s 3882, 56.62)
- Speicher: Z. 56, 102, 79, 125; Pool: Z. 58, 104, 81, 127
- GSM8K: Z. 48–50, 94–96, 71–73, 117–119; Antwortgleichheit aus ~/.cache/bench-scripts/ab_fscore_dsv4_a/*-gsm8k.json (got, Tokenzahl, Schluss, finish je Frage; 30/30 in allen fünf Armen), Vergleichsskript /tmp/claude-1000/-home-mp-Projekte-vllm-research/0197c9c1-5674-4590-b355-e9e32ee28cb0/scratchpad/dsv4ld/compare_gsm8k.py
- Acht Fragen: Z. 20–27 (nur T#1 gegen Produktionskontrolle)
- Schwelle 1 024 Schlüssel: vllm/config/sm70_sparse.py Z. 38 (VLLM_SM70_INDEXER_DECODE_CUBLAS_MIN_KEYS 1024); Schlüsselgrenze = max_seq_len // compress_ratio (vllm/v1/attention/backends/mla/indexer.py Z. 890); Buckets 2048 … laut journal-2-T.txt Z. 587; GSM8K zero-shot (gsm8k_api_client.py Z. 82, max_tokens 1024, längste Antwort 1024 Token). Dass die Frage plus Antwort unter 2 048 Token bleibt, ist erschlossen, nicht gezählt (Fragen um 100–200 Token, Vermutung). Für die acht festen Fragen ist die Promptlänge nicht geprüft, daher dort keine Aussage zur Schwelle.
- Route-Zeile fällt beim Graph-Capture: journal-2-T.txt Z. 587–612 (11:32:00 Buckets, 11:32:03 fused multi-row scorer, vor "bereit").
- Langkontext von Hand gelesen: ~/.cache/bench-scripts/quality_2026-09-27/longctx-fscore-dsv4-a-{2-T,3-P,4-T}.json (ctx150k, ctxmax), gelesen 02.11.; T#1 gegen T#2 nicht gleich -> kein Bytevergleich möglich.
- @@KERNEL@@: Die früheren Kernel-Zahlen (0.195/0.104 usw.) stehen nur in STAND.md Z. 130, ohne Rohlog, deshalb aus Text und Commit-Nachricht entfernt. Entweder ~/.cache/bench-scripts/dsv4_indexer_decode_microbench_2026-10-31.py gegen diesen Baum auf V100 und RTX mit Log laufen lassen und die Zeile füllen, oder den Platzhalter streichen.
- 60ee88f91 (gemessener T-Arm) rechnete im Decode mit round_fp16=False, also dieselbe fp32-Arithmetik wie c43d421ed; Prefill rundete dort in BEIDEN Armen auf fp16 (Stand vor ebf0f603f). Der fp32-Prefill-Fix selbst ist im Modell noch nicht gemessen; der eigene Boot (@@BOOT@@) deckt ihn mit ab.
- Zweite Reihe: ~/.cache/bench-scripts/ab_v5b_dsv4_2026-10-31a.log Z. 52/98 vs 75/121 (178k), 53/99 vs 76/122 (287k), 54/100 vs 77/123 (336k); Fremdlast Qwen3VL auf GPU 0 in T#4/P#5: STAND.md Z. 109.

Pass-512-Reihe: ~/.cache/bench-scripts/ab_pass512_dsv4_2026-11-01a.log (T = f1b54cf46 + 4 geänderte Dateien, P = f1b54cf46)
- ttft 178 480: Z. 55 (41.2), 103 (41.2), 79 (41.11), 127 (41.06)
- ttft 287 039: Z. 56 (73.53), 104 (73.51), 80 (73.39), 128 (73.38)
- ttft 336 216: Z. 57 (89.9), 105 (89.97), 81 (89.72), 129 (89.69); +0,3 % = 89,935 gegen 89,705 (+0,26 %)
- Prefill: Z. 50 (2.29 s 3898), 98 (2.31 s 3869), 74 (2.29 s 3903), 122 (2.28 s 3914)
- ms je Runde: Z. 55/103 vs 79/127, 56/104 vs 80/128, 57/105 vs 81/129 (tok/s 55.9, 50.8 vs 48.8, 52.5)
- Decode kurz: Z. 50, 98, 74, 122 (55.91/55.89 vs 55.91/55.89)
- Speicher: Z. 59, 107 vs 83, 131; KV je Worker: Z. 60, 108 vs 84, 132; Pool Z. 61, 109, 85, 133
- GSM8K: Z. 51–53, 99–101, 75–77, 123–125; Antwortgleichheit ab_pass512_dsv4_a/*-gsm8k.json, 30/30
- 750k/960k: ~/.cache/bench-scripts/ab_pass512c750k_dsv4_2026-11-01a.log Z. 33, 35, 37; ab_pass512c960k_dsv4_2026-11-01a.log Z. 33, 35, 37 (Tabelle STAND.md Z. 50–51)
- Produktionskontrolle mit 772cbe803: ~/.cache/bench-scripts/validate_v5c_dsv4_2026-11-01.log Z. 3 (Baum, 0 geänderte Dateien), Z. 72 (486 316 Token), Z. 75–82 (8/8 gleich)
- T-Arm lief 19:38–20:39 mit unversionierten Änderungen; ~/.cache/bench-scripts/chain_e_2026-11-01.log Z. 8–12 zeigt den Diffstat zur Messzeit: sparse_prefill_bmm.py "2 +-" (nur die Konstante), Commit 772cbe803 "5 +++-" (plus Kommentar), die anderen drei Dateien gleich. Gemessen wurde also der Code des Commits.
- Kerntests "68 passed" auf GPU 1 (V100) und GPU 2 (RTX): chain_e_2026-11-01.log Z. 16–17.
- Kopfzeile der Pass-Logs: "_vllm_fa2_C_sm75.abi3.so: Datei oder Verzeichnis nicht gefunden" im Baum 1Cat-vLLM-pass512 – T-Arm ohne Turing-FA2-Bibliothek gebootet. Deshalb im Contract nur "_C identisch", nicht "alle nativen Bibliotheken".

Statische Prüfungen (02.11., Stand c43d421ed/81cddeb8f auf ebf0f603f): /tmp/claude-1000/-home-mp-Projekte-vllm-research/0197c9c1-5674-4590-b355-e9e32ee28cb0/scratchpad/dsv4ld/p4b_precommit.log (18 bestanden, nur moe_runner 37 -> 39), p4b_mypy310.log (nur die drei bekannten Umgebungsfehler in flash_v100/runtime.py, weil im Baum jetzt das unversionierte flash_attn_v100/ liegt; ohne das Verzeichnis grün: mypy310.log vor dem Neuaufsetzen, fp32_mypy310.log im Fix-Baum). Mergeprobe gegen main 6db9c816c: Exit 0.
