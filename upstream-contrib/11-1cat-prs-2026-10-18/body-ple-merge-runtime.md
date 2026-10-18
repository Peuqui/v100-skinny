Title: [Bugfix][Qwen4Exp] Merge the resident PLE rows outside the engine config
Branch: Peuqui:pr-ple-merge-runtime (cc0934ee9 on main 6f3557c97; fork commit 6bbcd40be)

## Purpose

With the PLE offload worker and the disk cascade enabled, Qwen3.8-Flash-Next computes its n-gram embedding without the rows that live in device or pinned-host memory. PleOffloadLayer.forward asks offload_keeps_local_tables() on every call whether to gather its resident rows and merge the worker's rows into them. For Qwen4ExpNGramEmbedding that answer comes from ple_cascade_configured() or pinned_decode_active(). Since 41faaa6d1 the first reads kernel_config.ple_disk_cascade_active from get_current_vllm_config_or_none(), and the second (#996) reads kernel_config the same way. The worker sets that config around construction and weight loading, but not around execute_model, so at runtime the check answers "no" and the forward returns the offload worker's raw output buffer. The worker writes only the disk-tier slots of that buffer; every slot whose row is resident keeps whatever the buffer held, zeros on the first request and rows of earlier chunks afterwards.

How much of the table is affected depends on the placement. On our PP4 entry (131,045,319 of 320,001,536 rows in device memory) that was about 41 % of all n-gram head slots. On TP2xPP2 (125,439,175 of 160,000,768 rows per rank in device memory) most of the rows are resident. Because the stale rows depend on earlier requests, the same prompt got different log-probabilities on every pass within one boot.

The guarded constructor already decides once, while the config is current, whether the layer builds its resident tables. This PR records that decision in the instance and lets the forward merge exactly the tables it built. A regression test builds a layer while its keeps-local-tables answer is true, flips the answer as the missing runtime config does, and checks that the forward still merges its own rows; it fails on main.

Not a duplicate: gh pr list --repo 1CatAI/1Cat-vLLM --state all --search "offload_keeps_local_tables", "ple_disk_cascade_active", "resident PLE rows", "PLE cascade forward", "wait_offloaded_output", "Flash-Next nondeterministic" and "PLE zeros", and the same searches over issues, find no PR or issue about this; #856 (merged) fixed the cascade storage lookup under checkpoint layer names, #646 (closed) was the original cascade. The open PRs touching the same two files (#1020, #1023, #1094, #1167, #1214, #1217) do not change this check, and this PR adds no merge conflict to any of them (git merge-tree against each PR head; #1020 and #1023 conflict with main already).

AI assistance was used for this change. I reviewed every line and ran the tests and measurements below.

## Test Plan

```bash
pytest tests/v1/worker/test_ple_offload_worker.py tests/models/test_qwen4_ple_placeholder.py tests/models/test_ple_optional_cuda_import.py tests/models/qwen4_exp/test_gguf_pinned_policy.py tests/models/qwen4_exp/test_ple.py tests/compile/test_sm70_decode_graph.py tests/config/test_ple_cascade_capabilities.py tests/kernels/ple
pre-commit run --from-ref origin/main --to-ref HEAD
pre-commit run mypy-3.10 --hook-stage manual --from-ref origin/main --to-ref HEAD
```

Row check against the checkpoint: Qwen3.8-Flash-Next-180B-A4B NVFP4 with MTP, PP4 over two Tesla V100 and two Quadro RTX 8000, PLE offload worker with the disk cascade, --enforce-eager. A forward hook saved the n-gram embedding output and the worker's ids for a 3,707-token German prompt (prefill chunks 1,616 + 1,616 + 476 tokens); each slot was compared with the row read directly from the checkpoint's FP8 table (dequantized with the table scale for the fixed path, which returns the dequantized embedding).

Teacher-forced log-probabilities: the same prompt with prompt_logprobs, two passes in one boot, same setup.

Serving A/B on our fork (main 6f3557c97 plus our commits): fork commit 6bbcd40be (this change) against cf50e1a2c (its parent), PP4 and TP2xPP2, boots fix cold, base, fix, base, fix; the cold boot is not counted. Per boot: greedy answers to three prompts, four ~29k-token prompts, decode benchmark (prose, code, edit), eight quality prompts at temperature 0, and prompts of 184,134 and 249,992 tokens.

## Path coverage

| Path / provider | Format | Key conditions (shape, TP/EP, graph, capability) | Expected selection / fallback | Evidence (source / static / executed / measured) | Unverified scope |
| --- | --- | --- | --- | --- | --- |
| Qwen4Exp n-gram PLE, offload worker with disk cascade | FP8 table in safetensors (NVFP4 checkpoint) | PP4 and TP2xPP2, V100 and Quadro RTX 8000, eager and CUDA graphs | layer merges its resident rows with the worker's disk rows | executed + measured (row check, log-probabilities, serving A/B) | TP4 |
| Packed GGUF pinned decode (pinned_decode_active) | GGUF | same check in offload_keeps_local_tables() | records the same constructor decision | source | not run |
| Hybrid PLE policy | any | ple_policy().hybrid, bound at runtime | unchanged | static | |

## Acceleration and benchmark contract (required for performance changes)

Not a performance change. Python only; no new switches, fields or environment variables. The serving A/B below shows the speed is unchanged.

## Test Result

- Tests on main 6f3557c97 plus this PR: 260 passed, 34 failed. Main alone: 259 passed, 35 failed; the extra failure is the new test test_ple_layer_merges_local_tables_outside_the_engine_config. The 34 failures are identical with and without this PR and come from stale test fakes (CudaGraphManager has no capture_context, SimpleNamespace configs without kernel_config, the connector and ple_offload_layer modules without envs, Qwen4ExpNGramEmbedding fakes without layer_name); #1214 repairs part of them. pre-commit and mypy-3.10 pass.
- Row check, first 1,616-token chunk: base: disk-tier slots 15,264 of 15,264 equal the checkpoint row, resident slots 0 of 10,592. This PR: all 25,856 slots equal the dequantized checkpoint row (max abs difference 0). The other two chunks behave the same.
- Log-probabilities of the 3,707-token prompt: base: mean NLL 1.4115 in pass 1 and 1.4023 in pass 2, equal log-probabilities at 1 of 3,707 positions, same top-1 token at 3,466. This PR: mean NLL 1.3609 in both passes, all 3,707 log-probabilities identical.
- Serving speed, fix (two boots) against base (two boots): decode step times are equal (PP4 edit 54 ms = 92.3 against 92.5 tok/s, TP2xPP2 edit 45 ms = 110.0 against 110.0 tok/s); time to first token for ~29k tokens is equal (PP4 7.5-7.6 s against 7.5-8.0 s, TP2xPP2 11.8-11.9 s against 11.8 s; the first long request after each boot not counted); 184,134 tokens PP4 71.9 and 72.2 s against 73.3 and 74.5 s, TP2xPP2 99.3 and 99.6 s against 107.6 and 111.1 s; decode step at 249,992 tokens PP4 66.8 and 67.4 ms against 67.0 and 68.1 ms, TP2xPP2 59.2 and 57.9 ms against 58.8 and 58.4 ms. Prose tok/s follows the generated text through the MTP acceptance (PP4 47.1 against 45.6 tok/s, TP2xPP2 54.0 against 57.2 tok/s at acceptance 2.42 against 2.58; ~29k-token prompts PP4 37.5 against 35.5, TP2xPP2 43.8 against 41.8 tok/s).
- Outputs: the three fix boots per topology produced byte-identical greedy texts and quality answers, and the decode benchmark, which runs each prompt twice per boot, now gets the same token count and acceptance on both runs (base: 519 and 553 tokens for the same prose prompt). PP4 and TP2xPP2 now write the same greedy text for two of the three prompts and the same first 625 of 665 characters for the third; on base the two topologies share only the first 377, 211 and 66 characters. All quality answers read correct by hand (arithmetic, logic, code, summaries; a question about a non-existent person was declined without inventing a biography in both arms).
