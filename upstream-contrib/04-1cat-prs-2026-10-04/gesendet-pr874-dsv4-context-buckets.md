Gesendet 04.10.2026 ~03:40 als 1CatAI/1Cat-vLLM#874 (Branch pr-dsv4-spec-context-buckets 48f0cb970)
Titel: [Perf][SM70][DeepSeek-V4] Bucket speculative decode graphs by context so the sparse indexer can take cuBLAS

## Purpose

Under a full CUDA graph the SM70 sparse indexer decode refuses its cuBLAS route (`_decode_cublas_blocker`: "a live key bound rather than a fixed full-graph bucket"), because cuBLAS scores the whole static key bound and a full graph is bound by the model's maximum length. With speculative decoding (MTP, DSpark) that is the only kind of decode graph: `_get_sm70_dsv4_decode_context_buckets` derives no context buckets when a speculative config is set ("require a separate end-to-end long-context gate"). So DeepSeek-V4-Flash with DSpark always scores its indexer with the paged Triton kernel, whose cost grows with the live context.

This PR

- derives the same automatic DeepSeek-V4 decode context buckets for single-request speculative graphs as for single-row decode (index_topk x smallest compress ratio, times 1/2/8/32/64, below max_model_len; `VLLM_SM70_DSV4_DECODE_CONTEXT_BUCKETS` still overrides, and multi-request graphs stay unbucketed as before), and
- lets the indexer take cuBLAS inside a context-bucket graph, where the key bound is the bucket rather than the maximum length. A full graph without a bucket (contexts beyond the largest bucket) keeps the paged kernel, as do bounds below the existing 1024-key minimum and the existing workspace budget check.

The long-context gate the exclusion asked for is below: needles at 29,888, 124,471 and 198,959 prompt tokens (65,536 bucket, 131,072 bucket, no bucket) and the quality prompts.

On our rig DeepSeek-V4-Flash PP5 only starts with #873, so the measurements carry it in both arms; this change does not depend on it.

Not a duplicate: no open PR touches the speculative bucket exclusion or the indexer's full-graph cuBLAS block (gh pr list --state open --search "context bucket", "context buckets speculative", "indexer cuBLAS", "decode_cublas", "DSV4_DECODE_CONTEXT_BUCKETS": none).

AI assistance was used for this change. I reviewed every line and ran the tests and measurements below.

## Test Plan

```bash
pytest tests/models/test_deepseek_v4_sm70_routes.py tests/v1/cudagraph/test_cudagraph_dispatch.py tests/kernels/test_deepseek_v4_sm70_indexer.py tests/models/test_deepseek_v4_sm70_sparse_policy.py
pre-commit run --files vllm/models/deepseek_v4/sm70/indexer.py vllm/v1/cudagraph_dispatcher.py tests/models/test_deepseek_v4_sm70_routes.py tests/v1/cudagraph/test_cudagraph_dispatch.py
```

The routes test now checks all three cases (eager: allowed, full graph without bucket: blocked, full graph with a 65,536-token bucket: allowed); the dispatcher test that asserted no automatic buckets for MTP now asserts the derived bucket.

## Acceleration and benchmark contract (required for performance changes)

DeepSeek-V4-Flash at PP5 on 3x Tesla V100-PCIE-32GB + 2x Quadro RTX 8000, DSpark with 5 speculative tokens, `--max-num-seqs 1`, CUDA graph size 6, FP8 KV cache, main's defaults (no kernel-config entry). Two boots per arm after a cold one, the first long request after a boot not counted; decode ms per step, lower is better:

| prompt | without this PR | with this PR | our production fork |
| --- | ---: | ---: | ---: |
| 26 tokens | 75 | 69-70 | 84-85 |
| 36 tokens | 78 | 72 | 87-88 |
| 1,437 tokens | 82 | 76 | 91 |
| 35.7k tokens | 96-98 | 74-76 | 85-86 |

Prefill of the 35.7k prompts 15.3-15.6 s in both arms (production 15.1-15.3 s). The startup log shows the buckets (2048, 4096, 16384, 65536, 131072) and "SM70 indexer decode: cuBLAS route" next to the paged kernel for the unbucketed graph; start time and KV cache (399,133 tokens) are unchanged. With `--max-num-seqs 4` a single request still uses a bucket: 46.8-48.1 tok/s against 41.9-42.1 tok/s in production.

## Test Result

- Tesla V100-PCIE-32GB: 104 passed. Quadro RTX 8000: 104 passed.
- The changed cases on main f551e0afe without this change: 3 of 4 fail (the unbucketed full-graph case only on the reason text).
- pre-commit passes.
- Needles (four facts at different depths, temperature 0): 4/4 at 29,888, 124,471 and 198,959 prompt tokens, as in production.
- Quality prompts (8) read by hand in both runs: equivalent to production, the unknown-person prompt correctly declined.
- Measurements on our build of the #837 head plus #856/#857 (cef0a2e4b, extensions built from source, sm_70 only) with #873; indexer.py and cudagraph_dispatcher.py are identical on current main.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
