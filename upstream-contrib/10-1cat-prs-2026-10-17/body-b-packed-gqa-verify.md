## Purpose

FA2 packs the query heads that share a KV head only when a request has a single query token. Speculative decoding verifies several tokens per request, so in the verify step each query head of a KV head reads the whole KV cache on its own. For Qwen3.8-27B with MTP on two Quadro RTX 8000 (TP2: 12 query heads, 2 KV heads, head dim 256, 4 tokens per request) this attention dominates the decode step at long contexts: 182.5 ms per step at 249,950 tokens of context against 32-34 ms at short contexts.

This PR splits that attention exactly, for uniform multi-token decode batches on FA2:

1. The request context, which every new token sees unmasked: query heads x tokens become the rows of each KV head, so the KV cache is read once per KV head.
2. The new tokens among themselves, causal.
3. Both partial results are merged by their log-sum-exp (`merge_attn_states`).

`AttentionConfig.flash_attn_pack_gqa_max_query_len` bounds the tokens per request (default 16, 0 disables the split). The gate also requires FA2, more than one query head per KV head, a causal batch with the same number of tokens per request, no sliding window, ALiBi or sinks, an unquantized KV cache, and batch invariance off.

The second commit fixes the gate: it required `q_descale` to be None, but the backend expands the layer's query scale on every CUDA platform, so the split never ran when serving. FA2 ignores the descale factors; quantized KV caches stay excluded by `is_quantized_kv_cache`. A test now covers the gate itself.

Not a duplicate: gh pr list --repo 1CatAI/1Cat-vLLM --state all --search "pack_gqa", "packed GQA", "GQA verify", "flash_attn verify" and "spec decode attention long context" find no PR that changes the FA2 verify attention. #284 and #697 reduce the DFlash2 verifier cost on the Flash-V100 routes, a different backend. The open PR touching vllm/config/attention.py (#1190) changes other parts. This PR adds no merge conflict to any open PR (git merge-tree against the head of each open PR that touches the same files, including the generated runtime reference; the ones that conflict do so with main already).

AI assistance was used for this change. I reviewed every line and ran the tests and measurements below.

## Test Plan

```bash
pytest tests/kernels/attention/test_flash_attn_packed_gqa_verify.py tests/v1/attention/test_flash_attn_turing_gates.py tests/kernels/attention/test_cascade_flash_attn.py tests/kernels/attention/test_attention_selector.py tests/config/test_runtime_default_ownership.py tests/engine/test_arg_utils.py tests/v1/worker/test_sm70_long_attention_graphs.py
pre-commit run --from-ref origin/main --to-ref HEAD
pre-commit run mypy-3.10 --hook-stage manual --from-ref origin/main --to-ref HEAD
```

New tests (`tests/kernels/attention/test_flash_attn_packed_gqa_verify.py`): the split against FA2's causal paged attention for 12/2, 16/4 and 8/1 query/KV heads, head dim 128 and 256, 2 and 4 tokens, one and three requests with uneven contexts (atol 2e-3); and the gate for each of its conditions.

Kernel benchmark on a Quadro RTX 8000, Qwen3.8-27B TP2 shapes, one request: FA2 causal paged attention (as on main) against the split, eager and captured as CUDA graphs (decode runs as graphs when serving), median of 20 calls.

Serving on our fork (main 6f3557c97 plus our commits). Arms: fix (fork commit 9a41a8e67 = this PR on top of the base) against base (fork commit 9e81ffeb0), Qwen3.8-27B NVFP4 with MTP (3 speculative tokens, so 4 tokens per verify step) on two Quadro RTX 8000, TP2, MRV2 (VLLM_USE_V2_MODEL_RUNNER=1), boots fix cold, base, fix, base, fix; the cold boot and the first long request after each boot are not counted. Per boot: three greedy reference prompts, four ~29k-token prompts (temperature 1.0), short greedy prompts, eight quality prompts and a long-context probe (184,133 and 249,950 tokens from repository documents, greedy, 400 output tokens). Step time = decode time / engine decode steps.

## Path coverage

| Path / provider | Format | Key conditions (shape, TP/EP, graph, capability) | Expected selection / fallback | Evidence (source / static / executed / measured) | Unverified scope |
| --- | --- | --- | --- | --- | --- |
| FA2 multi-token decode (speculative verify) | fp16 query and KV cache | FA2, query heads per KV head > 1, 2-16 tokens per request, uniform causal batch, unquantized KV, FULL CUDA graph | split context / new tokens with log-sum-exp merge; otherwise unchanged FA2 call | executed + measured (serving and kernel benchmark on Quadro RTX 8000); unit tests on RTX 8000 | FA3 (excluded by the gate), batches with more than one request when serving (covered by unit tests only) |

## Acceleration and benchmark contract (required for performance changes)

- Default enabled or opt-in: default for qualifying batches; `flash_attn_pack_gqa_max_query_len=0` disables it.
- Required CLI options and environment switches: none.
- KV cache dtype used for the benchmark: auto (fp16).
- Wheel SHA or source commit: fork commit 9a41a8e67 against 9e81ffeb0 (both main 6f3557c97 plus our commits).
- PYTHONPATH, source overlays, or external native libraries used: fix arm PYTHONPATH on the fork tree with its own compile cache; same native build in both arms (this PR changes Python only).
- User-entry route-hit, speed, and output-quality evidence: see Test Result.
- Existing kernel/backend registry and capability-rejection reasons: unchanged.
- Per-engine configuration and deprecated-variable compatibility: new field `AttentionConfig.flash_attn_pack_gqa_max_query_len` (default 16, 0 = off).
- New/changed environment metadata and generated reference check: the generated runtime reference lists the new field.
- Route snapshot command and intentional changed rows: none.
- SM70 registered variables / unregistered reads / model-parameter restrictions, before -> after: unchanged.

## Test Result

- On a Quadro RTX 8000 (main 6f3557c97 plus this PR): 317 passed, 82 skipped, 42 failed. Main without this PR: 283 passed, 82 skipped, 42 failed, the same 42 tests: 36 cases of tests/kernels/attention/test_cascade_flash_attn.py::test_cascade ("FlashAttention on Turing (sm75) only supports fp16 data type") and 6 in tests/v1/worker/test_sm70_long_attention_graphs.py.
- On a Tesla V100: 242 passed, 155 skipped, 11 failed with this PR against 242 passed, 154 skipped, 11 failed on main, the same 11 tests (5 in tests/kernels/attention/test_attention_selector.py, 6 in tests/v1/worker/test_sm70_long_attention_graphs.py); the new test module skips there. pre-commit and mypy-3.10 pass.

- Kernel benchmark, ms per layer call, FA2 causal -> split, eager / captured as a CUDA graph: 2k tokens of context 0.187 -> 0.214 / 0.145 -> 0.101, 33k 1.566 -> 0.451 / 1.510 -> 0.365, 184k 6.519 -> 1.327 / 6.355 -> 1.218, 250k 8.703 -> 1.725 / 8.735 -> 1.655; maximum relative difference of the outputs 8.1e-4.

- Decode step in ms (fix: two boots; base: two boots) and output speed: ~29k-token prompts (temperature 1.0) 38-39 ms = 53.8-63.5 tok/s against 52 ms = 40.2-47.3 tok/s with the same acceptance per prompt; 184,133 tokens 55.2-55.3 ms = 50.1-50.2 tok/s against 142.9-143.0 ms = 19.4 tok/s; 249,950 tokens 63.2 ms = 41.5-41.6 tok/s against 182.5 ms = 14.2 tok/s.
- Short contexts cost about 1 ms per step more: prose 33 ms = 74.4-74.7 tok/s against 32 ms = 76.1-76.3 tok/s, code 33 ms = 114.1-115.3 tok/s against 32-33 ms = 116.4-117.7 tok/s; the 1.9k-token edit prompt is faster, 34 ms = 119.0-119.2 tok/s against 117.3-117.4 tok/s. A context-length threshold does not work inside FULL CUDA graphs, which are captured once per batch shape, so the split also runs at short contexts.
- Prefill unchanged (time to first token at 249,950 tokens 592.8-592.9 s against 592.9-593.0 s).
- Outputs: the three greedy reference prompts are identical in all five boots, and the eight quality prompts are word-identical to the base.
