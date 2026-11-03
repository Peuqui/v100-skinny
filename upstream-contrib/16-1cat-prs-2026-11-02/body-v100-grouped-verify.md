## Purpose

Speculative verification on Volta reads a long context once per query row in the XQA kernel. The grouped kernel reads it once for all rows, but its admission takes one KV head with six query heads on the 832-token page. A TP2 rank has two KV heads and twelve query heads, and an FP8 checkpoint has the 800-token page, so such a rank fell back to XQA, where each of the four rows reads the context itself: 5.46 ms per call at 222,203 tokens on a Tesla V100, against 1.29 ms for the packed Turing path.

This PR calls the grouped kernel once per KV head and admits the 800-token page next to 832; the kernel takes any page divisible by 16.

Not a duplicate: searched the open PRs of this repository (86 at the time of writing) for grouped verify and flash_v100. #1271 (Gemma4 on V100) appears for the V100 attention backend; it changes neither spec/verifier.py nor ops/sm70_grouped.py.

AI assistance was used for this change. I reviewed every line and ran the tests and measurements below.

## Test Plan

python -m pytest tests/kernels/test_sm70_fp16_grouped_verify.py
python -m pytest tests/v1/attention/test_flash_v100_kv_head_views.py
python -m pytest tests/v1/attention/test_sm70_fp16_grouped_admission.py
each on a Tesla V100 and on a Quadro RTX 8000
pre-commit run --from-ref origin/main --to-ref HEAD
pre-commit run mypy-3.10 --all-files --hook-stage manual

New in tests/kernels/test_sm70_fp16_grouped_verify.py: two KV heads on page 800 up to 131,072 tokens against a softmax reference.

## Path coverage

| Path / provider | Format | Key conditions (shape, TP/EP, graph, capability) | Expected selection / fallback | Evidence (source / static / executed / measured) | Unverified scope |
| --- | --- | --- | --- | --- | --- |
| FLASH_V100 speculative verify | fp16 KV, page 800 or 832 | several KV heads per rank, six query heads per KV head, SM70 | grouped kernel once per KV head | measured: TP2 on two Tesla V100; executed: kernel tests | TP sizes other than 2 in a model run |
| same | one KV head, page 832 | as before | unchanged: grouped kernel, one call | executed: kernel tests | none |
| same | other head ratio or page | admission rejects | XQA kernel as before | executed: admission tests | none |

## Acceleration and benchmark contract (required for performance changes)

- Default enabled or opt-in: default enabled, no switch.
- Required CLI options and environment switches: none for the change. Benchmark entry: Qwen3.8-27B FP8, --tensor-parallel-size 2 --dtype float16 --max-model-len 262144 --max-num-batched-tokens 2048 --block-size 16, MTP with 3 speculative tokens and attention_backend FLASH_ATTN, VLLM_USE_V2_MODEL_RUNNER=1, VLLM_1CAT_ENABLE_SM70_MTP_DEFAULTS=1.
- KV cache dtype used for the benchmark: the entry's default, fp16.
- Wheel SHA or source commit: fork build, reference tag verified-2026-10-31-prod-v5b, changed 671babc33, two commits on that tag (a build pin of the Turing library, which this entry on two V100 does not load, and this change), tag verified-2026-11-01-v100-verify in github.com/Peuqui/1Cat-vLLM. This branch carries the same commit on main.
- PYTHONPATH, source overlays, or external native libraries used: the fork tree on PYTHONPATH in both arms; native libraries identical in both arms. The tests of this PR ran with main's native build.
- User-entry route-hit, speed, and output-quality evidence: OpenAI API server entry, below.
- Existing kernel/backend registry and capability-rejection reasons: admission of the grouped verifier extended by the page size and by several KV heads; other shapes are rejected as before.
- Per-engine configuration and deprecated-variable compatibility: no new or changed setting.
- New/changed environment metadata and generated reference check: none.
- Route snapshot command and intentional changed rows: none.
- SM70 registered variables / unregistered reads / model-parameter restrictions, before -> after: no change.

## Test Result

Qwen3.8-27B FP8, TP2 on two Tesla V100, one reference run and one changed run through the same server entry (not an A/B/A/B; two earlier changed runs gave the same numbers within 0.6 ms), ms per decode round (lower is better) and tok/s (higher is better):
- 33,208 tokens: 64.8 ms reference, 53.8 ms changed.
- 184,134 tokens: 126.7 ms (21.0 tok/s) reference, 75.9 ms (35.3 tok/s) changed.
- 249,992 tokens: 156.1 ms (17.2 tok/s) reference, 84.8 ms (31.6 tok/s) changed.
- 259,552 tokens: 160.3 ms (16.8 tok/s) reference, 86.4 ms (31.2 tok/s) changed.

Short decode 48.17 tok/s (20.8 ms per token) reference, 47.26 tok/s (21.2 ms per token) changed, in this pair; after an 8,490-token prompt 50.93 tok/s (19.7 ms per token) reference, 53.21 tok/s (18.8 ms per token) changed. Prefill of 8,490 tokens 6.69 s (1,269 tok/s) reference, 6.64 s (1,278 tok/s) changed. First token at 259,552 tokens 480.5 s reference, 480.3 s changed. Memory peak per card 32,318 MiB reference, 32,344 MiB changed. KV pool 444,804 tokens in both.

Output: GSM8K-30 27/30 in both runs. Seven of eight fixed answers are byte-identical; the eighth keeps its result in other words (741 against 754 tokens).

Tests on this branch (main 4d57473db plus this commit; it merges cleanly into main 6db9c816c), main's native build, Tesla V100 / Quadro RTX 8000: tests/kernels/test_sm70_fp16_grouped_verify.py 15 passed / 15 skipped (the kernel is Volta-only); tests/v1/attention/test_flash_v100_kv_head_views.py 2 passed / 2 passed; tests/v1/attention/test_sm70_fp16_grouped_admission.py 14 passed / 14 passed. pre-commit run --from-ref origin/main --to-ref HEAD: every hook passes except check-layering, which reports vllm/model_executor/layers/fused_moe/runner/moe_runner.py (platform coupling 37 -> 39) exactly as main 4d57473db does. pre-commit run mypy-3.10 --all-files --hook-stage manual: no issues.

🤖 Generated with Claude Code (https://claude.com/claude-code)

https://claude.ai/code/session_0172gtUkBtN8FSrthHLCZe8W
