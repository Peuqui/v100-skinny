## Purpose

Two MLA memory reservations assumed many requests, even when the scheduler holds only a few. We found them while bringing up GLM-5.3-Flash in our fork (PP=5 over three 32 GiB Tesla V100 and two 48 GiB Quadro RTX 8000, --max-num-seqs 1), where they took gigabytes from every pipeline stage or made the profile run go out of memory:

1. Chunked-prefill workspace sized by the batch it can hold (fb7fc7c97) and capped at the batch's context (7cbf026a8). The workspace aimed at eight full-length requests and at least four pages per request, capped at 64k tokens. It now takes min(8, max_num_seqs) full requests and never more than max_num_seqs x max_model_len, still at least one page per request. The cap matters for hybrid models, which raise the attention page to the Mamba state page (GLM-5.3-Flash: 8,576 tokens), so four pages alone asked for 34,304 tokens. The profile run reserves the up-projected kv_b_proj output for this many tokens (GLM-5.3-Flash: 64 heads x 512 x 2 bytes per token). At max_model_len 16,384 with one sequence: 65,536 tokens (4 GiB per stage) before, 34,304 (2,144 MiB, the largest block in the memory snapshot of stage 0) with only the first change, 16,384 (1 GiB) with both. The size changes only when max_num_seqs is below 8 and max_model_len below 64k tokens; long-context engines keep the 64k cap. The result is rounded up to the DCP group size, since fewer than eight requests of an odd max_model_len would not divide across the DCP all-gather (no change without DCP).
2. Sparse indexer prefill buffer bound to max_num_seqs (2efa63d56). get_max_prefill_buffer_size always reserved 40 x max_model_len keys (132 bytes each), but the buffer only holds the keys of one prefill chunk's requests, at most max_num_seqs sequences of at most max_model_len tokens. With one sequence of 1,048,576 tokens that was 5.16 GiB per stage, and the boot ran out of memory in the profile run ("Tried to allocate 5.16 GiB" on a 48 GiB stage); at 262,144 tokens 1.29 GiB per stage. It now reserves min(40, max_num_seqs) sequences: 33 MiB for one sequence of 262,144 tokens.

Tests (e41ef0048 and the workspace test of 7cbf026a8): the workspace for one or few sequences, many sequences, hybrid pages and DCP groups of 1, 2 and 4; the indexer gather buffer for 1, 4, 40 and 256 sequences. The mutation that keeps 40 fixed is caught.

Not a duplicate: I searched the open PRs for workspace, "profile run", "chunked prefill", "indexer buffer", max_num_seqs, "MLA memory", "prefill buffer" and kv_b_proj, and compared the changed files with the complete file lists of all open PRs. #1260 (open) also changes mla_attention.py and backends/mla/indexer.py: it skips the profile run's kv_b_proj reservation for sparse MLA implementations (they never up-project the cached context; for GLM-5.3-Flash at long context that removes the 4 GiB the 64k cap still reserves) and changes the DeepGEMM gating of the indexer metadata builder. Our fork had the same sparse skip; it is left out here because #1260 covers it. Neither change of #1260 touches determine_chunked_prefill_workspace_size or get_max_prefill_buffer_size, so the two PRs only meet in nearby lines of mla_attention.py.

AI assistance was used for this change. I reviewed every line and ran the tests and measurements below.

## Test Plan

python -m pytest tests/v1/attention/test_mla_chunked_prefill_workspace.py tests/v1/attention/test_mla_sparse_profile_budget.py, one process per file.
pre-commit run --from-ref origin/main --to-ref HEAD; pre-commit run mypy-3.10 --hook-stage manual --from-ref origin/main --to-ref HEAD

## Path coverage

| Path / provider | Format | Key conditions (shape, TP/EP, graph, capability) | Expected selection / fallback | Evidence (source / static / executed / measured) | Unverified scope |
| --- | --- | --- | --- | --- | --- |
| MLA chunked-prefill workspace | any MLA | max_num_seqs < 8 and max_model_len < 64k, hybrid page size, DCP | min(8, max_num_seqs) requests, within max_num_seqs x max_model_len, rounded to the DCP group | executed: unit tests; measured: GLM-5.3-Flash boots in our fork (memory snapshot of stage 0) | dense MLA models (sizes computed, not booted); DCP > 1 not run |
| sparse indexer prefill buffer | DeepSeek-V3.2 / V4-style indexers, GLM-5.3 | max_num_seqs < 40 | min(40, max_num_seqs) x max_model_len keys | executed: unit test; measured: GLM-5.3-Flash boots at 262,144 and 524,288 tokens; DeepSeek-V4 on an 8-layer cut of the checkpoint (indexer entry 387.72 -> 10.67 MB, peak unchanged) | DeepSeek-V3.2 not run |

## Acceleration and benchmark contract (required for performance changes)

- Default enabled or opt-in: always on (memory sizing).
- Required CLI options and environment switches: none. Measured entry: GLM-5.3-Flash, PP=5 over three Tesla V100 and two Quadro RTX 8000, --max-num-seqs 1.
- KV cache dtype used for the benchmark: fp8_e4m3.
- Wheel SHA or source commit: fork builds (github.com/Peuqui/1Cat-vLLM); the commits are rebased onto main 6a143197e and unit-tested there, not booted with a full model on this base.
- PYTHONPATH, source overlays, or external native libraries used: none for the tests.
- User-entry route-hit, speed, and output-quality evidence: GLM-5.3-Flash boots with 524,288 tokens of context in our fork with these changes plus the sparse skip that #1260 also makes (production there since 2026-10-25); without the indexer bound the profile run ran out of memory at 1,048,576 tokens.
- Existing kernel/backend registry and capability-rejection reasons: not touched.
- Per-engine configuration and deprecated-variable compatibility: no setting added or changed.
- New/changed environment metadata and generated reference check: none.
- Route snapshot command and intentional changed rows: none.
- SM70 registered variables / unregistered reads / model-parameter restrictions, before -> after: no change.

## Test Result

Tests on main 6a143197e with this branch (Tesla V100, one process per file):
- tests/v1/attention/test_mla_chunked_prefill_workspace.py: 9 passed
- tests/v1/attention/test_mla_sparse_profile_budget.py: 4 passed
pre-commit run --from-ref origin/main --to-ref HEAD: every hook passes except check-layering, which reports vllm/model_executor/layers/fused_moe/runner/moe_runner.py (platform coupling 37 -> 39); this PR does not touch that file and main 6a143197e reports the same. pre-commit run mypy-3.10 --hook-stage manual: passed. The pre-commit CI job of this PR fails exactly like main's own run at 6a143197e (run 40009610580): mypy in vllm/model_executor/layers/quantization/gguf_turbomind_moe.py:609 and vllm/v1/kv_cache_interface.py:156-158, and the check-layering count above; none of these files is touched here.

🤖 Generated with Claude Code (https://claude.com/claude-code)

https://claude.ai/code/session_0172gtUiymppDzziDqKfy78Q
