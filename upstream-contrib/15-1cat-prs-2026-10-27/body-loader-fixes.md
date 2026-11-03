## Purpose

Two safetensors loader fixes found and verified in our fork while bringing up GLM-5.3-Flash-FP4-Local (ModelOpt mixed precision). Main cannot serve that checkpoint yet (its csf_32x2_v1 expert scales come with a separate PR), but the loader code involved is the same on main, and either fix applies to any checkpoint with the same layout. Each one stopped the boot or silently left weights empty:

1. Index shards in subfolders (4f9137429). The checkpoint keeps its BF16 attention tensors (KDA b/f_a/g_a/f_b/g_b, MLA kv_b, indexer wq_b) in tensors/native-QB.safetensors; model.safetensors.index.json points there, but the default loader only globbed weight files in the top folder. The file was never read and the parameters stayed empty without an error: perplexity 21,036 on German text and garbled output. Now, when the default pattern (without subfolders) matches, the shards the index names in subfolders are added if their file name matches the pattern, and without model-specific allow_patterns the loader stops when the index names a shard that would not be loaded (before: silently empty).
2. U16 and U32 through direct I/O (4998496f3). The checkpoint stores its csf_32x2_v1 scale page bases as U32 and the block offsets and exceptions as U16. Direct I/O did not know either dtype and stopped the boot ("does not know safetensors dtype 'U32'"); both now map like safetensors maps them, and the loader test checkpoint carries one tensor of each.

Not a duplicate: I searched the open PRs for "index shards", subfolder, allow_patterns, direct_io, U16, U32 and safetensors index, and compared the changed files with the complete file lists of all open PRs; none changes default_loader.py, weight_utils.py or direct_io.py. #1260 changes the GGUF loader only. The same bring-up also hit the excluded ModelOpt LM head crash ("'ParallelLMHead' object has no attribute '_layer_execution_policy'"); #1260 already fixes that, so it is not part of this PR.

AI assistance was used for this change. I reviewed every line and ran the tests and measurements below.

## Test Plan

python -m pytest tests/model_executor/model_loader/test_safetensors_index_filter.py tests/model_executor/model_loader/test_direct_io.py, one process per file.
pre-commit run --from-ref origin/main --to-ref HEAD; pre-commit run mypy-3.10 --hook-stage manual --from-ref origin/main --to-ref HEAD

## Path coverage

| Path / provider | Format | Key conditions (shape, TP/EP, graph, capability) | Expected selection / fallback | Evidence (source / static / executed / measured) | Unverified scope |
| --- | --- | --- | --- | --- | --- |
| DefaultModelLoader, safetensors with index | any | index names shards in subfolders, default allow pattern | subfolder shards loaded; unloaded indexed shard is an error | executed: test_safetensors_index_filter.py; measured: GLM-5.3 full-model weight comparison | models with their own allow_patterns: their subfolder shards are added too, only the missing-shard error is skipped; not booted |
| direct I/O safetensors reader | U16, U32 | any | read like safetensors | executed: test_direct_io.py | none |

## Acceleration and benchmark contract (required for performance changes)

- Default enabled or opt-in: bug fixes, always on.
- Required CLI options and environment switches: none.
- KV cache dtype used for the benchmark: n/a (no benchmark).
- Wheel SHA or source commit: written and first verified on a fork build (github.com/Peuqui/1Cat-vLLM); rebased onto main 6a143197e, unit-tested there.
- PYTHONPATH, source overlays, or external native libraries used: none for the tests.
- User-entry route-hit, speed, and output-quality evidence: GLM-5.3-Flash boots and answers correctly with these fixes in our fork (in production there since 2026-10-25); no speed effect intended.
- Existing kernel/backend registry and capability-rejection reasons: not touched.
- Per-engine configuration and deprecated-variable compatibility: no setting added or changed.
- New/changed environment metadata and generated reference check: none.
- Route snapshot command and intentional changed rows: none.
- SM70 registered variables / unregistered reads / model-parameter restrictions, before -> after: no change.

## Test Result

Tests on main 6a143197e with this branch (CPU, one process per file):
- tests/model_executor/model_loader/test_direct_io.py: 24 passed
- tests/model_executor/model_loader/test_safetensors_index_filter.py: 14 passed
pre-commit run --from-ref origin/main --to-ref HEAD: every hook passes except check-layering, which reports vllm/model_executor/layers/fused_moe/runner/moe_runner.py (platform coupling 37 -> 39); this PR does not touch that file and main 6a143197e reports the same. pre-commit run mypy-3.10 --hook-stage manual: passed. The pre-commit CI job of this PR fails exactly like main's own run at 6a143197e (run 40009610580): mypy in vllm/model_executor/layers/quantization/gguf_turbomind_moe.py:609 and vllm/v1/kv_cache_interface.py:156-158, and the check-layering count above; none of these files is touched here.

🤖 Generated with Claude Code (https://claude.com/claude-code)

https://claude.ai/code/session_0172gtUiymppDzziDqKfy78Q
