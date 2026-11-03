## Purpose

Two log lines that only the first pipeline stage printed, so a pipeline-parallel deployment could not be read from its log.

1. Available KV cache memory per worker (f1948edad). Each worker logs "Available KV cache memory: X GiB" with info_once, whose default scope "local" logs only on the first local rank, so on one node only stage 0 prints its figure. The KV pool is sized by the smallest worker, which is often another stage. The engine core now logs the figures it receives from all workers in one line. In a DeepSeek-V4-Flash PP5 boot (three Tesla V100 and two Quadro RTX 8000), the worker line said "Available KV cache memory: 8.47 GiB" (stage 0), while the new line reads "Available KV cache memory per worker: 8.47 GiB, 1.39 GiB, 1.39 GiB, 1.39 GiB, 0.50 GiB": the last stage limits the pool.
2. SM70 indexer decode route (99524c6ba). The two "SM70 indexer decode: ..." lines in vllm/models/deepseek_v4/sm70/indexer.py use info_once with the default scope, so only stage 0 named its decode route. They now pass scope="process", as the prefill route lines of #1280 do. In the same boot (our fork, which has the prefill lines with scope="process" and the decode line without), the prefill route came from all five stages ("fused SM75 scorer on cuda:0", "Triton scorer on cuda:1" ... "fused SM75 scorer on cuda:4") and the decode route from stage 0 alone ("cuBLAS route on cuda:0").

Not a duplicate: I searched the open PRs for "Available KV cache memory", "KV cache memory per worker", info_once, scope="process" and "indexer decode route", and compared the changed files with those of all open PRs. Our own #1276 changes vllm/v1/engine/core.py elsewhere, and our #1280, #1282 and #1286 change other parts of indexer.py (the prefill routes); a merge with each gives no conflict.

AI assistance was used for this change. I reviewed every line and ran the tests below.

## Test Plan

Native build of this branch in place (TORCH_CUDA_ARCH_LIST=7.0), then one pytest process per file:
python -m pytest tests/kernels/test_deepseek_v4_sm70_indexer.py tests/v1/engine/test_engine_core.py

pre-commit run --from-ref origin/main --to-ref HEAD; pre-commit run mypy-3.10 --hook-stage manual --all-files

The log lines were read in a DeepSeek-V4-Flash PP5 boot of our fork (production entry, 2026-10-28).

## Path coverage

| Path / provider | Format | Key conditions (shape, TP/EP, graph, capability) | Expected selection / fallback | Evidence (source / static / executed / measured) | Unverified scope |
| --- | --- | --- | --- | --- | --- |
| EngineCore._initialize_kv_caches | any | KV cache memory determined by the workers (not for attention-free models or a preset KV budget) | one extra info line naming every worker's figure | measured: fork PP5 boot log (tests/v1/engine/test_engine_core.py needs model weights we do not have offline, see Test Result) | multi-node: the line lists every worker of the executor, as determine_available_memory returns them |
| SM70 indexer decode route log | DeepSeek-V4 sparse indexer | any SM70-family stage | one line per process instead of per node | source; measured: the prefill lines with the same scope in the fork PP5 boot | none |

## Acceleration and benchmark contract (required for performance changes)

- Default enabled or opt-in: log lines only; no behaviour change.
- Required CLI options and environment switches: none.
- KV cache dtype used for the benchmark: n/a (no benchmark).
- Wheel SHA or source commit: this branch on main 6aea01fc6 (the tests ran with it on 9a33cee2a; the three later main commits #1290-#1292 touch neither the changed files nor these tests); the log evidence from our fork (github.com/Peuqui/1Cat-vLLM, prod-v5 89ce1e78b), which carries the KV line already.
- PYTHONPATH, source overlays, or external native libraries used (write "none" if absent): none.
- User-entry route-hit, speed, and output-quality evidence: n/a.
- Existing kernel/backend registry and capability-rejection reasons: not touched.
- Per-engine configuration and deprecated-variable compatibility: no setting added.
- New/changed environment metadata and generated reference check: none.
- Route snapshot command and intentional changed rows (write "none" for refactors): none.
- SM70 registered variables / unregistered reads / model-parameter restrictions, before -> after: unchanged.

## Test Result

On a Quadro RTX 8000 and on a Tesla V100: tests/kernels/test_deepseek_v4_sm70_indexer.py 26 passed. tests/v1/engine/test_engine_core.py: 10 failed, 1 skipped on both cards, the same test IDs on main 9a33cee2a with the same build (the tests load llava-hf/llava-1.5-7b-hf, whose weights are not on this machine, so EngineCore is not exercised here; the log line was read in the fork boot above).

pre-commit run --from-ref origin/main --to-ref HEAD: every hook passes except check-layering, which reports vllm/model_executor/layers/fused_moe/runner/moe_runner.py (platform coupling 37 -> 39), as on main 9a33cee2a; this PR does not touch that file. pre-commit run mypy-3.10 --hook-stage manual --all-files: the only errors are three arg-type errors in vllm/v1/attention/backends/flash_v100/runtime.py, which mypy reports on main as well in our environment and 1Cat's CI does not report; this PR does not touch that file.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
