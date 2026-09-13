## Purpose

With `VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH=1`, `VllmConfig` set `VLLM_DISABLE_COMPILE_CACHE=1` unless the user had set the variable, and `envs.disable_compile_cache()` defaulted to `1` on the same condition. The log line explained why: a reloaded AOT artifact had once reproduced deterministic greedy token drift. The cause of that drift was the compile-cache key, which did not include unregistered `VLLM_` switches and therefore reused an artifact compiled for a different kernel path. That was fixed in #536.

With the key fixed, the forced opt-out only hides the cache from users who did not ask for that. This change removes both defaults, so `VLLM_DISABLE_COMPILE_CACHE` means what its documentation says again, and removes `VLLM_SM70_ALLOW_COMPILE_CACHE_FOR_PROFILING`, whose only purpose was to bypass the forced opt-out (its mirror in `benchmarks/benchmark_sm70_decode.py` goes with it). Users who want the old behaviour set `VLLM_DISABLE_COMPILE_CACHE=1` explicitly.

## What the cache does on this graph, measured

Qwen3.8-27B-NVFP4 with DFlash2 n=7, TP2, greedy, 5x400 tokens per boot, `VLLM_DISABLE_COMPILE_CACHE=0` on this branch's parent (main dfef3342 plus our open PRs), 11 boots, output text identical (same SHA-256) in every one:

- Cold boot on an empty cache root: 170-210 s to ready, three AOT artifacts written.
- Warm boot that reloads the artifacts: 70-75 s to ready ("Directly load AOT compilation" for both ranks and all three graphs). Without reload, i.e. with the opt-out, a warm boot is 110-120 s because only Inductor's own cache is warm.
- Card type is part of the artifact key: booting the RTX 8000 pair on the cache root that a V100 pair had just filled produced a different key and compiled its own artifacts; a cold RTX boot on a fresh root produced that same key again. No cross-architecture reuse.
- One caveat on torch 2.10, which requirements/cuda.txt pins: the first warm boot after a cold compile fails to load the fresh artifacts ("Compiling model again due to a load failure ..., reason:" with an empty reason), recompiles and saves again; every later warm boot loads. With `VLLM_FORCE_AOT_LOAD=1` the failure is an `AssertionError` in `torch/_higher_order_ops/triton_kernel_wrap.py:get_kernel` (`assert idx in self.id_to_kernel`) while `AOTAutogradCache` computes the key of the loaded graph: user-defined Triton kernels are referenced by an index into a process-local table that the artifact does not carry. pytorch/pytorch#173556 (merged 2026-01-28, in torch 2.11 and later) serializes that table with the artifact. torch 2.11 dropped Volta from the CUDA 12.8 wheels, so this repository stays on 2.10 and keeps the one wasted warm boot per artifact generation; it costs the same as today's behaviour, nothing more.

## Test Plan

1. `pre-commit run --files benchmarks/benchmark_sm70_decode.py vllm/config/vllm.py vllm/envs.py` and `pre-commit run mypy-3.10 --hook-stage manual --files vllm/config/vllm.py vllm/envs.py`.
2. `grep -rn ALLOW_COMPILE_CACHE_FOR_PROFILING tests/ docs/` to confirm nothing else references the removed switch.
3. The boot matrix above (scripts `cache_devtype_chain.sh`, `cache_reload_chain.sh`, `cache_trace_chain.sh` in our v100-skinny repository; logs available on request).

## Test Result

1. All applicable hooks passed; mypy-3.10 passed.
2. No references outside the removed code.
3. 11 boots as described, text identical throughout, reload verified from the second warm boot on, card type verified in the key.

## Not a duplicate

`gh pr list --state open --search` for "compile cache", "DISABLE_COMPILE_CACHE" and "ALLOW_COMPILE_CACHE_FOR_PROFILING" and `gh issue list --search "compile cache"` return nothing related; no open PR touches these lines of `vllm/config/vllm.py` or `vllm/envs.py`.

AI assistance (Claude) was used to trace the cache path, run the boot matrix and prepare the change; I reviewed every line and ran the checks above.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

