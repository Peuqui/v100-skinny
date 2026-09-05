# PR-Entwurf: 1CatAI/1Cat-vLLM — KV-Budget misst den warmen Forward, nicht den kalten torch.compile

Stand 2026-09-05. Branch `memory-profile-warm-compile`, Basis origin/main 755baae.
Status: committed (Branch memory-profile-warm-compile, auf fork Peuqui/1Cat-vLLM gepusht), PR NICHT eröffnet. Fork-Port in fork_patches_150/gpu_worker.py deployt 2026-09-05 ~20:26.
Kein eigenes Issue — Befund aus unserer Kalibration (RTX 8000 cappte 94k -> 52k Kontext).

Geänderte Dateien (2):
- vllm/v1/worker/gpu_worker.py — Aufwärmlauf vor `memory_profiling`, `warmup_torch_residual` in `non_kv_cache_memory`
- tests/v1/worker/test_gpu_worker_memory_profile.py — NEU, Fake-Runner (1 GiB Scratch + 64 MiB Rest beim ersten Lauf)

Vorgeschlagener PR-Titel:

    [Core] Profile the KV budget on a warm forward, not on the cold torch.compile

Vorgeschlagene Commit-Message:

    [Core] Profile the KV budget on a warm forward, not on the cold compile

    determine_available_memory() measures the torch peak and the non-torch
    increase across profile_run(). The first forward also triggers
    torch.compile, and a cold compile (cache miss) allocates inductor
    autotuning scratch and AOT export buffers that have nothing to do with
    steady-state serving. Measured on Qwen3.8-27B-NVFP4 on a 48 GB RTX
    8000: torch peak 2.59 GiB cold vs 0.74 GiB warm, non-torch 0.86 vs
    0.22 GiB, so the KV budget came out at 3.53 GiB with a cold cache and
    6.03 GiB with a warm one. The same server therefore capped
    max-model-len differently from boot to boot (94k -> 93k -> 52k tokens)
    while the card could hold 90k.

    Run profile_run() once before the measured pass so the compiled graphs
    are in place, and charge what the warm-up left allocated: non-torch is
    already measured against the init snapshot, torch memory the warm-up
    kept beyond the weights is added as warmup_torch_residual. The budget
    can only get smaller from that term, never larger.

    Test drives determine_available_memory() with a fake runner whose
    first profile_run allocates 1 GiB of scratch and keeps 64 MiB: the peak
    is the second forward's 256 MiB and the 64 MiB are still charged.

    Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
    Signed-off-by: Peuqui <peuqui@github.com>

---------------------------------------------------------------------------
PR-BODY
---------------------------------------------------------------------------

## Purpose

`Worker.determine_available_memory()` derives the KV budget from the torch
peak and the non-torch increase observed across `profile_run()`. That first
forward is also where torch.compile happens, and a cold compile (compile
cache miss) is not steady-state serving: inductor autotuning and the AOT
export allocate scratch, and the compile leaves memory behind that a warm
boot never touches. The measurement therefore depends on the state of the
compile cache.

Measured on our rig (Qwen3.8-27B-NVFP4, TP=1 on a Quadro RTX 8000 48 GB,
`--gpu-memory-utilization 0.98`, DEBUG log of the profiling result):

    cold compile (cache miss, 74 s):  torch peak +2.59 GiB, non-torch +0.86 GiB -> KV 3.53 GiB
    warm compile (cache hit,  2.8 s): torch peak +0.74 GiB, non-torch +0.22 GiB -> KV 6.03 GiB

Every new `--max-model-len` is a new compile-cache key, so a boot sequence
that follows vLLM's own "max seq len larger than KV cache" hints walks
into a cold compile and caps far too low: 262144 -> 94080 (warm) -> 93296
(warm) -> 52528 (cold, 3.53 GiB) although a warm boot at 52528 then
reports room for 90,664 tokens. On Tesla V100 the effect does not show
(7.61 GiB cold and warm), which is why it went unnoticed on the SM70 path.

This PR runs `profile_run()` once before the measured pass, so the compiled
graphs are in place when the peak is taken, and charges what the warm-up
left allocated:

- non-torch memory is already measured against the init snapshot, so
  anything the compile keeps (worker contexts, handles) still counts;
- torch memory the warm-up kept beyond the weights is added explicitly as
  `warmup_torch_residual`. On the RTX that term is 0.43 GiB, which the old
  code silently handed to the KV cache.

The budget can only shrink from the residual term, never grow. The cost is
one extra dummy forward (seconds), and only the compile is no longer part
of the measurement.

After the change, same card, same model, new compile-cache key:

    cold compile: torch peak +0.73 GiB, non-torch +0.86 GiB, residual 0.5 GiB -> KV 4.96 GiB
    warm compile: torch peak +0.73 GiB, non-torch +0.22 GiB, residual 0.43 GiB -> KV 5.60 GiB

The remaining cold/warm difference is the non-torch memory a cold compile
keeps allocated for the lifetime of the process; it is real and stays
charged.

Duplicate check (per AGENTS.md): `gh pr list --state open --search
"determine_available_memory"`, `--search "memory profiling"`,
`--search "profile_run"` (none). Verified against origin/main 755baae today.

## Test Plan

Environment: 1Cat-vLLM 1.5.0 wheel venv (Python 3.12), checkout at
origin/main 755baae with the compiled extensions linked in, single Tesla
V100 (`CUDA_VISIBLE_DEVICES=4`) for the unit test; the end-to-end numbers
above from our 1.5.0 deployment with the same two hunks applied.

    python -m pytest tests/v1/worker/test_gpu_worker_memory_profile.py tests/v1/worker/test_gpu_worker_static_pp.py -q
    # new test with vllm/v1/worker/gpu_worker.py reverted to origin/main (git stash)
    python -m pytest tests/v1/worker/test_gpu_worker_memory_profile.py -q
    pre-commit run --files vllm/v1/worker/gpu_worker.py tests/v1/worker/test_gpu_worker_memory_profile.py
    pre-commit run mypy-3.10 --hook-stage manual --files <same two files>
    mypy --python-version 3.10 vllm/v1/worker/gpu_worker.py   # local, with the wheel's site-packages

## Test Result

With the fix (new test plus the existing worker tests in the same process):

    5 passed

New test with `gpu_worker.py` reverted to origin/main:

    1 failed: peak_activation_memory == 1088 MiB (the 1 GiB scratch of the
    first forward), expected 256 MiB

pre-commit: ruff-check, ruff-format, typos, SPDX headers, forbidden
imports, torch.cuda-call check, mypy-local and mypy-3.10 (manual stage)
all Passed. Local mypy on `gpu_worker.py`: 9 errors before and after,
identical set (pre-existing, none in the touched lines).

AI assistance: this change was developed with Claude (Anthropic) as a
coding assistant. Every changed line was reviewed by me and the test runs
above were executed on my hardware; I can defend the change end to end.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
