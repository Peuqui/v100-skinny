# PR-Entwurf: 1CatAI/1Cat-vLLM — Fix #439 (drafter guard on non-last PP ranks)

Stand 2026-09-05. Branch `pp-spec-drafter-guard-v2` im Checkout
`~/Projekte/vllm-research/1Cat-vLLM`, Basis origin/main 2b89b77
(origin/main ist inzwischen 7e3efe7; die neuen Commits berühren weder
`gpu_model_runner.py` noch die Testdatei, Merge-Probe konfliktfrei).
Status 2026-09-05: committed als 89aa181, gepusht nach fork Peuqui/1Cat-vLLM (Branch pp-spec-drafter-guard-v2). PR eröffnet: https://github.com/1CatAI/1Cat-vLLM/pull/511 Diff: `pr-439-drafter-guard.diff`
(daneben abgelegt) bzw. `git diff HEAD` im Checkout.

Vorgeschlagener PR-Titel:

    [Bugfix][Spec Decode] Bind self.drafter on non-last PP ranks (#439)

Vorgeschlagene Commit-Message (Autor Peuqui, Trailer wie bei #485):

    [Bugfix][Spec Decode] Bind self.drafter on non-last PP ranks

    The drafter is only built on the last pipeline-parallel rank, but four
    call sites that run on every rank assert on it -- memory profiling
    (_dummy_run -> _build_attention_metadata), initialize_metadata_builders,
    _check_and_update_cudagraph_mode and initialize_kv_cache. With
    speculative decoding and pipeline_parallel_size > 1 the non-last ranks
    die with AttributeError before the first request (#439).

    Bind self.drafter = None on those ranks and gate the drafter-only
    initialisation on get_pp_group().is_last_rank. load_model() and
    _sample() move to the getattr(self, "drafter", None) idiom the file
    already uses elsewhere; the latter also repairs the pre-existing
    failure of test_sample_passes_reordered_draft_probs_to_rejection_sampler.

    Add a regression test that boots a rank-0 runner with ngram,
    draft_model and extract_hidden_states speculative configs and runs
    initialize_kv_cache() plus a profiling _dummy_run() through
    _build_attention_metadata.

    Fixes #439

    Co-authored-by: Claude <noreply@anthropic.com>
    Signed-off-by: Peuqui <peuqui@github.com>

---------------------------------------------------------------------------
PR-BODY (Template des Repos: Purpose / Test Plan / Test Result)
---------------------------------------------------------------------------

## Purpose

Fixes #439 (`'GPUModelRunner' object has no attribute 'drafter'` with
speculative decoding and pipeline_parallel_size > 1).

`GPUModelRunner.__init__` only builds `self.drafter` under
`if self.speculative_config and get_pp_group().is_last_rank:` -- the draft
model lives on the last PP rank. Four code paths that run on every rank
nevertheless assert on the attribute, guarded only by
`self.speculative_config`:

- `_dummy_run` -> `_build_attention_metadata` (memory profiling, the frame
  in the reported traceback),
- `initialize_metadata_builders` (eagle / draft_model),
- `_check_and_update_cudagraph_mode` (eagle / extract_hidden_states),
- `initialize_kv_cache` (extract_hidden_states).

On PP rank 0 the attribute does not exist, so the worker dies during
`determine_available_memory()` before any request arrives.

This PR

- binds `self.drafter = None` on non-last ranks (the `isinstance` probes
  in `_build_attention_metadata` then fall through as intended),
- gates the three drafter-only initialisation blocks on
  `get_pp_group().is_last_rank`, next to the existing `speculative_config`
  check,
- switches `load_model()` and `_sample()` to the
  `getattr(self, "drafter", None)` idiom the file already uses in five
  other places (`_sample` previously did `hasattr(self.drafter, ...)`,
  which raised on a runner without the attribute -- that is also why
  `test_sample_passes_reordered_draft_probs_to_rejection_sampler` fails on
  current main, see Test Result),
- adds a regression test that boots a rank-0 runner with `ngram`,
  `draft_model` and `extract_hidden_states` speculative configs and runs
  the real `initialize_kv_cache()` plus a profiling `_dummy_run()` with
  `force_attention=True`, i.e. through `_build_attention_metadata`.

Scope: this only makes the non-last ranks survive initialisation and
profiling. It does not add the spec-state transport (sampled token matrix,
accepted counts, hybrid-state update) that non-last ranks need for
speculative decoding under PP to be correct end to end; that is the larger
change discussed in the #439 thread and is intentionally kept out of this
PR. eagle/mtp variants are not in the parametrisation because opt-125m
carries no draft head; the eagle branches share the guarded sites with the
draft_model / extract_hidden_states variants.

Duplicate check (per AGENTS.md): `gh issue view 439 --comments`,
`gh pr list --state open --search "439 in:body"` (no results),
`gh pr list --state open --search "drafter"` and `"pipeline parallel
speculative"` (no PR touches these sites). Verified against origin/main
7e3efe7 today.

## Test Plan

Environment: 1Cat-vLLM 1.5.0 wheel venv (Python 3.12, torch from the
wheel), checkout at origin/main 2b89b77 with the compiled extensions
linked in, single Tesla V100 (`CUDA_VISIBLE_DEVICES=4`), facebook/opt-125m
from the HF cache.

    # regression test, with the fix
    python -m pytest tests/v1/worker/test_gpu_model_runner.py -k non_last_pp_rank -q

    # regression test, fix reverted (git stash of gpu_model_runner.py only)
    python -m pytest tests/v1/worker/test_gpu_model_runner.py -k non_last_pp_rank -q

    # whole file, before and after
    python -m pytest tests/v1/worker/test_gpu_model_runner.py -q

    # lint / types (same versions as .pre-commit-config.yaml: mypy 1.19.1; ruff 0.15.8 vs pinned 0.14.0)
    ruff check vllm/v1/worker/gpu_model_runner.py tests/v1/worker/test_gpu_model_runner.py
    ruff format --check vllm/v1/worker/gpu_model_runner.py tests/v1/worker/test_gpu_model_runner.py
    mypy --python-version 3.10 vllm/v1/worker/gpu_model_runner.py
    mypy --python-version 3.10 --follow-imports skip tests/v1/worker/test_gpu_model_runner.py

## Test Result

Regression test with the fix:

    3 passed, 38 deselected in 12.35s

Regression test with the fix reverted (runner file at origin/main, test
kept) -- each variant dies on one of the four sites:

    test_non_last_pp_rank_profiles_with_speculative_config[ngram]
      vllm/v1/worker/gpu_model_runner.py:11000: in _dummy_run
      vllm/v1/worker/gpu_model_runner.py:5672: in _build_attention_metadata
      AttributeError: 'GPUModelRunner' object has no attribute 'drafter'
    test_non_last_pp_rank_profiles_with_speculative_config[draft_model]
      vllm/v1/worker/gpu_model_runner.py:12674: in initialize_kv_cache
      vllm/v1/worker/gpu_model_runner.py:12166: in initialize_metadata_builders
      AttributeError: 'GPUModelRunner' object has no attribute 'drafter'
    test_non_last_pp_rank_profiles_with_speculative_config[extract_hidden_states]
      vllm/v1/worker/gpu_model_runner.py:12659: in initialize_kv_cache
      vllm/v1/worker/gpu_model_runner.py:12125: in initialize_attn_backend
      vllm/v1/worker/gpu_model_runner.py:12219: in _check_and_update_cudagraph_mode
      AttributeError: 'GPUModelRunner' object has no attribute 'drafter'
    3 failed, 38 deselected in 11.93s

Whole file `tests/v1/worker/test_gpu_model_runner.py`:

    origin/main 2b89b77:  1 failed, 38 passed, 2 skipped
      FAILED test_sample_passes_reordered_draft_probs_to_rejection_sampler
        vllm/v1/worker/gpu_model_runner.py:7668: in _sample
        AttributeError: 'GPUModelRunner' object has no attribute 'drafter'
    this branch:          39 passed, 2 skipped in 40.92s

ruff check / ruff format --check: clean on both files.
mypy on `tests/v1/worker/test_gpu_model_runner.py`: no issues.
mypy on `vllm/v1/worker/gpu_model_runner.py`: 61 errors before and after,
identical set (all pre-existing, none in the touched lines).

AI assistance: this change was developed with Claude (Anthropic) as a
coding assistant. Every changed line was reviewed by me and the test runs
above were executed on my hardware; I can defend the change end to end.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
