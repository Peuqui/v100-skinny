# PR-Entwurf: 1CatAI/1Cat-vLLM — Fix #414 (MTP profile report never prints under PP)

Stand 2026-09-05. Branch `pp-mtp-profile-report-rank` im Checkout
`~/Projekte/vllm-research/1Cat-vLLM`, Basis origin/main 755baae.
Status: committed als e375cc3, gepusht nach fork Peuqui/1Cat-vLLM (Branch pp-mtp-profile-report-rank). PR eröffnet: https://github.com/1CatAI/1Cat-vLLM/pull/512 Fork-Port in fork_patches_150 deployt 2026-09-05 18:3x, Rig-Nachweis unten.
Diff: `pr-414-mtp-profile-report-rank.diff` (daneben) bzw. `git diff` im Checkout.

Geänderte Dateien (6, 201+/10-):
- vllm/distributed/parallel_state.py — neuer Helfer `is_last_pp_first_tp_rank()`
- vllm/v1/worker/gpu_model_runner.py — Gate im v1-Runner-Report
- vllm/v1/worker/gpu/model_runner.py — Gate im v2-Runner-Report
- vllm/v1/spec_decode/llm_base_proposer.py — Gate im Proposer-Report (totes try/except entfernt)
- tests/v1/spec_decode/test_sm70_mtp_safety.py — Helfer-Tests, v1-Runner- und Proposer-Report-Tests
- tests/v1/worker/test_gpu_model_runner_v2.py — bestehenden Test angepasst, v2-Report-Test ergänzt

Vorgeschlagener PR-Titel:

    [Bugfix][Spec Decode] Report SM70 MTP profiles from the last PP stage (#414)

Vorgeschlagene Commit-Message:

    [Bugfix][Spec Decode] Report SM70 MTP profiles from the last PP stage

    The VLLM_SM70_MTP_PROFILE reports (v1 runner, v2 runner and the
    proposer) collect their timing events on the last pipeline-parallel
    stage -- the sampling path and the drafter only run there -- but gate
    the log line on is_global_first_rank(), which is never on that stage
    when PP > 1. The profile is accumulated every interval and silently
    discarded (#414).

    Add is_last_pp_first_tp_rank() next to is_global_first_rank() and gate
    the three reports on it: the first TP rank of the last PP stage, True
    in a single process. The proposer's try/except around the old check
    was dead code (is_global_first_rank never raises) and goes away.

    Add tests that simulate PP=2 ranks through the parallel_state groups:
    the last-stage leader must report, the first stage and the other TP
    ranks must stay silent.

    Fixes #414

    Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
    Signed-off-by: Peuqui <peuqui@github.com>

---------------------------------------------------------------------------
PR-BODY
---------------------------------------------------------------------------

## Purpose

Fixes #414 (`VLLM_SM70_MTP_PROFILE` report never prints under pipeline
parallelism).

All three SM70 MTP profile reports collect their events on the last PP
stage and log them only from the global first rank:

- `GPUModelRunner._sm70_mtp_profile_report` (v1): the profile context is
  created after sampling in `execute_model`, a path the non-last stages
  leave early with their `IntermediateTensors`; the report then returns on
  `not is_global_first_rank()`.
- `GPUModelRunner._sm70_v2_mtp_profile_report` (v2): profiling is enabled
  only when `self.is_last_pp_rank` (see `_sm70_v2_mtp_profile_enabled`),
  the report is again gated on `is_global_first_rank()`.
- `SpecDecodeBaseProposer._sm70_mtp_profile_report`: the drafter lives on
  the last stage; same gate.

With PP > 1 the global first rank is on the first stage, so the reports
are accumulated every interval and discarded. With PP = 1 they work, which
is why this went unnoticed.

This PR adds `is_last_pp_first_tp_rank()` to `parallel_state` next to
`is_global_first_rank()` (True on the first TP rank of the last PP stage,
True when the model-parallel groups are not initialised) and gates the
three reports on it. The proposer's `try/except RuntimeError` around the
old check was dead code: `is_global_first_rank()` catches everything
itself.

Tests simulate PP=2 through the `parallel_state` group objects rather than
by patching the imported name, so they exercise the real gate: on
origin/main the last-stage leader stays silent and the first stage logs a
report it has no events for.

Duplicate check (per AGENTS.md): `gh issue view 414 --comments` (no
follow-up), `gh pr list --state open --search "414 in:body"` (none),
`--search "profile report"` and `--search "is_global_first_rank"` (no PR
touches these gates). Verified against origin/main 755baae today.

## Test Plan

Environment: 1Cat-vLLM 1.5.0 wheel venv (Python 3.12), checkout at
origin/main 755baae with the compiled extensions linked in, single Tesla
V100 (`CUDA_VISIBLE_DEVICES=4`).

    # both touched test files, with the fix
    python -m pytest tests/v1/spec_decode/test_sm70_mtp_safety.py tests/v1/worker/test_gpu_model_runner_v2.py -q

    # same, with the four source files reverted to origin/main (git stash), tests kept
    python -m pytest tests/v1/spec_decode/test_sm70_mtp_safety.py tests/v1/worker/test_gpu_model_runner_v2.py -q

    # lint / types (mypy 1.19.1 as pinned; ruff 0.15.8 vs pinned 0.14.0)
    ruff check <6 files>
    ruff format --check <6 files>
    mypy --python-version 3.10 vllm/distributed/parallel_state.py vllm/v1/spec_decode/llm_base_proposer.py vllm/v1/worker/gpu/model_runner.py vllm/v1/worker/gpu_model_runner.py
    mypy --python-version 3.10 --follow-imports skip tests/v1/spec_decode/test_sm70_mtp_safety.py tests/v1/worker/test_gpu_model_runner_v2.py

## Test Result

Both test files on origin/main 755baae, before the change:

    10 passed, 2 skipped in 5.55s

Both test files with the fix (14 new test cases):

    24 passed, 2 skipped in 8.92s

Tests kept, source files reverted to origin/main:

    11 failed, 13 passed, 2 skipped
    test_runner_mtp_profile_reports_from_last_pp_stage[pp2-last-stage-leader]   assert False is True  (no report from the stage that has the events)
    test_runner_mtp_profile_reports_from_last_pp_stage[pp2-first-stage]         assert True is False  (report from a stage without events)
    test_proposer_mtp_profile_reports_from_last_pp_stage[pp2-last-stage-leader] assert False is True
    test_proposer_mtp_profile_reports_from_last_pp_stage[pp2-first-stage]       assert True is False
    test_sm70_v2_mtp_profile_reports_from_last_pp_stage[pp2-last-stage-leader]  assert False is True
    test_sm70_v2_mtp_profile_reports_from_last_pp_stage[pp2-first-stage]        assert True is False
    test_is_last_pp_first_tp_rank[*] (5 cases)                                   AttributeError: no attribute 'is_last_pp_first_tp_rank'

ruff check / ruff format --check: clean on all six files.
mypy on both test files: no issues.
mypy on the four source files: 144 errors before and after, identical set
(all pre-existing, none in the touched lines).

End-to-end check on our rig (2x RTX 8000 + 2x V100, TP2 x PP2,
Qwen3.8-27B-NVFP4 with MTP k=5, the same three changes applied to our
1.5.0 deployment, `VLLM_SM70_MTP_PROFILE=1` on the server entry): all four
workers (PP0_TP0, PP0_TP1, PP1_TP0, PP1_TP1) are up; over a ~2000-step chat
both reports come from `Worker_PP1_TP0` only -- 125 runner reports and 125
proposer reports, none from the first stage or from TP rank 1:

    (Worker_PP1_TP0 pid=2217869) INFO 09-05 18:47:29 [gpu_model_runner.py:2212] SM70 spec runner profile avg_ms calls=1984 spec_steps=1977 num_tokens=6 num_reqs=1 target_forward=17.466 target_logits=1.565 target_rejection_sample=1.575 target_sample_no_spec=0.002 state_update_wall_cpu=0.419 state_update_validate_cpu=0.012 state_update_attn_compact_cpu=0.001 state_update_mamba_compact_cpu=0.001 state_update_input_batch_cpu=0.399 state_update_drafter_context_cpu=0.003 draft_total=10.787 draft_wall_cpu=11.394 bookkeeping=0.053 bookkeeping_wall_cpu=0.063
    (Worker_PP1_TP0 pid=2217869) INFO 09-05 18:47:29 [llm_base_proposer.py:873] SM70 MTP proposer profile avg_ms calls=1984 batch=1 tokens=6 total_gpu=9.916 total_wall_cpu=5.328 first_setup_cpu=0.584 first_forward=1.571 first_sample=0.700 loop_metadata_cpu=0.777 loop0_forward=1.010 loop0_sample=0.700 loop1_forward=1.006 loop1_sample=0.700 loop2_forward=1.013 loop2_sample=0.700 loop0_metadata_cpu=0.289

(Line numbers in that log are from our patched deployment, not from this
branch.)

AI assistance: this change was developed with Claude (Anthropic) as a
coding assistant. Every changed line was reviewed by me and the test runs
above were executed on my hardware; I can defend the change end to end.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
