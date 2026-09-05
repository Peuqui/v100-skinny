# PR-Entwurf: 1CatAI/1Cat-vLLM — #479 Scheibe 2: PLE-Gate folgt der Partition, nicht der PP-Größe

Stand 2026-09-05. Branch `pp-ple-partition-gate`, Basis origin/main 755baae.
Status: committed als f494c9b, gepusht nach fork Peuqui/1Cat-vLLM (Branch pp-ple-partition-gate). PR eröffnet: https://github.com/1CatAI/1Cat-vLLM/pull/516
Diff: `pr-479-ple-partition-gate.diff` (daneben) bzw. `git diff` im Checkout.

Geänderte Dateien (8):
- vllm/distributed/utils.py — `get_layers_outside_first_pp_rank()` neben `get_pp_indices()`
- vllm/models/qwen4_exp/common/ple.py — `check_ple_layers_on_first_pp_rank()` (SSOT für die drei Gates)
- vllm/v1/worker/gpu_model_runner.py — Runner-Gate umgestellt (lazy import)
- vllm/models/qwen4_exp/nvidia/model_state.py, .../amd/model_state.py — Model-State-Gates umgestellt
- tests/distributed/test_pipeline_partition.py — 6 Helfer-Fälle
- tests/models/qwen4_exp/test_ple.py — 7 Gate-Fälle (inkl. Off-by-one der 1-basierten ple_layer_ids)
- tests/v1/worker/test_gpu_model_runner_pp.py — 2 Runner-Fälle (echte GPUModelRunner-Konstruktion, PP=2)

Scope: Nur das PP-Verbot der PLE-Embedding (Gate 2 aus #479). Gate 1 (VLLM_PLE_CPU_OFFLOAD
verweigert PP im Worker) bleibt, weil der Offload-Kindprozess unter PP zusätzlich die
Partitionsvererbung braucht (Fork: ple_offload_worker.py) — das ist die nächste Scheibe.

Wesentliche Erkenntnis beim Bauen: `ple_layer_ids` sind 1-basiert — id L hängt das PLE-Modul
an Decoder-Layer L-1 (`Qwen4ExpDecoderLayer`: `if (self.layer_idx + 1) in ple_layer_ids`).
Die Prüfung rechnet deshalb mit L-1; der Fork prüfte L (eins zu streng). Test
`pp2-custom-split-off-by-one` deckt genau das ab.

Vorgeschlagener PR-Titel:

    [Core][Qwen4Exp] Gate PLE under pipeline parallelism on the partition, not the PP size (#479)

Vorgeschlagene Commit-Message:

    [Core][Qwen4Exp] Gate PLE under PP on the partition, not the PP size

    The n-gram PLE embedding refuses pipeline_parallel_size > 1 outright, in
    GPUModelRunner and in both Qwen4ExpModelState trees. The inputs a PLE
    layer needs (ngram_context, query_start_loc, input_ids) are prepared on
    the first pipeline rank only, and the model forward passes them along
    the layer loop, so the actual requirement is narrower: every decoder
    layer that carries a PLE module has to sit on the first rank. Which
    layers those are is decided by the pipeline partition (#479, gate 2).

    Add get_layers_outside_first_pp_rank() next to get_pp_indices() and a
    single check_ple_layers_on_first_pp_rank() used by all three gates.
    ple_layer_ids are 1-based (id L attaches the PLE module to decoder
    layer L-1), so the check maps them before comparing against the first
    rank's range; VLLM_PP_LAYER_PARTITION is honoured through get_pp_indices.

    Tests cover the partition helper, the check (even and custom splits,
    the off-by-one), and a real GPUModelRunner construction at PP=2 that
    now succeeds with the PLE layer on rank 0 and still refuses one on the
    second stage.

    Refs #479

    Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
    Signed-off-by: Peuqui <peuqui@github.com>

---------------------------------------------------------------------------
PR-BODY
---------------------------------------------------------------------------

## Purpose

Second slice of the PP enablement offered in #479 (the loader fix was #485).

The n-gram PLE embedding currently refuses `pipeline_parallel_size > 1`
outright, at three places: `GPUModelRunner.__init__` and both
`Qwen4ExpModelState` trees (nvidia, amd). The stated reason is right --
`ngram_context`, `query_start_loc` and the raw `input_ids` are only prepared
on the first pipeline rank (`gpu_model_runner.py`, `not is_first_rank` guard
in the PLE input preparation) -- but the conclusion is wider than needed. The
model forward passes those inputs down the layer loop and only a decoder
layer that owns a PLE module consumes them (`Qwen4ExpDecoderLayer.forward`:
`if self.ple is not None`). So PP works as long as every decoder layer with a
PLE module sits on the first rank, and that is decided by the pipeline
partition, not by the pipeline size.

This PR

- adds `get_layers_outside_first_pp_rank()` in `vllm/distributed/utils.py`
  next to `get_pp_indices()`, so the check honours `VLLM_PP_LAYER_PARTITION`
  exactly like the model does,
- adds `check_ple_layers_on_first_pp_rank()` in
  `vllm/models/qwen4_exp/common/ple.py` and uses it at all three gates
  (one rule, one message),
- maps `ple_layer_ids` correctly: they are 1-based -- id `L` attaches the PLE
  module to decoder layer `L - 1` (`if (self.layer_idx + 1) in ple_layer_ids`
  in `Qwen4ExpDecoderLayer.__init__`) -- so the check compares `L - 1`
  against the first rank's layer range. A `1,47` split of a 48-layer model
  with `ple_layer_ids=[2]` is therefore rejected (decoder layer 1 is on the
  second stage), while `2,46` passes.

With `ple_layer_ids=[2]` (Qwen3.8-Flash-Next) the default even split keeps
the PLE layer on rank 0 for any `pipeline_parallel_size`, so the common case
needs no `VLLM_PP_LAYER_PARTITION` at all.

Not in this PR: gate 1 of #479 (`VLLM_PLE_CPU_OFFLOAD` refuses PP in
`Worker._validate_ple_offload_config`). Lifting it needs the offload child
process to inherit the partition as well; that is a separate, reviewable
step. `PP=N` therefore still shows up as unsupported for the offload path.

Duplicate check (per AGENTS.md): `gh issue view 479 --comments` (no reply
since 2026-09-03), `gh pr list --state open --search "479 in:body"` (none),
`--search "PLE pipeline"` and `--search "ple_layer_ids"` (none). Verified
against origin/main 755baae today.

## Test Plan

Environment: 1Cat-vLLM 1.5.0 wheel venv (Python 3.12), checkout at
origin/main 755baae with the compiled extensions linked in, single Tesla
V100 (`CUDA_VISIBLE_DEVICES=4`), facebook/opt-125m from the HF cache for the
runner construction.

    # the three touched test files plus the PLE offload worker tests
    python -m pytest tests/distributed/test_pipeline_partition.py tests/models/qwen4_exp/test_ple.py \
        tests/v1/worker/test_gpu_model_runner_pp.py tests/v1/worker/test_ple_offload_worker.py -q

    # same, with the five source files reverted to origin/main (git stash), tests kept
    python -m pytest <same four files> -q

    # pre-commit hooks as configured (ruff, typos, forbidden imports, mypy-local) plus the CI-only mypy stage
    pre-commit run --files <8 files>
    pre-commit run mypy-3.10 --hook-stage manual --files <8 files>
    mypy --python-version 3.10 <5 source files>   # local, with the wheel's site-packages

## Test Result

Four test files with the fix (15 new cases):

    86 passed, 1 skipped in 29.59s

Same four files, source files reverted to origin/main, tests kept:

    15 failed, 71 passed, 1 skipped
    all 71 pre-existing tests pass on origin/main; the 15 failures are the new cases:
      helper / check: ImportError (functions do not exist)
      test_runner_accepts_pp2_with_ple_layers_on_first_rank:
        RuntimeError: N-gram PLE embedding requires pipeline_parallel_size=1   <- the old gate
      test_runner_rejects_pp2_with_ple_layer_on_second_rank:
        message mismatch (old gate refuses without naming the stage)

pre-commit: ruff-check, ruff-format, typos, SPDX headers, forbidden imports,
torch.cuda-call check, mypy-local and mypy-3.10 (manual stage) all Passed.
Local mypy on the five source files: 69 errors before and after, identical
set (all pre-existing in gpu_model_runner.py / parallel utils, none in the
touched lines); the three new/changed test files: no issues.

Production context: our fork has run this rule (first-rank check instead of
the PP ban) for Qwen3.8-Flash-Next at TP2/PP2 with MTP since 1.3.0, on
2x RTX 8000 + 2x V100. The fork variant compared the 1-based ids directly,
i.e. was one layer stricter; this PR uses the exact mapping.

AI assistance: this change was developed with Claude (Anthropic) as a
coding assistant. Every changed line was reviewed by me and the test runs
above were executed on my hardware; I can defend the change end to end.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
