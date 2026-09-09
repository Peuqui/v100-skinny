# PR-Entwurf: 1CatAI/1Cat-vLLM — Qwen4Exp-MTP unter PP bootbar machen

Stand 2026-09-08 nachts. Branch `qwen4exp-mtp-pp-stage-local`, Basis
`origin/main` **4f19ef7**, ein Commit `26a406a`, zwei Dateien, 138+/34−.

Vorgeschlagener Titel:

    [Bugfix][Qwen4Exp] Keep the MTP drafter stage-local under pipeline parallelism

---------------------------------------------------------------------------
PR-BODY
---------------------------------------------------------------------------

## Purpose

Qwen4Exp with `--speculative-config method=mtp` and `--pipeline-parallel-size`
greater than one does not boot at all. Every worker on the final pipeline rank
dies while compiling:

    (Worker_PP1_TP0) ERROR torch._dynamo.exc.Unsupported:
        Data-dependent assertion failed (cannot compile partial graph)
    (Worker_PP1_TP0) ERROR     assert intermediate_tensors is not None
    (Worker_PP1_TP0) ERROR     File ".../vllm/models/qwen4_exp/nvidia/mtp.py",
                                 line 333, in forward

The drafter is stage-local. `gpu_model_runner.execute_model` returns the
IntermediateTensors on every non-final pipeline rank *before* speculation is
reached, so `Qwen4ExpMultiTokenPredictor` only ever runs on the last rank. Its
weights are replicated rather than partitioned: `embed_tokens`, `fc_embedding`,
`fc_hidden` and every MTP layer are built on all ranks.

`forward` nevertheless branched on `get_pp_group().is_first_rank` — the
**target** model's pipeline position. On the last rank that is False, so the
drafter took the "receive from the previous stage" path and asserted on
intermediate tensors that nobody sends. Under the fullgraph AOT compile this is
a compile error rather than a runtime one, so it takes the whole boot down
instead of a single step.

The fix drops both pipeline branches in that module: always build the embedding
locally, and never return IntermediateTensors for a next stage that does not
exist for this module. Single-rank behaviour is unchanged — there
`is_first_rank` and `is_last_rank` are both True and the removed branches were
already dead code.

## Results

2x Quadro RTX 8000 (sm75) + 2x Tesla V100 (sm70), `CUDA_VISIBLE_DEVICES=0,2,1,3`,
Qwen3.8-Flash-Next-180B-A4B-NVFP4 with a quantized MTP block, TP2 x PP2,
partition 24,24, k=4, `--max-model-len 16384`.

| | before | after |
|---|---|---|
| boot | dies on both final-rank workers, quoted above | engine up, serves |
| long-context decode | not reachable | 25.2 – 28.0 tok/s |
| acceptance length | not reachable | 2.70 – 3.23 |

The "after" row is three questions of 30 sentences each behind 13004 tokens of
unrelated context, 1200 output tokens apiece, greedy with a fixed seed. All
three answers are complete and were read; the deliberately misspelled probe
term is recognized as a misspelling rather than hallucinated into a new
phenomenon.

## Test Plan

Environment: checkout at `origin/main` 4f19ef7 with the compiled extensions of
a 1Cat-vLLM 1.5.0 wheel linked in (see Limitations).

    python -m pytest tests/models/qwen4_exp/test_mtp_stage_local.py -q

    # complete affected directory; the AMD test cannot be collected on an
    # NVIDIA box, so it is excluded and run separately
    python -m pytest tests/models/qwen4_exp/ -q \
        --ignore=tests/models/qwen4_exp/test_qsa_amd.py
    python -m pytest tests/models/qwen4_exp/test_qsa_amd.py -q

    # counter-check: mtp.py reverted to origin/main, tests kept
    python -m pytest tests/models/qwen4_exp/test_mtp_stage_local.py -q

    pre-commit run --files vllm/models/qwen4_exp/nvidia/mtp.py \
        tests/models/qwen4_exp/test_mtp_stage_local.py
    pre-commit run mypy-3.10 --hook-stage manual --files <same two files>

    # duplicate-work checks
    gh pr list --repo 1CatAI/1Cat-vLLM --state open --search "<keywords>"

## Test Result

New file: **2 passed**.

Directory without the AMD test: **280 passed, 10 failed, 3 skipped**.
`tests/models/qwen4_exp/test_qsa_amd.py` alone: **10 skipped**.

The failures are all `test_sm70_gdn_projection_split.py::
test_cuda_public_op_graph_replay_is_bitwise`, and they are **pre-existing and
unrelated**: the same file on the unmodified tree gives the identical
`7 failed, 28 passed`, before and after this change. They assert that a fused
public op route is taken (`assert route_hits == [rows, rows]`, actual `[]`),
which our bench cannot satisfy because it runs main's Python against the
compiled ops of a 1.5.0 wheel.

Counter-check with `vllm/models/qwen4_exp/nvidia/mtp.py` reverted to
`origin/main`, both tests kept:

    2 failed
    test_drafter_embeds_on_the_last_pipeline_rank
    test_drafter_does_not_hand_off_to_a_next_stage

pre-commit over both files: ruff check, ruff format, typos, mypy-local, SPDX
headers, root lazy imports, forbidden imports, torch.cuda-call check,
config-docstring check, attention-backend docs and the boolean-ops check all
Passed. `pre-commit run mypy-3.10 --hook-stage manual`: Passed.

## Not a duplicate

Checked on 2026-09-08 against every open PR. One other open PR touches
`vllm/models/qwen4_exp/nvidia/mtp.py`: **#553** ("NVIDIA Qwen3.8 Flash-Next MTP
loader"). Its hunks in that file are at lines 22, 45, 67, 138, 152, 166, 209
and 493; this change is at 309–354. `is_first_rank`, `is_last_rank` and
`intermediate_tensors` do not occur in #553 at all. No open PR or issue
mentions pipeline parallelism together with the Qwen4Exp drafter.

## Limitations

The unit tests and linting run against this tree. The boot and throughput
numbers come from a 1Cat-vLLM 1.5.0 wheel deployment carrying the same change,
because building current main from source on this hardware is a multi-hour CUDA
build. For the code that carries the measurement the two trees differ only in
this fix: `vllm/models/qwen4_exp/nvidia/mtp.py` on `origin/main` and the
deployed file are otherwise identical, and the "before" run above was produced
by putting main's file into that deployment unchanged.

AI assistance: this change was developed with Claude (Anthropic) as a coding
assistant. Every changed line was reviewed by me and the test runs above were
executed on my hardware; I can defend the change end to end.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
