Title: [Bugfix][Spec Decode] Trim optimistic spec-decode tokens on every PP rank

## Purpose

Fixes #54260.

With async scheduling and speculative decoding, `_update_states` assumes every
draft token of the previous step was accepted and appends that many `-1`
placeholders to `req_state.output_token_ids`
(`req_state.output_token_ids.extend([-1] * optimistic_num_accepted)`). That
extend is not gated on the pipeline rank: `prev_num_draft_len` is set from the
scheduler output in `InputBatch.update_req_spec_token_ids`, which every rank
runs. The trim that removes the placeholders is the `elif` of
`if not is_last_rank`, so it only runs on the last rank, and the deferred
correction (`correct_spec_decode_token_counts`) adjusts `num_computed_tokens`,
not the placeholders. On every non-last rank `output_token_ids` therefore grows
by the draft length each step and never shrinks.

The change turns the `elif` into an `if`. It only fires when
`output_token_ids` is longer than `num_output_tokens`, so the last rank and
single-rank runs take exactly the path they took before, and the model output
is unchanged: sampling happens on the last rank only.

On our deployment (a fork of this code path, TP2 x PP2 with MTP) the overshoot
was not cosmetic: the inflated length fed the discard/chunked-prefill
bookkeeping, the discard mask flipped on the non-last rank only, the PP
broadcast guard then disagreed between ranks and the run wedged in NCCL. The
unit test below is the reduced, deterministic form of that.

**Why this is not a duplicate.** #54270 proposed a fix for this issue and was
closed without review. The checks from AGENTS.md
(`gh issue view 54260 --comments`, `gh pr list --state open --search "54260 in:body"`,
and searches for "non-last rank spec decode", "optimistic spec tokens
pipeline", "trim output_token_ids", "gpu_model_runner output_token_ids") find
no open PR for it. The open PRs that add speculative decoding under pipeline
parallelism (#44698, #45985, #46003, #57917, #57171, #56957) do not touch these
lines. In #54260, sohom-cs confirmed the analysis on eb0f2ca37 and offered to
open this PR; I am opening it as the reporter.

AI assistance (Claude) was used for this change. I reviewed every changed line
and ran the tests below.

## Test Plan

Environment: `uv venv --python 3.12`, `VLLM_USE_PRECOMPILED=1 uv pip install -e .`
with the precompiled wheel of 44af287eb (the newest main commit with a nightly
wheel; the one commit between it and this branch's base 0376f8153 does not
touch the changed files), `uv pip install -r requirements/test/cuda.in`,
torch 2.13.0+cu130, Quadro RTX 8000.

```bash
.venv/bin/python -m pytest tests/v1/worker/test_gpu_model_runner.py -k trims_optimistic -q
.venv/bin/python -m pytest tests/v1/worker/test_gpu_model_runner.py -q
# counter-check: vllm/v1/worker/gpu_model_runner.py reverted to main, test kept
.venv/bin/python -m pytest tests/v1/worker/test_gpu_model_runner.py -q
pre-commit run --files vllm/v1/worker/gpu_model_runner.py tests/v1/worker/test_gpu_model_runner.py
pre-commit run mypy-3.12 --hook-stage manual --files <same two files>
```

## Test Result

New test alone: 1 passed.

`tests/v1/worker/test_gpu_model_runner.py`: 49 passed, 2 failed. With
`gpu_model_runner.py` reverted to main and the new test kept: 48 passed,
3 failed; the extra failure is the new test:

```
assert [111, -1, -1, -1] == [111]
```

The two failures present on both trees,
`test_hybrid_attention_mamba_tensor_shapes` and
`test_mamba_cache_raises_when_max_num_seqs_exceeds_blocks`, select FlashInfer
and stop with "Selected backend AttentionBackendEnum.FLASHINFER is not valid
for this configuration. Reason: ['compute capability not supported']" on this
Turing card; they are unrelated. The existing PP consistency tests
`test_update_states_pp_non_async_multi_request_keeps_token_buffers_consistent`
and `test_update_states_pp_async_multi_request_keeps_rank_state_consistent`
pass with the change.

pre-commit: all hooks passed; mypy-3.12: passed.

Limitations: I have not run current main end to end with pipeline parallelism
and speculative decoding; our GPUs are Volta and Turing and the cu130 torch
wheels no longer support Volta. No model evaluation is attached: the change
does not alter the last rank's path, which is where tokens are sampled.

---
<details>
<summary> Essential Elements of an Effective PR Description Checklist </summary>

- [x] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [x] The test plan, such as providing test command.
- [x] The test results, such as pasting the results comparison before and after, or e2e results
- [ ] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model.
</details>
