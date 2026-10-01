Title: [Bugfix][PP] Check the token ids the last pipeline stage hands over

## Purpose

Under pipeline parallelism with speculative decoding, the last stage
broadcasts two payloads to the other stages over the gloo CPU group (#662):

- the sampled token matrix;
- the draft token ids.

The first stage embeds them. An id outside the vocabulary therefore surfaces
on a different rank than the one that produced it, as an anonymous
"device-side assert triggered" in the embedding lookup. It takes the engine
down without naming the request or the cause.

Both payloads already travel through host memory, so this PR checks them
there:

- on the last stage before sending, where the cause can still be named: the
  non-finite values per sampler input row, the sampling mode, the
  temperature, the computed tokens, and the step's input ids and positions;
- on arrival, on every other stage.

A discarded request's sampled token is never read, so it is skipped; draft
ids are always checked. The check is a range comparison on tensors that are
copied to the host anyway. The diagnostic text is only built when an id is
out of range.

This is how we found the `vocab_size` that `sample_recovered_tokens` returns
for an all-NaN logits row on DeepSeek-V4 (the cause behind #658, #715 and
#723). It has run in our fork on a five-stage DeepSeek-V4 + DSpark pipeline
since 2026-09-19. Our logs go back to 2026-09-21, and it has not fired since
then.

Why this is not a duplicate. #625 (merged) stops at the first out-of-range
id in `InputBatch.update_async_output_token_ids()` on the rank that sampled.
That is a different path: the PP broadcast to the non-last ranks has no
check. `gh pr list --state all --search` for "pipeline token ids", "PP spec
decode", "invalid token ids", "vocab_size token" and "device-side assert
embedding" finds no other PR that touches `_pp_check_token_ids` or the
broadcast sites in `gpu_model_runner.py`.

AI assistance was used for this change. Every line was reviewed and the tests
below were run by the submitter.

## Test Plan

```bash
pytest tests/v1/worker/test_pp_token_check.py
pytest tests/v1/worker/test_gpu_model_runner.py -k "pp or spec or pipeline"
pre-commit run --files <changed files>
```

## Test Result

The new test covers:

- valid ids pass;
- an id of `vocab_size`, or a negative one, raises an error naming the
  request;
- a discarded request's sampled id is skipped;
- draft ids of a discarded request are still checked;
- with the sampler input, the error lists the non-finite values per row.

| | Result |
|---|---|
| test_pp_token_check.py | 6 passed |
| test_gpu_model_runner.py -k "pp or spec or pipeline" (Tesla V100) | 11 passed |

pre-commit clean.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
