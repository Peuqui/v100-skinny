Title: [Bugfix][Spec Decode] Assert enough query rows for each request's drafts

## Purpose

`_calc_spec_decode_metadata` places a request's sampled rows at
`[cumulative_end - (draft_len + 1), cumulative_end)`. If a request is
scheduled with fewer query rows than `draft_len + 1`, nothing fails.
Instead, the verifier silently reads rows that belong to the preceding
request and accepts or rejects tokens against the wrong hidden states.

This PR asserts the invariant in `_prepare_inputs`, where the draft lengths
are collected. A scheduler/runner mismatch then fails at its source instead
of producing wrong tokens. The cost is one integer comparison per request
with drafts.

It has been in our fork since 2026-09-06 on DeepSeek-V4 + DSpark (PP5),
Qwen3.8-Flash-Next + MTP (PP4 and TP2 x PP2) and Qwen3.8-27B + DFlash2. Our logs go back to 2026-09-21, and it has not
fired since then.

Why this is not a duplicate. `gh pr list --state all --search` for
"draft_len", "spec decode metadata" and "num_scheduled_tokens draft" finds
no PR that guards this placement.

AI assistance was used for this change. Every line was reviewed and the tests
below were run by the submitter.

## Test Plan

```bash
pytest tests/v1/worker/test_gpu_model_runner.py tests/v1/spec_decode/test_dspark.py
pre-commit run --files vllm/v1/worker/gpu_model_runner.py
```

## Test Result

54 passed, 2 skipped on a Tesla V100; the assert does not fire on any of the
existing spec-decode paths. pre-commit clean.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
