Title: [CI] Reject new VLLM_* environment reads that bypass envs.py

## Purpose

envs.compile_factors() hashes every variable registered in vllm/envs.py into
the torch.compile cache key. A switch that is read straight from os.environ is
invisible to it: with the AOT compile cache on (the default since torch 2.10),
flipping such a switch can load an artifact compiled for the other setting.
We ran into this with our own route switches on a V100 + RTX 8000 rig: the
second boot after changing an unregistered switch loaded the previous graph
and failed; registering the switches in envs.py fixed it.

main currently reads 137 VLLM_* names directly that are not keys of
envs.environment_variables (831 registered). Many of them are dump, trace and
profiling switches that never change compiled code; those belong in
ignored_factors once registered, so they do not invalidate the cache. Others
select kernel routes (for example the FlashQLA prefill switches, the QSA
indexer switches or the SM70 indexer cuBLAS switches).

This PR does not register them. It adds a pre-commit hook so the list stops
growing:

- tools/pre_commit/check_env_registration.py flags os.getenv,
  os.environ.get, os.environ.setdefault and os.environ[...] reads of VLLM_*
  names under vllm/ that are not keys of envs.environment_variables (read
  with ast, so lambda, env_with_choices and named functions all count).
  Writes (os.environ[...] = ...) are not reads and pass.
- The 137 existing direct reads are listed as a baseline. The error message
  asks to register a new switch, and to add it to ignored_factors if it never
  changes compiled code.
- A test fails once a baseline name is registered but still listed, so the
  baseline only shrinks.

**Why this is not a duplicate.** No open PR or issue covers environment
registration or the compile cache key (gh pr list / gh search issues for
"envs.py", "compile_factors", "environment variable", "compile cache").
Related: #621 (compile cache opt-out) and #622/#672 (process-local addresses
in AOT artifacts) are about other ways an artifact can go stale.

AI assistance was used for this change. Every line was reviewed and the tests
below were run by the submitter.

## Test Plan

```bash
pytest tests/tools/test_check_env_registration.py
pre-commit run check-env-registration --all-files
```

## Test Result

5 passed. The hook passes on the whole tree (exit 0), rejects a new
os.getenv("VLLM_BRAND_NEW_SWITCH") and accepts a registered read
(VLLM_PORT) and an os.environ write. pre-commit and mypy-3.10 clean.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
