Title: [Bugfix] Hash unregistered VLLM_ env vars into the compile cache key

Branch: Peuqui/1Cat-vLLM  compile-cache-key-unregistered-env  (e9e5941)
Files:  vllm/envs.py (+15), tests/test_envs.py (+38)

## The bug

compile_factors() builds the environment part of the compile cache key from
every variable declared in vllm/envs.py, minus an explicit ignore list.
Anything read straight from os.environ is therefore invisible to the key.
171 of the 275 VLLM_-prefixed variables read via os.getenv/os.environ across
this tree are not declared in envs.py, among them kernel-route switches for
QSA, FlashQLA, the GDN paths, the SM70 indexer, the MoE variants and the
NVFP4 linear routes.

The AOT artifact store makes this reachable. Its key, by its own comment in
compilation/decorators.py, "contains all of the factors except for the
source files being traced through". So the source-file check cannot save a
switch that only selects a different runtime branch inside an unchanged
file, and the environment hash is the only thing left -- exactly where these
switches are missing.

## Reproduction (measured, not argued)

Qwen3.8-27B-NVFP4, one Quadro RTX 8000, TP1, greedy, dedicated
VLLM_CACHE_ROOT, VLLM_DISABLE_COMPILE_CACHE=0.

| Run | Configuration | Behaviour | Result |
|---|---|---|---|
| 1 | Marlin route, cache empty | compiled 90.2 s | correct answer |
| 2 | fork NVFP4 route, cache empty | compiled 91.5 s | correct answer |
| 3 | Marlin route, cache from run 2 | "Directly load AOT compilation", 16.3 s | engine start fails, `KeyError: 'skinny_codes'` |

Run 3 loaded the artifact compiled for the other kernel route. The graph
expects a weight stash that the Marlin route never creates, so the engine
core dies during startup. Before the change, flipping four such switches
left compile_factors() bit-identical (hash 1360e6c6ce7396c5 in both
directions).

## The change

After the declared variables, hash every VLLM_-prefixed variable that is
actually set and is neither already covered nor explicitly ignored. One
place, so out-of-tree and future switches are covered without anyone having
to remember to register them. An over-invalidated cache costs a recompile; a
wrongly reused one costs correctness.

## Verification

| Run | Configuration | Behaviour | Result |
|---|---|---|---|
| 4 | Marlin route, cache from run 2, with the change | recompiled 89.6 s, second artifact written | correct answer |
| 5 | Marlin route, cache empty, with the change | compiled 89.0 s | correct answer |
| 6 | Marlin route, cache from run 5, with the change | "Directly load AOT compilation", 15.9 s | correct answer, byte-identical text (sha f506073edf9b) |

Runs 4 and 6 together show the change blocks reuse only across switch
boundaries and still hits for an unchanged configuration, so the ~73 s saved
per boot are preserved. With no such switch set, the key is unchanged:
691 factors, hash 1360e6c6ce7396c5, identical to before the change.

Test commands and results:

    python -m pytest -q tests/test_envs.py
      -> 58 passed  (includes the two new tests)

    python -m pytest -q tests/test_envs.py tests/config/test_config_utils.py \
                       tests/config/test_multimodal_config.py
      -> 85 passed

    python -m pytest -q tests/entrypoints/openai/test_fingerprint.py
      -> 4 passed

    python -m pytest -q tests/compile/test_aot_compile.py
      -> 10 failed, 17 passed, 1 error
      Pre-existing on this tree, NOT caused by this change: the same module
      gives exactly 10 failed / 17 passed / 1 error with vllm/envs.py reset
      to origin/main. The failures are AttributeError: 'NoneType' object has
      no attribute 'model' at vllm/compilation/backends.py:994, i.e. missing
      engine context in the fixture, unrelated to the cache key.

    pre-commit run --files vllm/envs.py tests/test_envs.py
      -> ruff check, ruff format, SPDX headers, root lazy imports, forbidden
         imports, config validation, attention-backend docs: all Passed,
         no hook failed

## Not a duplicate

No open PR or issue in this repository addresses the compile cache key
(searched "compile cache", "compile_factors OR cache key OR env").
Upstream vllm-project/vllm carries per-factor fixes of the same bug class,
e.g. #55386 (LoRA wrap state into the AOT cache key) and #54642 (multimodal
limits into the compile-cache hash). This change is materially different: it
closes the class for environment-driven switches instead of adding one more
factor by hand.

## AI assistance

This change was prepared with AI assistance (Claude). Every changed line and
every measurement above is to be reviewed by the human submitter before the
PR is opened.
