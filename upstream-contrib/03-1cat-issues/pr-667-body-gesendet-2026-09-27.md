## Purpose

A cached DeepSeek-V4 prefix does not survive an unrelated request with main's
current prefix-cache code. A 23k prompt answers in 0.7 s when repeated; after one unrelated 30k
request it pays the full prefill again and no token comes from the cache.

This PR originally carried two parts. The first one, recycling blocks without
a hash before cached ones (with `prepend_n`), has landed in main in the
meantime through 1cbaa2c5, so it is gone from this branch. What remains is the
part main does not have yet:

Sliding-window groups cache the short run of blocks at each alignment boundary
(`reachable_block_mask`). Once the window has moved past a boundary, that run
can never serve a hit again, but it keeps its hash, so `free_blocks` appends it
behind other requests' cached blocks, and the next long prefill evicts other
requests' prefixes although it could reuse those runs.

This change lets sliding-window groups release the blocks their window has
moved past with `reuse_first`, which puts them at the front of the queue
whether they carry a hash or not. The one exception is the run at the last
alignment boundary of the prompt, which is exactly what a repeat of that
prompt looks up; it stays allocated until the request finishes and is then
released with everything else. The held range is derived with the same
formula as `reachable_block_mask`, including the EAGLE shift, and the
coordinator hands each manager its `lcm_block_size` the way it already hands
out `use_eagle`.

The branch is rebased onto current main; #657 has landed, so the note about
reviewing only the second commit no longer applies.

**Why this is not a duplicate.** 1cbaa2c5 on main recycles blocks without a
hash first; it does not touch blocks that still carry one, which is exactly
the case measured below (0 tokens served from the cache on main). #239 changes
*when* out-of-window blocks are freed under async scheduling; this PR changes
*where* freed blocks go in the free queue. Both touch the sliding-window
release path, so whichever lands second may need a small rebase there. No
other open PR or issue covers it (`gh pr list` searches for sliding window /
prefix cache eviction / free_blocks).

AI assistance was used for this change. Every line was reviewed and the tests
below were run by the submitter.

## Test Plan

```bash
pytest tests/v1/core/test_kv_cache_utils.py tests/v1/core/test_prefix_caching.py \
  tests/v1/core/test_single_type_kv_cache_manager.py \
  tests/v1/core/test_mamba_sparse_retention.py
```

End-to-end on 3x V100 + 2x RTX 8000, PP5, DeepSeek-V4-Flash (DeepSeek's MXFP4
checkpoint) with DSpark, 3000 blocks, chunk 512: prompt A (23009 tokens) cold,
repeated, then an unrelated 29948-token request B, then A again. Measured on
our fork (current main plus our open PRs and local patches), once with this
branch's three core files and once with main's versions of them, same build
otherwise.

## Test Result

290 passed on this branch (rebased onto 1e90d17f); `pre-commit` and
`mypy-3.10 --hook-stage manual` clean. New: `test_free_blocks_reuse_first_hands_out_dead_cached_blocks_first`,
which fails with main's `free_blocks`. The two tests this PR added before
(`prepend_n`, uncached before cached) stay: main has the behaviour but no test
for it yet.

| Measurement | main's core files | this PR |
|---|---|---|
| A cold | 10.8 s | 10.8 s |
| A repeated | 0.7 s (22784 cached) | 0.7 s (22784 cached) |
| B, unrelated 30k | 13.8 s | 13.8 s |
| A after B | 10.6 s (0 cached) | 0.7 s (22784 cached) |

🤖 Generated with [Claude Code](https://claude.com/claude-code)
