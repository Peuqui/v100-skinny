# PR-Entwurf B: Backport vllm-project/vllm#44082 (Präfix-Cache 0 % unter EAGLE/MTP/DSpark)

Worktree: `vllm-research/1Cat-vLLM-pr-swaeagle`, Branch
`swa-prefix-cache-eagle-mask` auf `origin/main` (b711d530). Stand
19.09.2026: Änderung fertig, pre-commit und mypy-3.10 grün, committet
(456a62f5, inhaltlich identisch mit dem Fork-Commit).

ERÖFFNET 19.09.2026 abends als https://github.com/1CatAI/1Cat-vLLM/pull/657 (Go Peuqui: „wir machen
beides"), Kommentar in #205: https://github.com/1CatAI/1Cat-vLLM/issues/205#issuecomment-5743686861. Im Fork enthalten als 0c250990.

WICHTIG vor dem Absenden: Zu diesem Fehler gibt es das offene Issue
1CatAI/1Cat-vLLM#205 von kezboard233 (13.08.2026, 0 Kommentare, kein PR). Er
hat dort einen vollständigen Backport-Patch gegen `main@4695b82` angehängt und
bewusst ein Issue statt eines PRs eingereicht („so you can carry the change
yourself in whatever form you prefer"). Unsere vier Handkorrekturen sind mit
seinen identisch. Was wir zusätzlich beitragen: aktueller Stand von `main`,
die Umbenennung in 1Cats eigenen `find_longest_cache_hit`-Varianten, und eine
Betriebsmessung mit laufendem Server, die er ausdrücklich nicht hatte.
Peuqui entscheidet: PR (wie unten) ODER Kommentar im Issue mit Verweis auf
unseren Branch.

## PR-Titel

`[Bugfix][Core] Backport vllm-project/vllm#44082: cache the EAGLE/MTP lookahead block in the SWA prefix-cache mask`

## PR-Body

## Purpose

Fixes #205.

With a sliding-window KV cache group whose block size is smaller than the
alignment (DeepSeek V4: block 64, window 128, lcm 256) and EAGLE, MTP or
DSpark speculation, the prefix cache never hits. The engine logs
`Prefix cache hit rate: 0.0%` on every interval and nothing else; every
follow-up turn of a conversation recomputes its whole context.

The cause and the fix are upstream's: vllm-project/vllm#44082 by ivanium,
merged 2026-06-02. The write-side mask registers only the last blocks of each
aligned segment, while the lookup under speculation needs one more
contiguous block, the one past the aligned boundary, which it then drops.
The two sets never meet. @kezboard233 reported this for this repository in
#205, with a root-cause analysis and a hand-merged patch against an older
`main`; this PR is the same backport against current `main`, and all credit
for the analysis is theirs. Our four hand-placed hunks came out identical to
the ones listed in #205 section 5, which we take as a good sign for both.

Applying `gh pr diff 44082 -R vllm-project/vllm` to b711d530: three files
apply cleanly, four hunks are rejected and were placed by hand.

1. `single_type_kv_cache_manager.py`: `self.use_eagle = False` in `__init__`.
   Rejected only because this base has `take_pending_boundary_state_offloads`
   right after `__init__`.
2. `kv_cache_coordinator.py`: `from typing import NamedTuple`.
3. `verify_and_split_kv_cache_groups` in upstream's `SpecGroup` form, keeping
   what this base adds: the `prefix_cacheable` skip, the "at least one
   cacheable group" assertion, and the `lcm_block_size` computation.
   `alignment_tokens` keeps `self.lcm_block_size` where upstream passes
   `self.scheduler_block_size`, which this base does not have.
4. The hunk that spans two methods. In `find_longest_cache_hit` the call site
   takes `drop_eagle_block`, the per-round flag; left as `use_eagle` it would
   pass the group-level flag and lose the convergence guard, silently. In
   `HybridKVCacheCoordinator.cache_blocks` an EAGLE group may cache one block
   past the aligned boundary. We made exactly the mistake #205 warns about
   in our own first attempt: a mask-only fix that gets hits but never
   registers that lookahead block.

Beyond upstream's diff, this base has its own `find_longest_cache_hit`
variants (the two no-op managers and the Mamba align manager) that still
named the parameter `use_eagle` in their bodies; they take
`drop_eagle_block` now, otherwise they raise `NameError`. Upstream's three
regression tests build the manager through a helper that passes
`scheduler_block_size`; here they call `KVCacheManager(...)` directly, this
base derives the lcm itself. `test_mamba_honors_eagle_cache_drop` passes the
renamed keyword.

## Test Plan

```
pytest tests/v1/core/test_prefix_caching.py \
       tests/v1/core/test_single_type_kv_cache_manager.py -q
pre-commit run --files <5 files>
pre-commit run mypy-3.10 --hook-stage manual --files <5 files>
```

## Test Result

With this change: `77 passed`.

Upstream's three regression tests against unmodified `main` (b711d530), test
file only:

```
FAILED test_eagle_swa_alignment_caches_extra_block
FAILED test_eagle_swa_boundary_caches_post_boundary_block
FAILED test_eagle_grouped_swa_siblings_use_same_cache_mask
AssertionError: EAGLE + SWA with sliding_window <= alignment failed to find
any cache hit; the +1 block past each segment boundary must be cached.
3 failed, 62 deselected
```

pre-commit: all hooks passed; mypy-3.10 manual stage passed.

Serving, which #205 says it could not measure. DeepSeek-V4-Flash NVFP4 +
DSpark (`num_speculative_tokens=5`), `--enable-prefix-caching`, pipeline
parallel over two RTX 8000 and three V100, on a fork of this repository that
carries the same backport:

- The same 5131-token request sent twice with the model loaded: 18.5 s, then
  1.5 s, with `cached_tokens: 4864` reported for the second one. 4864 is 19 aligned
  segments of 256; the twentieth is the lookahead block EAGLE drops.
- Follow-up turns at 18k context reach the first token in 1.4 to 1.6 s.
  Before, every turn took 85 to 99 s.
- `Prefix cache hit rate` in the engine log goes from 0.0% to 56.8% over the
  test session.
- A 30k-token needle test returns 4 of 4 both cold and from the cache, so the
  cached blocks are the right ones.

## Not a duplicate

```
gh issue view 205 -R 1CatAI/1Cat-vLLM --comments        # open, no comments
gh pr list -R 1CatAI/1Cat-vLLM --state all --search "205 in:body"
gh pr list -R 1CatAI/1Cat-vLLM --state all --search "44082"
gh pr list -R 1CatAI/1Cat-vLLM --state open --search "cache_block_mask"
gh pr list -R 1CatAI/1Cat-vLLM --state open --search "prefix cache sliding window"
```

No PR carries this fix. #603 mentions #205 in passing and touches none of the
five files; #239 matches the last search on wording only.

Overlap to be aware of: #617 (open, Mamba prefix-cache retention, backport of
upstream #45845/#47782) and #598 stacked on it change the same three core
files. #617 does not contain #44082: its diff has no `drop_eagle_block`, no
`SpecGroup`, no `_contiguous_blocks_for_hit` and none of the three regression
tests. It does add a `reachable_block_mask` classmethod to the Mamba manager,
with a different signature (`retention_interval`, `reachable_boundaries`),
and it edits `HybridKVCacheCoordinator.cache_blocks`. The two changes are
independent in substance but will conflict textually in
`single_type_kv_cache_manager.py` and `kv_cache_coordinator.py`. Whichever
lands second needs a rebase; I am happy to do that for this one.

## AI assistance

AI assistance (Claude) was used for the hand-merge, the comparison with
upstream and with #205, and this text. I have read every changed line and
ran the tests and the server measurements above on my own hardware.

---

## Offene Punkte für Peuqui vor dem Absenden

- Entscheidung PR oder Issue-Kommentar (siehe oben).
- Diff lesen: `git -C vllm-research/1Cat-vLLM-pr-swaeagle show 456a62f5`.
- Den GitHub-Namen @kezboard233 im PR-Text nennen ist gewollt (Würdigung);
  falls nicht erwünscht, auf „the author of #205" ändern.
- Push erst nach Go: `git push fork swa-prefix-cache-eagle-mask`.
