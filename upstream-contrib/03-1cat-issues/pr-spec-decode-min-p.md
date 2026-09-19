# PR-Entwurf C: min_p im Rejection-Sampler (Speculative Decoding)

Worktree: `vllm-research/1Cat-vLLM-pr-minp`, Branch `spec-decode-min-p` auf
`origin/main` (b711d530), Commit 384d44a0. Stand 19.09.2026: Änderung
fertig, pre-commit und mypy-3.10 grün, committet, GEPUSHT nach fork/spec-decode-min-p (19.09. abends) und ERÖFFNET als
https://github.com/1CatAI/1Cat-vLLM/pull/659 (Peuqui hat den Diff gelesen). Im Fork enthalten als 86c39898 (+ 6190a078 Typannotation).

Vor dem Absenden von Peuqui zu lesen: der Diff (4 Dateien) und dieser Text.

Kopplung: AIfred sendet seit bf13a12c min_p wieder an vLLM. Das setzt diesen
Stand voraus (im Fork erfüllt).

## PR-Titel

`[Feature][Spec Decode] Apply min_p in the rejection sampler`

## PR-Body

## Purpose

`min_p` is rejected outright as soon as speculative decoding is on:

```
The min_p and logit_bias sampling parameters are not yet supported with
speculative decoding.
```

Every deployment here that is worth running uses MTP, DSpark or DFlash, so a
client that sends the same sampling parameters to every backend has to strip
`min_p` for exactly these. Upstream vLLM still rejects it in this path as
well (checked at 4d433e4f).

This change applies `min_p` inside the verification step:

- `apply_sampling_constraints` cuts at `max_logit + log(min_p)` right after
  the temperature and before top-k and top-p. That is the place the regular
  sampler gives the argmax-invariant processors, so the order is the same.
  A request without `min_p` gets `log(0) = -inf` and loses nothing, which
  keeps the cut free of a per-request branch.
- The per-request values come from `MinPLogitsProcessor`, which is now built
  under speculative decoding too. In the regular path it filters the bonus
  token through the sampler, and the new cut covers the draft positions, so
  nothing is applied twice.
- In the combined-bonus fast path (`_forward_combined_bonus`) the cut runs
  over every sampled position, the bonus one included, so an active `min_p`
  keeps that path. Token matching hands per-token logits to a per-request
  processor and therefore stays off while `min_p` is active.
- `_validate_spec_decode` rejects `logit_bias` only, and the startup warning
  says so.

One existing test is corrected:
`test_combined_bonus_fast_path_matches_legacy_sampling` compares the two
paths token by token from the same RNG state. On sm75 and newer the regular
sampler draws the legacy bonus token with FlashInfer's own generator, so the
streams cannot match; the test failed on every run on an RTX 8000 and passed
on a V100. It now pins the native sampler
(`VLLM_USE_FLASHINFER_SAMPLER=0`), which is what it compares against.

## Test Plan

```
pytest tests/v1/sample/test_rejection_sampler.py -q
pytest tests/v1/logits_processors/test_correctness.py -q
pre-commit run --files <4 files>
pre-commit run mypy-3.10 --hook-stage manual --files <4 files>
```

Three new tests:

- `test_min_p_filters_draft_positions_like_the_regular_sampler`: three
  requests with `min_p` 0.0, 0.3 and 0.05 and different temperatures; the set
  of dropped tokens at every draft position equals
  `prob < min_p * max_prob` on the temperature-scaled distribution, and
  `min_p = 0` drops nothing.
- `test_min_p_keeps_the_combined_bonus_fast_path`.
- `test_speculative_decoding_accepts_min_p_and_rejects_logit_bias`.

## Test Result

Quadro RTX 8000 (sm75) and Tesla V100-PCIE-32GB (sm70), one card each,
`CUDA_DEVICE_ORDER=PCI_BUS_ID`.

With this change: `tests/v1/sample/test_rejection_sampler.py` `72 passed` on
both cards; `tests/v1/logits_processors/test_correctness.py` `29 passed`
(run as its own process, it forks).

Against unmodified `main` (b711d530), RTX 8000: the three new tests
`3 failed`; `test_combined_bonus_fast_path_matches_legacy_sampling`
`1 failed` (it fails there six times out of six).

pre-commit: all hooks passed; mypy-3.10 manual stage passed.

On our rig (DeepSeek-V4-Flash NVFP4 + DSpark, `num_speculative_tokens=5`,
temperature 1.0, top_k 40), running a fork that carries the same change:
requests with `min_p` 0.05 and 0.5 return coherent text where the server
answered with the error above before; a stress test of 120 short requests
with mixed sampling runs clean.

## Not a duplicate

```
gh pr list -R 1CatAI/1Cat-vLLM --state open --search "min_p"
gh pr list -R 1CatAI/1Cat-vLLM --state open --search "speculative min_p"
gh issue list -R 1CatAI/1Cat-vLLM --state open --search "min_p"
```

No hits.

## AI assistance

AI assistance (Claude) was used to write the change and the tests and to
draft this text. I have read every changed line and ran the tests above on
my own hardware.

---

## Offene Punkte für Peuqui vor dem Absenden

- Diff lesen: `git -C vllm-research/1Cat-vLLM-pr-minp show 384d44a0`.
- Das ist ein Feature, kein Bugfix. 1Cat könnte es ablehnen, weil Upstream es
  (noch) nicht hat. Alternativ zuerst als Issue anbieten.
- Push erst nach Go: `git push fork spec-decode-min-p`.
