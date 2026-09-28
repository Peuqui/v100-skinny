Title: [Bugfix][Spec Decode] Keep recovered tokens inside the vocabulary (backport vllm#44744)

## Purpose

Backport of vllm-project/vllm#44744 (advisory GHSA-8wr5-jm2h-8r4f), which
1Cat-vLLM does not have yet. `sample_recovered_tokens_kernel` takes a tiled
argmax over `BLOCK_SIZE = 8192` lanes; in the last tile the lanes past the
vocabulary load 0. vllm#44744 masks those lanes to `-inf` before `tl.max`
and clamps `recovered_id` to `vocab_size - 1`. This PR takes the kernel
change and its regression test `test_sample_recovered_tokens_vocab_boundary`
unchanged.

What we can reproduce on 1Cat main is a second trigger: an all-NaN target
row on the path without draft probabilities (DSpark, MTP, EAGLE). `tl.max`
skips the NaN lanes and picks a padded lane of the last tile, so the
recovered id is exactly `vocab_size` (1000, 10000, 129280 and 151936
measured; 8192 is aligned and returns 0). The next step looks that id up in
the embedding table and the engine dies on a device-side assert. This is
the chain #658 describes for DeepSeek-V4 under float16; #658 and #715
remove the NaN source there, this change keeps any remaining NaN row from
turning into an invalid token id.

With the NaN row the recovered token is still meaningless, but it is a
valid id: 0, or the draft token for a vocabulary below one tile.

To be precise about the advisory's scenario: with the Triton in our build
(3.6.0, torch 2.10.0) vllm#44744's regression test passes on 1Cat main as
well, so we have not reproduced the zero-tail case here. The added NaN test
fails on main.

**Why this is not a duplicate.** No open PR touches
`vllm/v1/sample/rejection_sampler.py` (`gh pr diff --name-only` over all
open PRs); `gh pr list --state all --search` for "sample_recovered_tokens",
"recovered token vocab", "GHSA-8wr5", "44744" and "rejection sampler nan"
finds only #658, which names the symptom but does not change the sampler.

AI assistance was used for this change. Every line was reviewed and the tests
below were run by the submitter.

## Test Plan

```bash
pytest tests/v1/sample/test_rejection_sampler.py
# counter-check: rejection_sampler.py reverted to main, tests kept
pytest tests/v1/sample/test_rejection_sampler.py -k "vocab_boundary or nan_rows"
```

## Test Result

On this branch (main 357d07bc): 86 passed on a Tesla V100 and on a Quadro
RTX 8000, including the existing comparisons of `sample_recovered_tokens`
against the torch reference. With main's `rejection_sampler.py`, on both
cards: the three new NaN cases without draft probabilities fail (recovered
id equal to `vocab_size`); the other 11 cases pass, including all eight of
vllm#44744's test. pre-commit and mypy-3.10 clean.
