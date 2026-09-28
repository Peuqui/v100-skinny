Title: [Bugfix][DeepSeek-V4] Keep hc_head finite under float16 for attention-sink rows

## Purpose

Follow-up to #658. #658 saturates the float16 stores of the mHC pre and post
kernels, so an attention-sink row (one residual channel far above the rest, as
the first token of DeepSeek V4 has in its last layers) stays finite. The fused
`hc_head` kernel (`hc_head_fuse_tilelang`) was left out: it accumulates the
sigmoid-weighted sum of the four streams in float32 and copies it straight into
the float16 output. For a sink row that sum exceeds 65504 and the head input
becomes inf.

The change clamps that store to the float16 range under `use_fp16`, with the
same line #658 uses for the other stores. bfloat16 runs are unaffected, and
rows inside the float16 range are unchanged.

We had worked around this in our fork with a torch reference for hc_head on
pre-Ampere float16; with this change the kernel itself is correct and the
workaround can go.

**Why this is not a duplicate.** #658 touched `mhc_pre`/`mhc_post` and the
Triton and torch paths, not `hc_head_fuse_tilelang`. No open PR touches
`vllm/model_executor/kernels/mhc/tilelang_kernels.py` (`gh pr list --state open
--search` for "hc_head", "mhc float16", "tilelang_kernels").

AI assistance was used for this change. Every line was reviewed and the tests
below were run by the submitter.

## Test Plan

```bash
pytest tests/kernels/test_mhc_fp16_range.py
# counter-check: tilelang_kernels.py reverted to main, test kept
pytest tests/kernels/test_mhc_fp16_range.py -k hc_head
```

## Test Result

On this branch (main 357d07bc): 26 passed (21 existing, 5 new) on a Tesla
V100 and on a Quadro RTX 8000. With main's `tilelang_kernels.py` the five new
cases fail on both cards with inf in the sink row:

```
assert torch.isfinite(out).all()
... tensor([[     inf,      inf,      inf,  ...,      inf,      inf,      inf],
        [  3.7051,  -0.9517,   5.5156, ...
```

The new test covers 6, 13, 16, 17 and 26 tokens (both sides of the 16-token
kernel switch used elsewhere in this file), checks that the sink row saturates
to 65504 and that the other rows match a float32 reference. `hc_base` is fixed
at 3.0 so that the sink row overflows independently of the random weights.
