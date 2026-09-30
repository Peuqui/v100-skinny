Title: [Bugfix][SM70] Repack DeepSeek-V4 MXFP4 experts for TurboMind without a second copy

## Purpose

`Mxfp4SM70MoEMethod.process_weights_after_loading` prepared every expert into
a Python list and stacked the list at the end, and the checkpoint tensors
stayed alive until both projections were stacked. For the layer being
processed that is the checkpoint experts, the per-expert list and the
stacked copy at the same time, and post-load processing runs when every
layer of the stage is already resident. On DeepSeek-V4-Flash with pipeline
parallelism over 2x Quadro RTX 8000 and 3x Tesla V100 (layer partition
10,8,8,8,9, DSpark drafter), all three V100 stages ran out of memory there
(`Tried to allocate 2.00 GiB` in the stack of `mxfp4_sm70_moe.py`).

Each prepared expert is now copied straight into a preallocated stack, and
the w13 checkpoint is released before w2 is repacked. The prepared tensors
are byte-identical to before; only the order of allocation and release
changes. The per-expert conversion moved into a small helper,
`_prepare_mxfp4_sm70_experts`, which serves both projections.

End to end on the same rig, default MoE path (TurboMind on the V100
stages, Marlin on the RTX stages), partition 10,8,8,8,9: before the change
the three V100 stages ran out of memory in this repack; with it they finish
the TurboMind warmup and the model serves (checked without the drafter,
three greedy prompts plus two 35.8k-token prompts). With the DSpark drafter
the last RTX stage still runs out, now in the Marlin repack of
`marlin_utils_fp4.py`, which builds a per-expert list and concatenates it
in the same way. That code is shared with upstream vLLM and is not part of
this PR.

Why this is not a duplicate. No open PR changes `mxfp4_sm70_moe.py`;
`gh pr list --state open --search` for "mxfp4_sm70_moe", "MXFP4 repack",
"TurboMind MXFP4 memory", "MXFP4 MoE OOM" and "DeepSeek-V4 V100 OOM" finds
only unrelated work and our own #742, which adds a different MoE backend and
does not touch this file. Found while measuring #742 against the default
path.

AI assistance was used for this change. Every line was reviewed and the tests
below were run by the submitter.

## Test Plan

```bash
pytest tests/quantization/test_sm70_mxfp4_moe.py
pre-commit run --files vllm/model_executor/layers/quantization/mxfp4_sm70_moe.py tests/quantization/test_sm70_mxfp4_moe.py
pre-commit run mypy-3.10 --hook-stage manual --files vllm/model_executor/layers/quantization/mxfp4_sm70_moe.py tests/quantization/test_sm70_mxfp4_moe.py
```

## Test Result

The new test runs the post-load repack on a DeepSeek-V4-Flash TP8 layer
(256 experts), checks every TurboMind tensor bit for bit against preparing
each expert and stacking afterwards, and bounds the allocation growth of the
repack by one prepared copy of both projections. On a V100 the repack now
grows the allocation by 284 MiB; with the previous code the same layer grows
it by 819 MiB and the test fails.

On main d3046986:

| | Tesla V100 | Quadro RTX 8000 |
|---|---|---|
| test_sm70_mxfp4_moe.py | 33 passed | 28 passed, 5 skipped |

pre-commit clean, mypy-3.10 passed.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

---
Status: gesendet 2026-09-30 spät als https://github.com/1CatAI/1Cat-vLLM/pull/743 (Zweig pr-sm70-mxfp4-repack-peak, 400baa11); #674 am selben Abend nachgezogen
