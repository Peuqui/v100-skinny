# PR-Entwurf A: mHC bleibt unter float16 endlich (Attention-Sink-Zeile)

Worktree: `vllm-research/1Cat-vLLM-pr-mhcfp16`, Branch `mhc-fp16-range`
auf `origin/main` (b711d530), Commit db900f3d. Stand 19.09.2026: Änderung
fertig, pre-commit und mypy-3.10 grün, committet, GEPUSHT nach fork/mhc-fp16-range (19.09. abends) und ERÖFFNET als
https://github.com/1CatAI/1Cat-vLLM/pull/658 (Peuqui hat den Diff gelesen). Im Fork enthalten als 3e9d20e6 (dort zusätzlich die Fork-eigenen
torch-Pfade und das Modellende, die es upstream nicht gibt).

Vor dem Absenden von Peuqui zu lesen: der Diff (4 Dateien, +195/−13) und
dieser Text. AGENTS.md verlangt, dass ein Mensch jede Zeile vertreten kann.

## PR-Titel

`[Bugfix][DeepSeek-V4] Keep mHC finite under float16 for attention-sink rows`

## PR-Body

## Purpose

DeepSeek V4 is trained in bfloat16. Its attention-sink row (the BOS token)
carries one residual channel far above the rest. Measured on
DeepSeek-V4-Flash, per-row maximum of the residual stream: about 2k at
layer 27, 20k at layer 33, 35k at layer 35, 63.5k at layer 40, and past
65504 from layer 41 on. Every other token stays below about 200. Under
float16 three places in the mHC kernels lose that row.

1. `hc_prenorm_gemm_tilelang` and its `block_m` variant read `x_val` as
   float16 and accumulate `sqr[0] += x_val * x_val`. The product of two
   float16 values is float16, so beyond |x| = 255.9 it is inf before it
   reaches the float32 accumulator. The square sum is inf, the norm factor 0,
   and the row's mixes fall back to `hc_base` alone. Only the large-batch path
   (more than 16 tokens, not SM70) does this; `mhc_fused_tilelang` and the
   SM70 Triton staging square in float32. A 13-token and a 26-token prompt
   therefore disagree on the very same BOS row, which is causal and cannot
   depend on the prompt. We saw it as different expert routing for token 0:
   one dominant expert at weight 1.45 on the small path, six smeared weights
   on the large one.

2. A residual beyond 65504 is stored as inf. The next attention turns that
   into NaN for every token that still has the row in its sliding window,
   which is every token of a prompt shorter than the window. Residual stores
   now saturate at the largest finite float16: `mhc_fused_tilelang`,
   `mhc_post_tilelang`, `sm70_mhc_post_fp32_stage_tilelang`, the SM70 Triton
   post kernel and `mhc_post_torch`. A kernel that reuses the value it stores
   reuses the saturated one, so what it computes on matches what is stored.

3. The unnormalized pre-mix sum of the four streams is staged in float16
   before the fused RMSNorm scales it down to about 64. Four streams near
   65504 times a pre weight of about 0.6 is about 157k. float16 runs now
   stage it in float32, which is exact instead of clipped. The SM70 Triton
   pre-norm kernel carried a deliberate `.to(tl.float16).to(tl.float32)`
   round trip that copied this staging; it is removed.

For bfloat16 runs the saturation and the float32 staging are gated on
`use_fp16` and do not apply. The one thing that changes there too is that the
two GEMM kernels now square in float32 instead of bfloat16, which has the
same range and is more precise. I have no bfloat16-capable card to measure
that on.

How it surfaced for us: with async scheduling a few steps run past the end
of a request with token id 0 in the first row, which is BOS for this
tokenizer. That BOS in the middle of a context overflowed, its logits row
became NaN, `sample_recovered_tokens` returns `vocab_size` for an all-NaN
row, and the next step looked that id up in the embedding table: "CUDA
error: device-side assert triggered", engine dead right after a request
ended. Greedy requests never showed it, argmax over NaN is a valid id.

Not covered here: `hc_head_fuse_tilelang` writes its output in the
activation dtype before the final norm and sums four streams as well. We
could not test it (its TileLang codegen does not build on our pre-Ampere
cards), so it is left alone rather than changed blind. #643 (hidden state
growing without bound, NaN spreading causally from an all-zero-id dummy)
reads like the same family on another model; we have not checked it.

## Test Plan

```
pytest tests/kernels/test_mhc_fp16_range.py \
       tests/kernels/test_mhc_sm70_fp16.py \
       tests/kernels/test_mhc_kernels.py -q
pre-commit run --files <4 files>
pre-commit run mypy-3.10 --hook-stage manual --files <4 files>
```

The new test drives `mhc_fused_post_pre_tilelang` on both sides of the
16-token switch (6, 13, 16, 17, 26 tokens) with a sink channel of 200, 3000
and 35000, and with a residual that has to exceed float16 (exact value
82800). It checks the stored residual, the mixes and the layer input of the
sink row against `mhc_post_torch` and `mhc_pre_torch` in float32.

## Test Result

Tesla V100-PCIE-32GB (sm70) and Quadro RTX 8000 (sm75), one card each,
`CUDA_DEVICE_ORDER=PCI_BUS_ID`.

With this change: V100 `88 passed, 9 skipped`, RTX 8000 `74 passed, 23
skipped` for the three files together.

The new test alone against unmodified `main` (b711d530):

- V100: `11 failed, 10 passed`. All 35000 cases and all overflow cases fail
  with `Greatest absolute difference: inf at index (3077,)` in the layer
  input, plus the torch store.
- RTX 8000: `13 failed, 8 passed`. The same, plus the 3000 cases at 17 and
  26 tokens, which is defect 1: the mixes come back as
  `[1.005, 0.98, 1.045, 1.04]` against a reference of
  `[0.718, 1.301, 0.752, 0.812]`.

pre-commit: all hooks passed; mypy-3.10 manual stage passed.

On our rig (DeepSeek-V4-Flash NVFP4 + DSpark, pipeline parallel over two RTX
8000 and three V100, float16) we run a fork of this repository. There the
same kernel changes, together with a saturating cast at the end of the model
that only exists in the fork's pre-Ampere torch paths, were measured: a
stress test of 120 short requests, three in flight, greedy and temperature
1.0 mixed, killed the engine within three requests before and runs clean now,
three times. Prompts of 13 and 14 tokens that came back empty answer
correctly. A 30k-token needle test finds 4 of 4, decode speed and draft
acceptance are unchanged. I have not run this exact branch as a server.

## Not a duplicate

```
gh pr list -R 1CatAI/1Cat-vLLM --state open --search "mhc"
gh pr list -R 1CatAI/1Cat-vLLM --state open --search "hc_prenorm"
gh pr list -R 1CatAI/1Cat-vLLM --state open --search "float16 overflow"
gh pr list -R 1CatAI/1Cat-vLLM --state open --search "attention sink"
gh issue list -R 1CatAI/1Cat-vLLM --state open --search "mhc"
gh issue list -R 1CatAI/1Cat-vLLM --state open --search "NaN"
```

No open PR touches `vllm/model_executor/kernels/mhc/`. The only `mhc` hit is
the dependabot PR #590. Open issue #620 is about the TileLang version, #643
is mentioned above.

## AI assistance

AI assistance (Claude) was used to trace the NaN to these kernels, to write
the change and the test, and to draft this text. I have read every changed
line and ran the tests above on my own hardware.

---

## Offene Punkte für Peuqui vor dem Absenden

- Diff lesen: `git -C vllm-research/1Cat-vLLM-pr-mhcfp16 show db900f3d`.
- Entscheidung: #643 erwähnen (wie oben, ausdrücklich ungeprüft) oder
  weglassen.
- Push erst nach Go: `git push fork mhc-fp16-range`, dann PR gegen
  `1CatAI/1Cat-vLLM:main`.
