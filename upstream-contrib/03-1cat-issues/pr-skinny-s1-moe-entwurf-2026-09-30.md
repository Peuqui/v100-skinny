Title: [Kernel][SM70/SM75] Skinny QPN MoE backend for NVFP4 and MXFP4

## Purpose

`--moe-backend sm70_skinny` serves NVFP4 and MXFP4 MoE layers on SM70 and
SM75 through the skinny QPN kernels from dnv2003/v100-skinny (MIT). The
QPN2 layout already in `csrc/sm70_turbomind/ops/` comes from the same
project; the new file sits next to it and uses the same
`LICENSE.v100-skinny`. This is the PR that #718 asked about.

- `skinny_moe_qpn_sm70`: grouped MoE kernel with device-side routing
  (permutation and group offsets on the device, inactive experts exit
  before reading their weights). The launch does not scale with the expert
  count and captures into CUDA graphs. It serves batches up to
  `VLLM_SM70_NVFP4_MOE_GROUPED_MAX_TOKENS` (default 512).
- `skinny_qpn_gemm_sm70`: the per-expert QPN GEMM for larger batches, which
  route on the host and leave the capture through
  `eager_break_during_capture(ignore_full_mode=True)`.
- Expert weights are permuted in place into mma.m8n8k4 fragment order at
  load time, so the footprint and the parameter shapes stay those of the
  checkpoint. NVFP4 carries one fp8-e4m3 scale per 16 codes, MXFP4 one E8M0
  scale per 32; the kernels take the scale mode as a template argument.
  MXFP4 scales are rebased per expert into the range the fp16 decode covers,
  with the exact inverse as the expert's global scale, and an NVFP4 raster
  that is MXFP4 in disguise is folded to one scale per 32.

The backend is only used when requested; automatic selection does not
change.

Why it is useful here. The numbers come from our fork, which carries this PR
plus the pieces named at the end; within each comparison only the MoE backend
differs, one boot per path.

- MXFP4 MoE on SM70 and SM75. DeepSeek-V4-Flash (256 MXFP4 experts, FP8
  attention) with pipeline parallelism over 2x Quadro RTX 8000 and 3x Tesla
  V100, five stages. Prefill is the time to first token of a second fresh
  35.8k-token prompt after start. Without the drafter, layer partition
  11,7,7,7,11:

  | MoE path | prefill, 35.8k tokens | decode | follow-up TTFT |
  |---|---|---|---|
  | sm70_skinny | 18.9 s | 15.5 tok/s | 0.4 to 0.9 s |
  | default (TurboMind on SM70, Marlin on SM75) | 31.9 s | 6.8 tok/s | 1.1 to 2.0 s |

  With the DSpark drafter (k=5) the default path does not load on this
  machine. The TurboMind repack holds the checkpoint experts of a layer, the
  list of prepared experts and their stacked copy at the same time, while the
  other layers of the stage are already resident. With 8 layers per V100
  (partition 10,8,8,8,9) all three V100 stages run out of memory in
  process_weights_after_loading; with 7 per V100 (11,7,7,7,11) the Turing
  stage that also holds the drafter runs out while creating its weights.
  sm70_skinny keeps the checkpoint layout and serves 10,8,8,8,9 with the
  drafter: 16.9 s to the first token of the 35.8k prompt, 90 ms per decode
  round (about 35 tok/s), 1.0 to 1.2 s follow-up, greedy outputs identical
  to our production reference.
- NVFP4 MoE on SM70 and SM75. Qwen3.8-Flash-Next NVFP4 (512 experts, top-10) with
  pipeline parallelism over 2x Quadro RTX 8000 and 2x Tesla V100, MTP k=4,
  layer partition 12,12,13,11. Prefill is the time to
  first token of a fresh 29k-token prompt once the kernels are warm, the
  decode step is one MTP round. The three greedy test prompts give identical
  outputs on all three paths.

  | MoE path | prefill, 29k tokens | decode step | follow-up TTFT | KV cache |
  |---|---|---|---|---|
  | sm70_skinny | 17.4 s | 71 ms | 2.3 s | 612k tokens |
  | default (TurboMind on SM70, Marlin on SM75) | 18.6 s | 80 ms | 3.3 s | 481k tokens |
  | marlin | 13.9 s | 80 ms | 2.3 s | 639k tokens |

  With the even partition 12,12,12,12 the default path does not fit: its MoE
  weights take 1.6 to 1.9 GiB more per V100 stage, and the last stage runs
  out of memory while loading the MTP drafter. sm70_skinny keeps the
  checkpoint footprint. Against Marlin the trade is an 11% faster decode for
  a 25% slower prefill, so both are worth having as options.

Why this is not a duplicate. `gh pr list --state open --search` for "skinny",
"sm70_skinny", "moe_qpn", "MXFP4 MoE SM75", "MXFP4 Turing" and "MXFP4 SM70
MoE" finds only dense-linear work: #519 (AWQ QPN M1 operator) and our own
#604, #621, #623 and #714. None of them touches the MoE path, and this branch
merges cleanly with #519. #718 asked whether 1Cat wants these kernels and
this is the answer in code. The kernel file is new; `csrc/ops.h`,
`csrc/torch_bindings.cpp` and `CMakeLists.txt` only gain the two ops and the
source.

AI assistance was used for this change. Every line was reviewed and the tests
below were run by the submitter.

## Test Plan

```bash
pytest tests/kernels/moe/test_skinny_sm70_moe.py
pytest tests/v1/cudagraph/test_breakable_cudagraph.py
pytest tests/quantization/test_sm70_mxfp4_moe.py tests/quantization/test_sm70_moe_backend_override.py
pytest tests/kernels/moe/test_unquantized_backend_selection.py tests/benchmarks/test_sm70_quasar_nvfp4_oracle.py
pre-commit run --files <changed files>
pre-commit run mypy-3.10 --hook-stage manual --files <changed files>
```

## Test Result

The new tests check the MXFP4 rebasing (exact, and refused where fp16
cannot represent it), the scale folding, that the prepack only permutes the
checkpoint bytes, the QPN GEMM at M 1..16 and the grouped kernel in both
scale modes against dequantized references, both serving paths of the
experts against an fp32 MoE reference (largest error about 0.1% of the
output scale), and that the backend is taken only on request.

On this branch (main d3046986), each file in its own process:

| | Tesla V100 | Quadro RTX 8000 |
|---|---|---|
| test_skinny_sm70_moe.py | 23 passed, 12 skipped | 23 passed, 12 skipped |
| test_breakable_cudagraph.py | 12 passed | 12 passed |
| test_sm70_mxfp4_moe.py | 32 passed | 28 passed, 4 skipped |
| test_sm70_moe_backend_override.py | 7 passed | 7 passed |
| test_unquantized_backend_selection.py | 14 passed, 1 skipped | 14 passed, 1 skipped |
| test_sm70_quasar_nvfp4_oracle.py | 6 passed | 6 passed |

The skipped cases of the new file are the other card generation. The
kernels are byte-for-byte the ones we run: against the v100-skinny build,
`skinny_qpn_gemm_sm70` and `skinny_moe_qpn_sm70` give bit-identical outputs
at M 1..16, 1..300 tokens and every (splitk, nacc) configuration, in both
scale modes. pre-commit clean, mypy-3.10 passed.

End to end, the same code serves Qwen3.8-Flash-Next (NVFP4, 512 experts)
and DeepSeek-V4-Flash (MXFP4, 256 experts) in our fork. It cannot run end to
end on this branch alone here: Flash-Next needs the PLE cascade (#646) on our
30 GiB host, and DeepSeek-V4 needs pre-Hopper paths for the Turing stages
that a follow-up PR brings. In the DeepSeek-V4 runs above the dense FP8
layers also use our QPN8 block kernel, identical in both rows; that kernel
is a separate follow-up.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

---
Status: gesendet 2026-09-30 abends als https://github.com/1CatAI/1Cat-vLLM/pull/742 (Zweig pr-sm70-skinny-kernels, 5ddc0475); Kommentar in #718: https://github.com/1CatAI/1Cat-vLLM/issues/718#issuecomment-5916182317
