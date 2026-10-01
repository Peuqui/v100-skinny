Title: [Perf][SM70/SM75] Block FP8 through the native QPN8 operators (opt-in)

## Purpose

The native `fp8_qpn8_*` operators serve block-FP8 linears faster than the
default routes on Volta and Turing. Main admits them only in two places:

- for listed layer names and shapes (`VLLM_SM70_FP8_QPN8`);
- for the serialized PP2 x TP4 contract.

Both sit inside the exact-SM70 TurboMind branch. Every other [128, 128]
block-FP8 linear, for example DeepSeek-V4's `fused_wqa_wkv`, `wq_b`, `wo_b`
and the indexer's `wq_b`, goes to TurboMind on Volta and to Marlin on Turing.
TurboMind's FP8 GEMM has no sm_75 kernel.

With `VLLM_SM70_FP8_BLOCK_QPN8=1`, `QPN8Fp8BlockScaledMMLinearKernel` serves
every [128, 128] block-FP8 linear whose N and K are multiples of 128, on both
generations. It does not depend on layer names or on the parallel layout.

- M <= 8 runs `fp8_qpn8_gemm` on the packed codes. Larger M runs
  `fp8_qpn8_prefill` into a dense buffer, as the existing dispatch does.
- Each call takes its dense buffer from the caching allocator on its own
  stream. DeepSeek-V4 runs the indexer's `wq_b` on an aux stream next to the
  main `wq_b`. With one shared buffer the two dequantized weights overwrite
  each other, and a test covers that case.
- The kernel registers first in the CUDA block-FP8 list and is supported
  only on Volta and Turing with the switch set. On Volta, `Fp8LinearMethod`
  then skips the TurboMind and dequant-fallback branches for block FP8.
  `Fp8Config` admits FP8 checkpoints on SM70 under the switch, as it does
  under the TurboMind switches.
- `is_bmm` layers (DeepSeek-V4 `wo_a`) take the grouped fp16 path from #747
  when QPN8 is the kernel, as they do with Marlin.

Default off, so nothing changes without the switch. Stacked on #747 because
of the `wo_a` path.

Why this is not a duplicate. `gh pr list --state open --search` for "QPN8",
"block fp8", "fp8 sm75" and "fp8 turing" finds:

- #405: a design document for the Qwen3.8-27B-FP8 TP4 QPN8 shapes, no code;
- #604: ModelOpt FP8 on Turing, a different quant method;
- #710: workspace addresses on AOT reload, touches other lines of fp8.py;
- the DeepSeek-V4 and MoE PRs: #742, #744, #745, #716, #747.

None of them routes `Fp8Config` block-FP8 linears to QPN8 in general or on
Turing.

AI assistance was used for this change. Every line was reviewed and the tests
below were run by the submitter.

## Test Plan

```bash
pytest tests/kernels/quantization/test_sm70_qpn8_block_fp8.py
pytest tests/quantization/test_fp8_grouped_bmm_dequant.py
pytest tests/models/test_deepseek_v4_sm70_routes.py tests/v1/cudagraph/test_cudagraph_dispatch.py
pytest tests/quantization/test_fp8.py
pre-commit run --files <changed files>
```

End to end: DeepSeek-V4-Flash (MXFP4 experts, FP8 block linears, DSpark k=5)
at PP5 on 3x Tesla V100 + 2x Quadro RTX 8000. The runs used main plus the
open PRs, #747, #716 (BMM switches on) and this PR, with
`VLLM_SM70_QUANT_BACKEND=auto`. The two boots differ only in
`VLLM_SM70_FP8_BLOCK_QPN8`. Both used the same fresh prompts.

## Test Result

`test_sm70_qpn8_block_fp8.py` covers:

- the opt-in, and the kernel selection with and without it;
- the geometry gate;
- M from 1 to 512 against the dequantized reference;
- two streams running the prefill path at once.

| | Tesla V100 | Quadro RTX 8000 |
|---|---|---|
| QPN8, grouped wo_a, routes, cudagraph dispatch | 92 passed | 92 passed |
| test_fp8.py | 8 failed, same 8 as on #747 without this PR | |

End to end:

| | switch off (TurboMind on V100, Marlin on RTX) | switch on (QPN8) |
|---|---|---|
| 35.7k / 35.8k prompt, cold prefill | 16.7 / 16.9 s | 15.2 / 15.4 s |
| decode step at 35.7k | 100-102 ms | 85-86 ms |
| decode step, short prompt | 101-104 ms | 85-88 ms |
| first request at a new prompt length | up to 24 s before the first token | no stall |
| P(end of sequence) as first token, 5 probes | 0 | 0 |
| 8 reference prompts | all correct | all correct |

The stall with the switch off is TurboMind measuring its FP8 GEMM
candidates on each V100 stage for a new M. In the run the eight reference
prompts took 2.6 to 27.2 s each, against 2.1 to 7.2 s with QPN8. With the
switch on, greedy output and all eight answers are identical to our fork,
which has run this kernel in production since 2026-09-30.

pre-commit clean.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
