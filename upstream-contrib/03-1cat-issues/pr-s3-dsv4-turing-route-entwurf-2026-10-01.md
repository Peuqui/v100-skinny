Title: [Feature][DeepSeek-V4] Run the SM70 route on Turing

## Purpose

DeepSeek-V4 has no route on Turing (sm_75) on current main.
`_select_v4_sparse_impl` sends every CUDA card except exact SM70 to FlashMLA
(SM90+), the fused Q/KV insert is sm_80+, the indexer needs DeepGEMM, and the
O projection needs DeepGEMM's `fp8_einsum` or the SM70 TurboMind grouped
kernel. A pipeline that mixes V100 and Turing cards cannot start.

The SM70 route itself does not depend on Volta. The sparse MLA, the Q/KV
insert, the indexer query and its scoring are FP16 Triton (or cuBLAS) with
software FP8 decode. The only Volta-only piece on the way is the TurboMind
grouped `wo_a`. This PR lets Turing take the route:

- One predicate, `current_platform.is_device_capability_family(70)` (Volta or
  Turing), replaces exact `(7, 0)` at the gates of the route:
  - impl selection and `_use_sm70_path` (attention.py);
  - `DeepseekV4SM70SparseBackend.supports_compute_capability`;
  - the FlashMLA tile-scheduler skip (sparse_swa.py) and the C128A fixed row
    stride (flashmla_sparse.py);
  - the FP16 indexer (sparse_attn_indexer.py, fused_indexer_q.py) and the
    cuBLAS indexer decode (sm70/indexer.py);
  - the compressor kernel choice;
  - DSpark's `main_proj` input scale and context KV insert;
  - the `--dtype half` check;
  - the compressed-index decode graph buckets (cudagraph_dispatcher.py).
- `wo_a`. Turing takes Marlin for block FP8, and Marlin packs one [N, K]
  matrix, so it cannot serve the grouped matmul of an `is_bmm` layer. When
  Marlin is the kernel, `Fp8LinearMethod` now dequantizes an `is_bmm` weight
  to fp16 once at load and applies it per group, with the same
  `[..., groups, K] -> [..., groups, R]` contract as the TurboMind grouped
  path. `sm70_grouped_output_projection` therefore stays as it is. The cost is
  one extra byte per `wo_a` weight on those stages: 32 MiB per layer for
  DeepSeek-V4-Flash (8 groups x 1024 x 4096). The same condition also applies
  on V100 when `VLLM_SM70_QUANT_BACKEND=marlin` selects Marlin for block FP8.

These stay exact SM70, because they are Volta kernels or opt-ins measured on
V100 only:

- TurboMind FP8 and the grouped TurboMind `wo_a`;
- the FP13 GEMV;
- `VLLM_SM70_DSV4_PRIVATE_COMPRESSOR_STATE`;
- the Flash-V100 graph buckets.

One test fix rides along, in a module this PR opens for Turing.
`test_deepseek_v4_sm70_indexer.py` captures a graph on a side stream. It took
torch's current stream as the main one, which is the default stream until
vLLM's `current_stream()` has run. `vllm.utils.torch_utils` patches
`torch.cuda.set_stream`, so leaving the stream contexts records the default
stream as vLLM's current stream, and every later graph capture in the same
pytest process fails. One example is `TestCUDAGraphWrapper` in
`tests/v1/cudagraph/test_cudagraph_dispatch.py` ("CUDA graphs must be
captured on a non-default stream"). The test now takes vLLM's
`current_stream()`, the way the model code does.

Why one predicate instead of per-feature thresholds: in a pipeline every
stage has to declare the same attention backend and metadata plan. In our
fork we once let Volta stages take one impl and Turing stages another, and
the prefill stage lost its top-k indices across the stage boundary. With one
predicate, both generations run `V4_SM70_TRITON_SPARSE`.

Stacked on #745. The cache ops need its software FP8 on Turing, and #745 in
turn sits on #744. #714 rewrites the cuBLAS indexer decode condition that
this PR widens. That is a one-line overlap; whichever PR lands second takes
`is_device_capability_family(70)`.

Why this is not a duplicate. `gh pr list --state open --search` for
"turing deepseek", "sm75 deepseek", "SM75", "Turing", "wo_a", "is_bmm",
"V4_SM70_TRITON_SPARSE" and "sm70 route" finds:

- #742: MoE, unrelated files.
- #604 and #623: linears and FlashAttention on Turing, no DeepSeek-V4 path.
- #716: an opt-in SM70 sparse MLA variant; it merges cleanly with this PR.
- #714: as above.
- #327: auxiliary GEMVs.

None of them selects a DeepSeek-V4 route for Turing.

AI assistance was used for this change. Every line was reviewed and the tests
below were run by the submitter.

## Test Plan

```bash
pytest tests/models/test_deepseek_v4_sm70_routes.py
pytest tests/kernels/test_deepseek_v4_sm70_indexer.py
pytest tests/kernels/test_deepseek_v4_sm70_qnorm_rope_kv_insert.py
pytest tests/kernels/test_deepseek_v4_sm70_fp8_software.py
pytest tests/quantization/test_fp8_marlin_bmm_dequant.py
pytest tests/v1/spec_decode/test_dspark.py
# all of the above in one process, with the graph wrapper tests last
pytest <the files above> tests/v1/cudagraph/test_cudagraph_dispatch.py
pre-commit run --files <changed files>
```

End to end: DeepSeek-V4-Flash (MXFP4 experts, FP8 block linears, DSpark k=5)
at PP5 on 3x Tesla V100 + 2x Quadro RTX 8000, stage order RTX, V100, V100,
V100, RTX. The run used main plus our open PRs and this one, with
`--moe-backend sm70_skinny` (#742) and `VLLM_SM70_QUANT_BACKEND=auto`: V100
stages on TurboMind FP8 and grouped `wo_a`, RTX stages on Marlin FP8 and the
dequantized `wo_a`.

## Test Result

Unit tests, on main d3046986 plus #744 and #745:

| | Tesla V100 | Quadro RTX 8000 |
|---|---|---|
| all test files above, one process, cudagraph dispatch last | 88 passed | 88 passed |

Without the stream fix, running the indexer module before
`test_cudagraph_dispatch.py` in one process fails `TestCUDAGraphWrapper` on
the V100 (1 failed, 2 errors). On main the module skips on Turing.

Before this PR, the indexer and Q/KV insert modules skip on the RTX 8000.
Run with their gate opened on main, they pass there unchanged (12 passed).

End to end, PP5 as above:

| | SM70 route as is | with #716 (`VLLM_SM70_DSV4_SPARSE_MLA_BMM*=1`) |
|---|---|---|
| boot, all five stages on `V4_SM70_TRITON_SPARSE` | yes | yes |
| 35.8k-token prompt, cold prefill | 42.3 s | 16.7 s |
| decode step at 35.8k, DSpark k=5 | 224 ms | 100 ms |
| P(end of sequence) as first token, 5 probes | 0 | 0 |
| 8 reference prompts (facts, arithmetic, code, refusal) | all correct | all correct |

The answers were read in full; a greedy diff against our fork's output is
not meaningful here, because the V100 stages run other FP8 kernels. The
#716 column needs its prefill fix (9e6e1200 on #716). The first request
at a new prompt length waits 17-24 s while TurboMind tunes the FP8 GEMMs of
the three V100 stages for that M. The time is spent on the V100s only, in
TurboMind's dispatch, and it is unrelated to this PR.

Until now our fork ran its own pre-Hopper path for this model: all stages
on the Triton impl that ROCm uses, with batched-matmul attention added
there. We replaced it with this PR plus #716 after an A/B on the same
build, same dense FP8 path and the same PP5 rig, fresh prompts per run:

| | our fork's path | this PR + #716 |
|---|---|---|
| DSpark k=5: 35.8k cold prefill | 15.8 / 16.0 s | 15.2 / 15.3 s |
| DSpark k=5: decode step at 35.8k | 91 ms | 84-86 ms |
| DSpark k=7: 35.8k cold prefill | 21.0 / 21.2 s | 20.3 / 20.4 s |
| DSpark k=7: decode step at 35.8k | 85-86 ms | 80-81 ms |

All eight reference prompts were answered correctly in every run.

pre-commit clean.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
