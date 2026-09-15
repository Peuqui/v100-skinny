# PR-Entwurf: [Bugfix][Qwen4Exp] Allow checkpoint FP8 MTP experts under pipeline parallelism

Status: ENTWURF 14.09.2026, nicht gesendet. Worktree `1Cat-vLLM-pr-fp8mtppp`,
Branch `qwen4exp-fp8-mtp-experts-pp` auf origin/main 02c87ab8, gestaged, nicht
committet, nicht gepusht. Fork-Stand: work-main d228c725 (Tag verified-2026-09-14c
auf 914333d6 + d228c725 Sync).

Titel: [Bugfix][Qwen4Exp] Allow checkpoint FP8 MTP experts under pipeline parallelism

## Purpose

`nvidia/Qwen3.8-Flash-Next-NVFP4` ships its MTP routed experts as ModelOpt
`FP8_PB_WO` blocks. Since fbdd1700 those load natively on SM70 through
`MTPExpertFp8Config`, but `_make_draft_vllm_config` rejects the draft whenever
`pipeline_parallel_size != 1`:

    ValueError: MTP FP8 experts require SM70, FP16, an AWQ/ModelOpt/FP8 draft
    checkpoint, tensor parallelism without PP or EP, and standard rejection sampling

The design note `docs/design/sm70_awq_fp8_mtp_experts.md` records this as the
scope of the first validation (TP4), not as a kernel constraint. Under pipeline
parallelism the drafter is stage-local: the V2 model runner creates the
speculator only when `is_last_pp_rank` (`vllm/v1/worker/gpu/model_runner.py`,
`init_speculator`), and the FP8 expert padding is computed from the
tensor-parallel size of that stage's `moe_parallel_config`. The same stage-local
property was the basis of #573.

The change:

- moves the support conditions into `_mtp_fp8_experts_supported` so they can be
  tested without a full `VllmConfig`;
- accepts checkpoint-native FP8 experts under pipeline parallelism;
- keeps `pipeline_parallel_size == 1` for online conversion
  (`mtp_expert_quantization="fp8"`), which I could not validate under PP;
- updates the error message and the design note accordingly.

SM70, FP16, supported draft quantization, no expert parallelism and standard
rejection sampling are still required. Configurations with PP=1 behave exactly
as before.

Not a duplicate: #553 converts the same nvidia MTP experts to resident FP16 for
TP4 and does not touch pipeline parallelism; the native FP8 path it predates
(fbdd1700) is what this PR extends. Searched open and closed PRs and issues for
"MTP FP8 experts pipeline", "mtp_fp8", "FP8 MTP", "pipeline parallel MTP",
"Flash-Next nvidia".

## Test Plan

    .venv/bin/python -m pytest tests/models/qwen4_exp/test_mtp_fp8_experts.py \
        tests/models/qwen4_exp/test_mtp_fp8_checkpoint.py \
        tests/models/qwen4_exp/test_mtp_stage_local.py
    pre-commit run --files <changed files>
    pre-commit run mypy-3.10 --hook-stage manual --files <changed Python files>

End to end on 2x Quadro RTX 8000 (pipeline stage 0) + 2x Tesla V100 (stage 1,
the drafter stage), TP2 x PP2, partition 24/24, `nvidia/Qwen3.8-Flash-Next-NVFP4`
revision fc694b54 loaded straight from its Hugging Face snapshot, fp16,
`--speculative-config '{"method":"mtp","num_speculative_tokens":4,"draft_sample_method":"greedy"}'`,
chat endpoint with thinking enabled, temperature 0, seed 1.

## Test Result

- Tests: 64 passed. New cases: checkpoint experts accepted at PP 2 and 5, online
  conversion accepted at PP 1 and rejected at PP 2, expert parallelism, bf16,
  non-SM70, unsupported quantization and synthetic rejection still rejected.
  With the former `pipeline_parallel_size == 1` condition restored, the two
  checkpoint PP cases fail.
- pre-commit and mypy-3.10 pass.
- Before: the engine refuses to start with the ValueError above on both
  last-stage workers.
- After: the engine boots and serves. The last stage loads 23.28 GiB, 0.70 GiB
  more than the same checkpoint with an NVFP4 MTP block transplanted in its place
  (22.58 GiB), which is the FP8 block with its padding. Two of the prompts produce
  byte-identical text with the FP8 head and with the transplanted NVFP4 head in
  three boots, as expected for a lossless drafter. Over nine prompts behind a
  13k-token context the decode rate is 62.6 to 78.2 tok/s with an acceptance
  length of 3.18 to 3.97; a 52k-token prompt prefills at 691 tok/s.

These end-to-end runs were made on our fork (1Cat main 80c88e8d plus our open
PRs); the checkpoint also needs the PLE detection fix submitted separately,
because it does not set `ple_embedding_dtype`.

AI assistance (Claude) was used for this change and this description. I reviewed
every changed line and ran the tests and measurements above.
