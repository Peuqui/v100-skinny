# PR-Entwurf: [Bugfix][Qwen3.5] Keep the MTP drafter stage-local under pipeline parallelism

Status: **GESENDET 14.09.2026 als #636** (Go Peuqui) — https://github.com/1CatAI/1Cat-vLLM/pull/636,
Branch `qwen3-5-mtp-pp` (Worktree `1Cat-vLLM-pr-mtppp`), Commit 6aa0dd29 auf origin/main 80c88e8d
(konfliktfrei gegen a6f5e834). Overlay in work-main trägt dieselbe Datei.
Duplikatprüfung: 1Cat offen "qwen3_5 mtp" leer; vllm-project offen "Qwen3_5MTP
pipeline parallel" nur #37429 (hybrider KV-Cache, andere Sache).
Vorbild: unser gemergter #573 (Qwen4Exp, gleiches Muster).

Titel: [Bugfix][Qwen3.5] Keep the MTP drafter stage-local under pipeline parallelism

## Purpose

Qwen3.5-family targets (Qwen3.5, Qwen3.6, Qwen3.8 dense and MoE) with
`--speculative-config method=mtp` and `--pipeline-parallel-size` greater than one
do not boot. There are three stacked causes, each hiding the next:

1. `Qwen3_5MTP` does not implement `SupportsPP`. The draft model config is
   verified against the parallel config, so the engine refuses to start:

       NotImplementedError: Pipeline parallelism is not supported for this model.
       Supported models implement the `SupportsPP` interface.

2. `Qwen3_5MultiTokenPredictor.forward` branches on `get_pp_group().is_first_rank`,
   which is the TARGET model's pipeline position. The drafter only runs on the
   last rank (`execute_model` returns the IntermediateTensors on every earlier rank
   before speculation), so it took the "receive from the previous stage" path and
   asserted on intermediate tensors that nobody sends. This is the same defect
   #573 fixed for Qwen4Exp.

3. With the default `VLLM_QWEN35_MTP_SHARE_IO_WEIGHTS=1` the drafter builds its
   embedding as a `PPMissingLayer` and expects the loader to share the target's.
   Both `load_eagle_model` (V2) and `_maybe_share_embeddings` (V1) deliberately
   skip embedding sharing under PP, because the target's embedding only exists on
   the first rank. The placeholder was therefore never replaced, and the first
   compile on the last rank failed with

       torch._dynamo.exc.Unsupported: Assertion failed on symbolic shapes
         assert hidden_states.shape[-1] == inputs_embeds.shape[-1]

Changes:

- `Qwen3_5MTP` implements `SupportsPP` and forwards
  `make_empty_intermediate_tensors` from the predictor, as `Qwen4ExpMTP` does.
- The predictor's forward drops both pipeline branches: it always embeds locally
  and never returns IntermediateTensors for a next stage it does not have.
- Under PP the drafter owns its embedding (loaded from the checkpoint's
  `embed_tokens`) instead of waiting for a share that the loaders skip. With a
  single pipeline rank nothing changes: sharing stays as before. The LM head is
  still shared, since the target's head lives on the drafter's rank.

Single-rank behaviour is unchanged: there `is_first_rank` and `is_last_rank` are
both True, the removed branches were dead, and `world_size == 1` keeps the shared
embedding.

## Test Plan

    .venv/bin/python -m pytest tests/models/qwen3_5/test_mtp_stage_local.py -v
    pre-commit run --files <changed files>
    pre-commit run mypy-3.10 --hook-stage manual --files <changed files>

End to end on 2x Quadro RTX 8000 (TP1 x PP2), RadixArk/Qwen3.8-27B-NVFP4,
`--speculative-config '{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"greedy"}'`,
fp16, V2 model runner, 400 tokens at temperature 0, five repetitions, plus the same
prompt on TP2 with a DFlash2 draft as a text reference.

## Test Result

- New tests: 3 passed. With the model file reverted to main: 3 failed
  (AssertionError on the intermediate tensors, missing SupportsPP).
- main, TP1 x PP2: the engine refuses to start with the NotImplementedError above.
- With causes 1 and 2 fixed but the embedding still shared: boot fails on the last
  rank with the dynamo assertion quoted above.
- This PR, TP1 x PP2: boots in 105 s and serves. 61.05 tok/s median (61.00 to
  61.10), acceptance length 2.963. The 400-token greedy text is byte-identical to
  the TP2 reference (sha256 prefix `0106659946c064b1`), so the drafter changes
  acceptance, never the output.

AI assistance (Claude) was used to write this change and this description. I
reviewed every changed line and ran the tests and measurements above.
