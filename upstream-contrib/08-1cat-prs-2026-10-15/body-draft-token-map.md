## Purpose

An MTP drafter that shares the target's lm_head projects every draft token onto the full vocabulary (248,320 rows on Qwen3.8-Flash-Next). On bandwidth-bound cards that head, not the drafter layer, dominates the draft step. This PR adds `speculative_config.draft_token_map`: a list of target token ids the drafter may propose (a JSON list or SGLang's `--speculative-token-map` `.pt`; EOS ids are always included). The drafter's shared head is restricted to those rows; the target still verifies every token with its full head.

It is the fixed-list part of vllm-project/vllm#59740 by @stecasta as of 2026-10-02 (the later dynamic rows are not included), ported to this fork's `EagleSpeculator` (`_sample_draft` is the single drafting point). Adaptations: `VocabParallelEmbedding` records `tp_rank`, the draft head declines the SM70 fused top-1 path so the generic local argmax handles its padding rows, and the tests stub the TP group the heads read at construction. Credit: @stecasta (Co-authored-by on the commit).

The field is opt-in (default `None`); without it nothing changes, so the full-vocabulary MTP4 defaults qualified in #821 stay as they are. It requires method `mtp` and an unquantized lm_head shared with the target.

Not a duplicate: gh pr list --state all --search "draft_token_map", "draft vocabulary", "sparse LM head", "59740", "reduced draft" finds #922 (open draft), which builds a CJK-complete FP8 draft vocabulary for the DFlash2 head (qwen3_dflash2.py, dflash/utils.py); this PR is for MTP drafters through EagleSpeculator and touches neither file.

AI assistance was used for this change. I reviewed every line and ran the tests and measurements below.

## Test Plan

```bash
pytest tests/v1/worker/test_gpu_draft_vocab.py tests/lora/test_layers.py tests/v1/spec_decode/test_ddtree_payload.py tests/v1/spec_decode/test_dspark.py tests/v1/spec_decode/test_sm70_mtp_head_scope.py
pre-commit run --from-ref origin/main --to-ref HEAD
```

Serving on our fork (main 6f3557c97 plus our commits, production commit baaa5a687): Qwen3.8-Flash-Next-180B NVFP4 + MTP (k=4), PP4 over Quadro RTX 8000 / Quadro RTX 8000 / Tesla V100 / Tesla V100, greedy drafting. The list has 98,304 ids, ranked by frequency in model-generated German/English text and code plus German running text and Python sources (coverage on held-out real sessions 99.8 %). Arms: the production entry with the list, and the same entry without `draft_token_map`, alternating boots (off, off, on, off, on). Per boot: four 29k-token prompts, short prose, code and a 1.9k-token edit task at production sampling, and three greedy reference prompts.

## Path coverage

| Path / provider | Format | Key conditions (shape, TP/EP, graph, capability) | Expected selection / fallback | Evidence (source / static / executed / measured) | Unverified scope |
| --- | --- | --- | --- | --- | --- |
| MTP draft head | FP16 lm_head, list rows | method mtp, `draft_token_map` set, PP4, CUDA graphs | restricted head, generic local argmax | executed + measured (serving), unit tests | quantized lm_head (rejected at config) |
| MTP draft head | FP16 lm_head | `draft_token_map` unset | full head, unchanged | executed (off arm) | none |

## Acceleration and benchmark contract (required for performance changes)

- Default enabled or opt-in: opt-in (`speculative_config.draft_token_map`).
- Required CLI options and environment switches: `--speculative-config '{"method":"mtp",...,"draft_token_map":"<list>"}'`.
- KV cache dtype used for the benchmark: FP16 (model default for this entry).
- Wheel SHA or source commit: our fork at baaa5a687 (main 6f3557c97 + our commits).
- PYTHONPATH, source overlays, or external native libraries used: none for the production entry; native build of the fork tree.
- User-entry route-hit, speed, and output-quality evidence: see Test Result.
- Existing kernel/backend registry and capability-rejection reasons: the restricted head declines the SM70 fused top-1 path.
- Per-engine configuration and deprecated-variable compatibility: speculative config field, no environment variable.
- New/changed environment metadata and generated reference check: architecture reference regenerated.
- Route snapshot command and intentional changed rows: none.
- SM70 registered variables / unregistered reads / model-parameter restrictions, before -> after: unchanged.

## Test Result

- On a Tesla V100 and on a Quadro RTX 8000: tests/v1/worker/test_gpu_draft_vocab.py passes; with the other four files 120 passed, 63 skipped, 1 failed. The failure (`test_sm70_mtp_head_scope.py::test_default_head_preparation_precedes_graph_mode_guard`, a fake without `sm70_draft_hot_vocab_ids`) fails the same way on main 6f3557c97. pre-commit passes.
- Mean over boots, tok/s / decode step, off (3 boots) -> on (2 boots): prose 41.8 / 59.0 ms -> 45.7 / 54.2 ms (+9 %), code 71.1 / 58.3 ms -> 77.3 / 54.8 ms (+9 %), edit 85.6 / 58.2 ms -> 92.7 / 54.0 ms (+8 %); at 29k context 37.1 / 62.9 ms -> 37.7 / 62.4 ms. Acceptance per round unchanged (prose 2.47 -> 2.50, code 4.18 -> 4.23, edit 4.99 both).
- Greedy texts are stable within each arm across boots. Between the arms, 1 of 3 is identical and 2 differ at a near-tie (e.g. "nutzbare mechanische Energie" vs "nutzbare Antriebsleistung"): the target still verifies every token with its full head, but a different acceptance pattern changes how positions are batched, and FP16 near-ties can flip, as between boots of different compile artifacts. Both versions read as correct (checked by hand).
- Smaller lists (65,536 and 49,152 ids) saved about 1 ms per step but lowered acceptance on code and long context, so 98,304 was kept for this measurement.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01Ly7LM3m9fa6DRJPYQjnt95
