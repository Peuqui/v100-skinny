# PR-Entwurf: SWA-Builder nimmt dieselbe Decode-Schwelle wie der Sparse-MLA-Builder

Worktree: `1Cat-vLLM-pr-swathreshold`, Branch `sparse-swa-spec-threshold`
auf `origin/main` (fe67339d). Stand 12.09. früh: Änderung fertig, pre-commit
und mypy-3.10 grün, NICHT committet, NICHT eröffnet. Als Overlay auf
work-main angewendet (`handover/2026-09-11/patches/sparse_swa_spec_threshold.diff`).

## Commit-Message

```
[Bugfix][DeepSeek-V4] Align the SWA decode threshold with the sparse MLA builder

Under parallel drafting (DSpark, DFlash) the sparse MLA builder derives
its decode threshold through _init_reorder_batch_threshold, which counts
1 + 2 * num_speculative_tokens (11 for k=5). The sliding-window builder
added num_speculative_tokens to its own threshold instead (6 for k=5).
A request with 7..11 query tokens - a short prompt, or the uncached tail
of a prefix-cache hit - was therefore a decode for the C128A metadata
(no prefill top-k indices built) and a prefill for the SWA metadata,
and the sparse attention asserted on the missing indices, killing the
engine. The SWA builder now takes its threshold from the same helper.

Co-authored-by: Claude <noreply@anthropic.com>
Signed-off-by: Peuqui <peuqui@github.com>
```

## PR-Titel

`[Bugfix][DeepSeek-V4] Align the SWA decode threshold with the sparse MLA builder`

## PR-Body

## Purpose

DeepSeek-V4-Flash with DSpark (`num_speculative_tokens=5`) kills the engine
on a chat prompt of 7 to 11 tokens after the template, and on any
prefix-cache hit that leaves 7 to 11 uncached tokens (typically the second
turn of a tool call):

```
File "vllm/models/deepseek_v4/amd/rocm.py", line 788, in _forward_prefill
    assert topk_indices is not None
AssertionError
```

followed five minutes later by `RPC call to sample_tokens timed out` and
`EngineDeadError`. Prompts of 4 to 6 tokens and of 12 tokens and more work.

Three metadata builders decide the decode/prefill split for one forward,
with two different thresholds:

- `FlashMLASparseMetadataBuilder` (C128A) calls
  `_init_reorder_batch_threshold(1, supports_spec_as_decode=True)`, which
  under parallel drafting counts `1 + 2 * num_speculative_tokens`
  (`backend.py`), i.e. 11 for k=5. DSpark sets `parallel_drafting = True`
  (`config/speculative.py`).
- `DeepseekSparseSWAMetadataBuilder` computed
  `reorder_batch_threshold + num_speculative_tokens`, i.e. 6, with a comment
  saying it must match the other builders.

A request with a query length in (6, 11] is a decode for the C128A builder
(`treat_short_extends_as_decodes`), so it builds no
`c128a_prefill_topk_indices`; for the SWA builder it is a prefill, so the
sparse attention takes `_forward_prefill` and asserts on the missing
indices. The window is exactly the gap between the two thresholds.

The SWA builder now takes its threshold from the same helper. Without
speculation nothing changes (both were 1); with MTP-style drafting
(`parallel_drafting=False`) nothing changes either (both were 1 + k). The
indexer builder (`indexer.py`) still adds `num_speculative_tokens`; it is
not part of this assert path and is left as is — happy to align it in the
same PR if wanted.

## Test Plan

1. `pre-commit run --files vllm/v1/attention/backends/mla/sparse_swa.py` and
   the `mypy-3.10` hook (`--hook-stage manual`).
2. Hardware, 2× Quadro RTX 8000 + 3× Tesla V100, DeepSeek-V4-Flash NVFP4,
   TP1 × PP5, DSpark k=5, chat prompts with exactly 4..16 tokens after the
   template (each prompt in a fresh request, `max_tokens=8`), before and
   after the change. Before: two boots, descending from 16 and ascending
   from 4, each stopped at the first crash. After: one boot, all lengths,
   then a two-round tool call (call, then the tool result as a `tool`
   message, with prefix caching).

## Test Result

1. All applicable hooks passed; mypy-3.10 passed.
2. Before: 16, 15, 14, 13, 12 OK, **11 assert**; 4, 5, 6 OK, **7 assert**
   (8 asserted three times in earlier runs). After: all thirteen lengths
   4..16 answer, no assertion in the worker log. Tool call with the tool
   result returned as a `tool` message (prefix-cache hit leaving 8 uncached
   tokens, the case that asserted before): the call is parsed
   (`get_weather`, `{"city": "Hamburg", "unit": "celsius"}`) and the second
   turn answers in 5.3 s with the returned values, no assertion.

## Not a duplicate

Checked on 2026-09-12 against `1CatAI/1Cat-vLLM`: `gh pr list --state open
--search` for "sparse_swa", "decode threshold speculative",
"parallel_drafting" and `gh issue list --search "topk_indices"` return no
PR touching this; `gh pr diff --name-only` over every open PR shows none
touching `sparse_swa.py` or `deepseek_v4/amd/rocm.py`. Related but
different: #208 (indexer decode workspace OOM and a shape assert in
`combine_topk_swa_indices` under 32-sequence load), #205 (prefix-cache hit
rate 0 with DSpark, an upstream SWA mask bug), #597 (DSML tool-call token
selection and decode speed on 8× V100).

AI assistance (Claude) was used to measure the window and trace the two
thresholds; I reviewed every line and ran the tests above.
