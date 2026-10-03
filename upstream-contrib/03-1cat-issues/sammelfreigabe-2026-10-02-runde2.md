Status: GESENDET 2026-10-02 ~20:55 nach Go von Peuqui. Vor dem Senden alle drei Branches auf main
6ffba351 (#789 dazugekommen) umgesetzt, range-diff unverändert, Tests wiederholt (gleiche Zahlen;
#646 allein: test_ple_offload_worker 32 statt 35 bestanden, im Kommentar korrigiert). Während der
Tests kam #790 (a3498d46): alle 14 PRs mergen sauber, Env-Hooks auf dem Merge-Ergebnis bestanden.
(1) #793 eröffnet, Branch pr-ple-kv-estimate-block-size 7225de1d8. (2) Force-Push #646 94416c613 ->
4ce22ea65, #717 9851eb019 -> 2451f407f; Kommentar #646 issuecomment-5959281342 (Basis-Satz eine
Minute später präzisiert: „onto main (6ffba351; they also merge cleanly into the newer a3498d46)“).
(3) #674 aktualisiert: 14 offen, main a3498d46, #791/#793 unter „Still open“, #646-Hinweis.
Gesendete Fassungen: scratchpad runde2_pr_body.md, runde2_comment646.txt, issue674_runde2.md.
Autor der #646/#717-Commits ist noch peuqui@github.com (unzugeordnet), nicht umgeschrieben.

Sammelentwurf Runde 2, 2026-10-02 abends. Freigabe Peuqui.
Reihenfolge: (1) neuen PR „Blockgröße vor der PLE-KV-Schätzung“ eröffnen (Branch
pr-ple-kv-estimate-block-size, 6426edc99 auf main 589a5d1e), (2) Force-Push #646/#717 + Kommentar
mit der neuen PR-Nummer, (3) #674 mit beiden PR-Nummern und main-Hash. Absätze beim Senden zu
Zeilen zusammenfügen. Belege: ple_diag_2026-10-02.log (DIAG-Zeilen), ab_next_fn2_2026-10-02.log,
Testläufe dieser Sitzung (V100 GPU 4, ohne GPU), range-diff, Gegenproben.

==================== 1. Neuer PR ====================
Titel: [Bugfix][Qwen4Exp] Settle the hybrid block size before estimating the PLE KV budget

## Purpose

`kv_cache_bytes_for_max_model_len()` (#760) runs while the model loads, to decide how much of the
PLE table stays on the device. The worker settles the hybrid block size only after
`load_model()` (`update_block_size_for_backend` in `MultiprocExecutor`), so the estimate reads the
KV cache specs at the provisional block size. On Qwen3.8-Flash-Next with `--block-size 16` the
CSA compressor state needs 2240 bytes per page while a 16-token compressed page holds 1024, so
`get_kv_cache_groups` rejects the layout:

```
ValueError: CSA+linear layer 3 violates cache geometry.
```

Right after loading, the worker raises the block size to 1616 tokens, where the layout is valid.
589a5d1e keeps hybrid PLE away from the estimate; every other caller still gets the provisional
layout: the automatic host budget without hybrid mode, and a placement that measures the device
such as the cascade in #646.

This PR calls `current_platform.update_block_size_for_backend(vllm_config)` before the specs
are read. The worker's later call ends with the same block size: phase 1 picks the same backend
size again (or is skipped for an explicit `--block-size`), and phase 2 raises it to the same
aligned value.

How much the provisional layout was off, measured on Qwen3.8-Flash-Next PP4 (3x V100 + 2x Quadro
RTX 8000, 262,144-token context): on stage 0, which places the PLE table, the per-layer sum at the
provisional size reserved 1.714 GiB, the grouped layout at the final size needs 1.793 GiB; the
grouped layout exceeds the per-layer sum by 15 to 77 MiB per stage. Small for this model's
compressed attention, but it grows with every model whose KV cache is padded more.

Not a duplicate: no open PR touches `kv_cache_bytes_for_max_model_len` or
`update_block_size_for_backend` (gh pr list / gh search for "violates cache geometry", "PLE KV
budget", "update_block_size_for_backend").

AI assistance was used for this change. I reviewed every line and ran the tests below.

## Test Plan

```bash
pytest tests/models/qwen4_exp/test_ple_cache_budget.py   # one pytest process per file
pytest tests/models/qwen4_exp/test_ple_auto_hybrid_budget.py
pytest tests/models/qwen4_exp/test_ple.py
pre-commit run --files vllm/models/qwen4_exp/common/ple.py tests/models/qwen4_exp/test_ple_cache_budget.py
pre-commit run mypy-3.10 --hook-stage manual --files <same>
```

## Acceleration and benchmark contract (required for performance changes)

Not a performance change.

## Test Result

The new test checks that the estimate settles the block size before it reads the specs; it
passes with this change and fails on main. test_ple_cache_budget 8 and
test_ple_auto_hybrid_budget 6 passed on a Tesla V100 and without a GPU; test_ple 78 passed on the
V100 (68 passed, 10 skipped without a GPU). The existing budget tests use stand-in configs
without an attention backend, so their block size is taken as final. pre-commit and mypy-3.10
pass.

End to end on our fork (main 24994ba9 plus our open PRs, including the cascade of #646):
Qwen3.8-Flash-Next at PP4 and at TP2xPP2 failed to start with the error above; with this change
both start, and the greedy output matches our production reference [TP2-ERGEBNIS NACHTRAGEN].

🤖 Generated with [Claude Code](https://claude.com/claude-code)

==================== 2. Force-Push #646/#717 + Kommentar in #646 ====================
#646: qwen4exp-ple-tier-cascade-pr 94416c613 -> fde9f1325
#717: ple-offload-pipeline-parallel 9851eb019 -> 250bb73fb (beide auf main 589a5d1e)
  git push --force-with-lease=qwen4exp-ple-tier-cascade-pr:94416c613… fork fde9f1325:refs/heads/qwen4exp-ple-tier-cascade-pr
  git push --force-with-lease=ple-offload-pipeline-parallel:9851eb019… fork ple-offload-pipeline-parallel

Rebased #646 and #717 onto main (589a5d1e). 589a5d1e keeps hybrid PLE away from the provisional
device/KV estimate; the cascade now does the same: with VLLM_SM70_QWEN38_HYBRID_PLE and no explicit
host budget it neither measures the device nor estimates the KV cache, and keeps the whole table
on host within the host cap. Your new test is ported to _plan_placement and fails without that
branch. With the cascade itself (VLLM_QWEN4EXP_PLE_DISK) the device is still measured, which on
Qwen3.8-Flash-Next needs #NNN: without it the grouped estimate of #760 reads the provisional
block size and rejects the layout. Following #782, VLLM_QWEN4EXP_PLE_DISK and
VLLM_PLE_DISK_RELEASE_PAGES are registered with env_var() and the reference is regenerated. #717
is unchanged.

Tests on a Tesla V100, one pytest process per file: test_ple 98, test_ple_auto_hybrid_budget 6,
test_ple_cache_budget 7, test_ple_disk_shard_guard 1, test_sm70_decode_graph 46, test_envs 58 and
test_check_env_registration 8 passed; test_ple_offload_worker 35 passed, 1 skipped, 1 failed (the
single-GPU case of #739). pre-commit and mypy-3.10 pass.

==================== 3. #674 ====================
Wie im ersten Sammelentwurf (Teil 5), zusätzlich:
- im Update-Absatz: „#646 and #717 are rebased onto 589a5d1e, and #NNN settles the hybrid block
  size before the PLE KV estimate; with the cascade, Qwen3.8-Flash-Next needs it to start. #791
  fixes the DFlash/DSpark start that #748 broke.“ — und die Zahl offener PRs neu zählen.
- unter „Still open“: #791 und #NNN (neu) eintragen, bei #646 „needs #NNN on current main“.
- main-Hash beim Senden.
