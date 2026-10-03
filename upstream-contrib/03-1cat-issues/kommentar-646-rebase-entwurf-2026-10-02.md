Status: GESENDET 2026-10-02 17:51 nach Go von Peuqui — Force-Push #646 94416c613, #717 9851eb019 (mit Lease), Kommentar https://github.com/1CatAI/1Cat-vLLM/pull/646#issuecomment-5956073570 (Absätze zu Zeilen zusammengefügt).
Anlass: #646 und #717 kollidieren mit main 3e6d0a3c in nvidia/ple_layer.py (1Cats e9cd8ed5e aus
#766). Neuer Stapel lokal im Worktree 1Cat-vLLM-pr-plecascade: #646 94416c613, #717 9851eb019
(vorher ccabdd725 / f996501d2). Push mit
  git push --force-with-lease=qwen4exp-ple-tier-cascade-pr:ccabdd725 fork 94416c613:refs/heads/qwen4exp-ple-tier-cascade-pr
  git push --force-with-lease=ple-offload-pipeline-parallel:f996501d2 fork ple-offload-pipeline-parallel
Belege: range-diff (nur Konfliktstelle + Tests geändert, #717 unverändert), Rücksetz-Prüfung,
Testlauf V100 (GPU 1) und RTX 8000 (GPU 2) je Datei, pre-commit + mypy-3.10.

==================== Kommentar in 1Cat #646 ====================

Rebased #646 and #717 onto current main (3e6d0a3c). #766 changed the derived host budget in
_resolve_host_budget: with VLLM_SM70_QWEN38_HYBRID_PLE the whole table goes to host memory, and
the host cap counts min(TP, local world size) x local DP ranks. #646 replaces that function with
_device_spill_bytes, _cap_derived_host_budget and _plan_placement, so both changes are carried
into the cascade: the hybrid override applies to the derived host budget before the host cap,
and the cap uses the same rank count. #717 is unchanged.

tests/models/qwen4_exp/test_ple_auto_hybrid_budget.py now drives _plan_placement with the same
stubs and expectations, counted in rows; without the hybrid line its two hybrid cases fail. One
cascade test now sets the vLLM config around materialize_tables(), since the cap reads the
parallel config, as it does while the model loads.

Tests, one pytest process per file, on a Tesla V100 and a Quadro RTX 8000:
test_sm70_decode_graph 46 passed on both; test_ple 98 passed on the V100, 95 passed and 3
skipped on the RTX 8000; test_ple_auto_hybrid_budget 5, test_ple_cache_budget 7 and
test_ple_disk_shard_guard 1 passed on both; test_ple_offload_worker 35 passed, 1 skipped and 1
failed on both, test_offload_distributed_sets_config_only_for_model_parallel, which fails the
same way on main with a single visible GPU (#739). pre-commit and mypy-3.10 pass.
