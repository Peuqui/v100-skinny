Entwürfe 2026-10-02 abends, NICHT gesendet (Freigabe Peuqui abwarten). Ersetzen die Entwürfe für
#767/#768 (kommentar-767-768-gpu-entwurf-2026-10-02.md): 1Cat hat #767/#768/#770/#771 im
Sammel-PR #773 gemergt, #744/#745 stecken im offenen #777.
Belege: tasks/bcyea07pz.output (main-Spalte, main 3e6d0a3c, V100 GPU 3 / RTX GPU 0),
tasks/bci95rig5.output + Skip-Zählung (#777 auf V100 GPU 4 / RTX GPU 2), Journal-Auswertung der
27B-Warmstarts (27.09.–01.10.), range-diffs, DSv4-PP5-Lauf ab_next_ds_2026-10-02.log.
Vor dem Senden: Absätze zu Zeilen zusammenfügen (GitHub bricht sonst jede Zeile um).

==================== Kommentar in 1Cat #773 ====================

GPU results for this merge on our rig (3x Tesla V100 + 2x Quadro RTX 8000), on main 3e6d0a3c
with native extensions from our fork's build (main d3046986 plus our open PRs; main's csrc is
unchanged since then), one pytest process per file.

The modules this PR touches pass natively on both cards: test_pp_token_check 17,
test_sm70_fp8_workspace_aot_reload 6, test_sm70_fp8_prefill_exact_dense 24,
test_sm70_fp8_qpn8_pp2_tp4 13, test_sm70_turbomind_adapter 13, test_qwen3_5_quantization 14,
test_qsa_xqa_page4_workspace 5 and its CPU variant 7, test_sm70_compressed_tensors_fastpaths 5,
test_sm70_ct_fp8_cache_workspace 1, test_spec_token_aliasing 2 and test_envs 58 passed.
test_sm70_mxfp4_moe passed 33 on the V100, including the repack allocation bound (28 passed and
the 5 V100-only cases skipped on the RTX 8000); test_sm70_flash_v100_multihead passed 70 on the
V100 (31 passed, 39 SM70-only skipped on the RTX 8000).

Cold/hot AOT: the workspace commit is unchanged from #710, and our fork has carried it since
2026-09-27. Since then Qwen3.8-27B-NVFP4 (TP2 with MTP, compile cache on) started warm from its
AOT artifacts on 2x Quadro RTX 8000 and served requests 19 times without the stale-pointer
error; earlier that day, before the change, four warm starts failed with "The specified pointer
resides on host memory and is not registered with any CUDA device." Its FP8 layers reach these
sites on Turing through the QPN8 route of #604, so those runs are our fork with #604 on top.

[PP-PRÜFUNG UNTER ECHTEM MEHRSTUFIGEM BETRIEB: DSv4-PP5-Ergebnis nachtragen — Zahl der
Anfragen/Schritte, keine "PP spec decode: invalid"-Meldung, Greedy/Qualität, Tempo gegen
Produktion]

==================== Kommentar in 1Cat #777 ====================

Native run of this branch (344bb9eb) on a Tesla V100 and a Quadro RTX 8000, one pytest process
per file, against main 3e6d0a3c, with the same extensions as above:

- test_compressor_kv_cache: 32 passed, 6 skipped on both cards; main fails 21 on the V100 and
  35 on the RTX 8000.
- test_deepseek_v4_fp8_capability: 16 passed on both.
- test_deepseek_v4_mega_moe: 3 passed, 1 skipped on both; main fails 1.
- test_fused_deepseek_v4_qnorm_rope_kv_insert, test_fused_indexer_q_rope_quant and
  test_fused_inv_rope_fp8_quant: all 139, 40 and 50 cases skip on both cards with their stated
  reasons (the fused op refuses pre-Ampere; Triton's fp8e4nv needs sm_89); main fails all of
  them.

git range-diff shows the #744/#745 commits unchanged apart from your sign-off.
