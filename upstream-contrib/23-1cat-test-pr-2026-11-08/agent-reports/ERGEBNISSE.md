# SM70-Testreihe 08.11.2026: Korrekturen der 78 roten Dateien (Stand 11:35, unfertig)

Reihe: 458 SM70-Testdateien von 1Cat main c98a8d664 + #1421 (ohne `tests/models`), V100, Standard-Triton 3.6.0, torch 2.10.0+cu128.
380 grün, 78 rot (681 rote, 10 837 grüne Fälle). Logs `~/.cache/bench-scripts/gputests_sm70_sweep_2026-11-08/` (+ `card{1,3,4}/`),
Übersicht `../sm70-sweep-2026-11-08-failures.txt`. `test_gpu_model_runner_v2.py` repariert unser offener #1349 → 77 Dateien.

Korrekturen liegen UNCOMMITTET im Baum `1Cat-vLLM-pr-sm70-tests-2` (Zweig `pr-sm70-stale-tests-2` = main c98a8d664, Bibliotheken aus
main-check), nur unter `tests/`. Fünf Helfer-Agenten, Gruppenlisten `group_*.txt` hier. Peuqui 08.11.: korrigieren, nicht löschen;
Hardware-Unmögliches begründet überspringen; echte Fehler im Code beheben.

## Stand je Gruppe
- **B1 GGUF/AWQ/INT8 (16, fertig):** 13 veraltet, 3 hardwareunmöglich (INT8-`tl.dot` im Standard-Triton erst ab sm80 →
  `moe/test_block_int8.py`, `quantization/test_int8_kernel.py` Skip ohne Fähigkeit 80; `quantization/test_block_int8.py` nutzt
  fp8e4nv → Skip ohne 89). Die „policy-argument ABI“-Meldung war eine Attrappe (ganz `torch.ops._C` ersetzt). Diff auch in
  scratchpad `b1/group_B1.diff` (vergänglich). GPU nötig: `test_gguf_device_transcode` [*-29], `test_gguf_dmv::test_iq2_plane_policy_changes_graph_hash`,
  `test_gguf_segment_prefill` (Genauigkeit dense bei M=33, GEMM bei M=65), `test_gguf_turbomind_model`, `test_gguf_turbomind_model_moe::test_mixed_canonical_experts_tp4_and_graph`,
  `test_gguf_packed_embedding::test_gpu_packets_match_cpu_and_changed_graph_ids`.
- **B2 übrige Quantisierung (13, fertig):** 12 veraltet/hardwarebedingt. `test_sm70_qpn8_block_fp8` (Referenz ohne gebundene Policy →
  anderer Dispatch-Cache) und `test_sm70_dflash_context_reduction` (FAST_TARGETS nur einmal je Prozess gelesen) sind KEINE Rechenfehler.
  `test_compressed_tensors`: Min-Capability-Marker je Schema (13 Fälle Skip). GPU nötig: `test_deepseek_v41_tail`,
  `test_sm70_dflash_context_reduction` (ganze Datei), `test_sm70_qpn8_block_fp8` (ganze Datei), `test_compressed_tensors`, `test_fp8`.
- **C Spec-Decode/Sampling/Worker (21, fertig):** alle veraltet, kein echter Fehler (`test_executor`, `test_mtp`: Attrappen ohne Felder,
  die eine echte VllmConfig hat). GPU nötig: `test_sm70_sampler_graph_branch`, `test_sm70_mtp_safety::test_prepare_inputs_padded_does_not_alias_target_seq_lens`,
  `test_rejection_sampler`, `test_sampler`, `test_gpu_input_batch` (cuda:0-Fall), `test_gpu_model_runner`, `test_qwen4_exp_v2`, `test_executor` (custom_executor).
- **D Attention/GDN (13, fertig):** veraltet bzw. Hardware/Umgebung, kein Funktionsfehler. `test_sm70_prefill_score_capacity`: seit
  #797 lädt der Import FA2 nicht mehr (1217bade4 reparierte drei Geschwister, diesen nicht). `test_sm70_tp2_e4m3_scalar_fast`: nur ninja
  (Wrapper behoben, Test unverändert). GPU nötig: prefill_score_capacity, tp2_e4m3_scalar_fast, triton_decode_attention (32 Skip),
  fused_sigmoid_gating_delta_rule, sm70_gdn_mixed_spec_verify, sm70_mtp_gdn_mixed_qkv, sm70_qsa_page4_plan.
- **A Konfiguration/Richtlinien (14): UNFERTIG**, Helfer um 11:33 gestoppt (Kontingent), kein Bericht. 12 Dateien schon geändert,
  `test_sm70_fp8_kernel_selection.py` und `test_sm70_native_policy.py` noch unverändert. Änderungen ungeprüft.

## Echte Fehler im Nicht-Test-Code (noch nicht behoben)
1. **SM70-Marlin lehnt Bias ab, FP8-Marlin übergibt ihn:** `vllm/model_executor/layers/quantization/utils/marlin_utils_fp8.py:67–72`
   gibt `b_bias=bias` an `ops.marlin_gemm`, `csrc/quantization/marlin/sm70_marlin_dispatch.cu:510–512`
   `TORCH_CHECK(!has_bias, "SM70 Marlin does not support bias. TODO…")`. Jedes FP8-Modell mit Linear-Bias (z. B. OPT-125m) startet auf
   der V100 nicht (`tests/quantization/test_fp8.py::test_online_quantization`, 4 Fälle, Test bewusst rot gelassen). Unsere
   Produktionsmodelle nicht betroffen (eigene FP8-Wege). Lösung: Bias auf SM70 nach dem GEMM addieren (oder Epilog).
2. **Falsche Doku:** `vllm/config/cache.py:86–88` sagt fp8 → fp8_e5m2 auf SM70 Flash-V100; der Code (`vllm/engine/arg_utils.py:141–165`,
   seit e6be041d6) macht fp8_e4m3.
3. **1Cats eigene Komplexitätsgrenze verletzt** (`tests/v1/attention/test_flash_v100_trace_golden.py::test_dependency_and_coupling_ratchet`):
   größte Funktion 206 > 200 (`dense_prefill.py try_sm70_fa2_d256_prefill`, #1287/#1298), `impl.py forward` 120 > 113 (#1287),
   cross_module_private 12 > 11 (#1298: `prefill.py` liest `_dense_prefill._SM70_79T_Q4096_MIN_QUERY_LEN`).

## Entscheidungen (Vorschlag in Klammern, Peuqui noch nicht gefragt)
- Ratchet: Code umbauen statt Grenze anheben (eigener PR, Laufzeitcode, GPU-Tests).
- Methoden-Hash-Fixtures (`test_flash_v100_impl_composition`, `test_sm70_grouped_codecs`): Helfer hat geänderte Methoden ausgenommen;
  (lieber Hashes in `fixtures/flash_v100_impl_methods.json` erneuern, Präzedenz 8c96e32e5 vorher ansehen).
- `test_triton_decode_attention` D_QK=576-Fall: Skip nur SM70 (lieber: Skip, wenn shared_memory_per_block_optin < Bedarf; trifft RTX und A100 richtig).
- INT8-Skips greifen auch auf der RTX (Standard-Triton kann es dort nicht) (so lassen).
- WNA16 in test_compressed_tensors auf SM70 nur mit Marlin/TurboMind-Freischaltung (Skip ohne Freischaltung; Alternative: im Test freischalten).
- Wirkungslose envs-Patches in heute grünen Fällen (`test_dflash2.py::test_selector_default_path_does_not_allocate_sparse_score_cache`,
  `test_sm70_long_attention_graphs.py` Z. 149/219) (in wirksame Form bringen, toter Code).
- `test_release_cleanup.py`: zusätzliche Identitäts-Assert für den Modulimport (prüfen, ob `# noqa: F401`-Import sauberer).
- Zu prüfen: `test_dflash2.py`, laut Helfer „Alignment-Schalter jetzt Umgebungsvariable“ – widerspricht 1ca9ea9f8 (Speculator-Trace); vor Übernahme lesen.
- Geprüft und ok: `test_gpu_input_batch.py` ruft `functools.partial` in `_compare_objs` auf; `bind_output_token_repair` ist ein reiner Wertzugriff (`policy.value`).
- Abdeckungslücke: Dense-Permute aus #1369 hat im fp4-mainflow-Vergleich keinen Aufrufreihenfolge-Test mehr.

## Kartenprüfung der Gruppen B1, B2, C, D (63 Dateien)
Start 11:32 auf V100 1/3/4 (Einheiten `sm70-fix-c{1,3,4}`, Wächter `sm70-fix-guard`), Logs `~/.cache/bench-scripts/gputests_sm70_fixcheck_2026-11-08/card*/summary.log`.
Danach: rote Fälle EINZELN wiederholen (Parallelartefakte ausschließen), Gruppe A fertigstellen und ebenso prüfen, Diffs durchsehen,
thematisch committen (Konfiguration, Quantisierung, Spec-Decode/Worker, Attention), PRs nach AGENTS.md (Vorlage #1421).

## Kartenprüfung erledigt (08.11. 11:32–11:36 und 11:48–11:58, V100 1/3/4)
Alle 77 korrigierten Dateien gelaufen (Logs `gputests_sm70_fixcheck_2026-11-08/card*/` und `rest-card*/`). Gruppen B1/B2/C/D: 61 von 63
grün; rot nur erwartet: `test_fp8.py` 4 (echter Fehler Marlin-Bias, Test bewusst rot), `test_flash_v100_trace_golden.py` 1 (Ratchet,
Entscheidung). Gruppe A: 11 von 14 grün; rot `compile/test_config.py` 5 (vorher 6), `test_sm70_fp8_kernel_selection.py` 1 und
`test_sm70_native_policy.py` 1 (beide vom Helfer noch nicht bearbeitet). Diese Läufe waren parallel zu dritt; rote vor der Wertung einzeln
wiederholen. Offen bleibt: Gruppe A fertig, alle Diffs selbst lesen, die Entscheidungen oben, die drei echten Fehler, Commits, PRs.
