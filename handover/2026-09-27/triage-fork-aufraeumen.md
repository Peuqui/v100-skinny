# Fork aufräumen — Triage des Fork-Anteils (27.09.2026)

Basis: Fork d8712e2a gegen main 1e90d17f, ohne die Dateien der sechs offenen
PRs. 63 Dateien, rund +4.960/−390. Drei Analyse-Durchgänge (DSv4; FP8+Runner;
Sonstiges+Tests+Kopfzeilen), tragende Aussagen stichprobenartig nachgeprüft.

## Wichtige Befunde vorab

1. **#604 läuft in unserer Produktion nicht.** `_SM70_MODELOPT` (Standard 1)
   leitet ModelOpt-FP8 in `process_weights_after_loading` sofort auf den
   fork-eigenen Pfad `_sm70_fp8_process` um; main's TurboMind und die
   SM75-Route aus #604 werden nie erreicht. NVFP4 im Mixed-Checkpoint geht
   ebenfalls fork-eigen (W4A16 → Skinny-QPN2). Nachgeprüft.
2. **Der DSv4-Tuning-Kern wurde nie eingereicht** (BMM-Decode, BMM-Prefill,
   Indexer-cuBLAS). #657/#659/#662 lagen in anderen Bereichen. Die Maintainer
   haben diese Dateien seit Ende August nicht angefasst → nichts überholt.
3. **#600 macht viele Fork-Stellen überflüssig:** main's `resolve_device_id`
   liefert ohne Argument die Karte des Workers (nachgeprüft). Jede Fork-Stelle,
   die dafür `torch.cuda.get_device_capability(current_device…)` nachbaut, ist
   überholt.
4. **Fehler im Fork gefunden:**
   - `gpu_worker.py` (PLE-Offload unter PP): Off-by-one, prüft `layer_id >= end`,
     die IDs sind 1-basiert; baut außerdem main's
     `check_ple_layers_on_first_pp_rank` nach.
   - `gpu_model_runner.py`: Mamba-Kopierfunktionen pro Typ, eine Aufrufstelle
     (Warmup, Z. ~2299) nicht umgestellt.
   - `fp8.py`: FP8-MoE-Skalen-Verfeinerung aus vLLM #53896 nur halb portiert
     (Lade-Hälfte fehlt, `weight_scale_refine` wird nie gelesen).
   - `sm70/gemv.py`: Env-Gate doppelt zu main, blockiert FP13 ohne FP16-Flag.
   - `modelopt.py`: Referenzpfad `VLLM_SM70_FP8_REFERENCE` ist ein stiller
     Fallback auf langsames fp16.
5. **PR #623 fehlen drei Tests:** `test_sm70_79t_stability.py`,
   `test_sm70_flash_v100_multihead.py`, `test_sm70_prefill_local_heads.py`
   nutzen `load_fa2_library`, die es nur mit #623 gibt → gehören in #623.

## 1. ÜBERHOLT — main kann es selbst, aus dem Fork entfernen

| Wo | Was | Beleg in main |
|---|---|---|
| utils/torch_utils.py | alte `checkpoint_kv_quant_allowed()` (Gerät 0, try/except, Schalter) + Aufruf | config/vllm.py `checkpoint_kv_quant_allowed(cfg)` (#613), Reset in `__post_init__` |
| models/utils.py | shard_id-Weitergabe, `get_rename_mapper`, `ShardId`, kv_scale-Warnung | 86b19124 |
| v1/kv_cache_interface.py | `head_size_v` in AttentionSpec, `prefix_cacheable`, CircularBuffer-Uniform | `CircularBufferSpec(head_size_v=0)`, `KVCacheSpec.prefix_cacheable` |
| v1/core/kv_cache_utils.py | `hashing_sizes` | Assertion in qsa_cache.py |
| attention/backends/fa_utils.py | FA-Version per current_device | #600 |
| alle workerlokalen Capability-Nachbauten | attention.py `_is_exact_sm70_cuda`, mhc/tilelang.py ×4, flashmla_sparse.py, sparse_swa.py (SM70-Zeile), gemv.py | #600 |
| nvidia/dspark.py | `SupportsPP`-Block, `embed_tokens`-Check | speculative.py:1067 (Drafter PP=1), cf0e5d6c |
| gpu_model_runner.py | gloo-Kommentare/Docstrings im PP-Pfad | main sendet schon über cpu_group |
| modelopt.py | `_sm70_implicit_unquantized` | main lässt nicht gelistete Layer unquantisiert |
| Kopfzeilen ohne Code | parallel_state.py, attention.py | Code identisch mit main |

## 2. LOKAL — Debug, Experimente, Kopfzeilen: entfernen

- custom_all_reduce.py: AR-EVT-Messwerkzeug (+153), „dormant in production"
- gpu_model_runner.py: `VLLM_SM70_MTP_THINK_ONLY` (Token-ID hart im Code),
  `STAGED_PREP_SPEC_FORCE`, `GDN_SLOT_DEBUG`, totes `only_gids`, Formatierung
- gdn_attn.py: `GDN_SLOT_DEBUG` (nicht capture-sicher)
- modelopt.py: Referenzpfad, `QPN8_TWOOP`/`qpn8_prefill`, Zensus-Logs;
  qpn8_blk.py: Zensus-Logs
- mhc/tilelang.py: toter `_mhc_pre_torch_generic`
- sm70/sparse.py: `splitk_workspace_specs`-Refactor (Rest, zurück auf main)
- Diagnose-`info_once` in rocm.py, sm70/indexer.py; `info_once`→`info` in gpu_worker.py
- alle 14 Lizenz-Kopfzeilen „Modified by the v100-skinny contributors"
- deutsche Kommentare in fused_indexer_q.py, fa_utils.py, triton_attn.py

## 3. PR-Kandidaten (nach Aufwand/Nutzen)

| # | Paket | Umfang | Abhängig | Anmerkung |
|---|---|---|---|---|
| P1 | Indexer-cuBLAS-Decode (inkl. Bugfix gepaddete Cache-Blöcke, flache Spec-Decode, mehrere Requests) | ~+210/−50 | – (Turing-Teil braucht P3) | am leichtesten; Route ist in main da, greift dort aber nie |
| P2 | SWA-Ragged-Copy-Breite für DSpark-Zeilen | +8/−2 | – | Bugfix, betrifft auch ROCm |
| P3 | DeepSeek-V4 auf Turing / gemischtem PP (Software-FP8 < sm89, CuteDSL ≥ sm80/90, Triton-SWA < sm90, fp16/e4b15-Decode, block_h für 64 KiB) | ~+200/−40 | QPN8-dequant aus P8 | vorher SSOT: `get_max_shared_memory_bytes`, `rocm_inv_rope_einsum` nutzen; SM75-Insertpfad vereinheitlichen |
| P4 | generische Triton-Sparse-Impl für alle Karten vor Hopper | ~+25/−13 | – | heikel: macht main's SM70-Impl tot; Alternative A/B in die SM70-Impl spiegeln |
| P5 | Sparse-MLA-Prefill als Gather+BMM (128er-Häppchen) | ~+230 | P4 | |
| P6 | Sparse-MLA-Decode als dequantisierender Gather+BMM | ~+310 | P5, P4 | größter Tempo-Hebel |
| P7 | PP-Spec-Decode-Invarianten (`_pp_check_token_ids`, Draft-Zeilen-Assert) | ~+60 | – | Diagnose straffen |
| P8 | QPN8-Kernel für Block-FP8 (SM70/75) | ~+250 | – | Vorarbeit: Kernel in csrc statt JIT aus v100-skinny, Prepack aus modelopt lösen, Env in envs.py |
| P9 | Legacy-Runner: Mamba-Kopierfunktionen pro Typ + PLE-Short-Conv im Spec-Decode | ~+50 | – | vergessene Aufrufstelle vorher umstellen |
| P10 | PLE-CPU-Offload unter PP | ~+5 | nach #646 | Off-by-one-Nachbau durch main's Helfer ersetzen |
| P11 | CuteDSL-Gate ab SM80 (Turing-KeyError) | +11/−2 | – | echter Turing-Fix |
| P12 | DSv4-MTP: Architektur-Guard + `DeepSeekV4MTP(SupportsPP)` | ~+30 | – | |
| P13 | Test-Hygiene: mhc-Default-Device, 4 Tests an main-Schnittstellen angepasst | ~+25 | – | main's eigene Tests sind dort kaputt |

## 4. BLOCKIERT — Skinny-Kernel (v100-skinny)

nvfp4_skinny_moe.py, nvfp4/marlin.py (Lader), qpn_dequant.py,
nvfp4_emulation_moe.py, oracle/mxfp4.py + nvfp4.py, quantization/mxfp4.py,
`sm70_skinny`-Literal in config/kernel.py (Docstring verrutscht, korrigieren),
`ignore_full_mode` in breakable_cudagraph.py, Mixed-NVFP4 → W4A16 in modelopt.py.
Hängt an der Vendoring-Entscheidung (1Cat #679, dnv2003/v100-skinny#8).

## 5. UNKLAR — erst messen

| # | Frage | Messung |
|---|---|---|
| U1 | SM75-Zweig im PP-Spec-Empfang nach Entfernen des Turing-GDN noch nötig? | Flash-Next PP mit Turing-Stufe + MTP, greedy k>0 gegen k=0 byteidentisch, ohne und mit Prefix-Caching, Zweig durch normalen Aufruf ersetzt |
| U2 | QPN8-blk gegen main's FP8-QPN8 auf SM70 | DSv4 V100 `VLLM_SM70_QPN8_BLK=0/1`, Tempo + Text |
| U3 | eigene ModelOpt-FP8-Route (~300 Zeilen) gegen main-TurboMind/#604 | 27B Mixed `VLLM_SM70_MODELOPT=0/1`; vorher Mixed-Gate (89) für die Marlin-Einträge klären |
| U4 | Skinny gegen 1Cat-TurboMind/QPN2 für Mixed-NVFP4 | Grundsatzfrage, mit A |
| U5 | FP8-MoE-Verfeinerung (#53896, halb portiert) | entfernen, sofern kein Modell sie braucht |
| G | DSpark-Aux in fp32 (`mhc_post_fp32`, `saturating_cast`) nötig, seit main saturiert (#658)? | DSpark-Annahme `--dtype half` main-Pfad gegen fp32-Pfad |
| H | hc_head-Torch-Fallback < sm80 | main's TileLang-hc_head auf V100/RTX aufrufen |
| T | Triton-3D für Spec-Verify + 64 Segmente, noch gebraucht nach #623? | prüfen, ob ein Eintrag Triton-Attention nutzt; sonst löschen |
| M | mamba_get_block_table_tensor_triton (Upstream-vLLM hat es verworfen) | Flash-Next mit/ohne, Schrittzeit + Bitgleichheit |
| C | GDN-Chain-Fast-Build (−1,4 ms laut Kopf, Default aus) | messen, dann Default oder löschen |

## Zusätzliche Prüfpunkte (Peuqui 27.09.)

- Turing-Schiene voll erhalten; Pfad-Ersatz durch main nur, wenn er auf Volta
  UND Turing in allen Topologien trägt (PP4, PP5, TP2×PP2, TP2).
- Turing-Pfade dürfen keine TP-Grenze fest einbauen (möglicher reiner
  Turing-Server mit TP4/TP8).
- Keine fest verdrahteten Speichergrößen (32/48 GB) — mögliche V100-64-GB-Umbauten.
- Fork muss auch auf 8× V100 TP4×PP2 laufen.

## Vorgeschlagene Reihenfolge

1. ÜBERHOLT + LOKAL entfernen (ein Aufräum-Commit je Gruppe), Bau, Tests,
   Abnahme wie heute (Greedy bitgleich + Tempo, drei Modelle).
2. Die fünf gefundenen Fehler beheben (gehören teils zu PR-Paketen).
3. U1–U5, G, H, T, M, C messen → jeweils behalten (→ PR) oder entfernen.
4. PR-Pakete in der Reihenfolge P1, P2, P11, P7, P13, P12, P3, P9, P10, dann
   P4–P6 und P8. Die drei #623-Tests in #623 nachtragen.

## Umsetzung im Worktree 1Cat-vLLM-cleanup (Branch cleanup-fork-2026-09-27, uncommittet)

Durchgang 1 (ganze Dateien auf main 1e90d17f): torch_utils.py,
layers/attention/attention.py, parallel_state.py, models/utils.py,
attention/backends/fa_utils.py, custom_all_reduce.py, deepseek_v4/sm70/sparse.py,
kv_cache_interface.py. Zurückgestellt: kv_cache_utils.py (hash_block_size für
Qwen4Exp — erst messen).

Durchgang 2: flashmla_sparse.py, deepseek_v4/sm70/gemv.py auf main.
**Verhaltensänderung gemv.py:** main schaltet VLLM_SM70_DSV4_FP13_GEMV
standardmäßig ein; der Fork-Vertrag verlangte zusätzlich den FP16-Schalter
(0) und blockierte main's FP13-GEMV auf den V100 still. Nach dem Zurücksetzen
ist der FP13-Pfad bei DSv4 aktiv -> in der Abnahme Tempo UND Qualität einzeln
bewerten (Greedy wird vermutlich nicht mehr bitgleich sein).

Weitere Umsetzung (Durchgang 2, Rest): Geräte-Nachbauten in deepseek_v4/attention.py
und sparse_swa.py entfernt (#600); GDN-Slot-Debug (gdn_attn.py + Runner),
`VLLM_SM70_MTP_THINK_ONLY`, `STAGED_PREP_SPEC_FORCE`, totes `only_gids` entfernt
(Hunks rückwärts angewendet); gpu_worker.py: PLE-PP-Prüfung auf main's
`check_ple_layers_on_first_pp_rank` (Off-by-one weg), `info_once` zurück.
Lizenz-Kopfzeilen bleiben (Peuqui), nur angepasst, wo sich der Inhalt ändert.

**FP13-GEMV:** greift bei uns nie — main's Vertrag verlangt x.shape == (1, K),
DSv4 mit DSpark K5 hat 6 Zeilen pro Schritt. Das Entfernen der Fork-Zeile ist
richtig (sie blockierte den Einzelzeilen-Fall), ändert unsere Produktion nicht.

Durchgang 3: 29 fork-eigene rechenweg-relevante Schalter in envs.py
registriert (Skinny-Gruppe, VLLM_SM70_MODELOPT, FP8_REFERENCE, QPN8-Gruppe,
NVFP4-MoE-Startkonfiguration, GDN_CHAIN_SPEC_FAST_BUILD, TRITON_3D_SPEC,
TRITON_SOFTMAX_SEGMENTS) und Lesestellen auf envs umgestellt -> sie gehen in
compile_factors ein (geprüft). Altlasten in marlin.py, qpn8_blk.py und den
Triton-Dateien behoben (Importe, deutsche Kommentare, torch.cuda ->
current_platform, mypy method-assign per setattr + noqa B010).
main hat selbst ~140 unregistrierte Schalter (DDTree, Dump, QSA, TurboQuant):
möglicher Hinweis/PR an 1Cat.

Tests nach Durchgang 3: V100 932, RTX 886 bestanden; test_skinny_mxfp4_moe.py
scheitert auf Merge- UND Aufräum-Stand ohne VLLM_SKINNY_NVFP4_SRC (Kernel-Quelle
liegt im v100-skinny-Repo), mit gesetztem Schalter 8/8 auf beiden Karten.
Test-Hygiene: der Test sollte sich ohne Quelle überspringen (P13).
