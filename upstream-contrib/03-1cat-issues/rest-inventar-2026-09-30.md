# Rest-Inventar: Produktion minus (1Cat main + alle offenen PRs) — 30.09.2026 abends

Basis: Worktree 1Cat-vLLM-prunion = main d3046986 + alle 20 offenen eigenen PRs
(604 611 621 623 646 667 710 711 714–717 720 723 725 726 740 741 742 743), verglichen
mit Produktion d9b689f2. 60 Dateien. Nutzung belegt aus journalctl -u llama-swap seit 27.09.

## A. Nur Fassungsunterschied zu offenen PRs (nichts Neues anzubieten)
- S1 #742: csrc-Kernel vs JIT im Fork, skinny_sm70_moe.py vs nvfp4_skinny_moe.py, Tests,
  oracle/*, config/kernel.py, breakable_cudagraph.py, quantization/mxfp4.py, envs-Teile.
  Fork übernimmt die PR-Fassung nach dem Merge.
- #711 (check_env_registration + pre-commit): im PR, nicht im Fork.
- #740 Direct-IO: Fork hat zusätzlich den gemeinsamen madvise-Helfer (SSOT direct_io ↔
  ple_layer) — bewusst aus #740 genommen; Nachzieher nach Merge von #740.
- #717 gpu_worker.py: nur Kommentar/Lizenzkopf.
- #733 (fremd) + #621: config/vllm.py (explizites mode=NONE, Compile-Cache-Kommentar).
- #664 (fremd): qsa.py E4M3-Arbeitsspeicher + torch.full — wörtlich im fremden PR.
- #714 / #716: ältere Fork-Fassungen (sm70/indexer.py, sm70/sparse.py, Tests).
  ABER: Produktion fährt die Gather+BMM-Variante im Triton-(„ROCM“-)Impl, #716 bietet sie
  im SM70-Impl an (richtig für reine V100-Nutzer) → siehe S3.

## B. Bleibt bewusst im Fork (entschieden)
- PP-Token-Prüfung + draft_len-Assert in gpu_model_runner.py (cd7bbf3c; Peuqui 30.09.).

## C. Echte Lücken — anzubieten
- **S2 QPN8-Block-FP8** (Produktion DSv4, 14.218 QPN8_BLK_CENSUS_LOAD):
  qpn8_blk.py, kernels/linear/__init__.py, compressed_tensors.py (_W8A8Fp8Sm70Block),
  fp8.py (use_sm70_fp8_qpn8_blk, get_min_capability 70), modelopt.py (_sm70_qpn8_prepack/
  _unpack/_indices), envs (QPN8_BLK_CFG, WMMA_MAX), skinny_kernels.cu (4 Ops: gemm_qpn8_blk,
  _mt2, _wmma, qpn8_blk_dequant), wo_a-fp16-Dequant-Route (is_bmm).
  Offene Designfrage: Standard oder auf Anfrage (1Cat hat TurboMind-FP8 + eigene QPN8-
  Tabellen nur für DSv4 TP4). Entscheidung nach Kernel-Matrix.
- **S3 DeepSeek-V4 auf Turing / gemischt Volta+Turing** (hängt an S2 wegen wo_a):
  import_utils.has_cutedsl (sm80+), compressor.py (CuteDSL nur sm90+), sparse_swa.py
  (Triton-SWA vor Hopper), fused_indexer_q / cache_utils / fused_compress_quant_cache
  (Software-FP8 < sm89 statt ==sm70), dspark.py (main_proj_input_scale < sm89,
  _insert_context_kv < sm80), sparse_attn_indexer._is_exact_sm70_cuda → < sm80,
  sm70/indexer.py cuBLAS-Decode auch sm75 (eigenes Gerät), attention.py (Triton-Impl vor
  Hopper, Q/KV-Insert < sm80 aus Einzelteilen, O-Projektion mit fp16-wo_a),
  rocm_aiter_mla_sparse.py (e4b15, fp16 auf Volta, Shared-Mem-Grenze), amd/rocm.py
  (BMM-Decode/Prefill im Triton-Impl), model.py (scale_fmt optional).
  DESIGN: Fork nimmt vor Hopper auf ALLEN Karten den Triton-Impl (PP-Stufen brauchen
  dasselbe Backend). Für 1Cat: reine V100 behalten SM70-Impl; sobald eine Turing-Karte im
  Verbund ist → alle Stufen Triton-Impl (1Cat hat _any_participating_device_is_capability).
- **DSpark SupportsPP** (dspark.py): ohne Interface stirbt `--speculative-config dspark`
  unter PP beim Start — trifft auch reine V100-PP-Nutzer. Am main-Code prüfen, dann
  eigener Bugfix-PR (unabhängig von S3). Dazu embed_tokens-Ladeprüfung.
- **ModelOpt-Übersteuerung VLLM_SM70_MODELOPT** (modelopt.py, Standard AN, greift bei
  Flash-Next MIXED_PRECISION in Produktion): SM70-Zulassung, NVFP4 → W4A16-Methode,
  _sm70_implicit_unquantized. Enthält `except Exception` „never fail a boot“ = stiller
  Fallback (gegen Regel). A/B nötig: Flash-Next mit VLLM_SM70_MODELOPT=0 gegen Produktion.
  Gleich → Übersteuerung im Fork zurückbauen; verschieden → Lücke anbieten.

## D. Toter Code im Fork (Rückbau vorschlagen, nicht anbieten)
- Dichte Skinny-NVFP4-Linear-Routen: marlin.py (+903), qpn_dequant.py (+152) — seit 27.09.
  kein „path enabled/dense prefill/self-check/lm_head“ im Log.
- ModelOpt-FP8-Skinny-Pfad (_sm70_fp8_process, gemm_qpn8 per-tensor): nur 27.09. in
  Merge-Tests; 27B-Produktion fährt VLLM_SM70_MODELOPT=0.
- Torch-Referenzpfade für Ampere/Ada (sparse_attn_indexer._torch_*_logits,
  attention._torch_indexer_q_rope_quant): auf unseren Karten unerreichbar, ungetestet.
- Emulations-Häppchen (nvfp4_emulation_moe.py, S1b gestrichen).

## E. Kleinkram Tests/Benchmarks
- test_mhc_kernels.py +9 (01e14ec6 Standardgerät zurücksetzen) — prüfen, ob #726 es abdeckt.
- test_dspark.py +4, test_kv_cache_utils.py, test_config.py, test_sm70_79t_stability.py,
  test_sm70_flash_v100_multihead.py, test_sm70_prefill_local_heads.py, benchmark_sm70_decode.py.

## Reihenfolge
1. DSpark-SupportsPP am main prüfen (klein, unabhängig).
2. Flash-Next-A/B VLLM_SM70_MODELOPT=0.
3. S2 (Kernel-Matrix → Designentscheidung → Port → Tests → A/B → Entwurf).
4. S3 (auf S2 aufsetzend).
5. Rückbau-Vorschläge D an Peuqui; E einsortieren.

## Ergebnisse 30.09. spätabends
- DSpark-SupportsPP: KEIN Bug in main — main legt DSpark/DFlash-Drafter auf PP=1
  (speculative.py:1067). Fork-Ergänzung überflüssig → Rückbauliste D.
- ModelOpt-Übersteuerung: Flash-Next PP4 mit VLLM_SM70_MODELOPT=0 greedy 3/3 bitgleich,
  gleiche Pfade (Checkpoint meldet ohnehin W4A16_NVFP4), Tempo gleich (29k-b 18,3 s/70 ms gegen
  17,5 s/69 ms, erster Boot nach Graphänderung) → kein PR, Rückbauliste D.
- S2 Kernel-Matrix (benchmarks/fp8_blk_three_way.py, DSv4-Formen): 1Cats nativer QPN8 ist
  bei M ≤ 8 gleich/schneller als Skinny, ab M 128 bis 2× schneller (dichter Prefill);
  Skinny gewinnt nur M 12–64 (RTX deutlich, V100 schlägt TM dort). TurboMind-FP8 hat
  KEINEN sm75-Kernel. Genauigkeit mit echten Gewichten + Ausreißern gleich (2e-4).
  → Kein Kernel-Port an 1Cat.
- Fork-Test (Worktree 1Cat-vLLM-s2test, unkommittet): QPN8-blk über 1Cats native Ops.
  FEHLER meiner Einbindung: ein gemeinsamer dichter Arbeitsspeicher für alle Schichten;
  DSv4 rechnet indexer.wq_b parallel zu attn.wq_b (attention.py:607, Nebenstream) →
  Überschreiben beim Prefill → P(Satzende, 1. Token) 5–12 % statt 0, Leerantworten.
  Genau deshalb schließt 1Cat indexer.wq_b aus seinem gemeinsamen Speicher aus.
  Mit eigenem Speicher für ".indexer." Erst-Token-Verteilung wie Produktion.
  Kurzer Prompt („Ein“/„**“) ist Rundungs-Gleichstand: Skinny mit cuBLAS ab M 17 gibt
  53/47, WMMA 78/21, nativ 41/59 — alles im Rauschband.
  E2E: DSv4-Prefill 35,8k 15,9 statt 16,9 s (−6 %), Decode 90 ms gleich.
- Offene Entscheidung (Peuqui): Produktion auf nativen QPN8 umstellen? Robustheit: Speicher
  je Nebenstream-Rolle statt Präfix-Heuristik überlegen.

## Neu beim Rückbau (01.10. nachts) — vorbestehende Testfehler auf V100 UND RTX (auch in der Produktion)
- tests/kernels/test_fused_inv_rope_fp8_quant.py: 50 failed — Triton fp8e4nv erst ab sm89;
  DSv4 ruft den Kernel vor Hopper nicht auf → Test braucht skipif (< sm89). Kleiner Test-PR
  an 1Cat (oder in #726 aufnehmen).
- tests/kernels/test_compressor_kv_cache.py: 21 failed, nur Indexer-Tests
  (indexer_k_quant_and_cache + cp_gather_indexer_k_quant_cache, „Token 0 diff 3,1 > 0,125“).
  Ursache offen: anderes Cache-Format auf SM70-Builds oder echter Op-Fehler. Produktion DSv4
  nachweislich sauber (Greedy/Erst-Token) → vermutlich Test/Format; UNTERSUCHEN.
