# Unabhängige Prüfung Turing-Kernel-PR (27.10. ~12:45): READY AFTER FIXES

In Ordnung: Kerne (Swizzle, ldmatrix, m16n8k8-Fragmente, Split-K-Reduktion, Grid-Grenze, Prepack-Tabellen) von Hand geprüft; 72/55 Cubins;
alle Unit-Test-Zahlen des Textes stimmen mit den Logs (Baum ff7685bf8); CONFIGURED_OPERATORS für diesen Zweig vollständig.
BLOCKER
1. 27B-FP8-Decode-Zeile beschreibt Routing, das der PR nicht liefert: Arm war 7f343a791 (sm75_mma vor _QPN8_MAX_M → M=1..512 über den neuen
   GEMM, Log „Block FP8 QPN8 on SM75: M=1.“, ab_q27fp8turing_2026-10-20.log); MTP-Decode M=4 geht am Kopf (c4b72f197) zurück zum gepackten
   GEMM → streichen oder als Zwischenstand kennzeichnen (Referenz 23e34b03c, Änderung 7f343a791), FP8-27B mit Endrouting nicht nachgemessen.
2. „keeps every Volta path as it is“ falsch: c27eeae9f (zwei Spaltenkacheln, Split 8 statt 16 für w13) gruppiert die FP32-Reduktion um →
   Volta-MoE nicht mehr bitgleich zu main; Fork-Beleg nur gebündelt in der Routen-A/B → offenlegen.
3. Duplikat-Absatz unvollständig: #1069/#1157 (kezboard233, fp8_qpn8_sm70.cu, Block-Scale-QPN8 M 9..32 bzw. 17..32 — gleicher Problemraum,
   Unterschied erklären); #822 (u. a. test_turing_qpn_kernel_selection.py → Konflikt); #1216 (eigener PR, CMakeLists.txt-Konflikt; host_reduce.cu
   in SM70_FAMILY_ARCHS aufnehmen, wer zuletzt landet); #1243 (auf #1031, gleiche zwei Skinny-MoE-Konflikte). „3 files for #1030, 6 for #1031“ =
   deren Konflikte mit main; durch diesen Zweig je 2 (skinny_moe_qpn_sm70.cu, skinny_sm70_moe.py). #1031 = DeepSeek-V4.1-Port.
SHOULD-FIX (Code)
4. Turing-Split-8-Regel leckt in die Per-Tensor-Kerne M 9..32: fp8_qpn8_launch_config (sm70_turbomind.py:674) speichert Split 8 im State, die
   Dispatch nutzt ihn für jedes M; M=17..32-Kandidat verlangt split_k 12/16 (fp8_qpn8_sm70.cu:2025-2030) → fällt still auf den M16-Körper,
   M 9..16 mit Split 8; nur M=1..8 gemessen → Belegungs-Split nur für den M≤8-Start, oder M 9..32 messen.
5. sm_75-Bau liefert überwiegend toten Code (exakte SM70-Hostprüfungen in sm70_router.cu:28, qsa_indexer_sm70.cu:179,
   sm70_ring_collectives.cu:118, qsa_device_history_sm70.cu:878, exact_row_reduce.cu:143, h3_w8a16.cu:126, h3_noncausal_sm70.cu:334,
   flash-attention-v100 h3/forward.cu:120, forward_sparse.cu:89; TurboMind sm70_884 Arch [700,750) leer auf sm_75); „1-4 % skinny MoE“-Begründung
   maß den m8n8k4-Kern, der nach f852ca00e auf Turing nicht mehr läuft; Modell-A/B ab_sm75_2026-10-21.log: DSv4 kein Unterschied (Prefill
   ±0,2 %, Texte gleich). Nötig ist sm_75 für den m16n8k8-Kern in skinny_moe_qpn_sm70.cu → SM70_FAMILY_ARCHS auf Quellen beschränken, die auf
   SM75 laufen (TurboMind-Ops-Liste, Sampler), Grund neu fassen, A/B nennen. 369cccdc1 nennt „host reduce“ (gibt es auf main nicht).
6. ldmatrix_x4/mma_m16n8k8 kompilieren unter SM75 zu nichts (mma_fragments_sm70.cuh:42-64) → __trap() im #else.
7. test_sm70_native_policy schlägt weiter wegen gguf_grouped_prefill_gemm_sm70_out fehl (main-Fehler; Fix im wo_a-PR) → nennen.
   Test Plan ergänzen: test_sm70_native_policy.py, test_block_fp8_qpn8_capabilities.py, test_sm70_moe_mainflow, test_sm70_fp4_mainflow,
   tests/tools/test_sm70_path_explanations, sm70_route_snapshot --check (moe/fp8); bekannte main-Fehler angeben.
SHOULD-FIX (Text)
8. Pfadtabelle „CSF scales“ — gibt es auf dem Zweig nicht.
9. „M 9..64 0,16-0,29x“ stammt aus Einzelaufruf-Ereigniszeiten (vor der Korrektur); CUDA-Graph-Wert „M 16..64 0,16-0,60x“ über 13 Formen (c4b72f197).
10. DSv4-A/B (c4b72f197): Greedy in 2 von 3 Prompts verschieden, 7/8 Antworten verschieden (gleichwertig gelesen); Code tok/s 73,5 → 67,7
    (Textpfad), Annahme gemischt (ab_qpn8turing_accept); Prompt ist 178 480 Token (nicht „150k“), 113,40-113,41 → 112,71-112,74 s.
11. „patched Triton on PYTHONPATH (identical in both arms)“ unbelegt: Läufe 20./21.10. ohne sm7x-Triton (kam am 23.10.).
12. Referenz/Änderung je A/B: 23e34b03c → 7f343a791; 23e34b03c → 4da0bcfa4; 23e34b03c → 6c0ef8efb; e1cb50fd9 → 79e07bfef; 27B-FP8-Eintrag nennen.
13. V100-Fehlschlag nur per Codelesen als unabhängig belegt → Test auf origin/main-Bau mehrfach laufen lassen.
14. Historie: Paare 77006c020+c4b72f197 und f5486ee76+69dfe6906 zusammenfassen/umformulieren; 77006c020 „Turing now runs M=1..512“; „v4 and v2“,
    „production _C“, „entry“; Signed-off-by 1/11.
NIT: Datumsangaben 20.10.2026 im Code (fp8_block.py:52, fp8_qpn8_mma_sm75.cu:308, skinny_sm70_moe.py:69, skinny_moe_qpn_sm70.cu:690) → ISO;
skinny_sm70_moe.py:515 veraltet; „SM8x+ would also pick“ irreführend (nur 7.0/7.5); +1,75/+1,79/+1,60 %; „K/16“; „replacing copies in …
skinny_moe_qpn_sm70.cu“; TurboMind-cvt_f16x8_e2m1-Herkunftsvermerk (Apache-2.0) wieder aufnehmen; alte Compile-Caches; Test für fehlenden Op
in can_implement; flash_attn_v100/ nicht hinzufügen.
