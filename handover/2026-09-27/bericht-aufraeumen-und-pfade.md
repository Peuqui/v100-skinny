# Fork aufräumen + Pfad-Vergleiche — Bericht 27.09.2026

## Kurzfassung

1. **Pfad-Vergleiche Fork gegen main:** Skinny ist auf unserem Rig nötig,
   nicht nur schneller (DSv4: main hat keinen MXFP4-MoE-Pfad für Turing;
   Flash-Next PP4: main's Weg passt nicht in den V100-Speicher; TP2×PP2: main
   Prefill 19,2 statt 10,5 s). **#691 wirkt in keiner Topologie mehr.**
   **Ausnahme 27B auf den RTX: #604 hat 17 % schnelleren Prefill** als unser
   Fork-Pfad (22,7 gegen 27,5 s) bei etwa gleichem Decode.
2. **Erster Aufräum-Durchgang** liegt uncommittet im Worktree
   `1Cat-vLLM-cleanup` (Branch `cleanup-fork-2026-09-27`): 15 Dateien,
   −517/+129 Zeilen. Alle Tests grün (V100 925, RTX 879). Abnahme auf den drei
   Modellen: siehe unten.

## Pfad-Vergleiche im Detail

| Modell / Topologie | Fork-Pfad | main's Pfad | Ergebnis |
|---|---|---|---|
| DSv4 PP5 MoE | Skinny-MXFP4 | TurboMind-MXFP4 | main startet nicht: kein Turing-Pfad (RTX-Stufen) |
| DSv4 PP5 Dense-FP8 | QPN8-blk | main's FP8-Route | nicht testbar: Fork-Schalter `VLLM_SM70_QPN8_BLK=0` kaputt (attention.py:436, O-Proj-Referenz setzt QPN8-Layout voraus) |
| Flash-Next PP4 | Skinny-NVFP4 | TurboMind | main: CUDA OOM auf V100 |
| Flash-Next TP2×PP2 | Skinny | TurboMind/Einzel-Experten | main langsamer: Prefill 19,2 gegen 10,5 s, Decode 58–68 gegen 55–62 ms |
| 27B TP2 RTX | Skinny-QPN2 + eigenes QPN8 | #604 | **#604 Prefill 22,7 gegen 27,5 s**, Decode 39–46 gegen 38 ms |
| #691 (Batch-GEMM-Voreinstellung) | – | – | ohne Effekt: FN PP4, FN TP2×PP2, 27B V100 TP2 |

Lehren aus den Messungen:
- Unregistrierte Fork-Schalter (`VLLM_SM70_MODELOPT`, `VLLM_SKINNY_QPN2`,
  `VLLM_SM70_QPN8_BLK`) gehen nicht in den Compile-Cache-Schlüssel -> nach dem
  Umschalten lädt vLLM den alten Graphen und stürzt ab. Für Tests eigener
  `VLLM_CACHE_ROOT`; für den Fork: registrieren oder entfernen.
- Die gestrige #691-Regression gehörte zum Zwischenstand b0346480 mit alter
  FA-V100-Bibliothek.

## Aufräum-Durchgang 1+2 (uncommittet)

Ganz auf main zurückgesetzt (Fork-Anteil vollständig überholt oder lokal):
utils/torch_utils.py (alte, doppelte checkpoint_kv_quant_allowed mit Gerät-0-
Bug), layers/attention/attention.py und distributed/parallel_state.py (nur
Kopfzeilen), model_executor/models/utils.py (86b19124 in main),
attention/backends/fa_utils.py (#600), device_communicators/custom_all_reduce.py
(AR-EVT-Messwerkzeug), deepseek_v4/sm70/sparse.py (Refactor-Rest),
v1/kv_cache_interface.py (main hat CircularBufferSpec), mla/flashmla_sparse.py
(#600), deepseek_v4/sm70/gemv.py.

Gezielt entfernt: Geräte-Nachbauten in deepseek_v4/attention.py und
sparse_swa.py (#600 erledigt das), GDN-Slot-Debug (gdn_attn.py + Runner),
`VLLM_SM70_MTP_THINK_ONLY`, `STAGED_PREP_SPEC_FORCE`, totes `only_gids`.

Fehler behoben: PLE-Offload-PP-Prüfung in gpu_worker.py (Off-by-one,
Nachbau) -> nur noch die PP-Sperre aufgehoben, Platzierung prüft main's
`check_ple_layers_on_first_pp_rank`.

**Verhaltensänderung:** gemv.py — der Fork-Vertrag blockierte main's
FP13-GEMV (Standard an) bei DSv4 auf den V100. Jetzt aktiv.

Bewusst NICHT angefasst: Lizenz-Kopfzeilen (siehe Entscheidungen),
kv_cache_utils.py (hash_block_size, erst messen), alle UNKLAR-Punkte, Skinny,
QPN8-blk, eigene ModelOpt-FP8-Route.

## Abnahme des aufgeräumten Stands

**Tempo und Greedy (ab_cleanup):** bitgleich zum Fork, Tempo gleich.

| Modell | Decode Fork / clean | Prefill Fork / clean |
|---|---|---|
| DSv4 | 91 / 91 ms | 8,7 / 8,6 s |
| Flash-Next PP4 | 68 / 67 ms | 8,2 / 7,5 s |
| 27B | 45 / 39 ms | 27,8 / 27,6 s |

**Qualität, händisch gelesen** (8 Prompts, Ausgaben in
`~/.cache/bench-scripts/quality_2026-09-27/`):

- **DSv4 prod = clean:** alle 8 Antworten identisch. Gelesen, alle bestanden
  (zug 84 km/h, kuanda ohne Erfindung).
- **27B prod = clean:** alle 8 Antworten identisch.
- **27B Fork gegen #604-Pfad:** zug, kuanda und english weichen ab.
  - zug: Beide werden bei 600 Token vor dem Endergebnis abgeschnitten, der
    Rechenweg (287·12/41) ist gleich und richtig; nur die Einleitung ist anders.
  - english: Beide erklären Rayleigh-Streuung und den längeren Weg bei
    Sonnenuntergang korrekt, nur mit anderen Worten.
  - kuanda: Beide nennen die Person unbekannt und erfinden trotzdem
    Nebendetails (Fork: Kongo/Tshombe-Bezug, #604: „Henri Koudou/Kagan“).
    Das ist dieselbe Schwäche des Modells und kein Unterschied zwischen den
    Pfaden.
  - Urteil: gleichwertig, die Abweichungen sind nicht bedeutend.
- **Flash-Next prod (#691 an) gegen clean-Testeintrag (#691 aus):**
  motor, zug, kuanda, summary und english weichen ab.
  - motor, summary, english: nur einzelne Wörter anders
    („Danach beginnt“ statt „Nach diesem vierten Takt beginnt“).
  - zug: Der Rechenweg ist identisch (287 : 41 = 7, v = 7·12). clean endet mit
    **84 km/h**, prod wird bei 600 Token genau davor abgeschnitten.
  - kuanda: Beide schreiben „Es gibt keine … bekannte Person namens Henri
    Kuanda“ und nennen Konan Bédié als Verwechslung. Keine der beiden
    erfindet etwas über Kuanda.
  - Urteil: gleichwertig.
  - **Ursache (Kontrolllauf geklärt):** Weder #691 noch das Aufräumen. Der
    Merge-Stand liefert frisch kompiliert genau die clean-Antworten (alle 8
    gleich, merge-fn-recheck). prod-fn lief auf dem warmen AOT-Graphen von
    gestern; ein neu kompilierter Graph wählt andere Inductor-Kernel und
    verschiebt die Rundung. Der Aufräumcode rechnet identisch.

**Produktionsabnahme** (Produktion auf c58a4ddc, 27B auf #604,
max_tokens 1500, prod_accept.sh):

- DSv4: Greedy bitgleich zur Referenz, alle 8 Antworten gleich zu prod-ds.
- Flash-Next PP4: Greedy bitgleich, alle 8 gleich zu clean-fn und merge-fn-recheck.
- 27B #604: Greedy 2/3 gleich, der dritte weicht wie erwartet ab (#604-Pfad).
  7 von 8 gleich zu clean-q27-604. zug läuft jetzt bis zum Ergebnis **84 km/h**
  mit Probe (41·84 = 3444); die Abweichung beginnt erst nach etwa 250 Token und
  betrifft nur die Formulierung.
- Tag `verified-2026-09-27-cleanup` auf c58a4ddc (nicht gepusht).

**FP13-GEMV (Punkt 3):** greift auf der Produktion nie (Vertrag x.shape ==
(1, K), DSpark liefert 6 Zeilen). Annehmen ändert nichts; bleibt drin.

## Deine Entscheidungen

1. **Lizenz-Kopfzeilen** „Modified by the v100-skinny contributors": Apache 2.0
   §4(b) verlangt Änderungshinweise in geänderten Dateien bei Weitergabe; der
   Fork ist öffentlich. Behalten (meine Empfehlung) oder Git-Historie genügt?
2. **27B-Produktion auf #604 umstellen?** +17 % Prefill-Tempo, Decode etwa
   gleich; Greedy-Text weicht in 1/3 Prompts ab (andere Kernel).
3. **FP13-GEMV bei DSv4** annehmen, wenn die Abnahme Tempo und Qualität bestätigt.
4. **Aufräum-Commit** freigeben (dann Produktion wie gestern umstellen).
