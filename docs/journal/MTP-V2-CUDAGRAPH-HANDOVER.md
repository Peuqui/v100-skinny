# MTP im V2-Model-Runner: CUDA-Graphen rechnen die Spekulationsform falsch

Stand 2026-09-07, ca. 15:30. SSOT fuer den offenen Defekt.
Vorgaenger-Kontext: QWEN4EXP-PORT-HANDOVER.md, logs/pp-mtp-deadlock-2026-09-06/.

---

## 1. Kurzfassung

Drei Fehler gefunden, **zwei behoben und gepusht**, einer offen.

Der offene Defekt: **Das Zielmodell rechnet unter Spekulation falsch, sobald
CUDA-Graphen benutzt werden.** Mit `--enforce-eager` ist die Ausgabe korrekt.
Das ist KEIN PP-Problem, KEIN Qwen4Exp-Problem und KEIN Drafter-Problem — der
27B reproduziert es auf zwei Karten ohne Pipeline.

Nicht zufriedengeben mit dem Notbehelf (Graphen aus). Der eigentliche Fehler
sitzt in den uebersetzten/aufgezeichneten Kernels fuer die Spekulationsform.

---

## 2. Behoben (nicht mehr anfassen)

**0b8f4cc** — Verklemmung in `pp_utils.py`. `pp_broadcast` schickte
`sampled_token_ids` ungepolstert, `pp_receive` legt immer
`[num_reqs, num_speculative_steps+1]` an. Erster Sampling-Schritt nach dem
Aufwaerm-Prefill hat eine Spalte: 4 gegen 20 int64, NCCL haengt.
Deterministisch. Betrifft auch `eagle`+PP. Dreifach verifiziert.

**28c6b54** — `eagle/utils.py` Zeile 66 benutzte `getattr(target_model,
"lm_head")` statt des Helfers `get_target_lm_head` aus derselben Datei (den
nur der dflash-Pfad benutzt). Bei `...ForConditionalGeneration` sitzt der Kopf
am Sprachmodell -> `None` -> das GESAMTE Teilen faellt aus (lm_head,
shared_head.head, topk_indices_buffer). Ohne den Fix bootet der 27B auf V2
gar nicht (`AttributeError: 'PPMissingLayer' object has no attribute
'maybe_get_sm70_lm_head_top1'`).

Beides unveraenderter 1Cat-Code, also echte Upstream-Fehler. Meldung offen
(Entscheidung Peuqui).

---

## 3. Der offene Defekt

### Messmatrix (27B, V2 erzwungen, TP2 auf GPU 0+2, KEIN PP, mtp k=3)

| Modus | Ziel-Hidden-States | Entwuerfe | Ausgabe |
|---|---|---|---|
| `--enforce-eager` | endlich | echt | **kohaerent** |
| FULL (Vorgabe) | **NaN** | `[0,0,0]` | Fixpunkt, 1 Zeichen |
| FULL + Split-Graphen | **NaN ab Schritt 4, identisch zu FULL** | `[0,0,0]` | leer |
| PIECEWISE | endlich | — | Kauderwelsch (CJK-Mojibake) |
| k=0 (keine Spekulation) | — | — | **kohaerent** |

Nur vollstaendig eager ist korrekt. Jede Graph-Form rechnet die
Spekulationsform falsch, auf unterschiedliche Weise.

### Harte Belege

- `target.hidden_out` misst `hidden_states` des ZIELMODELLS direkt vor der
  Uebergabe an `speculator.propose()`: `lhs_nan = [True,True,True,True]`.
  Der Drafter bekommt bereits NaN herein, er erzeugt es nicht.
- Die Marke 248320, die das Ziel im Fixpunkt endlos ausgibt, ist kein
  gesundes Token, sondern das deterministische Ergebnis eines Samplings ueber
  NaN-Logits.
- Akzeptanzrate 0,0 % ueber alle k: `Per-position acceptance rate: 0.000,
  0.000, 0.000`.
- **Das NaN entsteht exakt beim ERSTEN FULL-Replay der Spekulationsform**
  (Schritt 4: 4 Token, 1 Anfrage). Schritte 1-3 sind Prefill und laufen
  ohne Graph — sie testen die Spekulationsform nie. Verlauf identisch in
  `marks/diag27b.*` und `marks/splitcg.*`; `marks/piecewise.*` zeigt
  durchgehend False.

---

## 4. Reproduktion — schnelles Fahrzeug benutzen!

**NICHT mit Flash-Next debuggen** (4 Karten, 9-10 min/Boot). Der 27B zeigt
denselben Defekt auf zwei Karten in 60-90 s (warmer Compile-Cache).

```
logs/mtp-v2-cudagraph-2026-09-07/scripts/diag27b.sh    # k=3, Marken an
logs/mtp-v2-cudagraph-2026-09-07/scripts/eagertest.sh  # dito, --enforce-eager
logs/mtp-v2-cudagraph-2026-09-07/scripts/piecewise.sh  # dito, PIECEWISE
logs/mtp-v2-cudagraph-2026-09-07/scripts/splitcg.sh    # dito, Split-Graphen
logs/mtp-v2-cudagraph-2026-09-07/scripts/fast.sh       # 6 Varianten am Stueck
```

Kern: `VLLM_USE_V2_MODEL_RUNNER=1`, `CUDA_VISIBLE_DEVICES=0,2`, TP2, PP1,
`--max-model-len 32768`, `mtp` k=3, `FLASH_ATTN`.
Der 27B (`Qwen3_5ForConditionalGeneration`) waehlt sonst V1, weil quantisiert.

**Vor jedem Lauf `nvidia-smi` pruefen** — belegter Fremdspeicher macht die
Messung wertlos (ist einmal passiert). Die Skripte haben dafuer eine
Vorpruefung.

**Genau EIN Skript laufen lassen.** `pgrep -af 'bash \./run\.sh'` vorher.
Die Kommandozeile lautet `./run.sh`, ein Muster wie `fast/run[.]sh` trifft
sie NICHT — dieser Fehler hat einmal zwei Laeufe parallel gestartet, die sich
die Karten weggenommen haben.

---

## 5. Ausschlussliste — nicht erneut untersuchen

Alles Folgende ist gemessen oder am Code belegt ausgeschlossen:

1. **PP-Uebergabe**: 114 Schritte, Stufe 0 gegen Stufe 1, NULL Abweichungen in
   `prepare_inputs` und `gdn_metadata`. Eingaben, Entwuerfe, Metadaten sind
   bitgleich.
2. **Fehlende Entwurfsmarken unter PP**: war eine echte Luecke, ist in 0b8f4cc
   behoben, war aber NICHT die Ursache der verstuemmelten Ausgabe.
3. **Ausricht-Modus / Prefix-Caching** (`mamba_cache_mode == "align"`): ohne
   Prefix-Caching wird es SCHLECHTER (300 Token, kein Nicht-Leerzeichen).
4. **`spec_state_slot_selectors`**: existiert in V2 nicht, faellt aber
   dokumentiert auf `num_accepted_tokens` zurueck (gdn_attn.py:643), und V1
   setzt beide ohnehin auf denselben Wert. Die sm75-GDN-Schichten lesen ihn
   gar nicht.
5. **`use_local_argmax_reduction`**: an und aus liefern bitgleich dasselbe.
6. **`VLLM_1CAT_ENABLE_SM70_MTP_DEFAULTS`**: mit und ohne bitgleich dasselbe.
7. **k**: k=1 und k=3 liefern bitgleich dasselbe.
8. **Reihenfolge `postprocess` vor `propose`**: stimmt (1600 vor 1617).
9. **`prepare_eagle_inputs`**: zieht `num_rejected` korrekt ab, setzt
   `last_token_index` richtig.
10. **`combine_sampled_and_draft_tokens`**: Kernel korrekt, `input_ids` und
    `logits_indices` gemessen richtig.
11. **Eigene Puffer des Spekulators**: er hat eigene `InputBuffers` (Zeile 81),
    klobbert dem Ziel nichts.
12. **Dummy-Block-Tables/Slot-Mappings bei der Aufzeichnung**: liefern
    persistente Sichten auf die echten Puffer, ausdruecklich dokumentiert.
    ACHTUNG: nur am Code geprueft, NICHT gemessen — siehe naechste Schritte.
13. **`try_dflash2_sparse_target_rejection`**: steigt bei `mtp` sofort aus.
14. **Rejection-Sampler**: vergleicht gegen `input_batch.input_ids`, nicht
    gegen die `-1`-Platzhalter des Schedulers.
15. **`draft_tokens.zero_()`**: steht in `add_request`, laeuft nicht je Schritt.

---

## 6. Split-Graphgroessen: WIDERLEGT (Korrektur 07.09. abends)

`cudagraph_utils.py::_use_split_sm70_mtp_cudagraphs` teilt die
Aufzeichnungsgroessen auf: Ziel in **Token** (`decode_query_len=4`, sizes
`(4,8)`), Drafter in **Anfragen** (`decode_query_len=1`, sizes `(1,2)`).
Standardmaessig aus, zusaetzlich auf sm70 beschraenkt
(`is_device_capability((7,0))` prueft EXAKT) — die RTX 8000 ist sm75.

**Die urspruengliche Behauptung "Split an -> NaN verschwindet" war FALSCH.**

Der Schalter HAT gegriffen (Boot-Log: "Using split SM70 MTP CUDA graph manager
shapes ..." mit korrekter Aufteilung). Das NaN bleibt trotzdem, ab Schritt 4,
bitgleich zum Lauf ohne Split. Der Fehler lag beim Messen: `tail -2` auf die
Marken MITTEN im Lauf zeigte die Prefill-Schritte 1-3, die den
Spekulationspfad gar nicht durchlaufen.

**Lehre fuer die Nachfolge: Marken erst nach `FERTIG` auswerten, und immer den
ganzen Verlauf, nie den Schwanz.**

Der Mechanismus bleibt trotzdem konzeptionell interessant (die Token/Anfragen-
Unterscheidung ist real), aber er ist NICHT die Ursache und kein Ansatzpunkt.
Patch unter `logs/mtp-v2-cudagraph-2026-09-07/scripts/experiment_cudagraph_utils.py`,
NICHT deployt.

## 7. Naechste Schritte (Vorschlag)

1. **Punkt 12 der Ausschlussliste messen statt lesen.** Der gesamte
   FULL-Replay haengt daran, dass die Laufzeit in DIESELBEN Puffer schreibt,
   die beim Capture eingebacken wurden. Bisher nur am Code geprueft.
   `data_ptr()` synchronisiert nicht und laesst sich daher auch waehrend der
   Aufzeichnung erfassen. Dazu die echten `seq_lens` zur Replay-Zeit — beim
   Capture setzt `make_dummy` seq_len == query_len == 4, zur Laufzeit ist der
   Kontext 15+ Token lang.
2. **Diskriminator TRITON_ATTN gegen FLASH_ATTN.** Trennt in einem Lauf die
   FA2-Spur von GDN und den Quant-Kernels.
3. **Danach** die Fork-eigenen sm75-Schalter einzeln variieren
   (`VLLM_SKINNY_QPN`, `QPN2`, `NVFP4_TURBOMIND`) — sie waren in ALLEN
   Laeufen gesetzt und sind noch nie variiert worden.
4. Erst danach ueber einen Fix entscheiden.

---

## 8. Instrumentierung (UNCOMMITTED, im Arbeitsbaum)

Aktiv ueber `VLLM_SKINNY_PPDIAG=<pfad-prefix>`, pro Rang eine Datei,
`VLLM_SKINNY_PPDIAG_CAP` begrenzt je Marke.

| Datei | Marken |
|---|---|
| `fork_patches_150/skinny_ppdiag.py` | das Modul selbst (fork-eigen, neu) |
| `fork_patches_150/model_runner.py` | `prepare_inputs`, `nonlast.recv`, `last.send_*`, `target.hidden_out` |
| `fork_patches_150/gpu_spec_eagle_speculator.py` | `eagle.prefill`, `eagle.replay_in/out` |
| `fork_patches_150/mamba_hybrid_state.py` | `postprocess_state`, `gdn_metadata` |

**Wichtig:** `mark()` ueberspringt sich waehrend der CUDA-Graph-Aufzeichnung
(`is_current_stream_capturing`) — ohne diese Sperre kracht der Boot mit
`operation not permitted when stream is capturing`.

Zum Entfernen: die vier Dateien loeschen und die drei Deploy-Eintraege
(`skinny_ppdiag.py`, `mamba_hybrid_state.py`, `gpu_spec_eagle_speculator.py`)
aus DEPLOY-TARGETS.txt nehmen; `model_runner.py` per `git checkout` auf HEAD.

---

## 9. Archiv

`logs/mtp-v2-cudagraph-2026-09-07/` — Skripte, alle Generierungen, alle
Marken-Protokolle. `logs/` ist gitignoriert, liegt aber auf der Platte.
Aeltere Beweise zur Verklemmung: `logs/pp-mtp-deadlock-2026-09-06/`.
