# Nachtbericht 2026-09-05/06 — autonome Arbeit (Claude)

Auftrag (Peuqui, 2026-09-05 spät): Commit/Push, Watcher auf das Ende der laufenden
27B-Kalibration, Analyse des Laufs, Plan A (Dense-Prefill) scharfschalten, Neukalibration,
danach die drei Fragen im AIfred (Quantenphysik / Regenbogeneffekt / Kuanda-Effekt, je 30 Sätze).

Stand wird fortlaufend ergänzt; Zeiten lokal.

## 1. Commit/Push (erledigt 22:1x)
- v100-skinny `work` de0d358 (Plan A, env-gesperrt) auf Peuqui/v100-skinny.
- 1Cat-Branches auf Peuqui/1Cat-vLLM; offene PRs #511, #512, #514, #516, #518.

## 2. Laufende Kalibration (gestartet 21:38 durch Peuqui, DROP_CT=1 + Profiling-Fix aktiv)
- Watcher läuft (Log-Marker „Operating point saved" / Fehler).
- Bisher: TP1/RTX cappt bei 249k → 238k statt 52k (Profiling-Fix wirkt; ~4 % Rest durch
  kalten Compile, real belegt).

## 3. Analyse des Laufs (beendet 00:54, 3 h 16 min)

Sollverhalten, das erfüllt ist:
- TP1/RTX: Cap 262k → 249k → 238k, boot OK bei 238.336 (statt 52k gestern). Profiling-Fix + DROP_CT wirken.
- V100-Paar: boot OK bei GMU 0,97 ohne Sonden-OOM (das Speicherprofil lässt jetzt Luft), 40,7 / 35,3 tok/s.
- k-Sweep mit Produktions-Sampling: RTX-Paar k=3 gewinnt (53,6 tok/s lang, 68 % Akzeptanz) vor k=2 (48,6) und k=5 (46,5).
  Grid k=3 41,8, TP1/RTX k=3 36,3 bei 168k (Speed-Kandidat, nur Info).
- Bester Punkt: TP2 auf dem RTX-Paar, k=3, 262.144 Kontext — gegenüber gestern (Grid k=5, 51,2) ein Wechsel.

FEHLER gefunden:
- V100-Paar, k-Sweep: alle sieben k-Boots mit „fatal error during boot: CUDA out of memory" — im
  Boot-Log: `Failed to run autotuning code block: CUDA out of memory. Tried to allocate 1.19 GiB, 998 MiB free`.
  Ursache: der Compile der Spekulationsgraphen (k>0) läuft NACH der KV-Zuteilung (Warmup), sein
  Autotune-Scratch findet keinen Platz mehr. Die Sweep-Boot-Fehlerbehandlung kennt keinen OOM-Schritt
  (nur „braucht größeren Chunk"/„kleineren Kontext"); die Sonden-OOM-Leiter greift erst nach dem Boot.
  Folge: V100-Paar nur mit k=0 bewertet (35,3), obwohl es mit k>0 vermutlich ~45-50 erreichen würde.
  Fix (AIfred, nach Ende des Laufs, weil Hot-Reload): Boot-OOM im Sweep → GMU −0,02 und Retry,
  gelernte GMU für die restlichen k behalten (gleiche Regel wie beim Sonden-OOM). Test dazu.
- Kein weiterer Fehler; keine verworfene Sprosse, keine Inkohärenz.

Ergebnis persistiert 00:54: Betriebspunkt TP2 auf dem RTX-Paar (GPUs 0,2), k=3, GMU 0,96 (A/B),
Chunk 2048, ctx 262.144, 53,6 tok/s. llama-swap neu gestartet, Eintrag trägt DROP_CT=1.
Speed-Kandidat TP1/RTX k=3 36,3 tok/s bei 168k (nur Info, unterliegt).

Fix eingebaut und gepusht (AIfred 270493ea): Boot-OOM im k-Sweep → GMU −0,02 und Retry, gelernte
GMU bleibt für die restlichen k; Test dazu; alle 29 Kalibrationstests grün. Greift beim nächsten Lauf,
das V100-Paar bekommt dann seinen k-Sweep.

## 4. Plan A auf der RTX — End-to-End bestanden (00:56–01:2x, GPU 2, 27B TP1, GMU 0,9, mml 32768)

| | Marlin (DROP_CT=1) | Dense-Prefill (Plan A) |
|---|---|---|
| Ladevolumen | 29,49 GiB | 19,07 GiB (V100: 20,21) |
| KV-Budget | 11,09 GiB / 157.696 Token | 19,27 GiB / 274.432 Token |
| greedy-Antworten (Coandă, Primzahlen) | Referenz | byte-identisch |
| Prefill 15.624 Token, kalt | 341 tok/s | 414 tok/s (+21 %) |
| Routen | M>16 marlin | M>16 dense, M<=8 qpn2 (unverändert) |

Selbstcheck der QPN-Kernel gegen die cuBLAS-Referenz: rel. Fehler 6e-4. Keine Fehler im Log.
Damit hat die RTX 8000 dasselbe Gewichtsvolumen wie die V100 — ein Layout.

## 5. Scharfgeschaltet + Neukalibration
- 01:2x: `VLLM_SKINNY_DENSE_PREFILL: "1"` in data/vllm_runtime.yaml base_env (Backup
  backups/vllm_runtime.yaml.pre-dense-*). Gilt für Kalibrationsboots und die daraus gerenderten
  llama-swap-Einträge.
- Neukalibration über den Standalone-Runner gestartet (gleicher Flow wie die UI, llama-swap gestoppt,
  wird am Ende neu gestartet). Log: Scratchpad calibration_night.log. Enthält jetzt auch den
  Sweep-Boot-OOM-Fix, das V100-Paar bekommt seinen k-Sweep.

## 6. Kalibration #2 (Dense-Prefill, 01:05–02:41) — Ergebnis und Abbruch

Log: logs/calibration-2026-09-06-night-run1.log (Kopie des Scratchpad-Logs).

Topologien (alle ctx 262.144, kein Cap mehr nötig — mit Dense-Prefill passt der volle Kontext
überall auf Anhieb):

| Topologie | k=0 kurz / lang | bestes k | lang tok/s | Akzeptanz |
|---|---|---|---|---|
| TP1 RTX 8000 | 26,7 / 23,4 | — (Sweep abgebrochen, s.u.) | 23,4 | — |
| TP2 RTX-Paar | 40,7 / 36,3 | k=3 | 52,1 | 68 % |
| TP2 V100-Paar | 40,6 / 35,2 | k=2 | 51,0 | 97 % |
| TP2×PP2 Grid | 39,7 / 33,0 | k=3 | 42,1 | 66 % |

Sollverhalten erfüllt:
- Sweep-Boot-OOM-Fix greift: V100-Paar k=7 „boot hit OOM: retry with GMU 0.95", danach Sonden-OOM
  → 0,93, Rest des Sweeps läuft durch. Das V100-Paar hat damit erstmals einen vollständigen k-Sweep.
- V100-Paar k=2 mit 97 % Akzeptanz ist der zweitbeste Punkt (51,0 gegen 52,1 RTX-Paar).
- Grid ist mit 42,1 klar hinter beiden Paaren (PP-Latenz), wie gestern.

FEHLER: Lauf um 02:41 gestorben, TP1/RTX-Sweep k=7 „EngineCore failed to start", ab k=6 „exit 120"
im Sekundentakt. Ursache: Root-FS zu 100 % voll — ~/.cache/vllm/torch_compile_cache war auf 103 GB
gewachsen (jeder Kalibrationsboot legt ~0,7 GB neuen Compile-Cache an, eigener Key pro
max-model-len/k). ENOSPC hat auch das Persistieren des Betriebspunkts verhindert; der
Betriebspunkt aus Lauf #1 (RTX-Paar k=3, 53,6) bleibt gültig und deckt sich mit dem Sieger aus #2.

Aufräumen 03:xx: Compile-Cache älter als 6 h gelöscht → 98 GB frei (jetzt 6,3 GB Cache, 97 GB frei).
Offener Punkt: Policy für den Compile-Cache bei Kalibrationsboots (nach dem Lauf löschen oder
Cache für Kalibrationsboots abschalten) — Vorschlag an Peuqui, nicht umgesetzt.

## 7. Produktionsboot mit Dense-Prefill (06:29–06:37, llama-swap, RTX-Paar TP2 k=3)

- Model loading took 10,18 GiB pro Karte (vorher ~14,7 mit DROP_CT, ~25 mit drei Layouts)
- Available KV cache memory: 31,97 GiB pro Karte
- GPU KV cache size: 941.578 Token gepoolt = 3,59× 262.144 (Maximum concurrency)
- Cold Start bis erste Antwort 7:24 min (Compile ohne passenden Cache-Key nach dem Aufräumen)

## 8. Die drei Fragen (AIfred, Headless-Chrome, User ki, Modus Standard, Temperatur 1,0)

| Frage | TTFT | PP | Decode | Dauer | Token |
|---|---|---|---|---|---|
| Quantenphysik in 30 Sätzen | 5,83 s | 543,9 tok/s (2.974) | 72,5 tok/s | 1:06 min | 4.352 |
| Regenbogeneffekt in 30 Sätzen | 6,13 s | 535,2 tok/s (3.228) | 56,1 tok/s | 58,3 s | 2.924 |
| Kuanda-Effekt in 30 Sätzen | 7,27 s | 547,6 tok/s (3.954) | 57,7 tok/s | 25,0 s | 1.023 |

Q1 und Q2: kohärente, vollständige Butler-Antworten (Q1 ca. 30 Sätze, Q2 ca. 30 Sätze, Fachinhalt
korrekt: Descartes/Newton/Alexander-Band/Supernumerary). Kleinere Sprachrutscher wie sonst auch
(„rather", „indeed", „same", „exists" mitten im Deutschen; „Böcken-Bogen" ist erfunden).
Q3: AIfred hat den absichtlichen Schreibfehler diesmal NICHT als Coandă gedeutet, sondern
nachgefragt (Kundt-Rohr, Kundalini, …). Am 01.09. (Marlin-Pfad) wurde Coandă erkannt. Bei
Temperatur 1,0 ist das ein Sampling-Ergebnis, kein Beleg gegen Dense-Prefill: der Greedy-Vergleich
in Abschnitt 4 war auf genau dieser Frage byte-identisch. Zum Absichern: die Frage einmal
wiederholen (oder greedy) — steht als Punkt bei Peuqui.

Chat unter „Merken" gespeichert (Titel „Quantenphysik in 30 Sätzen erklärt"), abgemeldet,
Headless-Chrome beendet.

## 9. GitHub-Stand 07:00

- Upstream main unverändert auf 755baae (unsere Basis). Fünf neue Fremd-PRs (#515, #517, #519–#523,
  SM70-Kernel/AWQ-QPN, FlashInfer-GDN) — keine Überschneidung mit unseren Dateien.
- Unsere PRs #511/#512/#514/#516/#518: alle OPEN, MERGEABLE, keine Reviews. Der rote Check
  „pre-run-check" ist die Repo-Regel „Label ready/verified oder ≥4 gemergte PRs des Autors
  (gefunden 2)" — kein Fehler unsererseits, wartet auf ein Maintainer-Label.
- #516: Frage von DSYZayn (02:44Z), ob #485 nur Qwen4Exp ist; DSV4-Flash TP4×PP2 gehe weiter
  nicht. Antwort-Entwurf: upstream-contrib/03-1cat-issues/reply-516-dsyzayn-dsv4-pp.md — NICHT
  gepostet, wartet auf Freigabe.

## 10. Langer Greedy-Vergleich Marlin gegen Dense-Prefill (09:12–09:24, RTX PCI 2, TP1, mml 32768)

| Prompt | Marlin | Dense | Ergebnis |
|---|---|---|---|
| Quantenphysik 30 Sätze (greedy, max 1500) | 875 tok / 32,2 s | 875 tok / 31,3 s | byte-identisch |
| Regenbogeneffekt 30 Sätze | 454 tok / 16,2 s | 467 tok / 16,7 s | Divergenz nach 1.164 von ~1.900 Zeichen, beide kohärent |
| Kuanda-Effekt 30 Sätze | 488 tok / 17,3 s | 488 tok / 17,5 s | byte-identisch |
| 22.437-Token-Prompt, Zusammenfassung | 96 tok / 71,2 s | 96 tok / 59,9 s | byte-identisch, Prefill +19 % |

Ladevolumen 29,49 → 19,07 GiB, KV 11,09 → 19,28 GiB (157.696 → 274.432 Token). Bewertung: die eine
späte Divergenz ist die erwartete Rundungsdifferenz cuBLAS-fp16 gegen Marlin; Dense-Prefill bleibt scharf.
Skript: Scratchpad dense_bisect_long.sh, Ausgaben dense_long_{0,1}.json.

## 11. Kalibration #3 (09:57–13:38, Dense-Prefill, Cache beim Start geleert)

Vollständig, persistiert 13:38. Sieger nach der Gesamtzeit-Regel: **TP2 auf dem V100-Paar (GPUs 1,3),
k=2, 51,7 tok/s**, GMU 0,94 (Boot-OOM-Leiter hob die Reserve einmal), Chunk 2048 (4096 verliert: 51,6),
ctx 262.144. RTX-Paar k=3 52,1 (Prefill 513 vs 556 — daher 1,5 % Turnzeit hinten). Grid k=3 42,1,
TP1 k=3 34,9. Matrix im Log logs/calibration-2026-09-06-run3.log.

Befunde:
- Geleerter Compile-Cache kostete ~5 min je RTX-Boot (Sonden 8 statt 3 min), ~2 h Laufzeit — daher
  jetzt Obergrenze 40 GiB statt Leeren (AIfred 38648bb5).
- TP1-Sweep sinnlos nach dem Paar derselben Klasse (Faktor 1,49 vs 1,44) — Sweep-Verzicht nach
  Klasse (AIfred 24857770), hardwareagnostisch (eine Karte: TP1 wird gesweept).
- Kurzkontext: RTX-Paar 61,2 vs V100-Paar 55,5 (+10 %); KV-Pool RTX 3x. Entscheidung RTX vs V100 offen
  (Peuqui, Single-User), Nachmessmodus kann den RTX-Punkt in ~30 min persistieren.
- Sweep-Zeiten: V100-Sonden 3,4–4,1 min, RTX-Sonden 8 min (kalt), k=1 auf RTX 3,0 (Cache-Treffer von k=7/3).

## 12. Nachmesslauf RTX-Paar (13:50–14:29, Nachmessmodus, warmer Cache) — Betriebspunkt RTX

`scripts/vllm_resweep.py Qwen3.8-27B-NVFP4-vllm --topology "TP2 across RTX 8000 class"` (AIfred, neuer
Nachmessmodus). Sonden je 3,3 min (Cache-Treffer). Ergebnis: **TP2 RTX-Paar (GPUs 0,2), k=3, 52,1 tok/s
lang / 61,2 kurz, Prefill 516**, Chunk 2048 (4096: 52,2, kein Gewinn), GMU 0,98 (0,96: 51,7, verliert).
Persistiert 14:29, llama-swap neu gestartet, Eintrag ohne DROP_CT/DENSE-Env-Zeilen (Fork-Defaults).
Entscheid Peuqui: RTX wegen Kurzkontext (+10 %); Single-User, KV-Pool kein Kriterium. Neue Siegerregel
(AIfred b7e75a0d): Turnzeit am Alltagskontext (12.000 Token, gewichtet zwischen Kurz- und Langpunkt,
Prefill ueber die Lastannahme) — haette die RTX direkt gewaehlt.
Matrix: data/logs/vllm_calibration/Qwen3.8-27B-NVFP4-vllm/measurement-matrix.json.
