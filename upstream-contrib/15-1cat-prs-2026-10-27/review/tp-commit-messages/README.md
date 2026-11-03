# Vorgeschlagene Commit-Nachrichten, Zweig pr-tp-prefill (27.10.2026)

Hashes = heutiger Stand (`git log --abbrev=9 --format='%h %s' pr-turing-kernels..HEAD`, Kopf 5bd8a8d21). Je Datei die vollständige
neue Nachricht (Betreff unverändert). Trailer: Laut Auftrag bleibt das vorhandene ältere Paar Co-Authored-By/Claude-Session stehen,
danach Signed-off-by. Das Paar dieser Sitzung (session_0172gtUiymppDzziDqKfy78Q) ist nicht ergänzt; falls gewünscht, vor
Signed-off-by einfügen.

| Commit | Datei | Warum |
| --- | --- | --- |
| 78d8c2b18 Turing-Prefill-Attention | 78d8c2b18.txt | „about a quarter of the m16n8k8 rate“ durch Messwerte ersetzt (165,8 ms ≈ 23 TFLOPS gegen 61); Auswahl genau beschrieben (Anfrage ≥ 65 statt „batches with at least“, eigenes Gerät, nicht quantisierter Cache, Seiten 16/32/64, 32-Bit-Offsets – nach dem Review eingefaltet); Testzeile ergänzt; „nur sm_75 gebaut“. |
| 24fb9a581 Flash-V100-Verfügbarkeit | 24fb9a581.txt | „failed … on every V100 stage“ nicht belegt (nur der Fehler selbst) → „the first GLM-5.3-Flash boot failed with it“; Testzeile; Zeilen ≤ 72. |
| bfd35e7b5 Überlappung | bfd35e7b5.txt | Nachricht stammte von vor dem Review: Startablehnungen (DCP, Encoder-Decoder, MoE, Deepstack), all_reduce_sum2 über den Comm-Stream, Fehler der fusionierten AR+RMSNorm, Abbruch-Flag; „96 of 208 s“ als Profil MIT Turing-Attention gekennzeichnet; „about 2 GB/s“ entfernt (steht mit Zahl im Text); Testzeilen auf den Zweigstand (14 passed; runner_v2 5 passed, 2 Fehlschläge wie auf main) statt Fork-Zählungen und Fremdfehlern aus Fork-Läufen; Satz „without the comm-done wait the test fails“ gestrichen (auf dem Zweig nicht geprüft). |
| cacae3976 Copy-Engine-Modus | cacae3976.txt | Kern-Zuordnung falsch („two largest … 31 and 37 %, a smaller one 7 %“): +37 % ist der FP8-Kern (285 µs), kleiner als der +7-%-NVFP4-Kern (570 µs); FP8 3840 (+18 %) fehlte → „NVFP4 31 und 7 %, FP8 18 bis 37 %“; GEMMs „5 to 22 %“ → „largest GEMM 22 % per tile row“ (nur das nachgerechnet); „production profile“ → „baseline profile“; „8 MiB“ → 8,192,000 B (Bandbreiten passen nur dazu); eager; Capture-Regel und copy_engine_max_bytes > max_bytes ergänzt; CE-Profil 95,85 statt 118,22 s und Restverlangsamung 4–11 % ergänzt; Testzeile mit Capture-Fall. |
| 7c9951aa2 Split-KV | 7c9951aa2.txt | „production“ → „without split-KV“; Testzahl 16 → 21 (Auswahlfälle aus Commit 1 kamen dazu). |
| 5bd8a8d21 NVFP4-Kachel | – | Keine Fehler gefunden (Zahlen 2,20–2,32×, 11 passed je Karte stimmen mit dem Lauf 18:39/18:40). Nur Signed-off-by ergänzen. Offen: Welches Triton beim Sweep lief, steht nirgends. |

Testzahlen in den neuen Nachrichten: Läufe `~/.cache/bench-scripts/prtests_2026-10-27/` auf 5bd8a8d21 (tp-final-rtx 18:39, tp-final-v100
18:40, tp-final-pair 18:41, tp-overlap-rtx 18:43). Die Datei test_prefill_overlap.py ist in den späteren Commits unverändert, ebenso
test_paged_prefill_attention_sm75.py nach 7c9951aa2 und test_host_reduce.py nach cacae3976, die Zählungen gelten also für den
jeweiligen Commit. Nach dem Schlusslauf abgleichen.
