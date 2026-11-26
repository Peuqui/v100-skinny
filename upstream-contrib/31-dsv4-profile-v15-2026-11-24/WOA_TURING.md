# DeepSeek-V4 wo_a auf den RTX-Stufen direkt aus FP8 (Rang 1 aus ANALYSE.md)

Zweig `dsv4-wo-a-turing-fp8` (Baum `1Cat-vLLM-woa`, noch nicht gepusht).

## Commits

- 6b8d41945: gruppierter QPN8-Start für wo_a (8 Gruppen à 1024 × 4096, Block-FP8), keine FP16-Kopie mehr auf Turing (−0,74 GB auf den
  zwei RTX-Stufen). M ≤ 8 ein Start über alle Gruppen, größere M je Gruppe über die Leiter der Dense-Linears.
- fa3ae4c7c: über 128 Zeilen je Gruppe das ganze Gewicht in den geteilten Prefill-Arbeitsbereich entpacken (wo_a hat so viele Elemente
  wie wq_b und wo_b, der Bereich wächst nicht) und ein `at::bmm_out` über alle Gruppen. Grenze 128 aus den Zahlen von
  `~/.cache/bench-scripts/woa_2026-11-24/bench_c0.log` abgeleitet, Messung mit dem neuen Benchmark-Arm `qpn8_dense_bmm`.
- bd9fdf7ae (fp32-Reduktion als Scope im QPN8-Prefill-Op) per Rebase entfernt: Der Fehler bei M = 1024 trat nur im Test auf. Jeder Worker
  ruft vor dem Laden `set_high_precision_cuda_matmul_defaults()` (`vllm/v1/worker/gpu_worker.py`), das Boot-Log vom 24.11. 00:01 meldet
  „CUDA matmul precision policy: … disabled“. Die Produktion war nie betroffen. Tests und Benchmark setzen jetzt dieselbe Richtlinie.

## Erste A/B (`~/.cache/bench-scripts/ab_woa_dsv4_2026-11-24a.log`, Stand 6b8d41945 + bd9fdf7ae)

T3 gegen P4 (T1/P2 mit Qualitätsfragen):

| Messgröße | T (wo_a FP8) | P (v15) | Urteil |
|---|---|---|---|
| Prefill 8921 Token (s, weniger ist besser) | 2,41 (3704 tok/s) | 2,29 (3897 tok/s) | +5 %, schlechter |
| greedy Runde (ms, weniger ist besser) / tok/s (mehr ist besser) | 48,7 / 67,36 | 50,2 / 64,75 | −1,5 ms |
| AIfred-Sampling alle Runde (ms) / tok/s | 49,4 / 56,94 | 50,5 / 57,94 | Runde −1,1 ms; tok/s folgt der Annahme (1,87 gegen 1,93 je Runde) |
| GSM8K-Decode (tok/s, mehr ist besser); richtig | 75,49; 30 | 72,14; 28 | +4,6 % |

Ursache des Prefill-Rückstands: Bei M > 8 lief jede Gruppe einzeln (Mikrobenchmark RTX `bench_c0.log`: je Gruppe 1017 / 1988 µs bei
M = 512 / 1024 gegen 495 / 964 µs für das bmm auf dem fertig entpackten Gewicht). Behoben in fa3ae4c7c, zweite Kette
`chain_woa_2026-11-24b.sh` (Ausgaben `~/.cache/bench-scripts/woa_2026-11-24b/`, A/B `ab_woa_dsv4_2026-11-24b.log`).

## Texte der ersten A/B (händisch gelesen, 24.11. abends)

T1 gegen P2: prime und english tokengleich. motor, zug, kuanda, logik, code_bug und summary anders formuliert, alle richtig: Zug in beiden
84 km/h (T rechnet 287 / 3,4167, P über 41/12), Kuanda in beiden zurückgewiesen, Logik in beiden David jüngster und Ben zweitältester,
code_bug in beiden `+ 1` als Fehler mit derselben Korrektur. Motor und Zusammenfassung unterscheiden sich in einzelnen Wörtern
(„erwärmt“/„erhitzt“, „freigesetzt“/„als Nebenprodukt entsteht“).

## Kette c (24.11. 20:31, `~/.cache/bench-scripts/woa_2026-11-24c/`): Tests und Mikrobenchmark

Kette b war an der neuen Test-Fixture gescheitert (Aufräumen schrieb das nur lesbare torch-Attribut `…_split_k`; 66 Teardown-Fehler,
0 FAILED); Fixture stellt jetzt das Tupel über den Basisnamen zurück (im Commit e336f616e). Kette c ohne Bau: Tests 87/87 auf RTX und V100,
einschließlich der Bitgleichheit des bmm-Wegs gegen das alte einsum bei M = 129–1024.

Mikrobenchmark wo_a, µs je Aufruf im CUDA-Graphen (weniger ist besser), Worker-Richtlinie gesetzt:

| M | RTX FP16-einsum (Produktion) | RTX QPN8 neu | RTX nur bmm-Weg | V100 QPN8 gruppiert | V100 TurboMind (Benchmark, ungetunt) |
|---|---|---|---|---|---|
| 1 | 114,1 | 58,5 | 348,1 | 47,2 | 61,4 |
| 6 | 122,5 | 58,9 | 345,1 | 46,6 | 128,1 |
| 16 | 125,0 | 148,1 | 346,4 | 228,1 | 105,2 |
| 64 | 132,1 | 248,1 | 354,3 | 549,4 | – |
| 128 | 144,0 | 353,5 | 364,5 | 1040,2 | – |
| 256 | 257,3 | 450,3 (bmm) | 446,4 | – | – |
| 512 | 499,3 | 664,8 (bmm) | 664,3 | – | – |
| 1024 | 967,1 | 1107,4 (bmm) | 1107,8 | – | – |

Urteil: Grenze 128 auf der RTX bestätigt (bis 128 ist der Gruppenweg 3 % schneller als das bmm). Gegen die Produktion bleibt der Prefill
langsamer: +140 µs je Aufruf bei M = 1024, das ist das Entpacken (≈ 100 MB Verkehr). Decode (M ≤ 8) halbiert sich auf der RTX.

V100-Nebenbefund: Der Benchmark-TurboMind ist nicht getunt; in der Produktion braucht wo_a auf den V100-Stufen 59,4–60,4 µs (nsys,
ANALYSE.md 4.2). Echt bleibt QPN8 gruppiert 46,5 gegen ≈ 60 µs, ≈ 13 µs je Aufruf, ≈ 0,3 ms je Runde (Rang 7). Haken: Über M = 8 ist QPN8 auf
der V100 viel langsamer (M = 16: 228 gegen 105 µs), also bräuchte die V100 QPN8 für Decode und TurboMind für Prefill, zwei Kopien
(+33,6 MB je Schicht auf den V100-Stufen) wie bei den Dense-Layern (`_keep_volta_turbomind_copy`). Offen, nach der A/B entscheiden.

A/B: Kette d (`chain_woa_2026-11-24d.sh`, Variante `woa2`, weil die Cache-Kopien des ersten Laufs noch liegen), Log `ab_woa2_dsv4_2026-11-24d.log`.

## A/B Kette d (`~/.cache/bench-scripts/ab_woa2_dsv4_2026-11-24d.log`, T P T P, 20:35–21:30), Stand 6b8d41945 + e336f616e

| Messgröße | T (wo_a FP8) T1 / T3 | P (v15) P2 / P4 | Unterschied |
|---|---|---|---|
| Prefill 8921 Token (tok/s, mehr ist besser; s) | 3891 / 3888 (2,29 / 2,29 s) | 3910 / 3911 (2,28 / 2,28 s) | −0,5 % (≈ +10 ms) |
| Erstes Token 150k / 250k / 486k (s, weniger ist besser; je eine Messung) | 41,13 / 73,71 / 147,19 | 40,94 / 73,32 / 146,79 | +0,5 / +0,5 / +0,3 % |
| Decode kurz (tok/s, mehr ist besser; ms/Token) | 64,44 / 64,57 (15,5) | 61,20 / 61,13 (16,3–16,4) | +5,4 % |
| GSM8K-30-Decode (tok/s); richtig | 75,29 / 75,32; 29 / 29 | 72,10 / 72,15; 28 / 28 | +4,4 % |
| greedy Runde (ms, weniger ist besser) | 48,7 / 48,7 | 49,9 / 50,2 | −2,7 % |
| AIfred-Sampling alle: Runde (ms); tok/s (Annahme je Runde) | 49,3 / 49,3; 58,60 / 58,07 (1,91 / 1,93) | 50,5 / 50,5; 57,92 / 57,91 (1,93 / 1,93) | Runde −2,4 %, tok/s +0,7 % |
| Langkontext je Schritt 150k / 250k / 486k (ms, weniger ist besser) | 56,5 / 60,9 / 63,2 | 58,0 / 61,8 / 64,8 | −2,6 / −1,5 / −2,5 % |
| GPU-Spitze Karte 0 / Karte 2 (MiB, weniger ist besser) | 38 221–38 281 / 46 573–46 633 | 38 581–38 641 / 46 953–47 013 | −0,3 bis −0,4 GiB je RTX |

Texte: T1 und T3 wortgleich zum ersten Lauf (T deterministisch); P hat zwischen erster und zweiter A/B selbst bei zug, logik und code_bug
anders formuliert. T gegen P: prime und english tokengleich, die übrigen sechs anders formuliert, alle richtig (gelesen, siehe oben; neue
P-Fassungen von zug 84 km/h, logik David/Ben, code_bug `+ 1` ebenfalls richtig).

Urteil: Decode klar besser (kurz +5,4 %, Langkontext −1,5 bis −2,6 % je Schritt, Speicher −0,4 GiB je RTX), Prefill und erstes Token
langsamer um 0,3 bis 0,5 %. Rest-Ursache: das Entpacken des ganzen Gewichts je Prefill-Stück (≈ 140 µs je Schicht bei 1024 Zeilen, am
Bandbreitenlimit). Nach der Regel „keine andere Größe schlechter“ nicht ohne Entscheidung ins Bündel; Rückfrage an Peuqui 24.11. 21:40.

## Entpacken überlappt (Zweig `dsv4-wo-a-prefill-overlap` 5c14dc309, 25.11.2026): verworfen

Idee: Gruppe g+1 auf einem Nebenstrom entpacken, während Gruppe g auf dem Hauptstrom multipliziert wird (ein `at::mm_out` je Gruppe statt
eines bmm). Kette `chain_woaov3_2026-11-25.sh`, Tests 87/87 beide Karten. Ergebnis schlechter:
- Mikrobenchmark RTX (`woa_2026-11-24ov3/bench_c0.log`, µs je Aufruf, weniger ist besser): M = 512 1588 gegen 664 (bmm-Weg v16), M = 1024 1587 gegen 1107.
- A/B `ab_woa3_dsv4_2026-11-24e.log` (T P T P gegen v16): Prefill 8,9k 2,36 / 2,36 gegen 2,29 / 2,29 s (+3 %), erstes Token 150k/250k/486k
  42,28 / 75,41 / 150,25 gegen 41,13 / 73,64 / 147,30 s; Decode gleich.
- Ursache (Diagnose `woa_mm_diag.py` im Scratchpad, 25.11. 03:05, µs, Worker-Richtlinie gesetzt): cuBLAS ist für die Einzelform [M, 4096] × [4096, 1024]
  viel langsamer als für das gebatchte Produkt, auch mit zusammenhängendem Speicher. RTX: M = 512 bmm 539, 8 × mm 1805; M = 1024 1055 gegen 1450.
  V100: M = 512 625 gegen 725; M = 1024 1053 gegen 1502.
Übrig bliebe ein bmm in zwei Hälften mit Überlappung (≈ die Hälfte der 0,5 % Prefill zurück, ≈ 0,25 %); nach Peuquis Regel für Kleinkram nicht verfolgt.
Produktion bleibt v16 (bmm-Weg).
