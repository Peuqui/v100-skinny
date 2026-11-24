# E6: Skalen-Spanne je K-Scheibe, GLM-5.3-Flash NVFP4 (23.11.2026)

Nur Host, nur lesend, keine GPU, kein Bau. Skript: `e6_scale_spread.py` (Ausgabe `e6_scale_spread.out`, Rohdaten `e6_scale_spread.json`).
Aufruf: `CUDA_VISIBLE_DEVICES="" PYTHONPATH=/home/mp/Projekte/vllm-research/1Cat-vLLM-v14 python e6_scale_spread.py --json e6_scale_spread.json` (cwd ausserhalb des Baums, 101 s, 3 Prozesse).

## Methode und Belege

- **Daten:** echter Checkpoint `/home/mp/models/GLM-5.3-Flash-NVFP4-MTP`, je Schicht alle 288 Experten, W13 = gate (2048 Zeilen) + up (2048 Zeilen) = [4096, K/16 = 256 Gruppen], W2 = down [4096, 128 Gruppen]. Gelesen wurden nur die csf-Teile (`csf32_bases/codes/page_bases/offsets/exceptions`) je Matrix mit `safe_open`, zerlegt mit dem echten Decoder `decode_csf32` (`vllm/model_executor/layers/quantization/utils/csf32_scales.py`), dessen Ergebnis der GPU-Expander nachbildet (Tests `tests/kernels/quantization/test_csf32_scales.py`).
- **S5-Entscheidung (a):** `vllm/model_executor/layers/fused_moe/experts/skinny_sm70_moe.py:585-587` (`_csf_to_s5`): `spread = scale_spread(raster.view(experts, -1), planes.groups)`; `fits = spread <= S5_MAX_SPREAD`; `S5_MAX_SPREAD = 31` in `vllm/model_executor/layers/quantization/utils/s5_scales.py:33`. Die Spanne ist je Zeile (Kachelposition) max minus min der u8-Skalenbytes ueber alle K-Gruppen, danach das Maximum ueber alle Zeilen UND alle 288 Experten der Projektion: ein Experte reicht, und die ganze Projektion der Schicht bleibt csf. Entscheidung also je (Schicht, Projektion), nicht je Experte.
- **Prepack:** `qpn_prepack` (`skinny_sm70_moe.py:154`) ist eine reine Permutation der Spalten innerhalb der 32er-Kachel; die Spanne je Zeile haengt davon nicht ab. Das Skript rechnet direkt auf der dekodierten Matrix [N, K/16] und belegt je Projektion an den ersten zwei Experten (beide Fragment-Ordnungen M8N8K4 und M16N8K8) mit `assert`, dass echter `qpn_prepack` + echtes `scale_spread` (bzw. dieselbe Scheiben-Rechnung auf dem Raster) dieselben Zahlen liefern. Bei Spanne <= 31 zusaetzlich die Rundreise `decode_s5(encode_s5(raster)) == raster`. Alle Asserts bestanden (Lauf ohne Abbruch).
- **K-Scheibe (b):** `csrc/sm70_turbomind/ops/skinny_moe_qpn_sm70.cu:263-264` (V100, m8n8k4) und `:529-530` (RTX, m16n8k8): `G = K >> 4; Gq = G / SPLITK; g0 = warp * Gq` (RTX `gend = g0 + Gq`): jeder Warp bearbeitet eine **zusammenhaengende** Scheibe von Gq Gruppen zu 16 K. `grouped_splitk` (`skinny_sm70_moe.py:81-97`) gibt fuer K = 4096 und K = 2048 auf beiden Karten SPLITK = 8 (vom Skript geprueft). Scheibe = 32 Gruppen (512 K) bei W13, 16 Gruppen (256 K) bei W2. Das S5-Basisfeld ist heute je (Experte, Kachel, Position) (`kern.cu:272-276` bzw. `:545-552`), eine Basis je Scheibe waere also 8 Basen je (Experte, Kachel, Position).
- **Spalten der Tabelle:** `max Spanne global` = Wert, den der Code heute misst; `je Scheibe` = max ueber Zeilen, Experten und die 8 Warp-Scheiben; `je halbe/Viertel` = Zusatz (Scheiben mit 16/8 bzw. 8/4 Gruppen), falls die Warp-Scheibe nicht reicht; Prozentwerte mit absoluter Anzahl; `6 Bit global reicht` = Spanne <= 63; `Bits` = ceil(log2(Spanne+1)) des Maximums (also Bits je Skala mit globaler Basis je Zeile bzw. mit Basis je Scheibe). (c)(i) = Spalte `Bits je Scheibe` (5 = Basis je Scheibe + 5 Bit reicht fuer alle Experten), (c)(ii) = `6 Bit global reicht`.

## Ergebnis je Schicht und Projektion (Einheit: Spanne in e4m3-Bytewerten, weniger ist besser)

| Schicht | Proj. | max Spanne global (Bytes e4m3) | max Spanne je Scheibe (Warp, Gq Gruppen) | je halbe Scheibe | je Viertel | Experten global <= 31 % (Anzahl) | Experten Scheibe <= 31 % (Anzahl) | Zeilen global > 31 % (Anzahl) | Zeilen-Scheiben > 31 % (Anzahl) | 6 Bit global reicht (<= 63) | Bits global | Bits je Scheibe |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 14 | W13 | 29 | 26 | 25 | 24 | 100.00 (288/288) | 100.00 (288/288) | 0.00 (0/1179648) | 0.00 (0/9437184) | ja | 5 | 5 |
| 14 | W2 | 26 | 26 | 26 | 26 | 100.00 (288/288) | 100.00 (288/288) | 0.00 (0/1179648) | 0.00 (0/9437184) | ja | 5 | 5 |
| 35 | W13 | 28 | 28 | 26 | 26 | 100.00 (288/288) | 100.00 (288/288) | 0.00 (0/1179648) | 0.00 (0/9437184) | ja | 5 | 5 |
| 35 | W2 | 27 | 25 | 24 | 24 | 100.00 (288/288) | 100.00 (288/288) | 0.00 (0/1179648) | 0.00 (0/9437184) | ja | 5 | 5 |
| 36 | W13 | 29 | 28 | 28 | 27 | 100.00 (288/288) | 100.00 (288/288) | 0.00 (0/1179648) | 0.00 (0/9437184) | ja | 5 | 5 |
| 36 | W2 | 36 | 33 | 31 | 31 | 99.65 (287/288) | 99.65 (287/288) | 0.00 (1/1179648) | 0.00 (1/9437184) | ja | 6 | 6 |
| 37 | W13 | 28 | 25 | 25 | 25 | 100.00 (288/288) | 100.00 (288/288) | 0.00 (0/1179648) | 0.00 (0/9437184) | ja | 5 | 5 |
| 37 | W2 | 30 | 28 | 28 | 28 | 100.00 (288/288) | 100.00 (288/288) | 0.00 (0/1179648) | 0.00 (0/9437184) | ja | 5 | 5 |
| 38 | W13 | 29 | 28 | 27 | 27 | 100.00 (288/288) | 100.00 (288/288) | 0.00 (0/1179648) | 0.00 (0/9437184) | ja | 5 | 5 |
| 38 | W2 | 31 | 26 | 24 | 24 | 100.00 (288/288) | 100.00 (288/288) | 0.00 (0/1179648) | 0.00 (0/9437184) | ja | 5 | 5 |
| 39 | W13 | 27 | 26 | 25 | 25 | 100.00 (288/288) | 100.00 (288/288) | 0.00 (0/1179648) | 0.00 (0/9437184) | ja | 5 | 5 |
| 39 | W2 | 28 | 27 | 26 | 26 | 100.00 (288/288) | 100.00 (288/288) | 0.00 (0/1179648) | 0.00 (0/9437184) | ja | 5 | 5 |
| 40 | W13 | 30 | 30 | 28 | 28 | 100.00 (288/288) | 100.00 (288/288) | 0.00 (0/1179648) | 0.00 (0/9437184) | ja | 5 | 5 |
| 40 | W2 | 34 | 32 | 30 | 28 | 99.31 (286/288) | 99.65 (287/288) | 0.00 (2/1179648) | 0.00 (1/9437184) | ja | 6 | 6 |
| 41 | W13 | 39 | 38 | 34 | 33 | 99.31 (286/288) | 99.65 (287/288) | 0.00 (2/1179648) | 0.00 (2/9437184) | ja | 6 | 6 |
| 41 | W2 | 37 | 31 | 30 | 30 | 99.31 (286/288) | 100.00 (288/288) | 0.00 (2/1179648) | 0.00 (0/9437184) | ja | 6 | 5 |
| 42 | W13 | 31 | 29 | 29 | 29 | 100.00 (288/288) | 100.00 (288/288) | 0.00 (0/1179648) | 0.00 (0/9437184) | ja | 5 | 5 |
| 42 | W2 | 33 | 31 | 31 | 31 | 98.61 (284/288) | 100.00 (288/288) | 0.00 (4/1179648) | 0.00 (0/9437184) | ja | 6 | 5 |
| 43 | W13 | 39 | 39 | 39 | 34 | 97.92 (282/288) | 98.61 (284/288) | 0.00 (17/1179648) | 0.00 (6/9437184) | ja | 6 | 6 |
| 43 | W2 | 56 | 50 | 50 | 50 | 95.83 (276/288) | 97.92 (282/288) | 0.04 (424/1179648) | 0.00 (188/9437184) | ja | 6 | 6 |
| 44 | W13 | 47 | 42 | 40 | 40 | 93.40 (269/288) | 96.18 (277/288) | 0.01 (108/1179648) | 0.00 (43/9437184) | ja | 6 | 6 |
| 44 | W2 | 53 | 51 | 50 | 50 | 63.89 (184/288) | 81.25 (234/288) | 0.48 (5689/1179648) | 0.04 (3772/9437184) | ja | 6 | 6 |

Kontrolle: Schichten 14 und 35 (S5-Schichten) kommen mit global <= 31 heraus (14: W13 29, W2 26; 35: W13 28, W2 27), ebenso alle S5-Schichten 37 bis 39 (max 31 bei Schicht 38 W2). Genau die neun Projektionen mit global > 31 sind die der Analyse: 36 W2, 40 W2, 41 W13+W2, 42 W2, 43 W13+W2, 44 W13+W2 (= 9 csf-Expander-Launches je Runde). Die Methode reproduziert den bekannten Zustand.

## Befund

1. **Sechs Bit global reichen fuer alle neun csf-Projektionen** (hoechste globale Spanne 56 bei Schicht 43 W2, Grenze 63). Mit Hi-Wort je (Kachel, Gruppe) zweimal statt einmal, Basis je Zeile wie heute, ohne Kernlogik fuer Ausnahmen.
2. **Basis je Scheibe + 5 Bit reicht NICHT fuer alle.** Die Scheibenspanne ist nur 3 bis 12 kleiner als die globale und bleibt ueber 31 bei: 36 W2 (33, 1 Experte von 288), 40 W2 (32, 1 Experte), 41 W13 (38), 43 W13 (39), 43 W2 (50), 44 W13 (42), 44 W2 (51; 54 Experten). Nur 41 W2 und 42 W2 (beide genau 31) passen vollstaendig; bei 36 W2 und 40 W2 scheitert je genau ein Experte knapp. Da die Entscheidung je Projektion fuer alle Experten gilt, laeuft Basis je Scheibe fuer 7 von 9 Projektionen nicht, ohne Ausnahmebehandlung.
3. Auch mit Scheiben von Viertelgroesse (8 bzw. 4 Gruppen) bleiben 43 W13 (34), 43 W2 (50), 44 W13 (40), 44 W2 (50), 41 W13 (33) ueber 31: kleinere Scheiben helfen nicht, ausser bei 36 W2, 40 W2 (31 / 28 bei Viertel). Die Spanne entsteht zwischen einzelnen Gruppen derselben Zeile, nicht zwischen weit entfernten K-Bereichen.
4. Zeilen mit Spanne > 31 sind selten (44 W2: 0,48 % der Zeilen, 5689 von 1 179 648; alle anderen Projektionen praktisch 0 bis wenige Zeilen), aber ein einziger Ausreisser sperrt die Projektion.
5. Empfehlung: **Sechs-Bit-Planes** (Rang 2 der MOE_KERN.md): Nibble + zwei Hi-Bits. Basis je Scheibe nur, wenn man die zwei knappen Projektionen und Zusatzlogik (Ausnahmen) ausschliesst; das spart 7 von 9 nicht.

## VRAM-Mehrbedarf (aus den Tensorformen, je Experte; MB = 10^6 Byte; Kern liest `s5.base` [E,T,32] u8, `s5.nib` [E,T,G/2,32] u8, `s5.hi` [E,T,G] i32)

| Projektion | S5 (heute) | Sechs-Bit-Planes | Basis je Scheibe (8 Basen) | Raster (gemeinsam, je Experte) |
|---|---|---|---|---|
| W13 | 0.6595 | 0.7905 | 0.6881 | 1.0486 |
| W2 | 0.3318 | 0.3973 | 0.3604 | 0.5243 |

csf-Projektionen (globale Spanne > 31), MB je Projektion und Schicht (288 Experten):

| Schicht | Proj. | csf-Planes heute | Sechs-Bit-Planes | Basis je Scheibe | Basis je Scheibe passt |
|---|---|---|---|---|---|
| 36 | W2 | 87.6 | 114.4 | 103.8 | nein |
| 40 | W2 | 87.7 | 114.4 | 103.8 | nein |
| 41 | W13 | 174.7 | 227.7 | 198.2 | nein |
| 41 | W2 | 87.8 | 114.4 | 103.8 | ja |
| 42 | W2 | 88.0 | 114.4 | 103.8 | ja |
| 43 | W13 | 174.7 | 227.7 | 198.2 | nein |
| 43 | W2 | 88.1 | 114.4 | 103.8 | nein |
| 44 | W13 | 174.8 | 227.7 | 198.2 | nein |
| 44 | W2 | 89.3 | 114.4 | 103.8 | nein |
| Summe | | 1052.7 | 1369.6 | 1217.4 | |

Gemeinsames Raster je Karte (W13 + W2, 288 Experten, entfaellt, sobald keine Schicht der Karte mehr csf haelt): 453.0 MB.
Netto Sechs-Bit: -136.2 MB; Basis je Scheibe: -288.3 MB (nur gueltig, wenn alle Projektionen passen: nein).

Laufzeit 101 s

Rechnung (T = N/32 = 128 Kacheln, G = Gruppen): S5 = T * (32 + 16 G + 4 G) B; Sechs Bit = T * (32 + 16 G + 8 G) B (zweites Hi-Wort je Kachel und Gruppe); Basis je Scheibe = T * (8 * 32 + 16 G + 4 G) B; Raster = T * G * 32 B. W13: G = 256, W2: G = 128. Die S5-Bytes stimmen mit MOE_KERN.md 2.4 ueberein (9 048 064 + 4 526 080 = 13 574 144 B je Experte). csf auf der GPU nach `CsfExpertScales` (`csf32_expand.py:34`): bases u8, codes u8, page_bases/offsets/exceptions als int32, je Matrix ein int64-Start; Datensaetze je Projektion aus dem Checkpoint gezaehlt.

Gegenueber heute (csf-Planes + gemeinsames Raster auf der Karte, das erst frei wird, wenn keine csf-Schicht mehr auf ihr liegt):
- **Sechs-Bit-Planes: netto -136 MB** je Karte (S4): +317 MB Planes gegen csf, dafuer entfaellt das gemeinsame Raster von 453 MB (W13 302 + W2 151). Das ist besser als die +0,1 GB der Analyse. Voraussetzung: alle neun Projektionen werden umgestellt (sonst bleibt das Raster). Die MTP-Schicht (nextn) wurde hier nicht untersucht, falls sie csf haelt und auf derselben Karte liegt, bleibt das Raster; Ladespitze unveraendert (das Raster wird beim Laden gebraucht).
- Raster-Pfad (heute) gegen S5: +1,573 MB je Experte und Schicht beim Streamen (14,156 MB gegen 13,574 MB gerechnet aus den Tensorformen; die 14,147 MB der Auftragsvorgabe liegen 0,009 MB daneben); Sechs-Bit-Planes streamen 0,132 MB (W13+W2: 0,1310 + 0,0655 = 0,1965 MB) mehr als S5 je Experte, also 13,770 MB (+1,4 % gegen S5, -2,7 % gegen Raster), ohne Expander.
- Basis je Scheibe: +0,029 MB je Projektion und Experte gegen S5 (+0,2 %), netto -288 MB, aber nur gueltig, wenn alle Projektionen passen: tun sie nicht.

## Grenzen / Annahmen

- Die Spanne wird in e4m3-Bytewerten gemessen (wie der Code, positive Skalen ordnen wie Integer, `s5_scales.py` Kopfkommentar); `decode_csf32` ist der Referenzdecoder, nicht der GPU-Expander (Gleichheit der beiden ist Aufgabe der vorhandenen Tests, hier nicht neu geprueft).
- Gewichte der Schichten 15 bis 34 (ausser 14) wurden nicht ausgewertet; sie liegen laut Analyse in S5. Ausgewertet: 14, 35 bis 44.
- Rechnet man die Planes fuer W13 mit dem Gate/Up-Verbund (Zeilen 0-2047 gate, 2048-4095 up), ist das gleich dem Produktionslayout; die Reihenfolge beeinflusst die Spanne je Zeile nicht.
- Nicht gemessen: Auswirkung von Sechs Bit auf die Kernzeit (zusaetzliches Hi-Wort je Gruppe: 4 B mehr je Kachel und Gruppe, ein weiterer `LDG` je Gruppe); das ist GPU-seitig.
