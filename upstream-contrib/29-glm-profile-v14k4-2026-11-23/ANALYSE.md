# GLM-5.3-Flash Decode: Abstand der MoE- und Dense-Kerne zur Speicherbandbreite (nsys, 23.11.2026)

Nur Analyse (keine Änderung an Code, GPU oder Git). Quelle: `~/.cache/nsys-runs/glm-v14k4-decode16k-1123/` (Capture 1 = Decode kurz, Capture 2 = Decode bei 16k Kontext, nur Abschnitt 8).
Reproduktion: `analyse_nsys_stufen.py <sqlite>` in diesem Ordner (für Capture 2 zusätzlich die Filterung aus Abschnitt 8, im Skript nicht enthalten).

Kennzeichnung: **[gemessen]** = direkt aus der nsys-Datei, **[Header]** = aus den safetensors-Headern/config.json, **[abgeleitet]** = Rechnung aus gemessenen und Header-Werten, **[vermutet]** = Schluss ohne direkten Beleg.
Bandbreiten: Spitze V100-PCIE 900 GB/s, Quadro RTX 8000 672 GB/s (nominal). Als praktische Decke gelten 85 %: V100 765 GB/s, RTX 571 GB/s. Die besten gemessenen Kerne erreichen RTX 86–87 % (o_proj DSA, LM-Kopf, Entwurf-Gate/Up), V100 84 % (o_proj DSA); die Decke von 85 % ist damit durch eigene Kerne belegt. GB/s = Byte / µs / 1000 (dezimal, 1 MB = 10^6 B).

## 1. Zuordnung und Runden

**Zuordnung nsys-deviceId → Stufe [gemessen]:** deviceId 0..4 ist die Stufe (die Reihenfolge der Kernelketten bestätigt es): Die Hauptdurchläufe folgen strikt aufeinander (Stufe 0 endet, 0,27 ms später beginnt Stufe 1, usw.). Kernelnamen trennen die Karten: deviceId 0 und 4 laufen `skinny_nvfp4_moe_m16n8k8`, `turing_*gemm`, `fp8_qpn8_mma_sm75` (Turing = RTX); deviceId 1, 2, 3 laufen `skinny_nvfp4_moe_qpn`, `volta_*gemm`, `fp8_qpn8_dequantize_sm70` (Volta = V100). MoE-Starts je Durchlauf: 22/14/14/14/20 = 2 Kerne (Up, Down) je Sparse-Schicht, also 11/7/7/7/10 Sparse-Schichten = Partition 14,7,7,7,10 minus die 3 dichten Schichten 0–2 auf Stufe 0. Nur deviceId 4 hat LM-Kopf-GEMM (`fp8_qpn8` Gitter 4840 = 154 880 Zeilen) und die DFlash2-Kerne (`_dflash_*`). Die Tabelle `TARGET_INFO_GPU` der Datei (andere Reihenfolge, PCI-Reihenfolge) wurde nicht benutzt.

**Runden [gemessen]:** Capture 1 enthält 39 Vorwärtsdurchläufe je Stufe: der erste (0–195 ms, Prompt + Aufwärmen) ist Prefill, danach **38 Decode-Runden**. Beleg: 78 NCCL-Broadcast (2 je Runde) und 78 MoE-Kerne je Schicht (2 je Runde und Schicht), 39 Durchläufe.

| Größe | Wert | Einheit | Quelle |
|---|---|---|---|
| Rundenzeit (Beginn Stufe 0 bis Beginn der nächsten Runde) | 60,74 (min 56,70, max 63,94, Streuung 1,72) | ms je Runde, weniger ist besser | gemessen, 37 Abstände |
| Tokens je Runde | 127 / 38 = 3,34 | Token | abgeleitet (128 Ausgabetoken, 1 aus Prefill) |
| Rate unter nsys | etwa 55 | tok/s | abgeleitet (3,34 / 60,74 ms) |

## 2. Wo die Runde hinläuft: seriell, jede Karte meist leer

Bei einer Anfrage und einer Mikrobatch laufen die fünf Stufen **nacheinander**: Die Rundenzeit ist die Summe der Stufenzeiten plus Übergaben. Jede Karte wartet den Rest der Runde.

| Stufe (Karte) | Durchlauf | belegt (Kernel-Vereinigung) | Lücken im Durchlauf | Leerlauf je Runde | Auslastung der Runde |
|---|---|---|---|---|---|
| | ms | ms | ms | ms (weniger Wartezeit ist nur durch kürzere andere Stufen erreichbar) | % |
| S0 RTX (14 Schichten) | 18,77 | 16,86 | 1,91 | 43,9 | 27,8 |
| S1 V100 (7) | 8,02 | 7,67 | 0,34 | 53,1 | 12,6 |
| S2 V100 (7) | 7,99 | 7,52 | 0,47 | 53,2 | 12,4 |
| S3 V100 (7) | 7,34 | 7,04 | 0,30 | 53,7 | 11,6 |
| S4 RTX (10 + Entwurf + LM-Kopf) | 17,25 | 16,92 | 0,33 | 43,8 | 27,9 |
| Summe Durchläufe | 59,37 | 56,01 | 3,35 | | |
| Übergaben S0→S1, S1→S2, S2→S3, S3→S4, S4→S0 | 0,27 + 0,37 + 0,36 + 0,32 + 0,03 = 1,35 | | | | |
| Summe | 60,72 | | | | Rundenzeit 60,74 (stimmt) |

[gemessen]. Leerlauf = Rundenzeit minus belegt [abgeleitet]. Mittlere Auslastung der fünf Karten: 18,5 % [abgeleitet]. Die Partition 14,7,7,7,10 ist durch den VRAM bestimmt (eine Sparse-Schicht = 288 Experten · 13,44 MB = 3,87 GB, 7 Schichten ≈ 27 GB auf 32 GB; [abgeleitet]); für die Rundenzeit zählt nicht das Gleichgewicht, sondern die Summe. Eine Schicht kostet auf der RTX mehr (KDA+Sparse: 1,13–1,28 ms) als auf der V100 (0,96–1,01 ms) (Abschnitt 6) [gemessen]; jede Schicht, die von einer RTX auf eine V100 wanderte, sparte etwa 0,15–0,3 ms je Runde [abgeleitet], ist aber durch den V100-VRAM begrenzt.

**NCCL [gemessen, 38 Decode-Runden]:** je Karte 2 Broadcast je Runde (ein langer von 17–42 ms ist reines Warten auf die vorderen Stufen, ein kurzer von etwa 5 µs) und SendRecv-Kerne: S0 5, S1 11, S2 13, S3 15, S4 8 je Runde, dazu je Stufe 1 langer Warte-SendRecv (S1–S4: 15–26 ms, wartet auf die Eingangsstufe). Wirkliche NCCL-Laufzeit je Runde (SendRecv kurz + Broadcast kurz): S0 0,13 ms, S1 0,31, S2 0,41, S3 0,41, S4 0,22 ms. Auf dem kritischen Pfad bleibt je Übergabe 0,27–0,37 ms Lücke (Übertragung selbst nur etwa 165 kB: 5 Zeilen · 4 mHC-Ströme · 4096 · 2 B; [abgeleitet] also Latenz, nicht Bandbreite).

## 3. Modellgrößen (Header, exakt)

config.json (text_config): hidden 4096, moe_intermediate 2048, 288 geroutete Experten, Top-8, 1 Shared Expert (Zwischengröße 2048), dichte Schichten 0–2 (intermediate 12288), 45 Schichten: DSA-Schichten 3, 7, 11, …, 43 (11 Stück), die übrigen 34 sind KDA. hc_mult 4 (mHC). NVFP4-Experten mit „csf32“-komprimierten Skalen.

| Tensor (je Schicht) | Format | Bytes | Quelle |
|---|---|---|---|
| Experte gate / up / down: Gewichte | NVFP4 (U8 gepackt) | je 4 194 304, zusammen 12 582 912 | Header |
| Experte: Skalen csf32 (codes 262 144 + bases + exceptions + offsets 16 448 + page_bases 128) | U8/U16/U32 | gate ≈ 283,6 kB, up ≈ 283,6 kB, down ≈ 285,8 kB (schwankt mit den exceptions) | Header Schicht 3, 4, 44 |
| **Bytes je Experte** | | Up-Kern (gate+up) **8,956 MB**, Down-Kern **4,480 MB**, zusammen **13,436 MB** (Skalen 6,8 % Aufschlag; expandiert als FP8 wären es 12,5 %) | abgeleitet |
| Router `mlp.gate` BF16 [288,4096] | BF16 | 2,36 MB | Header |
| Shared Expert gate+up / down (FP8 + F32-Blockskalen) | F8 | 16,78 MB / 8,39 MB | Header |
| KDA qkv (3 · [8192,4096] FP8 + U8-Skalen je 32) | MXFP8 | 103,81 MB | Header |
| KDA o_proj [4096,8192] | MXFP8 | 34,60 MB | Header |
| KDA f_a, g_a, b, f_b, g_b (BF16, klein) | BF16 | zusammen 6,8 MB | Header |
| DSA q_a [1536,4096], kv_a [512,4096] | FP8 | 6,29 + 2,10 MB | Header |
| DSA q_b [16384,1536] | FP8 | 25,17 MB | Header |
| DSA kv_b [32768,512] | **BF16** | 33,55 MB | Header |
| DSA o_proj [4096,16384] | FP8 | 67,13 MB | Header |
| DSA Indexer wq_b [4096,1536] + wk + gate + weights_proj | **BF16** | 12,58 + 1,05 + 1,05 + 0,26 = 14,9 MB | Header |
| Dichte Schicht 0–2: gate+up / down (MXFP8) | | 103,81 MB / 51,90 MB | Header |
| LM-Kopf [154880,4096] MXFP8 | | 654,2 MB (634,4 + 19,8 Skalen) | Header |
| DFlash2-Entwurf je Schicht (fc einmal): qkv, o, gate+up, down; fc | FP8 | 25,2 / 16,8 / 100,7 / 50,3 MB; fc 83,9 MB | Header des Entwurfs |

Die Zuordnung „Kern → Tensor" ist aus Gittergrößen und Reihenfolge abgeleitet [abgeleitet]: `fp8_qpn8` Gitter 768 (b512) = N 24 576 (KDA qkv, bei Schichten 0–2 auch dichtes gate+up), 128 (b512) = N 4096 (o_proj, Shared gate+up, dichtes down), 128 (b256) = Shared down, 512 (b256) = DSA q_b, 4840 = LM-Kopf. Stückzahlen je Runde passen genau (z. B. S0: 11 qkv, 14 o, 11 Shared gu, 11 Shared dn, 3 q_b, 3 dichtes gu, 3 dichtes dn).

## 4. MoE-Gruppe

**Zeit je MoE-Kern [gemessen, Mittel über 38 Runden]:** Up-Kern etwa doppelt so lang wie Down (Verhältnis der Bytes 2,0; Anpassung Up = a + b·Down ergibt b = 1,95–2,08, a −16…+10 µs: also Zeit proportional zu den gelesenen Bytes, kaum fester Anteil) [gemessen/abgeleitet].

| Stufe (Karte) | Sparse-Schichten | Up-Kern Mittel [min…max] | Down-Kern Mittel [max] | Summe je Schicht | MoE-Kerne je Runde | Rand-Kerne (Router, Topk, Aktivierung, Reduce, csf32-expand) je Runde |
|---|---|---|---|---|---|---|
| | | µs | µs | µs | ms | ms |
| S0 RTX | 11 | 526 [327…641] | 266 [321] | 792 | 8,71 | 0,26 |
| S1 V100 | 7 | 367 [245…501] | 184 [249] | 552 | 3,86 | 0,22 |
| S2 V100 | 7 | 353 [194…491] | 176 [245] | 529 | 3,70 | 0,23 |
| S3 V100 | 7 | 334 [205…457] | 166 [234] | 500 | 3,50 | 0,22 |
| S4 RTX | 10 | 393 [235…612] | 196 [301] | 589 | 5,89 | 0,76 (davon `csf32_expand` 0,55) |
| **Summe** | 42 | | | | **25,66 (42 % der Runde)** | 1,69 |

Die Zeit je Schicht schwankt mit der Tiefe stark (S0: Schicht 3: 874 µs, Schicht 13: 685 µs; S4: Schicht 35: 544, Schicht 44: 730 µs). Gleiche Kerne gleicher Karte sind also nicht gleich schnell: Die Zahl der **verschiedenen Experten** je Schicht und Runde schwankt (datenabhängig) [vermutet, gestützt auf Dichte der Ergebnisse unten].

**Gelesene Bytes:** Der Kern läuft Gitter (64 Zeilenkacheln × 40 Plätze) = 5 Zeilen × Top-8 = höchstens 40 Experten-Plätze. Gelesen wird die Vereinigung der Experten (Beleg: Laufzeit variiert 2× zwischen Runden; bei 40 immer gelesenen Plätzen wäre sie konstant). Erwartete verschiedene Experten bei gleichverteilter Wahl: 288·(1 − (280/288)^5) = **37,8**; Höchstfall **40**.

Erreichte Bandbreite je Schicht (Up + Down, Mittelzeit) bei angenommener Expertenzahl N [abgeleitet; Bytes N · 13,436 MB: N=20 → 269 MB, N=30 → 403 MB, N=37,8 → 508 MB, N=40 → 537 MB]:

| Stufe | Karte | N=20 | N=30 | N=37,8 (gleichverteilt) | N=40 | größtes N bei ≤ 100 % der Spitze | größtes N bei ≤ 88 % der Spitze |
|---|---|---|---|---|---|---|---|
| | | GB/s (% Spitze) | GB/s (% Spitze) | GB/s (% Spitze) | GB/s (% Spitze) | | |
| S0 | RTX | 340 (51) | 509 (76) | 642 (96) | 679 (101) | 39,6 | 34,8 |
| S1 | V100 | 487 (54) | 731 (81) | 921 (102) | 974 (108) | 37,0 | 32,5 |
| S2 | V100 | 508 (56) | 762 (85) | 961 (107) | 1016 (113) | 35,4 | 31,2 |
| S3 | V100 | 538 (60) | 807 (90) | 1017 (113) | 1076 (120) | 33,5 | 29,5 |
| S4 | RTX | 456 (68) | 684 (102) | 863 (128) | 912 (136) | 29,5 | 25,9 |

**Befund 1 [abgeleitet, zwingend]:** Die Gleichverteilung (37,8 Experten) ist physikalisch unmöglich, denn sie ergäbe über 100 % der Spitze (alle Stufen außer S0 deutlich, S4 sogar 128 %). Die echte mittlere Zahl verschiedener Experten ist also deutlich kleiner (höchstens 29,5 auf S4 bis 39,6 auf S0 bei 100 % Spitze; bei 88 % höchstens 26–35). Die fünf Zeilen eines Verify-Schritts (Anker + 4 Entwurfstoken) wählen überlappende Experten.

**Befund 2 [abgeleitet]:** Die langsamsten einzelnen Kerne je Stufe (Maximum, wahrscheinlich nahe 40 verschiedene Experten) liegen bei Annahme N=40 auf **83 % (S0), 79 % (S1), 81 % (S2), 87 % (S3), 87 % (S4)** der Spitze (Up, Down gleich: 83,1/83,1; 79,4/80,0; 81,1/81,2; 87,2/85,2; 87,1/88,6). Hatte das Maximum weniger als 40 verschiedene Experten, sind es entsprechend weniger (N=36: 75 / 71 / 73 / 78 / 78 %). Die Zahl ist also eine obere Schranke der Effizienz bei vollem Lauf; Zeit ist proportional zu N, die Effizienz je Byte ist über N etwa konstant (siehe Anpassung oben).

**Wie weit vom Limit [abgeleitet]:** Bei Annahme „Maximum = 40 Experten" ist der MoE-Kern auf RTX und V100 bei 79–87 % und liegt damit bereits nah an der 85-%-Decke; bei Maximum 36 Experten bei 71–78 %. **Fehlende Messgröße:** die tatsächliche Zahl verschiedener Experten je Schicht und Runde (ein einmaliger Debug-Zähler in der Routing-Stufe, ein Lauf). Damit wäre die Effizienz exakt, nicht als Spanne.

## 5. Dense-Gruppe (`fp8_qpn8_sm70_kernel` und Verwandte)

Je Klasse: Stückzahl je Runde, Mittelzeit, Bytes, erreichte Bandbreite, Anteil der Spitze. Zahlen [gemessen] für Zeit/Stück, [Header] für Bytes, [abgeleitet] für GB/s. Zeit höher = schlechter; Bandbreite höher = besser.

| Klasse | Karte | Stück/Runde | µs je Aufruf | MB je Aufruf | GB/s | % Spitze | ms/Runde |
|---|---|---|---|---|---|---|---|
| KDA qkv | RTX S0 | 11 | 187,9 | 103,8 | 552 | 82,2 | 2,07 |
| KDA qkv | RTX S4 | 7 | 185,5 | 103,8 | 560 | 83,3 | 1,30 |
| KDA qkv | V100 S1/S2/S3 | 5/5/6 | 145,6/145,1/146,2 | 103,8 | 713/716/710 | 79,2/79,5/78,9 | 0,73/0,73/0,88 |
| KDA o_proj | RTX S0 / S4 | 11 / 7 | 76,1 / 75,1 | 34,6 | 455 / 461 | 67,7 / 68,6 | 0,84 / 0,53 |
| KDA o_proj | V100 | 5/5/6 | 57,9/57,4/57,9 | 34,6 | 598/603/598 | 66,4/67,0/66,4 | 0,29/0,29/0,35 |
| DSA o_proj | RTX S0 / S4 | 3 / 3 | 115,5 / 114,8 | 67,1 | 581 / 585 | 86,5 / 87,0 | 0,35 / 0,34 |
| DSA o_proj | V100 | 2/2/1 | 88,6/88,4/88,4 | 67,1 | 757/759/759 | 84,1/84,3/84,3 | 0,18/0,18/0,09 |
| Shared gate+up | RTX S0 / S4 | 11 / 10 | 34,6 / 33,9 | 16,8 | 485 / 494 | 72,1 / 73,6 | 0,38 / 0,34 |
| Shared gate+up | V100 | 7 | 29,2–29,7 | 16,8 | 566–575 | 63–64 | 0,20–0,21 |
| Shared down | RTX S0 / S4 | 11 / 10 | 19,3 / 17,6 | 8,4 | 434 / 477 | 64,6 / 70,9 | 0,21 / 0,18 |
| Shared down | V100 | 7 | 18,8–18,9 | 8,4 | 443–447 | 49–50 | 0,13 |
| DSA q_b | RTX | 3 | 44,3 | 25,2 | 568 | 84,5 | 0,13 |
| DSA q_b | V100 | 2/2/1 | 37,3–38,0 | 25,2 | 662–675 | 74–75 | 0,07–0,08 |
| dichtes gate+up (Schichten 0–2) | RTX S0 | 3 | 195,7 | 103,8 | 530 | 78,9 | 0,59 |
| dichtes down | RTX S0 | 3 | 93,4 | 51,9 | 556 | 82,7 | 0,28 |
| LM-Kopf (2 je Runde: Verify + Entwurf) | RTX S4 | 2 | 1131,5 | 654,2 | 578 | 86,0 | 2,26 |
| Entwurf gate+up / down / qkv / o / fc | RTX S4 | 5/5/5/5/1 | 173,8 / 86,9 / 49,0 / 31,3 / 161,8 | 100,7 / 50,3 / 25,2 / 16,8 / 83,9 | 579 / 579 / 514 / 536 / 518 | 86,2 / 86,2 / 76,5 / 79,7 / 77,2 | 0,87 / 0,43 / 0,24 / 0,16 / 0,16 |

Summe `fp8_qpn8` je Runde und Stufe: S0 4,84 ms, S1 1,61, S2 1,60, S3 1,69, S4 6,95 (davon LM-Kopf 2,26 und Entwurf 1,86).

**Zeit gewichtet, ganze Stufe [abgeleitet]:** S0 525 GB/s (78,1 % von 672), S1/S2/S3 656/658/650 GB/s (72,9/73,1/72,2 % von 900), S4 555 GB/s (82,6 %). Große Kerne (qkv, DSA o_proj, q_b, dichtes MLP, LM-Kopf, Entwurf) laufen auf beiden Karten bei 79–87 %; die kleinen (o_proj KDA 66–69 %, Shared gate+up 63–74 %, Shared down 49–71 %) drücken den Durchschnitt, besonders auf der V100.

### Frage 3: „13 ms je Schicht auf Stufe 0 (RTX) gegen 8,7 ms auf den V100-Stufen je Capture"

Das ist **kein ineffizienter Turing-Kern**, sondern Bytes mal Bandbreite:
- Rechnung aus der Datei: Stufe 0 `fp8_qpn8` 184,08 ms / 14 Schichten = 13,1 ms; Stufe 1 61,02 ms / 7 = 8,7 ms; Verhältnis 1,50 [gemessen; die 39 Durchläufe enthalten Prefill, daher auch je Runde: 0,346 gegen 0,229 ms, Verhältnis 1,51].
- Bytes je Schicht [abgeleitet]: S0 182 MB (enthält die drei dichten 12288er-MLP-Schichten mit je 155,7 MB zusätzlich und 11 KDA-/3 DSA-Schichten), S1 150 MB. Verhältnis 1,21.
- Bandbreitenverhältnis 900/672 = 1,34. Erwartete Zeitverhältnis bei gleicher Effizienz: 1,21 · 1,34 = 1,62 [abgeleitet]. Gemessen ist 1,50 [gemessen]. Die RTX ist also, bezogen auf ihre Spitze, 7 % **besser** als die V100 (78,1 % gegen 72,9 %).
- Gleiche Klasse gegen gleiche Klasse (reine Karte, gleiche Bytes) [gemessen]: qkv 187,9 / 145,6 µs = 1,29; o_proj KDA 76,1 / 57,9 = 1,31; DSA o_proj 115,5 / 88,6 = 1,30; Shared gate+up 1,18; Shared down 1,03; q_b 1,18. Große Kerne 1,29–1,31 (unter 1,34: kein Turing-Nachteil). Kleine Kerne sind auf beiden Karten nahezu gleich schnell (Shared down 19,3 gegen 18,8 µs, nur 50 % von V100-Spitze): sie haben eine Zeituntergrenze von rund 19 µs (Gitter 128 CTAs, wenige Warps je SM; Startlatenz) und lassen die V100 ihren Bandbreitenvorteil nicht ausspielen [vermutet: Ursache Belegung/Latenz].

### Kleine GEMMs (cuBLAS/`Kernel2`/`splitKreduce`/`volta_*`/`turing_*`)

Je Runde: S0 1,37 ms (129 Aufrufe), S1 0,99 (78), S2 0,98 (78), S3 0,70 (60), S4 1,21 (111); Summe **5,24 ms (8,6 % der Runde)** [gemessen]. Größter Block: DSA-Schichten, 307–346 µs je Schicht (RTX 307–315, V100 344–346 [gemessen]) bei etwa 55 MB gelesenen Gewichten (kv_b 33,55 + Indexer 14,9 + q_a/kv_a 8,4 MB, falls als BF16/FP8 gelesen [vermutet, Zuordnung über Gittergrößen]); die 85-%-Zeit dafür wären 96 µs (RTX) bzw. 72 µs (V100) [abgeleitet]. Einzelposten [gemessen]:
- kv_b: zwei Kerne je DSA-Schicht, RTX 47,6 + 46,0 = 93,6 µs, V100 41,4 + 47,3 = 88,7 µs für 33,55 MB → 358 GB/s (53 %) bzw. 378 GB/s (42 %) [abgeleitet]; bei 85 % 58,7 bzw. 43,9 µs.
- Indexer wq_b (`Kernel2` Gitter 8,32,1): 30,5 µs (RTX) / 30,0 µs (V100) für 12,58 MB → 412 GB/s (61 %) bzw. 419 GB/s (47 %).
- Sparse-Attention-Block (`gather_dequant` 26–28 µs, 5 × `s1688/s884gemm` je 11–14 µs + `splitKreduce` 3,5–6,6 µs, ein `tn`-GEMM 37 µs): zusammen ≈ 150 µs je DSA-Schicht bei kurzem Kontext; latenzgebunden, Byteanzahl hier klein.
- KDA-Schicht: 41–59 µs kleine GEMMs für etwa 9,2 MB (f/g/b-Projektionen + Router); Untergrenze durch Startlatenz.

## 6. Zeitverteilung je Stufe (Zeit je Runde in µs, Kernel-Summe; [gemessen])

| Gruppe | S0 RTX | S1 V100 | S2 V100 | S3 V100 | S4 RTX* | Summe ms | % der Runde |
|---|---|---|---|---|---|---|---|
| MoE-Kerne (Up + Down) | 8706 | 3862 | 3702 | 3498 | 5892 | 25,66 | 42,2 |
| MoE-Rand (Router, Topk, Aktivierung, Reduce, csf32-expand) | 257 | 219 | 227 | 215 | 758 | 1,68 | 2,8 |
| Dense `fp8_qpn8` (Zielmodell) | 4844 | 1606 | 1601 | 1691 | 2817 | 12,56 | 20,7 |
| kleine GEMMs (cuBLAS/`Kernel2`) | 1368 | 985 | 980 | 700 | 1209 | 5,24 | 8,6 |
| mHC (pre_norm, post_dot) | 622 | 400 | 401 | 403 | 450 | 2,28 | 3,8 |
| KDA (conv1d, recurrent, rmsnorm, pack) | 660 | 274 | 276 | 331 | 411 | 1,95 | 3,2 |
| DSA-Attention/Indexer-Kerne | 242 | 196 | 199 | 100 | 205 | 0,94 | 1,5 |
| Elementweise/Sonstiges | 165 | 134 | 143 | 104 | 114 | 0,66 | 1,1 |
| LM-Kopf (Verify + Entwurf), nur S4 | | | | | 2260 | 2,26 | 3,7 |
| Entwurf (fc + 5 Schichten, mit Attention), nur S4 | | | | | 2680 (Wand) | 2,68 | 4,4 |
| Sampling/Rejection + Rest nach Kopf 2, S4 | | | | | 240 | 0,24 | 0,4 |
| Lücken im Durchlauf (kein Kernel) | 1910 | 340 | 470 | 300 | 330 | 3,35 | 5,5 |
| Übergaben zwischen den Durchläufen | | | | | | 1,35 | 2,2 |

*S4: Spalte „Zielmodell-Bereich" (erste Schicht bis letzter MoE-Kern + 6 Kerne) = 12,07 ms; LM-Kopf, Entwurf, Sampling liegen danach. Entwurf: Kernel-Summe ergibt sich aus Wandzeit 2,68 ms (Kopf 1 endet, Kopf 2 beginnt) und enthält `fp8_qpn8` 1,86 ms. Die Spalten summieren sich auf 60,74 ms (Rundenzeit) ± 0,1.

**Zeit je Schicht [gemessen, Summe der Kernelzeiten]:** KDA-Sparse-Schicht: RTX S0 1282 µs, RTX S4 1133 µs, V100 963–1006 µs. DSA-Sparse-Schicht: RTX S0 1436 µs, S4 1301 µs, V100 1252–1317 µs. Anteil MoE-Kerne an der Schicht 52–55 % (KDA, V100), 52 % (DSA, RTX S0), 40–43 % (DSA, V100) [abgeleitet].

**Stufe 4 im Einzelnen [gemessen, Wandzeit]:** Schichten 35–44 11,96 ms; Kopf 1 (Verify) 1,13; Sampling 0,08; Entwurf (fc + 5 Schichten) 2,68; Kopf 2 (Entwurf) 1,13; Rest 0,16 ms. **Stufe 0:** vor dem ersten Schichtkern liegen 1,55 ms (min 1,12, max 2,96) Vorbereitung durch ~20 winzige Kerne mit Abstand von etwa 100 µs (`_post_update`, `_scatter_*`, `_prepare_*`, `_compute_slot_mappings`, …); das ist hostgebundenes Launch-Verhalten auf dem kritischen Pfad. In Capture 2 betrug derselbe Wert nur 0,26 ms (Abschnitt 8): die Differenz von etwa 1,3 ms ist also (unter nsys) nicht grundsätzlich, Ursache offen [vermutet: CPU-Last oder anderer Planerzustand].

## 7. Rangliste: wo noch Zeit liegt (ms je Runde bei 85 % der Spitze)

Rundenzeit 60,74 ms; jede Zeile gilt je Runde, weniger = besser. „Sicher" = aus den Messungen ohne weitere Annahme; „Spanne" = hängt von einer genannten Unbekannten ab.

| Rang | Posten | Zeit heute | erreichbar | Beleg | Sicherheit |
|---|---|---|---|---|---|
| 1 | Host-/Launch-gebunden, ohne Rechnung: Vorbereitung vor Stufe 0 (1,55 ms), Übergaben (1,35 ms), Lücken im Durchlauf auf S1–S4 (1,44 ms) | 4,34 ms | bis etwa 1,3 (Vorbereitung auf Capture-2-Niveau 0,26 ms); der Rest braucht neue Messung | Abschnitt 2 und 6 | Spanne; Vorbereitung belegt, Übergaben/Lücken nur als Obergrenze [vermutet] |
| 2 | MoE-Kerne (größter Block, 25,66 ms) | 25,66 | 0,6 (Maximum = 40 Experten) bis 3,0 (Maximum = 36 Experten) | Abschnitt 4, Befund 2 | Spanne; Zahl verschiedener Experten fehlt (ein Debug-Zähler löst es) |
| 3 | kleine GEMMs der DSA-Schichten (kv_b-Paar, Indexer wq_b): 11 DSA-Schichten | 5,24 (ganze Gruppe) | 0,55 benannt (kv_b 0,43 + wq_b 0,12); Obergrenze der ganzen Gruppe etwa 2,5 | Abschnitt 5, kleine GEMMs | benannter Teil sicher, Obergrenze [vermutet] |
| 4 | Dense `fp8_qpn8` auf 85 % (o_proj KDA 0,47, Shared gate+up 0,26, qkv 0,25, Shared down 0,25, Rest 0,13) | 12,56 | 1,36 | Abschnitt 5 | sicher (Zeit je Klasse gegen Bytes/(0,85·Spitze)) |
| 5 | `csf32_expand` auf S4 (342 Aufrufe in 38 Runden, 61 µs): nur S4 expandiert Skalen zur Laufzeit | 0,55 | bis 0,55 (Skalen vorab expandieren, kostet VRAM auf der engsten Stufe) | Abschnitt 4 | Menge sicher; Ursache der Verteilung „nur S4" [vermutet] |
| 6 | Entwurfs-Kopf auf Teilvokabular (kein Effizienzposten): 654 MB bei 86 % schon an der Decke; weniger Zeilen = weniger Byte | 1,13 (Kopf 2) | 0,43 bei 98 304 Zeilen, 0,9 bei 32 768 Zeilen [abgeleitet, linear in den Zeilen] | Abschnitt 5 | Folgen für die Annahmerate nicht gemessen |
| 7 | Entwurfs-Rest (ohne `fp8_qpn8` 0,8 ms: `Kernel2` mit 410 GB/s, Attention, Norm) | 0,8 | etwa 0,2 | Abschnitt 6 | [vermutet] |
| – | LM-Kopf Verify, DSA o_proj, q_b, Entwurfs-gate+up/down | (in Abschnitt 5) | 0 | bei 84–87 % | sicher |

Summe, nur die „sicheren" und „benannten" Posten (Rang 4: 1,36; Rang 3 benannt: 0,55; Rang 2 untere Grenze: 0,6; Rang 1 nur Vorbereitung: bis 1,3): **etwa 3,8 ms = 6 % der Runde**. Mit den oberen Spannen (MoE 3,0; kleine GEMMs 2,5; csf32 0,55; Entwurfs-Kopf 0,43–0,9): etwa 9–10 ms (15 %) [abgeleitet, Obergrenze]. Rechnerisch ergäben −3,8 ms bei gleicher Annahmerate etwa +6 % tok/s.

**Struktureller Befund [abgeleitet, außerhalb des 85-%-Maßstabs]:** Die fünf Karten sind je Runde nur zu 11,6–27,9 % (im Mittel 18,5 %) belegt, weil die Stufen strikt nacheinander arbeiten. Die Kerne selbst liegen nahe an der Decke; der Hebel, der die Rundenzeit unter die Summe der Stufenzeiten bringt, liegt nicht in den Kernen, sondern in der Parallelisierung (mehrere gleichzeitige Verify-Schritte oder Anfragen, oder weniger/nebeneinander laufende Stufen).

## 8. Capture 2 (Decode bei 16k Kontext, nur Decode-Runden)

Trennung der Decode-Runden von Prefill-Stücken: je Stufe nur Durchläufe mit 2·n MoE-Kernen auf Gitter (64,40), nicht Gitter y = 272, und Dauer unter 1,35 × kürzester Durchlauf (`ana2`-Variante mit `FILT=1`). Ergebnis: 35 Decode-Runden (Stufen 0–4 je 35 Durchläufe, ausgerichtet). [gemessen]

| Größe | Capture 1 (kurz) | Capture 2 (16k) | Einheit |
|---|---|---|---|
| Rundenzeit | 60,74 | 62,68 | ms |
| S0 / S1 / S2 / S3 / S4 Durchlauf | 18,77 / 8,02 / 7,99 / 7,34 / 17,25 | 17,73 / 8,16 / 8,37 / 7,79 / 19,35 | ms |
| S4 Entwurf (fc + 5 Schichten) | 2,68 | 3,99 | ms |
| S4 Schichten 35–44 | 11,96 | 12,75 | ms |
| DSA-Attention/Indexer-Kerne je Runde, S0 / S1 / S2 / S3 / S4 | 242 / 196 / 199 / 100 / 205 | 444 / 353 / 358 / 184 / 411 | µs |
| MoE-Kerne je Runde S0 / S1 / S2 / S3 / S4 | 8706 / 3862 / 3702 / 3498 / 5892 | 8794 / 3849 / 3932 / 3864 / 6450 | µs |
| Vorbereitung vor Stufe 0 | 1,55 | 0,26 | ms |

Bei 16k Kontext wächst vor allem die DSA-Attention (etwa +0,9 ms über alle Stufen) und der Entwurf (+1,3 ms, [vermutet: Aufmerksamkeit des Entwurfs über 16k]); der Rest bleibt gleich. Die Rangliste aus Abschnitt 7 ändert sich nicht grundsätzlich; Rang 6 (Entwurf) wird bei langem Kontext wichtiger. MoE-Zeiten schwanken in Capture 2 etwas stärker (S3 +0,37 ms): andere Texte, andere Expertenwahl.

## 9. Grenzen der Auswertung

- nsys misst unter Profil-Last: Zeitmessung der Kerne ist genau, Hostlücken (Abschnitt 2/6, 1,3 ms Unterschied der Vorbereitung zwischen den Captures) können verfälscht sein.
- Zahl verschiedener Experten je Schicht und Runde ist nicht in der Datei; alle MoE-Bandbreiten sind darum Spannen (Abschnitt 4).
- Ob der MoE-Kern die komprimierten csf32-Skalen direkt liest oder expandierte (S4: Expander-Kern vorgeschaltet, S0–S3: nicht beobachtet), ist aus der Datei nicht entscheidbar; die Bytes nehmen die komprimierte Form an (6,8 % Aufschlag; bei expandierten Skalen ergäbe sich 12,5 %, die Bandbreiten wären 5 % höher).
- Die Zuordnung der kleinen GEMMs zu Tensoren (kv_b, Indexer) stützt sich auf Gittergrößen und Reihenfolge, nicht auf Namen.
