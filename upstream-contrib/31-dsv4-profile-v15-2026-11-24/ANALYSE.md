# DeepSeek-V4-Flash 284B (DSpark k = 5) Decode: Stufen, Kernklassen und verbleibende Hebel (nsys, 24.11.2026)

Nur Analyse (keine Änderung an Code, Config, GPU oder Git). Quelle: `~/.cache/nsys-runs/dsv4-v15-decode16k-1124/` (Capture 1 = Decode kurz, 26 Prompt-Token, 128 Token in 2,17 s; Capture 2 = Decode bei 16 262 Token Kontext, 128 Token in 2,34 s), Log `~/.cache/bench-scripts/profile_nsys_dsv4_v15_decode_2026-11-24.log`, Baum `1Cat-vLLM-v15` (prod-v15 0991b1656), Eintrag `DeepSeek-V4-Flash-284B-A13B-MXFP4-FP8-DSpark-vllm` (PP5, `VLLM_PP_LAYER_PARTITION=10,8,8,8,9`, `CUDA_VISIBLE_DEVICES=0,1,4,3,2`, DSpark k = 5, Graph-Größen 5 und 6, `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`).
Reproduktion: `analyse_nsys_dsv4.py <sqlite> [--vs <zweite sqlite>] [--tokens 128] [--json aus.json]` in diesem Ordner (erweitert `../29-glm-profile-v14k4-2026-11-23/analyse_nsys_stufen.py`: gleiche Passerkennung und Rundenlogik, dazu Rollenklassifikation je Kern, Fenster je Schicht, Leerlauf- und Kleinkernstatistik, Hebelrechnung, Vergleich der Captures). Alle Zahlen unten stammen aus den Läufen von Capture 1 (`--vs` Capture 2 für Abschnitt 5).

Kennzeichnung: **[gemessen]** = direkt aus der nsys-Datei, **[Header]** = safetensors-Header bzw. config.json, **[Quelle]** = Quelltext im Baum v15 (Datei:Zeile), **[abgeleitet]** = Rechnung aus gemessenen und Header-Werten, **[vermutet]** = Schluss ohne direkten Beleg.
Bandbreiten: Spitze nominell V100 900, Quadro RTX 8000 672 GB/s; praktische Decke 85 % (V100 765, RTX 571 GB/s). GB/s = Byte / µs / 1000 (dezimal). **Neu gegenüber der GLM-Analyse:** `TARGET_INFO_GPU` meldet für die RTX 624,1 GB/s (V100 898,0), in beiden DeepSeek- und in den GLM-Captures [gemessen]; das entspricht einem Speichertakt von 6,5 statt 7,0 GHz unter Last [vermutet: P2-Zustand]. Die Hauptzahlen bleiben gegen 672 (vergleichbar mit der GLM-Analyse), die Zusatzspalte „Treiber“ gegen 624: der LM-Kopf läuft dann mit 95,7 %, die großen QPN8-Kerne mit 88–90 %, der Anstieg der QPN8-Geraden (622 GB/s) liegt bei 99,7 % der Treiberspitze. Die RTX-Kerne haben also weniger Luft, als die 672-Rechnung zeigt.

## 0. Kurzfassung

- **Runde 50,73 ms** (Beginn Stufe 0 bis Beginn der nächsten Runde, Mittel über 38 Abstände, min 48,81, max 53,56), **3,26 Token je Runde** (127/39), unter nsys ≈ 64 tok/s. Bei 16k Kontext 54,28 ms und 3,53 Token je Runde (anderer Text, andere Annahme).
- **Stufenanteile (Durchlauf / Rundenzeit):** S0 RTX (Schichten 0–9) 11,59 ms = 22,9 %, S1 V100 7,62 = 15,0 %, S2 V100 7,58 = 14,9 %, S3 V100 7,37 = 14,5 %, **S4 RTX (Schichten 34–42 + DSpark-Entwurf) 15,84 = 31,2 %**, Übergaben 0,75 = 1,5 %. Die Stufen laufen strikt nacheinander; jede Karte ist im Mittel nur zu 18,6 % belegt (V100 14 %, S0 21 %, S4 30 %).
- **Die fünf größten Posten (Kernsummen aller Stufen, Anteil an 50,73 ms):** (1) **MoE-Kerne Up + Down 18,13 ms = 35,7 %**, (2) **wo_a 4,39 ms = 8,7 %** (auf den RTX-Stufen ein cuBLAS-FP16-GEMM auf entpacktem Gewicht, 133–135 µs), (3) **LM-Kopf 2 × 1,77 = 3,55 ms = 7,0 %** (BF16, 1,06 GB je Lesung, schon bei 88,9 %/95,7 % der Spitze), (4) **kleine Compressor-/Indexer-GEMMs 3,30 ms = 6,5 %** (V100 0,8 ms je Stufe), (5) wo_b 2,54 ms = 5,0 % (knapp davor/dahinter wq_b 2,45, mHC 2,42, Elementweise 2,41, Attention-Hilfskerne 2,11).
- **Gewichte-streamende Kerne 34,96 ms (68,9 %), übrige Kerne 14,66 ms (28,9 %), Lücken im Durchlauf 2,79 ms (5,5 %), Übergaben 0,75 ms (1,5 %).** MoE, wq_b, wo_b, LM-Kopf liegen an der Decke (Abschnitt 4). Das Potenzial liegt im „Rest“ und in zwei Kernen unterhalb der Decke (wo_a, Entwurfsköpfe).
- **Hebel-Rangliste (ms je Runde, abgeleitet, Punkt [Spanne]; Abschnitt 7):** 1. wo_a auf den RTX-Stufen als FP8-Gruppenkern 1,6 [1,4–1,7]; 2. Attention-Kern fusionieren (kleine Kerne + Leerlauf) 1,5 [0,9–2,0]; 3. Entwurfsköpfe (Kopf 2 + Markov) auf FP8/Teilvokabular 1,3 [0,7–1,6]; 4. Router-Kette in einen Kern 0,7 [0,4–0,9]; 5. Compressor-/Indexer-GEMMs auf Bytezeit 0,7 [0,4–1,1]; 6. mHC pre_norm 0,5 [0–1,1]; 7. wo_a V100 0,3 [0,2–0,4]. Summe Punkt 6,6 ms (13 %), Spanne 4,0–8,8 ms; nicht alles ist gleichzeitig realistisch.
- **MoE ist am Limit:** in den drei Hash-Schichten (L0–L2, Expertenzahl fast bekannt) läuft der Up-Kern der RTX mit 586 GB/s (87,2 % von 672, 94 % von 624) und damit mit 98 % seiner Grenzrate aus dem GLM-Mikrobenchmark (592–597 GB/s). Mehr gibt der Kern nicht her; Zeit je Schicht folgt der Zahl verschiedener Experten (16–34 von höchstens 36).
- **Offen/neu:** (a) Gewichtsbytes von wo_a auf Turing (FP16, 67 MB) sind aus Quelltext und Kernname abgeleitet, nicht gemessen (ncu); (b) in Capture 2 läuft die Stufe-0-Vorbereitung 0,8 ms länger (winzige Kerne bimodal 2,5 µs oder 10 µs), Ursache offen; (c) die Indexer-Kette (nur ratio-4-Schichten, ab Kontext > Top-k) kostet bei 16k 1,42 ms je Runde und wächst mit dem Kontext; ein Langkontext-Profil fehlt.

## 1. Zuordnung, Runden, Stufen

**deviceId → Stufe [gemessen + Config]:** deviceId 0..4 = Stufe 0..4 (`CUDA_VISIBLE_DEVICES=0,1,4,3,2` mit `PCI_BUS_ID`; die Tabelle `TARGET_INFO_GPU` nennt die PCI-Reihenfolge und wird nicht benutzt). Kernnamen bestätigen es: Stufen 0 und 4 laufen `skinny_nvfp4_moe_m16n8k8`, `turing_*gemm` (Quadro RTX 8000), Stufen 1–3 `skinny_nvfp4_moe_qpn`, `volta_*gemm`, `gemm_kernel` (V100). Partition 10, 8, 8, 8, 9 Zielschichten; Stufe 4 trägt zusätzlich den DSpark-Entwurf (`mtp.0`–`mtp.2`, 3 Schichten mit je 256 Experten, 5 Zeilen = MoE-Gitter y = 30), den Markov-Kopf und die zwei LM-Kopf-Lesungen.

**Runden [gemessen]:** Capture 1 hat je Stufe 39 Decode-Durchläufe (Segmente mit ≥ 16 MoE-Kernen auf Gitter (64, 36); 36 = 6 Zeilen × Top-6) plus einen Prefill-Durchlauf (Gitter y = 156 = 26 Token × 6) und je Stufe 30–40 kurze Vorbereitungsdurchläufe ohne MoE. Capture 2: 36 Decode-Runden (Prefill-Stücke y = 256). Tokens je Runde = 127/39 = 3,26 bzw. 127/36 = 3,53 [abgeleitet: 128 Ausgabetoken, 1 aus dem Prefill]. Zur Probe: 39 × 50,73 ms = 1,98 s plus Prefill/API ≈ 0,19 s = 2,17 s laut Log.

| Größe | Capture 1 | Capture 2 (16k) | Einheit |
|---|---|---|---|
| Rundenzeit (Mittel, Median) | 50,73 (50,32) | 54,28 (54,44) | ms je Runde, weniger ist besser |
| Streuung min / max / sd | 48,81 / 53,56 / 1,29 | 51,77 / 57,56 / 1,19 | ms |
| Tokens je Runde | 3,26 | 3,53 | Token |
| Rate unter nsys | 64,2 | 65,0 | tok/s (abgeleitet) |

Die Rundenzeit 50,73 ms liegt vor der GLM-Runde (60,74 ms), obwohl DSpark eine Zeile mehr prüft (6 gegen 5): DeepSeek hat weniger Schichten (43 gegen 45), keine csf32-Expansion und eine kürzere Hostvorbereitung (Abschnitt 2).

## 2. Seriell, jede Karte meist leer

| Stufe (Karte) | Durchlauf | belegt (Kernel-Vereinigung) | Lücken im Durchlauf | Leerlauf je Runde | Auslastung | Anteil Runde |
|---|---|---|---|---|---|---|
| | ms | ms | ms | ms | % | % |
| S0 RTX (10 Schichten) | 11,59 | 10,89 | 0,70 | 39,84 | 21,5 | 22,9 |
| S1 V100 (8) | 7,62 | 7,13 | 0,49 | 43,60 | 14,1 | 15,0 |
| S2 V100 (8) | 7,58 | 7,04 | 0,54 | 43,69 | 13,9 | 14,9 |
| S3 V100 (8) | 7,37 | 6,87 | 0,49 | 43,86 | 13,6 | 14,5 |
| S4 RTX (9 + Entwurf + 2 Köpfe) | 15,84 | 15,27 | 0,57 | 35,46 | 30,1 | 31,2 |
| Summe Durchläufe | 50,01 | | 2,79 | | | |
| Übergaben S0→S1, S1→S2, S2→S3, S3→S4, S4→S0 | 0,17 + 0,18 + 0,18 + 0,14 + 0,08 = 0,75 | | | | | 1,5 |

[gemessen]; Summe 50,01 + 0,75 = 50,76 gegen Rundenzeit 50,73. Leerlauf = Rundenzeit minus belegt [abgeleitet]. Mittlere Auslastung der fünf Karten 18,6 % [abgeleitet]. Die Partition ist durch den VRAM bestimmt (je Schicht 256 Experten × 13,37 MB = 3,42 GB plus Attention/Dense etwa 0,2 GB; 8 Schichten ≈ 29 GB auf 32 GB V100; gesamt etwa 168 GB von 192 GB) [Header, abgeleitet]; für die Rundenzeit zählt nur die Summe. Eine Schicht kostet auf der RTX nicht mehr als auf der V100, sofern sie keine Hash-Schicht ist (Abschnitt 4.1): Kernsumme/Spanne je Schicht [gemessen]: S4 ratio-4 984/963 µs, ratio-128 921/953; V100 ratio-4 1001–1051/925–977, ratio-128 861–874/913–925; S0 Schichten 0–1 (ratio-0, Hash) 1240/1268, ratio-4 1167/1151 (enthält die Hash-Schicht 2), ratio-128 981/1019; Entwurfsschicht 788/805. Eine Umverteilung von Schichten bringt daher nichts, und der VRAM erlaubt sie ohnehin nicht.

**Übergaben und Hostvorbereitung [gemessen]:**
- Übergaben 0,14–0,18 ms je Stufenwechsel (GLM 0,27–0,37): DeepSeek reicht nur einen Tensor weiter. NCCL-Kerne im Durchlauf: nur 1 Broadcast (7 µs) auf S4; die Wartekerne liegen außerhalb der Durchläufe.
- Vorbereitung vor dem ersten Stufe-0-Schichtkern (Durchlaufbeginn bis zum ersten mHC-Kern): Mittel 0,37, **Median 0,22 ms** (zwei Ausreißerrunden 2,61 und 1,69 ms am Anfang), also deutlich weniger als bei GLM (1,55 ms). Die etwa 36 winzigen Vorbereitungskerne (2–4 µs, Abstand 3–5 µs, dann einzelne Lücken) laufen in 0,22 ms. Der Host braucht von `execute_model` (NVTX `execute_context_0(0)_generation_1(6)`) bis `cudaGraphLaunch` 3,09 ms (Median; 87 CUDA-Aufrufe, der Graph-Aufruf selbst 0,78 ms) und beginnt 3,51 ms vor dem Ende von Stufe 4; die GPU wartet daher nur 0,23 ms (Median) auf den Host [gemessen, NVTX/RUNTIME-Tabellen].
- Auf S1–S4 läuft vor jedem Hauptdurchlauf ein Vorbereitungsdurchlauf (23 Kerne, 1,5 ms), der parallel zur Vorstufe liegt und nicht auf dem kritischen Pfad ist.
- **Capture 2 weicht ab:** Stufe-0-Vorbereitung Median 1,05 ms (statt 0,22), obwohl Hostzeit gleich ist (3,01 ms bis `cudaGraphLaunch`). Die winzigen Vorbereitungskerne sind in 23 von 36 Runden ≈ 10 µs lang statt 2,5 µs (zwischen den Runden bimodal, Übergangswerte 8,7/8,1/7,1/6,6) [gemessen]. Die Schichtkerne selbst sind gleich schnell. Das kostet +0,83 ms je Runde (1,5 %). Ursache offen (Fremdlast, Taktzustand der im Mittel zu 79 % leeren Karte oder Hostspeicherzugriff der Kerne; [vermutet]); nur in einem von zwei Captures, in Produktion nicht nachgewiesen.

## 3. Modellgrößen (Header, exakt)

config.json: hidden 4096, moe_intermediate 2048, **256 geroutete Experten, Top-6**, 1 Shared Expert (2048), 43 Schichten (alle MoE; Schichten 0–2 Hash-Routing mit `tid2eid` [129 280, 6]), 64 Köpfe, head_dim 512, 1 KV-Kopf, o_groups 8, o_lora_rank 1024, q_lora_rank 1024, sliding_window 128, index_topk 512, hc_mult 4, hc_sinkhorn_iters 20, `compress_ratios`: Schichten 0, 1 = 0 (nur Fenster), ab 2 gerade = 4 (CSA mit Indexer), ungerade = 128 (HCA); DSpark: `dspark_block_size` 5, Entwurf `mtp.0`–`mtp.2` (Verhältnis 0).

| Tensor | Format | Bytes | Quelle |
|---|---|---|---|
| Experte w1 + w3 („Up“-Kern) | I8 (MXFP4 gepackt) [2048, 2048] + e8m0-Skalen [2048, 128], je Matrix | 2 × (4 194 304 + 262 144) = **8,913 MB** | Header |
| Experte w2 („Down“-Kern) | I8 [4096, 1024] + e8m0 [4096, 64] | **4,456 MB** (je Experte 13,37 MB; Skalen 5,9 % Aufschlag) | Header |
| Router `ffn.gate` BF16 [256, 4096] | BF16 | 2,10 MB | Header |
| Shared Expert w1 + w3 / w2 | FP8 [2048, 4096] / [4096, 2048] + e8m0 | 16,78 / 8,39 MB | Header |
| wq_a [1024, 4096] + wkv [512, 4096] (fusioniert) | FP8 | 6,29 MB | Header |
| wq_b [32768, 1024] | FP8 | 33,56 MB | Header |
| wo_a [8192, 4096] (8 Gruppen × 1024 Zeilen) | FP8 | 33,56 MB (RTX nach dem Laden FP16 67,11 MB, siehe 4.2) | Header + Quelle |
| wo_b [4096, 8192] | FP8 | 33,56 MB | Header |
| ratio-4: Compressor wkv + wgate [1024, 4096] je | BF16 | 2 × 8,39 MB | Header |
| ratio-4: Indexer-Compressor wkv + wgate [256, 4096] je; weights_proj [64, 4096]; wq_b [8192, 1024] | BF16; BF16; FP8 | 2 × 2,10 MB; 0,52 MB; 8,39 MB | Header |
| ratio-128: Compressor wkv + wgate [512, 4096] je | BF16 | 2 × 4,19 MB | Header |
| mHC `hc_attn_fn`, `hc_ffn_fn` [24, 16384] | F32 | je 1,57 MB | Header |
| LM-Kopf `head.weight` [129 280, 4096] | **BF16** | **1 059,06 MB** | Header |
| Markov-Kopf `markov_w1`, `markov_w2` [129 280, 256] | BF16 | je 66,19 MB | Header |
| Entwurf `main_proj` [4096, 12288] | FP8 | 50,33 MB | Header |

Zuordnung Kern → Tensor [abgeleitet aus Gitter, Reihenfolge, Stückzahl je Runde und Bytes]: `fp8_qpn8` (48, 512) = wq_a+wkv, (1024, 256) = wq_b, (128, 512) in der Attention-Hälfte = wo_b, (128, 512)/(128, 256) in der FFN-Hälfte = Shared gate+up / down, (256, 256) = Indexer wq_b (nur bei langem Kontext), (128, 512) mit 97 µs auf S4 vor den Entwurfsschichten = `main_proj`; `Kernel2 (8, 1010, 1)` = LM-Kopf (1010 × 128 = 129 280 Zeilen); `gemv2T_kernel_val (16160, …)` + `reduce_kernel (1, 1, 1)` = Markov-Kopf (5 Schritte); `turing_fp16_s1688gemm … tn (8, 1, 8)` bzw. `gemm_kernel (·, ·, 2)` = wo_a (Gitter-z 8 = 8 Gruppen).

## 4. Kernklassen je Stufe

Zeit je Runde in µs (Summe der Kernzeiten; Anteil an der Summe der Kernzeiten der Stufe in %; n = Aufrufe je Runde S0/S1/S4). [gemessen]

| Rolle | S0 RTX | S1 V100 | S2 V100 | S3 V100 | S4 RTX | Summe ms | % Runde | n S0/S1/S4 |
|---|---|---|---|---|---|---|---|---|
| MoE Up | 3681 | 1970 | 1899 | 1768 | 2838 | 12,15 | 24,0 | 10/8/12 |
| MoE Down | 1865 | 941 | 905 | 838 | 1430 | 5,98 | 11,8 | 10/8/12 |
| wo_a | 1328 | 475 | 480 | 483 | 1624 | 4,39 | 8,7 | 10/8/12 |
| LM-Kopf (2 je Runde) | – | – | – | – | 3547 | 3,55 | 7,0 | 0/0/2 |
| Compressor-/Indexer-GEMMs (`Kernel2`, `gemmSN`, `splitKreduce`) | 404 | 799 | 803 | 806 | 483 | 3,30 | 6,5 | 28/28/33 |
| wo_b | 609 | 402 | 403 | 404 | 726 | 2,54 | 5,0 | 10/8/12 |
| wq_b | 597 | 378 | 378 | 379 | 718 | 2,45 | 4,8 | 10/8/12 |
| mHC (pre_norm, post_dot, Staging, hc_head) | 448 | 462 | 462 | 465 | 578 | 2,42 | 4,8 | 41/33/54 |
| Elementweise (Torch-Kleinkerne) | 406 | 490 | 497 | 501 | 518 | 2,41 | 4,8 | 119/89/145 |
| Attention-Hilfskerne (KV-Einfügen, Dequant-Gather, Rope, Softmax, Indizes) | 400 | 394 | 394 | 399 | 519 | 2,11 | 4,2 | 85/64/109 |
| Shared gate+up | 331 | 245 | 246 | 247 | 393 | 1,46 | 2,9 | 10/8/12 |
| Attention QK/PV-GEMMs | 292 | 250 | 249 | 252 | 360 | 1,40 | 2,8 | 20/16/24 |
| Compressor-Kerne (partial states, kv_compress) | 181 | 206 | 203 | 207 | 218 | 1,01 | 2,0 | 24/24/28 |
| Shared down | 174 | 148 | 148 | 149 | 207 | 0,83 | 1,6 | 10/8/12 |
| wq_a+wkv | 149 | 136 | 136 | 138 | 178 | 0,74 | 1,5 | 10/8/12 |
| Markov-Kopf (5 × gemv + reduce) | – | – | – | – | 733 | 0,73 | 1,4 | 0/0/10 |
| Router Top-k (`topkGatingSoftplusSqrt`, Gruppenrouting) | 104 | 141 | 142 | 144 | 125 | 0,66 | 1,3 | 20/16/24 |
| Router Gate (`Kernel2` + `splitKreduce`) | 91 | 134 | 134 | 135 | 112 | 0,61 | 1,2 | 20/16/24 |
| MoE Aktivierung / Reduktion | 81 | 75 | 75 | 76 | 88 | 0,40 | 0,8 | |
| Shared Aktivierung, Norm | 52 | 65 | 65 | 66 | 80 | 0,33 | 0,6 | |
| `main_proj`, Kontext-Projektionen, Sampler (nur S4) | – | – | – | – | 164 | 0,16 | 0,3 | |
| **Kernsumme** | 11 193 | 7 711 | 7 619 | 7 457 | 15 640 | 49,62 | 97,8 | |

### 4.1 MoE-Kerne

**Zeit je Schicht [gemessen, Mittel über 39 Runden, µs]:**

| Gruppe | Up-Kern Mittel [min…max] | Down-Kern Mittel [max] | Up/Down | gleichwertige Expertenzahl N (bei Grenzrate) |
|---|---|---|---|---|
| S0 Hash-Schichten L0–L2 (6 Zeilen, Hash-Routing) | 509 [413…548] | 258 [277] | 1,98 | **34,0** (Rate 594 GB/s) |
| S0 L3–L9 | 308 [182…418] | 156 [212] | 1,97 | 20,5 |
| S1 V100 L10–L17 | 246 [177…380] | 118 [187] | 2,09 | 20,5 (Rate 742 GB/s) |
| S2 V100 L18–L25 | 237 [129…336] | 113 [163] | 2,10 | 19,8 |
| S3 V100 L26–L33 | 221 [143…327] | 105 [160] | 2,11 | 18,4 |
| S4 RTX L34–L42 (6 Zeilen) | 244 [143…385] | 123 [196] | 1,98 | 16,3 |
| S4 Entwurf L43–L45 (5 Zeilen, y = 30) | 212 [158…308] | 107 [79…156] | 1,98 | 14,1 |

**Der Kern liegt an der Decke [abgeleitet, belastbar nur an den Hash-Schichten]:** Die Zahl verschiedener Experten je Schicht und Runde steht nicht in der Datei. In L0–L2 bestimmt `tid2eid` die Experten aus den Token-IDs; für 6 Zeilen × 6 Experten, zufällig verteilt, ist der Erwartungswert verschiedener Experten 256·(1 − (250/256)^6) = **33,9** (höchstens 36 Plätze). Gemessen laufen diese Schichten mit 509 µs (Up) bei Maximum 548 µs. Setzt man N = 36 für das Maximum, ergibt das 87,1 % der Spitze (586 GB/s; 94 % der Treiberspitze); setzt man die Grenzrate des Kerns aus `../29-glm-profile-v14k4-2026-11-23/MOE_MESSUNG.md` an (RTX 592–597 GB/s, V100 mit Ring 739–746 GB/s), kommt aus der Mittelzeit N = 34,0, also genau der Erwartungswert. Zwei unabhängige Überschlagsrechnungen treffen sich: **der RTX-Kern läuft in den Hash-Schichten bei etwa 98 % seiner Grenzrate**. Die tiefen Schichten brauchen weniger Zeit, weil die 6 Zeilen eines Prüfschritts (Anker + 5 Entwurfstoken) überlappende Experten wählen (N ≈ 16–21 gleichwertig); die Zeit ist proportional zu N (Anpassung Up = a + b·Down mit b = 1,92–1,98; a = −1 µs RTX S0, 18–20 µs V100, 4 µs S4). Das V100-a von 18–20 µs je Schicht ist der bekannte feste Anteil (Rampe/Schwanz der zwei Kerne, `MOE_MESSUNG.md`). Der Hebel „Aktivierung und Reduktion als Epilog“ wurde dort bereits verworfen (24.11.).
Vergleich mit GLM [gemessen]: gleiche Kernfamilie, nahezu gleiche Bytes je Experte (8,913 gegen 8,956 MB); GLM-S0-Maximum 641 µs bei N = 40 (83 %), DeepSeek-S0-Maximum 548 µs bei N = 36 (87 %). **Kein Hinweis auf einen DeepSeek-Rückstand.** MXFP4-e8m0-Skalen brauchen keinen csf32-Expander (GLM S4: 0,55 ms entfällt hier). Hebel an dieser Stelle: nur die Zahl verschiedener Experten (Algorithmik, z. B. Prüflänge; `28-glm-adaptive-verify/ERGEBNIS.md`: lohnt nicht, < 1 %), nicht der Kern.

### 4.2 Dense-Projektionen (QPN8 und cuBLAS)

Bytes aus Abschnitt 3, Zeit je Aufruf [gemessen], GB/s [abgeleitet]; „Treiber“ = Anteil der nsys-Spitze (RTX 624, V100 898).

| Klasse | Karte | n/Runde | µs je Aufruf | MB | GB/s | % Spitze (Treiber) | Gerade a + B/BW (GLM) | ms/Runde |
|---|---|---|---|---|---|---|---|---|
| wq_a+wkv | RTX S0/S4 | 10/12 | 14,9 | 6,29 | 422/424 | 62,8/63,0 (67,6/67,9) | 17,7 | 0,15/0,18 |
| wq_a+wkv | V100 S1–S3 | 8 | 17,0–17,3 | 6,29 | 364–370 | 40–41 | 17,1 | 0,14 |
| wq_b | RTX S0/S4 | 10/12 | 59,7/59,8 | 33,56 | 562/561 | 83,6/83,5 (90,1/89,9) | 61,5 | 0,60/0,72 |
| wq_b | V100 S1–S3 | 8 | 47,2–47,4 | 33,56 | 708–711 | 78,6–79,0 | 49,1 | 0,38 |
| wo_b | RTX S0/S4 | 10/12 | 60,9/60,5 | 33,56 | 551/555 | 82,0/82,6 (88,3/88,9) | 61,5 | 0,61/0,73 |
| wo_b | V100 S1–S3 | 8 | 50,2–50,5 | 33,56 | 665–668 | 73,9–74,2 | 49,1 | 0,40 |
| **wo_a** | **RTX S0/S4** | 10/12 | **132,8/135,4** | **67,11 (FP16)** | 505/496 | 75,2/73,8 (81,0/79,4) | – | **1,33/1,62** |
| **wo_a** | **V100 S1–S3** | 8 | **59,4–60,4** | 33,56 (FP8) | 556–565 | 61,7–62,8 | – | 0,48 |
| Shared gate+up | RTX S0/S4 | 10/12 | 33,1/32,8 | 16,78 | 507/512 | 75,5/76,2 | 34,6 | 0,33/0,39 |
| Shared gate+up | V100 | 8 | 30,7–30,9 | 16,78 | 543–547 | 60,4–60,8 | 29,4 | 0,25 |
| Shared down | RTX S0/S4 | 10/12 | 17,4/17,2 | 8,39 | 483/486 | 71,8/72,4 | 21,1 | 0,17/0,21 |
| Shared down | V100 | 8 | 18,5–18,7 | 8,39 | 450–454 | 50,0–50,5 | 19,5 | 0,15 |
| `main_proj` (Entwurf) | RTX S4 | 1 | 97,0 | 50,33 | 519 | 77,2 (83,1) | 88,5 | 0,10 |
| LM-Kopf (Kopf 1 + 2) | RTX S4 | 2 | 1773,3 | 1059,06 | 597 | **88,9 (95,7)** | – | 3,55 |

- **Die QPN8-Kerne liegen alle auf der Geraden t = a + Byte/BW** aus `KLEINE_GEMMS.md` (V100 a = 9,7 µs, BW 852 GB/s; RTX a = 7,6 µs, BW 622 GB/s): Vorhersage gegen Messung V100 wq_b 49,1/47,2, wo_b 49,1/50,2, Shared gate+up 29,4/30,7, Shared down 19,5/18,5, wq_a+wkv 17,1/17,0; RTX wq_b 61,5/59,7, wo_b 61,5/60,9, Shared gate+up 34,6/33,1. Die Kerne sind dieselben wie bei GLM und erreichen dieselben Anteile (V100 qkv 79 %, RTX 82–84 %). **Kein DeepSeek-spezifischer Rückstand; verloren geht nur der feste Anteil a**, im Mittel 5 QPN8-Kerne je Schicht × 9,7 µs (V100) bzw. 7,6 µs (RTX): 1,16 + 0,84 = 2,0 ms je Runde, der durch kein Kern-Tuning (Skalen-Vorladen wurde am 24.11. verworfen) zu holen ist, sondern nur durch weniger Kerne.
- **wo_a ist der einzige Dense-Posten klar unter der Geraden.** V100: TurboMind-Gruppenkern `gemm_kernel` auf FP8 (33,56 MB in 59,4 µs = 565 GB/s; mit FP16-Bytes wären es 1130 GB/s, über der Spitze, also zwingend FP8 [abgeleitet]). RTX: `turing_fp16_s1688gemm … tn` (cuBLAS) mit Gitter-z 8; der Quelltext entpackt `is_bmm`-Schichten auf Turing beim Laden nach FP16 (`vllm/model_executor/layers/quantization/fp8.py:494-520`, `layer.dequantized_bmm = True`, Meldung „Block FP8 QPN8 unavailable: grouped BMM uses the existing dequantized grouped implementation“), die V100 nutzt dagegen `qpn/fp8.py:575-640` (TurboMind-FP8-Gruppen) [Quelle]. Bei 67,11 MB in 133 µs = 505 GB/s wäre das 75 % von 672 und selbst nach FP16-Bytes noch 15 % über der Geraden (7,6 + 67,11/0,622 = 115,5 µs). **wo_b (gleiche FP8-Bytes, N = 4096, K = 8192) braucht 60,9 µs**; ein FP8-Gruppenkern auf dem Niveau von wo_b spart 72–75 µs je Aufruf × 22 Aufrufe.
- **LM-Kopf:** 1773 µs für 1,059 GB, 597 GB/s = 95,7 % der Treiberspitze: der Kern ist am Limit. Er kostet 3,55 ms je Runde nur wegen der Bytes (BF16; GLM hat MXFP8 mit 654 MB und 1,13 ms je Aufruf, also 0,64 ms weniger je Lesung).

### 4.3 Attention-Seite, Compressor/Indexer, Router, mHC (die Fenster je Schicht)

Die kleinen Kerne sind latenzgebunden und stehen in Ketten. Wandzeit je Schicht und Fenster [gemessen, Capture 1, µs]; Fenster: **Compress** = Ende wq_a+wkv bis Start Norm; **Attention-Kern** = Ende wq_b bis Start wo_a; **FFN-Vorspann** = Ende wo_b bis Start MoE-Up (mHC post/pre, Router, Shared Expert); **MoE-Block** = Start Up bis Ende Down.

| Schichttyp (Karte) | Compress | Attention-Kern (Kerne / davon < 6 µs, µs / Leerlauf µs) | FFN-Vorspann | MoE-Block |
|---|---|---|---|---|
| ratio-0 (S0, Schichten 0–1, Hash-Routing) | – | 73 (11 / 5, 13 / 9) | 109 | 772 |
| ratio-4 (S0 RTX) | 49 | 126 (23 / 13, 38 / 16) | 111 | 552 |
| ratio-128 (S0 RTX) | 28 | 104 (18 / 12, 36 / 17) | 111 | 462 |
| ratio-4 (V100, S1–S3) | **70** | **160 (24 / 13, 50 / 21)** | **143–149** | 314–371 |
| ratio-128 (V100) | **43–44** | **141–144 (19 / 12, 46–47 / 21–23)** | **140–145** | 349–368 |
| ratio-4 (S4 RTX) | 50 | 124 (23 / 12, 34 / 15) | 106 | 358 |
| ratio-128 (S4 RTX) | 28 | 106 (18 / 12, 36 / 17) | 107 | 393 |
| Entwurfsschicht (S4, ratio-0) | – | 70 (11 / 6, 14 / 6) | 101 | 322 |

- **Kleine Kerne gesamt:** 183 (V100) bis 310 (S4) Kerne unter 6 µs je Durchlauf, zusammen 0,70–0,91 ms je Stufe; Leerlauf zwischen den Kernen im Durchlauf: 0,5–0,7 ms je Stufe, davon 276 Lücken < 1,5 µs (0,27 ms) und 46–48 Lücken von 3–10 µs (0,19 ms, V100). Gesamt 2,79 ms Lücken je Runde. Kerne je Runde: S0 487, S1–S3 je 390, S4 619, insgesamt 2 276; im Graph ist die Lücke 0,7–1,2 µs je Kern.
- **Compress-Fenster:** V100 ratio-4 70 µs, ratio-128 43,5 µs für 21,5 bzw. 8,4 MB BF16 (Bytezeit bei 85 %: 28 bzw. 11 µs) [abgeleitet]. Ursache sichtbar im Kernel-Trace [gemessen]: ein cuBLAS-`gemmSN_TN_kernel` mit Gitter (8, 1, 1) = 8 CTAs auf 80 SMs (66,5 µs), daneben zwei `Kernel2` von 41 µs (ratio-4), bzw. ein `Kernel2 (8, 4, 4)` mit 37,7 µs (ratio-128). RTX: 49/28 µs (Bytezeit 38/15 µs).
- **FFN-Vorspann V100 143 µs** (RTX 106–111): `post_dot` 8,6 → `pre_norm` 19,6 → Router `Kernel2` 13,4 + `splitKreduce` 4,0 + `unrolled` 8,4 → Shared gate+up 31 → act_and_mul 5 → Shared down 18 → Top-k 12,8 → Gruppenrouting 5,0 → Up-Kern [gemessen, Layer-Trace]. Die Bytezeit des Router- und Shared-Anteils (27,3 MB) wäre 36 µs bei 85 %.
- **mHC `pre_norm`:** 92 Aufrufe je Runde, **6 CTAs × 1024 Threads** (Gitter (6, 1, 1)), RTX 13,9 µs, V100 19,8 µs, zusammen 1,56 ms (3,1 % der Runde); `post_dot` (Gitter (6, 4, 8)) 8,6 µs. Gleicher Kern wie bei GLM (mHC dort 2,28 ms, 3,8 %). Er liest 1,57 MB `hc_fn`, rechnet aber 20 Sinkhorn-Iterationen; wie viel davon Latenzkette ist, zeigt nur ncu.

### 4.4 Stufe 4 im Einzelnen (Wandzeit, ms; Mittel [min…max])

| Abschnitt | ms | Anteil Runde |
|---|---|---|
| Zielschichten 34–42 (erste mHC bis letzter MoE-Down) | 8,58 [8,05…9,09] | 16,9 % |
| Schwanz bis Kopf 1 | 0,05 | 0,1 % |
| **Kopf 1 (Verify, BF16)** | 1,775 | 3,5 % |
| Abnahme + `main_proj` + 3 Kontext-Projektionen (bis erste Entwurfs-mHC) | 0,40 | 0,8 % |
| **Entwurfsschichten (3, je 5 Zeilen, 256 Experten)** | 2,42 [2,24…2,58] | 4,8 % |
| **Kopf 2 (Entwurf, BF16)** | 1,771 | 3,5 % |
| **Markov-Kopf, 5 Schritte + Rest bis Durchlaufende** | 0,83 | 1,6 % |
| Summe | 15,83 | 31,2 % |

Die Entwurfskette nach der Abnahme (0,40 + 2,42 + 1,77 + 0,83 = 5,42 ms) kostet 10,7 % der Runde; mit Kopf 1 sind es 7,2 ms (14,2 %). Markov-Kopf [gemessen]: 5 × `gemv2T_kernel_val` (Gitter 16 160, 113,9 µs für eine [129 280, 256]-BF16-Matrix, 66,19 MB: 581 GB/s = 93 % der Treiberspitze) + 5 × `reduce_kernel` mit **1 CTA** (32,7 µs, Argmax/Reduktion über 129 280 Werte), Schritt-Periode 155 µs. Die fünf Schritte sind abhängig (Token i+1 braucht das Argmax von Schritt i) und lesen dieselbe Matrix fünfmal.

## 5. Capture 2 (16k Kontext) gegen Capture 1

Rundenzeit 50,73 → 54,28 ms (+3,55). Durchläufe je Stufe S0 11,59 → 12,70 (+1,11), S1 7,62 → 7,96 (+0,34), S2 7,58 → 8,17 (+0,59), S3 7,37 → 7,90 (+0,53), S4 15,84 → 16,58 (+0,74); Übergaben 0,75 → 0,89. Kernsummen je Rolle [gemessen, µs je Runde, alle Stufen]:

| Posten | Capture 1 | Capture 2 | Δ | Anmerkung |
|---|---|---|---|---|
| **Indexer-Kette** (`_prefill_index_scores`, `persistent_topk`, `_gather_paged_index_k`) | 0 | 1004 | **+1004** | nur ratio-4-Schichten (21 Stück), ab Kontext > Top-k; 3,8–4,7 Aufrufe je Stufe und Runde |
| **Indexer wq_b** (`fp8_qpn8` (256), 8,39 MB) | 0 | 419 | **+419** | 17–24 µs je Aufruf |
| MoE Up + Down | 18 134 | 18 881 | +747 | **inhaltsabhängig** (andere Texte, andere Expertenzahl), nicht kontextbedingt [vermutet] |
| Elementweise | 2 413 | 2 688 | +275 | |
| Attention-Hilfskerne (u. a. `_dequant_gather_rows (6, 512)` +171) | 2 107 | 2 372 | +265 | Gather der 512 Top-k-Einträge statt Kurzkontext-Abkürzung |
| Compressor-Kerne | 1 015 | 1 276 | +261 | |
| wq_b | 2 450 | 2 552 | +102 | |
| Vorbereitung S0 (nicht in Kernsummen) | 0,22 ms | 1,05 ms | +0,83 ms | Abschnitt 2 |
| Lücken im Durchlauf | 2,79 ms | 3,74 ms | +0,95 ms | davon S0 +0,59 (Vorbereitung) |

Je ratio-4-Schicht wächst das Attention-Kern-Fenster [gemessen] von 126 → 184 µs (RTX S0), 160 → 234–238 (V100), 124 → 183 (S4); ratio-128 unverändert (+2 bis +4 µs); Compress-Fenster gleich; FFN-Vorspann +1 bis +26 µs (am meisten S0-Schichten 0–1 mit +26 und S4 mit +15 bis +18, vermutlich dieselbe Langsamkeit der Kleinkerne wie in der S0-Vorbereitung); MoE-Block inhaltsabhängig. **Mit dem Kontext wachsen also die Indexer-Kette und ihr FP8-GEMM (+1,42 ms bei 16k: 1,00 + 0,42), die Compressor- und Attention-Hilfskerne und die Stufe-0-Vorbereitung; Gewichtsstrom (MoE, Dense, Köpfe) und mHC bleiben gleich.** Die Indexer-Kette skaliert vermutlich linear mit der Zahl der komprimierten Schlüssel (16k/4 = 4 096; bei 500k wären es 125 000) [vermutet]; sie ist bei 16k mit 2,6 % noch klein und bei langem Kontext ungemessen (Abschnitt 8).

## 6. Vergleich mit GLM-5.3-Flash (dieselben Kerne)

| Größe | GLM (v14, k = 4) | DeepSeek (v15, k = 5) | Befund |
|---|---|---|---|
| Rundenzeit | 60,74 ms | 50,73 ms | |
| Tokens je Runde | 3,34 | 3,26 | |
| Übergaben | 1,35 ms | 0,75 ms | ein Tensor statt mehrerer |
| Vorbereitung vor Stufe 0 | 1,55 ms | 0,22 (Median) / 0,37 ms | |
| Lücken im Durchlauf | 3,35 ms | 2,79 ms | |
| MoE-Kern RTX S0 (Maximum, Experten-Annahme) | 641 µs (N = 40, 83 %) | 548 µs (N = 36, 87 %) | DeepSeek ≥ GLM |
| QPN8 groß (V100/RTX % Spitze) | qkv 79/82–83 | wq_b 79/84 | gleich |
| QPN8 128 CTAs (Shared gate+up V100/RTX) | 63–64/72–74 | 61/75–76 | gleich |
| Shared down (V100/RTX) | 49–50/65–71 | 50/72 | gleich |
| mHC je Stufe (V100, 7–8 Schichten) | 0,40 ms | 0,46 ms | gleicher Kern, gleiche Last je Schicht |
| LM-Kopf je Aufruf | 1,13 ms (MXFP8, 654 MB, 86 %) | 1,77 ms (BF16, 1 059 MB, 88,9 %) | gleich effizient, **1,6× Bytes** |
| csf32-Expansion S4 | 0,55 ms | entfällt | |
| wo_a / o_proj | DSA o_proj 84–87 % (kein Gruppen-GEMM) | **62 % (V100), 74–75 % bei doppelten Bytes (RTX)** | **einziger deutlicher Rückstand**; kein GLM-Gegenstück |

Hinweis aus dem Vergleich: DeepSeek liegt nirgends unter den GLM-Anteilen, wo derselbe Kern läuft (MoE, QPN8, mHC, Köpfe). Die Abstände entstehen dort, wo DeepSeek ein eigenes Format oder einen eigenen Weg hat: **wo_a (Gruppen-GEMM, auf Turing als FP16 entpackt)** und der **BF16-LM-Kopf**.

## 7. Rangliste der Hebel (ms je Runde, weniger ist besser)

Basis 50,73 ms. Spalte „Effekt“ = Punktwert / 50,73 (gleiche Annahmerate, ≈ tok/s-Gewinn). „Bit-gleich“ meint dieselbe Ausgabe wie heute. Hebel sind nicht überlappungsfrei addierbar, wo Fenster zusammenfallen (Rang 2 und 4/5/6 liegen in verschiedenen Fenstern und überlappen nicht; die Fenster stehen in 4.3). Rechnungen: `analyse_nsys_dsv4.py`, Abschnitt „Hebelrechnung“ (vorsichtig / Punkt / optimistisch).

| Rang | Hebel | ms Punkt [Spanne] | Effekt | generisch / DSv4 | bit-gleich | Aufwand | Risiko | Beleg |
|---|---|---|---|---|---|---|---|---|
| 1 | **wo_a auf den RTX-Stufen als FP8-Gruppenkern** (statt cuBLAS-FP16 auf entpacktem Gewicht): ein QPN8-Kern mit Gruppen-Dimension (Gitter n/32 × 8), Ziel Zeit von wo_b bei gleichen Bytes (60,7 µs); Nebeneffekt −33,6 MB × 22 = **0,74 GB VRAM** auf den RTX-Stufen | **1,6 [1,4–1,7]** | 3,2 % | DSv4-spezifisch (`is_bmm`-Schicht); gilt auch für V4.1 | nein: gleiche Gewichtswerte exakt, aber FP32-Summenreihenfolge anders; Token können kippen | mittel (Kern bekommt Gruppenzeiger/-Stride, Route `fp8.py` für `is_bmm` auf Turing) | gering bis mittel; Prüfstufe 2 (Tempo ist das Ziel) | 4.2; 22 Aufrufe × (133–135 − 61) µs |
| 2 | **Attention-Kern: kleine Kerne und Leerlauf fusionieren** (im Fenster 11–24 Kerne, davon 5–13 unter 6 µs = 13–50 µs plus 6–21 µs Leerlauf je Schicht; Ziel 25 µs je Schicht für alles zusammen: KV-Einfügen + Dequant-Gather + Rope + Index-/Längen-Elementweise in wenige Kerne, Hilfsrechnungen einmal je Runde statt je Schicht) | **1,5 [0,9–2,0]** | 3,0 % | generisch (DSv4 hat 43 Attention-Schichten, GLM-DSA 11); Ketten ähnlich | teils (Elementweises ja; Fusion von Softmax/GEMM ändert die Summenreihenfolge) | **hoch** (erst Kern→Python-Op zuordnen, dann Triton-Kerne; Indexpfade für ratio-0/4/128) | mittel (Indexlogik, Kurzkontext-Abkürzungen); Prüfstufe 1–2 | 4.3: 43 Schichten × (Kleinkerne + Leerlauf − 25 µs) |
| 3 | **Entwurfsköpfe nur für den Entwurf verkleinern:** Kopf 2 (1,77 ms) als FP8-Kopie (0,53 GB) → 0,89 ms oder Teilvokabular 98 304 (BF16, 0,80 GB) → 1,35 ms, FP8 + Teilvokabular → 0,67 ms; Markov-Gemv in FP8 −0,29 ms, Argmax im Gemv-Epilog statt 1-CTA-Reduce (5 × 33 µs) −0,12 ms | **1,3 [0,7–1,6]** (Kopf 2: 0,4 / 0,9 / 1,1; Markov: 0,3 / 0,4 / 0,5) | 2,6 % | DSpark-spezifisch (Mechanismus `draft_token_map` existiert für das 27B, für DSpark ungeprüft) | **Ausgabe ja** (Rejection-Sampling bleibt exakt), **Annahmerate nein** | gering bis mittel (FP8-Kopie + Ladepfad, Vokabularkarte für 129 280 Token) | **Annahmerate**: 1 ms Ersparnis (2 %) wird von −2 % Token je Runde (3,26 → 3,19) aufgehoben; VRAM +0,4–0,8 GB auf S4 | 4.4; Kopf 1 (Verify) bleibt BF16, verlustfrei nicht verkleinerbar |
| 4 | **Router-Kette in einen Kern** (Gate-GEMM + `splitKreduce` + Top-k-Sqrt-Softplus + Gruppenrouting; heute 19,5 µs je Schicht RTX, 34 µs V100, Ziel 12 µs) | **0,7 [0,4–0,9]** | 1,4 % | generisch (GLM hat dieselbe Kette) | ja möglich (gleiche Rechnung, Gate bleibt BF16-Skinny) | mittel | gering bis mittel (Hash-Schichten 0–2 mit `tid2eid` getrennt) | 4.3 FFN-Vorspann |
| 5 | **Compressor-/Indexer-GEMMs auf Bytezeit:** eigener Skinny-BF16-Kern statt `gemmSN_TN` (8 CTAs!) und zwei `Kernel2` (Vorbild: kv_b-Triton-Kern aus `KLEINE_GEMMS.md`, 34 → 26 µs); Ziel 1,5 × Bytezeit | **0,7 [0,4–1,1]** | 1,4 % | generisch (GLM: gleiche Klasse „kleine GEMMs“) | nein (Reduktionsfolge), verlustfrei | mittel | gering; V100-lastig (0,5 von 0,7 ms) | 4.3 Compress-Fenster |
| 6 | **mHC pre_norm auf mehr CTAs** (6 CTAs × 1024 Threads; 92 Aufrufe, 1,56 ms): Aufteilung über die 24 Ausgaben statt über K lässt die Summenreihenfolge je Ausgabe unverändert | **0,5 [0–1,1]** | 1,0 % | generisch (GLM: gleiche Kernfamilie, 2,28 ms) | ja möglich (Aufteilung nach Ausgabespalten; [vermutet]) | gering bis mittel | **unsicher**: die 20 Sinkhorn-Iterationen können die Untergrenze bestimmen; erst ncu | 4.3 |
| 7 | **wo_a V100 (TurboMind-Gruppenkern) auf wq_b-Niveau** (59,4 → 47,3 µs) | **0,3 [0,2–0,4]** | 0,6 % | DSv4-spezifisch | ja möglich | mittel (TurboMind-Gitter) | gering | 4.2 |
| – | Shared-Experte auf Aux-Stream | 0,4 (gemessen +0,8 %) | 0,8 % | generisch | ja | Mechanismus im Baum | **bekannt, auf Wunsch nicht weiterverfolgt** (GLM Prefill −2 %) | `MOE_MESSUNG.md` |
| – | Hostvorbereitung vor Stufe 0 | 0,0–0,2 | | | | | bekannt: bei GLM nichts Messbares; hier Median 0,22 ms; Capture 2 +0,8 ms Ursache offen | Abschnitt 2 |
| – | Übergaben gepackt | 0 | | | | | bekannt neutral (ein Tensor) | |
| – | MoE-Kerne (35,7 %), LM-Kopf Verify, wq_b, wo_b, Shared | 0 | | | | | **an der Decke** bzw. auf der Geraden (festes a nur über weniger Kerne) | 4.1, 4.2 |

**Summe Punktwerte Rang 1–7: 6,6 ms = 13,0 % der Runde** (+15 % tok/s bei gleicher Annahmerate), Spanne 4,0–8,8 ms; realistisch eher die Hälfte. Reihenfolge nach Aufwand/Risiko (leichtester zuerst): wo_a RTX (1) → Router (4) → mHC (6, nach ncu) → Köpfe (3, wenn die Annahmerate trägt) → Compress-GEMMs (5) → Attention-Fusion (2).

**Einordnung der Posten außerhalb der Rangliste:**
- Der strukturelle Befund der GLM-Analyse gilt hier unverändert: die Karten sind im Mittel zu 18,6 % belegt; die Kerne selbst liegen nahe an der Decke (68,9 % der Runde streamen Gewichte). Ein Hebel, der die Rundenzeit unter die Summe der Stufenzeiten bringt, wäre nur Parallelisierung (mehrere gleichzeitige Prüfschritte oder Anfragen), kein Kernumbau.
- Der DSpark-Entwurf ist von der Abnahme abhängig (Hidden States der verifizierten Positionen, `main_proj`); die Entwurfskette (5,4 ms) kann nicht hinter Kopf 1 verschoben werden, ohne alle sechs Annahmefälle vorab zu rechnen (Experten-Last × 2–3, bei gleicher Bandbreite) [abgeleitet, nicht empfohlen].
- Schichtverteilung: VRAM-begrenzt, Kosten je Schicht auf RTX und V100 ähnlich.

## 8. Welche Messungen die besten Hebel absichern

Keine GPU in dieser Analyse; alle folgenden Schritte brauchen freie Karten (GPU-Orc, Messfenster ankündigen).

1. **wo_a auf Turing (Rang 1), vor jedem Kern-Umbau:**
   - `ncu` auf `turing_fp16_s1688gemm … tn (8, 1, 8)` (Stufe 0/4, 1 Schicht): `dram__bytes_read.sum` soll ≈ 67 MB sein (belegt, dass das Gewicht als FP16 gelesen wird; wären es 33,5 MB, ist die Ursache eine andere und die Rechnung neu).
   - Mikrobenchmark im CUDA-Graph auf der RTX (M = 6, [8 × 1024 × 4096]): (a) heute (cuBLAS FP16 bmm), (b) 8 Starts des vorhandenen QPN8-Kerns mit N = 1024 (Erwartung: 8 × (7,6 + 4,2/0,622) = 113 µs, also schlechter), (c) ein QPN8-Kern mit Gitter-y = Gruppe (Erwartung 61,5 µs). Abnahme: ≤ 68 µs und Abweichung gegen die FP32-Referenz nicht größer als die des cuBLAS-Wegs.
2. **Attention-Kern (Rang 2):**
   - Kern → Python-Op zuordnen: ein einziger, eng begrenzter `torch.profiler`-Lauf (eine Schicht, ratio-4, eine RTX und eine V100) mit Stapelangabe (Speicher: Torch-Profiler nur eng begrenzt, Vollprofil → Swap-Sturm); danach lässt sich sagen, welche der 11–24 Kerne wegfallen können (Hilfsrechnungen, die nicht von der Schicht abhängen, einmal je Runde im Vorbereitungsdurchlauf).
   - Mikrobenchmark: die heutige Kette (KV-Einfügen, Rope, Dequant-Gather, QK, Softmax, PV, Inverse-Rope) als CUDA-Graph-Segment gegen eine Fassung ohne die Kleinkerne (Obergrenze der Fusion); Abnahme: ≥ 0,5 ms je Runde Ersparnis hochgerechnet, sonst Rang 2 streichen.
3. **Entwurfsköpfe (Rang 3):**
   - Abdeckung des Teilvokabulars aus bereits vorhandenen Ausgaben (AIfred-Texte DE/EN/Code, Tokenizer des Modells): Anteil der erzeugten Token innerhalb der 98 304 häufigsten (offline, ohne GPU); Ziel ≥ 99,5 %.
   - A/B (T P T P) der Annahmerate (Token je Runde) mit FP8-Kopf und mit Teilvokabular gegen heute auf den üblichen Texten; Abbruchregel: Token je Runde −2 % hebt eine Ersparnis von 1 ms auf.
   - Mikrobenchmark Markov-Schleife: Gemv mit Argmax-Epilog gegen `gemv2T` + 1-CTA-`reduce_kernel` (5 Schritte im Graph).
4. **Zusatz (vor Rang 6, kurz):** `ncu` auf `sm70_glm_mhc_pre_norm_kernel` (Gitter 6, 1, 1): Warp-Stall-Gründe, Latenzkette der Sinkhorn-Schleife; Messreihe mit 6 / 24 / 144 CTAs im Mikrobenchmark. Entscheidet, ob 0,0 oder 1,1 ms in Rang 6 stecken.
5. **Zusatz (Absicherung der Profilbasis):** Capture 2 wiederholen (S0-Vorbereitung bimodal 2,5/10 µs) und ein **Langkontext-Profil** (z. B. 128k und 250k Token) für die Indexer-Kette; in Produktion ohne nsys die Hostzeit der Runde messen, bevor Hostarbeit an Stufe 0 geplant wird.

## 9. Grenzen der Auswertung und offene Annahmen

- nsys misst unter Profil-Last: Kernzeiten sind genau, Hostlücken (Abschnitt 2) und die Stufe-0-Vorbereitung können verfälscht sein; der bimodale S0-Befund in Capture 2 ist nicht reproduziert.
- Die Zahl verschiedener Experten je Schicht und Runde steht nicht in der Datei; die MoE-Aussagen stützen sich auf die Hash-Schichten (Erwartungswert 33,9 bei Gleichverteilung der `tid2eid`-Tabelle, nicht geprüft) und auf Grenzraten des GLM-Mikrobenchmarks (RTX 592–597, V100-Ring 739–746 GB/s; der V100-Kern der DeepSeek-Läufe ist derselbe `skinny_nvfp4_moe_qpn` mit Ring seit v15 laut Auftrag). Fehlmessgröße wie bei GLM: ein Debug-Zähler der Expertenzahl.
- **wo_a-Bytes auf Turing (67 MB FP16)** sind aus Quelltext und Kernname abgeleitet, nicht gemessen (4.2); die V100-FP8-Bytes sind zwingend (sonst > Spitze).
- Die Zuordnung der kleinen GEMMs (Compress-Fenster: welches Gewicht welcher Kern) stützt sich auf Gittergrößen und Reihenfolge, nicht auf Namen; die Bytes des Compress-Fensters (21,5 bzw. 8,4 MB) sind die Summe der in der Schicht vorhandenen BF16-Gewichte, ob alle in diesem Fenster gelesen werden, ist [vermutet]. Der Indexer-wq_b (FP8) liegt nur bei langem Kontext im Fenster (Capture 2).
- Rollen sind per Zustandsautomat (mHC-pre_norm-Zählung, MoE-Zählung) und Kernnamen/Gitter klassifiziert; „Elementweise“ ist der Rest (alle namenlosen Torch-Kerne), „Attention-Hilfskerne“ die benannten Index-/Gather-/Rope-Kerne. Die Aufteilung zwischen beiden ist grob, die Summe (4,5 ms, 8,9 %) zählt.
- Die Tokenrate je Runde (3,26 / 3,53) stammt aus verschiedenen Prompts; Unterschiede der Rundenzeit zwischen den Captures mischen Kontext- und Inhaltseffekte (MoE +0,75 ms ist Inhalt, nicht Kontext).
- Hebelrechnungen sind Bandbreiten- bzw. Zielzeit-Überschläge; Ziele (12 µs Router, 25 µs Attention-Fenster, 1,5 × Bytezeit Compress, 5–11 µs mHC) sind Annahmen und im Skript änderbar.
- RTX-Spitze: Treiberwert 624 GB/s unter Last gegen nominell 672; Prozentangaben gegen 672 sind für die RTX 7,7 % zu niedrig, die Rangfolge der Hebel ändert das nicht, wohl aber die Aussage „wie weit vom Limit“.
