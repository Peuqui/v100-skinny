# GLM-5.3-Flash Decode: die kleinen GEMMs weit unter der Bandbreite (kv_b, Indexer wq_b, QPN8 klein), 23.11.2026

Nur Analyse: keine Änderung an Code, GPU, Git. Baum `1Cat-vLLM-v14` (Dateien gleich im Dev-Baum `-merge-v14`), Profil `~/.cache/nsys-runs/glm-v14k4-decode16k-1123/*.1.sqlite` (Capture 1, 37–38 Decode-Runden je Stufe), Vorlage `ANALYSE.md` in diesem Ordner. Hilfsskripte (nur Lesen der sqlite) lagen im Scratchpad, nichts im Projekt.

Kennzeichnung: **[gemessen]** = nsys-Datei oder Header, **[Code]** = Quelltext (Datei:Zeile), **[abgeleitet]** = Rechnung aus Messwerten, **[vermutet]** = Schluss ohne direkten Beleg. GB/s = Byte / µs / 1000. Spitze V100 900, RTX 672 GB/s; „85 %“-Decke = V100 765, RTX 571 GB/s. Zeiten in µs, weniger ist besser.

## 0. Kurzfassung

1. **kv_b ist keine GEMM, sondern zwei `torch.bmm` je DSA-Schicht** (W_UK-Absorption `q_nope @ W_UK` und Wertprojektion `attn_out @ W_UV`), cuBLAS, Gewicht FP16 (Checkpoint BF16, beim Laden nach FP16 gewandelt), 64 Köpfe × 256 × 512, je 16,78 MB, zusammen alle 33,55 MB je Runde. Die Zuordnung der Analyse stimmt.
2. **Korrekturen an ANALYSE.md:** (a) Im Decode läuft auf der RTX nie `fp8_qpn8_mma_sm75`, sondern auf beiden Kartenarten für alle M=5-Projektionen `fp8_qpn8_sm70_kernel` (der MMA-Kern gilt erst ab M=9). (b) Die fusionierte `q_a`+`kv_a`-Projektion (`fused_qkv_a_proj`) liegt nicht als FP8 vor, sondern als **FP16 (16,78 MB)** und kostet 34,5–36,2 µs je DSA-Schicht (V100 48–54 % der Spitze, RTX 72 %); sie fehlt in der Aufzählung der Analyse und ist ein vierter Posten derselben Klasse. (c) Die „19 µs Untergrenze“ ist keine Untergrenze, sondern der Achsenabschnitt einer Geraden: alle QPN8-Kerne mit 128 CTAs folgen **t = a + Byte/BW** mit a ≈ 9,7 µs (V100) bzw. 7,6 µs (RTX) und BW ≈ 852 GB/s (V100, 95 % der Spitze) bzw. 622 GB/s (RTX, 93 %). Die Kerne streamen also asymptotisch über der 85-%-Decke; verloren geht der feste Anteil a.
3. **Split-K-Feinabstimmung bringt hier nichts:** die Belegungsregel `fp8_qpn8_occupancy_split_k` ist für alle diese Formen inaktiv (Rechnung in 3.4), und die Formen liegen bereits auf der Geraden. Hebel sind andere: Scale-Vorladen im MXFP8-Kern (KDA o_proj), ein eigener FP16-Skinny-Kern für kv_b/wq_b/qkv_a, Fusion der Shared-Expert-Aktivierung.
4. **Gesamtes Potenzial je Runde (Rundenzeit 60,74 ms):** kv_b 0,28–0,42 ms, KDA o_proj 0,30–0,42 ms, Shared-Expert-Aktivierung 0,10–0,13 ms, qkv_a 0,10–0,16 ms, wq_b 0,04–0,12 ms; Summe etwa 0,8–1,3 ms = 1,3–2,1 % der Runde. Kein Posten allein erreicht die 1,5-%-Schwelle für generische Hebel; ein gemeinsamer FP16-Skinny-Kern deckt kv_b + wq_b + qkv_a (zusammen 0,4–0,7 ms) ab.
5. **Verlustbehaftet wäre:** kv_b oder wq_b nach FP8/MXFP8 (der Checkpoint hält sie in den Hauptschichten bewusst BF16; nur die MTP-Schicht 45 hat MXFP8-Fassungen). Das ist unten als verlustbehaftet gekennzeichnet und nicht empfohlen. `fused_qkv_a` nach FP8 ist dagegen nur eine andere Darstellung desselben Checkpoints (er liegt dort als FP8 vor), aber nicht bitgleich zum jetzigen Lauf; Freigabe nötig.

## 1. Zuordnung: Modul, Kern, Gewicht, Bytes

Modell: `Glm5NextForConditionalGeneration`, `--dtype half` (llama-swap-Eintrag `GLM-5.3-Flash-320B-A18B-NVFP4-DFlash2-vllm`, `config.yaml` Zeile 211) [Code]. Alle „BF16“-Tensoren des Checkpoints liegen daher als FP16 im Speicher, alle Aktivierungen FP16. M = 5 (Anker + 4 Entwurfstoken), CUDA-Graph mit Größe 5 (`cudagraph_capture_sizes [5]`, Config Zeile 219). Rechenstrom: genau ein Strom (nsys streamId 20), die Kerne laufen strikt nacheinander [gemessen]. Lücken zwischen Kernen im Graph sind 0–3 µs (z. B. V100 409,669+155,3 µs → nächster Start 409,825 ms) [gemessen].

Tabelle je DSA-Schicht (V100 = Stufen 1–3, RTX = Stufen 0 und 4). Zeiten Mittel über Runden, aus der sqlite (`n` = Aufrufe: V100 S1 74 = 2 Schichten × 37 Runden usw.).

| Posten | Modul (Datei:Zeile) | Gewicht im Speicher | N × K | Byte je Aufruf | Kern V100 [gemessen] | Kern RTX [gemessen] | µs V100 / RTX | GB/s (% Spitze) V100 / RTX |
|---|---|---|---|---|---|---|---|---|
| kv_b, Teil 1 (W_UK) | `MLAAttention.forward_impl`, `torch.bmm(mqa_q_nope, self.W_UK_T)`: `vllm/model_executor/layers/attention/mla_attention.py:758` | FP16, Sicht auf `kv_b_proj.weight` (Parameter aus `ColumnParallelLinear`, `models/glm5next/nvidia/attention.py:511`) | je Kopf 256 × 512, 64 Köpfe | 16,78 MB | `volta_fp16_s884gemm_fp16_64x64_ldg8_f2f_nn`, Gitter (8,1,64), 128 Threads, 106 Reg | `turing_fp16_s1688gemm_fp16_128x128_ldg8_f2f_stages_32x1_nn` (4,1,64), 234 Reg | 40,7 / 46,5 | 412 (45,8 %) / 361 (53,7 %) |
| kv_b, Teil 2 (W_UV) | `_v_up_proj`: `torch.bmm(x, self.W_UV, out=…)`: `mla_attention.py:986-1008` | FP16, andere Hälfte jedes Kopfblocks | je Kopf 512 × 256 | 16,78 MB | `…f2f_tn`, (4,1,64) | `…f2f_tn`, (2,1,64) | 45,5–46,2 / 46,3–46,6 | 364 (40,4 %) / 361 (53,7 %) |
| Indexer `wq_b` | `Indexer.forward`: `q, _ = self.wq_b(qr)`, `models/glm5next/nvidia/attention.py:343`; `ReplicatedLinear` ohne Quantisierung (`quant_config=None` der Attention, `models/glm5next/nvidia/model.py:426`), Definition `attention.py:271` | FP16 | 4096 × 1536 | 12,58 MB | cuBLAS `Kernel2`, Gitter (8,32,1), 32 Threads, 122 Reg | gleich | 30,8 / 29,8–30,1 | 408 (45,3 %) / 422 (62,8 %) |
| `fused_qkv_a_proj` (q_a + kv_a) | `DeepSeekV2FusedQkvAProjLinear`, `attention.py:478`; Kommentar `model.py:421-426`; Ladeweg `_try_load_fp8_attn_proj` dequantisiert FP8 → BF16: `model.py:1790-1858` | FP16 (Checkpoint FP8_PB_WO 128×128) | 2048 × 4096 (1536 + 512, `qk_rope_head_dim` = 0 im Checkpoint, also kein Pad) | 16,78 MB | `Kernel2` (8,8,4), 128 Threads, plus `splitKreduce_kernel` (64,1,1) 4,2 µs | `Kernel2` (8,16,1), 32 Threads, kein splitKreduce | 34,5–36,2 (+4,2) / 34,8–35,3 | 475 (52,8 %) / 482 (71,8 %) |
| Indexer-Rest (wk + weights_proj, fp32-`mm`, kpool-gate) | `attention.py:348`, `:357`, `:410` | FP16 / FP32 | klein (≤ 1,3 MB) | | `Kernel2` (4,2,10) 11,4 + (4,1,10) 9,2 + `volta_sgemm_32x32` 15,5, je + splitKreduce ~4 | `Kernel2` (8,2,13) 6,4, (8,1,16) 5,6, `volta_sgemm` 7,1 | je DSA-Schicht V100 ≈ 48, RTX ≈ 28 | latenzgebunden |

Bestätigung der Zuordnung der Analyse: kv_b = die beiden Batch-GEMMs mit z = 64 Köpfen (Gitter-z 64), Reihenfolge nn vor Attention, tn nach Attention und direkt vor dem o_proj-QPN8 (V100 411,225 ms → 411,283 ms) [gemessen]. wq_b = `Kernel2` (8,32,1): N = 4096 = 32 Kacheln × 128 [abgeleitet], Byte-Rechnung 12,58 MB × 1/30,8 µs passt [abgeleitet].

### QPN8-Projektionen (M = 5)

Alle laufen im Decode als `fp8_qpn8_sm70_kernel` auf **beiden** Kartenarten: Dispatcher `fp8_qpn8_gemm_sm70_out` für M ≤ 8 (`fp8_block.py:146-181`, `_QPN8_MAX_M = 8` Zeile 53, `_SM75_MMA_MAX_M = 512` Zeile 60; MXFP8-Weg `pre_ampere.py:207-230`), Kern `csrc/sm70_turbomind/ops/fp8_qpn8_sm70.cu:215-483`, Start `launch_fp8_qpn8_sm70`: Gitter `n/32` CTAs, `32·SplitK` Threads (:486-495). Kein `fp8_qpn8_mma_sm75` im gesamten Capture [gemessen, Name taucht in keiner Karte auf].

| Posten | Modul | Format | N × K | Gitter / Threads | SplitK | Byte | µs V100 / RTX | % Spitze V100 / RTX |
|---|---|---|---|---|---|---|---|---|
| KDA o_proj | `Qpn8Mxfp8LinearKernel` (`pre_ampere.py:171-196`), `prepare_mxfp8_qpn8_linear` (`sm70_turbomind.py:~790-815`), Zeilengruppen-Skalen, Kern-Zweig `RowGroupScales` (`fp8_qpn8_sm70.cu:327-335`, Start :1633-1656) | MXFP8, Skalen FP16 [K/32, N] | 4096 × 8192 | 128 / 512 | 16 (`fp8_qpn8_launch_config`, `sm70_turbomind.py:700-710`) | 33,55 Codes + 2,10 Skalen = **35,65 MB** | 57,9 / 76,1 (S0), 75,1 (S4) | 68 (V100: 616 GB/s) / 70 |
| Shared Expert gate+up | `Glm5NextMLP.gate_up_proj` (`model.py:189-246`), `QPN8Fp8BlockScaledMMLinearKernel` (`fp8_block.py:561`, Aufbereitung :669-766) | FP8_PB_WO 128×128 | 4096 × 4096 | 128 / 512 | 16 (`fp8.py:242-250`: `k % 256 == 0`, `k > 2048`, `n < 12288`) | 16,78 + ~0,02 | 29,2–29,7 / 34,6 (S0), 33,9 (S4) | 63–64 / 72–74 |
| Shared Expert down | `down_proj`, gleicher Kern | FP8_PB_WO | 4096 × 2048 | 128 / 256 | 8 (`k <= 2048`) | 8,39 | 18,8 / 19,3 (S0), 17,6 (S4) | 49–50 / 65–71 |

Zählung je Runde [Code + gemessen]: DSA-Schichten V100 2/2/1 = 5, RTX 3/3 = 6 (Schichten 3,7,…,43; Partition 14,7,7,7,10); KDA-Schichten V100 5/5/6 = 16, RTX 11/7 = 18; Sparse-Schichten mit Shared Expert V100 7/7/7 = 21, RTX 11/10 = 21 (S0 hat Schichten 0–2 dicht). Die Zählungen gegen die sqlite: kv_b-Aufrufe V100 74/75/38 = 2·37/2·37/1·37, RTX 111/114 = 3·37/3·38.

## 2. Frage 2: Warum kv_b BF16, und was wird wirklich gerechnet?

**Checkpoint [gemessen]:** Die Hauptschichten haben `kv_b_proj.weight` `[32768, 512]` BF16 (= 64 Köpfe × (256 nope + 256 v) × 512), ebenso `indexer.wq_b.weight` `[4096, 1536]` BF16 und `indexer.wk`/`weights_proj` BF16, alle in `tensors/native-QB.safetensors` (Index `model.safetensors.index.json`; das ist die Datei, die der Lader früher übersah, STAND.md Zeile 1211). In `hf_quant_config.json` steht für die Hauptschichten **kein** Eintrag für kv_b oder wq_b (nur q_a, kv_a, q_b, o_proj als `FP8_PB_WO`, Experten NVFP4, Shared Expert FP8_PB_WO); die **MTP-Schicht 45** hat kv_b, wq_b, wk, weights_proj dagegen als `MXFP8` (`F8_E4M3` + U8-Skalen `[32768,16]`). Warum die Hauptschichten sie BF16 lassen, steht weder in der README („Decoder attention: MXFP8 projection selection from the pinned source“, Zeile 123) noch in der Config [unbekannt]; die Auswahl liegt also bei NVIDIAs Quantisierung, nicht bei uns.

**Lader [Code]:** Es gibt keine Dequantisierung zur Ladezeit für kv_b. `MLAAttention.process_weights_after_loading` (`mla_attention.py:842-930`) ruft `get_and_maybe_dequant_weights(self.kv_b_proj, out_dtype=act_dtype)` (`quant_utils.py:388-407`); bei unquantisierter Schicht ist das `weight.to(out_dtype)`, bei gleichem Datentyp (FP16 = FP16) dieselbe Speicherfläche. Danach `W_UK, W_UV = kv_b_proj_weight.split([256, 256], dim=-1)` und `self.W_UV = W_UV.transpose(0,1)` (:927), `self.W_UK_T = W_UK.permute(1,2,0)` (:929): beides **Sichten** auf `kv_b_proj.weight`, kein zweites Exemplar [abgeleitet aus dem Code, Speicher nicht gemessen]. Kopf h belegt die Zeilen `h·512 … h·512+511` des Gewichts: erst 256 Zeilen W_UK (256 KB zusammenhängend), dann 256 Zeilen W_UV (256 KB). Beide Hälften liegen je Kopf zusammenhängend, die Schrittweite von Kopf zu Kopf ist 512 KB; das Layout ist für Streaming unproblematisch [abgeleitet].

**Decode-Rechnung [Code]:** Alle Sparse-MLA-Implementierungen laufen nur über `forward_mqa` (`mla_attention.py:678-689`: `is_sparse_impl` → `num_mqa_tokens = q.size(0)`). Je Runde und DSA-Schicht:
- `mqa_ql_nope = bmm(q_nope[64,5,256], W_UK_T[64,256,512])` (:758) → `[64,5,512]`, danach Sparse-Attention im Latentraum (Rang 512);
- `attn_out[5,64,512]` → `_v_up_proj`: `bmm(x[64,5,512], W_UV[64,512,256])` (:1008) → `[5,64,256]`, dann `o_proj` (QPN8).
Das sind **zwei Batch-Matmuls pro Schicht, nicht eine kv_b-GEMM**; beide lesen ihre Gewichtshälfte vollständig (kein Pruning über Köpfe oder Experten möglich): 2 × 64 × 256 × 512 × 2 B = 33,55 MB je Runde und DSA-Schicht [abgeleitet, stimmt mit der Header-Größe]. Die Rechenmenge (84 M FMA je bmm bei M = 5) ist gegenüber dem Speicherstrom klein, aber auf CUDA-Kernen allein nicht vernachlässigbar (siehe 3.1). Tokenstrom und KV-Cache-Aufbau (`_sm70_glm5_fp8_kv_insert`, `gather_dequant` 26–28 µs, QK-Kern `…tn (34,1,5)` 37–39 µs, 5 × PV-GEMM je ~12 µs + `splitKreduce` ~4,7 µs) sind eigene Posten und hier nicht Gegenstand.

**Warum cuBLAS hier langsam ist [abgeleitet, Mechanismus vermutet]:** Das Profil zeigt für M = 5 Standardkerne mit Allzweck-Kachel: V100 64×64-Kachel, 106 Register, 4 CTAs/SM möglich (65 536 / (106·128)); Gitter (8,1,64) = 512 CTAs → 6,4 CTAs je SM, also zwei Wellen mit schlecht gefüllter zweiter. RTX 128×128-Kachel, 234 Register, 32 KB Shared, 2 CTAs/SM (8 Warps je SM); Gitter (2,1,64) = nur **128 CTAs** für W_UV (1,8 CTAs/SM), (4,1,64) = 256 für W_UK. Turing hat kein `cp.async`; „ldg8“-Kerne puffern über Register, so dass je SM nur wenige Kilobyte in Flug sind. Beide Befunde (zu wenig parallele Last in Flug, Wellen-/Verteilungsverlust) erklären 40–54 % der Spitze plausibel, sind aber nicht durch einen Kern-Counter belegt [vermutet].

## 3. Frage 3: Was begrenzt jeden Posten, und welche Änderung hilft

### 3.0 Die Gerade t = a + Byte/BW (neu, [abgeleitet] aus gemessenen Zeiten)

QPN8-Kerne mit 128 CTAs (N = 4096), V100 gemessen: Shared down 18,8 µs (8,39 MB), Shared gate+up 29,4 (16,78 MB), DSA o_proj 88,5 (67,13 MB). Aus den beiden großen: Steigung 1,174 µs/MB = **852 GB/s (94,7 % der Spitze)**, Achsenabschnitt **a = 9,7 µs**; der kleine Kern wird damit zu 9,7 + 9,85 = 19,5 µs vorhergesagt, gemessen 18,8 µs. RTX: Shared gate+up 34,6, DSA o_proj 115,5 (S0) → 1,607 µs/MB = **622 GB/s (92,6 %)**, **a = 7,6 µs**; Shared down vorhergesagt 21,1, gemessen 19,3/17,6.

Folgerung: Der Kern streamt asymptotisch deutlich über der 85-%-Decke; die „Decke“ 84–87 % der großen Kerne ist selbst schon vom festen Anteil a gedrückt (DSA o_proj: a/t = 11 %). Kleine Kerne verlieren fast nur a. Das erklärt, warum Shared down (8,4 MB) 49–65 %, Shared gate+up 63–74 % erreicht. Was a ist, ist **nicht belegt** [vermutet]: Anlauf (CTA-Start, erste Scale-/Code-Ladungen, Latenz ~1 µs), Auslauf (Shared-Memory-Reduktion über 8–16 Warps, Ausgabe), Wellenverlust durch ungleiche CTA-Verteilung (128 CTAs auf 80/72 SMs = 1 oder 2 je SM), kein PDL auf Volta/Turing, so dass der nächste Kern erst nach dem Ende des vorigen startet. Eine Messung entscheidet: ein Kern-Sweep mit festem Gitter (128 CTAs) und K = 512…16384 auf beiden Karten (der Achsenabschnitt ist dann direkt der Leerlauf) und ein Lauf desselben Kerns mit 1 gegen 2 CTAs je SM (Gitter 80 gegen 160) trennt Wellenverlust von Anlauf.

### 3.1 kv_b (zwei Batch-bmm)

- **Begrenzung** [abgeleitet]: Allzweck-Kachel und zu wenig Last in Flug (2.); der feste Anteil je Kern liegt mit etwa 22–28 µs (Rest nach Abzug der Bytezeit bei 850 GB/s: 40,7 − 19,7 = 21; 46 − 19,7 = 26) weit über dem QPN8-Achsenabschnitt von 9,7.
- **Änderung (verlustfrei):** eigener FP16-Skinny-bmm-Kern (Gewicht FP16 direkt gelesen, 16-Byte-Ladungen, mehrere Ladungen je Thread in Flug, Tensorkern-Befehl `m8n8k4` wie in `fp8_qpn8_sm70.cu` über `VLLM_SM70_MMA_8N8K4`, FP32-Akkumulation). Je Kopf ein eigener Gewichtsblock; ein CTA je (Kopf, Spaltenkachel) liefert 64–256 CTAs. Reine CUDA-Kern-FMA reicht rechnerisch knapp nicht bequem: 84 M FMA je bmm bei 7,2 TFLOP/s FP32 (V100) = 11,6 µs gegen 22 µs Byte-Zeit; Tensorkerne gehören deshalb dazu [abgeleitet]. Wegen anderer Akkumulationsreihenfolge als cuBLAS nicht bitgleich (Tokenfolge ändert sich, kein Qualitätsverlust): **Prüfstufe 1** (A/B).
- **Erwartete Ersparnis je Runde** [abgeleitet]:
  - Decke (beide bmm bei 85 %): V100 33,55 MB / 765 GB/s = 43,9 µs gegen 86,9 µs gemessen → 43,0 µs je Schicht × 5 = **215 µs**; RTX 58,8 gegen 93,1 → 34,3 × 6 = **206 µs**; zusammen 0,42 ms.
  - Belastbare Schätzung (zwei Kerne auf der QPN8-Geraden: 2 × (a + 16,78 MB·Steigung)): V100 2 × 29,4 = 58,8 µs → 28,1 × 5 = **140 µs**; RTX 2 × 34,6 = 69,2 µs → 23,9 × 6 = **143 µs**; zusammen **0,28 ms** (0,47 % der Runde).
- **Verlustbehaftet, nicht empfohlen:** kv_b als FP8/MXFP8 (halbe Bytes): Zeit nach Gerade V100 2 × (9,7 + 9,85) = 39 µs (Ersparnis 48 µs/Schicht), RTX 2 × (7,6 + 13,5) = 42 µs (51 µs/Schicht), zusammen ≈ 0,55 ms. Der Checkpoint hat die Hauptschichten bewusst BF16 gelassen; BF16 → FP8 verwirft Mantissenbits (8 → 3): **verlustbehaftet**, gegen die Regel „Qualität vor Tempo“, keine Abnahme ohne Qualitätsvergleich.

### 3.2 Indexer `wq_b` (4096 × 1536 FP16, 12,58 MB)

- **Begrenzung** [abgeleitet]: cuBLAS wählt einen `Kernel2` mit **Gitter (8,32,1) aus 32-Thread-CTAs** = 256 Warps im Ganzen, also 3,2 Warps je SM auf der V100: zu wenig Parallelität, um Latenz zu verdecken (30,8 µs = 408 GB/s, 45 %). Auf der RTX gleicher Kern, 29,8 µs (63 %). [Mechanismus vermutet.]
- **Änderung:** derselbe FP16-Skinny-Kern wie für kv_b (ein Gewicht `[N,K]` zeilenweise gelesen, K-Teilung über Warps; Form 4096 × 1536 passt auf 128 Spalten-CTAs).
- **Erwartete Ersparnis je Runde:** Decke 85 %: V100 12,58/765 = 16,4 µs → 14,5 µs × 5 = **72 µs**; RTX 22,0 µs → 7,8 × 6 = **47 µs**; zusammen 0,12 ms. Nach der Geraden (a + Byte/BW): V100 24,5 µs → 6,4 × 5 = **32 µs**, RTX 27,8 µs → 2,0 × 6 = **12 µs**; zusammen **0,04 ms**. Auf der RTX bleibt wenig übrig [abgeleitet].
- Verlustbehaftet wäre MXFP8 für wq_b (MTP-Schicht hat eine Fassung): Bytes halb, Ersparnis ≈ 10 µs/Schicht V100; nicht empfohlen, wie kv_b.

### 3.3 `fused_qkv_a_proj` (Nachtrag, in der Analyse nicht benannt)

- **Befund** [gemessen/Code]: `fused_qkv_a_proj` ist absichtlich FP16/„BF16“ (Kommentar `model.py:421-426`, dessen Begründung „2624 Zeilen neben dem 128-Raster“ für dieses NoPE-Modell nicht mehr zutrifft: 1536 + 512 = 2048, `qk_rope_head_dim` = 0). Der Lader dequantisiert den FP8-Checkpoint beim Laden (`_try_load_fp8_attn_proj`, `model.py:1790-1858`). Kosten: V100 34,5–36,2 µs (+ `splitKreduce` 4,2 µs, also bis 40,4), RTX 34,8–35,3 µs.
- **Änderung A (verlustfrei, Kern):** derselbe FP16-Skinny-Kern. Decke 85 %: V100 21,9 µs → 13,3 × 5 = **66 µs**, RTX 29,4 → 5,4 × 6 = **32 µs**; nach Gerade V100 → 29,2 µs (+ splitKreduce entfällt) → 6–11 µs × 5 ≈ **30–55 µs**, RTX ≈ 0.
- **Änderung B (Darstellung, nicht bitgleich):** `fused_qkv_a_proj` als FP8_PB_WO-Block-QPN8 führen (Checkpoint ist FP8, N = 2048, K = 4096 sind 128-teilbar, vorhandener Kern `QPN8Fp8BlockScaledMMLinearKernel`; die gemergte q_a/kv_a-Ladung braucht Skalen-Mischung, `DeepSeekV2FusedQkvAProjLinear` ist eine `MergedColumnParallelLinear`). 8,39 MB statt 16,78: nach Gerade V100 9,7 + 9,85 = 19,5 µs (35,4 → 19,5 = 15,9 × 5 = **80 µs**), RTX 7,6 + 13,5 = 21,1 µs (35 → 21,1 = 13,9 × 6 = **82 µs**), zusammen 0,16 ms. Das ist keine Quantisierung im Verlustsinn (der Checkpoint liegt als FP8 vor; die jetzige Rechnung rundet Wert × Skala einmal nach FP16, QPN8 rundet die Skala nach FP16: relativer Unterschied ≤ 2^-11), aber **nicht bitgleich** und eine Änderung der Gewichtsdarstellung: Freigabe nötig, Prüfstufe 1.

### 3.4 QPN8 klein: KDA o_proj, Shared gate+up, Shared down

**Split-K-Belegungsregel (wie angekündigt gefunden):** `fp8_qpn8_occupancy_split_k` in `vllm/model_executor/layers/quantization/sm70_turbomind.py:719-748`; sie senkt Split-K 16 → 8 nur, wenn `ctas_per_sm_at[16] <= 2` (Turing), `0 < spill < sms/2` und `ctas <= sms·ctas_per_sm_at[8]`. Für die hier betrachteten Formen [abgeleitet aus Code und Kartendaten]:
- N = 4096 → `ctas = 128`. V100 (80 SMs, 2048 Threads/SM): `ctas_per_sm_at = {8: 8, 16: 4}`, `spill = 128 − 80·4 = −192` → Regel inaktiv. RTX (72 SMs, 1024 Threads/SM): `{8: 4, 16: 2}`, `spill = 128 − 144 = −16` → inaktiv (kein Überlauf in eine zweite Welle: alle 128 CTAs laufen gleichzeitig).
- q_b (N = 16384, K = 1536, 512 CTAs): Standard bereits Split-K 8 (`fp8.py:246`, `k % 256 != 0`); Regel: V100 `ctas_per_sm_at[16] = 4 > 2`, RTX `spill = 368 ≥ sms/2 = 36` → inaktiv.
Es gibt in diesen Formen also keinen Wellenüberlauf, den die Regel beheben könnte. Der Kommentar `sm70_turbomind.py:722-731` nennt außerdem Split-K 32 für Gitter ≤ SMs als zwar isoliert schneller, auf DeepSeek-V4 aber ohne Schrittgewinn und mit niedrigerer Annahmerate (also andere Reduktionsfolge, nicht „falsch“); 128 CTAs > 80 SMs ließe diese Regel ohnehin nicht zu.

**KDA o_proj (MXFP8, 128 CTAs):** auffällig schlechter als die Gerade [abgeleitet]. Nach der Block-Geraden wäre V100 9,7 + 35,65·1,174 = 51,6 µs (gemessen 57,9: +6,3), RTX 7,6 + 35,65·1,607 = 64,9 µs (gemessen 76,1: +11,2). Der Kern für Zeilengruppen-Skalen (`RowGroupScales`) lädt je zwei K-Gruppen einen Skalenwert in einer **bedingten, abhängigen Ladung mitten in der Schleife** (`fp8_qpn8_sm70.cu:327-335`: `if (scale_group != loaded_scale_group) loaded_scale = __ldg(...)`), die der Compiler nicht über Iterationen heben kann; bei nur 128 CTAs (16–32 Warps je SM) fehlt die Warp-Zahl, um das zu überdecken, während der große MXFP8-Kern (qkv, 768 CTAs) fast auf der Geraden liegt (RTX 187,9 gegen Gerade 7,6 + 110,3 MB · 1,607 µs/MB = 184,9; V100 145,6 gegen 9,7 + 110,3 · 1,174 = 139,2) [Mechanismus vermutet].
- **Änderung (verlustfrei, bitgleich erwartet):** die Skalen der ganzen K-Scheibe eines Warps vorab laden (je Warp `K/SplitK/32` = 16 Werte bei K = 8192, in Registern oder Shared Memory) bzw. um zwei Iterationen vorholen; die Rechenreihenfolge bleibt, nur die Ladereihenfolge ändert sich. Prüfstufe 0/1.
- **Ersparnis je Runde:** Decke 85 % (35,65 MB): V100 46,6 µs → 11,3 × 16 = **181 µs**; RTX 62,4 µs → 13,7 (S0) × 11 + 12,7 (S4) × 7 = **240 µs**; zusammen 0,42 ms. Nach Gerade: V100 6,3 × 16 = **101 µs**, RTX 11,2 × 11 + 10,2 × 7 = **195 µs**; zusammen **0,30 ms**. Zusatz [abgeleitet]: derselbe Eingriff trifft die MXFP8-qkv (V100 16 Aufrufe × ~6 µs = ~0,1 ms), nicht auf der RTX.

**Shared gate+up und down (Block-FP8):** liegen **auf** der Geraden; Tuning von Split-K oder Gitter holt dort nichts, die Decke 85 % wäre nur über Wegfall von a erreichbar. Rechnung der Decke (nicht als Ziel): gate+up V100 16,78/765 = 21,9 µs → 7,5 × 21; RTX 29,4 → 5,2 × 11 + 4,5 × 10; down V100 11,0 → 7,8 × 21; RTX 14,7 → 4,6 × 11 + 2,9 × 10; zusammen Decke 0,42 ms. Nach Gerade: 0.
Konkrete Änderungen, die den festen Anteil treffen:
- **Aktivierung fusionieren** (verlustfrei): zwischen gate+up und down läuft ein eigener Elementweise-Kern `triton_poi_fused_clamp_mul_silu_slice_0` (V100 3,7–3,8 µs × 21, RTX 2,1/2,3 µs × 21) [gemessen], weil `Glm5NextMLP` den Clamp (`swiglu_limit` 10,0) selbst über `SiluAndMulWithClamp` rechnet (`model.py:228-237`) und der fusionierte `fp8_qpn8_gated_pair_sm70`-Kern (`fp8_block.py:756-765`, `fp8_qpn8_sm70.cu:852`) weder genutzt wird noch den Clamp kann. Ersparnis V100 21 × 3,77 = 79 µs, RTX 11 × 2,1 + 10 × 2,3 = 46 µs; abzüglich Epilogkosten etwa **0,10–0,13 ms**.
- **Shared Expert nebenläufig zum Router und den Routed Experts** (Zweig im CUDA-Graph): Der Shared Expert hängt nur an `hidden_states`, das Routing und die Experten ebenso; heute laufen alle auf einem Strom (nsys streamId 20) nacheinander [gemessen]. Nur der feste Anteil (2 × a + Aktivierung: V100 ≈ 23 µs, RTX ≈ 18 µs je Schicht) ließe sich verstecken, die Bytezeit (V100 29,6 µs) bleibt; Obergrenze V100 21 × 23 + RTX 21 × 18 = **0,86 ms**, tatsächlich unsicher (der Skinny-MoE-Kern füllt die Karte mit 2560 CTAs, Überlappung nur in seinem Auslauf) [vermutet]. Hoher Aufwand und Risiko (Graph-Topologie, Speicherpools, Streamsynchronisation); erst nach Messung des Achsenabschnitts (3.0) entscheiden.
- **Gate+up+down in einem Kern** (ein Start statt zwei, kein zweites a): benötigt Synchronisation über alle CTAs (down braucht alle 2048 Zwischenwerte); nur kooperativ oder mit zweiter Welle, wäre ein neuer Kernentwurf; keine belegte Ersparnis, nur Obergrenze a ≈ 9,7 µs V100 / 7,6 µs RTX je Schicht = 0,20 + 0,16 ms [abgeleitet].

### 3.5 Die „19-µs-Untergrenze“ der Analyse

Die kleinsten gemessenen Kerne: RTX `Kernel2` (8,1,16) 5,4–5,6 µs (1 MB), V100 `Kernel2` (4,1,10) 9,2 µs (1 MB) plus `splitKreduce` ~4 µs [gemessen]; QPN8 kleinster V100 17,8 µs (Shared down). Der Wert 19 µs ergibt sich aus a + Bytezeit (3.0), nicht aus einer Hardwaregrenze; ein Start im Graph kostet 0–3 µs Lücke [gemessen]. Was a ist (Anlauf, Auslauf, Welle), steht nicht fest [vermutet].

## 4. Rangfolge

Ersparnis je Runde in ms (Rundenzeit 60,74 ms); „belastbar“ = auf der gemessenen Geraden, „Decke“ = 85 % der Spitze. Aufwand und Risiko nur als Rang, keine Zeiten.

| Rang nach Ersparnis | Posten | belastbar | Decke | Aufwand | Risiko / Prüfstufe |
|---|---|---|---|---|---|
| 1 | kv_b: zwei bmm → FP16-Skinny-Kern | 0,28 | 0,42 | hoch (neuer Kern, Tensorkern-Layout, Einbindung in `mla_attention.py` oder eigener Operator, CUDA-Graph, beide Karten) | mittel: Reduktionsfolge anders als cuBLAS, Tokenfolge ändert sich; verlustfrei; Prüfstufe 1 |
| 2 | KDA o_proj (und MXFP8-qkv V100): Skalen vorab laden | 0,30 (+ bis 0,1 V100-qkv) | 0,42 | gering bis mittel (Eingriff in vorhandene Kern-Vorlage `RowGroupScales`) | gering: Ladereihenfolge, Rechenreihenfolge bleibt, bitgleich erwartet; Prüfstufe 0/1 |
| 3 | Shared-Expert-Aktivierung (Clamp+Silu) in den gate+up-Kern | 0,10 | 0,13 | mittel (Clamp im Gated-Pair-Kern, Modellcode `Glm5NextMLP` auf fusionierten Weg) | gering bis mittel: Clamp-Semantik exakt halten (Test gegen `SiluAndMulWithClamp`); Prüfstufe 1 |
| 4 | `fused_qkv_a_proj`: Kern A (FP16-Skinny-Kern) oder B (FP8-QPN8) | A 0,03–0,06 / B 0,16 | A 0,10 | A: kein Zusatz, wenn Kern aus Rang 1 vorhanden; B: mittel (gemergte Block-Skalen laden, Loader `model.py:1790-1858` ändern) | A: wie Rang 1; B: nicht bitgleich, Freigabe nötig |
| 5 | wq_b: FP16-Skinny-Kern | 0,04 | 0,12 | gering, wenn Kern aus Rang 1 vorhanden | wie Rang 1 |
| 6 | Shared Expert nebenläufig (Graph-Zweig) | unbekannt | bis 0,86 (Obergrenze) | hoch | hoch (Strom, Pools, Reihenfolge); erst nach Messung von a |
| – | Indexer-Kleinstteile (kpool-gate + wk/weights_proj zu einem GEMM zusammenlegen; fp32-mm bleibt) | V100 ≈ 11 µs × 5 + RTX ≈ 6 µs × 6 = 0,09 | | mittel | Reduktionsfolge/FP32-Gate: fp32-Anteil muss getrennt bleiben (`attention.py:350-357`, Nahtie-Ranking) |
| – | Shared gate+up / down: Gitter/Split-K | 0 | 0,42 | | bringt nach der Geraden nichts |
| – | kv_b/wq_b nach FP8 | 0,55 (kv_b) | | | **verlustbehaftet, nicht empfohlen** |

Rang nach Aufwand/Risiko (leichtester zuerst): Skalen vorab laden (2) → Aktivierung fusionieren (3) → FP16-Skinny-Kern (1, 4A, 5 teilen sich ihn) → qkv_a nach FP8 (4B, Freigabe) → nebenläufiger Shared Expert (6).

Zusammen: belastbar ≈ 0,28 + 0,30 + 0,10 + 0,06 + 0,04 = **0,8 ms**, Decke ≈ 1,3 ms, das sind **1,3 bis 2,1 %** der Rundenzeit (≈ +1,3 bis +2,2 % tok/s bei gleicher Annahmerate [abgeleitet]). Nach der Regel „belegt und billig“ gehört Rang 2 und 3 rein; Rang 1 zählt als generischer Hebel (der FP16-Skinny-Kern nützt auch DeepSeek-V4 und Qwen) und liegt mit 0,47 % unter der 1,5-%-Schwelle der Einzelsuche, also nur als Teil des Bündels.

## 5. Was offen bleibt und wie es zu klären ist

1. **Was ist der feste Anteil a?** Messung auf der Karte: QPN8-Kern mit festem Gitter (128 CTAs, N = 4096) und K = 512 … 16384 sowie 80/160 CTAs; Kern-Counter (`sm__warps_active`, `smsp__warp_issue_stalled_long_scoreboard`, DRAM-Durchsatz-Zeitreihe) für Shared down. Erst danach entscheidet sich, ob Fusion (3.4) oder nebenläufiger Shared Expert (Rang 6) lohnt. Nicht von mir ausführbar (keine GPU).
2. **Kern-Mikrobench für kv_b:** FP16-Skinny-bmm (64 × 5×256 @ 256×512 und 64 × 5×512 @ 512×256) gegen cuBLAS, im CUDA-Graph, auf V100 und RTX; erreicht der Entwurf die Gerade a + Byte/BW (29,4 / 34,6 µs je Kern)?
3. **Speicher:** bestätigen, dass `W_UK_T`/`W_UV` wirklich Sichten sind (Speicherverbrauch kv_b je Schicht 33,55 MB, nicht 67 MB) – aus dem Code abgeleitet, nicht gemessen.
4. **Zählung `splitKreduce`:** je DSA-Schicht V100 7–8 Aufrufe à ~4 µs (QK, 5 × PV, cuBLAS-Zerlegung) = ~30 µs, nicht Gegenstand dieser Frage, aber derselbe Mechanismus (cuBLAS-Split-K für M = 5): eigener Posten für eine spätere Runde.
5. **Wiederholbarkeit der Zahlen:** Mittelwerte aus Capture 1 (kurzer Kontext); in Capture 2 (16k) ändern sich diese Kerne nicht (Gewichtsstrom ist kontextunabhängig), nicht einzeln nachgerechnet.

## Nachtrag 24.11.2026: QPN8-Zeilengruppen-Skalen vorladen (KDA o_proj) verworfen

Zweig `qpn8-rowgroup-scale-preload` ea95771f7 (Baum `1Cat-vLLM-qpn8-scales`), Mikrobenchmark `pkg_qpn8_scales_bench_2026-11-23.py`,
Kette `chain_qpn8sc_2026-11-24.sh`, Logs `~/.cache/bench-scripts/qpn8sc_2026-11-24/`, alt = Baum v14 (gleicher QPN8-Kern wie v15), je
zweimal im Wechsel. Bitgleich (0 von 32 Formen abweichend, beide Karten), aber langsamer, µs je Aufruf bei M = 5 (weniger ist besser):
V100 o_proj 4096×8192 47,3 → 47,6–47,9, 4096×4096 25,3 → 25,8, 2048×6144 21,9 → 22,7, 8192×4096 48,9 → 49,6; RTX 59,4 → 60,0,
30,9 → 31,3, 23,8 → 24,3, 58,9 → 59,2. Also +0,5 bis +3,6 %. Verworfen.

## Nachtrag 24.11.2026: kv_b als schmaler Triton-Kern übernommen (Entwicklungszweig)

Zweig `glm-kvb-skinny` 48a92aa11 (im Entwicklungszweig `merge-1cat-2d218ab03`): ein Programm je Kopf und Spaltenstreifen, `tl.dot` mit
FP32-Akkumulation, Kacheln je Architektur aus `benchmark_mla_kv_b_skinny.py --sweep --rows 1 5 7` (V100 64×32/4 und 64×64/2, RTX 128×32/4
und 64×64/2), Schalter `kernel_config.mla_kv_b_skinny` (nicht im Compile-Hash), über 8 Zeilen bleibt `torch.bmm`. Mikrobenchmark bei 5 Zeilen,
µs je Aufruf (weniger ist besser): V100 W_UK_T 34,1 → 26,1, W_UV 34,3 → 24,2; RTX 41,3 → 31,9, 40,4 → 32,1. Tests 33/33 je Karte.
A/B GLM `ab_kvb_glm_2026-11-24a.log` (T P T P): Runde greedy 53,4/53,5 gegen 53,7/54,1 ms, AIfred-Sampling +0,5 %, GSM8K-Decode +0,5 %, 29;
erstes Token 150k–516k gleich, Langkontext je Schritt gleich, Prefill 8,7k 3,87–3,88 gegen 3,85–3,86 s (Streuung), Texte 8/8 tokengleich.
Fährt mit dem nächsten Produktionsbündel mit (Eintrag: `"mla_kv_b_skinny":true` im `--kernel-config`).
