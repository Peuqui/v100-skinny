# Leistungshistorie der Produktionsmodelle auf dem Mini (Stand 02.10.2026, 23:48)

Zusammengestellt am 02.10. abends aus STAND, Journal, Memory und Messlogs (nur gelesen, nichts neu gemessen); Abschnitt 8 ergänzt am 03.10. die Abschlussmessung und die Produktionsumstellung. Jede Zahl hat Datum, Bedingungen und Datei:Zeile. Richtung: Zeiten (s, ms, min:s) – weniger ist besser; tok/s und Annahme – mehr ist besser. Zeiten über 60 s stehen als min:s.

**Quellen (Kürzel → absoluter Pfad)**
- STAND = /home/mp/Projekte/vllm-research/v100-skinny/STAND.md
- FNOP = /home/mp/Projekte/vllm-research/v100-skinny/FLASH-NEXT-OPERATING-POINT.md
- PLEK = /home/mp/Projekte/vllm-research/v100-skinny/docs/PLE-KASKADE-ENTWURF.md
- J/ = /home/mp/Projekte/vllm-research/v100-skinny/docs/journal/
  - TUR = TURING-COEXISTENCE-HANDOVER.md
  - DSH = DEEPSEEK-VLLM-HANDOVER.md
  - QWH = QWEN4EXP-PORT-HANDOVER.md
  - SLH = SPEC-LONGCTX-HUNT.md
- HO/ = /home/mp/Projekte/vllm-research/v100-skinny/handover/
- UC/ = /home/mp/Projekte/vllm-research/v100-skinny/upstream-contrib/03-1cat-issues/
- RES = /home/mp/Projekte/vllm-research/vllm-bench/RESULTS.md
- BS/ = /home/mp/.cache/bench-scripts/
- Q/ = /home/mp/.cache/bench-scripts/quality_2026-09-27/
- clean = /home/mp/.cache/bench-scripts/ab_next_clean_2026-10-02.log
- CFG = /home/mp/.config/llama-swap/config.yaml
- MEM/ = /home/mp/.claude/projects/-home-mp-Projekte-AIfred-Intelligence/memory/
  - dspark = project_dsv4_dspark_tuning_2026-09.md
  - dio = project_safetensors_direct_io.md
  - m26 = project_1cat_merge_2026-09-26.md
  - m30 = project_1cat_merge_2026-09-30.md
  - ho14 = project_session_handover_2026-09-14.md
  - turfix = project_three_turing_fixes_measured.md
  - drafter = project_drafter_skip_checkpoint_weight.md
  - memhigh = reference_llamaswap_memoryhigh.md
  - pcache = project_dsv4_prefix_cache_evicted.md
  - pledisk = project_ple_disk_fast_path.md

**Messverfahren (Kürzel in den Tabellen)**
- **M-lang:** BS/dsv4_bench.py --lang.
  - Prompt: Fülltext aus 22 Wörtern; temp 1,0, top_k 40, Thinking aus, max. 500 Tok.
  - TTFT = Zeit bis zum ersten Content-Token.
  - Schritt = (Ende − erstes Token) / Drafts.
  - Annahme/Runde = 1 + angenommene Token / Drafts.
  - Jeder Prompt läuft zweimal; der zweite Lauf trifft den Präfix-Cache.
  - Länge: 905 Sätze ≈ 18k DSv4-Tok; 900 Sätze ≈ 14,6k Qwen-Tok; 1800 Sätze ≈ 35,6–35,9k DSv4-Tok bzw. 29,2k Qwen-Tok.
- **M-kurz:** dasselbe Skript ohne --lang (Prosa „Regenbogen“, Code „LRUCache“).
- **M-greedy:** BS/lookup_bench_greedy.py; temp 0; Prosa, Code und Bearbeitung (eine Datei mit 1.460 Tok umbenennen).
- **M-probe:** BS/prefill_probe.py; gleicher Fülltext wie M-lang.
- **M-speed:** tools/mtp-diagnostics/speed_27b.sh bzw. speed_dflash.sh.
  - Greedy, 400 Tok, Median aus 5 Läufen, 32k Kontext, Präfix-Cache aus.
  - tok/s schließt die TTFT ein.
- **M-bench:** vllm-bench/bench.py (August); ninfer-Sampling temp 0,6, Thinking an. In QWH: fester Prompt, 200 Tok, ignore_eos.
- **M-deep:** HO/2026-09-14/flashnext_deep_driver.py; greedy, Thinking an, 13k-Vorkontext.
- **b+g (Boot):** Zeit von der ersten Anfrage bis „bereit + drei Greedy-Antworten fertig“. Enthält etwa 5–15 s reine Antwortzeit.

## Überblick: neuester Stand (Produktion fork-union 368e74fd, 02.10.)

| Modell / Topologie | Prefill kalt (s @ Tok) | Schritt lang / kurz (ms) | Decode kurz (tok/s) | Annahme (Tok/Runde) | Boot b+g (min:s) | Kontext / KV-Pool (Tok) | Quelle |
|---|---|---|---|---|---|---|---|
| DSv4 PP5 (sauber) | 15,3 @ 35.891 und 35.694 | 85–86 @ 35,9k / Prosa 85–86, Code 88–90 | M-kurz temp 1,0: Prosa 31,4–35,3, Code 50,2–51,3 | lang 2,88–3,47; Prosa 2,70–3,00; Code 4,42–4,61 | 4:02 | 307.200 / 399.133 | clean:33-58 |
| Flash-Next PP4 (sauber) | 16,9–19,6 @ 29,2k (hängt vom Boot ab) | 70–82 @ 29k / 66–69 greedy | Prosa 36,5–36,8, Code 63,6–64,7, Bearbeitung 73,3–76,3 | 2,52 / 4,39 / 5,00 | 3:59–4:04 | 262.144 / 476.878 | clean:62-154 |
| Flash-Next TP2×PP2 (Nachmittag, laut STAND:28 sauber) | 18,5–19,8 @ 29,2k | 59–61 / 49–53 greedy | Prosa 45,2–48,8, Code 79,6–84,1, Bearbeitung 94,0–94,6 | 2,41 / 4,21 / 5,00 | 4:18 | 262.144 / 750.178 | BS/ab_next_fn2_2026-10-02.log:59-73 |
| 27B TP2 RTX-Paar (Nachmittag, Nebenlast nicht ausgeschlossen) | 22,1–22,4 @ 14,6k | 48 @ 14,6k / 37–39 greedy | Prosa 66,2–66,3, Code 102,0–102,5, Bearbeitung 103,1–103,5 | 2,47 / 3,82 / 3,99 | 2:43 | 262.144 / 979.027 | BS/ab_next_fn_2026-10-02.log:72-86 |

Produktionszeilen in CFG:
- DSv4: CFG:35-60 – Partition 10,8,8,8,9, Häppchen 512, 3.000 Blöcke.
- 27B: CFG:88-103.
- Flash-Next PP4: CFG:116-137 – PLE: Host 0 GiB + Platte, `RELEASE_PAGES=1`.
- Flash-Next TP2×PP2: CFG:139-160.

## 1. DeepSeek-V4-Flash 284B (PP5, DSpark k=5)

Seit 22.09. läuft der DeepSeek-Original-Checkpoint MXFP4. Er ist bitgleich zur vorherigen NVIDIA-NVFP4-Fassung.

### 1a Prefill, kalt

| Datum | Prompt (Tok) | TTFT kalt (s bzw. min:s) | Durchsatz (tok/s) | Stand / Bedingungen | Quelle |
|---|---|---|---|---|---|
| 10.09. | 13k Vorkontext | – | ~120 | erster PP5-Eintrag, Partition 11,8,8,8,8 | STAND:1873-1876 |
| 11.09. (Basis) | 64.349 | 9:01 min | ~119 | Fenster 65.536, 600 Blöcke | STAND:1877-1881 |
| 19.09. Betrieb | AIfred-Anfragen | 1:25–1:32 min je Antwort | ~200 | Präfix-Trefferquote 0 % | MEM/dspark:17-19 |
| 19.09. früh (Basis M-lang) | 21.857 | 1:41 min | ~217 | Häppchen 64 | MEM/dspark:59 |
| 19.09. | 21.857 | 1:20,5 min, nach NaN-Fix 1:22,5–1:25 min | 263 (bei 1:23 min) | bmm-Prefill; in STAND als „18.09.“ geführt | MEM/dspark:59, 94, 236; STAND:347 |
| 19.09. | 21.857 | 46–47 s | ~470 | Häppchen 128 | MEM/dspark:347 |
| 19.09. | 21.857 / 61.719 | 19,2–19,4 s / 55 s (vorher 1:48 min) | ~1.130 / 1.122 | gebündelter MoE-Kernel auch im Prefill | MEM/dspark:402, 575 |
| 20.09. früh | 21.857 / 61.719 | 16,7–18,3 s / 49 s | 1.190–1.310 / 1.260 | Häppchen 256, Fenster 65k | STAND:347-348; MEM/dspark:506 |
| 20./21.09. | „18k“ (Original-Skript) | 14,4–14,6 s | – | Häppchen 512, Fenster 262k bzw. 307k | STAND:2032-2035, 2047, 2193-2194 |
| 22.09. | „18k“ (Original-Skript) | 14,5 s | – | MXFP4 statt NVFP4 | STAND:2324-2328 |
| 22.09. | „18k“ (Original-Skript) | 12,9–13,1 → 12,1–12,3 → 11,3–11,5 → 11,1–11,3 s | – | moe_qpn-Zeilenblock → Union → Partition 10,8,8,8,9 → Vorladen | STAND:2417-2425, 2470-2471, 2490-2491 |
| 23.09. | 17.785–18.064 | 8,3–8,7 s | ~2.100 | Skript korrigiert (905 Sätze) | STAND:2680-2682 |
| 26.09. | 17.982 / 18.103 | 8,7 s | ~2.070 | Produktion | BS/bench_dsv4_2026-09-26b.log:29-30 |
| 27.09. | 61.590 / 61.811 | 30,6–30,7 s | ~2.010 | | BS/ab_cublas62k.log:10-11, 21-22 |
| 28.09. | 17.959 / 18.068 | 8,6 s | ~2.090 | Abnahme | Q/prod_accept7.log:10-11 |
| 30.09. | 35.626 | 16,7 s (Produktion als erste Anfrage nach Start: 18,7 s) | ~2.130 | Merge a45523ef | Q/merge30_ab.log:8, 20 |
| 30.09. | 35.745 / 35.676 | 16,7–16,8 → 15,8 s | ~2.260 | 1Cats nativer QPN8 statt Skinny-QPN8-blk | Q/dsv4_prefill_ab.log:8-10, 19-21; STAND:107-110 |
| 01.10. | 35.819 / 35.701 | 15,8/16,0 → 15,2/15,3 s | ~2.350 | 1Cat-SM70-Weg (S3) + #716 | Q/ab_sm70route.log:14-16, 44-46; STAND:98-99 |
| 02.10. sauber | 35.891 / 35.694 | 15,3 s (fork-next 15,3–15,4 s) | ~2.340 | Produktion ohne Nebenlast | clean:13-16, 43-46 |

Vom ersten Wert (119 tok/s) bis heute (2.340 tok/s) ist der Prefill etwa 20-mal schneller geworden.

### 1b Decode

| Datum | Kontext (Tok) | Schritt (ms) | Decode (tok/s) | Stand / Bedingungen | Quelle |
|---|---|---|---|---|---|
| 25.08. | kurz | – | 0,07–0,36 | eager, Emulation | J/DSH:571 |
| 26.08. | kurz | – | 1,3–4,1 | Skinny-MoE, ein Aufruf je Experte | J/DSH:721, 785 |
| 01.09. | kurz | – | K=0: 4,13; DSpark: 3,6–4,5 | | J/TUR:285, 486-487 |
| 02.09. | kurz | ~330 → 286–293 | 7,97 → 8,76 → 9,36 | Embedding-Fix, CUDA-Graphen, fp32-aux | J/TUR:523-524, 548-551, 616, 628-631, 655 |
| 02.09. | kurz | – | Essay 13,1 / Code 20,4 | moe_simt | J/TUR:687 |
| 03.09. | kurz | 144/146,5 → 112,8/114,5 | 21,3/26,7 → 25,2/28,1 | mHC fp16, danach moe_qpn; temp 0 | J/TUR:813-815, 1052-1056, 1136-1142 |
| 04.09. | kurz | – | 27,4 / 28,0 | Rebase auf 1Cat 1.5.0; temp 0 | J/TUR:1639 |
| 10.09. | 13k | – | 15,1 / 19,1 / 15,4 | | STAND:1876 |
| 11.09. | 64k | – | 12,3 | | STAND:1880 |
| 19.09. Betrieb | AIfred | – | 12,7–13,4 | temp 1,0 (die September-Werte 21–28 tok/s waren temp 0) | MEM/dspark:17-18 |
| 19.09. früh (Basis M-lang) | ~300 / 2,5k / 9k / 18k | 130 / 181 / 193 / 210 | bei 18k: Prosa 14–15, Code 22–24 | Fenster 65k | MEM/dspark:23-26 |
| 19.09. | kurz / 18k | 106 / 118 | | qk_dsplit | MEM/dspark:44 |
| 19.09. | kurz / 18k | 79 / 92 | bei 18k: Prosa 32–33, Code 51–56 | Sparse-MLA als Gather + BMM | MEM/dspark:289-290 |
| 20.09. früh | kurz / 18k | 81 / 81 | bei 18k: Prosa 35–40, Code 50–62 | Indexer über cuBLAS, Fenster 65.536 | STAND:350-352, 378-384 |
| 20.09. | 18k | 79 (Fenster 65.536) vs. 101 (Fenster 524.288) | 35–40 vs. 30 | Zielkonflikt Fenstergröße gegen Tempo | STAND:2012-2017 |
| 20./21.09. | ~22k | 88–89 / 91 / 93 | 32–38 / 32–33 / 28–32 | Fenster 262k / 307k (gewählt) / 350k | STAND:2047, 2183-2187 |
| 22.09. | „18k“ | 90 / 93 | Prosa 32,0/34,9, Code 53,0/52,6 | MXFP4 | STAND:2326-2328 |
| 26.09. | kurz / 18k | 90–94 / 91–98 | Prosa 32,0–33,8, Code 45,6–52,8 | | BS/bench_dsv4_2026-09-26b.log:25-30 |
| 27.09. | kurz / 18k / 62k | cuBLAS an: 90 / 90–91 / 90–91; aus: 81–82 / 91 / 116 | bei 62k: an 33,7–36,0, aus 22,5–26,3 | A/B Indexer-cuBLAS | BS/ab_cublas62k.log:4-11, 15-22 |
| 28.09. | kurz / 18k | 90–94 / 90 | Prosa 31,9–32,6, Code 47,6–53,9 | | Q/prod_accept7.log:6-11 |
| 30.09. | 35,6k | 90–91 (erste Anfrage nach Start: 115) | 23,4–40,0 | | Q/merge30_ab.log:8-9, 20-21 |
| 01.10. | 35,8k | 91 → 84–86 | | S3 + #716 | Q/ab_sm70route.log:14-16, 44-46; STAND:98-99 |
| 02.10. sauber | 35,7–35,9k / kurz | 85–86 / Prosa 85–86, Code 88–90 | 33,6–40,7 / Prosa 31,4–35,3, Code 50,2–51,3 | fork-next gleich (erste Anfrage nach Start: 113) | clean:13-16, 25-28, 43-46, 55-58 |

Zwei llama.cpp-Referenzen:
- 25.08., M-bench: math 40,4 ± 1,1 und code 42,8 ± 0,4 tok/s (RES:393-402).
- 03.09., gleiche Prompts, temp 0: Essay 21,2 und Code 37,8 tok/s, Schritt ~110 ms (J/TUR:698-701, 1136-1139).

### 1c Präfix-Cache (TTFT der Folgeanfrage)

| Datum | Fall | TTFT (s bzw. min:s) | Quelle |
|---|---|---|---|
| 19.09. | identische zweite Anfrage / 18k-Folgefrage | 20,8 → 1,8 s / 3,2 s statt 1:39 min (Backport vllm#44082) | MEM/dspark:22 |
| 19.09. | 18k-Folgefrage | 2,6 → 1,4–1,6 s | MEM/dspark:256 |
| 20.09. | gecachter Präfix | 1,3 → 0,8 s | STAND:349 |
| 20.09. | 22k nach einer 125k-Anfrage | 14,6 s (komplett verdrängt) | STAND:2057-2061 |
| 21.09. | 22,9k nach einer 30k- bzw. 125k-Anfrage | 14,4 → 0,8 / 0,7 s | MEM/pcache:10-11, 41; STAND:2188 |
| 26.09. | A nach einer Fremdanfrage: main vs. #667 | 10,6 s vs. 0,7 s | UC/pr-667-body-gesendet-2026-09-27.md:68-71 |
| 02.10. | 35,7–35,9k wiederholt | 0,7–1,4 s | clean:44, 46 |

### 1d Boot und Kontext

| Datum | Boot (min:s) | Messart / Stand | Quelle |
|---|---|---|---|
| 12.09. | 11:50 | ds_accept.sh, tilelang 0.1.14 | STAND:1566-1568 |
| 12./13.09. (Basis) | 8:07 / 8:39 | exakter Befehl / warm über llama-swap inkl. einer Antwort; NVFP4 | HO/2026-09-12/abnahme_prod_rerun.out:14, 34 |
| 14.09. | 9:30 | nach Merge 80c88e8d | HO/2026-09-14/abnahme_merge.out:2 |
| 21.09. | 10:25 → 6:03 | Drafter liest den Checkpoint nicht mehr doppelt; Drafter-Phase 4:15 min → 17 s | MEM/drafter:13; UC/pr-drafter-skip-checkpoint-weights.md:84-85 |
| 22.09. | Gewichte lesen 3:44 statt 4:24–4:46 | MXFP4 | STAND:2329 |
| 28.09. | 9:16 → 5:48 (b+g); mit 12 GiB 8:48 | MemoryHigh 16 GiB für llama-swap | Q/pressure-dontneed.run.log:2, pressure-memhigh.run.log:2, pressure-memhigh12.run.log:2; MEM/memhigh:16-17 |
| 29.09. | 5:30 (mmap) → 5:07 → 4:39 → 4:22 | Direct-IO v1, v2, Produktion | MEM/dio:41; Q/pressure-directio.run.log:2; Q/directio_v2.log:24; Q/prod_accept8.log:16 |
| 30.09. / 01.10. | 4:04–4:25 | b+g | Q/merge30_ab.log:2, 14; Q/accept_union.log:2; Q/prod_union_accept.log:2 |
| 02.10. | 4:02 (Produktion) / 4:20 (fork-next) | b+g | clean:33, 2 |

Kontext:
- **11.09. (Basis):** 65.536 Tok, KV-Pool 93.622 (STAND:1877-1879).
- **20.09.:** 524.288 Tok erprobt und verworfen (STAND:1989-1996, 2012-2017).
  - Pool 721.221 Tok, Nadel bei 497k 4/4, Prefill ~13 min.
  - Preis: Decode-Schritt +27 %.
- **20.09.:** 262.144 Tok, Pool 294.958, Nadel 125k 4/4 (STAND:2040-2049).
- **21.09.:** 307.200 Tok gewählt, Pool 399.133, Nadel 124.471 4/4 (STAND:2180-2195).
- **Heute:** unverändert 307.200 / 399.133 (CFG:36; clean:36).

### 1e Annahme

Zwei Maße kommen vor und sind nicht direkt vergleichbar: „%“ = Anteil angenommener Entwurfstoken; „Tok/Runde“ = 1 + angenommene Token / Drafts.

| Datum | Annahme (% bzw. Tok/Runde) | Bedingungen | Quelle |
|---|---|---|---|
| 01.09. | ~5 % | Essay | J/TUR:486 |
| 02.09. | 36 % | Embedding-Fix | J/TUR:523 |
| 03.09. | 32 % / 64 % (llama.cpp 27,9 % / 68,8 %) | Essay / Code, temp 0 | J/TUR:698-701 |
| 03.09. | 51 % | mHC fp16 | J/TUR:813-814 |
| 19.09. | 2,20 Tok/Runde | AIfred-Betrieb, temp 1,0 | MEM/dspark:18 |
| 19.09. (Basis M-lang) | Prosa ~3,0, Code ~4,7 Tok/Runde | unabhängig vom Kontext | MEM/dspark:26 |
| 26.09. | Prosa 2,88–3,05, Code 4,19–4,97, 18k 2,77–3,07 Tok/Runde | | BS/bench_dsv4_2026-09-26b.log:25-30 |
| 02.10. | lang 2,88–3,47, Prosa 2,70–3,00, Code 4,42–4,61 Tok/Runde | | clean:43-46, 55-58 |

Coding-K7-Eintrag (k=7, Fenster 65.536):
- 01.10.: Prefill 35,8k 21,0/21,2 → 20,3/20,4 s; Schritt 85–86 → 80–81 ms (Q/ab_sm70route.log:72-74, 102-104).
- Produktionsabnahme: 20,2 s / 80 ms (Q/prod_sm70route_accept.log:42).
- Sonst keine Historie: Peuqui hält den Eintrag aus den Messreihen heraus.

## 2. Qwen3.8-Flash-Next 180B-A4B (NVFP4, MTP k=4)

Wechsel von Checkpoint und Topologie:
- Bis 14.09.: RadixArk-MTPQ-Transplantat, TP2×PP2.
- Ab 14./15.09.: nvidia-Checkpoint fc694b54 mit FP8-MTP-Kopf (MEM/ho14:17).
- Ab 23.09.: PP4.
- Ab 30.09.: zusätzlich wieder ein TP2×PP2-Eintrag.

### 2a Prefill, kalt

| Datum | Topologie | Prompt (Tok) | TTFT kalt (s bzw. min:s) | Durchsatz (tok/s) | Stand / Bedingungen | Quelle |
|---|---|---|---|---|---|---|
| 30.08. | TP2×PP2 | 13k | 33 → 7,7 s | 392–448 → 1.482–1.696 | QSA-Kacheln für Pre-Ampere; alter Stack, MTPQ | J/SLH:274-277 |
| 07.09. | TP2×PP2 | – | – | 413–466 | 1Cat 1.5.0, MTPQ | STAND:453 |
| 14.09. (Basis, heutiger Checkpoint) | TP2×PP2 | 13.049–13.092 / 51.999 | 18,4–19,2 s (erste Anfrage 26,6 s) / 1:15 min | 680–708 / 691 | nvidia, M-deep, echter Text | HO/2026-09-14/flashnext_deep.out:125-135 |
| 16.09. | TP2×PP2 | 13k / 39k Fülltext | 11,8 / 35,6 s | 1.092–1.100 | Fülltext ist zu schnell; echter Text 22k/74k: 653–664 tok/s | PLEK:349-351, 419-426 |
| 22.09. (Basis 29k) | TP2×PP2 | ~29k (als „18k“ beschriftet) | 26,6–27,2 → 19,3–19,4 s (Marlin 17,7–17,8 s) | ~1.080 → ~1.510 | SM70-Gate berücksichtigt `--moe-backend`: TurboMind → Skinny | STAND:2560-2571, 2683-2686 |
| 22./23.09. | TP4 | ~29k | 36,1 s | ~810 | verworfen | STAND:2578-2584 |
| 23.09. | TP2×PP2 → PP4 | 29.159–29.247 | 18,6–19,1 → 13,5–14,0 s | ~1.540 → ~2.160 | M-probe | STAND:2748-2757; FNOP:79-85 |
| 23.09. | PP4 | ~29,2k / 15–17k echter Text | 13,5 s / 7,7–12,5 s | | Produktionsabnahme (PLE in Karten + Host 3 GiB) | STAND:2870-2871, 2900-2909 |
| 24.09. | PP4 | 12 echte Texte, 8–17k | Summe 1:45 min (Host 0 + Platte) vs. 1:43 min (Host 3) | | Rückbau; seitdem Host 0 + Platte 28,8 GiB | STAND:2988-2997 |
| 26.09. | PP4 | 14,7k | 7,5–8,5 s (erste Anfrage 11,5 s) | ~1.730–1.960 | | BS/bench_fn_prod_2026-09-26.log:13; BS/bench_fn_prefill_repeat_2026-09-26.log:7-8 |
| 27.09. | TP2×PP2 | 14,7k | 10,5 s (Fork) vs. 19,2 s (1Cat-main-Pfad) | ~1.400 vs. ~770 | | BS/ab_rest2_2026-09-27.log:10, 35 |
| 30.09. | PP4 | 4.933 / 14.732 / 29.203 / 60.064 | 4,1 / 9,8 / 16,9* / 39,0 s | 1.200 / 1.500 / 1.730 / 1.540 | *erste lange Anfrage nach Start | Q/topo_bench_pp4.log:7, 9; Q/longctx_bench.log:3, 5 |
| 30.09. | TP2×PP2 Host 0 | gleiche Prompts | 7,8* / 9,7 / 18,9* / 43,2 s | 630 / 1.520 / 1.550 / 1.390 | *erste Anfrage nach Start | Q/topo_bench_host0.log:7, 9; Q/longctx_bench.log:23, 25 |
| 30.09. | PP4 / TP2×PP2 | 29.149 | 13,5 / 18,6–18,7 s | 2.160 / 1.560 | zweite frische Anfrage; die erste nach Start dauert 18,8–21,5 bzw. 19,9–23,0 s | Q/merge30_recheck.log:4-27 |
| 30.09. | PP4 (Partition 12,12,13,11) | 29.149 | Skinny 17,4 / TurboMind 18,6 / Marlin 13,9 s | | MoE-Backend-A/B, je ein Boot | Q/moe_backend_ab.log:12, 25, 38 |
| 02.10. | PP4 | ~29,2k | 13,5–18,4 s je nach Boot | | Häppchen 2048/4096/8192 ohne belastbaren Unterschied | STAND:984-994 |
| 02.10. sauber | PP4 | 29.167–29.243 | Produktion 16,9–19,6 s; fork-next 13,6–18,5 s | 1.490–1.730 / 1.580–2.150 | Major-Faults: Produktion 0,86–1,06 Mio., fork-next B 165k | clean:68-75, 90, 101-108, 123, 132-139, 154, 165-172, 187 |
| 02.10. | TP2×PP2 | 29.193 / 29.287 | 18,5 / 19,8 s | ~1.480–1.580 | | BS/ab_next_fn2_2026-10-02.log:64, 66 |

### 2b Decode

| Datum | Topologie | Kontext (Tok) | Schritt (ms) | Decode (tok/s) | Stand / Bedingungen | Quelle |
|---|---|---|---|---|---|---|
| 27.08. | TP2×PP2 | kurz | – | k=0: 31,4; k=4 mit BF16-MTP-Kopf: 14,4 | M-bench; llama.cpp ohne Spekulation: 33,2 | J/QWH:925-935, 1178-1183 |
| 28.08. (Basis) | TP2×PP2 | kurz | – | k=0: 32,2; MTPQ k=4: 49,2 schwer / 67,2 vorhersagbar; mit Capture [1,2,4,5,8]: 51,9 / 68,2 | MML 16.384 | J/QWH:2444-2452, 2512-2519; FNOP:64-68 |
| 30.08. | TP2×PP2 | 13k | – | 26–29 → 22,5–25,6 | Kachel-Umbau senkte die Annahme | J/SLH:274-277 |
| 07.09. | TP2×PP2 | kurz / lang repetitiv / ~9–10k echter Text | – | 56,3 / 75,6 / 20–21 | MTPQ, MML 262.144 | STAND:444-462 |
| 14.09. | TP2×PP2 | kurz / 13k / 52k | – | 64,7 / 62,6–78,2 / 60,4 | nvidia, M-deep | HO/2026-09-14/flashnext_deep.out:125-135 |
| 14./15.09. | TP2×PP2 | Betrieb | – | ~47–50 | Produktion nvidia | MEM/ho14:17 |
| 16.09. | TP2×PP2 | 4 Sondenfragen | – | 56,5–66,8 | | PLEK:341-343 |
| 22.09. | TP2×PP2 | ~29k | 53–56 → 52–55 (Marlin 56–59) | Code 70–73 → 77–78 | TurboMind → Skinny | STAND:2563-2569 |
| 22./23.09. | TP4 / TP2×PP2 | ohne MTP | – | 30,1 / 28,9–31,2 | Decode gleichauf | FNOP:83-92 |
| 23.09. | TP2×PP2 / PP4 | 29k | – | 32,5–40,4 / 40,5–41,0 (Store-Karte) bzw. 35,5–43,7 (SSD) | M-probe | STAND:2753-2757 |
| 23.09. | PP4 | 29k | – | Mittel 36,9 (32,0–43,3); Abnahme 40,0–43,4 | Produktion | STAND:2834-2839, 2870-2871 |
| 24.09. | PP4 / TP2×PP2 | 8–17k echter Text | – | Mittel 40,8 / 49,0 | PP4 ist Produktion mit Host 0 + Platte; TP2×PP2 nur Nachmessung für #646 | STAND:2996, 3045-3049 |
| 26.09. (Basis ms PP4) | PP4 | kurz / 14,7k | Prosa 65–69, Code 75–76 / 63–64 | Prosa 32,1–33,2, Code 51,3–52,0 / 32,9–39,5 | M-kurz / M-lang | BS/bench_fn_prod_2026-09-26.log:9-13; BS/bench_fn_prefill_repeat_2026-09-26.log:7-8 |
| 27.09. (Basis ms TP2) | TP2×PP2 | kurz / 14,7k | 55–56 / 58 | Prosa 40,6–41,7, Code 68,1–70,6 | 1Cat-main-Pfad: 57–68 / 60 | BS/ab_rest2_2026-09-27.log:6-10, 31-35 |
| 28.09. | PP4 | kurz / 14,7k | 66–68 / 69 | | | Q/prod_accept7.log:25-30 |
| 30.09. | PP4 | kurz / 4,9k / 14,7k / 29k / 60k | 64–68 / 69–70 / 69 / 77–78 / 75–81 | | | Q/topo_bench_pp4.log:3-10; Q/longctx_bench.log:3-6 |
| 30.09. | TP2×PP2 Host 0 | kurz / 4,9k / 14,7k / 29k / 60k | 55–63 / 57–58 / 59–60 / 60–61 / 62–63 | | | Q/topo_bench_host0.log:3-10; Q/longctx_bench.log:23-26 |
| 30.09. | PP4 (12,12,13,11) | 29k | Skinny 69–71, TurboMind 79–80, Marlin 80–81 | | | Q/moe_backend_ab.log:11-13, 24-26, 37-39 |
| 01.10. | PP4 | kurz / 29k | 64–69 / 69–71 | | union = Produktion | Q/ab_union.log:3-16 |
| 02.10. | PP4 | kurz | K=4 ~66, K=3 ~60, K=2 ~54 | K=4: Prosa 33,6, Code 57,0, Bearbeitung 83,5 | K-Sweep | STAND:1024-1031 |
| 02.10. sauber | PP4 Produktion | kurz greedy / 29k | 66–69 / 70–82 | Prosa 36,5–36,8, Code 63,6–64,7, Bearbeitung 73,3–76,3 / 26,5–38,8 | | clean:68-81, 132-145 |
| 02.10. sauber | PP4 fork-next | kurz greedy / 29k | 53–63 / 67–73 | 41,0–48,9 / 67,2–78,4 / 79,0–91,0 | Entwurfsvokabular 98.304 | clean:101-114, 165-178 |
| 02.10. | TP2×PP2 Produktion | kurz greedy / 29k | 49–53 / 59–61 | Prosa 45,2–48,8, Code 79,6–84,1, Bearbeitung 94,0–94,6 / 41,0–42,5 | | BS/ab_next_fn2_2026-10-02.log:64-73 |

### 2c Boot und Kontext

| Datum | Topologie | Boot (min:s) | Messart / Stand | Quelle |
|---|---|---|---|---|
| 28.08. (Basis) | TP2×PP2 | ~7 min | MTPQ | FNOP:59 |
| 07.09. | TP2×PP2 | ~9 min | Skript-Weg | STAND:439 |
| 10.09. | TP2×PP2 | 10:12 kalt / 5:16 warm | | STAND:247-250 |
| 12./13.09. | TP2×PP2 | 5:16 / 5:01 | exakter Befehl / warm inkl. Antwort | HO/2026-09-12/abnahme_prod_rerun.out:10, 30 |
| 13.09. | TP2×PP2 | 5:40 kalt, warm 5:12 / 5:43 | | STAND:268-271 |
| 23.09. | PP4 | 9:30 | Store-Transfer 9:05 min → 50,8 s durch MADV_SEQUENTIAL | STAND:2730-2743, 2870 |
| 29.09. | PP4 | 4:48 (mmap) → 3:57 | Direct-IO + Filter im PLE-Worker | MEM/dio:30-34, 43; Q/directio_ab3.log:3; Q/prod_fn_again.log:1 |
| 29./30.09. | TP2×PP2 | 9:31 → 6:24; neuer Eintrag Host 0: 5:21 beim ersten Start, danach 4:12 | Direct-IO, einmaliges Lesen je TP-Gruppe | UC/pr-direct-io-entwurf-2026-09-30.md:119-120; Q/prod_accept9.log:2; Q/tp2_readonce.log:2; MEM/dio:61-64 |
| 30.09. / 01.10. | PP4 / TP2 | 7:24–8:59 bzw. 8:25–9:09; union-Test 3:29 | nach Leeren der Caches bzw. Code-Wechsel wird kalt kompiliert | Q/merge30_ab.log:50, 62; Q/merge30_recheck.log:2-23; Q/accept_union.log:52, 70; Q/prod_union_accept.log:46, 61; Q/ab_union.log:2, 10; MEM/m30:55-58 |
| 02.10. | PP4 | Produktion 3:27–4:04; fork-next 3:56–4:08 | b+g | clean:63, 94, 127, 158; BS/ab_next_fn2_2026-10-02.log:2, 30 |
| 02.10. | TP2×PP2 | Produktion 4:18; fork-next 5:22 | b+g | BS/ab_next_fn2_2026-10-02.log:60, 88 |

Kontext und KV-Pool:
- **27.08.:** MTP-Läufe mit MML 4.096 (J/QWH:1633); k=0 mit 262.144 und 797.226 Slots (J/QWH:1848).
- **28.08.:** MML 16.384 (J/QWH:2446).
- **Ab 07.09.:** 262.144 mit MTP (STAND:434, 1868-1871).
- **14.09.:** Pool 319.131 beim ersten Boot, warm 645.599 (HO/2026-09-14/flashnext_deep.out:138; MEM/ho14:17).
- **22.09.:** Pool 592.612 (TurboMind) / 550.781 (Skinny) / 446.202 (Marlin); Nadeln 30k und 124k 4/4 (STAND:2563-2569).
- **23.09., PP4:** Pool 564.725; Nadeln 24.488 und 101.605 4/4 (STAND:422-424, 2762).
- **24.09.:** 437.836 (STAND:3003).
- **30.09.:** Nadeln in beiden Topologien 4/4 (Q/longctx_bench.log:7-20, 27-40).
- **01.10.:** PP4 472.695 / TP2 659.543 (Q/prod_union_accept.log:48, 63).
- **02.10.:** PP4 476.878, fork-next 425.286; TP2 750.178 / 762.727 (clean:66, 99; BS/ab_next_fn2_2026-10-02.log:63, 93).

### 2d Annahme

| Datum | Annahme (% bzw. Tok/Runde) | Bedingungen | Quelle |
|---|---|---|---|
| 27.08. | 23 % / 1,92 | BF16-MTP-Kopf | J/QWH:932-935 |
| 28.08. (Basis) | 73,1 % / 3,92; mit Capture [1,2,4,5,8] 69,8 % / 3,79 | MTPQ | J/QWH:2455-2457, 2519 |
| 30.08. | lange Prompts 19,0 → 13,3 % | Kachel-Numerik | J/SLH:276-277 |
| 07.09. | 3,71 von 5 Tok/Runde | MTPQ | STAND:454-455 |
| 14.09. | 3,18–3,97 (nvidia) / 3,15–3,93 (MTPQ) Tok/Runde | M-deep, greedy | HO/2026-09-14/flashnext_deep.out:79-89, 125-135 |
| 26.09.–01.10. | Prosa 2,00–2,22, Code 3,21–4,35, lang 1,87–2,68 Tok/Runde | temp 1,0, PP4 | BS/bench_fn_prod_2026-09-26.log:9-13; Q/topo_bench_pp4.log:3-10; Q/longctx_bench.log:3-6; Q/ab_union.log:3-16 |
| 02.10. | greedy, Produktion: 2,52 / 4,39 / 5,00; fork-next: 2,60 / 4,21 / 5,00; 29k bei temp 1,0: 2,12–2,83 Tok/Runde | | clean:68-81, 101-114 |

## 3. Qwen3.8-27B NVFP4 (TP2 auf dem RTX-Paar, MTP k=3)

### 3a Prefill, kalt

Vor dem 26.09. habe ich in den gelesenen Quellen keinen 27B-Prefill-Wert gefunden.

| Datum | Prompt (Tok) | TTFT kalt (s) | Durchsatz (tok/s) | Stand / Bedingungen | Quelle |
|---|---|---|---|---|---|
| 26.09. (Basis) | 14.692–14.728 | 27,4–27,8 | ~530 | Fork-Pfad, M-lang | BS/bench_27b_2026-09-26.log:10-11, 21-22 |
| 27.09. | 14.713 / 14.735 | Fork 27,5 vs. #604 22,7 | ~535 vs. ~650 | Produktion läuft ab 27.09. mittags auf dem #604-Pfad | BS/ab_paths_2026-09-27.log:102; BS/ab_rest2_2026-09-27.log:56; MEM/m26:189-190, 230-235 |
| 28.09. | 14.666 / 14.710 | 22,3 / 22,8 | ~650 | | Q/prod_accept7_q27.log:10-11 |
| 30.09. | 29.203 | 47,1 | ~620 | | Q/merge30_ab.log:32, 44 |
| 01.10. | 14.585–14.602 | 22,5–22,6 | ~645 | | Q/ab_union.log:19, 27; Q/prod_union_accept.log:79 |
| 02.10. Nachmittag | 14.555 / 14.594 | Produktion 22,1–22,4; fork-next 19,8–20,2 | ~655 / ~730 | Nebenlast nicht ausgeschlossen | BS/ab_next_fn_2026-10-02.log:77, 79, 103, 105 |

### 3b Decode

| Datum | Kontext (Tok) | Schritt (ms) | Decode (tok/s) | Stand / Bedingungen | Quelle |
|---|---|---|---|---|---|
| 24.08. | kurz | – | 2× V100, k=7: math 88,1 / code 58,6 / prose 86,3; k=3: 80,5 / 60,2 / 81,1 | Original-v100-skinny-Fork, M-bench | RES:25-36 |
| 24.08. | kurz | – | 1× RTX: vLLM-AWQ k=3 52,2 / 36,3 / 46,0; llama.cpp Q8 n=3 31,5 / 26,3 / 34,8 | Referenzen | RES:27-30 |
| 25.08. | kurz | – | NVFP4 2× V100: K=0 40,5, k=7 88,2; 2× RTX: K=0 43,2, k=7 79,1 | Korrektheit auf Turing nicht belegt (RES:54) | RES:193-199, 230-233 |
| 08.09. | kurz / 13k | – | RTX, k=3: Fork-sm75 74,33, Upstream mit Fixes 69,64; k=0: 43,5 / 25,0 | M-speed | MEM/turfix:29-40 |
| 09.09. (Basis) | kurz | – | RTX MTP k=3: 73,35; V100: 66,13 | M-speed, gemeinsamer GDN-Pfad | STAND:524-535, 557-560 |
| 14.09. | kurz | – | RTX: 73,34; PP2: 61,05 | | STAND:119-121 |
| 26.09. (Basis ms) | kurz / 14,7k | 39–46 / 48 | Prosa 48,6–55,8, Code 91,3–99,8 / 45,7–47,6 | temp 1,0, Fork-Pfad | BS/bench_27b_2026-09-26.log:6-11 |
| 27./28.09. | kurz / 14,7k | 38–46 / 48 | Code 84,0–96,9 | #604 | BS/ab_rest2_2026-09-27.log:52-56; Q/prod_accept7_q27.log:6-11 |
| 30.09. | 29k | 56–57 (erste Anfrage nach Start: 83–84) | 39,4–39,6 | | Q/merge30_ab.log:32-33 |
| 01.10. | kurz / 14,6k | 39 / 48 (erste Anfrage nach Start: 81–85) | | | Q/ab_union.log:19-32 |
| 02.10. Nachmittag | kurz greedy / 14,6k | 37–39 / 48 (erste Anfrage nach Start: 73–74) | Prosa 66,2–66,3, Code 102,0–102,5, Bearbeitung 103,1–103,5 | M-greedy / M-lang | BS/ab_next_fn_2026-10-02.log:77-86 |

### 3c Boot, Kontext, Annahme

| Datum | Boot (min:s) | Messart / Stand | Quelle |
|---|---|---|---|
| 07.09. (Basis) | 6:30 (RTX-Paar) / ~2:00 (V100-Paar) | probe.sh | STAND:512-516 |
| 10.09. | 8:31 kalt / 2:01 warm | | STAND:247-249 |
| 12./13.09. | 2:25 / 2:02 | | HO/2026-09-12/abnahme_prod_rerun.out:2, 22 |
| 13.09. | 7:56 kalt, 1:25 / 1:20 warm | Backport torch #173556 | STAND:272-276 |
| 29.09. | 3:27–3:29 (mmap) → 2:35–2:47 | Direct-IO | Q/directio_ab3.log:21, 30; Q/prod_accept8.log:9; MEM/dio:42 |
| 30.09. | 2:25 (Gewichte 34 statt 50 s) | einmaliges Lesen je TP-Gruppe | Q/prod_accept9.log:11-13; MEM/dio:61-62 |
| 30.09. / 01.10. | 7:45–8:28; union-Test 2:35 | Kaltkompilierung | Q/merge30_ab.log:26; Q/accept_union.log:88; Q/prod_union_accept.log:76; Q/ab_union.log:18, 26 |
| 02.10. | 2:43; fork-next 8:16 (eigener Cache, kalt) | b+g | BS/ab_next_fn_2026-10-02.log:72, 97 |

Kontext:
- **07.09.:** 32.768 Tok im Bench; KV 661.796 (STAND:518-519).
- **Ab 10.09.:** Eintrag mit 256K und Präfix-Cache (STAND:249).
- **Heute:** 262.144 Tok; KV 955.335 (01.10.) bzw. 979.027 (02.10.) (CFG:89; Q/prod_union_accept.log:78; BS/ab_next_fn_2026-10-02.log:75).
- **Längster gemessener Prompt:** 29.203 Tok. Einen Nadeltest habe ich nicht gefunden.

Annahme:
- **24./25.08.:** 39,5 % (V100, k=7) bzw. 36,6 % (RTX, k=7) (RES:195, 232).
- **09.09. (Basis):** 2,963 Tok/Runde (STAND:526-528).
- **26.09., temp 1,0:** Prosa 2,15–2,22, Code 3,52–3,89, 14,7k 2,22–2,26 (BS/bench_27b_2026-09-26.log:6-11).
- **02.10., greedy:** 2,47 / 3,82 / 3,99; bei 14,6k und temp 1,0: 1,95–2,16 (BS/ab_next_fn_2026-10-02.log:77-86).

## 4. DFlash2 (nur 27B, nie in Produktion)

Alle Werte unter M-speed. In allen Läufen derselbe Text-SHA (0106659946c064b1).

| Datum | Karten | Verfahren | Decode (tok/s) | Annahme (Tok/Runde) | Quelle |
|---|---|---|---|---|---|
| 09.09. | 2× RTX | MTP k=3 / DFlash2 vor Gate-Patch / nach Gate-Patch / mit Block-Pack | 73,36 / 21,35 / 69,13 / 72,72 | 2,963 / 1,015 / 3,353 / 3,353 | STAND:560-563 |
| 09.09. | 2× V100 | MTP k=3 / DFlash2 / mit Block-Pack | 66,13 / 74,09 / 74,02 | 2,963 / 3,381 / 3,381 | STAND:557-559 |
| 10.09. | 2× RTX / 2× V100 | mit quantisiertem Entwurfskopf | 77,13 / 76,33 | 3,325 | STAND:496-498, 564-565 |
| 10.09. | 2× RTX | Eintrag `Qwen3.8-27B-NVFP4-DFlash2-vllm` (256K) | Boot 7:10 kalt / 2:20 warm | – | STAND:251 |
| 12.09. | 2× RTX | tilelang 0.1.14 | 77,14 | 3,325 | STAND:1570-1571 |
| 12./13.09. | RTX / V100 | Produktionskopf (Kopf incoai: 72,68–73,15 / 74,08–75,09) | 76,88–76,92 / 76,42 | 3,325 | STAND:198-203 |
| 14.09. | RTX / V100 | nach Merge | 76,72 / 76,48 | – | STAND:119-121 |

Danach gibt es keine Messung mehr. Der Eintrag steht nicht mehr in CFG, und DFlash2 wurde nie unter Produktionsbedingungen gemessen (STAND:506-510, 1038-1039).

## 5. Meilensteine mit gemessener Wirkung

### DSv4

| Datum | Änderung | Wirkung | Quelle |
|---|---|---|---|
| 02.09. | Embedding-Fix im Drafter | Annahme 5 → 36 %; 3,6 → 7,97 tok/s | J/TUR:523-524 |
| 02./03.09. | moe_simt → mHC fp16 → moe_qpn | Schritt ~290 → 113 ms; Essay 9,4 → 25,2 tok/s | J/TUR:655-687, 813, 1136-1142 |
| 19.09. | Präfix-Cache-Backport vllm#44082 | Folgefrage 1:39 min → 3,2 s | MEM/dspark:22 |
| 19.09. | qk_dsplit + Sparse-MLA als Gather/BMM | Schritt bei 18k: 210 → 92 ms | MEM/dspark:44, 289-290 |
| 19./20.09. | bmm-Prefill, Häppchen 128 → 256, gebündelter MoE | 21,9k: 1:41 min → 18,3 s; 62k: 1:48 min → 49 s | MEM/dspark:59, 347, 402, 506 |
| 20.09. | Indexer über cuBLAS | 18k: 91 → 81 ms. Im A/B vom 27.09.: 62k 116 → 90 ms, kurzer Kontext +8–9 ms | STAND:378-384; BS/ab_cublas62k.log |
| 20./21.09. | Häppchen 512 + Fenster 307k | Prefill 16,3 → 14,4–14,6 s; Preis: Schritt 81 → 91 ms durch das größere Fenster | STAND:2032-2050, 2183-2187 |
| 21.09. | Drafter-Skip | Boot 10:25 → 6:03 min | MEM/drafter:13 |
| 22.09. | moe_qpn-Umbau, Partition, Vorladen | „18k“: 14,5 → 11,1–11,3 s (−23 %) | STAND:2417-2491 |
| 26.09. | #667 | Präfix nach Fremdanfrage: 10,6 → 0,7 s | UC/pr-667-body-gesendet-2026-09-27.md:68-71 |
| 28./29.09. | MemoryHigh 16 GiB, dann Direct-IO | Boot 9:16 → 5:48 → 4:22 min | Q/pressure-*.run.log:2; MEM/dio:41 |
| 30.09. | Kontrolle: ohne Skinny-MoE (ohne MTP) | 35,8k-Prefill 31,9 s statt 18,9 s; Decode 6,8 statt 15,5 tok/s | UC/pr-skinny-s1-moe-entwurf-2026-09-30.md:42-43 |
| 30.09. | nativer QPN8 | 16,8 → 15,8 s | STAND:107-110 |
| 01.10. | S3 + #716 | 15,8 → 15,2 s; Schritt 91 → 84–86 ms | STAND:98-99 |

### Flash-Next

| Datum | Änderung | Wirkung | Quelle |
|---|---|---|---|
| 28.08. | NVFP4-MTP-Block (MTPQ) | k=4: 14,0 → 49,2 tok/s; Annahme 51 → 73,1 % | J/QWH:2449-2457 |
| 30.08. | QSA-Kacheln | 13k-Prefill 33 → 7,7 s | J/SLH:274-277 |
| 22.09. | SM70-MoE-Gate → Skinny-MoE | 29k: 27 → 19,4 s; Code 70–73 → 77–78 tok/s | STAND:2560-2571 |
| 23.09. | PP4 statt TP2×PP2 | 29k: 18,6–19,1 → 13,5–14,0 s; danach misst sich der Decode-Schritt aber langsamer (64–78 ms gegen 55–61 ms bei TP2) | STAND:2753-2757; Q/topo_bench*.log |
| 23.09. | MADV_SEQUENTIAL | Store-Transfer 9:05 min → 50,8 s | STAND:2742 |
| 24.09. | Rückbau auf Host 0 + Platte | Prefill-Summe 1:43 → 1:45 min, Decode gleich, mehr freier RAM | STAND:2993-2997 |
| 29.09. | Direct-IO | Boot PP4 4:48 → 3:57; TP2×PP2 9:31 → 6:24 | MEM/dio:43; UC/pr-direct-io-entwurf-2026-09-30.md:119-120 |

### 27B

| Datum | Änderung | Wirkung | Quelle |
|---|---|---|---|
| 09./10.09. | DFlash2-Fixes | RTX 21,35 → 77,13 tok/s | STAND:561-565 |
| 13.09. | torch-Backport | Warmstart 2:55 → 1:20–1:25 min | STAND:272-276 |
| 27.09. | #604 | Prefill 14,7k: 27,5 → 22,7 s (−17 %), Decode gleich | MEM/m26:189 |
| 29./30.09. | Direct-IO + einmaliges Lesen | Boot 3:27 → 2:25 min | MEM/dio:42, 62 |

## 6. Lücken und nicht vergleichbare Werte

1. **Die Messmethode hat mehrfach gewechselt.**
   - Bis 04.09.: DSv4 mit temp 0, die 21–28 tok/s (MEM/dspark:18).
   - August: M-bench mit temp 0,6 und Thinking an.
   - Ab 19.09.: temp 1,0 / top_k 40.
   - 02.10.: zusätzlich greedy.
   - M-speed rechnet die TTFT ins tok/s ein.
   - tok/s-Werte aus verschiedenen Methoden lassen sich nicht vergleichen. Die Schrittzeit in ms ist robuster.
2. **„18k“ ist kein fester Prompt.**
   - 19./20.09.: 21.857 Tok.
   - 22.09. bis 23.09. vormittags: Messfehler, in Wahrheit 35.812 Tok. Die A/B-Verhältnisse bleiben gültig (STAND:2670-2688).
   - Ab 23.09.: 17,8–18,1k Tok.
   - Ab 30.09.: 35,6–35,9k Tok.
   - Vor dem 23.09. wurde die Tokenzahl nicht je Lauf mitgeschrieben.
3. **Der Fülltext schmeichelt dem Prefill.** 16- und 22-Wort-Fülltexte fallen für MoE und PLE-Cache zu günstig aus: 1.100 tok/s gegen 653–664 tok/s bei echtem Text (PLEK:419-426; STAND:2896-2899). Alle Flash-Next-Werte bei 29k sind deshalb zu optimistisch.
4. **Die DSv4-Schrittzeit hängt vom Kontextfenster ab.**
   - 65k: 79–81 ms; 262k: 88–89 ms; 307k: 91 ms (STAND:2013-2016, 2183-2187).
   - Der Indexer über cuBLAS kostet bei kurzem Kontext etwa +9 ms.
   - Die 81 ms vom 20.09. früh gegenüber 85–90 ms heute sind also kein Rückschritt.
5. **Checkpoints haben gewechselt.**
   - DSv4: NVFP4 → MXFP4 am 22.09., bitgleich.
   - Flash-Next: MTPQ → nvidia am 14./15.09.
   - Die August-Werte von Flash-Next gelten für einen anderen Checkpoint und einen Stand vor moe_qpn; FNOP:94-101 nennt sie selbst „als Vergleich untauglich“.
6. **Flash-Next PP4 streut zwischen Boots um etwa ±20 %.**
   - Ursache ist der Seiten-Cache der PLE-Plattenstufe: 165k statt ~950k Major-Faults (STAND:35-41; MEM/pledisk:11-17).
   - Die erste lange Anfrage nach einem Start kostet beim Prefill etwa +8 s und beim Schritt +25–40 ms (MEM/m30:53-55). Bei DSv4 sind es 18,7 statt 16,7 s bzw. 113–115 statt 85–91 ms.
   - Prefill-A/B-Vergleiche aus nur einem Boot taugen deshalb nicht.
7. **Nebenlast am 02.10.** Die Nachmittagsläufe sind durch eigene Nebenlast um bis zu +50 % verfälscht (STAND:35-37). Die 27B-Werte vom 02.10. stammen aus diesem Fenster und wurden nicht sauber wiederholt.
8. **Boot-Zeiten folgen mehreren Faktoren.**
   - Zustand des Compile-Caches: Kaltstart ~+200 s, nach einem Code-Wechsel sind zwei Starts langsam (MEM/m30:55-58; MEM/dio:80-85).
   - Außerdem Seiten-Cache, MemoryHigh und die Messart (exakter Befehl, llama-swap oder b+g mit drei Antworten).
9. **KV-Pool.** Der Pool taugt nicht als A/B zwischen Boots (STAND:2960-2966) und ist bei einem einzigen Nutzer kein Kriterium.
10. **Annahme.** „%“ und „Tok/Runde“ sind verschiedene Maße. Beide hängen vom Prompt ab und fallen greedy höher aus als bei temp 1,0.
11. **Fehlende Daten.**
    - DSv4: Prefill zwischen 25.08. und 10.09.; Boot-Zeiten zwischen 14.09. und 21.09. nicht durchgehend.
    - Flash-Next: keine ms-Werte vor dem 22.09.; kein Prefill bei 29k vor dem 22.09.
    - Flash-Next: Der Sprung von 1.482–1.696 tok/s (30.08.) auf 413–466 tok/s (07.09.) ist in den Quellen nicht erklärt (Stack- und Promptwechsel).
    - 27B: kein Prefill vor dem 26.09.; kein Langkontext über 29k; kein Nadeltest.
    - DFlash2: keine Messung nach dem 14.09.
    - Turing-Korrektheit der RTX-Werte vom August ist nicht belegt.

## 7. Entwicklungsstände, nicht in Produktion

- **fork-next 46220f58:** Messung vom 02.10.
  - DSv4 gleich schnell wie die Produktion.
  - Flash-Next PP4: kurzer Schritt 53–63 statt 66–69 ms.
  - Die Produktionsumstellung wartete auf Peuquis Okay (STAND:16-30); umgestellt am 03.10., siehe Abschnitt 8.
- **Branch ple-disk-fast-path:** Messreihe 02.10., 22:39–23:48, Flash-Next PP4 bei 29k.

| Variante | Prefill 29k (s) | Schritt 29k (ms) | Major-Faults |
|---|---|---|---|
| v0 | 17,7–18,5 | 71–75 | 958k |
| v1 | 16,5–16,9 | 62–63 | 14,5k |
| Zeilen-Cache 0,5–4 GiB | 13,5–13,7 (erste Anfrage nach Start 24,7–26,0) | 63–64 | 6–21k |

  - Kurzer Schritt greedy: 52–57 ms.
  - Stand 02.10. nicht abgenommen (BS/ab_pledisk_2026-10-02.log:7-205; MEM/pledisk:11-34); Abnahme und Umstellung am 03.10., siehe Abschnitt 8.

## 8. Nachtrag 03.10.: Endstand, Prüflauf, Produktionsumstellung

### 8a Abschlussmessung Produktion gegen Endstand (03.10., 06:20–07:56)

Endstand = fork-next 46220f58 + PLE-Plattenstufe (c84aaaee, d0cb9bce, 138059c8) + Entwurfsliste
(98.304 Token) + Zeilen-Cache 0,5 GiB. Reihenfolge A/B/A/B; vor jedem Boot wird die PLE-Datei per
posix_fadvise aus dem Seiten-Cache genommen. Skripte BS/ab_final.sh und BS/ab_final_tp2.sh, Logs
BS/ab_final_2026-10-03.log und BS/ab_final_tp2_2026-10-03.log, Auswertung je Lauf: zweite Anfrage
jedes Paares für die kurzen Prompts, Prompts 2–4 für den Prefill (die erste lange Anfrage nach
einem Boot trägt Einmalkosten).

| Lauf | Prefill 29k, Prompts 2–4 (s) | Präfix-Folgeanfrage (s) | Schritt kurz greedy (ms) | Prosa / Code / Bearbeitung greedy (tok/s) | Decode 29k, Mittel (tok/s) | Platte gelesen (MiB) | memory.high-Treffer |
|---|---|---|---|---|---|---|---|
| PP4 Produktion A | 13,5 | 2,3 | 58–60 | 43,6 / 74,5 / 83,8 | 33,0 | 754 | 154 |
| PP4 Endstand A | 13,5–13,6 | 2,3–2,4 | 52–54 | 48,6 / 79,0 / 91,9 | 35,1 | 736 | 361 |
| PP4 Produktion B | 18,3 | 2,6 | 68 | 36,7 / 63,9 / 73,5 | 31,8 | 4.055 | 1.059 |
| PP4 Endstand B | 13,6–13,8 | 2,3–2,4 | 52–56 | 48,1 / 75,4 / 92,2 | 35,9 | 785 | 405 |
| TP2×PP2 Produktion (Wiederholung) | 18,5 | 1,9–2,0 | 49–51 | 48,8 / 83,5 / 98,9 | 38,8 | 179 | 114 |
| TP2×PP2 Endstand (Wiederholung) | 18,5–18,6 | 1,9–2,0 | 48–49 | 49,0 / 88,7 / 101,7 | 40,3 | 209 | 153 |

- Boot b+g 207–258 s; die beiden ersten Endstand-Boots 504 bzw. 526 s (neuer Compile-Hash, einmalig).
- Erste lange Anfrage nach dem Boot: Produktion 15,2 / 17,2 s, Endstand 16,4 s (Endstand A 24,5 s
  nach Neukompilierung).
- Greedy: PP4 4/4 identisch zur Referenz. TP2 Frage 3 kippt bei Zeichen 625 von 662 zwischen zwei
  Fassungen, in Produktion und Endstand gleichermaßen.
- Der erste TP2-Produktionslauf wurde um 06:45 von AIfreds Scheduler gestört (DSv4 eingewechselt,
  8 min Wartezeit) und ist deshalb oben durch die Wiederholung ersetzt; seine langen Prompts liefen
  vor der Störung (18,5 s, gleich).
- Kernaussage: Die Produktion schwankt mit dem Speicherdruck der llama-swap-cgroup (13,5–18,3 s
  Prefill, 58–68 ms Schritt), der Endstand nicht.

### 8b Determinismus und Prüflauf des Zeilen-Caches (03.10., 09:08–09:32)

Qualitätsfragen (acht Prompts, Q/<lauf>-ds.json) wortgleich zur eigenen Grundfassung:

| Variante | Läufe | wortgleich |
|---|---|---|
| Produktion PP4 (02./03.10.) | 5 | 5 von 5 |
| fork-next PP4 ohne Zeilen-Cache (02.10.) | 5 | 5 von 5 |
| PP4 mit Zeilen-Cache (c05, c1, c2, c4, Endstand A/B) | 6 | 4 von 6 |
| Produktion TP2×PP2 | 2 | 2 von 2 |
| Endstand TP2×PP2 | 2 | 5 von 8 Fragen verschieden |

Alle Antworten inhaltlich korrekt; die Abweichungen sind immer dieselben Alternativ-Formulierungen
an Fast-Gleichständen. Prüflauf BS/ab_verify.sh (Debug-Commit 34faff7a6 auf ple-disk-verify-debug,
nie gepusht): jede vom Cache-Pfad ausgelieferte Zeile wird mit frischem Direktlesen der Datei
verglichen.

| Lauf | geprüfte Zeilen | davon Cache-Treffer | Abweichungen |
|---|---|---|---|
| pp4-verifyA | 1.397.561 | 1.224.571 | 0 |
| pp4-verifyB | 1.397.493 | 1.223.580 | 0 |
| tp2-verify | 484.098 | 437.921 | 0 |

Die beiden PP4-Prüfläufe sind untereinander wortgleich, liegen aber bei drei Fragen auf den anderen
Fast-Gleichstands-Fassungen als Endstand A und fork-next — bei belegt identischen Daten. Die
wechselnden Formulierungen kommen also vom Timing bzw. vom Compile-Artefakt (vgl. FORTSCHRITT,
„Compile-Münze“), nicht vom Cache.

### 8c Produktionsumstellung (03.10., ab 09:50)

- fork-union 368e74fd → 138059c8 (reiner Fast-Forward, gepusht); alter Stand als Tag
  archive-fork-union-2026-10-03, neuer Stand als verified-2026-10-03-pledisk; fork-next ebenfalls
  auf 138059c8.
- CFG: alle 8 Flash-Next-Einträge mit draft_token_map und VLLM_PLE_DISK_ROW_CACHE_GIB=0.5; Backup
  ~/.cache/prod-switch-2026-09-29/config.yaml.vor-prod-pledisk-2026-10-03.
- Rauchtest (BS/smoke_prod_2026-10-03.log): PP4 7:40, TP2×PP2 8:43, 27B 7:49 (je einmal neu
  kompiliert), DSv4 4:22; Greedy 3/3 identisch (TP2: 2/3 + Kippfassung 018b693f), keine Fehler.

### 8d Gesamtbild vom Ausgangspunkt bis 03.10.

| Modell | Kennzahl | Ausgangspunkt | Stand 02./03.10. | Faktor |
|---|---|---|---|---|
| DSv4 PP5 | Prefill-Durchsatz (tok/s) | 119 (11.09., 64k in 9:01 min) | ~2.010 bei 61,6k (27.09.); 2.340 bei 35,9k (02.10.) | ~17–20× |
| DSv4 PP5 | Decode-Schritt (ms) | 210 bei 18k (19.09.) | 85–86 bei 35,9k (02.10.) | ~2,5× bei doppeltem Kontext |
| DSv4 PP5 | Folgefrage mit Präfix (s) | 99 (19.09.) | 0,7–1,4 (02.10.) | ~100× |
| DSv4 PP5 | Boot (min:s) | 8:07–8:39 (12./13.09.) | 4:02 (02.10.) | ~2× |
| Flash-Next | Prefill 29k (s) | 27 (22.09., TP2×PP2) | 13,5–13,8 (03.10., PP4) | 2× |
| Flash-Next | Schritt kurz greedy (ms) | 66–69 (02.10., PP4 vor der Umstellung) | 52–56 (03.10., PP4) | ~1,25× |
| Flash-Next | Schritt bei 29k, temp 1,0 (ms) | PP4 70–82, TP2×PP2 59–61 (02.10.) | PP4 61–69, TP2×PP2 56–57 (03.10.) | ~1,1× |
| Flash-Next | Boot warm (min:s) | 5:16 (10.09.) | 3:27–4:18 (03.10.) | ~1,4× |
| 27B TP2 | Prefill 14,7k (s) | 27,4–27,8 (26.09.) | 19,8–20,2 (02.10., fork-next, Nebenlast nicht ausgeschlossen) | ~1,4× |
| 27B TP2 | Schritt bei 14,6k, temp 1,0 (ms) | 48 (26.09.) | 48 (02.10.) | gleich |
| 27B TP2 | Boot (min:s) | 6:30 (07.09.) | 2:43 (02.10.) | ~2,4× |
