# Übergabe — Stand 21.10.2026, 15:55 (Projekt vllm-research)

Claude Code startet in `/home/mp/Projekte/vllm-research` (Projektregeln `CLAUDE.md`, Gedächtnis unter
`~/.claude/projects/-home-mp-Projekte-vllm-research/memory/`, AI-Connect-Peer `Mini:vllm-research`).
Betriebsstand (SSOT): `STAND.md`, Nachträge 21.10. 12:50 bis 15:40 ganz oben unter „Laufzeitumgebung“.
Vorige Übergabe (17.10.) liegt in der Git-Historie dieses Repos.

## Auftrag von Peuqui (21.10.)

- 12:47: sm_75-A/B abgebrochen. „Wir verwenden die neuen Kernel“, „wir optimieren Turing jetzt komplett fertig, dann
  testen“, „eigene Kerne für Turing, die genau optimiert sind, die wichtigsten für unsere Modelle, generisch für zukünftige
  Generationen oder auch alte“. Auf Rückfrage zur Reihenfolge: „japp, alles umsetzen“.
- Reihenfolge (mit Peuqui abgestimmt): GLM zum Laufen bringen → GLM-Profil → Volta-MoE übernehmen → FP8-QPN8-Turing-
  Durchlauf → Fused-NVFP4-Prefill für 27B. **Erst danach** die große A/B (DSv4 + 27B, ggf. GLM) gegen Produktion und
  Umstellung nur mit Peuquis Freigabe.
- Flash-Next ist eingefroren (keine Messungen), Schwerpunkt DSv4, dann GLM-5.3.

## Stand GLM-Start (letzter Lauf 15:39–15:46, nichts läuft mehr)

- **Sechster Start: Speicher passt jetzt** (GMU 0,98, MLA-Puffer gedeckelt). Budget je Stufe (Bytes aus
  `journalctl -u llama-swap | grep "GPU startup budget bytes"`): V100 PP1–PP3 Gewichte 29,9 GB, Aktivierungsspitze 1,15 GB,
  Graph-Reserve 1,15 GB, KV +1,04 bis +1,06 GB; PP0 (RTX) 45,5 GB, KV +3,02 GB; PP4 (RTX) 47,96 GB, KV +0,55 GB.
  **GPU KV cache size 41 506 Token.** Speicherspitze Boot (MiB): 0=45421 1=30582 2=47739 3=30584 4=30446.
  Ladezeit je Stufe 4:49–5:37 min.
- **Neuer Fehler (Nr. 6), danach gescheitert:** bei der Graph-Aufnahme auf PP0 (RTX) in
  `vllm/model_executor/layers/sparse_attn_indexer_kpool.py:347` `assert q_quant.dtype == torch.float16` — die quantisierte
  Query des Indexers kommt auf der RTX in einem anderen Datentyp an. Nächster Schritt: Herkunft von `q_quant` für SM75
  prüfen (Weichen nach Gerätefähigkeit, u. a. `kpool_compress.py` mit `has_device_capability(89, …)`, Indexer-Pfad
  `is_pre_ampere_cuda_platform`), Fix mit Test, dann neu starten (`glm53_boot_2026-10-21.sh text`, danach `image`).
- Config ist zurück (36110 Byte, 31 Modelle), GPUs frei, kein Skript läuft.
- Falls später doch Speicher fehlt: `ENTRY_EXTRA_ENV="VLLM_V2_CUDAGRAPH_MEM_MIB=…"` (SM70-V2 reserviert für Graph-Pools die
  ganze Aktivierungsspitze, `get_sm70_cudagraph_memory_reserve`) oder Weg C (MXFP8-Attention als FP8, ~5 GB).
- Peer-Status steht noch auf „GLM-5.3 erster Start … alle GPUs belegt“: bei Pause zurücksetzen und das Fensterende melden
  (Mini:ViroSpore war heute aktiv; Mini:AIfred-Intelligence wartet auf die GLM-Speicherzahlen mit und ohne Bildteil).

## Zweige und Bäume

| Baum | Zweig / Stand | Inhalt |
|---|---|---|
| `1Cat-vLLM-work` | `prod-fork-main` = 79e07bfef | **Produktion, unverändert** (Tag `verified-2026-10-21-moegrid`) |
| `1Cat-vLLM-fork-main` (Baubaum) | `turing-kernels-2026-10-21` = 7ee89582f, gepusht | 79e07bfef + sm_75-Bau (2f70712b0, 5028a3fc4) + Turing-MoE (039c1b6af) + Turing-Prefill-Attention (f9ac81b6b) + Volta-MoE NT2/Split 8 (7ee89582f) |
| `1Cat-vLLM-sm75` (Prüfbaum für fm-Boots) | detached 7ee89582f, `_C` c869d134c3ee | Bau per `build_turing_2026-10-21.sh <commit>` |
| `1Cat-vLLM-glm53` | `glm53-turing-2026-10-21` = 0cac9351d, gepusht | Merge glm53-flash-fp4local auf f9ac81b6b (1ddebbbd7) + sechs GLM-Fixes; **noch ohne 7ee89582f**, `_C` 62375240caca (Stand f9ac81b6b, passt zu seinem Python mit `nacc`) |

- Beim Nachziehen von 7ee89582f in den GLM-Zweig ändert sich die Op-Signatur (`nacc` entfällt) und
  `grouped_splitk(k, order)` (ohne `reduces_hidden`); danach `_C` aus `1Cat-vLLM-fork-main/vllm/*.so` in den GLM-Baum
  kopieren. Der GLM-Zweig ändert kein C++.
- **Für 7ee89582f laufen die MoE-Tests noch aus** (`run_pytest_per_file.sh -g 0,1 … 1Cat-vLLM-sm75
  tests/kernels/moe/test_skinny_sm70_moe.py`). Der Stand davor (039c1b6af/f9ac81b6b) war grün: 136 bestanden, 13
  übersprungen; Attention-Test 4/4.

## Ergebnisse heute (alle gemessen, Dateien genannt)

**Turing-MoE m16n8k8** (`skinny_nvfp4_moe_m16n8k8`, Gewichtslayout je Karte über `qpn_fragment_order`, gemeinsamer
Header `csrc/sm70_turbomind/ops/mma_fragments_sm70.cuh`): RTX µs je Aufruf alt → neu (`scratchpad/turing/final_rtx.log`):
DSv4-512 w13 6555 → 4303, w2 3271 → 2238; GLM-512 8566 → 5316, 4205 → 2760; Flash-Next-748 −40/−44 %; Decode ±1 %.
RTX im MoE-Prefill jetzt 7–33 % schneller als V100. Alte Ordnung bitgleich zum Prod-`_C`, Turing-Ordnung ≤ 6,3e-4 vom
Ausgabemaximum. Konfiguration: Split 8 (K < 2048: 4), NT 2, RB 4 Prefill / 1 Decode, eine Gruppe vorausladen.

**Turing-Prefill-Attention** d=256 (`paged_prefill_attention_sm75.cu`, Weiche `_use_turing_prefill` im FLASH_ATTN-Backend
ab `flash_attn_turing_prefill_min_query_len` = 65): 128 GQA-gepackte Zeilen je CTA, Q in Registern, K/V doppelt gepuffert,
32-Bit-Offsets, Seitengröße als Template. 27B-Ranggrößen, ms FA2 → eigen: 2048 Token 150k 165,8 → 61,5 (61 TFLOPS),
512 Token 52,5 → 20,0; bis 64 Token bleibt FA2. Fehler gegen fp32 gleich FA2. Diagnose per ncu: L2-Treffer 98 %, nicht
bandbreiten-, sondern latenzgebunden (`scratchpad/turing/attn/`).

**Volta-MoE** (7ee89582f): V100 µs alt → neu (`volta2_v100.log`): DSv4-512 w13 4979 → 4374, w2 2441 → 2395; GLM-512
6878 → 5954, 3567 → 3218; Decode 0,80–0,97.

**Profile** (`nsys_top.py` im Scratchpad):
- DSv4 fm 150k (`nsys/ds-stages-turing-1021`): Prefill 55,7 s unter nsys (gestern 90,7 s, inkl. Indexer/Softmax/Grid),
  alle fünf Stufen 53–55 s ausgeglichen, Aufteilung 10,8,8,8,9 bleibt optimal. RTX-Rest: FP8-QPN8 10 %, cuBLAS 20 %
  (nativ), Indexer-Epilog 5 %, HC-Prenorm 5 %, gather_rows 4 %.
- 27B fm 150k (`nsys/q27-stages-turing-1021`): 267,9 → 207,9 s (−22 %), Attention 110 → 46,8 s; jetzt Allreduce 46 %,
  cuBLAS 20,6 %, NVFP4-Entpacken 4,8 %, FP8-Entpacken 1,2 %.
- sm_75-Bau allein: DSv4 neutral (≤ 0,2 %, Texte bytegleich) — ist aber Voraussetzung der Turing-Kerne.

## GLM-5.3-Flash: Fehlerkette (alle mit Test, auf `glm53-turing-2026-10-21`)

1. 3856a7b67 MLA-Prefill-Auswahl band die Flash-V100-Laufzeit vor dem Binden (1Cat-Upstream-Code, `get_flash_dense_forward`).
2. 434b350c2 Direct-I/O kannte safetensors-`U16`/`U32` nicht (CSF-Seitenbasen/Offsets).
3. 5cc9779e1 CSF-Skalenteile wurden nach dem Umschreiben des Namens (`experts.N.down_proj.` → `experts.w2_`) nicht erkannt.
4. 08fbcdf68 Ausgenommener/unquantisierter LM-Head bekam `UnquantizedLinearMethod` (braucht im Fork LinearBase-Attribute)
   → jetzt `None`, LM-Head nutzt `UnquantizedEmbeddingMethod` wie bei compressed-tensors; upstream-Test angepasst.
5. 7104b7d46 + 0cac9351d MLA-Kontextpuffer: „8 volle Anfragen“ bzw. „4 Seiten“ bei Hybrid-Seiten 8576 Token → 2,1 GiB
   Profilierungspuffer je Stufe (Speicherschnappschuss Stufe 0: ein 2144-MiB-Block); jetzt ≤ max_num_seqs × max_model_len.
- Vorbestehend rot (schon vor dem Merge identisch): `test_pp.py` (2), `test_sm70_glm53_modelopt_nvfp4.py` (1),
  `test_sm70_modelopt_nvfp4.py` (3). GLM-Testliste: `~/.cache/bench-scripts/glm53_testfiles_2026-10-21.txt` (23 Dateien).
- `test_compressed_tensors.py` (nur zur Gegenprobe gelaufen): 15 rot in 8 min — noch nicht geprüft, ob vorbestehend
  (lädt echte Modelle); vor einem PR klären.
- Speicherbilanz vor Fix 5 (GB, dezimal): V100-Stufen Gewichte 29,97 von 33,05, KV −1,65; PP4 47,96 von 49,23, KV −1,14;
  PP0 45,51, KV +1,34. Ladezeit je Stufe 3:46–4:48 min.

## Offene Arbeit (Reihenfolge)

1. GLM-Start zum Erfolg bringen (siehe oben), Speicher mit/ohne Bildteil an AIfred-Intelligence (Englisch nicht nötig, Peer;
   Zahlen aus dem Log), danach nsys-Profil GLM (`profile_glm53_stages.sh` von 20.10. ist nsys-Variante des Eintrags) und
   GLM-eigene Turing-Kerne nach Rangliste.
2. 7ee89582f-Tests laufen lassen, GLM-Zweig nachziehen.
3. FP8-QPN8-Turing-Kern (`fp8_qpn8_mma_sm75.cu`) wie den MoE per Variantendurchlauf tunen (10 % der DSv4-RTX-Stufe).
4. Fused NVFP4-Prefill-GEMM für 27B statt Entpacken + cuBLAS (Vorlage `fp8_qpn8_mma_sm75.cu`).
5. Kleinere Turing-Posten (Indexer-Epilog, HC-Prenorm TileLang, gather_rows); QPN2-Decode-Kerne auf Turing-Ordnung nur bei
   Bedarf (Decode an der Bandbreitengrenze).
6. Danach große A/B (DSv4 + 27B, mit Langprompts bis Max-Kontext, memory.used je Karte) Prod 79e07bfef gegen
   `turing-kernels`; Texte sind auf den RTX nicht bytegleich (Summationsreihenfolge) → Qualitätsfragen lesen; Umstellung nur
   mit Freigabe.
7. 27B-Allreduce (46 % Prefill) ist Kommunikation — eigener Strang (Gedächtnis: Überlappung, 2-Rang-Host-Allreduce).
8. Upstream-Angebote an 1Cat nur auf Peuquis Ansage: Flash-V100-Verfügbarkeitsprüfung, ModelOpt-LM-Head, MLA-Puffer,
   Direct-I/O U16/U32, Turing-Kerne.

## Werkzeuge und Fallen von heute

- Mikrotests: `scratchpad/turing/moe_turing_bench.py` + `moe_bind.cu` (bindet die echte Quelldatei des Baubaums; Umgebung
  `SWEEP=decode|prefill`, `VOLTA_SWEEP=1`; Anker = `_C` des Baubaums), `scratchpad/turing/attn/attn_bench.py`
  (`CHECK=1`, `CHUNKS=…`), `scratchpad/turing/nsys_top.py`, `scratchpad/glm/memsnap_peak.py`
  (Scratchpad = `/tmp/claude-1000/-home-mp-Projekte-vllm-research/508c7d38-9687-45d1-9c5d-79a5efeb3358/scratchpad`).
- Bau: `~/.cache/bench-scripts/build_turing_2026-10-21.sh <commit>` (Baubaum muss sauber auf dem Commit stehen, setzt
  den Prüfbaum `1Cat-vLLM-sm75` und kopiert die `.so`). fm-Profil: `profile_fm_2026-10-21.sh <Prod-Eintrag> <tag>`.
- GLM-Boot: `glm53_boot_2026-10-21.sh text|image` mit `flock`, eindeutigem Backup, Signal-Trap, Abbruch ohne
  Qualitätsfragen bei gescheitertem Boot, `ENTRY_EXTRA_ENV` für Zusatzumgebung.
- **Config-Zwischenfall 14:45:** zwei Bootskripte gleichzeitig → llama-swap-Config 52 s leer (Gedächtnis
  `feedback_config_scripts_one_at_a_time`). Folgelauf erst nach Prozessende, nie nach Journal-Fehler.
- llama-swap hat `PrivateTmp=yes`, schreibbar u. a. `~/.cache` — Diagnosepfade dorthin.
- bash führt einen Trap erst nach dem laufenden Kindprozess aus: hängendes `curl` gezielt beenden.
- Jede Qualitätsanfrage an ein gescheitertes Modell startet es in llama-swap neu (5 min GPU je Versuch).
- Register-Deckel: `__launch_bounds__(512)` → 128 Register; Laderinge mit Laufzeitindex wandern in lokalen Speicher;
  Schleifen mit von äußeren Variablen abhängigen Grenzen nicht voll ausgerollt → Fragmente im lokalen Speicher (SASS auf
  `LDL/STL` prüfen, `cuobjdump -res-usage`).
