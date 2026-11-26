<!-- 24.11.2026: Prüfbericht eines Sonnet-Agenten (nur gelesen), Kernaussagen vom Koordinator nachgeprüft: Bedingung _keep_volta_turbomind_copy (fp8_block.py:641, _QPN8_MAX_M = 8; K7 1×8 = 8 greift nicht), num_speculative_tokens je Eintrag, 27B-KV 28,91 GiB, DeepSeek Stufe 0 8,68 GiB. -->

# Bericht: VRAM für Tempo in den vier vLLM-Produktionseinträgen (Mini), Stand Baum v15, Boots vom 24.11.2026

Es wurde nur gelesen (Code, Config, Journal, Safetensors-Header). Es gab keine Boots, keine GPU-Läufe und keine Änderungen.
Pfade: Baum `/home/mp/Projekte/vllm-research/1Cat-vLLM-v15/vllm/…` (im Folgenden `v15/`), Config `/home/mp/.config/llama-swap/config.yaml`.
Belege: `journalctl -u llama-swap` (Boots 00:29 GLM, 00:54 DeepSeek, 01:10 K7, 01:23 27B) und `~/.cache/bench-scripts/validate_v15_*_2026-11-24.log`.

## Kurzantwort

1. **Die zwei im Auftrag genannten Hauptverdächtigen greifen in keinem der vier Einträge.**
   - Das ist die TurboMind-Zweitkopie `_keep_volta_turbomind_copy` und der dauerhafte FP16-Prefill-Arbeitsbereich (`_ensure_block_qpn8_prefill_workspace`).
   - Belege stehen in den Zeilen 1 und 2 der Tabelle.
2. **Es gibt genau eine Stelle, die belegt Kontext für Tempo kostet: das eingeschränkte Entwurfsvokabular des 27B.**
   - Gemessen: KV 29,78 → 28,92 GiB je RTX, Pool 876 615 → 851 394 Token (−0,86 GiB, −25 221 Token, −2,9 %).
   - Dafür gibt es Decode kurz +4,6 % (`v100-skinny/STAND.md:95`).
   - Der Eintrag läuft heute noch so: 28,91 GiB und 851 394 Token im Kontrolllauf `validate_v15_fp8rtx_2026-11-24.log`.
3. **Sonst gibt es nur kleine Tempo-Zweitformen.**
   - Auf den V100-Stufen von DeepSeek und K7 sind das gepackte FP13-Gewichte, berechnet ca. 106 MiB je Stufe.
   - Bei GLM sind es dichte K/V-Zeilen im Entwurf (80 MiB), PP-Pack-Puffer (≤ 3,6 MiB je Karte) und die Graphen.
   - Alle liegen auf Stufen, die den Kontext nicht begrenzen, oder sind vernachlässigbar.
4. **Die Entwurfsmodelle sind die großen Posten, aber das ist die Spekulation selbst.**
   - DSpark belegt ca. 11 GiB auf RTX GPU 2 (DeepSeek, K7).
   - Der GLM-Entwurf belegt 1,2 GiB.
   - Die MTP-Schicht des 27B belegt ca. 0,2 GiB je Rang.
5. **Wo der Kontext wirklich begrenzt wird (gemessen):**
   - DeepSeek: Stufe 4 (RTX, GPU 2).
   - K7: Stufe 4 (RTX, GPU 2), mit nur 0,22 GiB Luft.
   - GLM: Stufe 0 (RTX, GPU 0).
   - 27B: beide RTX, der Pool nimmt alles.
   - Zahlen im Abschnitt „Kontext-Bindung“.

## Die Stellen (Nr. = Zeile)

Spalte „greift“ steht für DeepSeek / K7 / GLM / 27B: J = ja, N = nein.
Das Journal loggt `info_once` nur auf dem lokalen Rang 0, das ist die RTX-Stufe (`v15/logger.py:150-155`, `_should_log_with_scope`). Meldungen der V100-Stufen fehlen deshalb im Journal. Für die V100-Stufen stützen sich die Belege auf Code, Config, geladene Größen und die TurboMind-Warmup-Zeile.

| Nr | Stelle | Was gehalten wird | greift | Größe | Zweck | abschaltbar |
|---|---|---|---|---|---|---|
| 1 | `v15/model_executor/kernels/linear/qpn/fp8_block.py:632` (Kopie anlegen: 696-712, Aufrufstelle 700) | Zweite TurboMind-Kopie jedes Block-FP8-Gewichts, nur Volta | N/N/N/N | Wäre bei DeepSeek ca. 1,0 GiB je V100-Stufe (berechnet: FP8-Dichte 1040 MiB je 8 Schichten aus dem Checkpoint-Header). Realisiert: 0 | Tempo für M>8 | Ja: `kernel_config.sm70_fp8.block_qpn8_volta_turbomind_prefill=false` |
| 2 | `fp8_block.py:69` (`_ensure_block_qpn8_prefill_workspace`), `:103` (`_graphs_capture_dense_prefill`), `:719/:738` (Aufruf) | Dauerhafter FP16-Arbeitsbereich K×N, gemeinsam je Gerät | N/N/N/N | Wäre je Schicht-Form K×N×2 B, z. B. 27B gate_up je Rang 17408×5120×2 = 170 MiB (berechnet). Realisiert: 0 | Tempo/Graphen | Ja, nur durch die Bedingung (siehe unten) |
| 3 | `fp8_block.py:127` (`_block_qpn8_prefill_dense`) | FP16-Gewichtspuffer je Aufruf für Prefill M>128 bzw. >512 (RTX), nur während des Aufrufs | J/J/J/J (GLM über MXFP8, `pre_ampere.py:165-190`, `x.new_empty((K,N))` je Aufruf) | Teil der Aktivierungsspitze, siehe Nr. 23 | Prefill-Tempo | Nein (Code) |
| 4 | `v15/model_executor/layers/quantization/fp8.py:497-520` | DeepSeek-`wo_a` (gruppierte BMM) wird auf Turing einmal nach FP16 dequantisiert und zusätzlich gehalten | J/J/N/N (nur die RTX-Stufen 0 und 4). Beleg: Journalzeile `fp8.py:520` „grouped BMM uses the existing dequantized grouped implementation“ | +32 MiB je Schicht (FP16 64 gegen FP8 32 MiB). Stufe 0: 10 Schichten = 320 MiB. Stufe 4: 9 Schichten + 3 DSpark-Schichten = 384 MiB (berechnet; DSpark-Schichten nehmen dieselbe Attention-Klasse an, nicht geprüft) | Pflicht: Marlin/QPN8 können kein gruppiertes BMM | Nein |
| 5 | `v15/model_executor/kernels/linear/qpn/fp8.py:~640-766` (BMM-Zweig; Zeiger-Tabellen `:727-760`) | TurboMind-gepacktes `wo_a` (nur eine Form) und kleine Zeiger-Tabellen für Decode | J/J/N/N (V100-Stufen 1-3). Beleg: TurboMind-Warmup „1 FP8 dense layer shapes“ auf V100 | 256 MiB je Stufe (8×32 MiB), gleich groß wie FP8, kein Zusatz (berechnet). Zeigertabellen Bytes | Pflicht/Format | Nein |
| 6 | `v15/models/deepseek_v4/sm70/gemv.py:267-297` (Aufruf `model_executor/layers/linear.py:367-374`; Flaggen `gate_linear.py:60`, `attention.py:957`, `compressor.py:306`) | Gepackte FP13-Gewichte neben dem FP16-Gewicht: Router-Gate, indexer `weights_proj`, Kompressor und Indexer-Kompressor `wkv/wgate` (K=4096, N∈{64,256,512,1024,2048}) | J/J/N/N, nur V100-Stufen (Gate `gemv.py:271-273` `is_device_capability((7,0))`). Beleg: Journal (attention.py:447, 23:50:53) „SM70 exact FP16 auxiliary fusion: existing_fp13_route_preserved“ belegt `dsv4_fp13_gemv=True`; Default `envs.py:399` True | ca. 105,6 MiB je V100-Stufe (berechnet: 6656 B je Zeile × (4 Schichten à 2880 Zeilen + 4 Schichten à 1280 Zeilen)) | Decode-Tempo (GEMV) | Ja: `VLLM_SM70_DSV4_FP13_GEMV=0` |
| 7 | `v15/models/deepseek_v4/attention.py:347-380` mit `gemv.py:~325-355` | Gefusstes FP16-Aux-Gewicht (Kopie der drei Projektionen) | N/N/N/N: `reason = existing_fp13_route_preserved`, Puffer bleibt `None` | 0 | Tempo | `kernel_config.fused_fp16_aux_gemv` (Default True, `config/kernel.py:1021`), hier durch Nr. 6 verdrängt |
| 8 | `v15/model_executor/kernels/lm_head/sm70.py:104-137,181-312,392-437` | Gepackte oder QPN8-Zweitform des LM-Heads | N/N/N/N: setzt Volta (7,0) und FP16-Kopf voraus. DeepSeek/K7: Kopf liegt auf RTX GPU 2 (Stufe 4). GLM: Kopf MXFP8 und `VLLM_SM70_LM_HEAD_TOP1=0` (Journal „Auto-setting“). 27B: RTX | 0 (Wäre ca. 1 GiB) | Tempo | Ja (Env) |
| 9 | `v15/model_executor/kernels/linear/sm70_dense.py:133` (`prepare_dense`), `layers/linear.py:331-390`, Markierungen `qwen3_5.py:349`, `qwen3_5_mtp.py:98`, `qwen3_dflash2.py:503` | TurboMind-FP16-Zweitkopie kleiner dichter Projektionen | N/N/N/N: `dense_f16` Default 0 (`envs.py:6070`), `force_enable` nur Qwen/MTP (RTX, Gate `sm70_dense.py:~150` Kapazität (7,0)) | 0 | Tempo | Env/Config |
| 10 | `v15/model_executor/layers/quantization/sm70_dflash2_fp16.py:24-60`, `sm70_dflash2_fp8.py:69-100` | Gepackte FP16-/FP8-Zweitformen der DFlash2-Entwurfsprojektionen | N/N/N/N: nur TP 2/4 (`qwen3_dflash2.py:~290`), GLM ist TP1 PP5 | 0 | Tempo | Config |
| 11 | `v15/model_executor/layers/quantization/sm70_mtp_draft_qpn8.py:122-133` | Requantisierte MTP-Entwurfsprojektionen (ersetzt FP16, kein Zusatz) | N/N/N/N: `is_device_capability(70)` falsch auf RTX | 0 | Entwurfstempo | `kernel_config.sm70_mtp_draft_qpn8` (Default True) |
| 12 | `v15/model_executor/models/qwen3_5_mtp.py:~700-725`, `qwen3_5.py:1100-1106`, `qwen3_5_mtp_refined_head.py:431`, `qwen3_5_mtp_bounded_head.py:873`, `config/kernel.py:1040-1077` | Heißer Entwurfskopf, „refined head“ (+0,64 GB), „bounded target head“ (+0,32 GB je Rang) | N/N/N/N: Journal „hot vocabulary 0“. Defaults hängen an `prefer_native_mtp_runner` (nur Kapazität (7,0), `runtime_defaults.py:523-541`), und beide Köpfe prüfen zusätzlich `is_device_capability(70)` | 0 | Entwurfstempo | `sm70_*` Config |
| 13 | `v15/model_executor/layers/quantization/sm70_online_qpn8.py:~255-330` (Qwen4Exp) | Online-QPN8-Zweitform | N/N/N/N: nur Qwen4Exp TP4 | 0 | Tempo | Env |
| 14 | `v15/v1/worker/gpu/spec_decode/draft_vocab.py:96/157-171` (`DraftVocabHead`, `index_select` + `cat`), Vokabular-Datei in der `--speculative-config` des 27B | Eigene Kopie der LM-Head-Zeilen des 98304-Entwurfsvokabulars je TP-Rang | **N/N/N/J**. Journal 01:22:35: „Draft vocabulary: 98304 of 248320 tokens, 89566 lm_head rows per TP rank“ | 89566×5120×2 B = 874,7 MiB = 0,854 GiB je RTX (berechnet). **Gemessen**: Kontext-Wirkung −0,86 GiB, −25 221 Token (A/B `ab_dv98kv5d_fp8rtx_2026-11-08a`, journal-4-T gegen journal-5-P) | Entwurfstempo (+4,6 % Decode kurz) | Ja: den Schlüssel `draft_token_map` aus der `--speculative-config` des 27B entfernen |
| 15 | `v15/models/deepseek_v4/nvidia/dspark.py:140-220`, `model_executor/models/shared_weights.py:~96-104` | DSpark-Entwurf (3 Schichten mit je 256 MXFP4-Experten, Markov-Köpfe) plus eigene Entwurfs-Einbettung auf Stufe 4 | J/J/N/N. Journal 23:51:21 „draft model's vocab embedding will be loaded separately“ | Roh 10,12 GiB (mtp.*, berechnet aus Safetensors-Header) + 0,99 GiB Einbettung. Stufe 4 lädt gemessen 42,98 GiB (V100-Stufen 27,03, Stufe 0 35,11). Entwurfs-KV steckt im Pool von Stufe 4 (0,94 gegen 0,75 GiB der anderen, gemessen, Anteil nicht getrennt) | Spekulation (Tempo) | Nein (Eintrag-Eigenschaft) |
| 16 | GLM-Entwurfsmodell `/home/mp/models/GLM-5.3-Flash-DFlash2-Drafter-1.2B-FP8-Block128` (`speculative-config` im GLM-Eintrag) | DFlash2-Entwurf, FP8 Block128 | N/N/J/N | 1,205 GiB roh (berechnet). Stufe 4 lädt gemessen 40,64 GiB gegen berechnet 38,12 GiB für Schichten und Kopf | Spekulation | Nein |
| 17 | `v15/model_executor/models/qwen3_dflash.py:634-651` (`_fuse_dense_kv_weight`), geerbt von `qwen3_dflash2.py`, aufgerufen aus `v1/worker/gpu/spec_decode/dflash/speculator.py:680/855` | Dichter FP16-Stapel aller K/V-Zeilen der 5 Entwurfsschichten (der Entwurf ist FP8, daher keine „dichte“ Gewichtsform im Layer) | N/N/J/N, zur Laufzeit beim ersten Kontext-Aufruf | 2048 Zeilen × 4096 × 2 B × 5 Schichten = 80 MiB (berechnet). Zeitpunkt (vor/nach KV-Bemessung) nicht geklärt | Entwurfstempo (eine GEMM für alle Schichten) | Nein (Code) |
| 18 | `v15/model_executor/models/qwen3_5_mtp.py` (MTP-Schicht FP16, TP2) | Entwurfsschicht und `fc` des 27B | N/N/N/J | ca. 0,22 GiB je Rang (grob berechnet: Vollattention-Schicht 0,347 + fc 0,098 GiB, halbiert) | Spekulation | Nein |
| 19 | `v15/v1/worker/gpu/pp_pack.py:123-170` | Gepackte PP-Übergabeblöcke je Zeilenzahl 2..7 | N/N/J/N (`pp_pack_handover` nur im GLM-Eintrag) | Journal 00:32:33 „nbytes“: höchstens 3,77 MB = 3,6 MiB je Karte (Rang 3, Senden + Empfangen); Rang 0 1,27 MiB (gemessen) | Übergabetempo | Ja: `pp_pack_handover` aus dem `--kernel-config` |
| 20 | Standard-PP-Übergabepuffer, `topk_indices_buffer` (`deepseek_v4/nvidia/dspark.py:151`, `model.py:1017`), `compressor.private_state` (`compressor.py:180`), KV-Zeroer-Tabellen (`v1/worker/utils.py:~800-830`) | Kleine Hilfspuffer | J/J/J/J (27B ohne PP) | < 1 MiB bis wenige MiB (berechnet, nicht einzeln gemessen) | Pflicht/Graphen | Nein |
| 21 | `v15/v1/worker/gpu/cudagraph_utils.py:~688-727` | Graph-Pools | J/J/J/J | Gemessen „Graph capturing took“: DeepSeek 0,15 (Stufe 0) / 0,23 (V100) / 0,20 (Stufe 4) GiB, 16 Kontext-Buckets. K7 0,14/0,21/0,18, 13 Buckets. GLM 0,09/0,08/0,13 (Tail-Graphen 7..2). 27B 0,11 | Tempo (Graphen) | Capture-Größen und Buckets in `--compilation-config` |
| 22 | `v15/v1/worker/gpu_worker.py:~857-870` mit `v1/worker/gpu/cudagraph_utils.py:110-123` | Graph-Reserve, vom KV-Budget abgezogen (V100: Aktivierungsspitze statt echter Pool) | J/J/J/N (nur V100; RTX `graph=0`) | DeepSeek/K7 0,305 GiB gegen tatsächlich 0,23/0,21. GLM 0,64 GiB gegen tatsächlich 0,08: 0,56 GiB zu viel reserviert je V100-Stufe (gemessen). Alle diese Stufen sind nicht bindend (Abschnitt unten) | Sicherheits-/Tempo-Reserve | Env `VLLM_V2_CUDAGRAPH_MEM_MIB` |
| 23 | `--max-num-batched-tokens` (1024; 27B 2048) und `activation_peak` | Prefill-Aktivierungsspitze | J/J/J/J | Gemessen: DeepSeek/K7 V100 327 321 600 B (312 MiB), Stufe 4 377 653 248 B (360 MiB). GLM V100 ca. 689 MB (657 MiB), Stufe 0 754 550 784 B (720 MiB). 27B 413 403 136 B (394 MiB) | Prefill-Tempo | Ja (Parameter) |
| 24 | `warmup_live` | Torch-Speicher zwischen Laden und Profil, Zusammensetzung unklar | J/J/J/J | Gemessen: DeepSeek V100 930 571 776 B (887 MiB), Stufe 0 861 365 248 B (821 MiB), Stufe 4 912 371 712 B (870 MiB). K7 620 747 264 / 551 540 736 / 602 579 456 B. GLM V100 60-76 MB, Stufe 0 19 960 320 B, Stufe 4 187 242 496 B. 27B 55 123 968 B | unbekannt | **nicht geklärt**, siehe unten |
| 25 | `non_torch` | C++-Speicher (u. a. TurboMind) | J/J/J/J | Gemessen: DeepSeek/K7 V100 54 525 952 B (52 MiB), RTX 14-17 MB. GLM V100 16 777 216 B. Mehrbetrag 36 MiB nur bei DeepSeek/K7 V100 (Zuordnung zu TurboMind-Tuning vermutet, nicht belegt) | unklar | **nicht geklärt** |
| 26 | `v15/distributed/device_communicators/host_reduce.py:~120-190`, `config/kernel.py:736` (`copy_engine_max_bytes` 20 MiB im 27B) | Host-Reduce zwischen den beiden RTX | N/N/N/J | Vier Slots à 20 MiB im gepinnten **Host**-Speicher (`kernel.py:736-742`). VRAM nur das Ausgabe-`empty_like` je Aufruf (≤ 20 MiB, eager) | Prefill-Allreduce | Ja: `host_reduce.copy_engine_max_bytes=0` |
| 27 | `v15/v1/worker/gpu/prefill_overlap.py` (27B `tp_prefill_overlap_min_tokens` 1024) | Zwei Hälften auf zwei Streams | N/N/N/J | Kein dauerhafter Puffer | Prefill-Tempo | Ja (Config) |
| 28 | `v15/model_executor/layers/fused_moe/experts/skinny_sm70_moe.py:507-558` | Skinny-MoE, Experten und Skalen **in place** umgeordnet, keine Kopie im Checkpoint-Layout; Skalen auf MXFP4 gefaltet bzw. als Bitebenen; Aufrufe je Chunk ≤ 128 MiB | J/J/J/N | Keine zweite Form (Docstring `skinny_sm70_moe.py:509-515`; geladene Größen Nr. 36) | Format | Nein |
| 29 | `v15/model_executor/kernels/linear/qpn/pre_ampere.py:165-200` + `sm70_turbomind.py:790-860` | MXFP8-Dichte (GLM) als QPN8-Codes, Rohgewicht und Skala werden durch leere Tensoren ersetzt | N/N/J/N | Einzelform (Code `pre_ampere.py:~190`) | Format | Nein |
| 30 | `v15/model_executor/layers/fused_moe/sm70/workspace.py:19-130` (`Fp8MoEWorkspace`) | Dauerpuffer für FP8-MoE | N/N/N/N: Experten sind MXFP4/NVFP4 | 0 | Pflicht | – |
| 31 | `v15/v1/attention/backends/mla/flashmla_sparse.py:344-390`, `indexer.py:469-540` (Metadaten-Puffer, auch C128A-Tabellen mit Breite aus `max_model_len`) | Persistente Puffer für Graphenstabilität | DeepSeek/K7 wahrscheinlich, GLM/27B anders | Je Builder-Instanz: C128A zwei Tabellen 1024×3968×4 B = 32,5 MiB bei 500k, ca. 19 MiB bei 307k; Blocktabelle 1024×1954×4 B = 7,6 MiB (berechnet) | Graphen | **nicht geklärt**: Zuordnung zur SM70-Route nicht belegt |
| 32 | Env `VLLM_SM70_INDEXER_PREFILL_TILE_MB=64`, `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64` (`sm70_sparse.py:37`: Default Kachel 192; `envs.py:220`: Default Logits 512) | Deckelt Indexer-Arbeitsspeicher | J/J/N/N | **spart** Speicher (je Aufruf 64 statt 192 bzw. 512 MiB) | Speicher sparen | Env |
| 33 | `--cpu-offload-gb 1.4 --cpu-offload-params embed_tokens` im GLM-Eintrag | Einbettung (UVA) im Host | N/N/J/N | spart 1,18 GiB VRAM (Journal „UVA offload … (1.18)“) | Speicher sparen | Parameter |
| 34 | `NCCL_BUFFSIZE=1048576` im 27B, `PYTORCH_ALLOC_CONF=expandable_segments:True` (DeepSeek, K7, GLM) | NCCL-Puffer klein, weniger Fragmentierung | N/N/N/J bzw. J/J/J/N | spart Speicher | Speicher sparen | Env |
| 35 | `v15/config/kernel.py:1779-1784` | Setzt `block_qpn8_volta_turbomind_prefill=False`, wenn Quantisierung `fp8`/`compressed-tensors` und Op `fp8_qpn8_block_native_sm70_out` vorhanden (Op-Vorhandensein nicht geprüft; `grep` im `_C.abi3.so` fand den Namen nicht) | zusätzliche Absicherung zu Nr. 1 | – | – | – |
| 36 | Gewichtsbilanz je Stufe (Header gegen Journal) | Test auf versteckte Gewichtskopien | alle vier | DeepSeek V100: roh 26,667 → geladen 27,03 GiB (+0,36; eine TurboMind-Kopie wäre ≥ +1,016). K7 gleich. GLM V100: roh 26,25 → 26,91-26,92 (+0,66). 27B je Rang: 14,37 (halbe Rohgröße) + 0,85 (Nr. 14) + ca. 0,22 (Nr. 18) = 15,44 gegen geladen 15,54 GiB (+0,10) | Kontrolle | – |

### Belege für Nr. 1 und Nr. 2 im Einzelnen (greift nicht)

- **Nr. 1, Bedingung** (`fp8_block.py:632-668`): Die Kopie entsteht nur, wenn `max_num_seqs × (1 + num_speculative_tokens) > 8` und mindestens 18 GiB frei sind.
  - DeepSeek: 1 × (1+5) = 6.
  - K7: 1 × (1+7) = **8**, exakt an der Kante (`<= _QPN8_MAX_M`).
  - GLM: 1 × (1+6) = 7.
  - 27B: 3 × (1+3) = 12, würde zutreffen, läuft aber nur auf RTX (Zeile 698 verlangt Kapazität 70 oder 72).
- **Nr. 1, Laufzeitbeleg:** TurboMind-Warmup auf den V100-Stufen (`awq_sm70_warmup.py:421-426` überspringt Block-QPN8-Schichten ohne Kopie) nennt bei DeepSeek (15 Zeilen) und K7 (3 Zeilen) „1 FP8 dense layer shapes“. Das ist die `wo_a`-BMM. Bei Kopien stünden dort mindestens fünf Formen (wq_b, wo_b, wq_a, wkv, Shared-Experten).
- **Nr. 1, Gewichtsbilanz:** Siehe Nr. 36.
- **Nr. 2:** `_graphs_capture_dense_prefill` (`fp8_block.py:103-124`) hält den Arbeitsbereich nur, wenn eine Capture-Größe über dem Schwellwert liegt. Der Schwellwert ist 512 auf RTX und 8 auf V100.
  - DeepSeek [5,6], K7 [7,8], GLM [7] und 27B [1,2,4,8] liegen alle darunter.
  - `kv_cache_memory_bytes` ist nicht gesetzt (Default None, Journal: KV aus dem Budget berechnet).

## Kontext-Bindung (gemessen, Journal der Prod-Boots)

„Verfügbar“ = „Available KV cache memory per worker“, „benötigt“ = „KV pool … GiB“ je Stufe. Stufen: 0 = RTX GPU 0, 1-3 = V100, 4 = RTX GPU 2.

| Eintrag | verfügbar (GiB) Stufe 0/1/2/3/4 | benötigt (GiB) | bindende Stufe | Umrechnung |
|---|---|---|---|---|
| DeepSeek | 8,68 / 1,59 / 1,59 / 1,59 / 0,71 | 0,75 ×4 / 0,94 | Stufe 4 (0,71 < 0,94). Der Pool ist per `--num-gpu-blocks-override 4220` festgelegt; ohne Override ergäben sich dort 3182 Blöcke = ca. 384k Token. Die 0,23 GiB Überhang kommen aus dem Rest (frei 2140 MiB) | 100 MiB auf Stufe 4 ≈ 53k Token (0,94 GiB für 509 292 Token) |
| K7 | 8,96 / 1,88 ×3 / 1,00 | 0,62 ×4 / 0,78 | Stufe 4 mit 0,22 GiB Luft | 100 MiB ≈ 40k Token (0,78 GiB für 317 261 Token) |
| GLM | 1,80 / 2,57 / 2,57 / 2,58 / 4,46 | 1,52 / 1,01 / 1,01 / 0,51 / 2,65 | Stufe 0 (RTX GPU 0, Reserve 0,28 GiB). Pool 583 555 Token für 524 288 | 100 MiB ≈ 32k Token (1,80 GiB für 583 555 Token) |
| 27B | 28,91 beide | 28,90 (Pool nimmt alles) | beide RTX | 100 MiB ≈ 2,8k Token (36,5 KiB je Token) |

Folgerungen:

- **Freie Luft ohne Kontext-Wirkung:**
  - DeepSeek: Stufe 0 ca. 7,9 GiB, V100 je 0,84 GiB.
  - K7: Stufe 0 ca. 8,3 GiB, V100 je 1,26 GiB.
  - GLM: V100 1,56 / 1,56 / 2,07 GiB, Stufe 4 1,81 GiB.
  - Alles, was dort liegt, kostet keinen Kontext. Das betrifft u. a. Nr. 6 (V100), Nr. 17 (GLM Stufe 4), Nr. 22, wie Nr. 5.
- **Auf RTX-Stufen werden Graphen nicht vom KV abgezogen** (`graph=0`). Sie kommen aus dem Restspeicher (Headroom, GLM Stufe 0: 1568 MiB frei nach KV).
- **Auf den bindenden Stufen** sitzen:
  - DeepSeek Stufe 4: Nr. 4 (384 MiB, Pflicht), Nr. 15 (DSpark, Tempo/Spekulation), Aktivierungsspitze 360 MiB, Graph 0,20 GiB.
  - K7 Stufe 4: dasselbe.
  - GLM Stufe 0: Aktivierungsspitze 720 MiB, Graph 0,09 GiB (nicht abgezogen), PP-Pack 1,27 MiB.
- **27B:** Nr. 14 (0,854 GiB) und Nr. 18 (ca. 0,22 GiB) liegen auf der bindenden Stufe und kosten Kontext.

## Zählung

- Geprüfte Stellen: **36 Zeilen** (Nr. 1-36; Nr. 35 ist nur eine Absicherung und Nr. 36 ist die Kontrolle, beide ohne eigenen Speicher).
- Nicht greifend in allen vier Einträgen: **9** (Nr. 1, 2, 7, 8, 9, 10, 11, 12, 13) plus Nr. 30 = **10 Stellen**.
- Greifend je Eintrag. Die Zahl nennt die Tempo-, Entwurfs- oder Puffer-Stellen aus der Tabelle, ohne Pflicht-Formate (Nr. 4, 5, 28, 29) und ohne die unklaren Nr. 24, 25, 31:
  - DeepSeek: **7** (Nr. 3, 6, 15, 20, 21, 22, 23).
  - K7: **7** (dieselben).
  - GLM: **8** (Nr. 3, 16, 17, 19, 20, 21, 22, 23).
  - 27B: **6** (Nr. 3, 14, 18, 21, 23, 26).
- Pflicht, aber speicherwirksam: DeepSeek und K7 je 2 (Nr. 4 und Nr. 5).
- Sparend (kein Speicher für Tempo): Nr. 32, 33, 34.

## Nicht geklärt (mit Grund)

1. **Zusammensetzung von `warmup_live` (Nr. 24).**
   - Der Unterschied DeepSeek gegen K7 ist auf allen fünf Stufen identisch: 309 792 256 B (295,5 MiB).
   - Das spricht für einen Puffer je Prozess, der mit `max_model_len` bzw. den Kontext-Buckets wächst (500 000 gegen 307 200).
   - Die Zuordnung ginge nur mit einem Speicher-Schnappschuss (Werkzeug `memdiag`, braucht einen Boot, nicht erlaubt). Es liegt kein Schnappschuss mehr vor.
2. **`non_torch`-Mehrbetrag auf den V100-Stufen (Nr. 25), 36 MiB.** Vermutlich TurboMind-C++-Arbeitsbereich (`fp8_tune_small_shapes`), nicht belegt.
3. **Metadaten-Puffer des FlashMLA-Sparse-Builders (Nr. 31).** Unklar, ob die SM70-Route diese Builder-Klasse und die C128A-Tabellen nutzt.
4. **Zeitpunkt der 80-MiB-Anlage in Nr. 17.** Unklar, ob vor oder nach der KV-Bemessung. Es steht kein Logeintrag dazu im Journal.
5. **Op `fp8_qpn8_block_native_sm70_out` (Nr. 35).** Das Vorhandensein in `_C` ist nicht geprüft. Für das Ergebnis ist es unerheblich, weil die Bedingung in Nr. 1 ohnehin falsch ist.
6. **Nicht einzeln gemessen:** K7 je Stufe fehlt in den gefilterten `ab_*`-Journalen. Die K7-Werte stammen aus dem Prod-Boot (01:10) im Journal. Die Stufen-Zuordnung der GLM-Headroom-Zeilen (1568 / 3108 / 3594 / 2994 / 3014 MiB) folgt der `allocated`-Größe, nicht den PIDs.
7. **GLM Stufe 4 lädt +1,3 GiB über dem Rohwert von Schichten, Kopf und Entwurfsdatei.** Gewichtsbilanz: 40,64 gemessen gegen 39,33 berechnet. Je Schicht liegt auf GLM-V100-Stufen +0,094 GiB als Format-Erweiterung (Skalen), das passt zu etwa 0,9 GiB für zehn Schichten. Eine zweite Einbettung des Entwurfs liegt nach Code (`share_embeddings`, Entwurf ohne eigene Einbettung, Host-Einbettung per UVA) nicht auf der GPU; eine Gegenprobe per Schnappschuss fehlt.
