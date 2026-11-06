# Prüfung A1: spekulatives Dekodieren und Pipeline-Parallelität

Stand 06.11.2026, Baum `/home/mp/Projekte/vllm-research/1Cat-vLLM-main-v6` (HEAD 5f1c557fb), nur gelesen.
Geprüft: 82 Kandidaten aus `area_A1_specdecode_pp.json` plus die sechs vom Auftraggeber zusätzlich genannten aus
anderen Bereichen (#62129, #47616, #58668, #61174, #52436, #50693).

Betriebsfakten, auf die sich die Urteile stützen (alle nachgesehen):
- Alle drei Einträge haben `draft_sample_method` greedy (GLM und 27B explizit, DSpark über den Standard
  `vllm/config/speculative.py:398`); `draft_logits` gibt es nur bei probabilistic (`spec_decode/speculator.py:176`).
  Alles, was nur probabilistisches Entwerfen betrifft, trifft uns nicht.
- Unser MRV2-Rejection-Sampler kennt nur `standard` und `synthetic` (`spec_decode/rejection_sampler.py:49-59`);
  Block- oder adaptive Verifikation gibt es im Baum nicht (grep auf `block_verification`, `adaptive_verification`:
  kein Treffer).
- AIfred schickt Temperatur 0,2 als Standard (`AIfred-Intelligence/aifred/backends/base.py:75`), immer
  `max_tokens` (`base.py:689`, Standard 64000 `aifred/lib/config.py:417`), keine Stopp-Strings, keine Logprobs,
  kein `min_tokens`, keine `bad_words`, kein `response_format`.
- Bei Temperatur > 0 greift der SM70-Greedy-Verify nicht (`sm70_greedy_verify.py:100-111` braucht
  `can_use_sm70_greedy_token_fastpath`, das alle Temperaturen 0 verlangt, `sample/sampler.py:73`). Dann läuft für
  DSpark (DeepSeek) und MTP (27B) der allgemeine `rejection_sample` (`model_runner.py:1449-1462`); beim 27B nimmt
  `EagleSpeculator.try_sample_target` nur mit `sm70_sparse_target_sampling` einen eigenen Weg
  (`eagle/speculator.py:409-412`, im Eintrag nicht gesetzt), GLM nutzt den DFlash2-Sparse-Weg
  (`dflash2/speculator.py:361`) und fällt sonst auf denselben allgemeinen Weg zurück.
- FA2 statt FA3: `aot_schedule` nur bei FA-Version 3 (`v1/attention/backends/flash_attn.py:340`). Das 27B wählt
  FLASH_ATTN selbst (Log `~/.cache/bench-scripts/ab_v6_fp8rtx_2026-11-06g/`: 24× „Using FLASH_ATTN attention
  backend").

## Rangliste der Empfehlungen

| Rang | PR | Kurztitel | Warum es uns trifft | Beleg im Code | Empfehlung | A/B nötig | Abhängigkeiten |
|---|---|---|---|---|---|---|---|
| 1 | #66048 | Race in `_insert_resampled_kernel` | Der Kern liest `num_sampled`, schreibt sofort `num_sampled + 1` zurück und rechnet danach mit dem gelesenen Wert weiter; Warps desselben Programms können den schon erhöhten Wert lesen, dann folgt ein Lesen hinter dem Ende von `expanded_idx_mapping` bzw. aus `temp_ptr` mit Müllindex (Absturz) oder ein verkehrt eingesetztes Token. Läuft bei uns in jeder Verifikation mit Temperatur > 0: DeepSeek-DSpark und 27B-MTP immer, GLM im Rückfall. | `vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py:462-470` (Laden, Rückschreiben, Weiterrechnen), Aufruf `:1107-1120` ohne `num_warps` (der Nachbar `_rejection_kernel` hat `num_warps=1`, `:1066`) | übernehmen (`git apply --check` sauber, eine Zeile) | nein: nur Startkonfiguration eines Kleinstkerns (ein Programm je Anfrage); Kontrolllauf mit Temperatur > 0 reicht | keine |
| 2 | #54436 | PP: decodierende Anfrage nie aus dem Sampled-Token-Broadcast werfen | Fehlercode steht wörtlich bei uns: `not_finishing = max(old_computed, prefill_len) + 1 < max_seq_len`. Mit asynchronem Scheduling zählt `num_computed_tokens` die Entwürfe des noch laufenden Schritts schon mit (`scheduler.py:1483`, Abzug erst `:1907-1918`), und der Scheduler plant eine Anfrage weiter ein, solange `num_computed + 2 - placeholders < prompt + max_tokens` (`scheduler.py:731-744`). Dann fällt die Zeile auf den früheren Stufen aus dem Broadcast, Stufe 0 bettet veraltete Token ein, die letzten bis zu k Token und ihr KV werden falsch. Heute nicht erreichbar: mit `max_num_seqs 1` blockiert der Engine-Kern nach einem leeren Schritt auf das älteste Ergebnis (`v1/engine/core.py:663-720`: leerer Schritt → `batch_queue.pop()` `:699` → `update_from_output` `:718`), sodass die Korrektur vor der nächsten Einplanung (frühestens nach `pp_size` Schritten, `async_scheduler.py:59`) angewandt ist. Erreichbar, sobald unter PP `max_num_seqs > 1` läuft; die Grenze ist real, weil AIfred immer `max_tokens` setzt. | `vllm/v1/worker/gpu/pp_utils.py:32-42` | anpassen (vorbeugend): `not_finishing` streichen, Maske = `produces_sample`; dann das nur dafür gebaute `max_seq_len_np` mit entfernen (`model_runner.py:1303-1304`, `:1328`, `input_batch.py:77`, `:189`, `prefill_overlap.py:107`, `states.py:69`, `:100`). Upstream-Diff passt nicht wörtlich (pcp_manager, andere PP-Fassung) | nein: mit `max_num_seqs 1` sind die Masken nachweislich gleich; Kontrolllauf DeepSeek/GLM mit kleinem `max_tokens` | keine; #64458 und #55745 werden dafür nicht gebraucht |
| 3 | #47911 | Entwurfsmodell-Config in den Compile-Hash | Unser `SpeculativeConfig.compute_hash` nimmt bei DFlash/DSpark nur „aux an/aus" und die Layer-IDs, nicht das Entwurfsmodell. Der Compile-Cache liegt unter `VLLM_CACHE_ROOT/torch_compile_cache/<hash>/rank_*/<model_tag>` mit Hash aus Umgebung, Config, Code, Compiler; der Entwurf landet als `dflash_head` im Verzeichnis des Ziels. Ein Tausch der Entwurfsgewichte (wie BF16 → Block-FP8 am 06.11.) im selben Cache-Wurzelordner würde alte Entwurfsartefakte laden. Heute nur durch unsere Gewohnheit abgefangen, je Eintrag eigene Cache-Ordner zu nehmen (`~/.cache/bench-scripts/draft8b_entry_glm_2026-10-31.yaml:12-22`). | `vllm/config/speculative.py:445-512`; `vllm/compilation/backends.py:1050-1081`; Ladepfad `spec_decode/dflash/utils.py:83-84` | anpassen: `factors.append(self.draft_model_config.compute_hash())` im Zweig `uses_aux_hidden_states` (Diff passt nur wegen der Nachbarzeilen nicht) | nein, aber der Hash ändert sich für alle drei Einträge: erster Boot kompiliert neu und zählt nicht | keine |

Wichtiger Gegenbefund: **#46243 (min_tokens off-by-one) steht maschinell auf `fehlt_passt`, darf aber nicht übernommen
werden.** 1Cat hat den Fehler an der Quelle behoben: `min_len = prompt_len + min_tokens - 1`
(`vllm/v1/worker/gpu/sample/logit_bias.py:92`) mit unverändertem `pos < min_len` (`:230`); upstream rechnet
`min_len = prompt_len + min_tokens` und `pos + 1 < min_len` (Upstream-Datei zum Merge-Stand). Beides sperrt bis
Position `prompt_len + min_tokens - 2`; der Upstream-Diff auf unseren Baum ließe EOS einen Schritt zu früh durch.

## Alle geprüften Kandidaten

Legende: drin = bei uns schon gelöst (ggf. anders), nein = trifft uns nicht, Rest siehe Rangliste.

| PR | Kurztitel | Urteil | Begründung mit Beleg | A/B |
|---|---|---|---|---|
| #66048 | Race insert_resampled | übernehmen | siehe Rang 1 | nein |
| #54436 | PP-Broadcast verwirft Decode-Zeile | anpassen | siehe Rang 2 | nein |
| #47911 | Entwurfs-Config im Compile-Hash | anpassen | siehe Rang 3 | nein |
| #64458 | MRV2-PP Draft/Sampled-Broadcast | nein | 1Cats eigene PP-Fassung (ein Paket aus Sampled und Drafts, Ring der Tiefe PP). Die behobenen Fehler gibt es nicht: der Gather aus `req_states.draft_tokens` läuft auf dem Main-Stream, bevor der Broadcast-Stream wartet (`model_runner.py:2077-2087`); ausgeschlossene Zeilen werden über `idx_mapping = -1` ohne Bool-Maske und ohne Sync behandelt (`pp_utils.py:145-161`, Kern `:72-106`); Record-Stream für alle Tensoren (`pp_utils.py:193-194`, `:235-243`). Rest ist Kleinoptimierung (Paketbreite) | – |
| #55745 | record_stream idx_mapping im PP-Draft-Broadcast | nein | `idx_mapping` wird bei uns nie auf dem Broadcast-Stream benutzt; das Packen nimmt nur die schon gesammelten Tensoren (`pp_utils.py:209-243`, Gather `model_runner.py:2083`) | – |
| #62129 | Warmup-Deadlock mit PP (zusätzl. Gierig-Runde) | nein | Vorgänger #59260 fehlt: unser `warmup_kernels` macht genau einen Prefill- und einen Decode-Durchgang auf allen Stufen (`warmup.py:93-246`), keine zweite Runde | – |
| #65270 | Token-Budget im gemischten Warmup | nein | gemischter Prefill/Decode-Warmup (#56492) fehlt bei uns; `warmup.py` hat keinen gemischten Schritt | – |
| #61161 | Zufallswerte über Verifikationsrunden | nein | Verzerrung nur bei Block-Verifikation (PR-Tabelle: deterministisch + standard 0,5039 vorher und nachher); wir haben nur standard (`rejection_sampler.py:49-59`) und Greedy-Entwürfe | – |
| #61174 | Veraltete geteilte Block-Table-Zeilen | nein | Mechanismus vorhanden (`block_table.py:134-151` nullt nur bis `num_reqs_padded`, der Entwurf liest bis zu seinem eigenen Padding `eagle/speculator.py:705-709`), aber bei uns nicht erreichbar: GLM/DeepSeek haben immer genau eine Anfrage in Zeile 0; beim 27B (`max_num_seqs 3`, Größen 1,2,4,8) polstert der Entwurf nur 3 → 4, Zeile 3 hat nie echte Block-IDs bekommen | – |
| #56734 | Dummy-Entwurfsschritte schreiben KV über alte Zeilen | nein | Dummy-Batches im Betrieb gibt es nur bei DP (`v1/engine/core.py:2082-2090`, Klasse `DPEngineCoreProc` ab `:1892`); beim Start existiert noch kein Präfix-Cache-Eintrag | – |
| #64059 | DFlash-Annahme 0 bei FULL-Graphen | nein | `map_to_kernel_block_table` gibt es bei uns nicht; die Umrechnung auf Kernelblöcke steckt im Gather (`block_table.py:48-50`); GLM-Annahme im Betrieb hoch (gsm8k-Annahmelänge 5,51 in `ab_draft8b_glm_2026-11-06b.log`) | – |
| #54373 | DFlash-RoPE-Layout aus eigener Config | drin | `qwen3_dflash.py:348-353` liest `is_neox_style` aus der Entwurfs-Config, mit derselben Begründung im Kommentar; die Ableitung vom Ziel (#51655) gibt es nicht | – |
| #48524 | DFlash-fc bei abweichender Schichtzahl | drin | `qwen3_dflash.py:47`, `:96-104` (`get_eagle3_aux_layers_from_config`) | – |
| #48113 | DFlash Schichtzahl-Prüfung | drin | `qwen3_dflash.py:1107-1115` | – |
| #63436 | DFlash-fc im Entwurfs-dtype | drin | `qwen3_dflash.py:1116-1121` (Cast auf das FC-dtype) | – |
| #46435 | DFlash lm_head teilen | drin | 1Cat-Fassung über `has_own_lm_head`: `model_executor/models/shared_weights.py:30-35`, `:146-170`, Aufruf `dflash/utils.py:101-109` | – |
| #51256 | DFlash-Budget reserviert Bonus-Slot | drin | `speculative.py:1625-1629` gibt für DFlash K zurück | – |
| #51538 | DSV4 Sparse-MLA SM120 (inkl. DFlash-Eingaben) | drin/nein | DFlash-Teil drin (`dflash/speculator.py:132-134` `sample_idx_mapping = -1`, `:888-920` gültiger Kontext, Nullblock-Schutz); Rest SM120/FlashInfer | – |
| #44492 | Draft `seq_lens_cpu_upper_bound` | drin | `spec_decode/speculator.py:316-323` | – |
| #47429 | DSpark `draft_id_to_target_id` | drin | `models/deepseek_v4/nvidia/dspark.py:310` | – |
| #51602 | DSpark parallel_drafting_token_id (MRV1) | nein | nur MRV1; MRV2 nutzt `get_parallel_drafting_token_id` (`spec_decode/utils.py:60`, `dflash/speculator.py:90`) | – |
| #65816 | DSv4-MTP teilt Einbettung unter PP | nein | betrifft den MTP-Kopf; wir nutzen DSpark, das schon `has_own_embed_tokens = False` hat (`dspark.py:308-309`). Diff passt; nur nötig, falls DeepSeek-MTP je unter PP läuft | – |
| #50693 | DSpark-Warmup ohne Sparse-Index-Puffer | nein | FlashMLA-Pfad; unser SM70-Pfad hat die Assertion zwar ebenfalls vor dem SWA-Zweig (`models/deepseek_v4/sm70/sparse.py:158`), DSpark bootet aber im Betrieb (also Puffer gesetzt oder Pfad nicht ohne Metadaten durchlaufen) | – |
| #56448 | DFlash/DSpark-Profilierungsbatch deckeln | nein | Puffer bei uns nach `max_num_reqs × num_query_per_req` bemessen (`dflash/speculator.py:62`); 1 × 8 ≪ 1024 | – |
| #50065 | DFlash-Abfragepuffer bei Graph-Padding (MRV1) | nein | MRV1-Datei; in MRV2 ist die Abfragezahl je Eintrag selbst Graph-Größe (GLM 8 ∈ [8], DSpark 5 ∈ [5,6], K7 7 ∈ [7,8]) | – |
| #50869, #52223 | Falsche DSpark-Startassertion entfernen | nein | Assertion steht bei uns (`speculative.py:1103-1118`), greift aber erst bei k < `dspark_block_size` = 5 (DeepSeek-Config); wir fahren k = 5 und 7 | – |
| #46243 | min_tokens off-by-one | nein (anders gelöst) | siehe Gegenbefund oben; Übernahme würde den Fehler umdrehen | – |
| #50183 | NaN in `tl.argmax` des Rejection-Samplers | nein | Muster vorhanden (`rejection_sampler_utils.py:241`, `:487`), tritt nur bei NaN-Logits auf; die wären selbst der Fehler, der Fix verdeckt sie (keine Fallbacks ohne Grund); Diagnose über `get_num_nans` existiert (`rejection_sampler.py:110`) | – |
| #54826 | Entwurfs-`attention_backend` in MRV2 | nein | Feld wird für MTP tatsächlich ignoriert (`eagle/utils.py:24-29` lädt mit der Ziel-Config, bei DFlash/DSpark beachtet: `dflash/utils.py:71`), das 27B-Ziel wählt aber selbst FLASH_ATTN (Log oben) → gleiche Wahl | – |
| #53104 | Eagle-lm_head-Teilen bei quantisiertem Kopf | nein | Lücke vorhanden (`shared_weights.py:44` liest `draft.weight` ungeprüft), 27B-`lm_head` ist unquantisiert (`modules_to_not_convert` enthält `lm_head`), alle Einträge booten | – |
| #50417 | Multimodal-Fähigkeit des Entwurfs | nein | Regression einer späteren Upstream-Umstellung; bei uns getrennt geprüft (`spec_decode/speculator.py:222-229`), 27B-MTP mit Bildern läuft (`eagle/speculator.py:142-146`, `:370-380`) | – |
| #47066 | Veraltetes idx_mapping beim Graph-Padding | nein | wirkt nur beim Schreiben in `draft_logits` (probabilistisch) | – |
| #53017 | Spaltenschritt im Draft-Logits-Cache | nein | nur mit `draft_logits` (probabilistisch) | – |
| #54282 | Gumbel-Rauschen von Entwurf und Ziel trennen | nein | laut PR nur probabilistisch | – |
| #58784 | Nie vorgeschlagene Draft-Slots verwerfen | nein | P/D-Decode-Instanz; ohne Block-Verifikation und mit Greedy-Entwürfen bleibt standard exakt | – |
| #47383 | int32-Überlauf Block-Verifikation | nein | keine Block-Verifikation im Baum | – |
| #50939 | -1-Platzhalter bei Block-Verifikation | nein | keine Block-Verifikation im Baum | – |
| #60945 | Lookahead in adaptiver Block-Verifikation | nein | keine adaptive Verifikation im Baum | – |
| #64137 | Logits-Stats-Kerne ohne num_logits-Spezialisierung | nein | unser Kern `_compute_block_stats_kernel` hat kein `num_logits`-Argument (`rejection_sampler_utils.py:52`); #61451 fehlt | – |
| #46560 | int32-Überlauf in Sampler-Kernen | nein | Überlauf ab Zeile × Vokabular ≥ 2³¹; bei uns höchstens 12 Logit-Zeilen × 248 320 ≈ 3 Mio. | – |
| #60736 | int64 idx_mapping | nein | Überlauf ab Slot × Stride ≥ 2³¹; Slots 0 (PP-Einträge) bzw. 0–2 (27B) bei Strides ≤ 3 × 262 144 | – |
| #58231 | Split-Top-p maskiert alles | nein | Split-Top-p-Pipeline (#54651, `_topp_sb_mask_kernel`) fehlt in `v1/sample/ops/topk_topp_triton.py` | – |
| #62046 | Split-Top-p Gleichstandszähler | nein | wie #58231 | – |
| #53002, #53336 | FA-Metadaten aus Gruppengeometrie | nein | betrifft den FA3-AOT-Plan; bei uns FA2 (`flash_attn.py:340`) | – |
| #54374 | FA-AOT-Plan für Sliding-Window-DFlash | nein | wie #53002 (FA3 only) | – |
| #48261 | Veraltete Metadaten beim Capture des Speculator-Prefills | nein | nur Capture, Absturz im FA3-Combine-Kern; Replay laut PR nicht betroffen | – |
| #44253 | Warmup und Capture mit getrennten Attention-States | nein | betrifft Backends mit verzögerter Metadaten-Initialisierung (FlashMLA); 27B-Entwurf auf FA2, DeepSeek-Entwurf ist DSpark, kein Eagle | – |
| #48754 | Punkte im Entwurfsnamen | nein | absoluter Pfad enthält „/" und wird nicht als `custom_class` gelesen (`speculative.py:797-804`) | – |
| #49343 | Eagle-Entwurf max_position_embeddings | nein | nur `eagle`/`eagle3`; DFlash2-Entwurf hat 1 048 576 Positionen ≥ 524 288 | – |
| #42944 | GLM-5.1 PP-Laden (`deepseek_mtp.py`) | nein | DeepSeekMTP/GLM-MTP-Kopf läuft bei uns nicht (GLM mit `dflash`, DeepSeek mit `dspark`) | – |
| #55442 | GLM-MTP-Kopf verzögern | nein | GLM-MTP nicht in Betrieb | – |
| #61079 | GLM-5.3-Flash MTP-Index-Teilen | nein | GLM-MTP nicht in Betrieb (Entwurf ist DFlash2) | – |
| #44821 | DSv4-MTP-Projektionen mit Präfix | nein | NVIDIA-Datei schon drin, AMD nicht im Einsatz, MTP nicht im Einsatz | – |
| #47616 | Stopp-String unter Spekulation kürzt Tokens | nein | AIfred schickt keine Stopp-Strings (grep in `aifred/`), Text wird ohnehin richtig gekürzt | – |
| #58668 | Gekürzte Spec-Tokens aus num_computed_tokens | nein | betrifft nur fortsetzbare Streaming-Input-Sitzungen; normale Anfragen werden nach dem Stopp freigegeben | – |
| #52436 | DSpark adaptive Verifikation + Structured Output | nein | weder adaptive Verifikation noch Structured Outputs bei uns | – |

## Aussortiert nach Titel und Beschreibung (30)

- #42079 runai_streamer/S3, nicht genutzt
- #43150 FP64-Gumbel-Schalter für V1-Pfade, MRV1 und Schalter nicht genutzt
- #45002 Qwen3.5-Laden mit EP/MoE, 27B ist dicht, kein EP
- #45352 aufrufbare `hf_overrides`, nicht genutzt
- #45467 `top_logprobs=-1`, keine Logprobs
- #45956 Parallel Drafting probabilistisch in MRV1
- #46820 Transformers-Backend FP8-MoE
- #47238 MTP-Top-k-Puffer in MRV1 (`llm_base_proposer`)
- #47953 Embedding-Breiten-Wächter in MRV1 (Gemma4-MTP)
- #48429 Qwen-MTP mit MoE-Sequenzparallelität
- #48630 Rejection-Sampler-OOM bei großen Batches, wir höchstens 12 Logit-Zeilen
- #49178 MTP-Vollständigkeit bei IPC/NCCL-Gewichtstransfer (RL)
- #49415 DSpark-Shared-Expert-Padding für TP > 8
- #50685 Qwen3Next-Sequenzparallelität mit EP/DP
- #50734 Qwen3.5-MTP für reine Text-Checkpoints, unserer ist multimodal
- #51368 DSv4 mHC-Puffer bei `--load-format dummy`
- #51866 CPU-Sampler-Seeds
- #52311 `bad_words` unter Spekulation, nicht genutzt
- #54416 fastsafetensors-Deadlock, wir laden mit `--safetensors-load-strategy direct`
- #54442 Structured Output auf ungemaskter Zeile, nicht genutzt
- #54788 Entwurfs-`moe_backend`, nicht gesetzt
- #55914 DSv4 MegaMoE mit EP
- #56930 EAGLE/dichter Entwurf mit EP
- #57053 dynamisches K (`num_spec_tokens_to_schedule` gibt es bei uns nicht)
- #58921 mehrschichtiges MTP, 27B hat `mtp_num_hidden_layers` 1
- #60513 Speculator-Grenzen nach `--max-model-len auto`
- #61219 `logprob_token_ids` im V2-Rejection-Sampler
- #61563 EAGLE3 mit einem Zustand
- #65350 MRV1-DFlash `dataclasses.replace`
- #65463 `--max-model-len auto` mit erweiterbarem KV-Cache
