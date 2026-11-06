# Upstream-Backport-Durchsicht Bereich A2: Engine-Core, Executor, Scheduler, KV-Cache, shm_broadcast, MRV2 (ohne Spec-Decode), Konfiguration

Stand 06.11.2026. Geprüfter Baum (nur gelesen): `/home/mp/Projekte/vllm-research/1Cat-vLLM-main-v6` (HEAD 5f1c557fb = fork-main-v6 + Port vllm#60472). Kandidaten: 222 aus `area_A2_core_sched_kv.json`. Davon 3 empfohlen, 83 weitere im Code geprüft (Urteil „nein“ oder „schon drin“), 136 nach Titel/Beschreibung aussortiert.

Betriebsdaten, die in die Urteile eingehen (aus llama-swap-Config und Boot-Logs): DSv4/K7 PP5, `max_num_seqs` 1, `--num-gpu-blocks-override` 4220/3500, kein Mamba; GLM PP5, `max_num_seqs` 1, KDA (MambaSpec) + MLA fp8 + KpoolTail (Block 4) + DFlash2-SWA-Entwurf (Block 1152, FLASH_ATTN), Hash-Block = ggT = 4; 27B TP2, `max_num_seqs` 3, GDN + Attention, Hybrid-Blockgröße 800 → Hash-Block 800, MTP k=3, FLASH_ATTN; transformers 5.16.1, torch 2.10, pyzmq 27.2; AIfred fordert weder Logprobs noch Stoppstrings an.

## Rangliste der Empfehlungen

| Rang | PR | Kurztitel | Warum es uns trifft | Beleg im Code | Empfehlung | A/B | Abhängigkeiten |
|---|---|---|---|---|---|---|---|
| 1 | #45224 | shm_broadcast: Leerlauf-Wartezeit der Leser begrenzen, Leseslot im `finally` freigeben | Worker lesen die RPC-Queue mit `indefinite=True`; dann liefert `timeout_ms()` `None` und der Leser parkt ohne Obergrenze auf dem ZMQ-Notify (PUB `SNDHWM=1` verwirft still, SUB `CONFLATE`). Geht ein Notify verloren, steht der Worker trotz geschriebener Daten, die EngineCore stirbt nach 300 s am RPC-Timeout. Wirft der Aufrufer im `with`-Block (z. B. `pickle.loads`), bleibt der Slot belegt und der Schreiber blockiert. Bei uns bisher nicht beobachtet (keine „RPC call … timed out“ im Journal seit 01.10.). | `shm_broadcast.py:659-681` (`return warning_wait_time` = None), `:723` (wait), `:738-749` (Lesemarke nach `yield` ohne `finally`); Worker `multiproc_executor.py:1008-1010`; Notify-Sockets `shm_broadcast.py:139-165` | übernehmen (Patch passt unverändert auf fork-main-v6; im 1Cat-PR-Baum `1Cat-vLLM-pr-backport-60472` ist der Kontext anders, dort anpassen) | nein (nur der Leerlaufpfad wacht alle 5 s auf, der Busy-Pfad bleibt gleich) | vor #63165 (dessen Kontext); unabhängig vom #60472-Port in derselben Datei |
| 2 | #44490 | Host-Speicherleck durch nie geleerte `new_block_ids` | DSv4/K7 haben keine Mamba-Schichten, also `needs_kv_cache_zeroing = False`; der Scheduler leert `new_block_ids` nur, wenn genullt wird, die Manager tragen aber jeden neuen MLA-Block ein (1Cats Ring-/Tail-Manager sogar ohne Typprüfung). Die Liste wächst je allokiertem Block, solange die Engine läuft (Größe nicht gemessen). GLM/27B betrifft es nicht (Mamba → wird geleert). | `scheduler.py:1369-1373`; `kv_cache_interface.py:1080-1081`; Einträge `single_type_kv_cache_manager.py:276-282`, `310-316`, `703-710` (`_claim_ring_block`), `1082-1089` (`_allocate_tail_block`) | anpassen: nur den Scheduler-Hunk übernehmen (jeden Schritt `take_new_block_ids()` aufrufen, Ergebnis nur bei Nullung verwenden). Der Erfassungs-Schalter im Manager-Hunk passt nicht (konflikt) und deckt 1Cats Ring/Tail-Manager nicht ab; das Leeren auf der Verbraucherseite deckt alle ab | nein (keine GPU-Arbeit ändert sich) | keine |
| 3 | #51694 | Inkrementelles Block-Hashing mit mehreren MM-Elementen | Wird ein Block erst mit generierten Token voll (letzter Promptblock, auch Lookahead-Hash), startet die MM-Suche beim letzten Bild; ein früheres Bild im selben Block fehlt in den Hash-Schlüsseln. Beim 27B ist der Hash-Block 800 Token groß, zwei kleine Bilder plus kurze Frage passen hinein. Folge: anderer Hash als bei frischer Berechnung (Folgetreffer endet dort) und theoretisch ein falscher Treffer für gleiche Token mit anderem Bild. Praktisch gering: Qwen-Templates entfernen das Denken aus dem Verlauf, der Block trifft im Folgeturn ohnehin selten. GLM (Hash-Block 4) und DSv4 (keine Bilder) nicht betroffen. | `kv_cache_utils.py:774-781` (`curr_mm_idx = -1`), `:420-484`; Hash-Block = ggT `kv_cache_utils.py:740-743`; Boot-Log 27B „Setting attention block size to 800 tokens“ | anpassen (niedrig): `curr_mm_idx` immer bei 0 starten. Die frühe Rückkehr in `:446-449` hält Decode-Blöcke billig, der Helfer `get_mm_features_in_window` aus upstream ist nicht nötig | nein (nur Hash-Schlüssel); Kontrolle: 27B mit zwei kleinen Bildern, `prompt_tokens_details.cached_tokens` im Folgeturn | keine |

Hinweise zu weiteren vorgemerkten PRs (alle „nein“, Gründe in der Tabelle unten):
- #41357 wirkungslos bei uns (SHM-Pfad wirft schon bei ≤ 0); #63165 vernachlässigbar, nur nach #45224; #61000 ohne Auslöser (keine numpy-Puffer ≥ 1 MiB in der Queue).
- #50960 und #54113 (Bereich C) betreffen nur entfernte Leser bzw. TCP-Frontends; bei uns laufen API-Server und EngineCore über lokale IPC-Pfade (`v1/utils.py:173-184`).
- #50053 braucht Logprobs/Tensoren in der Ausgabe, #51979 hält im Leerlauf nur einen kleinen Decode-Payload.
- #47272 ist bei 1Cat schon gelöst (Null-Block in `_kv_cache_memory_needed`); #53955/#53682 betreffen Code, den 1Cats MRV2 nicht hat (kein Graph-Capture beim Profilieren); #61044 nie beobachtet.
- MM/Encoder-Cache: #52412, #65271, #59721 nur mit `disable_chunked_mm_input` (bei uns aus); #54284 passt textlich, braucht aber `request_cached_ids` (fehlt, AttributeError); #45347, #52482, #59942, #60553 bei unserem älteren Shift-Schema ohne Auslöser.
- #60767: GLM durch Seitengröße 1152 = 9 × 128 sicher. Unverändert übernommen würde der Patch den KpoolTail-Puffer (Unterklasse von `SlidingWindowSpec`) nullen, GLM wäre kaputt. #62206 gilt nur für SWA-Retention, die 1Cat nur für Mamba hat.
- #65463, #65225 und #65554 betreffen den Extensible-KV-Cache, den es bei uns nicht gibt.
- #55341 (Workspace wächst nach dem Capture): 1Cat reserviert den Worst Case vorab. MRV2 sperrt den Workspace nach `capture_model` aber nicht (V1 schon). Ein `lock_workspace()` dort wäre ein billiger Wächter gegen stilles Überschreiben. Das gehört nicht zu einem Kandidaten und ist nur eine Idee zur Entscheidung.
- Fallen bei `fehlt_passt`: #54284 (fehlender Vorgänger), #56050 (würde für GLM die Aktivierungsquant-Fusion einschalten, anderer Kompilierpfad), #45059 (passt, ändert aber alle Compile-Hashes).

## Alle geprüften Kandidaten (Patch und unser Code gelesen)

| PR | Titel | Status maschinell | Urteil |
|---|---|---|---|
| #39562 | [Bugfix]: Fix assertion in MambaManager.allocate_slots() | fehlt_passt | nein — Assertion nur bei schwankender Entwurfslänge (Suffix-Decoding); mit festem K (MTP 3, DFlash 7) schrumpft `num_tokens` nicht (`single_type_kv_cache_manager.py:1468-1476`). |
| #41357 | [Bugfix] Prevent stale multiproc RPC deadlines from becoming unbounded waits | fehlt_passt | nein (optional) — Ein abgelaufener Termin erreicht bei uns nur den ZMQ-Überlaufpfad (Nachrichten > max_chunk_bytes); der SHM-Pfad wirft schon bei ≤ 0 ms (`shm_broadcast.py:659-681`). Patch passt und ist harmlos, ändert aber nichts. |
| #42971 | Fix DFlash prefix cache corruption due to missing lookahead block | fehlt_passt | nein — V1-Runner; der Scheduler-Teil (Lookahead auch für neue Anfragen) ist bei uns schon richtig (`scheduler.py:1192-1195`). |
| #43988 | [Bugfix] Use storage_block_size in KV cache reshape for compressed specs (DeepSeek V4) | konflikt | nein — DSv4 bootet auf MRV2 (1Cat-Pfad), Fehler wäre beim Start sichtbar. |
| #44082 | [Bugfix] Cache the EAGLE/MTP lookahead block in the SWA prefix-cache mask | konflikt | schon drin (`reachable_block_mask`, `_contiguous_blocks_for_hit`, `SpecGroup`: `kv_cache_coordinator.py:492`, `645-654`). |
| #44490 | [Bugfix][Core] Fix host memory leak from undrained new_block_ids | konflikt | siehe Rangliste oben |
| #45059 | fix: AOT compile cache collision for dataclass-based HF configs | fehlt_passt | nein — transformers 5.16 ist betroffen (`config/utils.py:234ff`, Dataclass-Zweig vor `to_json_string`), eine Kollision bräuchte aber Configs, die sich nur in dynamisch gesetzten Attributen unterscheiden; jeder Eintrag hat eigene Cache-Verzeichnisse. Übernahme würde alle Compile-Hashes ändern (Neukompilierung). |
| #45100 | [Bugfix] Avoid racy accepted counts in async spec decode | konflikt | nein — Rennen im V1-Runner; MRV2 betrifft es nicht. |
| #45159 | fix(distributed): propagate distributed_timeout_seconds to NCCL device groups | teilweise | schon drin (`parallel_state.py:542-546`, `616-620`; `distributed/utils.py:531-539`). |
| #45224 | [Bugfix][Core] shm_broadcast: bound idle reader waits and release read slots | fehlt_passt | siehe Rangliste oben |
| #45347 | [BugFix] Avoid prematurely freeing cached mm encoder outputs | fehlt_passt | nein — Platzhalter (`num_output_placeholders`) entstehen erst nach dem Prefill, MM-Elemente enden vor dem Promptende (Template-Text folgt); eine frühere Freigabe wirkt beim Worker erst nach dem laufenden Schritt. Patch passt, bei uns ohne Wirkung. |
| #45564 | [Bugfix][V1] Split V2 model-runner attention groups on num_heads_q | fehlt_passt | nein — Absturz im FlashInfer-Metadaten-Builder; kein FlashInfer bei uns. |
| #45868 | [ModelRunnerV2] Various model/config compatibility fixes | teilweise | nein — Punkt 3 (`is_embed` überschrieben statt ODER) wirkt nur bei überlappenden MM-Bereichen; Bilder überlappen nicht. |
| #45984 | [Bugfix] Fix thinking_token_budget not enforced after natural </think> re-entry | konflikt | nein — `thinking_token_budget` nicht genutzt. |
| #46315 | [Bugfix][Spec Decode] Fix EAGLE drafter multimodal encoder cache misses | teilweise | nein — unser älteres Schema ist in sich konsistent: Scheduler-Shift 1 (`scheduler.py:830/1170`, `1676-1744`), Drafter-Gather `computed+1` (`gpu/model_runner.py:1969-1985`). Übernahme zöge #60553 nach sich. |
| #46747 | [Bugfix][V1][Multimodal] Recover from P0/P1 processor cache drift (#46747) | teilweise | nein — Drift P0/P1 braucht nebenläufige MM-Anfragen oder Clients, die nur Hashes senden; großer Protokollumbau (EngineCoreOutput-Feld, Frontend). Bei Auftreten scheitert nur die Anfrage. |
| #47029 | [Bugfix] Prevent padding placeholders from reaching embeddings | fehlt_passt | nein — V1-Runner; `-1`-Platzhalter erzeugt unser Scheduler nur mit Grammatik (`scheduler.py:2364-2368`). |
| #47050 | [BugFix] Gate MRV2 mixed sparse-MLA warmup on `max_num_seqs` > 1 | konflikt | nein — FlashInfer-Sparse-MLA-Mixed-Warmup (`run_mixed_prefill_decode_warmup`) gibt es bei uns nicht. |
| #47272 | [Bugfix][Core] Reserve the KV null block when validating max_model_len | konflikt | nein, schon gelöst (1Cat): `_kv_cache_memory_needed` zählt den Null-Block mit (`kv_cache_utils.py:3191-3204`, Prüfung 3460-3465); die öffentliche `check_enough_kv_cache_memory` wird nicht aufgerufen. DSv4-Override 4220 / K7 3500 hat Reserve. |
| #47381 | [Bugfix][Model Runner V2] Order uniform decodes first so spec decodes aren't misclassified as prefills | fehlt_passt | nein (optional) — Reihenfolge ändert bei uns die Ausgabe nicht: GDN wählt Spec-Zeilen per Maske (`gdn_attn.py:1247-1254`), FA2 ohne Decode/Prefill-Split, GLM/DSv4 nur eine Sequenz. Patch passt (`gpu/model_runner.py:1171`). |
| #47574 | [Bugfix] Zero new KV blocks for quantized + sliding-window hybrid caches | fehlt_passt | nein — GLM (KDA) und 27B (GDN) nullen schon (`needs_kv_cache_zeroing` = Mamba vorhanden, `kv_cache_interface.py:1080-1081`); DSv4 liest per Indexliste nur geschriebene Slots. Würde bei DSv4 das Nullen einschalten (Tempo, A/B). |
| #47616 | [Bugfix][Spec Decode] Trim token_ids/logprobs left past a stop string under speculative decoding | konflikt | nein — Stoppstrings. |
| #47716 | [Bugfix]Fix DeepSeek-V4 fp8_ds_mla KV cache reshape | konflikt | nein — FlashMLA-Layout (SM90). |
| #47728 | [Bugfix][V1] Free out-of-window blocks on the processed-token basis under async scheduling | konflikt | schon drin (1Cat): Freigabe auf Basis `num_computed_tokens - num_in_flight_tokens` (`kv_cache_manager.py:672-679`). |
| #48085 | [Bugfix] Fix race condition in KVBlockZeroer | konflikt | schon gelöst (1Cat anders): frische Pinned-Quelle je Aufruf (`worker/utils.py:548-558`). |
| #48245 | [BugFix] Fix `num_output_placeholders` preemption underflow | teilweise | unklar/nein — Absturz nur bei KV-Druck-Preemption unter async Scheduling; DSv4/GLM (1 Sequenz, Pool ≥ max_model_len) preemptieren nie, 27B nur mit mehreren langen Anfragen gleichzeitig. Patch hängt an weiteren Zählern (`num_stale_output_tokens`, Request-Felder). |
| #48256 | [Bugfix][KV Cache] Don't route uniform-page-size MLA+SWA models into DeepseekV4 packing | fehlt_passt | nein — `SlidingWindowMLASpec` nur bei DSv4. |
| #48776 | [Bugfix][KV cache] Support sparse-MLA targets with SWA drafts | konflikt | nein — 1Cat löst GLM-Sparse-MLA + SWA-Entwurf selbst (Boot-Log: „DFlash auxiliary attention uses block_size=1152 and shares padded … MLA tensor slots“). |
| #49274 | [Bugfix] Make metadata send non-blocking in GroupCoordinator.isend_tensor_dict | konflikt | nein — blockierender Metadaten-Send nur bei Sende/Empfangs-Versatz (vllm-ascend); bei uns kleine Metadaten bzw. metadatenfreier SM70-PP-Transfer (`gpu_worker.py:1444-1458`). |
| #49391 | [Bugfix][Spec Decode] Select earliest-completing stop string in check_stop_strings | fehlt_passt | nein — Stoppstrings; AIfred sendet keine. |
| #49623 | [Bugfix] Detect mixed precision in packed KV cache specs | konflikt | nein — Folge-PR von #47574. |
| #49675 | [Bugfix][Core] Stop zero-progress preemption cascades for deferred KV frees | konflikt | nein — der Deferred-Free-Zaun (#45357) fehlt bei uns. |
| #49704 | [Bugfix] Support non-uniform page sizes in KVBlockZeroer | konflikt | schon gelöst (Segmente mit eigener Seitengröße, `worker/utils.py:370-381`). |
| #50053 | [Bugfix] Don't reuse engine core payload buffer while zmq is sending it | konflikt | nein — Fehler braucht Hilfs-Frames (Tensor/ndarray ≥ 256 B) in `EngineCoreOutputs`; ohne Logprobs (AIfred fordert keine an) ist der Tracker der des Payload-Frames. Code alt gleich (`core.py:1744-1801`). |
| #50276 | [Bugfix] Fix packed KV block zeroing stride | konflikt | schon gelöst (`seg_block_strides` getrennt von der Seitengröße, `worker/utils.py:538-546`). |
| #50432 | [Bugfix][Hybrid] Fix cross-block race on num_accepted in MRv2 align prefix cache | konflikt | nein — 1Cats MRV2-Align nutzt einen Schnappschuss (`gpu/mamba_align.py:210-239`), der V1-Kern schreibt in `_out` (`mamba_utils.py:244`). |
| #50960 | [Bugfix] Fix ZMQ port TOCTOU race in shm_broadcast MessageQueue | fehlt_passt | nein — nur entfernte Leser (`n_reader > n_local_reader`, mehrere Knoten); wir ein Knoten. |
| #51694 | [Bugfix][KV Cache] Fix incremental multimodal block hashing | konflikt | siehe Rangliste oben |
| #51749 | [Bugfix] Generalize KV block zeroing to `AttentionSpec` | konflikt | nein — Akzeptanzeinbruch upstream nur mit FlashInfer/TRTLLM-gen-SWA, mit FlashAttention stabil; unser GLM-Entwurf läuft auf FLASH_ATTN. Unser Zeroer nullt nur FullAttention + Mamba (`worker/utils.py:396-436`). |
| #51843 | [Bugfix] Disable fine-grained prefix-cache hits for incompatible hybrid KV layouts | konflikt | nein — wie #52419. |
| #51865 | [Bugfix][MRV2] Require all requests to be decoding for uniform-decode dispatch | teilweise | schon drin (Kommentar „upstream #51865“, `cudagraph_utils.py:193-199`, `gpu/model_runner.py:1525-1545`). |
| #51979 | [Bugfix] Release worker RPC payload before next dequeue | konflikt | nein (niedrig) — gleiches Halten von `args` in `multiproc_executor.py:1004-1030`, aber Bilddaten stehen nur im ersten Schritt einer Anfrage (`NewRequestData`); im Leerlauf bleibt ein kleiner Decode-Payload liegen. |
| #52058 | [Bugfix] Bound KV block zeroing launch geometry | konflikt | schon gelöst (3-D-Grid `(n_blocks, n_segs, max_chunks)`, `worker/utils.py:559`). |
| #52412 | [Bugfix] Don't split a multimodal item that a prefill chunk starts exactly on | fehlt_passt | nein — wirkt nur mit `disable_chunked_mm_input`; bei uns aus (kein mm-prefix-LM: `model_arch_config_convertor.py:263-275`, `platforms/cuda.py:307-317`). |
| #52419 | [Bugfix][Spec Decode] Keep EAGLE cache registration on the partial-hash-hit path | konflikt | nein — fein-granulare Teiltreffer (#50062) fehlen bei uns. |
| #52482 | [Bugfix][V1][Multimodal] Ignore stale same-step encoder cache evictions | fehlt_passt | nein (optional) — unser MRV2 gibt im selben Schritt erst frei und kodiert dann neu (`gpu/model_runner.py:1058-1061` vor `model_states/default.py:115-120`), ein Fehltreffer ist bei uns nicht nachweisbar. Harmloser Einzeiler, passt. |
| #53456 | [Bugfix] Keep grid dims for XD-RoPE models on a prefix-cache hit | teilweise | nein — `strip_covered_mm_data` gibt es bei uns nicht (MM-Daten gehen immer vollständig mit), XD-RoPE nicht genutzt. |
| #53682 | [Bugfix][MRV2] Run cudagraph memory profiling in a throwaway graph pool | konflikt | nein — wie #53955 (kein Wegwerf-Graph-Pool nötig, es wird beim Profil nichts gefangen). |
| #53955 | [Bugfix] Release CUDA graph profiling memory before KV cache allocation | konflikt | nein — 1Cats MRV2-`profile_cudagraph_memory` fängt keine Graphen, sondern liefert einen festen Vorbehalt (`gpu/model_runner.py:931-935`, `get_sm70_cudagraph_memory_reserve` in `cudagraph_utils.py:97-110`); der Code des Fixes existiert nicht. |
| #53962 | [Bugfix][Scheduler] Don't pad spec decode up to `max_model_len` | konflikt | nein — Spec-Padding im Wartepfad fehlt bei uns. |
| #54284 | [Bugfix][V1] Keep an encoder cache entry until its last occurrence is freed | fehlt_passt | nein — Folge ist nur eine doppelte Encoder-Rechnung bei mehrfach gleichem Bild in einer Anfrage. ACHTUNG: `git apply --check` passt, der Patch nutzt aber `self.request_cached_ids`, das es in unserem `encoder_cache_manager.py` nicht gibt → AttributeError. |
| #54642 | [Bugfix] Include multimodal limits in the model compile-cache hash | konflikt | nein — MM-Limits je Eintrag fest, eigene Cache-Verzeichnisse je Stand. |
| #54713 | [BugFix] Retain both replay boundaries so an EAGLE resend of a block-aligned prompt still hits | konflikt | nein — betrifft EAGLE-Drop + Mamba/SWA-Retention; 27B/GLM nutzen Lookahead-Hashes ohne Drop, DSv4 (Drop) hat keine Mamba-Retention. |
| #55112 | [Bugfix][Core] Copy every overlaid KV view in copy_kv_cache_blocks_inplace | konflikt | nein — `copy_kv_cache_blocks_inplace` (CoW bei Teiltreffern) gibt es bei uns nicht; 1Cats `partial_prefix_cache` ist aus (`config/cache.py:111`). |
| #55341 | [Bugfix][V2] Warm up kernels before capturing CUDA graphs | konflikt | nein — 1Cat reserviert den Worst-Case-Workspace vor dem Capture (`sparse_attn_indexer_kpool.py:284-289`, `deepseek_v4/sm70/sparse.py:150-175`, `glm5next/sm70/sparse.py:170-180`); FA nutzt ihn nur mit DCP. Hinweis: MRV2 sperrt den Workspace nach `capture_model` nicht (V1 schon, `gpu_model_runner.py:11955`); ein Sperren wäre ein billiger Wächter gegen stilles Überschreiben. |
| #55504 | [Bugfix] Bound GDN speculative decode query widths | konflikt | nein — Folge von #56531. |
| #55891 | [Bugfix] Set worker runtime threads before profiling and compilation | konflikt | nein — `set_torch_threads_for_runtime` gibt es bei uns nicht. |
| #56050 | [Bugfix][Quantization] Detect NVFP4 in ModelOpt mixed-precision checkpoints | fehlt_passt | nein — würde für GLM (modelopt `MIXED_PRECISION`) `enable_act_fusion` einschalten (`config/vllm.py:158-168`) → anderer Kompilierpfad auf SM70, Risiko für Tempo und Ausgabe. |
| #56372 | [Bugfix][Multimodal] Fix flat/scoped mm_processor_kwargs merge and resolution | konflikt | nein — betrifft Kombinationen aus Engine-/Request- und modalitäts-gescopten kwargs; wir setzen nur flach `max_pixels`, das in `qwen3_vl.py:918` greift. |
| #56531 | [Bugfix] Route every speculation-capable row through the speculative path so recurrent state stays consistent | konflikt | nein — Auslöser sind Proposer ohne Entwurf (n-gram); MTP/DFlash liefern immer K. Unser GDN wählt den Zustand ohnehin per `num_accepted` (`gdn_attn.py:1305-1311`). |
| #57270 | [Bugfix][Model Runner V2] Route dummy tokens to MoE experts during profiling | konflikt | nein — `VLLM_MOE_SKIP_PADDING`/`is_padding` gibt es bei uns nicht. |
| #57317 | [Bugfix][KV Cache][GLM-5.3-Flash] Disable slot mapping kernel for the kpool tail buffer | fehlt_passt | nein — die KpoolTail-Blocktabelle ist bei uns `cdiv(max_model_len, 4)` breit (`gpu/model_runner.py:596-608`), die generische Slot-Map liest im Bereich; geschrieben wird über `compute_kpool_tail_slot_mapping` (`attention/backends/mla/indexer.py:288-373`). 524k-Prompts abgenommen. |
| #57696 | [Bugfix][V1] Reject encoder-cache hits with mismatched embedding counts | konflikt | nein — nur bei vom Client wiederverwendeten MM-UUIDs; AIfred sendet keine. |
| #57926 | [Bugfix][Core][MRV2] Profile the sampler with the params the warm-up actually uses | konflikt | nein — Sampler-Speicher bei 1–3 Sequenzen winzig. |
| #58430 | [Bugfix] Stop allocator fragmentation from shrinking the KV cache during memory profiling | konflikt | nein — betrifft nur die KV-Kapazität, nicht die Korrektheit; würde AIfreds Kalibrierung verschieben. |
| #58434 | [Bugfix][MRV2] Treat padded prompt tails as spec-decode rows for hybrid models | konflikt | nein — Ankunft eines P/D-Decode-Requests. |
| #58668 | [Bugfix][Core] Uncount stop-trimmed spec tokens from num_computed_tokens | konflikt | nein — wirkt nur bei streaming-input-Sessions. |
| #58756 | [Bugfix] Register IR providers before hashing configuration | fehlt_passt | nein — Folge ist nur `system_fingerprint` „nohash“ (`entrypoints/openai/fingerprint.py:63`). |
| #59030 | [Bugfix][Core] Requeue encoder-stalled waiting requests | konflikt | nein — Head-of-line-Stau mehrerer wartender Anfragen; bei uns 1 bzw. 3 Sequenzen, Single-User. |
| #59105 | [Bugfix][Spec Decode] Reserve the bonus KV slot for fill-in DSpark | konflikt | nein — nur DSpark mit `sample_from_anchor=false`; unser Checkpoint hat den Schlüssel nicht (Anchor-Sampling), unser Code kennt ihn nicht. |
| #59721 | [Bugfix][Core] Check multimodal chunk boundaries before encoder cache reuse | konflikt | nein — wie #52412. |
| #59942 | [Bugfix][Core] Retain encoder cache references for repeated multimodal inputs | konflikt | nein — nur Mehrfachrechnung/Stau bei doppelten Bildern im selben Schritt; hängt an #57696 (`_reject_on_encoder_cache_embed_mismatch`). |
| #60553 | [Bugfix][Scheduler] Fix encoder cache deadlock with drafter look-ahead | konflikt | nein — der Deadlock entsteht upstream aus der verzögerten Freigabe von #46315, die bei uns fehlt. Mit unserem Schema hinge nur Lücke 0 zwischen zwei Bildern; Qwen/GLM setzen ≥ 2 Trenn-Token (vision_end/vision_start). GLM-DFlash sammelt keine MM-Einbettungen (`dflash/speculator.py:55`). |
| #60767 | [Bugfix][V1] Null sliding-window block-table entries below the window | konflikt | nein — GLM: Entwurfs-SWA-Seite 1152 Token (Boot-Log 06.11. 19:15) = 9 × 128, FA2-Kacheln liegen daher nie auf einer ganz unterhalb des Fensters liegenden, freigegebenen Seite. DSv4: SWA wird über Indexlisten gelesen (`deepseek_v4/sm70/sparse.py`). WARNUNG: unverändert übernommen würde der Patch `KpoolTailSpec` (Unterklasse von `SlidingWindowSpec`, `kv_cache_interface.py:721`, 1 Block je Anfrage) nullen → GLM kaputt. |
| #61000 | [Bugfix][Distributed] Own SHM out-of-band buffers before releasing read slots | konflikt | nein — betrifft nur Out-of-band-numpy-Puffer ≥ 1 MiB (`shm_broadcast.py:758-765`, Leser hält `memoryview` in den Ringslot). Unser Verkehr: keine Logprobs/R3-Arrays, Torch-Tensoren werden in-band gepickelt. |
| #61016 | [Bugfix][MRV2] Keep input_ids on later PP stages for raw-token models | teilweise | nein — DSv4-Hash-Routing-Schichten (brauchen `input_ids`) liegen auf PP0; fehlte es, käme ein lauter ValueError (`deepseek_v4/nvidia/model.py:552-555`). |
| #61044 | [Bugfix] Create cuBLAS handles before memory profiling to avoid OOM | fehlt_passt | nein — `CUBLAS_STATUS_ALLOC_FAILED` bei uns nie beobachtet (Logs, Notizen); Patch passt, würde nur die Handle-Anlage vor den Snapshot ziehen. Bei Auftreten nachziehen. |
| #61174 | [Bugfix][Spec Decode] Clear stale shared block-table rows | konflikt | nein — GLM/DSv4 eine Sequenz; 27B-MTP-Entwurf ohne rekurrenten Zustand; Füllzeilen des Ziels nullt der Gather-Kern (`gpu/block_table.py:140-150`). |
| #61647 | [Bugfix] Engage indexer TP row sharding on total rows, not per-rank rows | konflikt | nein — Indexer-TP-Zeilenteilung; DSv4/GLM laufen mit PP. |
| #62129 | [Bugfix][MRV2] Fix speculative warmup deadlock with pipeline parallelism | konflikt | nein — der zweite Greedy-Warmup-Durchlauf (#59260), den das repariert, fehlt bei uns. |
| #62206 | [Bugfix][Core] Keep the EAGLE-reachable SWA tail at a shared-prefix junction | konflikt | nein — Retention über `shared_prefix_boundary` gibt es bei 1Cat nur für Mamba-Gruppen (`config/cache.py:104-110`, `single_type_kv_cache_manager.py:1596`), nicht für SWA. |
| #63154 | [Bugfix][MRV2] Release the warmup memory cached on the capture streams | konflikt | nein — nur ohne FULL-Graphen (PIECEWISE + breakable); wir fangen FULL-Decode-Graphen. |
| #63165 | [Bugfix] shm_broadcast: poll the notify socket from busy readers | konflikt | nein — Rückstau ≈ 41 B je Nachricht nur, solange ein Leser nie > 1 s untätig ist; jeder Leerlauf leert ihn. Single-User: vernachlässigbar. Braucht #45224 als Kontext (Hunk-Kontext `SHM_READER_RECHECK_INTERVAL_MS`). |
| #65225 | [Bugfix] Pad extensible KV cache segments instead of rounding the block count | konflikt | nein — Extensible-KV-Cache (#56492) fehlt bei uns. |
| #65271 | [Bugfix][Scheduler] Don't split a multimodal item when a prefix cache hit ends inside it | konflikt | nein — wie #52412 (nur `disable_chunked_mm_input`). |
| #65554 | [Bugfix] Fix startup illegal memory access from unaligned extensible KV cache segments | konflikt | nein — wie #65225. |

## Aussortiert nach Titel/Beschreibung (136)

- #43161 [Bugfix] Fix UBatchWrapper CUDA graph key to sum all ubatches, not just first two — UBatch/DBO
- #43637 [Feature] Detect all2all peer fault with fault tolerance backend and prevent corrupted output — EP all2all
- #43931 [Bugfix] V1: clear stale allowed_token_ids mask in InputBatch.condense — V1-InputBatch, allowed_token_ids
- #44002 [Bugfix] Reject negative values for max_logprobs and long_prefill_token_threshold — Config-Validierung
- #44057 [Bugfix] Reject non-positive values for ParallelConfig int knobs — Config-Validierung
- #44070 fix(config): reject negative max_logprobs (except -1) and long_prefill_token_threshold — Config-Validierung
- #44207 fix(config): validate max_num_scheduled_tokens >= 0 on all paths — Config-Validierung
- #44409 [Bugfix] Two-phase KV allocation for cross-group prefix cache hits (supersedes #33775) — externer KV/Connector
- #44560 [BugFix] Resolve multiple async kv load deadlock — async KV-Load (Connector)
- #44568 [Model Runner V2] Fix v2 `AttributeError: 'CohereASRDecoder' object has no attribute 'embed_input_ids'` — CohereASR
- #44603 fix: pad dummy run query_start_loc — V1-Runner Dummy-Run
- #45195 [Bugfix][V1] Clean up compiled-model bytecode hooks on VllmRunner exit — VllmRunner-Exit (offline)
- #45196 [Bugfix][Model] Validate DefaultModelLoader / LoadConfig and fail with clear errors — Loader-Validierung
- #45383 [BugFix] Fix prompt_embeds for multimodal models — prompt_embeds
- #45673 [BugFix] Support async scheduling with prompt embeds for multimodal models — prompt_embeds
- #45849 [BUG] fix hidden states nan for hybrid attention models — HiddenStates-Connector
- #45989 [Bugfix][RL] Set vLLM config during weight reload — RL-Gewichtsupdate
- #46066 [Bugfix][Core] Fix num_output_placeholders underflow with async scheduling + spec decode — reset_prefix_cache/RL/Sleep
- #46220 [Bugfix][Config] Keep pydantic validation for fields with a TYPE_CHECKING Literal alias — pydantic-Literal (Validierung)
- #46236 [Bugfix][Quant] Raise actionable error instead of bare assert for group-size/TP mismatch (#46230) — compressed-tensors Gruppen-Assert
- #46399 [Bugfix][Spec Decode] Disable dynamic speculative decoding for fixed-K proposers instead of crashing — dynamisches Spec-Decoding
- #46627 [Bug] Fix `IndentationError: expected an indented block after 'with' statement` — Syntaxfehler (nicht bei uns)
- #46753 [ModelRunner V2] Fix cross-attention block table sizing — Cross-Attention (Enc-Dec)
- #46974 [BugFix][MRV2] Ensure all req slots are accounted for when scheduling — streaming input / PRIORITY
- #47314 [BugFix] Fix packed HND KV cache reshape for FlashAttention — Offload-Connector
- #47356 [Bugfix] Exclude kv_cache_memory_bytes from CacheConfig.compute_hash — kv_cache_memory_bytes (nicht gesetzt)
- #47428 [ModelRunner V2] Fix Mamba2 crash on non-spec-decode — Mamba2
- #47460 [Bugfix] Initialize draft CUDA-graph keys for the native draft_model proposer — draft_model-Proposer V1
- #47483 [ModelRunner V2][BugFix] Free all model refs on shutdown — Shutdown
- #48132 [Bugfix][MRV2] Reset num_accepted_tokens on add_request in all modes — P/D-Decode-Rolle
- #48135 [Bugfix] Preserve tensor causal metadata for grouped attention — Regression aus #47914 (fehlt)
- #48317 [Bugfix] Count per-group blocks in get_max_concurrency_for_kv_cache_config — Log
- #48341 [Bugfix][Spec Decode] Auto-enable async scheduling for draft models — draft_model async
- #48411 [Bugfix] Include inline per-token-head scales in offloaded page transfer width — KV-Offload
- #48419 [Bugfix] V1: fix allowed_token_ids_mask aliasing in InputBatch.swap_states — V1-InputBatch
- #48622 [Bugfix][R3] Exclude draft routers from expert capture — R3 Expert-Capture
- #48843 [BugFix] Set graph_pool_id before FULL CUDA graph capture in ModelRunner V2 — NCCL symm mem
- #49134 [Bugfix] Reject contradictory custom-op directives — Config-Validierung
- #49204 [Core] Fix internal LB load-balancing — DP-Load-Balancing
- #49206 fix: resolve silent request skipping in PRIORITY scheduling — PRIORITY-Scheduling
- #49302 [Bugfix] Fix DSA crash under breakable piecewise cudagraphs — DSA + breakable piecewise (DeepGEMM/TMA)
- #49751 [BugFix][MRV2] Don't create dummy requests longer than `max_model_len` — Dummy-Länge > max_model_len (bei uns ≤ 2048)
- #49774 [Bugfix][Spec Decode] Preserve draft buffers across level-2 sleep — Sleep-Mode
- #49821 [Bugfix][CLI] Include inherited field docstrings in get_attr_docs — CLI-Doku
- #50020 [Bugfix][MRV2] Support encoder timing stats in model runner V2 — Encoder-Metriken
- #50302 [Bugfix] Universally align block table width to 128 tokens — Blocktabellen-Ausrichtung DCP/DSA-Indexer
- #50344 [BugFix] Scope divergent hybrid cache hits to capable connectors — Connector
- #50388 [Core] Fix ValueError on KV load failure with a hybrid KV cache — Connector
- #50462 [Bugfix][Core] Log KV cache capacity after block-size resolution — Log
- #50488 [Bugfix][Spec Decode] Capture the widest uniform decode batch by default — Default-Capture-Größen (wir explizit)
- #50769 fix(config): apply presence_penalty/frequency_penalty from override-generation-config — override-generation-config (nicht gesetzt)
- #50809 [Bugfix][V1] Sync mamba_block_size via EngineCoreReadyResponse — Metrik
- #50894 [Bugfix] Scale KV page size for hidden states extraction with TP — Hidden-State-Extraktion
- #50969 [BugFix] Bind RayExecutorV2 TCPStore before publishing its port — Ray
- #51108 [BugFix][KV Cache] Fix hybrid prefix caching with hidden-state extraction — Hidden-State-Extraktion
- #51218 [Bugfix] Report FULL_ATTENTION for uniform-base UniformTypeKVCacheSpecs groups instead of UNKNOWN — Connector/Metrik-Art
- #51260 [Bugfix] Skip fetching revision for model when model and weights_model are different — weights_model
- #51304 [V1] Copy NaN-in-logits counts to host asynchronously — V1-Runner
- #51438 [Bugfix][MRV2] Reserve spec-decode lookahead blocks in V2 warmup — V2-Warmup-Lookahead (Warmup läuft)
- #51469 [BugFix] Reject invalid data-parallel RPC ports — DP-RPC-Ports
- #51482 [BugFix][Core] free_blocks: restore prepend (LIFO) reuse order when prefix caching is off — Präfix-Cache aus
- #51665 Fix weight tying — Weight tying (unsere Configs untied)
- #51678 [Bugfix][Profiler] Fix API server crash on double /stop_profile — Profiler
- #51768 [Bugfix] Guard DeepSeek V4 MRV1 piecewise CUDA graphs — DSv4 MRV1
- #51795 [Bugfix] Reject negative token ids as out-of-vocabulary — Validierung
- #51899 [Core][BugFix] Tag prefix-cache extra keys by source — LoRA/cache_salt-Schlüssel
- #51953 [Bugfix][MRV2] Fix DiffusionGemma runtime OOM via tiled logits projection — DiffusionGemma
- #52161 [Bugfix] Detect all attention-spelling variants in ModelConfig.is_hybrid — is_hybrid granite-micro
- #52401 [Bugfix] Pick the DeepSeek V4 eager cudagraph region per model runner — DSv4 MRV1/ROCm
- #52707 [Bugfix][KV Cache] Prevent negative external block allocation — Connector
- #54162 [Bugfix][MRV2] Release model and KV cache on in-process engine shutdown — Shutdown
- #54246 [Bugfix][MRV2] Release layer-bound KV cache memory in shutdown() — Shutdown
- #54418 [Bugfix][Spec Decode] Keep default CUDA graph sizes memory-safe — Default-Capture-Größen
- #54782 [Bugfix] Raise for unavailable piecewise CUDA graphs — Fehlermeldung
- #54803 [Bugfix] `adjust_dcp_kv_cache_interleave_size` for NixlConnector only — DCP/Nixl
- #55146 [Bugfix][V1] Honor enable_jit_warmup for V2 kernel warmup — enable_jit_warmup=False
- #55234 [Bugfix][MLA] Restore DSpark cache-group capability under optimized Python — python -O
- #55237 [Bugfix] Fix cuda profiler missing bug — Profiler
- #55390 [Bugfix] Annotate MTP draft KV cache groups positionally on the hybrid grouping path — Offload-Connector-Annotation
- #55455 [Bugfix] Defer adaptive verification until after kernel warmup — adaptive verification (fehlt)
- #55650 [Bugfix] Make KV cache and MFU log lines backend-neutral — Log
- #55702 [Bugfix] Remove unsupported comma-separated detailed trace values — Trace-Optionen
- #55949 [Bugfix] Handle null RoPE parameters for NoPE layers — NoPE/Cohere
- #56079 [MoE][Bugfix] Skip SP padded rows in grouped MoE routing — Sequence Parallelism
- #56791 [Bugfix] Trim stale consequence claims from unannotated-eagle warning — Warnungstext
- #56844 [Bugfix] Validate routed-expert prompt offsets before engine submission — routed experts
- #56925 [Bugfix] Restore KV cache metadata GET method for external event consumer — KV-Events
- #57006 [Bugfix][Frontend] Validate mixed prompt embedding mask lengths — prompt_embeds
- #57355 [Bugfix] Max-load throughput cliff when `max_num_seqs` is not a multiple of 8 — Default-Capture-Größen
- #57356 [bugfix] Mark draft tokens to rebuilt their embeddings. — prompt_embeds V1
- #57447 [Bugfix][Scheduler] Preserve logprobs across streaming continuations — streaming input
- #57676 [Bugfix][Scheduler] Refresh max tokens for streaming continuations — streaming input
- #57757 [Bugfix] Clean up async client sockets after event loop closure — Client-Sockets beim Schließen
- #57988 [Bugfix] Release prompt_embeds tensor when its InputBatch slot is freed — prompt_embeds
- #58021 [Bugfix] Reject a prefix_match_unit that a single KV cache group cannot honor — prefix_match_unit/FlashKDA
- #58124 [Bugfix][Model Loader] Support reload_weights with runai_streamer load format — runai
- #58143 [Bugfix] Preserve padding token types for cross encoders — Cross-Encoder
- #58149 [Bugfix][V1] Read ModelState max_model_len from model config — max_model_len auto-fit
- #58212 [Bugfix] Skip VllmConfig re-validation for with_hf_config submodel views — NIXL/Submodel-Config
- #58243 [Bugfix][MRV2] Reserve encoder graph memory without decoder graphs — Encoder-Graphen ohne Decoder-Graphen
- #58259 [Bugfix] Fix resumable request + async scheduling handoff race — resumable sessions
- #58275 [Bugfix] Capture prefill kernels for mixed FULL graphs — FULL-mixed-Capture
- #58287 [Bugfix] Disable prefix caching for encoder-only before model config hooks — Encoder-only
- #58288 [Bugfix][Core] Keep every multimodal feature in the partial-block KV event — KV-Events
- #58462 [Bugfix][MRV2] Align dummy idx_mapping dtype to avoid runtime jit — Folge #57980 (fehlt)
- #58593 [Bugfix] Keep JIT warmup under enforce-eager when fault tolerance is on — Fault Tolerance
- #58611 [Bugfix][FT] Pass stateless process-group timeouts explicitly — Fault Tolerance
- #58744 [Bugfix][MLA] Gather nvfp4_ds_mla chunked-context KV with the NVFP4 kernel — NVFP4-KV (wir fp8)
- #58778 [Bugfix][Core] Keep resumable sessions that stop while preempted — resumable sessions
- #58918 [Bugfix][Compile] Replace Transformers AOT caching with graph caching — Transformers-Backend
- #58941 [Bugfix][Frontend] Preserve streaming chunk boundaries in RequestOutputCollector — streaming input
- #58942 [Bugfix] Terminate streaming sessions on ABORT/ERROR and prevent zombie sessions — streaming input
- #59015 [Bugfix][Frontend] Avoid generation for empty streaming input — streaming input
- #59029 [Bugfix][Core] Schedule encoder-only prompts larger than one step — Encoder-only
- #59103 [Bugfix][Core] Drop stale block hashes when a session truncates tokens — Sessions
- #59504 [Bugfix][Core] Exempt exactly the blocks an async KV load writes from zeroing — async KV-Load (Connector)
- #59541 [Bugfix][Spec Decode] Keep heterogeneous-vocab draft models on Model Runner V1 — heterogenes Entwurfsvokabular
- #59695 [Bugfix][Multimodal] Fix reference counting of the SHM processor cache — SHM-Prozessorcache (nicht genutzt)
- #60098 [Bugfix] Use FusedMoEWithLoRA for shared MoE LoRAs on 3D models — LoRA
- #60240 [Bugfix] Warn only on per-request do_normalize/do_rescale overrides — Warnung
- #60271 [Bugfix] Check batch-invariance support for every attention backend — Batch-Invarianz
- #60306 [Bugfix][Core] Count unscheduled running requests toward max_loras — LoRA
- #60412 [Bugfix] Don't warn about missing watermark config for default requests — Warnung
- #60447 [Bugfix][MoE] Size the permute scratch from its inputs — Humming/CUTLASS-MoE
- #60714 [Bugfix] Reject max_num_scheduled_tokens=0 at config time instead of hanging — Config-Validierung
- #60966 [Fast Start][Bugfix] Snapshot GPU memory after external weights are ready — Fast Start
- #61509 [Bugfix][MRV2] Don't count adaptive verification profiling cache as CUDA graph memory — adaptive verification (fehlt)
- #61976 [Bugfix][Core] Keep scheduling running requests after a priority self-preemption — PRIORITY
- #62164 [Bugfix] Fix multimodal sender-cache invalidation race with queued renders — Folge #46747
- #62880 [Bugfix] Don't enable MM embedding inputs on KV consumers in language-model-only mode — KV-Consumer
- #63224 [Bugfix][KV Events] Prevent replay truncation when the send queue fills — KV-Events
- #63431 [Bugfix][MRV2] Skip adaptive verification cost profiling in the CUDA graph memory-profiling pass — adaptive verification (fehlt)
- #63853 [Bugfix][Spec Decode] Apply the draft max_model_len limit before model config hooks — Prefix-Cache aus + Entwurf kürzer
- #65127 [Bugfix] Validate global model seed range — Seed-Validierung
- #65399 [Bugfix][Scheduler] Respect disabled chunked prefill for multimodal requests — Chunked Prefill aus
- #65761 [Bugfix][Multimodal] Encoder CUDA graph: cap each graph's frame slots at its token budget — Encoder-Graphen mit Video (bei uns video 0)
