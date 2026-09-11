# Overlay-Inventur: work-main gegen 1Cat main

**Stand 2026-09-10.** Grundlage ist `git diff origin/main work-main` im Worktree
`1Cat-vLLM-work` (origin/main = fe67339d, work-main = 82301e6b): 73 Dateien,
+5.837/−406 Zeilen. Davon sind 8 Testdateien unserer offenen PRs, eine ist
`setup.py`, 64 liegen unter `vllm/`.

Die Frage war: Was davon ist PR-würdig, was ist lokal, was ist überholt? Beim
Durchgehen kamen Merge-Reste zum Vorschein. Die stehen vorne, weil sie vor jedem
PR aufgeräumt sein müssen, und zwei davon ändern Verhalten.

Methode für die Merge-Reste: Für jede Datei wurden die hinzugefügten Zeilen gegen
die Upstream-Fassung derselben Datei abgeglichen (Zeilen > 30 Zeichen, ohne
Kommentare). Hohe Überdeckung heißt: Wir fügen hinzu, was Upstream schon hat.

---

## 1. Merge-Reste (vor jedem PR aufräumen)

| # | Datei | Befund | Wirkung | Vorschlag |
|---|---|---|---|---|
| 1 | `vllm/_custom_ops.py` | Zwei verirrte Zeilen seit dem Overlay vom 06.09. (beim 1.5.0-Rebase automatisch gemergt): `output_gate_activation: str = "silu",` mitten im Docstring von `swap_blocks_batch`, und `output_gate_activation,` als zweiter Pflichtparameter in `gather_and_maybe_dequant_cache`. Keine Op nutzt den Namen. | **Bug.** Der einzige Aufrufer (`mla_attention.py`, Chunked-Context-Prefill) übergibt Keywords → `TypeError`, fehlendes Argument. Trifft klassische MLA-Modelle (DeepSeek V2/V3, Kimi, GLM-MLA) bei langem Prompt. Unsere Modelle nicht (DeepSeek V4 hat eigene Sparse-Attention). | Beide Zeilen raus, Datei ist dann gleich Upstream. |
| 2 | `vllm/config/speculative.py` | Der Qwen4Exp-MTP-Block steht zweimal in `hf_config_override`. Unser älterer Block läuft als zweiter und überschreibt `index_share_for_mtp_iteration` mit `False` (Upstream: `True`, Commit 70b63a1e „Reduce Qwen4Exp MTP cost"). Zusätzlich ist der Prüfblock für `index_share_for_mtp_iteration` in `num_speculative_state_tokens()` gerutscht. | **Verhaltensänderung.** Flash-Next läuft zwingend über Model Runner V2; dessen Speculator nutzt die QSA-Indizes aus Schritt 0 für die MTP-Schritte 1+ nur bei `True`. Unsere Produktion hat diese Upstream-Optimierung also abgeschaltet. Der Checkpoint setzt den Wert nicht selbst. | Unseren Block und den verrutschten Prüfblock entfernen, dann A/B auf Flash-Next. Die geteilten Indizes betreffen nur die Entwürfe; der Zielkopf prüft bei greedy jeden Token, der Text sollte also gleich bleiben und nur Akzeptanz und Tempo sich ändern. Garantiert ist das ohne batch-invariante Kernel nicht, darum SHA prüfen. **Entscheidung Peuqui.** |
| 3 | `vllm/model_executor/models/config.py` | `_strip_qwen4_exp_mrope` und `Qwen4ExpForConditionalGenerationConfig` doppelt definiert, dazu doppelte Schlüssel in `MODELS_CONFIG_MAP`. | Harmlos: beide Fassungen sind inhaltlich gleich (nur Formatierung). | Unseren Block entfernen. |
| 4 | `vllm/v1/worker/gpu_model_runner.py` | `self.drafter = None` auf Nicht-letzten PP-Rängen doppelt; Upstream hat es seit unserem #511. | Harmlos. | Unsere Zeilen raus. |
| 5 | `vllm/config/vllm.py` | `_any_visible_device_has_capability` wird nirgends aufgerufen. Vor #514 geschrieben; Upstream nutzt `_any_participating_device_is_capability`. | Toter Code. | Raus. |
| 6 | `vllm/v1/core/kv_cache_utils.py` | Unser CSA+Linear-Zweig steht **vor** dem inzwischen gelandeten Upstream-Zweig. In `get_kv_cache_config_from_groups` wird der Upstream-Zweig dadurch nie erreicht. In `_max_memory_usage_bytes_from_groups` rechnen beide verschieden: wir nehmen das Maximum der Gruppenseiten × Bytes pro Block, Upstream die Summe über die Gruppen. | Die Zuteilung ist identisch: beide Zweige rufen dieselbe Funktion `_get_kv_cache_config_csa_linear`, und für Qwen4Exp greift keiner der Zweige dazwischen. Anders ist nur die Zulassungsprüfung beim Start. Alle Gruppen ziehen ihre Blöcke aus einem gemeinsamen Pool, eine Anfrage braucht also die Summe; unser Maximum unterschlägt die wenigen Blöcke der Mamba- und Ringgruppen und lässt etwas zu viel zu. | Unseren Zweig entfernen, Upstreams Summe gilt. Kein Einfluss auf den Text. Einziges Risiko: Flash-Next bootet mit der heutigen `max-model-len` nicht mehr, falls die Rechnung knapp war. Boot-Test. |
| 7 | `vllm/v1/core/single_type_kv_cache_manager.py` | `CircularBufferSpec: CircularBufferManager` doppelt in `spec_manager_map`. | Harmlos. | Unsere Zeile raus. |
| 8 | `vllm/v1/worker/gpu/model_states/mamba_hybrid.py` | Einziger Diff ist ein Kommentar über eine „zweite Bedingung" für den sm75-GDN-Builder; die Bedingung selbst ist nicht mehr da. Den Builder gibt es seit dem gemeinsamen GDN-Pfad (09.09.) nicht mehr. | Verwaister Kommentar. | Raus, Datei ist dann gleich Upstream. |
| 9 | `vllm/v1/worker/gpu_model_runner.py` | E5-Metadaten-Cache, rund 900 Zeilen: Vorgabe **an** (`VLLM_SM70_E5_CACHE` Standard `"1"`). Alle sieben vLLM-Einträge in llama-swap schalten ihn mit `=0` ab, Upstream hat ihn entfernt, auf CSA/QSA-Modellen stürzte er ab (#413). | Wer unseren Stand ohne unsere Umgebung startet, bekommt den abgeschalteten Pfad. | Vorgabe auf aus, oder ganz ausbauen samt E5-Werkzeugen (`E5_PROF`, `E5_NVTX`, `E5_DIFF`). **Entscheidung Peuqui.** |

Zu 1 bis 8: Keine dieser Bereinigungen sollte den erzeugten Text ändern;
Nr. 2 ändert Akzeptanz und Tempo, Nr. 6 kann höchstens den Start mit knapper
`max-model-len` verweigern.
Nach dem Aufräumen genügt deshalb die übliche Abnahme (Text-SHA, Tempo) auf
27B, Flash-Next und DeepSeek.

---

## 2. Einordnung aller Dateien

### A. Schon als PR bei 1Cat offen (nur nachziehen, wenn gemergt)

| PR | Dateien |
|---|---|
| #572 Turing bootbar | `config/vllm.py` (Teil: pre-Ampere-Gate), `layers/mamba/gdn/qwen_gdn_linear_attn.py`, 3 Tests |
| #573 Qwen4Exp-MTP stufenlokal | `models/qwen4_exp/nvidia/mtp.py`, 1 Test |
| #574 Output-Trim auf allen PP-Rängen | `v1/worker/gpu_model_runner.py` (ein Hunk), 1 Test |
| #576 SM70-Quant-Gate pro Gerät | `layers/quantization/sm70_turbomind.py`, 2 Tests |
| #592 DFlash quantisierter Entwurfskopf | `models/qwen3_dflash.py`, 1 Test |

### B. Editable-Bau (Punkt 8)

| Datei | Inhalt |
|---|---|
| `setup.py` | fünf pybind11-Module brauchen `py_limited_api=False`, sonst bricht `pip install -e .` nach 45–70 min beim Kopieren ab. Der Weg aus #320 (`USE_SABI 3`) geht bei pybind11 nicht. Entwurf: `03-1cat-issues/pr-editable-soabi-modules.md` |
| (Paketierung) | `flash_attn_v100` fällt wegen des absoluten `package_dir` aus dem editable Mapping; die GDN-Erweiterung landet nur in `build-lib` |

### C. Gemischte Hardware und Turing (Punkt 11)

| Datei | Inhalt |
|---|---|
| `vllm_flash_attn/flash_attn_interface.py` | FA2-Bibliothek pro Gerät laden (sm70-Neufassung und sm75-Build teilen sich den Modulnamen `_vllm_fa2_C`), Fähigkeit worker-lokal |
| `v1/attention/backends/flash_attn_v100.py` | lädt die passende FA2 vor dem d256-Pfad |
| `v1/attention/backends/fa_utils.py` | FA-Version pro Worker statt von Gerät 0 |
| `v1/attention/backends/flash_attn.py` | sm75: FA2 ab Fähigkeit 7.5, nur fp16 |
| `platforms/cuda.py` | Turing: FLASH_ATTN vorziehen, FlashInfers Paged-Prefill scheitert dort |
| `models/qwen3_dflash2.py` | Bereichserhaltender Pfad für alles unter SM80, entschieden am Gerät des Workers (Turing: Akzeptanz 1,015 → 3,353) |
| `v1/attention/ops/triton_unified_attention.py`, `v1/attention/backends/triton_attn.py` | 3D-Split-KV auch für Spekulations-Verify, 64 statt 16 Softmax-Segmente (RTX bei 31k Kontext: 2,6 → 58 tok/s). Vorgabe an. Vor dem PR prüfen, welche heutigen Konfigurationen noch über Triton-Attention laufen |

Außerhalb des 1Cat-Baums nötig: die sm75-FA2-Bibliothek
(`_vllm_fa2_C_sm75.abi3.so` aus unserem flash-attention-Fork) und der Bauhinweis
`TORCH_CUDA_ARCH_LIST=7.0` (bei `7.0;7.5` fällt SM70-Marlin samt MoE weg).

### D. DeepSeek V4 auf Pre-Ampere und gemischter Hardware

| Gruppe | Dateien | Inhalt |
|---|---|---|
| D1 „kein natives FP8" statt „genau SM70" | `deepseek_v4/common/ops/cache_utils.py`, `…/fused_compress_quant_cache.py`, `…/fused_indexer_q.py`, `v1/attention/ops/rocm_aiter_mla_sparse.py`, `deepseek_v4/nvidia/dspark.py` (Skalierungs- und Fused-Op-Gate) | Die sm75-Stufen nahmen den Hardware-FP8-Pfad, den es dort nicht gibt |
| D2 Fähigkeit worker-lokal | `v1/attention/backends/mla/sparse_swa.py`, `deepseek_v4/sm70/gemv.py`, `v1/attention/backends/mla/flashmla_sparse.py`, `deepseek_v4/attention.py` (O-Projektion) | Gerät 0 entschied für alle Stufen |
| D3 fp16-Korrektheit | `deepseek_v4/nvidia/model.py` (Aux-Strom in fp32), `deepseek_v4/nvidia/dspark.py` (BOS-Überlauf; Akzeptanz ~5 % → normal), `kernels/mhc/tilelang.py` (Torch-Referenz für `hc_head` unter SM80, fp32-Post) | `--dtype half` sprengt den fp16-Bereich |
| D4 Pipeline-Parallelität | `config/speculative.py` (Guard: V4-MTP nicht auf V3 umbiegen), `deepseek_v4/nvidia/mtp.py` (SupportsPP, Quantisierung aus dem Draft-Checkpoint), `deepseek_v4/nvidia/dspark.py` (SupportsPP), `v1/executor/multiproc_executor.py` (PP-Batch-Warteschlange deckeln, sonst Deadlock bei PP5) | Ohne PP passt DeepSeek hier nicht. SupportsPP: prüfen, ob 1Cat es seit „Draft-PP=1" noch braucht |
| D5 Ersatzpfade vor Hopper | `deepseek_v4/attention.py` (fp8_einsum-Referenz), `layers/sparse_attn_indexer.py` (DeepGEMM-Logits-Referenz), `deepseek_v4/compressor.py` (Triton statt CuteDSL, wie AMD), `utils/import_utils.py` (CUTLASS-DSL kennt erst sm_80), `deepseek_v4/nvidia/model.py` (`scale_fmt` optional für compressed-tensors) | 1Cat hat eigene SM70-Pfade für DeepSeek. Vor einem PR prüfen, welcher unserer Ersatzpfade auf frischem main überhaupt noch erreicht wird |
| D6 allgemeine Bugfixes | `deepseek_v4/amd/rocm.py` (SWA-Kopie nach tatsächlicher Zeilenbreite; Drafting-Zeilen sind breiter als das Fenster), `fused_moe/experts/nvfp4_emulation_moe.py` (nur geroutete Experten dequantisieren, in Portionen: OOM → läuft) | hardwareunabhängig |

### E. Skinny (Punkt 9), hängt am Kernel außerhalb des 1Cat-Baums

Kernel: `v100-skinny/kernels/skinny_kernels.cu`, JIT über `VLLM_SKINNY_NVFP4_SRC`.

| Datei | Inhalt |
|---|---|
| `kernels/linear/nvfp4/marlin.py` | Skinny-NVFP4 für M ≤ 64, Dense-Prefill |
| `kernels/linear/nvfp4/qpn_dequant.py` | Triton: QPN-Prepack → dichtes fp16 für Prefill über cuBLAS |
| `kernels/linear/scaled_mm/qpn8_blk.py`, `kernels/linear/__init__.py` | block-skaliertes FP8 über QPN8 |
| `layers/quantization/fp8.py`, `…/compressed_tensors/compressed_tensors.py`, `…/modelopt.py` | Routen QPN8 und TurboMind |
| `fused_moe/experts/nvfp4_skinny_moe.py`, `fused_moe/oracle/nvfp4.py`, `config/kernel.py` | Skinny-MoE-Backend, als `sm70_skinny` wählbar |
| `compilation/breakable_cudagraph.py` | Break auch im FULL-Modus für hostgesteuerte Ops |
| (`envs.py`) | fehlt: `VLLM_SKINNY_*` anmelden, im Skinny-PR |

Reihenfolge wie besprochen: erst den Routenzähler auf den
Produktionskonfigurationen, dann (a) Turing-Pfad für 1Cats QPN-Kopie, (b)
Block-Pack, (c) MoE-Backend erst nach Rückfrage in #441.

### F. Qwen4Exp / Flash-Next

| Datei | Inhalt | Einschätzung |
|---|---|---|
| `v1/kv_cache_interface.py` | `head_size_v` in die Basisklasse, CircularBufferSpec entscheidet Uniformität selbst, Mamba-Erkennung in UniformType-Gruppen | mit Befund 6 zusammen prüfen |
| `v1/core/kv_cache_utils.py` | nur prefix-cachebare Gruppen ins Hashing; CSA-Zweig | siehe Befund 6 |
| `v1/core/single_type_kv_cache_manager.py` | Ringpuffer nie in den Nullungs-Pfad | prüfen, ob Upstream das inzwischen selbst ausschließt |
| `v1/attention/backends/utils.py` | Triton-Kernel für den align-Blocktabellen-Gather der Mamba-Zustände | Herkunft prüfen, sieht nach einem vLLM-Backport aus |
| `transformers_utils/model_arch_config_convertor.py` | Schichtzahl für Qwen4ExpMTP | klein, PR-fähig |
| `v1/spec_decode/llm_base_proposer.py` | Qwen4ExpForConditionalGeneration in die Kopf-Teilen-Liste | klein, PR-fähig |
| `v1/worker/gpu/spec_decode/eagle/utils.py` | LM-Kopf über die Hilfsfunktion finden; bei `…ForConditionalGeneration` blieb der MTP-Kopf sonst ungeteilt | allgemeiner Bugfix, PR-würdig |
| `models/utils.py` | `shard_id` im AutoWeightsLoader durchreichen, `get_rename_mapper` | Upstream hat Stacked-Shard selbst gebaut; prüfen, ob unsere Durchreichung noch nötig ist |
| `v1/ple_offload/worker.py`, `v1/worker/gpu_worker.py` | PLE unter PP: Offload-Kind isoliert, Gate nach Partition | #479-Serie; Offload auf dem Mini nicht testbar (30 GB RAM) |
| `models/qwen4_exp/nvidia/ple_layer.py` | nur ein Docstring | raus oder mit dem nächsten #479-PR |
| `v1/worker/gpu_model_runner.py` (V1-Teile) | PLE-Eingaben, Short-Conv-Builder, gemischte Mamba-Kopierfunktionen | Qwen4Exp läuft seit 1Cat-main zwingend über V2. Im V1-Runner vermutlich tot, prüfen und ausbauen |
| `v1/attention/backends/gdn_attn.py` | Chain-MTP-Schnellbau (Vorgabe aus, −1,4 ms/Schritt, bitgleich), Slot-Debug | in Produktion nicht gesetzt. Entweder messen und als Vorgabe an, oder raus |

### G. Allgemein PR-würdig

| Datei | Inhalt | Einschätzung |
|---|---|---|
| `distributed/parallel_state.py` | NCCL-Untergruppen bekommen `--distributed-timeout-seconds`; bisher behielten TP/PP-Gruppen PyTorchs 600 s, ein kalter PP-Boot riss am Wachhund | klarer Bug, klein |
| `utils/torch_utils.py`, `layers/attention/attention.py` | Die KV-Quant-Angabe eines Checkpoints gilt unter SM80 nicht mehr als Anweisung (V100, 27B: +4,82 ms/Runde durch FP8-KV ohne FP8-Hardware) | Politikfrage, gut belegt |
| `config/vllm.py` | erzwungenes `VLLM_DISABLE_COMPILE_CACHE=1` für den 0DOT3-Compile-Graph entfernt; die Ursache ist seit #536 behoben | klein, direkte Folge von #536 |
| `models/qwen3_5_mtp.py` | SupportsPP für Qwen3_5MTP, Verzweigung nach Eingabe statt PP-Rang | Upstream fehlt es noch, PR-würdig |
| `v1/core/kv_cache_utils.py` | Fehlermeldung nennt den (negativen) Speicherwert | Kleinigkeit, eventuell mitnehmen |

### H. Diagnose und Experimente, lokal lassen

| Datei | Inhalt |
|---|---|
| `distributed/device_communicators/custom_all_reduce.py` | AllReduce-Verweildauer per CUDA-Events (`VLLM_SM70_AR_EVT`) |
| `v1/worker/gpu_model_runner.py` | E5-Profiler, NVTX, Diff (siehe Befund 9), `STAGED_PREP_SPEC_FORCE`, `GDN_SLOT_DEBUG`, `MTP_THINK_ONLY` (Spekulation nur im Denkteil, Vorgabe aus, nicht in Produktion) |
| `v1/attention/backends/gdn_attn.py` | Slot-Debug |
| `v1/worker/gpu_worker.py` | KV-Verfügbarkeit pro Rang loggen |

---

## 3. Was einem Nachbauer außerhalb von 1Cat fehlt

1. Skinny-Kernel `kernels/skinny_kernels.cu` (Paket E).
2. sm75-FA2-Bibliothek aus unserem flash-attention-Fork (Paket C).
3. Bauhinweise: `TORCH_CUDA_ARCH_LIST=7.0`; `CPATH` auf die CCCL-Header der venv,
   wenn das nvcc-Paket keine mitbringt.
4. TileLang ≥ 0.1.12 statt 1Cats Pin 0.1.10 (Punkt 10, Test auf V100 und RTX
   steht aus).

## 4. Nächste Schritte

1. Merge-Reste 1, 3, 4, 5, 7, 8 bereinigen (verhaltensneutral), Abnahme.
2. Befund 2 und 6 mit A/B auf Flash-Next klären, Befund 9 entscheiden.
3. PR-Pakete in dieser Reihenfolge: B (editable), G (Timeout, Compile-Cache),
   C (gemischte Hardware), D (DeepSeek), E (Skinny, nach dem Routenzähler).
   Jeder PR gegen frisch gefetchtes main, AGENTS.md frisch lesen.
