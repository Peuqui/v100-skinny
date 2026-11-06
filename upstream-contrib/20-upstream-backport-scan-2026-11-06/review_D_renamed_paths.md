# Bereich D: Bugfixes an umbenannten oder verschobenen Pfaden (Stand 06.11.2026)

Geprüfter Baum: `/home/mp/Projekte/vllm-research/1Cat-vLLM-main-v6` (HEAD 5f1c557fb), nur gelesen. Liste:
`area_D_renamed_paths.json` (211 PRs). #49139 ist laut Auftrag schon im Einbau und wurde nicht geprüft.

Grundlagen, im Code belegt:
- **Parser-Architektur weicht grundsätzlich ab.** Upstream hat die DeepSeek-, Qwen3- und GLM-Parser auf die neue ParserEngine
  umgestellt (`vllm/parser/<modell>.py`, `vllm/parser/engine/`). Upstreams `vllm/tool_parsers/` enthält zum Stand #54838
  (300f68832) nur noch `deepseekv4_engine_tool_parser.py`, `qwen3_engine_tool_parser.py` usw. Unser Fork nutzt die alten Klassen:
  Tool-Parser `deepseek_v4` → `DeepSeekV4ToolParser(DeepSeekV32ToolParser)` (`tool_parsers/__init__.py:37-40`,
  `deepseekv4_tool_parser.py:14-23`), `qwen3_coder` → `qwen3coder_tool_parser.py`, `glm47` → `glm47_moe_tool_parser.py`
  (erbt `glm4_moe_tool_parser.py`). Reasoning `deepseek_v4` → `DeepSeekV3ReasoningParser` (`reasoning/__init__.py:31-34`),
  `qwen3` → `qwen3_reasoning_parser.py`, GLM `deepseek_r1`. Alle Parser-Patches dieser Liste ändern nur Engine-Dateien und
  passen deshalb nie wörtlich. Geprüft wurde jeweils, ob derselbe Fehler in unserer alten Klasse steckt.
- **GLM-Bildteil:** unser Port liegt in `vllm/models/glm5next/nvidia/multimodal.py` (upstream `common/multimodal.py`), mit
  eigenem Prozessor `vllm/transformers_utils/processors/glm5next.py` (Stand vor upstream #57387).
- **AIfred** verkleinert hochgeladene Bilder auf höchstens 3840 px Kantenlänge (`aifred/lib/vision_utils.py:607-660`,
  `VISION_MAX_IMAGE_DIMENSION = 3840` in `aifred/lib/config.py:1780`). Es setzt `enable_thinking` immer ausdrücklich und
  `reasoning_effort` nur als Stufe, nie `"none"` (`aifred/lib/agent_settings.py:223-230`, `aifred/backends/base.py:413-419`).
  `parallel_tool_calls` sendet es nicht.

## Rangliste der Empfehlungen (übernehmen / anpassen)

| # | PR | Kurztitel | Warum es uns trifft | Beleg im Code | Empfehlung | A/B nötig | Abhängigkeiten |
|---|---|---|---|---|---|---|---|
| 1 | #59565 | GLM-5.3: Encoder-Cache aus der exakten Token-Obergrenze | Die Größenprobe ist quadratisch: 89 × 89 = 7921 Token statt 8000. Der Encoder-Cache wird auf 7921 angelegt (Produktionslog 06.11. 22:37: „Encoder cache will be initialized with a budget of 7921 tokens“). Bilder mit 7922 bis 8000 Token lehnt der Server mit 400 ab. Mit unserer `smart_resize` nachgerechnet: 3840×2160 → 7973 Token, 3840×2880 (AIfreds Verkleinerung eines 4032×3024-Handyfotos) → 7931, 3508×2480 → 7950. Jedes 4K-Bildschirmfoto und jedes Handyfoto im Format 4:3 scheitert also an GLM. Im Journal seit 01.11. kein Treffer, der Bildteil läuft erst seit 05.11. | Geerbte Probe `model_executor/models/glm4_1v.py:948-961` (Quadrat 9999999 × 9999999), `Glm5NextProcessingInfo` (`glm5next/nvidia/multimodal.py:668-765`) überschreibt sie nicht. Budget `v1/core/encoder_cache_manager.py:312-314`, Ablehnung `v1/engine/input_processor.py:459-467` | anpassen: die Upstream-Methode `get_image_size_with_most_features` wörtlich hinter `_get_video_max_pixels` (`multimodal.py:707`) einfügen und `import math` ergänzen. Nur Import- und Folgekontext weichen ab, `git apply --check` mit umgeschriebenem Pfad scheitert an Zeile 2. Die neue Leinwand 2240×2800 übersteht unsere `smart_resize` unverändert (nachgerechnet 8000 Token) | Tempo nein. Bootprüfung ja: Der Profil-Dummy wächst von 7921 auf 8000 Token (+1 % ViT-Spitze auf PP0), danach KV-Kapazität ≥ 524 288 bestätigen. Bildprobe 3840×2160 | vor #59126 (Upstream-Reihenfolge, gleiche Datei). Der Upstream-Test `tests/models/multimodal/processing/test_glm5next.py` fehlt bei uns, also kleinen eigenen Test schreiben |
| 2 | #55954 | DSML-Aufruf ohne `<｜DSML｜tool_calls>`-Öffner | Upstream-Issue #48931 beschreibt genau unseren Code (`deepseekv32_tool_parser.py`, gleiche Stellen): DeepSeek-V4-Flash lässt ab etwa 95k Kontext gelegentlich den Öffner weg. Dann geht der ganze Block als Text an den Client, und es gibt keinen Tool-Aufruf. Wir fahren bis 500k Kontext | Schnellprüfung nur auf den Öffner `deepseekv32_tool_parser.py:191`, Streaming `:409` | anpassen (eigene Umsetzung im alten Parser): `<｜DSML｜invoke name="…">` außerhalb eines Blocks als impliziten Blockstart werten, im Streaming die Teilmarken-Rückhaltung (`partial_tag_overlap`) für beide Marken. Die Fälle aus dem Upstream-Test `tests/tool_parsers/test_deepseekv4_tool_parser.py` (z. B. `test_missing_tool_calls_wrapper_is_recovered`) passen auf unsere Parser-API | nein (reiner Parser, CPU-Tests) | nach #54838, vor #56141. Es ist eine Toleranz für fehlerhafte Modellausgabe, deshalb wegen der No-Fallback-Regel kurz mit Peuqui abstimmen. Vorbild im Fork: `_repair_param_dict` (`:136-155`) |
| 3 | #54838 | DSML-Parameter implizit schließen | Fehlt `</｜DSML｜parameter>`, frisst der Wert den nächsten Parameter. Nicht-streamend matcht `(.*?)</｜DSML｜parameter>` bis zum Schließer des Folgeparameters, streamend sucht der String-Modus nur den Schließer. Upstream: 64 von 64 Proben mit fehlendem Schließer verloren oder verfälscht (Issue #53227) | `deepseekv32_tool_parser.py:90-93` (Regex), `:451` (Streaming) | anpassen: Lookahead `(?=<｜DSML｜parameter\s+name=)` in `parameter_complete_regex`. Im Streaming den nächsten Parameter-Öffner als Wertende werten und seine Teilmarke zurückhalten (das ist der Implicit-Close-Teil, den upstream erst in #58780 nachzog) | nein | zuerst von den vier DSML-PRs |
| 4 | #56271 | DSML-Parameter ohne `string=` | Unsere Regexe verlangen `string="(true|false)"`. Fehlt das Attribut, verliert der Aufruf nicht-streamend den Parameter. Streamend findet `parameter_start_regex.match` keinen Treffer, der Puffer hängt bis zum Stream-Ende, und AIfred bekommt einen Aufruf ohne Argumente. Upstream: „the model sometimes omits it“, für V4.1 zuerst behoben | `deepseekv32_tool_parser.py:91`, `:96`, `:503` | anpassen: Attribut optional (`(?:\s+string="([^"]*)")?`), fehlend wie `string="false"` behandeln, also Schema-Koerzierung über `_convert_params_with_schema` (`:157`). Der Rust-Teil entfällt | nein | nach #56141 (Upstream-Reihenfolge) |
| 5 | #56141 | Falsch geschriebener DSML-Öffner | Issue #51914 (DeepSeek-V4-Flash-0731 mit DSpark wie bei uns; ob unser Checkpoint die 0731-Fassung ist, habe ich nicht geprüft) zeigt `<｜DSML｜toolcalls>` und `<｜DSML｜tool>`. Unser Parser erkennt nur die exakte Schreibung | `deepseekv4_tool_parser.py:22`, Erkennung `deepseekv32_tool_parser.py:191`, `:409` | anpassen: nur die zwei beobachteten Schreibweisen als zusätzliche Öffner in `DeepSeekV4ToolParser`, als ein Tupel neben `tool_call_start_token` (SSOT). Erkennung in `extract_tool_calls` und `_process_streaming_buffer` | nein | nach #55954. Gleiche Abstimmung wie bei #55954 |
| 6 | #46091 | Leerer Tool-Block verschluckt den Rest (Qwen3Coder) | Nach `<tool_call></tool_call>` ohne `<function=` bleibt unser Streaming-Parser für immer im ersten Block: `tool_text` endet am ersten `</tool_call>`, ohne Funktionskopf kommt nur `return None`. Alles danach geht verloren, Text und weitere Aufrufe. Nicht-streamend ist es korrekt. Tritt nur auf, wenn das Modell Tool-Markup zitiert | `qwen3coder_tool_parser.py:681-691`, `:701`, `:767`; nicht-streamend `:523-527` | anpassen (eigene Umsetzung): einen geschlossenen Block ohne Funktionskopf überspringen (Index weiterzählen, `is_tool_call_started` zurücksetzen) | nein | keine |
| 7 | #59126 | GLM-Bildturm: Rope-Tabelle mit 8192 Zeilen | Rasterseiten über 8192 Patches lesen hinter die Tabelle. Das gibt einen device-side assert, und die Engine stirbt. Unser Prozessor begrenzt das Seitenverhältnis nicht. Nachgerechnet: 200000×20 px → Rasterseite 14286. Das ist erst ab etwa 2000:1 erreichbar. AIfred-Uploads (≤ 3840 px) kommen höchstens auf 275, offen sind nur Bild-URLs. Der Sync-Teil des PR trifft uns nicht (kein `VLLM_GPU_SYNC_CHECK`, `utils/gpu_sync_debug.py` fehlt) | `glm5next/nvidia/multimodal.py:446`, `:527-532`; `transformers_utils/processors/glm5next.py:172-230` | anpassen, niedrig. Upstream setzt `max_position=text_config.max_position_embeddings`, das sind 1 048 576 × 32 × fp16 = 64 MiB auf der Bildturm-Karte, bei GLM mit 0,972 und 524k kritisch. Besser die Länge aus dem Pixelbudget des Prozessors ableiten: größte Rasterseite = max_pixels / (temporal_patch_size · 28 · 14) = 16 000 bei 8000 Token, also 1 MiB. Entscheidung Peuqui | nein (für bisherige Raster bitgleich). Bei der Upstream-Variante KV-Kapazität prüfen | nach #59565 |
| 8 | #61297 | Ganzzahl-Argument als „3.0“ | `coerce_to_schema_type` versucht `int("3.0")`, bekommt ValueError und landet bei `json.loads`, also 3.0 als float statt 3. Betrifft `qwen3_coder` und DSv4 (`string="false"`). Selten | `tool_parsers/utils.py:566-569`, Ende `:590-593` | anpassen: im integer-Zweig ganzzahlige Floats annehmen (nicht inf/NaN), im selben Zug wie #43984 und #55743 aus Bereich C (gleiche Funktion) | nein | nach #43984 (Bereich C) |

Offen bzw. bewusst nicht empfohlen:
- **Quadratischer Streaming-Aufwand bei sehr langen Tool-Argumenten** (#61253, #64535 GLM; #64675 Qwen3): Auch unsere alten
  Parser arbeiten je Delta über den ganzen bisherigen Aufruf. GLM: `_extract_tool_call_regions` + `func_arg_regex.findall` +
  `_json_escape_string_content` des offenen Werts (`glm4_moe_tool_parser.py:339`, `:394`, `:445`). Qwen3Coder:
  `_parameter_start_positions` kopiert je Parameter `text[:position]` (`qwen3coder_tool_parser.py:166`), also O(Parameter ×
  Länge) je Delta. Spürbar wird das nur bei Endlosschleifen des Modells über viele Parameter. AIfred begrenzt auf 64000 Token,
  es gibt nur einen Nutzer. Die Upstream-Fixes hängen an der Engine (`arg_value_extender`) und sind nicht übertragbar. Vorerst
  nein; falls gewünscht, als eigener Umbau (Positionen inkrementell merken).

## Alle im Code geprüften Kandidaten

| PR | Titel kurz | Urteil |
|---|---|---|
| #59565 | GLM-5.3 Encoder-Cache 7921 statt 8000 | anpassen, Rang 1 |
| #55954 | DSML ohne Öffner | anpassen, Rang 2 |
| #54838 | DSML-Parameter implizit schließen | anpassen, Rang 3 |
| #56271 | DSML ohne `string=` | anpassen, Rang 4 |
| #56141 | DSML-Öffner falsch geschrieben | anpassen, Rang 5 |
| #46091 | Leerer Tool-Block (Qwen3) | anpassen, Rang 6 |
| #59126 | GLM-Bildturm Rope-Tabelle / Sync | anpassen (nur Rope-Teil, niedrig), Rang 7 |
| #61297 | Ganzzahl als „3.0“ | anpassen, Rang 8 |
| #58780 | DSML-String-Argumente inkrementell streamen | nein, schon gelöst: unser String-Modus sendet jedes Stück sofort (`deepseekv32_tool_parser.py:451-490`, Zweige `:458` und `:481`); der Fehler war die Engine-Sperre `arg_structural_chars` |
| #51296 | DSv4-Parser: Thinking-Standard wie Tokenizer | nein: bei uns sind Tokenizer (`tokenizers/deepseek_v4.py:32-35`) und Reasoning-Parser (`deepseek_v3_reasoning_parser.py:31-41`) beide standardmäßig „chat“, und AIfred setzt `enable_thinking` immer selbst |
| #45763 | Qwen3: Tool-Beispiel im Prompt beendet Reasoning | nein, schon gelöst: unsere Rückwärtssuche stoppt am `<think>` der Generierungsaufforderung (`qwen3_reasoning_parser.py:65-71`), die Vorlage endet immer darauf (`chat_template.jinja` des 27B, letzte Zeilen) |
| #54089 | Reasoning-Ende auf aktuelle Runde begrenzen | nein, schon gelöst: dieselbe Rückwärtssuche (`qwen3_reasoning_parser.py:65-71`, `basic_parsers.py:72-81` für `deepseek_r1`) |
| #46047 | Qwen3: Teilwerte mit `<` | nein: Code fehlt, unser Parser hat kein `[^<]*$` (`qwen3coder_tool_parser.py:67-79`) |
| #46351 | Qwen3: String-Argumente streamen | nein: unser Parser sendet jeden Parameter erst vollständig (`qwen3coder_tool_parser.py:801-887`). Das ist Verhalten, kein Fehler; AIfred führt Werkzeuge ohnehin erst nach Abschluss aus |
| #59928 | Qwen3: nicht geschlossenes `<function=` in Prosa | nein: Engine-Konfiguration, nicht übertragbar. Unser alter Parser hat dort eigene Heuristiken (`_tool_start_positions` `:345-362`, Kopf `:701-767`); Fall selten |
| #64675 | Qwen3: geschlossene Parameter einmal parsen | nein (vorerst), siehe „Offen“ |
| #61253 | GLM: offenes Argument nicht je Stück neu parsen | nein (vorerst), siehe „Offen“ |
| #64535 | GLM: geschlossene Argumente einmal parsen | nein (vorerst), siehe „Offen“ |
| #56635 | Engine: Reasoning-Ende und Turn-Grenzen | nein: Engine-Code; GLM nutzt `deepseek_r1` mit Rückwärtssuche bis `<think>` (`basic_parsers.py:72-81`) |
| #56994 | GLM-5.3: Reasoning erzwingen trotz `enable_thinking=false` | nein: `deepseek_r1` ignoriert die Kwargs und parst immer Reasoning, die GLM-Vorlage endet immer auf `<think>` |
| #48748 | EOS leckt in Reasoning | nein: Engine-Fehler; unser Detokenizer lässt das Stopp-Token weg (`v1/engine/detokenizer.py:107-111`) |
| #44955 | `parallel_tool_calls: null` | nein: Fehler besteht auch bei uns (`entrypoints/openai/utils.py:27`), aber AIfred sendet das Feld nicht, Vorgabe `True` (`chat_completion/protocol.py:231`) |
| #52120 | ReDoS im Fehlertext-Sanitizer | nein: die anfällige Pfad-Regex fehlt, unser `sanitize_message` entfernt nur Speicheradressen (`entrypoints/utils.py:317-319`) |
| #54684 | Validierungsfehler-Antwort begrenzen | nein: auch bei uns ungebremst (`entrypoints/openai/server_utils.py:419-433`), aber nur AIfred auf 127.0.0.1 ist Client |
| #52030 | GDN-Packed-Decode: Grid > 65535 | nein: 27B mit max_num_seqs 3 und TP2 liegt weit unter 65535 Programmen (`fla/ops/fused_recurrent.py:574`) |
| #53877 | GDN-Packed-Decode: beta in FP32 | nein, schon gelöst von 1Cat (#403): `beta_val = tl.sigmoid(b_val)` ohne Umweg (`fla/ops/fused_recurrent.py:439`) |
| #56974 | GLM-KDA: Batch × Köpfe aus gridDim.z | nein: gleiches Grid bei uns (`fla/ops/kda.py:117`), aber N ≤ max_num_seqs 1 (`glm5next/nvidia/kda.py:1078-1090`), also 64 Programme in z |
| #60991 | GLM-KDA: q/k/v-Convs als ein Parameter | nein: Fehler nur beim RL-Refit; wir laden einmal, die zwischengespeicherte Verkettung (`glm5next/nvidia/kda.py:814-823`) ist danach korrekt |
| #59147 | GLM: `--mamba-ssm-cache-dtype` für KDA | nein: wir setzen den Schalter nicht, der Zustand bleibt float32 wie gewollt |
| #60763 | GLM: Aux-Hidden-States mit Upstream-Config | nein: Folge von upstream #57387 (Config-Umbenennung), unser DFlash2-Eintrag läuft |
| #58061 | GLM: dichte MLP auf SP-Shard | nein: nur DP + TP + EP (Sequenz-Parallel-MoE); wir fahren PP5 |
| #55270 | GLM-kpool: DeepGEMM-Varlen | nein: DeepGEMM/SM100, laut PR selbst „nothing to fix“ |
| #57563 | Mistral-Large-3: YaRN-Regression | nein: Folge-PR zu upstream #56446 (`attention_factor`), das bei uns fehlt; wir haben noch `apply_yarn_scaling` (`deepseek_v4/nvidia/model.py:711-716`) |
| #54815 | DSv4-RoPE für SWA-Schichten | nein, schon gelöst: factor 1 und volle Positionslänge für `compress_ratio <= 1` (`deepseek_v4/nvidia/model.py:721-740`); Checkpoint hat flaches `rope_scaling` |
| #52288 | DSpark erbt Ziel-Backend | nein: `dspark/utils.py` fehlt, eigener Lader (`spec_decode/dspark/speculator.py:82-111`); Ziel-Backend nicht gepinnt |
| #52809 | DSpark-Erbe auf DSv4 begrenzen | nein: Folge von #52288 |
| #55472 | DSpark: DCP übernehmen | nein: kein DCP |
| #56138 | EPLB/MLA-H2D-Puffer pinnen | nein: nur unter strikter Sync-Prüfung relevant, Datei fehlt |
| #58560 | DSv4.1-Kompressor-Ring im Null-Block | nein: der Ring-Kern existiert nur in V4.1, unser `deepseek_v4/compressor.py` hat ihn nicht |
| #60613 | DSv4.1: −1-Sentinel im Prefill-Workspace | nein: wie #55299 (Bereich B, schon gelöst), unsere Kerne maskieren über die Länge |
| #59986 | DSv4.1: Sparse-Prefill aus FULL-Graphen | nein, schon gelöst: unsere Sparse-MLA- und SWA-Builder melden `UNIFORM_BATCH` (`v1/attention/backends/mla/flashmla_sparse.py:246`, `sparse_swa.py:281`), das Minimum entscheidet |
| #51318 | DSv4: adaptive C128A-Packung zurücknehmen | nein: die zurückgenommene Änderung #50004 ist bei uns nie angekommen (kein `active_topk_width`) |
| #43362 | FusedMoE: Skalen der Form (1,) | nein: `routed_experts.py` fehlt; unsere NVFP4-/MXFP4-Checkpoints laden |
| #50137 | RoutedExperts: Skalen nicht transponieren | nein: Regression aus upstream #47058, das fehlt |
| #50937 | RoutedExperts: leerer Experten-Bias | nein: ebenso #47058 |
| #50386 | Latent-MoE-Residualpuffer | nein: Latent-MoE-Runner fehlt |
| #53409 | SiLU-Block-Quant int32-Überlauf | nein: bräuchte über 100 000 Token je Batch, wir haben höchstens 2048 |
| #61755 | `silu_and_mul_nvfp4_quant` K-Padding | nein: der Kern wird nur für SM100/SM120 gebaut (`CMakeLists.txt:1190-1237`) |
| #63382 | NVFP4 Act+Quant: uninitialisierte Skalen | nein: ebenso nur SM100/SM120 |
| #61944 | Hadacore: letzter Warp | nein: Hadacore nur für compressed-tensors-Transformationen, nicht unsere Modelle |
| #63614 | Hadacore: `wait_group` | nein: ebenso |

## Aussortiert nach Titel und Beschreibung (159)

- #42644 scale-out/disagg
- #42748 disagg
- #45657 Harmony/gpt-oss
- #45718 MiniMax M3
- #45795 Gemma4
- #45832 Gemma4
- #46038 Fehlertext-Kosmetik (`param`-Feld)
- #46382 MiniMax M2
- #46506 SM120/FlashInfer
- #47062 Harmony
- #47260 scale-out derender
- #47268 Marlin AWQ int4-FP8
- #47834 derender
- #47933 scale-out
- #48262 Gemma4
- #48608 Video (bei uns aus)
- #48868 Helion
- #49195 scale-out
- #49306 FA4-Warmup (SM100)
- #49328 KV-Offload
- #49344 Logo-Farben
- #49602 Cohere
- #49876 Inkling
- #49987 Transformers-Backend
- #50074 Online-NVFP4 beim Neuladen
- #50305 MiniMax M3
- #50420 Kimi K3
- #50528 Inkling
- #50567 Kimi K3
- #50595 Structured Outputs/Harmony
- #50886 Kimi K3
- #51366 SageMaker
- #51391 Inkling
- #51395 SM120
- #51442 Online-MXFP4
- #51455 GPU-Sync-Debug (fehlt)
- #51556 Cohere
- #52222 GPT-OSS
- #52399 scale-out
- #52430 Gemma4
- #52552 LoRA
- #52632 DeepEP
- #53326 B12X
- #53353 LoRA
- #53369 MM-LoRA
- #53444 Gemma4
- #53574 SM120
- #53616 Mypy Gemma4-DSpark
- #53625 hf_token im Log
- #53657 Gemma4
- #53750 Cohere
- #54005 Kimi-K3-DSpark
- #54177 Mypy
- #54264 Seed-OSS
- #54324 scale-out
- #54844 Mistral
- #54913 `vllm launch render`
- #55102 gRPC
- #55133 Qwen3-DSpark (nicht genutzt)
- #55458 DP-Padding
- #55538 NVFP4-DS-MLA-KV (wir fp8)
- #55647 GLM-Video (bei uns aus)
- #55699 SSL
- #55761 Responses
- #56070 scale-out EC
- #56102 SM110
- #56387 DSv4.1 EPLB
- #56472 Weight-Cache-Daemon
- #56676 ohne Triton
- #56904 GPU-Sync-Debug
- #56969 FlashInfer SM90
- #57058 tokens-only
- #57098 Kimi K3
- #57454 DSv4.1 Kandidatenblöcke
- #57667 DSA-Kerne (Hopper/Blackwell, fehlen)
- #57729 Generative Scoring
- #57874 DSv4.1 mHC
- #57898 DSv4.1 Vision
- #57914 Engram
- #57948 run-batch
- #57990 derender
- #58179 Batch-Invariant
- #58215 DSA DeepSelect (fehlt)
- #58372 Kimi K3
- #58488 tokens-in/out
- #58499 DSv4.1 ViT
- #58704 FlashInfer SM90 Sparse-MLA
- #58788 Generative Scoring
- #58792 Inkling
- #58814 Kimi K3
- #59007 HiSparse
- #59068 Engram
- #59072 Cohere
- #59106 Batch-Invariant
- #59236 RL-Dev-Routen
- #59254 Harmony
- #59257 Kimi K3
- #59282 HiSparse
- #59373 DSv4-VL (wir nur Text)
- #59627 Structured Outputs
- #59639 Engram
- #59645 Cooperative-TopK (fehlt)
- #59661 CRIU
- #59779 Watermarking
- #59900 FlashInfer CuteDSL MegaMoE
- #59962 RecoverSSM (fehlt)
- #60179 Weight-Cache
- #60234 Qwen4Exp
- #60300 Ngram
- #60309 Cohere
- #60310 Request-Logging
- #60337 Humming
- #60507 run-batch
- #60586 Cohere
- #60682 HiSparse
- #61041 DSv4.1 MegaMoE
- #61061 Offline-LLM
- #61065 derender
- #61089 Cohere
- #61118 Batch-Invariant
- #61212 render/Beam
- #61236 render
- #61241 render
- #61390 KV-Offload
- #61463 CuTe-DSL-Router-GEMM (SM90+)
- #61494 Gemma4
- #61542 HYV4
- #61574 Structured Outputs
- #61643 DSv4.1
- #61659 HiSparse/TRT-LLM
- #61757 derender
- #61759 Video WebM
- #61825 Cohere
- #61852 derender
- #61973 run-batch
- #62114 derender
- #62163 derender
- #62269 Engram
- #62301 DP-Draft
- #62517 DSv4.1-Parser (Tags mit Leerzeichen)
- #62583 Weight-Cache
- #62767 Dummy-Laden
- #62967 gRPC
- #63012 B12X
- #63068 Engram
- #63190 MM-LoRA
- #63476 DBO
- #63492 MiniMax M3
- #63914 DSv4.1
- #63999 Triton-Dispatcher (fehlt)
- #64751 Standalone-Draft
- #64781 derender
- #65118 Mistral
- #65125 Muse Spark
- #65185 TopK-Compilerwarnung (Datei fehlt)
- #65229 Akzeptanzschätzer (fehlt)
- #65389 scale-out
- #65457 Extensible Tensor (fehlt)
- #65778 FlashInfer-ViT (bei uns kein FlashInfer)

Nicht geprüft (Auftrag): #49139 (persistent top-k, schon im Einbau).
