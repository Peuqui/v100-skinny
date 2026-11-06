# Bereich C: OpenAI-Frontend, Parser, Renderer, Tokenizer, SamplingParams (Stand 06.11.2026)

Geprüfter Baum: `/home/mp/Projekte/vllm-research/1Cat-vLLM-main-v6` (HEAD 5f1c557fb). Die in den Produktionsbäumen
laufenden Frontend-Dateien sind gleich: `chat_completion/serving.py`, `parser/abstract_parser.py`, `tool_parsers/utils.py`,
`tokenizers/deepseek_v4*.py`, `chat_utils.py` und die drei Tool-Parser sind in v6c (GLM) und v6e (DeepSeek) byte-gleich mit
main-v6. In v5d (Qwen3.8-27B) weichen nur `serving.py` (dort fehlt `_PrefixCacheBoundaries` aus 1Cat #1297) und
`deepseekv32_tool_parser.py` ab. Die Stellen, an denen die Empfehlungen ansetzen, sind dort ebenfalls vorhanden.
Alle Zeilennummern beziehen sich auf main-v6.

Betriebsannahmen, aus dem Code und der llama-swap-Config belegt:
- Alle drei Einträge laufen mit `--enable-auto-tool-choice --enable-prompt-tokens-details`. `VLLM_ENFORCE_STRICT_TOOL_CALLING`
  ist nicht gesetzt, Vorgabe 0 (`envs.py:15848`). Deshalb gibt es KEINE Structural Tags (`abstract_tool_parser.py:98-101`).
  JSON-Schema-Zwang greift nur bei `tool_choice` required oder named (`abstract_tool_parser.py:121-128`).
- AIfred (`aifred/backends/base.py:826-835`) streamt immer mit `stream_options.include_usage`, `max_tokens=64000`
  (`config.py:417`) und `tools` ohne `tool_choice` (also auto). Es setzt keine logprobs, kein `n`, kein `response_format`
  und keine Continuous-Usage-Stats. Nicht-Streaming nutzt es nur ohne Tools (z. B. Verlaufs-Kompression,
  `context_manager.py:652-662`).
- AIfred verwirft Tool-Aufrufe, wenn `finish_reason == "length"` gemeldet wird (`base.py:923-933`), und hat einen Workaround
  für das fehlende `cached_tokens: 0` (`aifred/backends/vllm.py:216-227`).
- DeepSeek-V4 rendert über den Python-Encoder (`tokenizer_mode` auto → `deepseek_v4`, `config/model.py:637-638`; das
  Modellverzeichnis hat keine Jinja-Vorlage).
- Die GLM-5.3-Vorlage hat keinen `enable_thinking`-Schalter. Der Generierungs-Prompt endet immer auf `<think>`
  (`chat_template.jinja:260`).

## Rangliste der Empfehlungen (übernehmen / anpassen)

| # | PR | Kurztitel | Warum es uns trifft | Beleg im Code | Empfehlung | A/B nötig | Abhängigkeiten |
|---|---|---|---|---|---|---|---|
| 1 | #51856 | DSv4-Renderer: Request-Tools an die bestehende Systemnachricht | Jede DeepSeek-Anfrage von AIfred mit Systemprompt und Tools bekommt heute vorneweg eine leere synthetische Systemnachricht mit dem Tool-Block. Der Prompt lautet dann `<BOS>` + `\n\n## Tools …` + AIfred-Systemprompt, ohne Trenner dazwischen. DeepSeeks Referenz (`encoding/README.md:77` im Modellordner: Tools stehen im `tools`-Feld der Systemnachricht) ergibt dagegen `<BOS>` + Systemprompt + `\n\n## Tools …`. Falsche Prompt-Struktur im Normalfall. | `vllm/tokenizers/deepseek_v4.py:39-41` (`messages.insert(0, {"role": "system"})`); `copy` ist in Zeile 3 schon importiert | übernehmen (`git apply --check` sauber) | **ja**, Qualität: Text- und Tool-Aufruf-Vergleich auf DSv4 und K7. Tempo nein, die Tokenzahl bleibt praktisch gleich. | keine; dieselbe Datei wie #51262 ist es nicht (dort `deepseek_v4_encoding.py`) |
| 2 | #46303 | Streaming: `length` statt `tool_calls`, wenn ein Tool-Aufruf an `max_tokens` abreißt | Unser Streaming-Pfad setzt `tool_calls`, sobald der Parser irgendeinen Aufruf gesehen hat (`auto_tools_called`), auch wenn die Engine mit `length` endet. AIfred verwirft abgeschnittene Aufrufe nur bei `length` (`base.py:923-933`), führt sie heute also mit halben Argumenten aus. | `chat_completion/serving.py:1037-1046` (Bedingung `auto_tools_called or (tools_streamed[i] and not tool_choice_function_name) or harmony`) | anpassen: die ganze Bedingung zusätzlich an `output.finish_reason == "stop"` binden. Der Upstream-Hunk passt nicht wörtlich, weil Upstream dort nur `tools_streamed` prüft. | nein; Funktionsprobe mit kleinem `max_tokens` und Tool-Aufruf reicht | gleiche Funktion wie #40986, aber andere Stelle |
| 3 | #51262 (Bereich B, hier mitgeprüft) | DSv4-Encoder: System als letzte Nachricht oder vor einer Assistant-Nachricht | Nach AIfreds Verlaufs-Kompression lautet der Verlauf `[Zusammenfassung als system] + llm_remaining`. Die Kompression nimmt Nachrichten einzeln von vorne weg (`context_manager.py:572-573, 807`), sodass der Rest mit einer Assistant-Nachricht beginnen kann. Unser Encoder hängt nach `system` kein `<｜Assistant｜>` an, die folgende Assistant-Runde steht dann ohne Rollen-Token im Prompt. | `vllm/tokenizers/deepseek_v4_encoding.py:390` (Übergang nur bei `user`/`developer`) | übernehmen (Python-Teil `git apply --check` sauber; der Rust-Teil entfällt) | nein für Tempo; Textprobe mit konstruiertem Verlauf system → assistant → user | optional vorher #53071 (dieselbe Datei, anderer Hunk). Wechselwirkung mit 1Cats `_PrefixCacheBoundaries` geprüft: Dessen Systemprompt-Probe (`serving.py:158-170`) würde für DSv4 nicht mehr passen. Der Teil-Präfix-Cache ist für DSv4 aber ohnehin aus, weil er nur für Modelle mit Mamba-Gruppen gilt (`v1/core/kv_cache_manager.py:238-250`). |
| 4 | #40986 | Streaming-Fehler vor dem ersten Token gehen verloren | Endet eine Anfrage mit `finish_reason="error"` und 0 Token, verwirft der Leer-Chunk-Filter die Ausgabe, bevor `_raise_if_error` läuft. Der Client bekommt Rollen-Chunk, Usage und `[DONE]`, aber keinen Fehler und kein `finish_reason`. AIfred sieht eine stille leere Antwort. Erreichbar über Vorverarbeitungsfehler je Anfrage in der EngineCore. | Filter `chat_completion/serving.py:800-806` vor `:964`; Quelle `v1/engine/core.py:1803-1810` (`_handle_request_preproc_error` → `FinishReason.ERROR` ohne Token) | anpassen: `self._raise_if_error(output.finish_reason, request_id)` direkt nach der `finish_reason_sent`-Prüfung (Zeile 758) aufrufen und den Aufruf in Zeile 964 entfernen. Der Hunk passt wegen anderer Logprobs-Zeilen nicht wörtlich. | nein | nach #46303 oder unabhängig |
| 5 | #43984 | `coerce_to_schema_type`: `inf`, `1e999`, NaN | Bei einem Parameter vom Typ `number` wirft `int(float("inf"))` einen `OverflowError`. Gefangen werden nur `ValueError`/`TypeError`. In den Streaming-Pfaden von `qwen3_coder` (Qwen3.8) und `deepseek_v4` (DSML mit `string="false"`) steht kein try/except darum, der Stream bricht mit einer Fehlermeldung ab. Dazu kommt ungültiges JSON (`Infinity`). Selten, aber ein Absturz der Anfrage. | `tool_parsers/utils.py:571-576`, `:590-593`; Aufrufer `qwen3coder_tool_parser.py:854-861` und `deepseekv32_tool_parser.py:367` (beide ohne try) | anpassen: Der Funktionskörper ist mit dem Upstream-Stand vor dem Patch identisch, nur der Import-Kontext (Zeile 3) weicht ab. Hunks einzeln übernehmen, `import math` ergänzen. | nein | vor #55743 |
| 6 | #55743 | Große Ganzzahlen bei Typ `number` bleiben exakt | Der `number`-Zweig rechnet über `float`, Werte über 2^53 werden verfälscht. Betrifft dieselben zwei Parser. Sehr selten. | `tool_parsers/utils.py:571-576` | anpassen (nach #43984; dann passt der Hunk) | nein | #43984 zuerst |
| 7 | #44383 | `prompt_tokens_details` auch bei 0 Cache-Treffern | Mit `--enable-prompt-tokens-details` fehlt das Feld beim ersten Turn, wegen einer Wahrheitsprüfung statt `is not None`. AIfred umgeht das heute mit dem Workaround in `vllm.py:216-227`, der nach dem Port überflüssig, aber weiter verträglich ist. | `chat_completion/serving.py:1102` (Stream) und `:1534` (Batch) | übernehmen (`git apply --check` sauber; Completion und Disagg kommen mit) | nein | keine; #64177 und #65036 werden nicht gebraucht |
| 8 | #42752 | `tool_choice="none"` im Streaming | Anfragen ohne Tools haben `tool_choice="none"` (Vorgabe in `chat_completion/protocol.py:209-215`). `parse_delta` ruft trotzdem den Tool-Parser auf. Schreibt das Modell Tool-Markup, streamt der Server `tool_calls` mit `finish_reason="tool_calls"`, Batch liefert dagegen Text. Sichtbarer Effekt nach dem Fix: Ein solches Markup erscheint in diesem seltenen Fall als Text. | `parser/abstract_parser.py:575-630` (kein Wächter), Batch-Gegenstück `serving.py:1400-1401` | übernehmen (`git apply --check` sauber) | nein | Upstream-Folge-PR #63345 (für `null`) nicht nötig, siehe Tabelle |

Offen bzw. `unklar` (nicht in der Rangliste):
- #57035 (`max_tokens` als Obergrenze statt Reservierung): Heute lehnt der Server Prompts über `max_model_len − max_tokens`
  mit 400 ab (`renderers/params.py:188-193`). AIfred schickt immer 64000. Bei GLM liegt die Grenze damit bei 460 288 Token,
  bei DSv4 bei 436 000, beim 27B bei 198 144. Der PR ist eine Verhaltensänderung und kein Fehler im engeren Sinn
  (Konflikt, mit Pooling-Anteil, braucht #62010-Logik). Zu klären: Stößt AIfreds Kontextführung je an diese Grenze?
  Wenn ja, ist das eher eine AIfred-Einstellung als ein Backport.
- #56446 (YaRN an Transformers angleichen): eigentlich Bereich B (RoPE-Numerik). Für DSv4 bleibt die abgeleitete Länge
  rechnerisch gleich (65536 × 16 = 1 048 576 = `max_position_embeddings`). Die neue `mscale`/`attention_factor`-Behandlung
  könnte die Numerik ändern. Upstream schreibt selbst „Model evaluation not yet run“. An Bereich B, dort nur mit A/B.

## Im Code geprüfte Kandidaten ohne Empfehlung

| PR | Titel kurz | Status maschinell | Urteil |
|---|---|---|---|
| #35076 | Server-Standard-`stop_token_ids` | fehlt_passt | nein: `get_diff_sampling_param` übernimmt nie `stop_token_ids` (`config/model.py:1500-1507`); nur gpt-oss/Harmony setzt sie (`serving.py:296-302`) |
| #42376 | speculators-Format-Overrides | fehlt_passt | nein: greift nur, wenn das Zielmodell im speculators-Format vorliegt; unsere Ziele sind normale Checkpoints |
| #43862 | `arguments: "null"` | konflikt | schon gelöst: `chat_utils.py:1832-1866` (1Cat-Fassung setzt None und Nicht-Objekte auf `{}`) |
| #44105 | leere `tool_calls` weglassen | fehlt_passt | nein: AIfred prüft `if delta.tool_calls` (`base.py:769-771`), eine leere Liste ist harmlos; nur Schema-Kosmetik |
| #44303 | `http_requests_total` 4xx als 5xx | konflikt | nein: nur ein Prometheus-Label; AIfred liest vLLMs eigene Histogramme |
| #45147 | Nicht-Funktions-Tools (web_search) | fehlt_passt | nein: AIfred sendet nur Funktions-Tools; `ToolParser.__init__` filtert ohnehin (`abstract_tool_parser.py:72-77`) |
| #45290 | named tool_choice mit leeren Parametern | konflikt | nein: nur named tool_choice (Structured Outputs) |
| #45389 | Klammern in required-Streaming-Strings | fehlt_passt | nein: nur `tool_choice="required"` |
| #45460 | Tokenizer-Pool lässt sich nicht pickeln | fehlt_passt | nein: betrifft Ray und Pickle-Pfade; laut PR ist `renderers/hf.py` nicht betroffen |
| #45629 | FastAPI ≥ 0.137, Router ohne `path` | fehlt_passt | nein: venv hat prometheus-fastapi-instrumentator 8.1.0 mit `_effective_routes` (`routing.py:41-56`), der Fehler ist dort behoben; Produktion läuft mit FastAPI 0.141.1 |
| #45919 | /derender, FFFD bei Byte-Fallback | teilweise | nein: /derender wird nicht genutzt; der api_server-Hunk betrifft den Render-State, der bei uns konsistent zu Chat ist (beide `structured_outputs_config.reasoning_parser`, `api_server.py:391,481`) |
| #46159 | U+FFFD am Übergang Reasoning → Inhalt (Engine-Parser) | konflikt | nein: der ParserEngine-Pfad fehlt im Fork (`vllm/parser/engine/` existiert nicht); unser `parse_delta` arbeitet textbasiert (`abstract_parser.py:689-698`) ohne Einzeltoken-Decode |
| #46308 | Nicht-ASCII-Argumente ohne `\uXXXX` | fehlt_passt | nein: betrifft hunyuan, seed_oss, xlam; unsere drei Parser nutzen überall `ensure_ascii=False` (`glm4_moe_tool_parser.py:128,214,344-390`, `deepseekv32_tool_parser.py:182,212,317,333`, `qwen3coder_tool_parser.py:276,313,861`) |
| #46875 | Spezialtokens lecken (Engine-Parser) | konflikt | nein: ParserEngine fehlt |
| #47512 | GLM required → kein JSON-Zwang | konflikt | nein für den Betrieb: Der Fehler existiert bei uns (`abstract_tool_parser.py:121-128` ignoriert `supports_required_and_named=False` von glm47), greift aber nur bei required/named; AIfred nutzt auto |
| #47562 | unvollständiges Tool-Markup im Batch | konflikt | nein: ParserEngine-Pfad; AIfred-Batch läuft ohne Tools |
| #47606 | Flush am Übergang Engine-Reasoning → Tool | konflikt | nein: nur Engine-Reasoning-Parser; unsere sind textbasiert |
| #48333 | Medienabrufe lecken bei Teilfehler | konflikt | nein: nur im Fehlerfall (eine Bild-URL schlägt fehl), die Abrufe enden von selbst; geringe Auswirkung |
| #48711 | GLM-5.2-Config `layer_types` | konflikt | nein: betrifft `model_type` `glm_moe_dsa`; unser GLM ist `glm5_next` und bootet |
| #48922 | Tool-Argumente im Verlauf absichern | konflikt | schon gelöst: `chat_utils.py:1832-1866` (gleicher Text wie Upstream) |
| #49668 | Signal-Handler-Race beim Start | fehlt_passt | nein: llama-swap stoppt über `vllm-swap-stop`, das den ganzen Prozessbaum selbst beendet |
| #50642 | `model_config` an verschachtelte Reasoning-Parser | konflikt | nein: von unseren Parsern liest keiner `_model_config` (nur Cohere, `cohere_command_reasoning_parser.py:510`) |
| #53071 | DSv4-Encoder: unbekannte Rolle 400 statt 501 | fehlt_passt | nein (nur Fehlercode); billig, kann mit #51262 mitgehen |
| #53747 | DSv4-Encoder: nackte `assert` | konflikt | nein: erreichbar ist nur die leere developer-Nachricht (500 statt 400); `reasoning_effort` wird im Wrapper normalisiert (`deepseek_v4.py:43-52`); kein `python -O`. Upstream hat dort eine neuere Encoder-Fassung (`REASONING_EFFORT_PROMPTS`) |
| #53824 | Inhaltsformat bei Makro-Parametern | fehlt_passt | nein: Die Makros in Qwen3.8 (`render_content(content, …)`) und GLM (`visible_text(content)`) heißen `content` und werden schon erkannt |
| #54023 | Revert: MM-Warmup im Hintergrund | teilweise | nicht nötig: #52764 ist nie im Fork gewesen (kein `start_mm_warmup_in_background`); Warmup läuft synchron (`renderers/base.py:226-268`) |
| #55448 | Renderer-Warmup an Prefill-Budget binden | fehlt_passt | nein: `seq_len` wirkt bei GLM nur auf Video-Frames (`glm4_1v.py:1300-1302`); Video ist bei allen Einträgen 0 und wird beim Warmup ausgefiltert (`renderers/base.py:213`); Bildgröße hängt nicht von `seq_len` ab |
| #56270 | Batch verliert Text nach dem Tool-Aufruf (deepseek_v3) | fehlt_passt | nein: anderer Parser. Das Analogon in `deepseekv32_tool_parser.py:223-224` gibt es; laut Referenz-Encoder folgt bei DSv4 auf den Block aber nur EOS (`parse_message_from_completion_text`: „Unexpected content after tool calls“), und AIfred streamt |
| #56446 | YaRN-Angleich | teilweise | unklar → Bereich B (siehe oben) |
| #57035 | `max_tokens` als Obergrenze | konflikt | unklar (siehe oben) |
| #58583 | Logprobs unterdrückter Chunks | konflikt | nein: AIfred fordert keine Logprobs an |
| #58911 | gepufferter Text ohne Tool-Parser (Engine) | konflikt | nein: nur Engine-Reasoning-Parser; bei uns ist immer ein Tool-Parser gesetzt |
| #58939 | frischer Parser je Choice (Batch) | konflikt | nein: nur `n>1`; im Batch erzeugt `_parse_tool_calls_from_content` ohnehin je Aufruf einen frischen Tool-Parser (`engine/serving.py:545`) |
| #59491 | `max_token_id` um eins zu hoch | fehlt_passt | nein: wirkt nur, wenn der höchste Vokabel-Index unter `vocab_size` liegt, und nur bei Token-ID-Eingaben bzw. `logit_bias`; AIfred sendet Text; Server hört nur auf 127.0.0.1 |
| #60354 | mehrere Tool-Aufrufe in einem Delta | konflikt | nein: nur der JSON-Pfad für `tool_choice="required"`. Die Zähländerung in `parse_delta` (`abstract_parser.py:731-736`) wirkt nur auf Kimi-IDs; unsere Auto-Parser (GLM: neu parsen und diffen) sind nicht betroffen |
| #60362 | mehrzeiliges JSON (deepseek_v3/v31) | fehlt_passt | nein: DSv4 nutzt DSML, kein JSON |
| #61857 | Reasoning schon im Prompt geschlossen | konflikt | nein: der Fix greift nur in der ParserEngine; Legacy-Parser (`deepseek_r1`) bleiben laut PR unverändert. Bei uns: GLM-Vorlage endet immer auf `<think>`, Qwen3 behandelt `enable_thinking=False` selbst (`qwen3_reasoning_parser.py:43-46`), DSv4 nutzt im Chat-Modus den Identity-Parser (`deepseek_v3_reasoning_parser.py:29-41`) |
| #62010 | Auto-Kürzung ohne Ausgabelimit | fehlt_passt | nein: nur mit `truncate_prompt_tokens`; AIfred setzt es nicht und schickt immer `max_tokens` |
| #63345 | tool_choice-Regeln Batch = Stream (auch `null`) | konflikt | nein: großer Umbau (neue Dateien `message_assembly.py`, `stream_processor.py`). Für AIfreds letzte Runde (`tools`/`tool_choice` = None) ändert sich nichts, AIfred ignoriert dort Tool-Deltas (`base.py:1143`, `:1156`: `_consume_stream(..., None)`) |
| #63606 | Prozessoren mit mehreren Vorlagen | fehlt_passt | nein: unsere Prozessoren liefern genau eine Vorlage |
| #64177 | `annotations: null` und `prompt_tokens_details: null` weglassen | fehlt_passt | nein: nur strenge Schema-Prüfer; das OpenAI-SDK in AIfred akzeptiert null. Nach #44383 ist das Feld mit unserem Schalter ohnehin immer gefüllt |
| #64615 | GLM-4.7-Structural-Tag: Arrays und Objekte frei lassen | konflikt | nein: Das GLM-4.7-Tag (Upstream #56403) gibt es im Fork nicht, die Registry kennt nur `deepseek_v4` und `qwen_3_5` (`structural_tag_registry.py:138,247`); Structural Tags sind ohne `VLLM_ENFORCE_STRICT_TOOL_CALLING=1` ohnehin aus |
| #65036 | Cache-Aufteilung im ersten Usage-Chunk | konflikt | nein: betrifft nur Continuous-Usage-Stats bzw. Anthropic `message_start`; AIfred nutzt nur `include_usage` |

## Aussortiert nach Titel und Beschreibung (73)

- #44617 Docstring der Offline-`LLM`-Klasse
- #44965 Temperatur über 2 ablehnen (Verschärfung; AIfred bleibt im Bereich)
- #45346 Structured Outputs
- #45376 Anthropic-API
- #46175 /v1/completions
- #46255 ernie45-Reasoning-Parser
- #46486 poolside-Tool-Parser
- #46612 Fehlercode bei `logit_bias`-Schlüsseln
- #47311 poolside-Tool-Parser
- #47450 Structured Outputs
- #47815 /v1/completions mit echo
- #47838 Logprobs (AIfred nutzt keine)
- #48252 /v1/completions `vllm_xargs`
- #48473 Fehlercode 400 statt 500 bei Bildern an ein Textmodell
- #48674 SentencePiece-Logprobs (Legacy Completions)
- #49001 HF-Cache-Race bei `--api-server-count > 1` (wir: lokale Pfade, 1 API-Server)
- #49144 Fehlercode bei nicht-numerischen Logprobs
- #49400 Kimi-K2.5-Vision-Chunks
- #49727 A.X-K1-Modell
- #50254 VLLMValidationError-Migration (nur `param`-Feld im Fehler)
- #50491 dito
- #51654 Fehlercode bei Nicht-Objekt-JSON-Body
- #51796 Structured Outputs (Regex)
- #52298 Structured Outputs (guidance)
- #52431 ModelScope
- #52528 Fehlercode bei Nicht-Objekt-JSON-Body (andere Endpunkte)
- #52529 Batch-Chat-Completions
- #52622 /detokenize
- #52825 `vllm launch`
- #53124 Reasoning-Parser-Plugins mit headless Engine
- #53433 `bad_words` (nicht genutzt)
- #53763 Responses-API
- #53999 Fehlermeldung bei zu vielen interleaved MM-Platzhaltern
- #54022 Template-`raise_exception` als 400 statt 500 (nur Fehlercode)
- #54113 ZMQ-Port-TOCTOU (Supervisor- und Rust-Umbau, Vorgänger #42585/#51033)
- #54192 Mistral-Tokenizer
- #54407 `truncate_prompt_tokens`-Offsets (nicht genutzt)
- #54509 dito
- #54539 dito
- #54622 nur eine Warnung zum Inhaltsformat
- #54692 Render-Offsets bei Linkskürzung
- #54850 `$defs`-Fehlercode (nur JSON-Schema-Pfad für required/named)
- #55288 Offline `LLM.chat()`
- #55305 Responses-API
- #55325 Anthropic-API
- #55451 `prompt_embeds`
- #55606 Fehlercode bei `extra_args`-Ganzzahlen
- #55710 CLI-Client `vllm chat`/`complete`
- #55717 /v1/completions-Logprobs
- #56086 `response_format` (Structured Outputs)
- #56988 Responses-MCP
- #57076 `truncate_prompt_tokens` mit MM-Expansion
- #57189 Laguna-Modell
- #57222 Responses-API
- #57243 Offline `LLM.chat()`
- #57461 Revisions-Pins für Hub-Artefakte (wir: lokale Pfade)
- #58336 Text einer Fehlermeldung
- #58927 Responses-API
- #58929 Batch-Chat
- #59173 Responses-API
- #59555 KV-Transfer, wiederverwendete Prompt-IDs
- #59652 Responses-API
- #59749 Parser-Tokenizer-Prüfung beim Start (unsere Parser laufen)
- #59879 Tool-Grammatiken (Structured Outputs)
- #60668 /v1/completions
- #61272 Tokenizer-Snapshot und `code_revision` (Folge von #60825; wir: lokale Pfade)
- #61611 dito
- #61876 Completions-Echo
- #62205 /v1/completions-Logging
- #62911 `skip_tokenizer_init`
- #63482 Tokenizer-Snapshot und Hub-ID (Folge von #60825)
- #63922 Tokenizer-Snapshot lädt Gewichte (Folge von #60825)
- #66012 `HF_HUB_OFFLINE` (nicht gesetzt)

Reverts: Unter den 7786 gemergten PRs in `merged_prs.jsonl` ist keiner, der einen Kandidaten dieses Bereichs zurücknimmt.
