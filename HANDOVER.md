# Übergabe — Stand 03.10.2026 mittags (Umzug in das Projekt vllm-research)

Ab jetzt arbeitet die vLLM-Forschung als eigenes Projekt: Claude Code startet in
`/home/mp/Projekte/vllm-research` (Projektregeln in dessen `CLAUDE.md`, eigenes Gedächtnis,
AI-Connect-Peer `Mini:vllm-research`). Allgemeine Arbeitsregeln stehen in `~/.claude/CLAUDE.md`.

## Produktion

- **fork-union = fork-next = 138059c8**, Tag `verified-2026-10-03-pledisk` (vorher 368e74fd, Tag
  `archive-fork-union-2026-10-03`). Enthält die PLE-Plattenstufe (Seiten gebündelt anfordern,
  Seitentabellen-Freigabe, Zeilen-Cache) und das reduzierte Entwurfsvokabular.
- llama-swap: alle 8 Flash-Next-Einträge mit `draft_token_map` und
  `VLLM_PLE_DISK_ROW_CACHE_GIB=0.5`; Backup `~/.cache/prod-switch-2026-09-29/config.yaml.vor-prod-pledisk-2026-10-03`.
- Abnahme: `ab_final*.sh`, `ab_verify.sh` (Zeilen-Cache bytegenau, 3,28 Mio. Zeilen), Rauchtest
  aller vier Modelle; Zahlen in STAND.md (Nachtrag 03.10.) und `docs/journal/LEISTUNGSHISTORIE.md`.
- In der main-Gegenprobe bestätigt: PP4 Prefill 29k 13,5–13,6 s, Schritt kurz 52–54 ms; TP2×PP2
  18,5–18,6 s / 48–49 ms; DSv4 35,9k in 15,3 s, Schritt 85–86 ms; 27B 41,4 s bei 29k.

## Womit anfangen

1. **Startfehler auf 1Cat main klären** (`~/.cache/bench-scripts/ab_main.sh`, Log
   `ab_main_2026-10-03.log`, Worktree `1Cat-vLLM-main-val` auf a692497bc mit unseren .so; main
   hat seit d3046986 nur GGUF-Operatoren nativ geändert). Die main-Einträge = Produktionseinträge
   ohne unsere Fork-Schalter, ohne `--moe-backend sm70_skinny`, ohne Entwurfsliste.
   - **Flash-Next PP4:** OOM in `ple_layer.py` `materialize_tables` — main will die ganze
     PLE-Tabelle (47,69 GiB) auf GPU 0 legen, die Kaskade greift nicht. TP2×PP2: nur 0,03 GiB KV
     übrig, vermutlich dieselbe Ursache. Nächster Schritt: `kernel_config.ple_disk_cascade_reason`
     auslesen (VllmConfig für den Eintrag bauen, ohne Gewichte; Zulassung in `vllm/config/vllm.py`
     `_qwen4exp_ple_cascade_requested`). Danach entscheiden: Konfiguration oder Lücke in main.
   - **DSv4 PP5:** main wählt ohne Vorgabe Marlin; auf den V100-Stufen: „DeepSeek-V4 MXFP4 MoE on
     SM70 requires the native TurboMind backend“. Nächster Versuch mit ausdrücklichem
     TurboMind-Backend (Namen in main prüfen); das Skinny-Backend (#742) integriert 1Cat gerade.
   - **27B TP2 (RTX 8000): main = Produktion** (Greedy 3/3 identisch, 41,5 gegen 41,4 s, gleiche
     tok/s) — Beleg für 1Cats #804 auf echter Turing-Hardware.
   - Hinweis: In main ist `kernel_config.ple_disk_release_pages` standardmäßig aus; eingeschaltet
     gibt es ganze Shards frei (unser alter Weg v0).
2. **Branch `fork-main` aufbauen** (Peuqui: komplettes main übernehmen, Zusätze obendrauf, als PR
   anbieten): offene PRs #715 und #611 (bis 1Cats #822 gemergt ist), FLA-Shared-Memory-Fix
   (841656ec), Entwurfsvokabular (2ccdc953, Backport vllm#59740), PLE-Plattenstufe auf 1Cats
   Kaskade aus #806 portieren (Zeilen-Cache als KernelConfig-Feld, Seitentabellen-Freigabe unter
   `ple_disk_release_pages`). #742 übernimmt 1Cat selbst. Dann A/B gegen die Produktion,
   Umstellung, PRs.
3. **#674-Leistungsbeitrag:** Historie (LEISTUNGSHISTORIE.md) + Fork + main, nur methodengleiche
   Werte, Entwurf vor dem Posten Peuqui zeigen.

## Wartet auf andere

- **#742 (Skinny-MoE):** 1Cat integriert (KernelConfig, 13 SM75-Fälle mangels Turing übersprungen).
  Sobald Branch/PR steht: SM75-Fälle auf den RTX 8000 und Vollmodell-Läufe fahren und die
  Ergebnisse ungefragt posten (Peuqui, 03.10.).
- **#822 (unser #611):** Entwurf; 1Cat untersucht Abweichungen zwischen frischem Prozess und
  wiederverwendetem Compile-Cache — dasselbe Muster wie unser Befund vom 03.10. (Timing bzw.
  Compile-Artefakt, Daten nachweislich identisch).
- **#715** offen; **#739** (zwei Testfehler auf main) offen, Ursachen auf 3efe512d unverändert.
- Community: #836 (MTP4 über TP4×PP2, nahe an unseren PP-/MTP-Fixes), #746 (Flash-Next auf 4× V100),
  #724 (Flash-Next bei langem Kontext langsam — Ansatzpunkt PLE-Plattenstufe).

## Aufräumen, wenn Zeit ist

- Worktree `1Cat-vLLM-pledisk-verify` und Branch `ple-disk-verify-debug` (lokal, nie gepusht).
- Im 1Cat-Repo blockiert `.git/worktrees/1Cat-vLLM-work/gc.log` (02.10.) den Auto-gc; harmlos,
  gc nur außerhalb von Messungen.
- AI-Connect: Die Wächter-Laufzeit (`timeout 7200000`) steht seit 04ad4ee in der
  peer_read-Beschreibung; wirksam nach `sudo systemctl restart ai-connect-mcp.service` (Peuqui).

## Entschieden, nicht neu aufrollen

- Store-Stufe/Kartenliste zurückgebaut (Archiv-Branch, STAND 44); Untergrenze für freien Host-RAM
  verworfen; `HOST_GIB` bleibt.
- TP4 bei uns verworfen (STAND 30); FP8-Drafter auf RTX nicht bauen (STAND 46).
- Zeilen-Cache 0,5 GiB (1–4 GiB ohne Mehrwert), MemoryHigh bleibt 16 GiB; Schritt 4 (Prefetch)
  nicht bauen, lohnt kaum noch.
- Kleine Helfer (bge-m3, Qwen3-VL) bleiben auf llama.cpp.

## Dauerhafte Regeln

- Vor jedem Rückbau Archiv-Branch + Tag pushen.
- Speichertests brauchen eine Mutationsprobe; Tests dürfen Invarianten nicht vortäuschen.
- KV-Budget ist kein A/B zwischen Einzelboots; erste lange Anfrage nach einem Boot zählt nicht.
- Echte Texte für PLE-Messungen; lange Messungen als `systemd-run --user`.
- pre-commit liegt in `~/.venv/precommit/`.
