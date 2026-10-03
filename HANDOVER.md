# Übergabe — Stand 03.10.2026 abends (Projekt vllm-research)

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

1. **Branch `fork-main` aufbauen** (Peuqui: Basis = komplettes 1Cat main, unsere Zusätze obendrauf als
   KernelConfig-Felder, als PRs anbieten; Ziel: auf unserem Rechner in Qualität und Tempo wie die
   Produktion). Basis: main ≥ 8002bc107 plus #856 und #857 (bis 1Cat sie mergt). Die Gegenprobe vom
   03.10. nachmittags (STAND, Nachtrag) zeigt, was main fehlt:
   - **DSv4 PP5: Decode 125–127 statt 85–86 ms, Prefill 35,7k 17,1 statt 15,3 s** (#837-Baum). Belegt:
     Indexer-Decode unter vollen Graphen auf Paged-Triton statt cuBLAS (`_decode_cublas_blocker`,
     „live key bound“). Weiter zerlegen: QPN8 nur M=1..8 (unser #750 voll), Skinny-Split-K für MXFP4
     gegen unsere Werte, Compile-Weg („no-compile decode graph is requested“). Einzeln an/aus messen.
   - **Flash-Next PP4:** 1Cats Kaskade nutzt die VRAM-Reste der Pipeline-Karten nicht (30,34 GiB auf der
     Platte), kein Zeilen-Cache → unsere PLE-Plattenstufe auf #806 portieren (Zeilen-Cache als
     KernelConfig-Feld, Seitentabellen-Freigabe unter `ple_disk_release_pages`, VRAM-Stufe auf den
     Pipeline-Karten). Entwurfsvokabular: vorher 1Cats #821 prüfen (bringt `sm70_mtp_greedy_draft_vocab`).
   - Weiter obendrauf: #715, #611 (bis #822 gemergt), FLA-Shared-Memory-Fix (841656ec).
   - Testbaum mit nativem Bau: `1Cat-vLLM-pr837` (Branch `test-837-pleadmit`, cef0a2e4b). Für
     `fork-main` neu bauen (Rezept im Gedächtnis), Läufe nach `ab_837*.sh`.
2. **PR-Kandidat:** doppelte Speicherspitze beim Stapeln der FP8-MTP-Experten für TurboMind
   (`fp8_sm70_moe.py:272`), ließ PP4 ohne Skinny nicht starten; wie #743 für MXFP4.
3. **#674-Leistungsbeitrag:** Historie (LEISTUNGSHISTORIE.md) + Fork + main, nur methodengleiche
   Werte, Entwurf vor dem Posten Peuqui zeigen.

## Erledigt 03.10. nachmittags

- Startfehler auf main geklärt: PP4/TP2×PP2 = Fehler in main → **PR #856**; DSv4 zuerst unser Eintrag
  (`VLLM_SM70_QUANT_BACKEND=marlin`), auf #837 dann zweiter main-Fehler → **PR #857**. 27B: main =
  Produktion.
- **#742 gemergt über #837**; native Gegenprobe gepostet (#837 issuecomment-5970301676): 13 SM75-Fälle
  grün, Vollmodelle, Skinny an/aus. #674 aktualisiert. Texte: `upstream-contrib/03-1cat-issues/
  gesendet-2026-10-03-nachmittag.md`.

## Wartet auf andere

- **#856, #857** (unsere Fixes, 03.10.), **#715**, **#611** (#822 Entwurf; 1Cat untersucht
  Abweichungen zwischen frischem Prozess und wiederverwendetem Compile-Cache). Alle vier mergen
  sauber in main 8002bc107. **#739** offen.
- Community: #836 (MTP4 über TP4×PP2), #746 (Flash-Next auf 4× V100), #724 (Flash-Next bei langem
  Kontext langsam).

## Danach (Peuqui, 03.10. nachmittags): Platz schaffen, dann GLM-5.3-Flash prüfen

- **Erst nach allen anstehenden Arbeiten.** Löschliste erstellen und Peuqui zur Freigabe zeigen:
  Python-/Compile-Caches (`~/.cache/uv` 16 GiB, `pip` 3,1 GiB, `vllm*` 11 GiB, gemessen 03.10.),
  alte Modelle im HF-Cache (192 GiB, gegen llama-swap-Config abgleichen), erledigte PR-Worktrees
  (Projektordner 42 GiB). Vor jedem Löschen `readlink -f`; nichts Wichtiges.
- **amd/GLM-5.3-Flash-Quark-MXFP4:** 185,1 GB = 172,4 GiB (62 safetensors), MIT,
  `Glm5NextForConditionalGeneration`, `quant_method: quark`. SSD hat 196 GB frei → vorher aufräumen.
  Recherche (Gedächtnis `project_model_candidates_2026-10`): Vision ja (Bild + Video), KDA + DSA + mHC,
  KV billig (~12 KiB/Token geschätzt); Quark-FP8-Schichten verlangen Capability 89 → Lader für
  SM70/SM75 und MXFP4-Experten an Skinny anschließen. NVFP4-Fassungen (181–190 GiB) passen nicht.
  Eigenes Projekt nach dem ganzen Rest (Peuqui).

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
