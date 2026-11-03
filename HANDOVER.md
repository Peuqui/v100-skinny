# Übergabe vllm-research (v100-skinny / 1Cat-vLLM auf dem Mini) — Stand 25.10.2026 ≈ 10:55

Geschrieben auf Peuquis Bitte kurz vor der Kontextkompression, aus dem Gedächtnis der Sitzung ohne Dateilesen. Maßgeblich bleiben
`STAND.md` (Betriebsstand, SSOT), der Code und das Projekt-Gedächtnis (`MEMORY.md`). Vorherige Fassungen:
`handover/archiv/HANDOVER-2026-10-25-1053.md` (vorige), `…-1050.md`, `…-1015.md`, `…-0250.md`.

**Fortschreibung 02.11. 06:05:** Diese Übergabe ist veraltet. Maßgeblich ist der oberste Nachtrag in `STAND.md` (Produktion, PR-Stapel, nächste Schritte); die letzte ausführliche Übergabe steht in `.agent-orc/handover.md` (29.10.).

**Fortschreibung 25.10. 19:20:** Die jüngste Übergabe ist `.agent-orc/handover.md` (18:25); danach sind die 1Cat-PRs **#1263** (Scheduler-Fix) und **#1264** (GDN-Metadaten-Gates für Turing) gesendet worden (oberster Nachtrag in `STAND.md`, Gedächtnis `reference_1cat_main_red_tests_and_ci_2026-10-25`). Produktion unverändert (GLM prod-sched, 27B q27-meta, DSv4 dsv4-sched). Nächster Hebel: GLM-`lm_head` auf 8 Bit.

## 1. Ziel

vLLM auf dem Mini für AIfreds große Modelle: 3× Tesla V100 (PCI 1/3/4) + 2× Quadro RTX 8000 (PCI 0/2). Produktion:
DeepSeek-V4-Flash 284B (PP5, DSpark), Qwen3.8-27B (TP2 auf den RTX), **GLM-5.3-Flash (PP5, DFlash2) = Schwerpunkt dieser Sitzung**.
Flash-Next 180B ist eingefroren (kein Aufwand). Fork `github.com/Peuqui/1Cat-vLLM` (Remote `fork`; `origin` = 1CatAI/1Cat-vLLM).
Peuqui (21./25.10.): **Optimierung hat Vorrang, 1Cat ist sekundär, aber im Auge behalten**; am Ende Upstream-Tag, dann (nur mit
ausdrücklichem Ja je Ziel) Veröffentlichung inkl. Triton-Patches (1Cat UND Triton upstream; lehnt 1Cat ab → eigener Fork bzw.
vllm-research als eigenes Repo, später entscheiden).

## 2. Aktueller Stand

**Produktion:**
- DSv4 und 27B (unverändert): `prod-fork-main` = 1addd22db, Worktree `1Cat-vLLM-work`, Tag `verified-2026-10-24-tuning-freeze`.
- **DeepSeek-V4 seit 25.10. 16:49 und GLM seit 17:00 mit Scheduler-Fix für langen Kontext** (keine komplette Token-Liste mehr je Decode-Schritt an den V2-Runner; DSv4 Tag `verified-2026-10-25-dsv4-sched`, Baum `1Cat-vLLM-dsv4-sched` f4faec141, Zweig `dsv4-sched-v2`; GLM Tag `verified-2026-10-25-glm53-sched`, Baum `1Cat-vLLM-glm53-prod-sched` 31d73aefa, Zweig `glm53-prod-sched`; beide gepusht; Quell-Zweig `sched-v2-no-token-history` in `1Cat-vLLM-glm53-dev2`): Schritt bei 287k Token DSv4 85,8 → 72,9 ms, GLM bei 516k 114,5 → 96,7 ms (weniger ist besser), kurzer Kontext unverändert, Texte/Pool/Speicher gleich (`prodcheck_2026-10-25_dsv4_sched.log`, `…_glm53_sched.log`). Rücksprünge: Config-Sicherungen `config.yaml.vor-dsv4-prod-sched-2026-10-25` und `config.yaml.vor-glm-prod-sched-2026-10-25`. Das 27B (TP) hat diesen Pfad nicht. Ursache und Zahlen: oberster Nachtrag in `STAND.md`, Gedächtnis `project_sched_token_history_2026-10-25`.
- **27B seit 25.10. 15:29 mit dem schnelleren MTP-Metadatenpfad auf den RTX (Tag `verified-2026-10-25-q27-meta`, Baum `1Cat-vLLM-q27-meta` = 806b1b2ee, Zweig `q27-gdn-meta`, gepusht; DSv4 bleibt auf `1Cat-vLLM-work`):** Decode kurz +5,1 %, nach 14k +5,6 %, GSM8K-30 +4,9 % (mehr tok/s ist besser), Prefill/Speicher/Annahme gleich, Texte 8/8 byteweise gleich, 259 552 Token durchlaufen (`prodcheck_2026-10-25_q27_meta.log`). Rücksprung: Config-Sicherung `config.yaml.vor-q27-prod-meta-2026-10-25`. Der Eintrag nutzt Cache-Kopien `*-q27meta-2026-10-25`. Zahlen und Ursache: oberster Nachtrag in `STAND.md`.
- **GLM seit 25.10. 14:21 mit fusioniertem GDN-Metadaten-Aufbau auch auf der RTX-Stufe (Tag `verified-2026-10-25-glm53-meta`, Baum `1Cat-vLLM-glm53-prod-meta` = 2ae98d460, Zweig `glm53-prod-meta` = `gdn-metadata-turing`, gepusht):** Decode kurz +2,3 %, GSM8K-60 +2,1 % (mehr tok/s ist besser), Prefill/Pool/Speicher gleich, Texte byteweise gleich, Kontrolllauf bis 516k bestanden (`prodcheck_2026-10-25_glm53_meta.log`). Rücksprung: Config-Sicherung `config.yaml.vor-glm-prod-meta-2026-10-25` (→ Baum prod-s5). Details und Ursache (Lücke zwischen den Runden, 6 einzelne Metadaten-Builds auf Stufe 0): oberster Nachtrag in `STAND.md`, Gedächtnis `project_round_turn_gap_2026-10-25`.
- **Davor: GLM seit 25.10. 11:01 auf Fünf-Bit-Skalen umgestellt und abgenommen (Kontrolllauf 11:01–11:44, Tag `verified-2026-10-25-glm53-s5`):** Eintrag
  `GLM-5.3-Flash-NVFP4-DFlash2-vllm`, Baum `1Cat-vLLM-glm53-prod-s5` = 5fef3ad60 (Zweig `glm53-prod-s5`, gepusht). Rücksprung: Config-Sicherung
  `~/.cache/prod-switch-2026-09-29/config.yaml.vor-glm-prod-s5-2026-10-25` über `~/.config/llama-swap/config.yaml` kopieren (zeigt dann auf Baum
  `1Cat-vLLM-glm53-prod` 31f961300, Tag `verified-2026-10-25-glm53-csf`). Kontrolle `prodcheck_2026-10-25_glm53_s5.log`; Einzelheiten und Zahlen
  im obersten Nachtrag von `STAND.md`; `GLM-AIFRED-ANBINDUNG.md` ist nachgezogen. Fensterende war bei AIfred und Agent-Orc um 11:45 gemeldet.
- Kontrolllauf-Kurzfassung: 8 von 8 Qualitätstexten byteweise gleich zur Vorgängerproduktion (drei von Hand gelesen), GSM8K-30 29/30, Robustheit 5/5,
  Maximalkontext 516 196 Token durchlaufen (Prefill 251,63 s, 118,9 ms/Schritt, 25,2 tok/s), Pool 645 277 Token, Boot ≈ 11:35 min.

**Abgenommene Fünf-Bit-Skalen (S5) für GLM — Beleg**
- Zweig `moe-s5-scales` (Fork gepusht, auf `fork-main-v3` = 31f961300): e6483cf55, 3b49b07ba, 64cca01f4, **5fef3ad60** (= Abnahmestand);
  5bbe3fa84 (nur Log-Zeile „Available KV cache memory per worker", nur in `1Cat-vLLM-glm53-dev2`, nicht in der Produktion).
- Vollmodell-A/B `s5d` (S P S P, Prefill direkt, Raster frei): Decode kurz +5,6 % (S 35,51/35,20 gegen P 33,35/33,60 tok/s), nach 14k
  Kontext +6,7 % (48,30/48,39 gegen 44,23/46,36), Prefill 8,7k −4,2 % Zeit (4,30 s gegen 4,49 s), Annahme 5,5109, GSM8K-60 59/60,
  Qualitätstext S#1 bytegleich. **Kosten: Pool 645 277 statt 667 683 Token (−3,4 %)**, begrenzt von Stufe 4 (7 der 42 Schichten mit
  Spannweite > 31 behalten CSF + Raster: Schichten 36, 40–44 in Stufe 4).
- Störungen im Lauf (AIfred-sandbox-install CPU/RAM, Agent-Orc CPU, keine GPU) ohne erkennbare Wirkung (gestörte = ungestörte Arme).
- Zustand 11:45: alle fünf Karten 0 MiB, llama-swap ohne geladenes Modell (lädt GLM bei der nächsten Anfrage), Config = neuer Eintrag.

**Umstellungswerkzeug (benutzt):** Baum `1Cat-vLLM-glm53-prod-s5` (`_C` 490e1c9bbbc3), Caches `~/.cache/{torchinductor,triton-cache,vllm}-glm53-prod-s5-2026-10-25`;
`~/.cache/bench-scripts/`: `glm53_prod_entry_s5_2026-10-25.yaml`, `install_glm_prod_s5_entry_2026-10-25.sh`, `validate_glm_prod_s5_2026-10-25.sh`,
`validate_glm_prod_s5_max_2026-10-25.sh`, Kette `run_glm_prod_s5_validation_2026-10-25.sh` (Einheit `glm-s5-validate-2026-10-25`). Der eingebaute
Textvergleich der Abnahmeskripte geht jetzt gegen `validate_glm_prod2_2026-10-25/quality.json` (die frühere Referenz E512 war vor dem Denkfeld-Fix).

## 3. Getroffene Entscheidungen (nicht neu aufrollen)

- Produktionsumstellung nach belegter Abnahme (Tempo UND Qualität mindestens gleich) selbst durchführen, ohne Rückfrage (Peuqui 23.10.);
  danach Kontrolllauf, Verified-Tag, STAND.md, Bericht. Pool −3,4 % im Bericht nennen.
- Optimierung vor Upstream; Upstream-Tag erst nach dem Optimierungspaket; Veröffentlichung nur mit ausdrücklichem Ja je Ziel
  (vorher Lizenzvermerke: MIT der skinny-Kerne aus dnv2003/v100-skinny, Apache-2.0 vLLM, Triton MIT; keine internen Pfade/Mail-Adressen/
  Hostnamen; Secrets-Scan; Doku Englisch; keine Markdown-Auszeichnung in nach außen gehenden Texten, jede Zahl aus einer Messdatei).
- Kachel-Scheduler für den RTX-MoE-Kern gemessen und VERWORFEN (Zweig `moe-scheduled-rtx` als Archiv). Chunk 1020 bei DSv4 bleibt.
  Indexer-Puffer-Fix (min(40,max_num_seqs)) bringt bei DSv4 keinen Speicher.
- Hop-Packing zwischen den Stufen bringt wenig (≈ 0,4 ms je Hop, bandbreitengebunden).
- `__launch_bounds__(32*SPLITK, (S5 && RB > 1 && SPLITK <= 8) ? 2 : 0)` — Wert 0, nicht 1 (sonst +23 Register bei Nicht-S5-Varianten).
- Nichts ungefragt hochladen (keine Artefakte/Docs/Gists); keine Pakete spontan installieren; kein clang-format vorhanden (nicht installiert).

## 4. Nächste Schritte (in dieser Reihenfolge)

1. **Erledigt (25.10. 11:45): GLM-Umstellung auf S5** — Installation, Kontrolllauf (normal + Maximalkontext), Tag, Push, STAND.md,
   Projekt-CLAUDE.md, `GLM-AIFRED-ANBINDUNG.md`, Fensterende gemeldet. Peuqui bekommt den Bericht (Pool −3,4 % genannt). Falls das Gedächtnis
   noch nicht nachgezogen ist: `project_glm_dflash2_pp5_kv_context_2026-10-24` und `project_s5_scales_fused_decode_2026-10-25` auf „Produktion" setzen.
2. **Erledigt (25.10. 15:45; Peuqui: „folge deinem Vorschlag"): 27B-Metadatenpfad umgestellt** (Zahlen oben und im obersten Nachtrag von `STAND.md`). Vorarbeit: Quelltext zuerst gelesen (Regel „vor aufwendigen Tests Sperren suchen"): der MTP-Pfad des 27B (Eintrag `Qwen3.8-27B-NVFP4-vllm`, `method: mtp`, k = 3, TP2 auf den RTX) ist nur durch `mamba_hybrid.py:128` (`is_device_capability(70)`) gesperrt, die Richtlinien `shared_mtp_metadata`/`fused_mtp_metadata` sind Standard an, Cache-Modus ist `align`. Die vorhandenen nsys-Aufnahmen (`nsys/q27-tpo-ce-1023`) zeigen 1,2…1,9 ms Lücke je Decode-Schritt (Schritt 32,7 bzw. 44,3 ms) mit demselben wiederholten Kernelblock wie bei GLM. Umgesetzt mit Commit 756620999 (Zeile 128 → `is_device_capability_family(70)`), Testbaum `1Cat-vLLM-q27-meta`, eigener A/B `ab_q27meta_2026-10-25.sh` (Start über `start_ab_q27meta_2026-10-25a.sh`), Umstellung über `start_q27_prod_meta_2026-10-25.sh`. Als Nächstes: DSv4 bei langem Kontext (Lücke 4,4 ms von 78 ms) per `VLLM_SM70_ASYNC_CPU_TRACE`, GLM-Wendezeit nach dem Fix neu vermessen (nsys).
   **Arbeitsweise (Peuqui 25.10.):** keine zusammengesetzten Befehle (`cd …&&`, Schleifen, `$()`, Variablen, `;`-Ketten) — jedes Mal eine Rückfrage; Mehrstufiges als Skript in den Scratchpad und mit `bash /Pfad.sh` / `python3 /Pfad.py` starten (Gedächtnis `feedback_avoid_approval_commands`).
3. **Ab hier GPUs frei** (kein Fenster offen). Nächste Messfenster wieder vorher bei aktiven Peers ankündigen (`peer_list` zuerst, nie an `*`),
   Wächter für jeden Job, ein blockierter Schritt heißt „Ich warte auf X" + Wächter (Gedächtnis `feedback_blocked_step_say_waiting_set_watcher`).
4. Offen (nach Wert): Wendezeit 4,1 ms je GLM-Runde (inzwischen durch den Metadaten-Fix um etwa 2 ms je Runde verkürzt, Rest ungemessen) (Stufe 0 bereitet die nächste Runde CPU-gebunden mit 119 Kleinkerneln vor;
   `VLLM_SM70_ASYNC_CPU_TRACE`); Pool-Diagnose (Log aller Stufen) und ggf. „Raster nur ≤ 64 Experten + Prefill in Expertenstücken" — nur wenn
   Peuqui den Pool will; V100-S5-Decode spaltenweise planen (−0,1 ms/Schicht); `lm_head`/Entwurf in 8 Bit (−2…4 ms); MoE-Bandbreite 90 → 96 %;
   DSv4 gewichts-stationärer MoE-Kern nur nach Peuquis Priorität.
   **Stand 19:20:** `lm_head`/Entwurf in 8 Bit ist der nächste Hebel (≈ −2 ms je Runde, ≈ 0,6 GiB frei auf Stufe 4; Plan in `.agent-orc/handover.md` Abschnitt 5.1). Die kleinen 1Cat-PRs (#1263, #1264) sind draußen; ihre CI zeigt dasselbe Rot wie main.
4. **Upstream-Tag** nach dem Optimierungspaket: `origin/main` (6a143197e, 89 Commits/237 Dateien, fast nur Flash-Next) in `fork-main-v3`
   mergen (Probe: 3 triviale Konflikte in `docs/design/architecture/runtime_reference.md`, `vllm/config/kernel.py`,
   `vllm/v1/worker/gpu/model_states/mamba_hybrid.py`), Tests, Mini8, GLM-Qualität bytegleich, kurzes A/B; #1216/#1217/#1235 rebasen
   (Force-Push nur auf Ansage); PR-Pakete ohne Triton-Abhängigkeit (Turing-MoE-Kern, 27B-Prefill-Paket, DSv4 wo_a + Tuning-Freeze,
   GLM-Paket, danach S5). AGENTS.md-Pflichtregeln beachten (Duplikatsuche, Testkommandos mit Ergebnissen, KI-Hinweis). Erst danach,
   mit ausdrücklichem Ja, Veröffentlichung inkl. Triton-Patches.

## 5. Wichtige Dateien und Bäume

- Bäume (`/home/mp/Projekte/vllm-research/`): `1Cat-vLLM-glm53-dev2` (Entwicklung, Branch `moe-s5-scales`, HEAD 5bbe3fa84),
  `1Cat-vLLM-glm53-prod-s5` (neu, eingefroren), `1Cat-vLLM-glm53-prod` (GLM-Prod 31f961300), `1Cat-vLLM-work` (DSv4/27B-Prod),
  `1Cat-vLLM-fork-main` (Fork-Arbeit; hat `origin/main` geholt). Editable-Mapping der venv `/home/mp/vllm/venv` zeigt auf `1Cat-vLLM-work`.
- Code S5: `csrc/sm70_turbomind/ops/skinny_moe_qpn_sm70.cu`, `csrc/ops.h`, `csrc/torch_bindings.cpp`,
  `vllm/model_executor/layers/quantization/utils/s5_scales.py` (neu), `…/fused_moe/experts/skinny_sm70_moe.py`,
  `…/quantization/utils/csf32_expand.py` (schwache `_SHARED_RASTERS`), `…/quantization/modelopt.py`;
  Tests `tests/kernels/moe/test_skinny_sm70_moe.py`, `tests/kernels/quantization/test_s5_scales.py` (RTX: 162 passed, V100: 134 passed).
- Skripte/Logs: `~/.cache/bench-scripts/` (`ab_s5_2026-10-25.sh`, `build_moe_dev_2026-10-25.sh`, `triton-sm7x/*`, `gpu_log.sh`, `watch_job.sh`,
  `scan_arch_gates.sh`, `validate_glm_prod_2026-10-25.sh` als Vorlage, `prod_robustness_probe.py`).
- Doku: `v100-skinny/STAND.md` (SSOT), `FORTSCHRITT.md`, `docs/journal/LEISTUNGSHISTORIE.md`, `v100-skinny/GLM-AIFRED-ANBINDUNG.md`,
  `upstream-contrib/`; Gedächtnis `project_s5_scales_fused_decode_2026-10-25`, `project_1cat_upstream_state_2026-10-25`,
  `feedback_optimization_first_upstream_secondary`.

## 6. Fallen und Regeln, die Zeit gekostet haben

- Pool = Minimum über alle Pipeline-Stufen; bei S5 begrenzt Stufe 4 (letzte RTX).
- Der A/B-Lauf bricht jetzt bei gescheitertem Arm ab; AIfreds Last (CPU/RAM) und Agent-Orc-Last während gezählter Messläufe vorher abfragen.
- Wartedienste mit `systemctl --user` brauchen `XDG_RUNTIME_DIR`/`DBUS_SESSION_BUS_ADDRESS`; `sleep ≥ 45` im Vordergrund wird blockiert.
- Während gezählter Boots nichts Schweres (kein nvcc, Bau, Checkout, Tests); Messbaum nicht ändern; Messläufe aus Skriptkopien.
- Karten über GPU-Orc (`POST http://localhost:5070/free`) freigeben; STT-Container nicht stoppen; AIfred-Dienst bleibt aus bis Projektende;
  AIfreds Scheduler lädt täglich 06:45 DeepSeek-V4 (Messfenster nicht darüber); Borg-Backup sonntags 01:00, Auto-Reboot Mo/Mi/Fr 05:30.
- Tempo-Aussagen nur aus A/B/A/B, Texte byteweise vergleichen, Tests auch mit großen Prompts bis nahe Max-Kontext, `memory.used` je Karte protokollieren.
- Force-Push, Löschen von Branches/Tags und PRs in fremden Projekten nur auf ausdrückliche Ansage; Commits mit echten Umlauten,
  Autor `Peuqui <43776522+Peuqui@users.noreply.github.com>`.
- `vllm-research/vllm-bench` (Peuquis Repo `Peuqui/vllm-bench`, Benchmark-Kampagne 08/2026): lokal auf e9c1529, `origin/main` ist 2 Commits weiter (MIT-Lizenz d10bf1d, README en/de 37a89d4); lokal nicht committet nur die Pfadanpassung `BENCH=` in `run-quality-matrix.sh` (neuer Ort unter vllm-research). Vor einem Push dort `git pull --rebase`; ob die Pfadänderung mitgehen soll, entscheidet Peuqui (Hinweis Agent-Orc 25.10. 11:47, nichts angefasst).
