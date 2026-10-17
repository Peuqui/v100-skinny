# Übergabe — Stand 17.10.2026 nachts (Projekt vllm-research)

Claude Code startet in `/home/mp/Projekte/vllm-research` (Projektregeln in `CLAUDE.md`, Gedächtnis unter
`~/.claude/projects/-home-mp-Projekte-vllm-research/memory/`, AI-Connect-Peer `Mini:vllm-research`).
Betriebsstand (SSOT): `STAND.md`, Nachtrag 16./17.10. nachts unter „Laufzeitumgebung“.

## Stand 17.10.2026 nachts

- **Produktion seit 17.10. ~04:25 = `fork-main-v2b` 9e81ffeb0** (Freigabe Peuqui) = df0941355 + #1219 + Testreparatur
  + Turing-Registerdeckel für `delta_h`. Kontrolle aller vier Modelle bestanden (STAND 17.10. früh), Tags
  `verified-2026-10-17-turing-maxnreg`, `archive-prod-fork-main-2026-10-17` (df0941355).
- **AIfred-Dienst hat Peuqui am 16.10. abends gestoppt** — Neustart macht Peuqui.
- **Fork-Baum `1Cat-vLLM-fork-main` steht auf Branch `fa2-packed-gqa-verify` (df08fe7a9, gepusht)** = 9e81ffeb0 + zerlegte
  Verify-Attention für FA2. Tests grün; Serving-A/B 27B läuft (`ab-gqapack-1017`). Bei Erfolg: `fork-main-v2b`
  vorspulen, Produktion umstellen (Okay Peuqui), sonst Branch liegen lassen.
- **Als Nächstes:** PLE-Stufe „freie GPU“ neu bauen (Druckprofil spricht dafür, STAND 17.10. früh), vorher VRAM-Bedarf
  von STT-GPU-Worker, Qwen3-TTS, Qwen3-VL-4B messen; QSA-Indexer-Weiche für Turing (A/B PP4-Prefill).
- **Entscheidung offen (Empfehlung: nein):** probabilistische Entwürfe für Flash-Next — Prosa +2,5 %, Code −2 %.
- **Nächste Schritte (Peuqui 16.10.: „allen Punkten nachgehen“), nach Nutzen:**
  1. 27B-Lang-Kontext-Decode: MTP-Verify-Attention zerlegen (Kontext gepackt ohne Maske + kausaler Schwanz, LSE-Merge)
     oder `sm70_grouped_long` auf Turing prüfen; Ursache belegt (Gedächtnis `project_q27_longctx_decode_slowdown`).
  2. RTX-Prefill: frisches Profil (Aufteilung 11,10,15,12 + maxnreg), dann GDN-TileLang für sm_75 (≤ 64 KB Shared
     Memory), QSA-Tuning für Turing, Skinny-MoE-Prefill auf Turing (Volta-Befehle?) — `project_turing_rtx_prefill_levers`.
  3. Streuung über Boots bei Flash-Next: derselbe Prompt mehrfach in einem Boot ohne Präfix-Cache; Präfix-Cache aus;
     PLE-Überlappen/Plattenstufe aus — `project_greedy_spread_triton_autotune` (Autotune widerlegt).
  4. PLE bei langem Kontext: Anteil der Plattenstufe am 184k/250k-Prefill messen (Speicherdruck), dann über eine
     Stufe „freie GPU“ entscheiden (Code nicht mehr im Fork).
  5. Klein: `git fsck` im 1Cat-Repo + Sperrdatei `gc.log` (Okay Peuqui vorhanden), die 14 alten Test-Attrappen,
     K=3 für Prosa, Prompt-Lookup, 27B DFlash2 gegen MTP, 27B-V2-OOM auf V100 (#1220).
  Neue PRs an 1Cat erst, wenn 1Cat auf die 11 offenen reagiert (Stand 16.10.: keine Reaktion; main seit 13.10. ruhig).

## Stand 16.10.2026 mittags

- **Produktion seit ~11:40 = `fork-main-v2b` df0941355** (Tag `verified-2026-10-16-longctx-indexer`): DSv4-Indexer
  für Kontext über 131.072 Token behoben (Score-Kacheln im cuBLAS-Dekodierpfad, letzter Bucket bei der Modelllänge).
  287k: 242 → 93 ms je Schritt (13 → 34 tok/s), kurz unverändert. Alle Produktionsmodelle kontrolliert (STAND 16.10.
  vormittags). Archiv `archive-prod-fork-main-2026-10-16b` (62b127eab).
- **AIfred-Endloslauf 04:51–08:05 war KEIN V2-Fehler:** V1/V2 bei langem Kontext gleich, Kopiertest gleich. Auslöser
  im AIfred-Auftrag (Sandbox `documents` nur lesend → Codine schrieb base64 in den Kontext, keine Längengrenze).
  **Behoben im AIfred-Repo (Peuqui 16.10. nachmittags), Commit a5ab62d0 auf `main`, gepusht:** `execute_code_write`,
  `delete_file`, `delete_folder` auf Stufe 2 (`chromadb_clear` bleibt 3), `MAX_OUTPUT_TOKENS = 64000` als `max_tokens`
  jedes OpenAI-kompatiblen Aufrufs ohne eigenes `num_predict`. Mit Peuquis Freigabe zusammen mit den 7 ungepushten
  Personarium/Vision-Commits vom 14.10. auf die 3 Traffic-Bot-Commits rebased und gepusht. AIfred-Neustart macht Peuqui.
- **Neue Regel (Peuqui):** jede Messreihe auch mit Prompts bis nahe Max-Kontext (`longctx_decode_probe.py`, kürzt
  per `/tokenize` auf die `max_model_len` des Eintrags); fest im Standard-A/B-Skript `run_ab_q27v2v100_2026-10-15.sh`
  (Backup `.vor-longctx-2026-10-16`). Flash-Next PP4 geprüft: 33k / 184k / 250k = 55,5 / 62,8 / 66,7 ms je Schritt,
  kein Einbruch.
- **Nachmittags erledigt (Freigabe Peuqui, „PRs doppelchecken“):**
  - **#1221** DSpark auf MRV2 + Context-Buckets (Zweig `pr-mrv2-dspark` 450e65c52) und **#1222** Indexer-Fix über
    131k (Zweig `pr-dsv4-longctx-indexer` f755ff05a, dritter Commit repariert die drei auf main roten Wrapper-Tests in
    `test_cudagraph_dispatch.py`). Beide gegen main 6f3557c97 auf V100 und RTX getestet, Duplikat- und Merge-Prüfung,
    jede Zahl gegen die Messdatei, Live-Text = Entwurf. Texte `upstream-contrib/09-1cat-prs-2026-10-16/`.
  - **#1219** umgebaut (93780942b: P2P-Prüfung nur in der TP-Gruppe auf deren Geräten, ohne Welt-Barriere), Abnahme
    auf `fork-main-v2b-p2p` 3b60d0303 (27B TP2 V100 ohne Workaround kein Hänger, Flash-Next TP2×PP2 startet, Greedy
    je 3/3), Text ersetzt, „ready for review“, Kommentar live. Produktion trägt #1219 weiterhin nicht.
  - Fork `fork-main-v2b` = 4e9ae3c3a (df0941355 + Testreparatur), gepusht; Produktion bleibt df0941355.
  - **Lang-Kontext-Lauf im Standard-A/B-Skript** (`run_ab_q27v2v100_2026-10-15.sh`, Funktion `run`, Backup
    `.vor-longctx-2026-10-16`); `longctx_decode_probe.py` kürzt Prompts per `/tokenize` auf die `max_model_len` des
    Eintrags. Test Flash-Next PP4: 33.208 / 184.133 / 249.950 Token → 55,5 / 62,8 / 66,7 ms je Schritt (kein Einbruch
    bei Flash-Next).
- **Nächste Schritte:**
  1. Die alten Attrappen aus der Liste unten reparieren (`test_dflash_mrv2_config.py` 4,
     `test_gpu_model_runner_v2_greedy.py` 9, `test_gpu_model_runner_v2_prefill_dispatch.py` 1 — alle auch auf main rot).
  2. Turing: ptxas-64-Register-Deckel für Triton auf sm_75 (Gedächtnis `project_turing_triton_register_cap`) —
     GDN-Prefill auf den RTX.
  3. Greedy bei langem Kontext über Boots nicht wortgleich (alle Arme) — Ursache suchen.
  4. 27B DFlash2 gegen MTP auf V2 (RTX), 27B-V2-OOM auf V100 (verwandt mit 1Cat-Issue #1220?).
  5. AIfred-Befund (Sandbox `documents` nur lesend, base64 im Kontext, keine Längengrenze) — Peuqui fragt noch.

## Stand 16.10.2026 nachts (vor dem Auto-Reboot 04:30)

- **Richtung (Peuqui 15.10.):** neue Arbeit nur noch für den V2-Runner; V1 nur Fehlerpflege
  (Gedächtnis `feedback_focus_v2_runner`).
- **Produktion:** `fork-main-v2b` 62b127eab = baaa5a687 + DSpark auf V2 + Context-Buckets für Compressed-Index-Graphen
  (Tag `verified-2026-10-16-v2-dspark`). Der P2P-Fix ist NICHT drin (Deadlock bei TP2×PP2, siehe unten). DSv4 (K=5 und Coding K=7), Flash-Next und 27B laufen alle auf V2.
  Archiv: Tag `archive-prod-fork-main-2026-10-16` (baaa5a687). DSv4 V2 gegen V1: +11–12 % kurz, −5 ms je Schritt
  bei 35,7k, Greedy identisch (STAND 16.10. nachts).
- **PRs 15./16.10.:** #1213, #1214, #1216, #1217, #1218 offen; #1219 (P2P-Prüfung) auf ENTWURF gesetzt: Deadlock bei
  gemischter PP (Welt-Barriere in `gpu_p2p_access_check`) und Gruppenrang als Geräteindex. Zuerst umbauen (Prüfung in
  der TP-Gruppe auf deren Geräten, Ergebnis über die Gruppe abstimmen), auf TP2 V100 UND TP2×PP2 testen, dann wieder öffnen.
- **Nächste Schritte:**
  1. DSpark-auf-V2 + Context-Buckets als PR an 1Cat vorbereiten (Upstream-Gutschrift vllm-project/vllm
     DSparkSpeculator; Duplikatsuche). Tests liegen auf `fork-main-v2b` 02138d3ee
     (`tests/v1/worker/test_mrv2_dspark_context_buckets.py`, 49/49 mit den Long-Attention-Tests auf V100 und RTX).
  2. (erledigt 03:10) DSv4-Coding (K=7) auf V2: +10–12 %, umgestellt.
  3. 27B DFlash2 auf V2 gegen MTP auf V2 (RTX), dann Decode-Profil 27B/V2 auf RTX (Turing-Hotspots).
  4. 27B-V2 auf V100 OOM beim 29k-Prefill (V2-KV-Schätzung großzügiger) — für Produktion irrelevant, für 1Cat-Bericht.
  5. Alte Attrappen reparieren (rot auch auf main): `tests/v1/spec_decode/test_dflash_mrv2_config.py` (4),
     `test_draft_profile_scope.py::test_qwen4exp_mtp_model_config_is_marked_before_revalidation` (`kernel_config` fehlt).
- **Für Peuqui:** TTS-Suite legt Qwen3-TTS auf die GPU mit dem meisten freien Speicher (23:16 auf GPU 0) —
  kollidiert mit vLLM-Boots (alle brauchen GPU 0). Reboot-Einzeiler für Mo+Do statt Mo/Mi/Fr steht im Chat.

## Stand 15.10.2026 früh

- **Produktion:** fork-main-1013 aa7087485 (1Cat main 6f3557c97 + unsere Commits + GDN-Prefill-Norm-Fix), Tag
  `verified-2026-10-15-forkmain1013`; alte Produktion Tag `archive-prod-fork-main-2026-10-15` (2991fbfdc), als Worktree
  `1Cat-vLLM-oldprod` für Vergleiche. Details STAND-Nachtrag 15.10. früh.
- **Fork-Branch `fork-main-1013`** steht eine Teständerung weiter (9fad9aaa0, PLE-Tests); Produktion braucht sie nicht.
- **PRs gesendet 15.10.:** #1213 (GDN-Prefill-Norm, Absturz auf Turing-Stufen ohne opake Grenze) und #1214 (vier rote
  PLE-Tests). Texte `upstream-contrib/08-1cat-prs-2026-10-15/body-*.md`. Offen von uns weiter #715, #1183, #1187.
- **Mitmessen bei der nächsten Reihe:** PP4 langer Kontext ~1 % (im Rauschen, GPU-Arbeit identisch).
- **Nächstes laut Peuqui:** Kernel-Optimierung (Warteschlange nach Sinnfälligkeit: PP4-Prefill-PLE-Überlappung auf
  Stufe 0, Oktober-Plan 4/5c/6, Turing-Doppelbau).

## Stand 11.10.2026 abends

- **Produktion:** fork-main-next 2991fbfdc (1Cat main be3722bc0 + unsere Commits + 3 DSpark-Policy-Fixes), Tag
  `verified-2026-10-11-forkmainnext`. DSv4 prosa −0,7 % akzeptiert, bei der nächsten DSv4-Reihe mitmessen.
- **Upstream-Runde abgeschlossen:**
  - #1183 DSpark-Policy-Regressionen offen.
  - #1187 FP8-Small-Shape-Tuning standardmäßig aus, mit Test, offen; Branch `pr-sm70-fp8-tune-default-off` 739083e52.
  - #1006 geschlossen mit Verweis auf #1187.
  - #715 offen und auf main weiter nötig (konfliktfrei).
  - Texte in `upstream-contrib/07-1cat-prs-2026-10-11/`.
- **Kaputte Tests auf 1Cat main:**
  - `test_sm70_warmup.py` hat 14 Fehlschläge (`warmup.envs`, `native_policy`-Fakes); #903 (areslp) schreibt sie um,
    deshalb kein eigener PR (Duplikat).
  - `test_legacy_dispatch_snapshot` schlägt ebenfalls fehl.
- **Nächstes:** Warteschlange nach Sinnfälligkeit — PP-Aufteilung, Oktober-Plan 1/3/4/5/6/7, sync-PP+MTP,
  Turing-Doppelbau.
- **Vorhandene Messzweige im Worktree `1Cat-vLLM-bisect`** (lokal):
  - `meas-fp8default-plus-1183`
  - `pr-fp8-default-backup-e9732e9e0`
  - `pr-fp8-with-testrepair-23ee7c9fa`

## Stand 05.10.2026 spätabends (historisch)

## Läuft gerade

- Nichts. Alle GPUs frei, AIfred-Dienst aus (Peuqui startet ihn selbst). `whisper-stt` läuft nur mit CPU-Modell; vor
  Messläufen GPU-Modell mit `curl -X POST 'http://localhost:5080/unload?device=cuda'` freigeben, nie den Container stoppen.

## Produktion

- **fork-main 65a3bf176** (1Cat main 2164365ab + unsere Zusätze), Worktree `1Cat-vLLM-work`, Branch `prod-fork-main`,
  Tag `verified-2026-10-05-forkmain-2164365`. Alte Produktion: Tag `archive-prod-fork-union-2026-10-05` (138059c82).
- llama-swap: 13 vLLM-Einträge umgeformt (alte Env raus, PLE als `--kernel-config`); Backup
  `~/.cache/prod-switch-2026-09-29/config.yaml.vor-prod-forkmain-2026-10-05`.
- Greedy-Referenzen: `~/.cache/bench-scripts/greedy_refs.sh` (SSOT).
- Leistung gegen alte Prod: DSv4 Decode −13 % Schrittzeit, Flash-Next Decode gleichauf/leicht vorn, Prefill PP4 +0,3 s /
  TP2 +0,2 s (offen), 27B wortgleich.

## Heute geklärt (Details STAND)

- TTFT-Aufschlag und Boot-Instabilität DSv4: TurboMind-Mess-Tuning kleiner FP8-Formen → KernelConfig-Feld
  `sm70_fp8.small_shape_tuning` (Standard aus) + native Setter-Op.
- sync unter PP: zwei Fehler in 1Cats PP-Spekulation behoben (Scheduler + Stufe-0-Token); sync bitgleich mit async, aber
  bei uns nicht schneller (Prefill 4× langsamer, Decode etwas langsamer).
- `Alfinaa9442/v100-skinny` = Schadsoftware-Köder (Kopie von dnv2003 + ZIP). Nicht anfassen.

## Nacht 05./06.10. (autonom)

- **Host-Allreduce** (mzen17, fork-main 81e95a1cd + fixup 06cd400bc: Feld aus dem Compile-Hash): 27B TP2 Decode −10 %,
  Flash-Next TP2×PP2 −4 %, bitgleich; nicht in Produktion (Peuquis Okay), PR erst nach Rücksprache + Hinweis an mzen17.
- **21 rote Tests auf 1Cat main geklärt:** 10 = Fehler aus #885 (PLE-Policy im Ladepfad + falscher Fehlertyp nativ) →
  Branch `pr-ple-row-gather-policy` (23e8b97bb, mit C++); 11 = veraltete Test-Attrappen/Erwartungen → Branch
  `pr-main-test-repairs` (43cce7ded). Beide auch in fork-main (d12c663b8, fae938623). Entwürfe in
  `upstream-contrib/05-1cat-prs-2026-10-05/` (dazu sync-PP und FP8-Tuning von gestern). Fingerprint-Test: Bisect →
  a096d6d28; mehrere ungenutzte Felder gehen in den Compile-Hash (im Entwurf als Beobachtung).
- Prüfbaum `1Cat-vLLM-mainchk` (main 17310de95) wurde einmal voll gebaut, um die PRs auf reinem main zu belegen.

## 06.10. vormittags

- 27B PP2 gegen TP2 gemessen (STAND-Nachtrag 06.10. vormittags): PP2 Prefill −40 %, Decode +40–62 %. TP2 bleibt;
  Overlap von Allreduce und Rechnung im TP2-Prefill wartet auf Peuquis Entscheidung.

## 06.10. mittags — ERLEDIGT bis auf sync-MTP (Stand 13:25)

- **Produktionsumstieg erledigt** 13:00: ee5813de2 + PP4 11,11,14,12, Smoke ok, Tag `verified-2026-10-06-forkmain-hostreduce`,
  Referenzen in `greedy_refs.sh` (STAND-Nachtrag 06.10. mittags). Messfenster bei den Peers beendet.
- **Offen:** sync unter PP mit MTP (Flash-Next) liefert kaputte erste Token → sync-PR bleibt zurück, erst Ursache.
  Danach Qwen3.6-27B-FP8-Vergleich für #1006 (Download läuft) und Overlap-Bewertung.

### Verlauf (historisch)

- **PRs an 1Cat eingereicht:** #1004 (PLE-Policy), #1005 (zwölf Test-Reparaturen), #1006 (FP8-Tuning-Feld, Standard
  aus). Texte: `upstream-contrib/06-1cat-prs-2026-10-06/`. **sync-PP-PR noch zurückgehalten** (Branch
  `pr-sync-pp-spec-decode` 3c0cdb4fe, Text dort): braucht den Flash-Next-PP4-sync-Nachweis aus Schritt 3 der Kette,
  dann Platzhalter `FLASHNEXT_SYNC_RESULT` füllen und senden.
- **fork-main ee5813de2 gepusht** (65a3bf176 + Host-Allreduce + PLE-Fix + Warmup folgt FP8-Feld + Test-Reparaturen),
  Fork-Tests V100 1390 / RTX 1301 grün.
- **Messkette `chain-acc-2026-10-06`** (systemd-User-Unit, Skript `~/.cache/bench-scripts/chain_acc_2026-10-06.sh`):
  1) A/B q27/tp2/pp4 (PP4 mit 11,11,14,12) → `ab_acc_2026-10-06.log`; 2) DSv4 once → `ab_accds_2026-10-06.log`;
  3) Flash-Next PP4 `--no-async-scheduling` once → `ab_syncpp4_2026-10-06.log`. 27B und TP2 ausgewertet: Decode −10 %
  bzw. −4 %, Qualität ok. Steht die Unit auf SIGSTOP (Agent-Orc-Slot), mit `systemctl --user kill --signal=SIGCONT
  chain-acc-2026-10-06` fortsetzen.
- **Danach Produktionsumstieg** (Peuqui hat Host-Allreduce und PP4-Aufteilung freigegeben): `1Cat-vLLM-work` Branch
  `prod-fork-main` auf ee5813de2, native Dateien aus `1Cat-vLLM-fork-main` kopieren (rsync + cmp, Muster in STAND),
  PP4-Einträge (4 Stück, alle `VLLM_PP_LAYER_PARTITION=12,12,12,12`) auf 11,11,14,12, Config-Backup vorher; neue
  Greedy-Referenzen in `greedy_refs.sh`; Verified-Tag; Messfenster-Ende bei allen Peers melden.
- **Download Qwen/Qwen3.6-27B-FP8** (Unit `dl-qwen36-27b-fp8`, startet nach der Kette) nach
  `/home/mp/models/Qwen3.6-27B-FP8-Dense-GDN`: für den direkten Vergleich zu #1006 (1Cats Modell) auf 2× V100 TP2,
  Tuning an/aus.
- **Slot-Regel:** Agent-Orc fragt Slots an → nach dem laufenden Lauf anhalten, „jetzt“, nach „fertig“ weiter
  (Gedächtnis `feedback_slots_for_other_agents`).
- Danach: Overlap-Bewertung (Vorstudie im Gedächtnis `project_todo_2026-10-05_sync_bitstable_ttft`).

## Nächste Schritte

1. PR-Entwürfe an 1Cat: sync-PP-Korrektur (Commit 17024e531) und FP8-Tuning-Feld (65a3bf176) — Peuqui vorher zeigen.
   Duplikatsuche ist gemacht (kein offener PR). PR C (PLE-Plattenstufe auf #885) jetzt möglich; Druckmessung fehlt.
2. mzen17s Host-Allreduce (`skinny_ar.cu`) als 1Cat-Allreduce-Backend für TP2 ohne P2P (Gedächtnis
   `project_mzen17_host_allreduce_candidate`); 27B TP2 A/B messen.
3. Prefill-Rückstand fork-main PP4/TP2 zerlegen; sync-Prefill 4× und async-TTFT +0,2 s als Luft prüfen.
4. 21 auf 1Cat main rote Tests (`upstream-contrib/04-1cat-prs-2026-10-04/tests-main-2164365ab-failing-2026-10-05.txt`)
   als Korrektur-PR-Kandidaten.
5. Ältere Liste: Entwurfsvokabular gegen #821/#922, FP8-MTP-Speicherspitze, #674, GLM-5.3-Flash, DeepSeek-V4-Flash-0731.

## Dauerhafte Regeln

- Auto-Reboot Mo/Mi/Fr 04:30; AIfred lädt 06:45 DSv4 (erster Boot nach Compile-Hash-Wechsel dauert länger).
- Messfenster bei allen Peers ankündigen (`peer_send *`), Agent-Orc baut nur in freigegebenen Lücken („jetzt“/„fertig“);
  Lücke per `systemctl --user kill --signal=SIGSTOP/SIGCONT <unit>` zwischen zwei Läufen.
- Jeder Messlauf mit Wächter starten (Ende + Fehlstart); nach `systemctl stop` die Config selbst aus dem Backup holen.
- `ab_forkmain2_ds.sh`: `MODELS="pp4 tp2 q27 ds"`, `RUN_TAG`, `FM_TREE`, `FM_EXTRA_ENV`, `FM_EXTRA_ARGS`, Modi `ttft`,
  `ttftgreedy`, `syncrepro`, `pvgreedy`, `pyspy`.
