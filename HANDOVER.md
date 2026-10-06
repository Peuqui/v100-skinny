# Übergabe — Stand 05.10.2026 spätabends (Projekt vllm-research)

Claude Code startet in `/home/mp/Projekte/vllm-research` (Projektregeln in `CLAUDE.md`, Gedächtnis unter
`~/.claude/projects/-home-mp-Projekte-vllm-research/memory/`, AI-Connect-Peer `Mini:vllm-research`).
Betriebsstand (SSOT): `STAND.md`, Nachträge 05.10. ganz oben.

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
