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
