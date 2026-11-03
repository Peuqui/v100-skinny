# Aufräumen der 1Cat-vLLM-Arbeitsbäume (02.11.2026, 21:30)

**Ausgeführt von Peuqui am 02.11. bis 22:48:** Gruppen `alt` und `offene-prs` entfernt, 35 Bäume. Danach waren 16 Bäume übrig, genau die Behalten-Liste, und keiner war verwaist. Die Platte hatte 110 G frei statt vorher 58 G. Stichprobe: Die Zweige `prod-v4`, `nf-probe-2026-11-02`, `pr-ple-disk-tier`, `sync-probe-2026-11-02` und `pr-dsv4-prefill-fp32` bestehen weiter. Offen ist noch die Gruppe `nach-v5d`.

Grundlage: Inventar `~/.cache/bench-scripts/worktree_inventory_2026-11-02.tsv` (Skript `worktree_inventory_2026-11-02.py`) über alle 51 Arbeitsbäume.
Für jeden Baum wurden geprüft:

- die Bezüge aus der laufenden llama-swap-Config,
- die Config-Sicherungen, die die Rücksprungskripte zurückspielen,
- die Editable-Installationen der venv,
- Links aus anderen Bäumen,
- laufende Prozesse,
- nicht eingecheckte Änderungen,
- ob der Stand nur im Baum existiert.

Entfernt wird nur der Checkout. Jeder Zweig, jeder Tag und jeder Commit bleibt im Repository. Alle hier vorgeschlagenen Zweige sind außerdem im Fork oder in einem Tag gesichert. Ausnahmen sind die vier lokalen Diagnose- und Probezweige `nf-probe*` und `sync-probe` (bewusst nie gepusht) sowie `pr-dsv4-prefill-fp32`, `pr-triton-sm7x-fp32-scores` und `pr-ple-disk-tier`. Sie bleiben als lokale Zweige erhalten.
Bei Bedarf entsteht ein Baum neu mit `git worktree add`, für native Tests mit einem Bau, der über ccache kurz ist.

Löschskript: `~/.cache/bench-scripts/remove_worktrees_2026-11-02.sh <gruppe> [delete]`. Ohne `delete` macht es nur einen Trockenlauf.
Unmittelbar vor jedem Entfernen prüft es noch einmal Link, Registrierung, Config, venv, Prozesse (Befehlszeile und Arbeitsverzeichnis), Links aus anderen Bäumen und Änderungen. Trifft eines davon zu, verweigert es den Baum.
Probe am 02.11.: v5d, work, stack-all und main-check wurden korrekt verweigert.
Nicht während eines gezählten Boots oder A/B ausführen, denn das Löschen erzeugt viel Last auf der SSD.

## Behalten

| Baum | Grund |
| --- | --- |
| 1Cat-vLLM | Haupt-Checkout mit `.git`, alle Bäume hängen daran |
| 1Cat-vLLM-work | venv `/home/mp/vllm/venv` zeigt per Editable auf ihn (vllm, flash_attn_v100), Links aus 1Cat-vLLM |
| 1Cat-vLLM-v5d, -v5c, -v5b, -v100verify | Produktion (llama-swap-Config) und Ziele der aktuellen Rücksprünge |
| 1Cat-vLLM-stack-all, -main-check | laufender Test-Sweep; danach entbehrlich |
| 1Cat-vLLM-pr-tp-prefill, -pr-dsv4-prefill2, -pr-triton-sm7x, -pr-glm53 | PRs #1279/#1280/#1282/#1286, Push heute Nacht |
| 1Cat-vLLM-pr-dsv4-longctx-decode, -pr-glm53-prefill-head | neue PRs P4/P5, Boots heute Nacht |
| 1Cat-vLLM-v5, -v5a | eine Rücksprungstufe der Einträge auf v5b (`rollback_prod_v5b_2026-10-31.sh`, `rollback_rest_v5b_2026-10-31.sh`); entbehrlich, sobald alle Einträge auf v5d laufen (Gruppe `nach-v5d`) |

## Gruppe `alt`: 20 Bäume, 34 GiB (Empfehlung: entfernen)

Alte Rücksprungstufen vom 27.10. und früher, seit drei Produktionsumstellungen überholt. Danach gehen die Rücksprungskripte `rollback_prod_v4_2026-10-27.sh`, `rollback_prod_v4b_2026-10-27.sh`, `rollback_prod_v5_2026-10-28.sh` (→ v4b) und `rollback_glm_prod_lmhead_2026-10-26.sh` nicht mehr. Die Stände bleiben über die Tags `verified-*` erreichbar.

| Baum | Zweig | MiB | Bemerkung |
| --- | --- | --- | --- |
| v4 | prod-v4 | 1 480 | Rücksprung prod-v4b → prod-v4 |
| v4b | prod-v4b | 1 280 | Rücksprung prod-v5 → prod-v4b |
| dsv4-sched | dsv4-sched-v2 | 1 998 | DSv4-Produktion 25.–27.10., Tag `verified-2026-10-25-dsv4-sched` |
| glm53-prod-sched | glm53-prod-sched | 1 298 | GLM-Produktion 25.–27.10., Tag `verified-2026-10-25-glm53-sched` |
| q27-meta | q27-gdn-meta | 1 998 | 27B-Produktion 25.–27.10., Tag `verified-2026-10-25-q27-meta` |

Mess-, Diagnose- und Doppelbäume:

| Baum | Zweig | MiB | Bemerkung |
| --- | --- | --- | --- |
| dsv4-scorer-dev | dsv4-decode-fused-scorer | 1 330 | Messzweig des Scorers, aufgegangen in P4 |
| fork-main | tuning-freeze-native | 2 967 | alte Fork-Bauarbeit (fork-main-v2b-Zeit), nichts zeigt mehr darauf |
| fork-mypy | fork-mypy-fixes | 291 | im Fork |
| fp32-scores | fp32-index-scores | 5 011 | Fix-Zweig, = prod-v5d (eigener Baum v5d); leere Eintragsdatei vorher entfernt (0 Byte, echte Fassung in bench-scripts) |
| nf-probe, nf-probe-fix, nf-probe-v5c | nf-probe-* | 1 124 / 1 279 / 1 114 | Diagnose der Index-Scores, nie pushen, Zweige bleiben lokal |
| pass512 | dsv4-prefill-pass512 | 1 048 | = prod-v5c |
| pr-dsv4-fp32 | pr-dsv4-prefill-fp32 | 5 036 | gleicher Kopf wie pr-dsv4-prefill2 (dff18c253) |
| pr-triton-fp32 | pr-triton-sm7x-fp32-scores | 306 | aufgegangen in #1282 (bf1dbb215) |
| pr-ple-disk-tier | pr-ple-disk-tier | 151 | nicht eingereicht (Entscheidung 02.11.), Zweig bleibt |
| sync-probe | sync-probe-2026-11-02 | 152 | Probe, Zweig bleibt |
| ubatch-dev | dsv4-compress-maxnreg | 1 364 | Stack-Fix, im Fork und in #1344 |
| v5-merge | fork-main-v5 | 1 525 | Prüfbaum fork-main-v5 (= prod-v5) |
| v5c-rebuild | v5c-rebuild-2026-11-02 | 4 859 | Neubau-Probe von v5c |

## Gruppe `offene-prs`: 15 Bäume, 17 GiB (Empfehlung: entfernen)

Bäume offener PRs außerhalb der laufenden Runde. Jeder Zweig liegt lokal und im Fork. Für ein Rebase reicht ein frischer Baum ohne Bau, für native Tests ein Bau über ccache.

| Baum | PR | MiB |
| --- | --- | --- |
| pr-drafttokenmap | #1218 | 1 148 |
| pr-dsv4-compress-stack | #1344 | 1 292 |
| pr-dsv4-woa-freeze | #1273 | 1 224 |
| pr-fa2-split-heuristic | #1346 | 293 |
| pr-gdnturing | #1228 | 1 219 |
| pr-hostreduce | #1216 | 1 216 |
| pr-logs | #1293 | 1 298 |
| pr-lookahead2 | #1276 | 1 144 |
| pr-mrv2-dspark | #1221 | 1 288 |
| pr-pleoverlap | #1217 | 1 150 |
| pr-plewindow | #1235 | 1 221 |
| pr-ppsend | #1294 | 1 219 |
| pr-qsapage4main | #1236 | 1 303 |
| pr-turing-kernels | #1278 | 1 222 |
| pr-v100-grouped-verify | #1345 | 1 288 |

## Gruppe `nach-v5d`: 2 Bäume, 2 GiB

v5 und v5a, erst wenn alle Einträge auf v5d laufen und kontrolliert sind (geplant nach dem DeepSeek-A/B).

## Zusammen mit dem Cache-Aufräumen

`cleanup_2026-11-02.sh` (Caches ohne Eintrag und `build/` der PR-Bäume) macht nach den Läufen heute Nacht 38 GiB frei. Drei seiner `build/`-Kandidaten mit 10,5 GiB liegen in Bäumen der Gruppe `alt`.
Beide zusammen ergeben etwa 80 GiB. Heute (21:30) sind 58 G von 916 G frei.

## Befehle für Peuqui

Erst der Trockenlauf je Gruppe, dann das Entfernen:

    bash ~/.cache/bench-scripts/remove_worktrees_2026-11-02.sh alt
    bash ~/.cache/bench-scripts/remove_worktrees_2026-11-02.sh alt delete
    bash ~/.cache/bench-scripts/remove_worktrees_2026-11-02.sh offene-prs
    bash ~/.cache/bench-scripts/remove_worktrees_2026-11-02.sh offene-prs delete
