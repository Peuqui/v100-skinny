# Auftrag: Backport-PRs an 1CatAI/1Cat-vLLM vorbereiten (07.11.2026)

Unser Fork (`fork-main-v6`, Baum `/home/mp/Projekte/vllm-research/1Cat-vLLM-main-v6`) enthält 31 Upstream-vLLM-Backports, abgenommen
und seit 07.11. 02:23 in unserer Produktion (Baum `1Cat-vLLM-v6f`). 1Cat zieht upstream nicht nach, Upstream-Fixes kommen nur als
Backport-PRs (Beispiele im Titelstil: 1Cat #657 „[Bugfix][Core] Backport vllm-project/vllm#44082: …“, #1406 von heute). Jetzt sollen
sie als sechs Bereichs-PRs auf das aktuelle 1Cat main (origin/main = c98a8d664). Zuordnung der Fork-Commits je PR:
`/home/mp/Projekte/vllm-research/v100-skinny/upstream-contrib/21-1cat-backport-prs-2026-11-07/groups.json`. Hintergrund je Fix:
`v100-skinny/upstream-contrib/20-upstream-backport-scan-2026-11-06/` (PLAN.md, review_*.md) und die Commit-Nachrichten im Fork.

Regeln aus `~/.claude/CLAUDE.md` und `/home/mp/Projekte/vllm-research/CLAUDE.md` gelten (bes. Außenwirkung: penibel, jede Zahl aus
einer Datei oder einem Testlauf, Texte nach außen Englisch). AGENTS.md im Baum (Duplikatsuche, KI-Hinweis, Testbefehle mit
Ergebnissen, keine Fleißarbeit). Die 1Cat-PR-Vorlage steht in `.github/PULL_REQUEST_TEMPLATE.md`; ein fertiges Beispiel unserer Form
(Abschnitte, Tabelle, Vertrag) ist `v100-skinny/upstream-contrib/19-1cat-pr-2026-11-06/body-port-60472.send.md`.

## Je PR
1. Frischen Arbeitsbaum anlegen: `git -C /home/mp/Projekte/vllm-research/1Cat-vLLM-main-v6 worktree add -b pr-bp-<name> ../1Cat-vLLM-pr-bp-<name> origin/main`
   (origin/main ist c98a8d664; vorher `git -C … fetch -q origin main` und prüfen, ob main sich bewegt hat — dann den neuen Stand nehmen und
   im Bericht nennen).
2. Die Fork-Commits der Gruppe in der Reihenfolge aus `groups.json` per `git cherry-pick` übernehmen. Konflikte: Absicht des Fixes auf
   1Cats Code übertragen, 1Cats eigene Änderungen behalten, NICHTS aus unserem Fork mitnehmen, was nicht zum Fix gehört. Danach
   `git diff origin/main --stat` und den vollen Diff lesen: nur die Backport-Änderungen, kein Zurücksetzen von main-Änderungen.
   Commit-Nachrichten behalten (Autor Peuqui, Signed-off-by, die Co-Authored-By-/Claude-Session-Zeilen); wo du beim Konfliktlösen etwas
   anpasst, im Rumpf ergänzen.
3. Duplikatsuche bei 1Cat für jede Upstream-PR-Nummer der Gruppe und Stichworte (`gh pr list --repo 1CatAI/1Cat-vLLM --state all
   --search "<nr>"`, dazu 2–3 Stichworte), Ergebnis in den Text. Gibt es schon einen offenen PR für einen Fix, den Fix aus der Gruppe
   nehmen und berichten.
4. Tests im PR-Baum: alle betroffenen CPU-Tests über `CUDA_VISIBLE_DEVICES="" ~/.cache/bench-scripts/pytest_1cat.sh <datei> -q -p no:cacheprovider`;
   der Baum hat noch keine nativen Bibliotheken — für reine Python-PRs genügt das meist (Import von vllm ohne `_C` geht für die meisten
   CPU-Tests; wo nicht, im Bericht sagen). GPU-Tests NICHT ausführen (Karten koordiniere ich), sondern als Befehlsliste in den Bericht.
   `pre-commit run --from-ref origin/main --to-ref HEAD` (`/home/mp/.venv/precommit/bin/pre-commit`) und `pre-commit run mypy-3.10
   --hook-stage manual --from-ref origin/main --to-ref HEAD`; `check-layering` ist auf main rot (fused_moe.py 25 → 32, moe_runner.py 37 → 39)
   — nur melden, wenn neue Dateien auftauchen.
5. PR-Text nach Vorlage schreiben: `…/21-1cat-backport-prs-2026-11-07/<name>/title.txt` (eine Zeile, Stil „[Bugfix][<Bereich>]
   Backport vllm-project/vllm#A, #B, …: <kurz>“) und `…/<name>/body.md`. Purpose: je Backport ein Absatz (Upstream-PR mit Nummer, Merge-SHA
   9-stellig und Autor; was falsch war; warum es 1Cat-Nutzer auf V100/Turing trifft, mit Datei; Anpassungen gegenüber upstream). Danach
   „Not a duplicate:“ mit der Suche, „AI assistance was used …“, Test Plan, Path coverage (Tabelle), Acceleration contract (bei
   Bugfixes: „not a performance change“, aber mit dem Hinweis, dass das Bündel auf unserem Fork per A/B auf DeepSeek-V4 und GLM-5.3 PP5
   ohne Tempo- oder Textänderung abgenommen ist, Logs `~/.cache/bench-scripts/ab_v6f_{glm,dsv4}_2026-11-07a.log`), Test Result (CPU-Ergebnisse
   echt eintragen, GPU-Teil als `[FILL GPU]`). Am Ende die zwei Zeilen
   `🤖 Generated with [Claude Code](https://claude.com/claude-code)` und `https://claude.ai/code/session_0172gtUn2th65t88H9KvYkwK`.
6. NICHT pushen, keinen PR öffnen, keine Kommentare posten. Das mache ich nach Durchsicht.

## Bericht an mich (höchstens 30 Zeilen je PR)
Baum, Zweig, Commit-SHAs, Konflikte und wie gelöst, Diff-Statistik gegen origin/main, Duplikatsuche, CPU-Tests mit Ergebnis,
pre-commit/mypy, GPU-Befehlsliste, offene Punkte.
