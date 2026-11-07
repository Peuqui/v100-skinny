# Auftrag: unsere Verbesserungen vom 06./07.11.2026 als 1Cat-PRs vorbereiten (Peuqui 07.11.: „einfach hochladen“)

Unser Fork: `fork-main-v6` (Baum `/home/mp/Projekte/vllm-research/1Cat-vLLM-main-v6`, 31d5473d0), Remote `fork` = Peuqui/1Cat-vLLM,
`origin` = 1CatAI/1Cat-vLLM (main = c98a8d664). Regeln aus `~/.claude/CLAUDE.md` und `/home/mp/Projekte/vllm-research/CLAUDE.md` gelten,
besonders Außenwirkung: jede Zahl aus einer Datei oder einem Testlauf, Texte nach außen Englisch, Git-Autor
`Peuqui <43776522+Peuqui@users.noreply.github.com>`, Trailer `Signed-off-by`, `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`,
`Claude-Session: https://claude.ai/code/session_0172gtUn2th65t88H9KvYkwK`. AGENTS.md im Baum (Duplikatsuche, KI-Hinweis, Testbefehle mit
Ergebnissen). PR-Vorlage `.github/PULL_REQUEST_TEMPLATE.md`; Beispiel unserer Form:
`v100-skinny/upstream-contrib/21-1cat-backport-prs-2026-11-07/glm-kpool/body.md` (Purpose, Not a duplicate, AI assistance, Test Plan,
Path coverage, Acceleration contract, Test Result, zwei Schlusszeilen `🤖 Generated with [Claude Code](https://claude.com/claude-code)` und
`https://claude.ai/code/session_0172gtUn2th65t88H9KvYkwK`). Commit-Nachrichten und Texte neutral (kein „our production“, keine AIfred-,
llama-swap- oder Eintragsnamen); Messergebnisse unseres Forks sind als solche zu kennzeichnen („on our fork (github.com/Peuqui/1Cat-vLLM …)“).

Messbelege (Logs unter `~/.cache/bench-scripts/`, Zusammenfassung in `v100-skinny/STAND.md` Nachträge 06.11. und 07.11.):
- Prefill-Stückgröße nach Promptlänge (a8244aad8): `ab_adachunk_glm_2026-11-06a.log`, `ab_adachunk_dsv4_2026-11-06{a,b}.log`
  (GLM lange Prompts −14 %, DeepSeek kurze 1k/2k/4k −24/−15/−6 %, sonst gleich; Preis KV-Pool/GPU-Spitze siehe STAND).
- Entwurfs-Configs teilen Laufzeitressourcen (e7e4c7f82): STAND Nachtrag 06.11. 20:10, `ab_draft8b_glm_2026-11-06b.log`.
- Feinere Decode-Buckets: `ab_fbuckets_dsv4_2026-11-07a.log` (+ `…-fortsetzung.log`), `ab_fbuckets_k7_2026-11-07a.log` (+ `…-fortsetzung{,2}.log`),
  Profile `~/.cache/nsys-runs/dsv4-v6f-*-1107`; STAND Nachtrag 07.11. 07:00 und 14:50.
- Decode-Kachel (31d5473d0): Commit-Nachricht, `tile_sweep_*`, `tile_check_*`, `tile_gpu_checks_*`, `ab_tile_dsv4_2026-11-07a.log` (+ Fortsetzung).
- mHC-Tensorkern-Weg (0c5cbfe9b): `ab_mhctc_glm_2026-11-06a.log`, STAND Nachtrag 06.11. 11:55.

Je PR: frischer Baum `git -C …/1Cat-vLLM-main-v6 worktree add -b <zweig> ../1Cat-vLLM-<zweig> origin/main` (vorher `git fetch -q origin main`),
Commits per cherry-pick, Konflikte im Sinn des Fixes auf 1Cats Code, nichts Fork-Fremdes mitnehmen; `git diff origin/main` vollständig
lesen; CPU-Tests `CUDA_VISIBLE_DEVICES="" ~/.cache/bench-scripts/pytest_1cat.sh <datei> -q -p no:cacheprovider`; pre-commit
(`/home/mp/.venv/precommit/bin/pre-commit run --from-ref origin/main --to-ref HEAD`, dazu `mypy-3.10 --hook-stage manual`; check-layering
meldet auf main fused_moe.py und moe_runner.py, nur Neues melden). GPU-Tests NICHT ausführen (die Karten laufen für Messungen), sondern als
Befehlsliste in den Bericht; im Text `[FILL GPU]`. Duplikatsuche bei 1Cat (`gh pr list --repo 1CatAI/1Cat-vLLM --state all --search …`).
Texte unter `v100-skinny/upstream-contrib/22-1cat-prs-2026-11-07/<name>/{title.txt,body.md}` bzw. für PR-Ergänzungen `<name>/comment.md`.
NICHT pushen, keine PRs eröffnen, keine Kommentare posten, kein `rm`, kein Force-Push, keine Pakete installieren.
