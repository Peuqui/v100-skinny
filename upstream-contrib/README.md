# Upstream-Beiträge

Entwürfe und Diffs liegen in den Unterordnern. Regeln für neue Beiträge:
`AGENTS.md` im 1Cat-Repo (Duplikatsprüfung, Testkommandos samt Ergebnis im
PR-Text, KI-Einsatz deklarieren) — Verstoß kann eine Sperre nach sich ziehen.
Jede Behauptung vor dem Senden gegen frisch gefetchtes Upstream-main prüfen
(Lehre aus #455, das nur auf dem eigenen Fork verifiziert war und
zurückgezogen wurde). **Nichts senden ohne Freigabe von Peuqui.**

## 1Cat-vLLM: wo der Stand steht

Der laufende Stand wird hier nicht als Liste geführt (die alte Liste war
binnen Tagen veraltet), sondern an zwei Stellen, die nach jeder Merge-Welle
stimmen:

- GitHub selbst: `gh pr list --repo 1CatAI/1Cat-vLLM --author Peuqui --state all`
- die öffentliche Übersicht [1Cat #674](https://github.com/1CatAI/1Cat-vLLM/issues/674)
  (offene PRs mit Abhängigkeiten, gemergte PRs); gesendete Fassungen und
  Entwürfe als `03-1cat-issues/issue-674-*` und `overview-674-*`.

**Momentaufnahme 2026-10-02 abends** (per API gezählt): 35 PRs gemergt.
Fünf weitere hat 1Cat in eigene PRs übernommen; GitHub führt sie deshalb als
„closed“: #727 → #733, #725 → #757 (mit #733), #741 → #758, #752 → #762,
#667 → #765. #601 haben wir selbst geschlossen (main deckt es ab), #455
zurückgezogen. 20 offen, alle mergen sauber auf main `b5f36b66`
(`git merge-tree`); davon stecken vier in 1Cats offenen Integrations-PRs:
#743 → #767, #710 → #768, #726 → #770, #749 → #771.

1Cats Integrationen laufen über Branches `codex/pr-<thema>-<datum>` und ein
Merge-Gate nur auf der CPU (`tools/merge_gate.sh`: keine GPU, keine
Modellgewichte). GPU-Nachweise auf seinem exakten Branch sind deshalb das,
was wir beitragen können.

Übernommene Commits tragen „Peuqui <peuqui@github.com>“ als Autor oder
Co-Autor. Diese Adresse gehört zu keinem GitHub-Konto, die Commits erscheinen
deshalb nicht im Profil. Seit 02.10. ist die Commit-Adresse
`43776522+Peuqui@users.noreply.github.com`.

Sobald 1Cat einen PR von uns merged oder übernimmt, fällt unsere Fassung beim
nächsten Hereinholen von main aus dem Fork heraus; nach dem Merge prüfen, dass
keine Doppelung bleibt („konfliktfrei“ heißt nicht „sauber“, Methode in
`OVERLAY-INVENTUR.md`).

---

## Erste Runde — alle veröffentlicht am 2026-08-28

ALLE VERÖFFENTLICHT am 2026-08-28 (Freigabe Peuqui):
- PR:  https://github.com/dnv2003/v100-skinny/pull/7
- vLLM: https://github.com/vllm-project/vllm/issues/54260 (Bug in main
  bestätigt, Zeile 1425, Stand 28.08.)
- 1Cat: https://github.com/1CatAI/1Cat-vLLM/issues/412 (Device-0),
  /413 (E5×QSA), /414 (MTP-Profiling PP)
- HF:  https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4/discussions/6
- Werkzeug-Repo: https://github.com/Peuqui/mtp-quant-transplant

| # | Ziel | Art | Inhalt | Voraussetzung |
|---|------|-----|--------|---------------|
| 1 | vllm-project/vllm | Issue (+PR-Angebot) | PP+async+spec: Output-Trim läuft nur auf letzter Stufe (elif→if) | gegen aktuellen main verifizieren |
| 2 | dnv2003/v100-skinny | PR | Branch pp-mtp-merge (PP×TP+MTP, sm75-Paket, Device-0-Fixes, PLE-Kaskade) | GitHub-Fork unter Peuquis Account, Branch pushen |
| 3 | 1CatAI/1Cat-vLLM | Issue 1 | Capability-Gates fragen Device 0 statt aller sichtbaren GPUs | — |
| 4 | 1CatAI/1Cat-vLLM | Issue 2 | E5-Metadaten-Cache crasht an CSA/QSA-Modellen (shape [] vs [1]) | — |
| 5 | 1CatAI/1Cat-vLLM | Issue 3 | MTP-Profiling-Report bei PP blind (is_global_first_rank-Gate) | — |
| 6 | HF RadixArk/Qwen3.8-Flash-Next-NVFP4 | Discussion | Unquantisierter MTP-Block = Spekulation wird Verlustgeschäft auf Pre-Hopper | — |

Messgrundlage: docs/journal/QWEN4EXP-PORT-HANDOVER.md (Abschnitte 28.08.) und
docs/journal/MERGE-PROJECT-HANDOVER.md. Hardware: 2x Quadro RTX 8000 (sm75) + 3x
Tesla V100 (sm70), TP=2/PP=2, 1Cat-vLLM 1.3.0 + v100-skinny-Patches.
