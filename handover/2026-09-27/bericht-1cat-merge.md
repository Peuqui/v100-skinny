# 1Cat-Merge 26./27.09. — Bericht für Peuqui

## Ergebnis in einem Satz

Der Fork ist mit 1Cat main@1e90d17f zusammengeführt, gebaut und gegen die
Produktion abgenommen: Ausgaben bei allen drei Modellen bitgleich, Tempo
gleich — bei Flash-Next nur, wenn die neue #691-Voreinstellung abgeschaltet ist.

## Was du entscheiden musst

1. **Merge committen** (Worktree `vllm-research/1Cat-vLLM-merge`, Branch
   `merge-1cat-main-2026-09-26`, alles gestaged, MERGE_HEAD = 1e90d17f).
2. **Produktion umstellen** wie am 22.09.: Fork-Branch per Fast-Forward
   nachziehen, Bauartefakte aus dem Merge-Worktree übernehmen, vorher Sicherung
   als tar. Dazu im Flash-Next-Produktionseintrag vier Zeilen ergänzen:
   `VLLM_SM70_BATCH_GEMM_LAYOUTS=0`, `VLLM_SM70_AWQ_WARMUP_MAX_M=16`,
   `VLLM_SM70_FP8_DENSE_TUNE_MAX_M=16`, `VLLM_SM70_NVFP4_DENSE_TUNE_MAX_M=16`.
3. **PR-Force-Pushes** #604, #611, #646, #667 (alle lokal fertig, siehe unten)
   und die neuen Texte für #667 und #674 freigeben.
4. **Optional:** 1Cat auf #691 hinweisen (Voreinstellung bremst gemischte
   Volta/Turing-Rigs mit PP).

## Abnahme (alt = Produktion 28ff9252, neu = Merge)

| Modell | Greedy | Tempo |
|---|---|---|
| Flash-Next PP4 | bitgleich | gleich MIT #691 aus; mit #691 an Prosa-Decode −20..27 % |
| DeepSeek-V4 PP5 | bitgleich | gleich (18k-Prefill 8,6–8,7 s, 90 ms/Schritt) |
| Qwen3.8-27B TP2 | bitgleich | gleich (27,5 s Prefill 14,7k, 38–46 ms/Schritt) |

Tests auf dem Merge-Bau: V100 760 bestanden, RTX 743 bestanden (Rest übersprungen).
Logs: `~/.cache/bench-scripts/` (bench_*_2026-09-26*.log, regression_merge2_*.log).

## Was beim Merge schiefging und behoben ist

- **DSv4-Deadlock:** git hatte `_pp_broadcast_draft_token_ids()` in
  gpu_model_runner.py ohne Konfliktmeldung GEDOPPELT. Die letzte PP-Stufe sendete
  zweimal, die anderen empfingen einmal -> Hänger beim ersten Request. Entfernt,
  Dopplungsprüfung über alle 68 beidseitig geänderten Dateien: sauber.
- Zwei doppelte Funktionsdefinitionen (mypy), alte ruff/typos-Befunde im
  modelopt-Overlay, drei PLE-Tests, die main's neue Shard-Ansichten nicht kannten
  (Segfault durch Seitenfreigabe auf anonymem Speicher im Test).

## #667 lohnt sich weiter

DSv4, 23k-Präfix nach einer fremden 30k-Anfrage: main allein 10,6 s (nichts aus
dem Cache), mit #667 0,7 s (22.784 Token aus dem Cache). main hat inzwischen die
erste Hälfte (prepend_n, „ungecacht zuerst"), der Rest bleibt nötig.

## PR-Branches (lokal, nichts gepusht)

| PR | Branch / Worktree | Stand |
|---|---|---|
| #604 | sm70-ops-on-turing (1Cat-vLLM-pr-turing-ops-rebase) | a1d8be72 auf 1e90d17f, 77 Tests grün |
| #611 | sm70-qpn2-block-pack-pr (1Cat-vLLM-pr-blockpack) | 64ad1de8 auf 1e90d17f, Kernel-Test grün |
| #646 | ple-cascade-pr-on-main (1Cat-vLLM-pr-plecascade) | Squash der Serie, GESTAGED, braucht Commit-Message; 13/15 Dateien = Fork |
| #667 | prefix-cache-reuse-dead-window-blocks (1Cat-vLLM-pr-evict) | 6048d14d, alte Message; neue im Entwurf |
| #621, #623 | unverändert | mergen sauber |

Entwürfe: `upstream-contrib/03-1cat-issues/pr-667-rebase-2026-09-26.md`,
`issue-674-overview-body-2026-09-27.md` (erst NACH den Force-Pushes posten).

## Eine Sache von mir, die du wissen solltest

#674 habe ich am Nachmittag schon einmal editiert, bevor die Rebases standen und
ohne dir den Text zu zeigen. Er ist inhaltlich richtig, aber veraltet (nennt
#603/#658 als offen). Der neue Entwurf ersetzt ihn.

## Offen nebenbei

- vllm#58016 braucht einen Rebase (mergify-Konflikt seit 25.09.).
- aifred-intelligence.service stand gestern Abend auf „failed" — beim Start
  kurz ins Log schauen.
