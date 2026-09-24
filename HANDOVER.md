# Übergabe — Stand 24.09.2026 mittags

Produktion: Flash-Next PP4, `HOST_GIB=0`, PLE-Rest auf der Platte,
`VLLM_PLE_DISK_RELEASE_PAGES=1` (in allen acht Flash-Next-Einträgen von
llama-swap). Fork `03b8cb99` = work-main, Tag `verified-2026-09-24-release`.
Alles committet und gepusht (Fork, v100-skinny, AIfred).

## Womit anfangen

`STAND.md` Punkte 44–47 (Archiv, Rückbau + Messungen, FP8-Drafter,
Freigabe-Schalter). Punkte 40–43 nur als Hintergrund.

## Wartet auf andere

1. **PR #646** am 24.09. aktualisiert (Branch a0f93cf7: VRAM → Host → Platte,
   Kopier-Korrektur, Schalter `VLLM_PLE_DISK_RELEASE_PAGES`). Hängt an
   #622/#640/#639; noch keine Reaktion der Maintainer. Lokaler Branch
   `ple-disk-only` im Worktree `1Cat-vLLM-pr-plecascade`.
2. **1Cat-Draft #684** (yangzhuxinyzx): gleiche Dateien, Schnellweg für kurze
   Plattenlesevorgänge. Wenn gemergt: #646 darauf rebasen, Schnellweg in
   `_gather_mapped_rows` ziehen, im Fork messen (STAND 47, Probe-Merge:
   ein Konflikt in `_disk_embedding_lookup`).
3. **Issue #679** (chenmacnica, MiMo): geschlossen, MiMo läuft bei ihm über
   unsere Skinny-Kernel. Er bietet sich als Tester für einen MXFP4-PR an;
   Maintainer-Entscheidung zur PR-Form (Variante 2 oder 3) steht aus.

## Entschieden, nicht neu aufrollen

- Store-Stufe/Kartenliste zurückgebaut (Archiv-Branch, STAND 44).
- Untergrenze für freien Host-RAM verworfen.
- `HOST_GIB` bleibt (bei 1Cat gemergt, nützt Systemen mit viel RAM).
- TP4 bei uns verworfen (STAND 30); FP8-Drafter auf RTX nicht bauen (STAND 46).

## Dauerhafte Regeln

- Vor jedem Rückbau Archiv-Branch + Tag pushen.
- Speichertests brauchen eine Mutationsprobe; Tests dürfen Invarianten nicht
  vortäuschen.
- KV-Budget ist kein A/B zwischen Einzelboots.
- Echte Texte für PLE-Messungen; lange Messungen als `systemd-run --user`.
- pre-commit liegt in `~/.venv/precommit/`.
