# Übergabe — Stand 24.09.2026 morgens (nach dem Rückbau)

Produktion läuft mit PP4, `HOST_GIB=0` und dem PLE-Rest auf der Platte. Die
Store-Stufe ist zurückgebaut und archiviert. Fork `83324b0e` = work-main, Tag
`verified-2026-09-24-rollback`, alles gepusht.

## Womit anfangen

`STAND.md` Punkte 44 (Archiv) und 45 (Messung, Rückbau, Abnahme). Punkte 40–43
nur als Hintergrund. Nicht mit den Logbüchern anfangen.

## Offen, Entscheidung Peuqui

1. **PR #646 aktualisieren**: Entwurf
   `upstream-contrib/03-1cat-issues/pr-646-update-2026-09-24.md`, Code im
   PR-Worktree `1Cat-vLLM-pr-plecascade` auf dem lokalen Branch
   `ple-disk-only` (PR-Stand + Kopier-Korrektur + Rückbau, Signed-off-by,
   Tests grün, nicht gepusht). Vor dem Push: `pre-commit` fehlt in der venv
   (AGENTS.md-Pflicht, Installation absprechen). Frage im Entwurf: reicht PP4
   als End-to-End-Beleg?
2. **Untergrenze für freien Host-RAM** (STAND 42): durch Host 0 und die
   Seitenfreigabe kaum noch dringend (18 GiB frei). Vermutlich verwerfen.

## Weiter offen (ohne Eile)

- Produktion an einem anderen Tag nachmessen (Punkt 38).
- Messwerkzeuge: `~/.cache/bench-scripts/pagecache.py` (Seitencache
  messen/räumen ohne root), `cold_probe.py` (echte Texte, Major-Faults, Swap,
  RssFile je Anfrage); Rohdaten in `~/.cache/bench-scripts/logs-2026-09-24/`.

## Dauerhafte Regeln

- **Vor jedem Rückbau Archiv-Branch + Tag pushen** (Punkt 44).
- **Speichertests brauchen eine Mutationsprobe.**
- **Tests dürfen Invarianten nicht vortäuschen** (Stub + `MADV_DONTNEED` =
  Absturz).
- **KV-Budget ist kein A/B zwischen Einzelboots** (Punkt 43).
- **Echte Texte für PLE-Messungen**; derselbe Text zweimal misst den
  Präfix-Cache.
- **Lange Messungen als `systemd-run --user`-Unit** (`XDG_RUNTIME_DIR`,
  `DBUS_SESSION_BUS_ADDRESS` setzen).
