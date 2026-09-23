# Übergabe — Stand 24.09.2026 früh (nach der Nachtschicht)

**Drei Entscheidungen von Peuqui stehen aus, dann ist das Paket rund.**
Produktion läuft (PP4, PLE auf den Pipeline-Karten, Tag `verified-2026-09-24`),
Fork `ddc146b2` = work-main, alles gepusht.

## Womit anfangen

`STAND.md` Punkte 40–43 (Kartenliste, echte Plattenmessung, Mappings
freigegeben, KV kein A/B). Nicht mit den Logbüchern anfangen.

## Offene Entscheidungen

1. **Untergrenze für freien Host-RAM** (STAND 42): Die Startprüfung garantiert
   nur „Rest ≥ 0" nach der Engine-Reserve (die Engine braucht ~7,75 GiB,
   genau die Reserve). Vorschlag: `VLLM_QWEN4EXP_PLE_HOST_MIN_FREE_GIB`
   (z. B. 4 GiB) in der Startprüfung einrechnen und nach dem Capture messen;
   unterschritten → Startabbruch mit empfohlenem `HOST_GIB`. Verhaltensänderung.
2. **PR #646 aktualisieren**: Entwurf
   `upstream-contrib/03-1cat-issues/pr-646-update-2026-09-24.md`, Code im
   PR-Worktree auf Branch `ple-cardlist-try` (141 Tests grün auf zwei V100,
   Signed-off-by gesetzt, nicht gepusht). Fragen dort: später Auslöser in
   `gpu_worker.py` mitnehmen?; Kopier-Korrektur kommt mit (eigener PR
   verworfen, STAND 43). `pre-commit` fehlt in der venv — vor dem Push nötig.
3. **llama-swap aufräumen** (Bulk-Edit an deiner Config, daher nicht nachts):
   `…-k0-test`, `…-prof2-test` (auch in der Gruppe `main`), `…-pp4-test`,
   `…-pp4-disk-test`, `…-pp4-cards-test`, `…-pp4-cards-host3-test`,
   `…-pp4-alldisk-test`, `…-tp2pp2-copyfix-test`. Sicherungen in
   `~/.config/llama-swap/backups/`.

## Weiter offen (ohne Eile)

- Produktion an einem anderen Tag nachmessen (Punkt 38).
- Messwerkzeuge der Nacht: `~/.cache/bench-scripts/pagecache.py`
  (Seitencache messen/räumen ohne root) und `cold_probe.py` (echte Texte,
  Major-Faults, Swap, RssFile je Anfrage); Rohdaten in
  `~/.cache/bench-scripts/logs-2026-09-24/`.

## Dauerhafte Regeln, die heute Nacht teuer waren

- **Speichertests brauchen eine Mutationsprobe.** Zwei neue Tests waren
  zunächst zu klein (640 KB, 16-Byte-Zeilen) und hätten den Fehler nie gesehen.
- **Tests dürfen Invarianten nicht vortäuschen.** Ein 1Cat-Test ersetzte die
  Mapping-Prüfung durch einen Stub; mit `MADV_DONTNEED` stürzte er ab.
- **KV-Budget ist kein A/B zwischen Einzelboots** (Compile-Cache, STAND 43).
- **Echte Texte für PLE-Messungen**, nicht die 22-Wörter-Prompts; derselbe
  Text zweimal misst vLLMs Präfix-Cache, nicht die Platte.
- **Lange Messungen als `systemd-run --user`-Unit** starten
  (`XDG_RUNTIME_DIR`/`DBUS_SESSION_BUS_ADDRESS` setzen) — ein VS-Code-Neustart
  beendet sonst alle Hintergrundskripte.
