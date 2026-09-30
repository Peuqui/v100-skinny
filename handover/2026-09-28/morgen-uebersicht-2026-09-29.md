# Morgen-Übersicht 29.09.

## Stand Abend 28.09.
- Produktion auf Fork f398d4c6 (Tag verified-2026-09-28-pass10, gepusht). acc7: DSv4, Flash-Next, 27B je 8/8
  Qualitätsantworten zeichengleich zu acc6 (händisch gelesen), Tempo 90–94 / 66–69 / 39–41+48 ms.
- Aufräumen abgeschlossen: P12 zurück auf main (Boot ohne P12 belegt), QPN8_BLK raus, SWA-Puffer,
  Sampler-Backport, Test-Korrekturen, #614-Fix. Fork-Suite (eigener Prozess je Datei): V100 1206, RTX 1121 grün.
- Worktrees 46 → 22; pr-turing-ops als archive/pr-turing-ops-wip-2026-09-13 gesichert und entfernt.

## Bei 1Cat (17 PRs offen, alle konfliktfrei auf 357d07bc)
#604 #611 #621 #623 #646 #667 #710 #711 #714 #715 #716 #717 #720 #723 #725 #726 #727; Issue #718
(Skinny-Kernel, Antwort an valentijnvenus gepostet). Kommentare: #614 (Bestätigung), #623 (Rebase).

## System
- llama-swap MemoryHigh: 16G richtig, 12G zu eng (DSv4-Boot 528 statt 348 s, vLLM swappt selbst).
  **Steht wieder auf 16G** (29.09. nachts gesetzt).
- GPUDirect Storage geht nicht (Modelle auf USB-SSD). O_DIRECT-Loader wie llama.cpp nur, falls 16G nicht reicht.
- AI-Connect: Ping-Pong-Fix, Wächter (integrations/claude-code/aiconnect_watch.py), Doku — alles gepusht.

## Offen / wartet
- Skinny-Paket + DSv4 auf Turing: wartet auf 1Cats Antwort zu #718.
- 9 der 13 V100-test_config-Fehler: 1Cats Designentscheidung (SM70-Politik überschreibt ausdrückliche Werte), in #727 benannt.
