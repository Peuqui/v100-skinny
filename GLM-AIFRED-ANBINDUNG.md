# GLM-5.3-Flash für AIfred: was beim Anbinden zu wissen ist (Stand 25.10.2026)

Modell in llama-swap (Port 11435, OpenAI-kompatibel): **`GLM-5.3-Flash-320B-A18B-NVFP4-DFlash2-vllm`** (Gruppe `main`, tauscht sich mit den
anderen großen Modellen aus). Eintrag: `~/.config/llama-swap/config.yaml`, Vorlage `~/.cache/bench-scripts/glm53_prod_entry_2026-10-25.yaml`.

## Eckdaten (gemessen, siehe STAND.md „25.10. 02:30“)
- Kontext **524 288 Token** (`max_model_len`), der KV-Pool fasst 645 277 Token (seit der Umstellung auf Fünf-Bit-Skalen 25.10. 11:01; vorher
  667 683); ein Prefix-Cache-Block sind 9 216 Token.
- Tempo (Kontrolllauf 25.10. 17:40): ≈ 37–41 tok/s bei Kurzprompts mit wenig vorhersagbarem Text, ≈ 60–61 tok/s beim Rechnen/Denken (GSM8K);
  Schrittzeit je Runde 85,5 ms bei 31k, ≈ 91 ms bei 283k und 96,7 ms bei 516k Kontext (vor dem Scheduler-Fix 86 / 100 / 114,5 ms; tok/s hängt
  im Langkontext von der Annahme je Lauf ab und schwankt, die Schrittzeit ist das stabile Maß); Prefill ≈ 2 000 tok/s (8,7k Token in 4,3 s;
  283k Token in 134 s; 516k Token in 251 s).
- Prefix-Cache wirkt: derselbe 62k-Verlauf noch einmal → erstes Token nach 4,6 s statt 29,4 s; nächste Chat-Runde mit gleichem Verlauf 4,0 s.
  (Gecacht werden volle 9 216-Token-Blöcke, der Rest wird neu gerechnet.)
- Genauigkeit: GSM8K-30 29/30, wie ohne Entwurf und wie vor der Umstellung; Qualitätsfragen von Hand gelesen in Ordnung, die acht Texte sind
  byteweise gleich zum Stand davor.

## Verhalten, auf das der Client Rücksicht nehmen muss
- **Nur ein Sitz** (`--max-num-seqs 1`): gleichzeitige Anfragen werden nacheinander bedient (die zweite wartet, scheitert nicht). Helfer-Aufrufe
  (Titel, Zusammenfassung) während einer langen Erzeugung warten also mit.
- **Starten dauert 7–12 min** (Gewichte von der Platte, abhängig vom Swap); der Server meldet sich erst danach (llama-swap `healthCheckTimeout`
  2400 s). Timeouts des Clients beim ersten Aufruf nach einem Modellwechsel entsprechend hoch setzen. Der Wechsel zu einem anderen großen Modell
  entlädt GLM.
- **Denken lässt sich nicht abschalten**: das GLM-Template hängt `<think>` immer an. Der Server trennt das Denken seit dem Fix vom 25.10.
  (`--default-chat-template-kwargs '{"enable_thinking":true}'`) in das Feld **`reasoning`** (Antwort in `content`); in der Antwort heißt das
  Feld `reasoning`, nicht `reasoning_content`. `enable_thinking: false` im Request schaltet NICHT das Denken ab, nur die Trennung.
  `reasoning_effort` low/high/max wird vom Template gelesen (Standard max).
- Werkzeugaufrufe laufen über `--tool-call-parser glm47` (geprüft: `get_weather` mit Argumenten); Antwortformat OpenAI `tool_calls`.
- Ein Abbruch durch den Client mitten in der Erzeugung ist unkritisch (Server antwortet danach sofort wieder).
- Maximale Länge nicht über ≈ 641k anheben (Top-k-Kern der RTX-Stufen, siehe `docs/design/sm70_glm53_pp5_dflash2_mixed.md`).

## Betrieb
- Karten: 3× V100 + 2× RTX 8000 komplett belegt (Spitzen `memory.used` 48,1 / 30,5 / 47,9 / 30,5 / 30,5 GB). Vor dem Boot die Karten über GPU-Orc
  (0.2.0, `gpu-orc@mp`, seit 25.10. etwa 00:56 mit den neuen STT-/TTS-Suiten kompatibel) freigeben lassen:
  `curl -X POST http://localhost:5070/free -H 'Content-Type: application/json' -d '{"model":"GLM-5.3-Flash-320B-A18B-NVFP4-DFlash2-vllm","reason":"…"}'`
  (läuft das Modell schon: `already_running`; Vorschau ohne Wirkung mit `"dry_run":true`; Belegung: `GET /status`).
- Baum/Version: `~/Projekte/vllm-research/1Cat-vLLM-glm53-prod-sched` (Zweig `glm53-prod-sched`, Tag `verified-2026-10-25-glm53-sched`, 31d73aefa, seit 25.10. 17:00;
  Spitzen `memory.used` des Kontrolllaufs 48,2 / 30,5 / 47,7 / 30,5 / 30,4 GB). Vorgänger `1Cat-vLLM-glm53-prod-meta` (2ae98d460), `1Cat-vLLM-glm53-prod-s5`
  (5fef3ad60) und `1Cat-vLLM-glm53-prod` (31f961300) bleiben als Rücksprung.
- Auto-Reboot des Mini montags/mittwochs/freitags 04:30 beendet das Modell; llama-swap lädt es bei der nächsten Anfrage neu.

## Was in AIfred dafür nachzuziehen ist (nur gelesen, nichts geändert)
- Feldname des Denkens: der Server liefert `reasoning`. Auf AIfred-Seite ist dafür nichts nachzuziehen (Auskunft `Mini:AIfred-Intelligence`
  25.10. 05:33, am Code geprüft): `aifred/lib/calibration/verifier.py` ist der Verifier für llama-server, dort ist `reasoning_content` richtig;
  den vLLM-Pfad bedienen `aifred/lib/calibration/vllm_probe.py` (Zeile 446 liest `reasoning`) und das Chat-Backend `aifred/backends/vllm.py`
  (`REASONING_FIELD = "reasoning"`).
- Kalibrierung: Kontext 524 288 Token, 1 Sitz, Boot 7–12 min; der Server-Speicher ist mit `--gpu-memory-utilization 0.98` fest verplant
  (keine freie Reserve für parallele Modelle auf den fünf Karten).
