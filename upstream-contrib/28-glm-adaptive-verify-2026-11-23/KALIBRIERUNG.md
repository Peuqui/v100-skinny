# Kalibrierung variable Prüflänge GLM (23.11.2026)

Kette `~/.cache/bench-scripts/chain_avcal_glm_2026-11-23.sh` (Zweig `glm-adaptive-verify` c9edcaa357, Baum `1Cat-vLLM-adaptive-verify`),
Logs `ab_avf{4,6,3,1}_glm_2026-11-23a.log`. Entwurfsblock immer K_max = 6 (Capture-Größe 7, Kürzungsgraphen 7…2 auf allen Stufen,
keine QSA-Kurzkontext- oder Bucket-Varianten im Log), Prüflänge je Boot über `forced_tokens` erzwungen. Vergleich: Produktion v14 k = 4
(`ab_k4_glm_2026-11-23a.log`, Arme T1/T3) und alte Produktion k = 7 (dort P2/P4).

## Test T5 (Pflicht): erzwungen 4 gegen Produktion k = 4

- Greedy-Qualitätstexte gegen `validate_v14k4_glm_2026-10-31/quality.json`: 8 von 8 tokengleich (motor 791, prime 1500, zug 611, kuanda 1137,
  logik 278, code_bug 483, summary 306, english 427 Token). GSM8K 29 richtig.
- Kosten des größeren Entwurfsblocks (6 statt 4): Runde greedy 57,4 gegen 57,1 ms, AIfred-Sampling je Textsorte innerhalb ±0,2 %.
  Der größere Block kostet nichts Messbares.

## Rundenzeit und Annahme (spec_length_probe, je ein Boot)

Runde greedy in ms (weniger ist besser):

| Prüflänge n | 1 | 3 | 4 | 6 | 7 (k = 7, Block 8) |
|---|---|---|---|---|---|
| Runde greedy | 42,8 | 52,4 | 57,4 | 65,7 | 69,9 |

Ausgleichsgerade über n = 1, 3, 4, 6: t(n) = 38,5 ms + 4,60 ms · n, also **ρ = c / t0 = 0,119** (Platzhalter war 0,108).

AIfred-Sampling (T 1,0 / min_p 0,1) in tok/s (mehr ist besser):

| Textsorte | n = 1 | n = 3 | n = 4 | n = 6 | Produktion k = 4 |
|---|---|---|---|---|---|
| Prosa | 36,09 | 35,59 | 34,23 | 29,91 | 34,25 |
| Erklärung | 40,17 | 44,82 | 40,87 | 38,06 | 41,01 |
| Code | 42,55 | 58,00 | 58,20 | 55,58 | 58,29 |
| Rechnen | 43,07 | 58,62 | 57,98 | 59,78 | 58,01 |

Decode lang nach 8687 Token (Client, tok/s): n = 1 41,9, n = 3 52,4, n = 4 61,6, n = 6 55,0.

Annahme je Position bei n = 6 (Erklärung): 0,67 0,39 0,19 0,11 0,07 0,03. Das Reglermodell E(n) / t(n) mit diesen Werten ergibt
39,0 / 42,9 / 41,1 / 37,4 tok/s für n = 1 / 3 / 4 / 6 (gemessen 40,2 / 44,8 / 40,9 / 38,1) und wählt wie die Messung n = 3.

Das „alle“-Feld der Sonde zeigt bei tok/s denselben Wert wie Erklärung (Median über die Textsorten), nicht einen Mittelwert.

## Folgerung

Die beste feste Länge hängt von der Textsorte ab: Prosa n = 1, Erklärung n = 3, Code 3 bis 4, Rechnen 6, langer Kontext 4. Gegen k = 4
wäre mit der jeweils besten Länge Prosa +5 %, Erklärung +9 %, Rechnen +3 %, Code gleich möglich. Nächster Schritt: Adaptiv-Boot mit
ρ = 0,119 als Stufe-2-A/B gegen Produktion k = 4 (`chain_avadapt_glm_2026-11-23.sh`).
