# Variable Prüflänge GLM: Stufe-2-A/B gegen Produktion k = 4 (23.11.2026)

Kette `~/.cache/bench-scripts/chain_avadapt_glm_2026-11-23.sh`, Log `ab_avad_glm_2026-11-23a.log` (Arme T1 P2 T3 P4, 15:17–16:20).
T = Zweig `glm-adaptive-verify` c9edcaa357 (Baum `1Cat-vLLM-adaptive-verify`), Entwurfsblock 6 (Capture 7), Regler n ∈ [1, 6], ρ = 0,119
(KALIBRIERUNG.md); P = Produktion v14 k = 4. Journal T: „Adaptive verify length (throughput) enabled: n in [1, 6], first round 4“.
Arm T1 zählt (Caches der Kalibrierung). Während der ersten Minute des T1-Ladens lief versehentlich ein ccache-Bau (gestoppt); die Bootzeit
von T1 (299 s gegen 275–279 s) zählt deshalb nicht, die Messungen danach sind nicht betroffen.

| Größe (mehr ist besser, außer ms) | T1 | T3 | P2 | P4 | T gegen P |
|---|---|---|---|---|---|
| AIfred-Sampling „alle“ tok/s | 43,96 | 43,90 | 41,00 | 41,12 | +6,9 % |
| Prosa / Erklärung tok/s | 36,20 / 43,96 | 36,08 / 43,90 | 34,38 / 41,00 | 34,44 / 41,12 | +5,1 % / +6,9 % |
| Code / Rechnen tok/s | 58,85 / 61,41 | 58,75 / 61,23 | 58,42 / 58,28 | 58,56 / 58,34 | +0,6 % / +5,2 % |
| Runde AIfred „alle“ ms (weniger ist besser) | 50,3 | 50,5 | 56,0 | 55,8 | −5,5 ms |
| Greedy-Sonde „alle“ tok/s | 49,12 | 48,94 | 50,58 | 50,67 | −3,1 % |
| Decode kurz tok/s (Client) | 49,20 | 49,13 | 47,24 | 47,38 | +3,9 % |
| Decode lang nach 8687 Token tok/s | 66,04 | 63,51 | 61,44 | 61,30 | +5,6 % |
| GSM8K-Decode tok/s / richtig | 67,85 / 29 | 67,74 / 29 | 64,43 / 29 | 64,46 / 29 | +5,2 % / gleich |
| Langkontext 150k / 250k / 516k tok/s (je eine Messung) | – | 57,8 / 55,4 / 56,0 | 60,0 / 55,8 / 41,4 | – | −3,7 % / −0,7 % / +35 % |
| Langkontext ms je Schritt (weniger ist besser) | – | 68,4 / 70,5 / 74,2 | 65,8 / 66,8 / 68,0 | – | Schritte länger, mehr Token je Schritt |
| GPU-Spitze MiB Karte 0…4 | 47375 30072 46067 31100 30440 | 47905 30460 46635 31100 30500 | 47863 30428 46593 31100 30448 | 47375 30040 46025 31100 30440 | gleich |

- Qualität: 7 von 8 Texten tokengleich zur Produktionskontrolle k = 4; „kuanda“ weicht ab Zeichen 3874 von 4525 ab (1134 statt 1137 Token),
  händisch gelesen: gleiche Aussage (Name existiert nicht, Kenneth Kaunda als naheliegende Person, Zurückweisen = bestanden), andere
  Formulierung zweier Aufzählungspunkte (Namibia statt Angola, beides sachlich richtig). Ursache: andere Prüflängen = andere Batch-Formen.
- Greedy-Sonde −3,1 %: der Regler kürzt bei greedy die hinteren Positionen (Annahme je Position 0,73 0,50 0,28 0,16 0,08 0,05 gegen
  0,72 0,50 0,34 0,24); AIfred nutzt Sampling (T 1,0, min_p 0,1), dort +6,9 %. Decode kurz und GSM8K-Decode (beide greedy im Client)
  sind dagegen +3,9 / +5,2 %.
- Langkontext: Einzelmessungen; die P-Arme streuen am gleichen Prompt 34–54 tok/s (k = 4-Stufe-2), 150k −3,7 % liegt darin.

Urteil: Erfolgskriterium aus DESIGN.md erfüllt (AIfred-Sampling ≥ +3 %: +6,9 %; Prosa und Rechnen nicht schlechter: +5,1 / +5,2 %;
langer Kontext im Rauschen). Einbau in die Produktion mit dem nächsten Bündel (v15), nicht als Einzelbaum (keine Versionsmischung).
