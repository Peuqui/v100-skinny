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

- Qualität: 6 von 8 Texten tokengleich zur Produktionskontrolle k = 4 (korrigiert 25.11.; vorher stand hier 7, das Log nennt auch
  „motor“: ab Zeichen 1065 anders formuliert, 782 statt 791 Token, gelesen: gleiche vier Takte, gleiche Aussagen, nur Wortwahl wie
  „öffnet sich“, „voller Zyklus“); „kuanda“ weicht ab Zeichen 3874 von 4525 ab (1134 statt 1137 Token),
  händisch gelesen: gleiche Aussage (Name existiert nicht, Kenneth Kaunda als naheliegende Person, Zurückweisen = bestanden), andere
  Formulierung zweier Aufzählungspunkte (Namibia statt Angola, beides sachlich richtig). Ursache: andere Prüflängen = andere Batch-Formen.
- Greedy-Sonde −3,1 %: der Regler kürzt bei greedy die hinteren Positionen (Annahme je Position 0,73 0,50 0,28 0,16 0,08 0,05 gegen
  0,72 0,50 0,34 0,24); AIfred nutzt Sampling (T 1,0, min_p 0,1), dort +6,9 %. Decode kurz und GSM8K-Decode (beide greedy im Client)
  sind dagegen +3,9 / +5,2 %.
- Langkontext: Einzelmessungen; die P-Arme streuen am gleichen Prompt 34–54 tok/s (k = 4-Stufe-2), 150k −3,7 % liegt darin.

Urteil: Erfolgskriterium aus DESIGN.md erfüllt (AIfred-Sampling ≥ +3 %: +6,9 %; Prosa und Rechnen nicht schlechter: +5,1 / +5,2 %;
langer Kontext im Rauschen). Einbau in die Produktion mit dem nächsten Bündel (v15), nicht als Einzelbaum (keine Versionsmischung).

## Nachtrag 24.11.2026: variable Prüflänge für DeepSeek-V4 (DSpark) lohnt nicht

DSpark ist ein Blockentwurf (erbt von DFlash, ein Durchlauf für alle k Positionen, danach Markov-Schleife), der Regler wäre mit wenigen
Zeilen zu öffnen; 1Cats DSpark-Konfidenzschnitt läuft nur im V1-Sync-Pfad und war am 11.10. verworfen. Messung ohne Code-Änderung
(`chain_dscal_2026-11-24.sh`, `spec_length_probe` je zweimal, Logs `~/.cache/bench-scripts/dscal_2026-11-24/`): Produktion k = 5 Runde
50,4 ms (AIfred-Sampling), Coding-K7 55,3 ms → c = 2,45 ms je Position, t0 = 38,2 ms, ρ = 0,064. Annahme je Position (kumulativ) unter
AIfred-Sampling, K7: alle 0,76 0,54 0,36 0,24 0,14 0,08 0,04; Prosa 0,73 0,52 0,33 0,22 0,12 0,07 0,03; Rechnen 0,84 0,60 0,52 0,38 0,27
0,16 0,10. Beste feste Länge je Textsorte (Obergrenze für einen Regler, tok/s aus E(n)/t(n)) gegen n = 5: AIfred alle +0,3 %, Prosa +0,8 %,
Erklärung +0,9 %, Code +0,3 %, Rechnen 0; greedy Prosa +2,0 %, Rechnen +1,5 %, Code (n = 7, = K7-Eintrag) +2,3 %. Der Gewinn bliebe unter
1 % für AIfred; nicht gebaut. (Die Schätzung von +4–7 % ging von 1,2 angenommenen Token je Runde aus; gemessen sind 1,93 bei k = 5.)
