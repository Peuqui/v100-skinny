# Gepackte PP-Übergaben: Stufe-2-A/B (23.11.2026)

Zweig `pp-pack-handover` (a43e966b1b + df927238c0 auf `merge-1cat-2d218ab03`), Baum `1Cat-vLLM-pp-pack`, Bibliotheken bytegleich zu v14.
T = Produktion v14 + `kernel_config.pp_pack_handover`, P = Produktion v14. Kette `~/.cache/bench-scripts/chain_ppack_2026-11-23.sh`.

## GLM-5.3-Flash (k = 4), `ab_ppk_glm_2026-11-23a.log`

Arme T1 P2 T3 P4 (Compile-Hash unverändert, T1 zählt). P4 hat die Hauptmessung abgeschlossen; Sonde und GSM8K von P4 fehlen,
weil AIfred um 14:01 das Ende des Holds anforderte.

| Größe | T1 | P2 | T3 | P4 | T gegen P |
|---|---|---|---|---|---|
| Decode kurz tok/s (mehr ist besser) | 47,77 | 47,39 | 47,74 | 47,31 | +0,8 % |
| Runde greedy ms (weniger ist besser) | 56,5 | 56,9 | 56,6 | – | −0,35 ms |
| AIfred-Sampling „alle“ tok/s | 41,42 | 41,07 | 41,42 | – | +0,85 % |
| AIfred Prosa / Code / Rechnen tok/s | 34,74 / 58,99 / 58,80 | 34,47 / 58,57 / 58,34 | 34,71 / 58,93 / 58,75 | – | +0,7 / +0,7 / +0,8 % |
| Prefill 8687 Token s (weniger ist besser) | 3,85 | 3,85 | 3,84 | 3,85 | gleich |
| Langkontext ms je Schritt 150k / 250k / 516k | – | 65,4 / 66,9 / 68,5 | 65,1 / 66,7 / 68,7 | – | gleich |
| GPU-Spitze MiB Karte 0…4 | 47299 30006 45977 31100 30440 | 47863 30428 46593 31100 30448 | 47863 30366 46593 31100 30440 | – | gleich oder weniger |

- Qualität: 8 von 8 Texten tokengleich zur Produktionskontrolle k = 4, GSM8K 29 in T1, P2, T3. Annahme je Position in allen Armen gleich.
- Layout-Abgleich beim Start auf allen Stufen gleich (Sender = Empfänger), eine Nachricht je Übergabe:
  246 528 / 287 488 / 328 448 / 369 408 Byte bei 5 Zeilen (wächst mit den DFlash2-Hilfszuständen je Stufe).
- Erwartet waren 0,5–0,85 ms je Übergabe; gemessen spart die Runde mit vier Übergaben nur rund 0,35 ms. Die einzelnen isends liefen
  offenbar schon weitgehend überlappt, übrig war nur der Startaufwand je NCCL-Aufruf.

Urteil GLM: kleiner, in allen Größen gleichgerichteter Gewinn (+0,6 bis +0,9 % Decode), unter der Schwelle für neue generische Hebel
(1,5 %), aber gebaut, geprüft und verlustfrei → nach der Regel „belegte, billige Verbesserungen kommen rein“ in die Entwicklungslinie,
in die Produktion mit dem nächsten Bündel, sobald DeepSeek gemessen ist.

## DeepSeek-V4, `ab_ppk_dsv4_2026-11-23a.log` (Kette `chain_ppack_dsv4_2026-11-23.sh`, 16:20–17:06)

Arme T1 P2 T3 P4. DeepSeek übergibt je Stufe nur einen Tensor (`hidden_states` fp16 [Zeilen, 4096]), das Layout-Log zeigt eine
Ein-Segment-Signatur: Packen ändert dort nichts an der Zahl der NCCL-Nachrichten.

| Größe | T1 | P2 | T3 | P4 |
|---|---|---|---|---|
| Decode kurz tok/s (mehr ist besser) | 61,09 | 61,12 | 61,14 | 61,17 |
| Decode lang nach 8921 Token tok/s | 63,08 | 57,70 | 58,23 | 62,30 |
| Prefill s (weniger ist besser) | 2,28 | 2,27 | 2,28 | 2,28 |
| Langkontext ms je Schritt 150k / 250k / 486k (weniger ist besser) | – | 58,1 / 61,7 / 64,4 | 58,1 / 62,0 / 64,6 | – |
| GPU-Spitze MiB Karte 0…4 | 38581 31322 46953 31322 31322 | 38641 … 47013 … | 38641 … 47013 … | 38581 … 46953 … |

Texte T gegen P 8 von 8 tokengleich, GSM8K 28 in allen Armen. Urteil DeepSeek: neutral (erwartet, ein Tensor je Übergabe), kein Nachteil.

## Gesamturteil

GLM +0,6 bis +0,9 % Decode, DeepSeek neutral, Qualität und Speicher gleich. Einbau mit dem nächsten Bündel v15.
