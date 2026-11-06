# Upstream-Backport-Bündel (Durchsicht 06.11.2026)

Grundlage: alle 7755 upstream gemergten vLLM-PRs seit 1Cats Basis (29.05.2026), 1903 Bugfixes, 606 Kandidaten in vier Bereichen,
von vier Prüf-Agenten am Code geprüft (Berichte `review_*.md` hier, Rohdaten `~/.cache/upstream-backport-scan-2026-11-06/`).
Die wichtigsten Aussagen (#51856, #58454, #66048) habe ich selbst am Code bestätigt; die übrigen prüfe ich beim Einbau.

Regeln: Prüfstufen und Schwellen der Projekt-CLAUDE.md; jeder Fix ein eigener Commit („Backport of vllm-project/vllm#N“),
Upstream-Test mit, wo vorhanden; danach je Fix ein Backport-PR an 1Cat.

## Stufe 1 (heißer Pfad, ohne Tempo-Absicht) — ein Bündel, verkürzte A/B je Modell
| PR | Was | Betrifft | Form |
| --- | --- | --- | --- |
| #57477 | GLM-Prefill-Tail rechnet mit dichtem Stride, schreibt in fremde Indexer-Blöcke (ab 2048 Token, Präfix-Cache) | GLM | anpassen, vor #58454 |
| #58454 | kpool-Ring nur 4 Plätze, DFlash k=7 überschreibt angenommene Einträge | GLM | anpassen (Ring 16) |
| #63475 (+ Zeilenend-Sync aus #62936, #62467) | persistent_topk/radix_topk ohne Sync ab ≈131k Kontext | DeepSeek, GLM | anpassen, `_C` neu bauen |
| #66048 | Race in `_insert_resampled_kernel` bei Temperatur > 0 | alle mit Spekulation | übernehmen (num_warps=1) |
| #45224 | shm_broadcast-Leser: Leerlauf-Warten begrenzen, Leseslot im finally freigeben | alle | übernehmen |
| #44490 | Host-Speicherleck `new_block_ids` (nur Scheduler-Hunk) | DeepSeek, K7 | anpassen |
| #54436 | PP-Broadcast verliert fertig werdende Anfrage (heute nicht erreichbar, max_num_seqs 1) | PP | anpassen, vorbeugend |
| #51694 | Block-Hash bei mehreren Bildern (27B, praktisch kaum Wirkung) | 27B | anpassen, niedrig |

## Stufe 0 (keine Wirkung auf die Rechnung) — Tests + ein Kontrolllauf je Modell
| PR | Was | Betrifft | Form |
| --- | --- | --- | --- |
| #51856 | DSv4-Renderer: Tools an die vorhandene Systemnachricht statt leerer Zusatz-Systemnachricht | DeepSeek, K7 | übernehmen, Textvergleich |
| #51262 | DSv4-Encoder: Übergang System → Assistant fehlt (AIfreds Kompression) | DeepSeek, K7 | übernehmen |
| #46303 | Streaming meldet `tool_calls` auch bei an `max_tokens` abgerissenem Aufruf | alle | anpassen |
| #40986 | Fehler vor dem ersten Token wird verschluckt (stille leere Antwort) | alle | anpassen |
| #43984, #55743 | `inf`/große Zahlen in Tool-Argumenten: Stream-Abbruch bzw. Genauigkeit | alle | anpassen |
| #44383 | `cached_tokens: 0` wird gemeldet | alle | übernehmen |
| #42752 | `tool_choice="none"` im Streaming ruft trotzdem den Tool-Parser | alle | übernehmen |
| #44974 | EXIF-Ausrichtung/Transparenz bei Bildern | GLM, 27B | übernehmen (Frage `contextlib.suppress`) |
| #52142 | Compile-Cache-Kopien mit absoluten Pfaden auf alte Ordner | alle | übernehmen |
| #47911 | Compile-Hash ohne Entwurfsmodell | alle mit Entwurf | übernehmen (erster Boot kompiliert neu) |

## Ausdrücklich NICHT übernehmen
- #46243 (min_tokens): maschinell sauber, aber 1Cat hat es an der Quelle anders gelöst; der Diff ließe EOS einen Schritt zu früh durch.
- #60767 unverändert: würde den KpoolTail-Puffer nullen (GLM kaputt).
- #50183 (NaN im Rejection-Sampler): verdeckt NaN nur (No-Fallback-Regel).
- Alle übrigen: Begründung je PR in den Berichten.

## Offen
- #44974: upstream umschließt `exif_transpose` mit `contextlib.suppress(Exception)` (kaputte EXIF-Daten → Bild ungedreht statt Fehler).
- 27B läuft noch aus v5d; seine Fixes (#66048, #51694, #44974, Frontend) greifen erst, wenn es auf die v6-Linie wechselt (27B-Ausreißer, Termin Sa 07.11.).
- Optional: `lock_workspace()` nach `capture_model` in MRV2 als Wächter (kein Upstream-Fix, Anlass #55341).
