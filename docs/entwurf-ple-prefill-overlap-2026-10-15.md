# Entwurf: PLE-Abruf im PP4-Prefill überlappen (15.10.2026)

Status: Entwurf zur Freigabe durch Peuqui, nichts gebaut. Reihenfolge laut Absprache: erst Aufteilungs-Sweep
(Nacht 15./16.10., `chain_partsweep_2026-10-15.sh`), dann dieser Umbau, falls sich der Rest lohnt.

## Befund (Profil 15.10. 06:05, Produktion aa7087485, 9.697 Token, `nsys/pp4-prefill-1015/`)

| Stufe | Kernelzeit (s) | PLE-Warten (s) | Summe (s) |
| --- | --- | --- | --- |
| 0 (RTX 8000), 11 Schichten | 3,31 | 6 × 0,108 = 0,65 | 3,96 |
| 1 (V100), 11 Schichten | 3,63 | – | 3,63 |
| 2 (RTX 8000), 14 Schichten | 2,75 | – | 2,75 |
| 3 (V100), 12 Schichten + lm_head | 2,61 | – | 2,61 |

Die Pipeline taktet nach Stufe 0. `ple_offload.wait_d2h` dauert je Häppchen 276 ms (11.10.: ~280 ms). Ohne Warten und bei
idealer Verteilung lägen alle Stufen bei ~3,08 s, mit Warten und idealer Verteilung bei ~3,24 s. Der Umbau bringt also
über einen guten Aufteilungs-Sweep hinaus etwa 5 % beim langen Prefill. Decode ist nicht betroffen (dort bleibt alles,
wie es ist).

## Warum heute nichts überlappt (Code aa7087485)

1. `PleOffloadConnector._launch` reiht die D2H-Kopie der Token-IDs in den **Modell-Stream** ein. Die Eingaben für
   Häppchen N+1 entstehen dort erst nach dem Forward von Häppchen N; `_process_request` wartet per `event.synchronize()`
   darauf (das ist `wait_d2h`) und schickt erst dann die Anfrage an den PLE-Worker.
2. Der Worker (`ple_offload/worker.py`, Schleife über die Schichten) wartet vor dem Sammeln auf `wait_reset` bzw.
   `wait_host_resets`, also bis die GPU den Ausgabepuffer nach dem **ganzen** Forward freigegeben hat.
3. Transport ist bei uns „mapped“ (`host_result_capability` auf RTX und V100 erfüllt, MTP zugelassen): Der Worker sammelt
   direkt in den Host-Puffer, aus dem die GPU nach `cuStreamWaitValue32` liest. Ein Vorziehen braucht daher einen
   zweiten Puffer.

## Entwurf (drei Bausteine, keine Locks, nur die vorhandenen Flag-Semaphoren)

**A. Frühe Anfrage für reine Prefill-Batches.** Die Token-IDs liegen im V2-Runner in `req_states.all_token_ids`
(UVA, Host-Speicher) mit CPU-Spiegel `num_computed_prefill_tokens`. Für einen Batch, in dem jede Anfrage nur
Prompt-Token rechnet (keine Entwurfs- oder gerade gesampelten Token), stellt die CPU `input_ids`, `query_start_loc`
und `ngram_context` selbst zusammen und schickt die Anfrage sofort, ohne D2H-Event. Alle anderen Batches (Decode,
Spec-Verify, erster Schritt nach dem Prefill) nehmen den heutigen Weg unverändert; das ist eine Zulassungsbedingung, kein
Ausweichpfad.

**B. Eingaben im Worker übernehmen.** Mit A kann die nächste Anfrage eintreffen, während der Worker noch für die
vorige sammelt. Der Worker kopiert deshalb die Eingaben beim Empfang in einen eigenen Puffer und setzt ein Freigabe-Flag;
der Connector schreibt die nächsten Eingaben erst nach diesem Flag (bei zwei Batches im Flug genügt ein zweiter
Eingabe-Slot).

**C. Zwei Ergebnis-Slots für Batches ohne CUDA-Graph.** Je PLE-Schicht ein gestapelter Ergebnispuffer `[2, N, D]`
und zwei Flags. `ple_offload_wait` bekommt beide Slots und liest den aktuellen Slot zur Laufzeit aus einem
Schichtzustand, den der Runner pro Forward setzt (der Op ist im kompilierten Graphen opak; ausgetauschte Attribute würden
neu kompilieren oder einen alten Zeiger einbacken). Prefill-Häppchen wechseln den Slot; FULL-CUDA-Graph-Decode bleibt fest
auf Slot 0 und startet erst, wenn Slot 1 freigegeben ist. Der Worker sammelt Häppchen N+1 in den freien Slot, während
die GPU den anderen liest.

Reihenfolge-Invarianten (statt Locks): Worker schreibt einen Slot nur nach dessen Reset; die GPU liest einen Slot nur
nach dessen Signal; Decode-Graphen sehen nur Slot 0. Speicher: der Host-Ergebnispuffer verdoppelt sich je PLE-Schicht
(max_num_batched_tokens × ple_dim × 2 Byte), wird beim Bau gemessen und geloggt.

## Messung und Abnahme

- nsys-Prefill wie oben: Lücke vor `_dequantize_ple_fp8_bytes_kernel` auf GPU 0 muss verschwinden, `wait_d2h` entfällt
  für Prefill.
- A/B/A/B PP4 und TP2×PP2 (Prefill 29k, Decode, Greedy, Qualität); Bit-Gleichheit der PLE-Ergebnisse per Test
  (gleiche Zeilen aus beiden Slots).
- Tests: Zulassung (nur reine Prefill-Batches), Slot-Wechsel, Decode-Graph bleibt auf Slot 0, Worker-Eingabeübernahme.

## Offene Fragen an Peuqui

- Lohnt der Umbau, wenn der Sweep schon den Großteil holt (erwartet über den Sweep hinaus ~5 % Prefill PP4)?
- Upstream: als KernelConfig-Feld (Standard aus, bis gemessen) und danach als PR an 1Cat anbieten?
