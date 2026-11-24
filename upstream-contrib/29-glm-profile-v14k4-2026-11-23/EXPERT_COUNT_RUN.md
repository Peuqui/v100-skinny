# Messung: verschiedene Experten je MoE-Schicht und Schritt (GLM-5.3-Flash, Wegwerfzweig)

Baum `1Cat-vLLM-expert-count`, Zweig `diag-expert-count` (Commit 9b561f25dc, aus `merge-1cat-2d218ab03`). Nie mergen, nie hochschicken.

## Was der Zweig tut
- `vllm/diag_expert_count.py`: je MoE-Runner ein Histogramm `[9 Zeilenklassen, 289]` (int32, 10 KB je Schicht, beim Bau des Runners
  angelegt, nicht im Graph). Zeilenklasse i < 8 = i+1 Zeilen, Klasse 8 = mehr als 8 Zeilen (Prefill).
  Der Zähler ist ein registrierter Custom-Op `vllm::diag_expert_count` aus reinen Tensor-Operationen (kein `.item()`, kein `unique`),
  also graph-tauglich. Er liest nur `topk_ids` (`moe_runner.py` nach dem Routing, vor dem Kern).
- Aufgefüllte Zeilen: gezählt werden nur die ersten `query_start_loc[-1]` Zeilen (prepare_inputs füllt alle Einträge hinter
  der letzten Anfrage mit der echten Tokenzahl; der Puffer ist persistent und wird im Graph gelesen).
- Nach der Graph-Aufnahme (`capture_model`) werden alle Histogramme auf null gesetzt (Aufwärm- und Aufnahmeläufe zählen nicht).
- Schreiben: `/home/mp/.cache/bench-scripts/expert_count_2026-11-23/rank<PP-Stufe>.json` alle 200 echten `execute_model`-Aufrufe
  und beim Herunterfahren des Runners (`shutdown`). Pfad = Konstante `OUTPUT_DIR` in `vllm/diag_expert_count.py`.

## Boot
Der Entwurf muss 4 Token lang sein (5-Zeilen-Schritte) und die Capture-Größe 5 haben. Steht im installierten GLM-Eintrag etwas anderes,
mit `MAKE_ENTRY_CMD_OLD` / `MAKE_ENTRY_CMD_NEW` auf `num_speculative_tokens` 4 und `cudagraph_capture_sizes` [5] setzen (Kopfkommentar des Skripts).
```bash
rm -rf ~/.cache/bench-scripts/expert_count_2026-11-23
MAKE_ENTRY_FRESH=1 bash ~/.cache/bench-scripts/make_variant_entry_2026-10-31.sh expcount expert-count glm
```
`MAKE_ENTRY_FRESH=1`, weil sich der Code (neuer Op im Graph) ändert und alte Compile-Caches nicht passen. Das Skript schreibt
`expcount_entry_glm_2026-10-31.yaml`; die Eintragsdatei in die llama-swap-Config einsetzen wie bei den A/B-Läufen (`ab_variant_2026-10-31.sh`),
Boot nur über llama-swap, Hold bei GPU-Orc vorher.

## Anfragen
```bash
cd ~/.cache/bench-scripts && MODEL=GLM-5.3-Flash-320B-A18B-NVFP4-DFlash2-vllm-expcount python3 spec_length_probe_2026-11-22.py /tmp/expcount_probe.json
```
(Der Name ist der Produktionsname mit Suffix `-expcount`; im erzeugten YAML nachsehen.) Das sendet Prosa, Erklärung, Code und Mathe,
greedy und mit AIfreds Sampling. Danach Modell entladen (schreibt beim Herunterfahren die Endstände) oder mindestens 200 Schritte abwarten.

## Auswertung
```bash
python3 /home/mp/Projekte/vllm-research/v100-skinny/upstream-contrib/29-glm-profile-v14k4-2026-11-23/eval_expert_count.py
```
Gibt je Schicht und PP-Stufe Schritte, Mittel, Median und p90 der verschiedenen Experten bei 5-Zeilen-Schritten aus, dazu die
Bytes je Schicht und Schritt (13,44 MB je Experte) und den Vergleich mit 37,8 bei gleichverteilter Wahl (40 Plätze, 288 Experten).

## Zu beachten
- Zeilenklasse 5 enthält auch Prefill-Reste und kurze Anfragen mit genau 5 Token; bei Last durch AIfred (Titel, Vigilantia) können
  fremde 5-Zeilen-Schritte dazukommen: Modell vorher bei GPU-Orc halten.
- Wird eine Anfrage mit weniger als 4 Entwürfen gefahren (Ende einer Anfrage), landet der Schritt in einer kleineren Klasse und stört nicht.
- Eine MTP-Schicht (falls der Eintrag eine hätte) erscheint als eigener Schlüssel; die Werte der Entwurfsschicht stimmen dann nicht
  (anderer Tokensatz). DFlash2 hat keine MoE-Schichten.
