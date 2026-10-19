# Entwurf: Angepasste Prüfbreite je Schritt, auch unter PP (19.10.2026)

Status: Entwurf, nichts gebaut außer dem Schätzer-Port (siehe Schritte). Freigabe Peuqui 19.10. („Hebel 3“, alle
Vorschläge umsetzen). Gilt für jeden Entwurf mit Logits: MTP (27B, Flash-Next), DSpark (DSv4), später GLM.

## Warum

- Flash-Next PP4, feste K (02.10., `k_sweep_2026-10-02.log`): K=4 Prosa 33,6 / Code 57,0 / Bearbeitung 83,5 tok/s,
  K=2 Prosa 36,2 (+8 %) aber Code −11 %, Bearbeitung −28 %. Jede Entwurfsposition kostet ~6 ms. Keine feste Tiefe ist
  überall die beste.
- DSv4 11.10. (`dsv4_av_2026-10-11.log`): 1Cats `dspark_confidence_threshold` kürzt zwar, kostet aber mehr, als es
  spart: Sie verlangt synchrones Scheduling (Prefill unter PP 3,6× langsamer), und kürzere Prüfungen fallen aus den
  CUDA-Graphen (Schritt 72 → 82 ms bei Schwelle 0,3).

## Was Upstream macht und warum es hier nicht passt

- #47808 (Infrastruktur) + #52228 (Online-Schätzer ohne Konfidenzkopf), Doku
  `docs/features/speculative_decoding/adaptive_verification.md`:
  - Schätzer je Position: `p = sigmoid(w · logit(q_max) + b_k)`, `q_max` aus den Entwurfslogits (nicht aus dem
    gezogenen Token, sonst nicht verlustfrei); Online-Newton-Schritt alle 100 Schritte.
  - Budget auf der CPU aus den Konfidenzen des VORIGEN Schritts (asynchrone D2H-Kopie) und einer beim Start
    gemessenen Kostentabelle je Token-Anzahl: `argmax_n (1 + Σ_{i<n} Überleben_i) / Kosten(n)`.
  - Verteilung über Anfragen auf dem Gerät (frische Konfidenzen), Attention-Backends müssen gerätebestimmte
    Query-Längen vertragen, volle CUDA-Graphen mit variabler Länge.
- Ausdrücklich NICHT mit Pipeline-Parallelität („cost curves and confidences exist only on the last rank“) und nicht
  mit `use_local_argmax_reduction`. Damit fielen DSv4 (PP5), Flash-Next (PP4, TP2×PP2) und GLM (PP5) heraus. 1Cat hat
  keinen Teil davon.

## Unsere Fassung

Einzelnutzer, meist eine Anfrage: Die Verteilung über Anfragen entfällt, die CPU-Entscheidung aus den Konfidenzen des
vorigen Schritts ist die ganze Entscheidung. Damit sind die Query-Längen auf der CPU exakt; kein Backend muss
gerätebestimmte Längen können.

1. **Konfidenz:** In `sample_draft` (MTP, DSpark) das Merkmal `logit(q_max)` je Position mit dem Triton-Kern aus #52228
   (`acceptance_estimator.py`, Online-Schätzer, Kalibrierung gegen das Prüfergebnis). Bei `use_local_argmax_reduction`
   (27B TP2) lokal je Rang Maximum und Exponentsumme bilden und mit dem Argmax-Austausch mitschicken (zwei Zahlen je
   Rang statt der Logits).
2. **Entscheidung:** Auf der letzten PP-Stufe nach dem Entwurf: Überlebenswahrscheinlichkeiten aufmultiplizieren,
   `n* = argmax_n (1 + Σ_{i<n} S_i) / Kosten(n+1)` mit der Kostentabelle des Ziels.
   Ergebnis `n*` je Anfrage reist im PP-Paket mit (`PPHandler.broadcast`, `pp_utils.py`), das ohnehin Sample- und
   Entwurfstoken an alle Stufen schickt. Ohne PP: dieselbe Zahl, lokal.
   - Async-Scheduling: Der Scheduler plant weiter k Platzhalter. Jede Stufe liest `n*` des vorigen Schritts (eine
     Stufe veraltet, wie Upstream) aus einem doppelt gepufferten Pinned-Puffer ohne Sync und kürzt den Schritt auf
     n*+1 Token. Die gekürzten Entwürfe zählen als verworfen (`num_rejected`); der Scheduler rollt sie zurück wie
     jede Verwerfung.
3. **Graphen:** 1Cat kann schon Ziel-Decode-Graphen für mehrere Query-Längen aufnehmen
   (`CudaGraphManager.decode_query_lens`, `cudagraph_utils.py` ~216–330, heute für DFlash2-Tails q1–q7 bei einer
   Anfrage). Für AV: q = 2 … k+1 bei einer Anfrage, nur Ziel; der Entwurf läuft weiter mit voller Breite.
4. **Kosten:** beim Start je q einmal den Ziel-Graphen messen (Events, nach dem Warmup, vor der Speicherzuteilung des
   KV-Caches) und auf alle Ränge verteilen; unter PP zählt die Summe über die Stufen, also misst jede Stufe und die
   letzte addiert.
5. **Schalter:** `speculative_config.adaptive_verification` (eigene Fassung, nicht Upstreams
   `enable_adaptive_verification`, weil die Semantik anders ist), Vorgabe aus.

## Verlustfreiheit

Das Merkmal hängt nur von der Entwurfsverteilung ab, nicht vom gezogenen Token; ein gekürzter Entwurf ist ein
kürzerer Vorschlag. Das Ziel prüft wie immer, die Ausgabe bleibt gleich (greedy wortgleich bis auf die bekannte
Prüfbreiten-Kippe bei Fast-Gleichständen, siehe K=2 02.10.: 1/3 anders).

## Schritte

1. Schätzer-Port (#52228 `acceptance_estimator.py` + Test) — ohne GPU-Abhängigkeit vom Runner, zuerst.
2. K-Messung DSv4 (k=3/5/7, Prosa/Code/Bearbeitung), um das Potenzial vor dem Bau zu belegen.
3. Graphen für q=2…k+1 und Kostentabelle, Entscheidung ohne PP (Flash-Next nicht, 27B TP2 schon) → A/B.
4. PP-Paket um `n*` erweitern → A/B Flash-Next PP4, DSv4 PP5.
