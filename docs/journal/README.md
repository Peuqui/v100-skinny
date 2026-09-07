# Journal — chronologische Logbücher

Diese Dateien sind **eingefroren**. Sie beantworten „warum ist es so", nicht
„wie ist es". Der aktuelle Betriebsstand steht in `../../STAND.md`, der laufende
Auftrag in `../../HANDOVER.md`.

**Nicht hier anfangen.** Die Logbücher sind gewachsene Messprotokolle; wer den
Stand sucht, liest `STAND.md` und kommt nur hierher, wenn dort ein Verweis auf
eine Begründung zeigt.

| Datei | Stand | Zeilen | Worüber es Auskunft gibt |
|---|---|---|---|
| `MERGE-PROJECT-HANDOVER.md` | 24.08. | 385 | PP×TP-Merge: wie das heterogene 2×2-Gitter zustande kam, Referenzkonfiguration, 85-tok/s-Beleg |
| `DEEPSEEK-VLLM-HANDOVER.md` | 25.08. | 865 | DeepSeek-V4-Flash unter vLLM und die Entscheidung gegen Block-FP8 — Begründung für den Schwenk auf Flash-Next |
| `QWEN4EXP-PORT-HANDOVER.md` | 27.08. | 2626 | Portierung von Qwen3.8-Flash-Next (Qwen4Exp) auf 1Cat 1.3.0. **Hier stehen die Begründungen zum Betriebspunkt**: warum k=4, warum `capture_sizes [1,2,4,5,8]`, warum Kartenreihenfolge `0,2,1,4` |
| `SPEC-LONGCTX-HUNT.md` | 29.08. | 279 | Warum die MTP-Spekulation bei langem Kontext einbrach |
| `FP8-EVALUATION.md` | 31.08. | 543 | FP8 gegen NVFP4 nach der Klage „zerfällt bei langer deutscher Generierung" — Formatvergleich |
| `REBASE-150-EVAL.md` | 03.09. | 172 | Klassifikation aller Deploy-Ziele beim Rebase von 1.3.0 auf 1.5.0 |
| `TURING-COEXISTENCE-HANDOVER.md` | 04.09. | 1926 | sm75-Koexistenz. **Hier steht das nsys-Rezept** (Abschnitt 03.09. 10:20) und die Kernel-Rangliste je Layer·Step |
| `NIGHT-REPORT-2026-09-06.md` | 06.09. | 173 | Autonome Nachtschicht: Plan A (Dense-Prefill), Neukalibration, der Drei-Fragen-Test |
| `MTP-V2-CUDAGRAPH-HANDOVER.md` | 07.09. | 207 | Die Jagd nach dem NaN unter CUDA-Graphen. **Teilweise überholt**: Abschnitt 6 (Split-Graphen) ist widerlegt, der Defekt selbst ist behoben — siehe `HANDOVER.md` |
| `baseline-/final-/postfix-matrix-*.txt` | 29./30.08. | je 28 | Messmatrizen der Kalibrationsläufe |

## Beim Nachschlagen beachten

Die Dokumente sind zu ihrem Zeitpunkt korrekt gewesen, aber der Stack hat sich
bewegt — besonders durch den Rebase auf 1Cat 1.5.0 (03.09.) und den V2-Runner.
Zwei bekannte Beispiele:

- Die Startzeile in `../../FLASH-NEXT-OPERATING-POINT.md` bootet seit 1.5.0
  nicht mehr (fehlendes `QUANT_BACKEND`), obwohl die Messwerte dort gelten.
- `MTP-V2-CUDAGRAPH-HANDOVER.md` nennt „60–90 s Boot" für den 27B; gemessen
  sind es 6,5 min auf Turing.

Wer hier eine Aufrufzeile oder eine Umgebungsvariable findet, prüft sie gegen
`STAND.md`, bevor er sie benutzt.
