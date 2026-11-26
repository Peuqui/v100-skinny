# Zweiter Stapel-Rebase auf 1Cat main 28f321c2c, Entscheidungen zur Prüfung

Stand 25.11.2026. Basis alt 7db135fa0, neu 28f321c2c (17 neue main-Commits, keiner nativ: `git diff --stat 7db135fa0 28f321c2c -- csrc cmake CMakeLists.txt setup.py` leer).

## Punkt 1: unser Commit "fix(flash_v100): check the dense prefill entry without binding it" ist durch main #1548 (edcc66fa5) doppelt

Beide Seiten (Datei `vllm/v1/attention/backends/flash_v100/dense_prefill.py`, `flash_v100_dense_prefill_lse_available`, Zeile ~1298):

- main (edcc66fa5, Jozkah): `try: return _ops.get_flash_dense_forward() is not None` / `except RuntimeError: return _ops._flash_attn_forward_lse is not None`, mit Kommentar.
- unser Commit (419e1c7ed): trennt in `ops.py` den Lader `_load_flash_dense_forward` vom gebundenen Zugriff und ruft in `dense_prefill.py` `_ops.flash_dense_forward_available()` auf; dazu die Testdatei `tests/v1/attention/test_flash_v100_dense_prefill_availability.py`.

Beide beheben denselben Fehler ("Flash-V100 resources must be bound before execution" beim ersten GLM-Boot). Konfliktregel "main-Seite bleibt, unser Commit obendrauf" lässt zwei Lesarten zu: unser Aufruf ersetzt mains try/except (entfernt main-Zeilen) oder mains Fassung bleibt und unsere ops.py-Änderung wird totes Gerüst.

Gewählt (und im Zweig umgesetzt, jederzeit umkehrbar über die alten Köpfe in `scratchpad/w/oldheads2.txt`): main bleibt unverändert, `ops.py`/`dense_prefill.py` unseres Commits entfallen, übrig bleibt nur die Testdatei als Commit `test(flash_v100): cover the dense prefill availability check before the runtime is bound` (beide Tests laufen grün gegen mains Fassung, 2 passed). Begründung: keine zweite Lösung desselben Fehlers neben einer gerade gemergten (AGENTS.md: keine Fleißarbeit), kein Rücksetzen fremder Zeilen, der Test bleibt als Absicherung.

Alternativen: (a) den Commit ganz weglassen (Test fällt weg), (b) unsere Fassung über mains setzen (entfernt Jozkahs Zeilen, nicht empfohlen). Der Fork enthält weiter unsere Fassung (419e1c7ed-Zwilling); beim nächsten Fork-Merge von main die doppelte Lösung auf mains Fassung zurückführen.

## Punkt 2 (Information): `_graphs_capture_dense_prefill` im Turing-Zweig von #1278

main #1535 (2413bfe68) setzt den Block-FP8-Prefill-Arbeitsbereich nur noch bei aufgenommenem Prefill. Unser #1278 führt für Turing einen eigenen Zweig (`if layer._qpn8_sm75_mma:`) ein, der vor #1535 keinen Arbeitsbereich anlegte. Die Fork-Fassung (0a694f29f) legt ihn dort an, wenn die Graphen mehr als `_SM75_MMA_MAX_M` (512) Zeilen aufnehmen. Ohne diese Zeilen würde mains neuer Test `test_prefill_workspace_is_held_only_for_captured_prefill` auf Turing fehlschlagen (`resident` erwartet den Arbeitsbereich bei Aufnahmegröße 1024 und bei festem KV-Budget; Schluss aus dem Code, nicht auf der Karte geprüft, GPU war für diesen Lauf gesperrt). Übernommen als fixup in den #1278-Commit "feat(sm75): tensor-core GEMM ..." (Fork-Beleg: `git show prod-v16:vllm/model_executor/kernels/linear/qpn/fp8_block.py`, Zeilen ~881–888). Nur die main-Fassung von #1535 plus diese Turing-Anpassung; 6597ec462 (Erweiterung des Forks) ist in main #1535 ohnehin enthalten.

## Punkt 3: main #1541 (d0c81f651) wirft bei jedem unbekannten `quant_algo`; unser #1286 bedient MXFP8

`ModelOptMixedPrecisionConfig.get_quant_method` (modelopt.py ~2830): main fügt vor dem letzten `return UnquantizedLinearMethod()` ein `if quant_algo: raise ValueError("Unsupported quant_algo ...")` ein. In #1286 stehen MXFP8-Schichten aber nicht auf dem Ablehnungsweg: in den Commits 4 bis 11 bleiben sie unquantisiert (der Lader dequantisiert sie), ab Commit 12 bekommen die Linearschichten `ModelOptMxFp8LinearMethod`, der MXFP8-LM-Kopf bleibt dem Lader überlassen (Test `test_modelopt_mixed_precision_leaves_the_mxfp8_lm_head_to_the_loader`, in main nicht vorhanden, in unserem Stapel grün auf der CPU nur mit dieser Lösung). Eine wörtliche Übernahme von mains Zeile würde GLM-5.3-Checkpoints mit MXFP8-Schichten beim Laden abbrechen.

Gewählt: in #1286 Commit 4 und 12 lautet die Bedingung `if quant_algo and quant_algo != "MXFP8":` (eine main-Zeile bewusst geändert); in #1348 liefert der MXFP8-Zweig für alle Schichten (auch den Kopf) eine Methode, dort steht wieder mains `if quant_algo:`. Alternative: mains Prüfung im Stapel unverändert lassen und den MXFP8-Zweig im ersten Commit (4) vorziehen. Ich halte die gewählte für die kleinere Änderung.

Folge: mains neuer Test `test_preserves_unquantized_controls` (#1541) erwartet für den LM-Kopf `UnquantizedLinearMethod`; unser Commit "fix(modelopt): leave a full-precision LM head to its embedding method" gibt dem Kopf bewusst `None` (eigene `UnquantizedEmbeddingMethod`, sonst fehlt `_layer_execution_policy`). Der Test folgt im selben Commit (LM-Kopf erwartet `None`), wie schon der Alttest in `test_modelopt.py`.

## Punkt 4 (Information): zwei in der ersten Runde verlorene Fork-Testanpassungen, jetzt behoben

- `tests/models/glm5next/test_sm70_kda.py::test_sm70_exact_kda_projection_dispatch`: der Commit "keep the MXFP8 projections as 8 bits in QPN8" lässt `forward` über `_project_inputs` laufen; der Fork ruft deshalb `_project_inputs(layer, hidden)` mit `split_qkv=False` auf (prod-v16 Zeile ~428–435). Im Stapel stand noch der alte `forward`-Aufruf (8 Fehlschläge auf der CPU, auf main grün). Als fixup in diesen Commit.
- `tests/quantization/test_qpn8_grouped_bmm.py::test_large_rows_dequantize_the_whole_weight_into_the_shared_workspace`: der Fork patcht `_block_qpn8_prefill_dense` (prod-v16 Zeile 125), weil die gruppierte Schreibfunktion mit #1535 darüber läuft; im Stapel patchte der Test noch `_ensure_block_qpn8_prefill_workspace`. `_qpn8_native_grouped_linear_write` ruft jetzt `_block_qpn8_prefill_dense` (wie der Fork, Zeile 423). Als fixup in den wo_a-Commit.
