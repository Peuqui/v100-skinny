# Offene Fragen nach dem Rebase auf 1Cat main 73da3445d (#1616 Inkling 2 Mini), 25.11.2026

Alle 13 PR-Zweige liegen lokal auf 73da3445d (nicht gepusht). Alte Köpfe: `scratchpad/w/oldheads3b.txt`.
Belege: `scratchpad/w/rd3b_<zweig>.txt` (range-diff), `pc3b_<baum>.log` (pre-commit), `t3b_<label>/summary.log` (Tests).

## 1. Echter Befund: #1286 bricht ModelOptNvFp4SM70MoEMethod.create_weights (Wechselwirkung mit #1616)

- `ModelOptNvFp4SM70MoEMethod.__init__` (vllm/model_executor/layers/quantization/nvfp4_sm70_moe.py, ca. Zeile 702) ruft
  `FusedMoEMethodBase.__init__` statt `ModelOptNvFp4FusedMoE.__init__` und erbt `create_weights`.
- Commit 883986ac7 (feat(glm53): load GLM-5.3-Flash-FP4-Local, Zweig pr-glm53-r1125, #1286) liest in `create_weights`
  (modelopt.py, ca. Zeile 1723) `self.csf_scales`; das Attribut wird nur in `ModelOptNvFp4FusedMoE.__init__` gesetzt.
- Nachweis (CPU, Mock mit gesetztem `use_global_sf`, `scratchpad/w/csf_repro3b.py`): im Baum pr-r-turing (ohne #1286)
  `create_weights OK`; in pr-r-glm53 und allen Zweigen darüber (pr-glm53-prefill-head, pr-sm70-moe-kernel-pack,
  pr-sm70-moe-routing-sort-v2) `AttributeError: ... has no attribute 'csf_scales'`.
- Vorher latent (Produktion wählt den Skinny-Pfad ausdrücklich). Mit #1616 baut `Inkling2QuantConfig.get_quant_method`
  (vllm/models/inkling2/quant_config.py) genau diese Methode direkt für jede FusedMoE-Schicht: Inkling 2 Mini würde auf main plus #1286
  beim Laden abstürzen. Auch jedes andere NVFP4-Modell auf V100 über TurboMind ohne ausdrücklichen Backend-Wunsch.
- Vorschlag (nicht umgesetzt, ändert PR-Inhalt): in 883986ac7 `csf_scales: bool = False` als Klassenattribut von
  `ModelOptNvFp4FusedMoE` (oder `self.csf_scales = False` im SM70-`__init__`) plus ein Test in
  tests/quantization/test_sm70_glm53_modelopt_nvfp4.py, der `create_weights` der SM70-Methode aufruft; Fixup, danach
  #1286 und alles darüber (pr-glm53-prefill-head, moe-kernel-pack, routing-sort-v2) neu aufsetzen.
  Frage: Soll ich das so machen (Fixup + Restapel)?

## 2. pre-commit: rot durch main selbst, nicht durch unsere Zweige

Referenzlauf auf main (`pc3b_MAIN.log`, Bereich 28f321c2c..73da3445d) ist rot in: ruff-check (Autofix), markdownlint (MD056 in
docs/design/sm70_inkling2_mini_port.md), mypy-local und mypy-3.10 (u. a. model_runner.py:1448 `try_sample_target`, dspark/widths.py:52,
moe_ref.py:138/142, fp4_decode_gemv.py:768, fp4_prefill_grouped.py:757), check-architecture-reference (+ Test), check-layering.
- check-layering: 33 gemeldete Dateien, in allen 13 Zweigen identisch mit main (Baseline `tools/pre_commit/layering_baseline.json`
  nicht aktualisiert). Kein Zweig fügt eine Kopplung hinzu.
- check-architecture-reference: `runtime_reference.md` ist auf main veraltet (nur die Zeilenanker von SpeculativeConfig
  und VllmConfig). Der Regenerier-Diff ist in allen 13 Zweigen byte-identisch mit dem von main (md5 gleich). Ein Fixup in unsere Commits
  würde fremde Zeilen mitnehmen; deshalb nicht gemacht.
- mypy: in den acht Zweigen, die model_runner.py ändern, derselbe eine Fehler wie auf main (model_runner.py:1448, bei pp-pack:1446).
Frage: Soll 1Cat darauf hingewiesen werden (eigener Fix-PR für Layering-Baseline, Architekturreferenz, mypy), oder
soll jeder Zweig den Fix von #1616 mittragen? Letzteres würde Fremdzeilen in unsere PRs mischen und bei einem Fix durch 1Cat konfliktieren.

## 3. Tests

Auf main (73da3445d) schlagen auf CPU ohne GPU in den gewählten Dateien viele Tests fehl (18 von 41 Dateien), darunter veraltete Tests
(z. B. `_v41_ahead_map`, `SimpleNamespace` ohne `model`, `RequestState(model_dtype=...)`) und reine GPU-Tests. Die Zweige haben
dieselben Fehlschläge, Name für Name, und keine neuen. Die veralteten Alt-Tests auf main sind nicht unser Gegenstand; Frage, ob wir sie
in einem eigenen PR korrigieren sollen.

## 4. Sonstiges

- Die Bäume prmain2 und die Zweigbäume sind sauber (`git status --porcelain --untracked-files=no` leer). pre-commit-Autofixes auf main
  (ruff, markdownlint) hatten prmain2 kurz verändert; zurückgesetzt.
- Nichts gepusht, nichts gelöscht.
