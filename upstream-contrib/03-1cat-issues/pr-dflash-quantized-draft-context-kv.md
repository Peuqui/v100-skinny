# PR-Entwurf: DFlash + quantisierter Entwurfskopf

**Status:** UNGEPOSTET, wartet auf Freigabe von Peuqui.
**Branch:** `dflash-quantized-draft-context-kv` auf `Peuqui/1Cat-vLLM`, Commit
`be2ba891`, Basis `origin/main` = `0a0d4d67`.
**Ziel-Repo:** `1CatAI/1Cat-vLLM`

---

## Vor dem Absenden von Peuqui zu prüfen

Laut `AGENTS.md` sind reine Agent-PRs verboten; du musst **jede geänderte
Zeile** gelesen haben und die Änderung end-to-end verteidigen können. Der
Diff ist klein — 53 hinzugefügte Zeilen in einer Datei plus eine neue
Testdatei:

```bash
cd ~/Projekte/vllm-research/1Cat-vLLM
git show be2ba891 --stat
git show be2ba891 -- vllm/model_executor/models/qwen3_dflash.py
```

Die drei Stellen, auf die es ankommt:

1. `_has_dense_qkv_weight()` — neuer Helfer, entscheidet allein über
   `isinstance(quant_method, UnquantizedLinearMethod)`.
2. `_build_context_kv_buffers()` — der bisherige Pfad läuft unverändert
   weiter, wenn alle Schichten dicht sind; sonst wird `_fused_kv_weight`
   auf `None` gesetzt.
3. `_fuse_dense_kv_weight()` + der Aufruf oben in `_project_context_kv()` —
   der aufgeschobene Bau.

---

## Duplikatsprüfung (AGENTS.md, Pflicht)

Ausgeführt am 2026-09-10 gegen `1CatAI/1Cat-vLLM`:

```bash
gh pr list --repo 1CatAI/1Cat-vLLM --state open --search "dflash quantized draft"
gh pr list --repo 1CatAI/1Cat-vLLM --state open --search "dflash2 nvfp4"
gh pr list --repo 1CatAI/1Cat-vLLM --state open --search "context_kv"
gh pr list --repo 1CatAI/1Cat-vLLM --state open --search "_build_context_kv_buffers"
gh issue list --repo 1CatAI/1Cat-vLLM --state all --search "dflash draft quantized"
gh issue list --repo 1CatAI/1Cat-vLLM --state all --search "context kv shapes cannot be multiplied"
gh issue list --repo 1CatAI/1Cat-vLLM --state open --search "dflash"
```

Ergebnis: **kein offener PR und kein Issue** zu diesem Absturz.

- Die Suche `dflash2 nvfp4` liefert drei offene PRs: **#561**, **#576**,
  **#572**. #576 und #572 sind unsere eigenen (SM70-Gates bzw. Turing).
- **#561** („[Kernel] Reduce DFlash2 weight and scale memory on SM70",
  DRAFT) liegt thematisch am nächsten. Geprüft: er teilt NVFP4-Codes
  zwischen QPN2 und TurboMind im **Zielmodell** und fasst
  `vllm/model_executor/models/qwen3_dflash.py` **nicht** an. Keine
  Überschneidung.
- Nächstliegendes Issue ist **#478** („DFlash2 acceptance=0 on V100 TP4") —
  ein Annahmeproblem bei laufendem Server, kein Ladefehler. Anderes Thema.

---

## Der Fehler

`_build_context_kv_buffers` legt die K/V-Projektionen aller Entwurfsschichten
in eine Matrix zusammen, damit die Kontextvorberechnung mit einem GEMM
auskommt:

```python
kv_weights = [a.qkv_proj.weight[a.q_size :] for a in layers_attn]
```

`.weight` trägt die dichte `[N, K]`-Matrix nur bei einer unquantisierten
Schicht. Ein quantisierter Entwurfs-Checkpoint hält dort die **gepackten
Codes** — NVFP4 packt zwei Werte je Byte, der Tensor ist also `[N, K // 2]`.
Die Fusion kommt halb so breit heraus, und `_project_context_kv` bricht ab:

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (2048x5120 and 2560x5120)
```

Der Abbruch passiert in `determine_available_memory` → `profile_run` →
`speculator.propose`, also **vor dem ersten Token**. DFlash ist damit mit
jedem quantisierten Entwurfskopf unfahrbar, unabhängig von der Kartenklasse.

Nachvollzogen mit Qwen3.8-27B und einem NVFP4-Entwurfskopf, der aus
`incoai/Qwen3.8-27B-DFlash2` abgeleitet ist. Der Fehler steht unverändert in
`origin/main` (`0a0d4d67`), Zeile 498.

## Der Fix

Die dichten Zeilen über die Quantisierungsmethode der Schicht selbst holen:
eine Einheitsmatrix durch `quant_method.apply` liefert das transponierte
Gewicht, **unabhängig vom Packformat** — also ohne Annahme über NVFP4, FP8
oder marlin. Der unquantisierte Pfad bleibt unberührt und fusioniert weiter
sofort.

Der Bau muss **aufgeschoben** werden: `_build_fused_kv_buffers` läuft am Ende
von `load_weights`, also vor `process_weights_after_loading`; dort ist
`quant_method` noch nicht benutzbar. Die Fusion entsteht deshalb bei der
ersten Kontextprojektion. Kosten: eine dichte fp16-Kopie der K/V-Zeilen (rund
52 MiB je Rang beim Qwen3.8-27B) und ein GEMM je Schicht, einmalig. Der
Decode-Pfad je Schritt bleibt quantisiert — dort sitzt der Gewinn, die
Kontextvorberechnung ist Prefill-Arbeit.

## Messung

Qwen3.8-27B, TP2, DFlash2 k=7, greedy, je 5 Läufe, Median:

| Karten | fp16-Entwurfskopf | NVFP4-Entwurfskopf | |
|---|---:|---:|---|
| 2× Quadro RTX 8000 | 72,72 | **77,13** tok/s | +6,1 % |
| 2× Tesla V100-PCIE | 74,02 | **76,33** tok/s | +3,1 % |

Der Antworttext ist in **jedem** Lauf byteidentisch (sha256-Präfix
`0106659946c064b1`), die Annahmelänge geht 3,353 → 3,325. Genau das ist das
Sicherheitsargument für einen quantisierten Entwurfskopf: angenommen werden
nur Token, die das Zielmodell ohnehin erzeugt hätte. Quantisierung kostet
Annahmerate, nie Korrektheit.

## Tests

Neue Datei `tests/kernels/attention/test_dflash2_context_kv_quantized.py`,
CPU-only, mit einer minimalen Fake-Quantisierungsmethode (kein Checkpoint
nötig): `.weight` ist gepackt, das dichte Gewicht nur über
`quant_method.apply` erreichbar.

```
$ .venv-sm70-150/bin/python -m pytest \
    tests/kernels/attention/test_dflash2_context_kv_quantized.py \
    tests/kernels/attention/test_dflash2_context_pipeline.py -v
4 passed, 15 warnings in 2.30s
```

Die **komplette** bestehende Datei `test_dflash2_context_pipeline.py` ist
mitgelaufen (Lehre aus PR #485), dazu die benachbarten DFlash2-Tests:

```
$ ... -m pytest tests/v1/spec_decode/test_dflash2_alignment_rank.py \
    tests/v1/spec_decode/test_dflash2_ngram_assist.py \
    tests/v1/spec_decode/test_dflash2_structured_output.py -q
38 passed, 19 warnings in 28.68s
```

**Gegentest**, dass die neuen Tests den Fehler wirklich fangen — Fix
zurückgenommen (`git checkout origin/main -- <datei>`), dieselben Tests:

```
test_unquantized_draft_head_fuses_eagerly        PASSED
test_quantized_draft_head_defers_then_fuses_dense FAILED
test_quantized_and_unquantized_fusions_agree      FAILED
2 failed, 1 passed
```

Der dichte Pfad bleibt also grün (der Fix ändert ihn nicht), die beiden
quantisierten Fälle sind vorher rot und nachher grün.

## Linter

```
$ pre-commit run --files vllm/model_executor/models/qwen3_dflash.py \
    tests/kernels/attention/test_dflash2_context_kv_quantized.py
ruff check ... Passed
ruff format ... Passed
typos ... Passed
Run mypy locally for lowest supported Python version ... Passed
(alle übrigen Hooks Passed oder Skipped)

$ pre-commit run mypy-3.10 --hook-stage manual --files <dieselben Dateien>
Run mypy for Python 3.10 ... Passed
```

---

## Vorgeschlagener PR-Text (englisch, für GitHub)

### Title

```
[Bugfix] DFlash: fuse context K/V through quant_method so a quantized draft head loads
```

### Body

## Purpose

DFlash never finishes its profile run when the draft checkpoint is quantized.
`DFlashQwen3Model._build_context_kv_buffers` fuses the per-layer K/V
projections into one matrix by slicing rows out of `qkv_proj.weight`:

```python
kv_weights = [a.qkv_proj.weight[a.q_size :] for a in layers_attn]
```

That attribute holds the dense `[N, K]` matrix only for an unquantized layer.
A quantized checkpoint keeps packed codes there — NVFP4 packs two values per
byte, so the tensor is `[N, K // 2]` — and the fused matrix comes out half as
wide. `_project_context_kv` then fails:

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (2048x5120 and 2560x5120)
```

The abort happens in `determine_available_memory` → `profile_run` →
`speculator.propose`, before a single token is served, so DFlash is currently
unusable with any quantized draft head on any device class. Reproduced on
Qwen3.8-27B with an NVFP4 draft head derived from
`incoai/Qwen3.8-27B-DFlash2`; the line is unchanged on `main`.

## Fix

Reach the dense rows through the layer's own quantization method: feeding an
identity matrix through `quant_method.apply` returns the transposed weight
whatever the packing is, so this needs no knowledge of NVFP4, FP8 or marlin
layouts. The unquantized path is untouched and still fuses eagerly.

The rebuild has to be deferred. `_build_fused_kv_buffers` runs at the end of
`load_weights`, before `process_weights_after_loading`, so `quant_method` is
not usable yet; the fusion is built on the first context projection instead.
It costs one dense fp16 copy of the K/V rows (~52 MiB per rank on
Qwen3.8-27B) and one GEMM per layer, once. The per-step decode path keeps
using the quantized weights, which is where the win is — the context
projection is prefill work.

## Result

Qwen3.8-27B, TP2, DFlash2 k=7, greedy, 5 runs, median, fp16 draft head
against the NVFP4 one:

| GPUs | fp16 draft | NVFP4 draft | |
|---|---:|---:|---|
| 2x Quadro RTX 8000 | 72.72 | **77.13** tok/s | +6.1 % |
| 2x Tesla V100-PCIE | 74.02 | **76.33** tok/s | +3.1 % |

The answer text is byte-identical in every run (sha256 prefix
`0106659946c064b1`); acceptance length moves 3.353 → 3.325. That is expected
and is the safety argument for quantizing a draft head at all: only tokens
the target model would have produced anyway are accepted, so quantization can
cost acceptance rate but never correctness.

## Not a duplicate

Checked on 2026-09-10 against `1CatAI/1Cat-vLLM`:

```bash
gh pr list --repo 1CatAI/1Cat-vLLM --state open --search "dflash quantized draft"
gh pr list --repo 1CatAI/1Cat-vLLM --state open --search "dflash2 nvfp4"
gh pr list --repo 1CatAI/1Cat-vLLM --state open --search "context_kv"
gh pr list --repo 1CatAI/1Cat-vLLM --state open --search "_build_context_kv_buffers"
gh issue list --repo 1CatAI/1Cat-vLLM --state all --search "dflash draft quantized"
gh issue list --repo 1CatAI/1Cat-vLLM --state open --search "dflash"
```

No open PR or issue covers this crash. The nearest open PR, #561 (*Reduce
DFlash2 weight and scale memory on SM70*, draft), shares NVFP4 codes between
QPN2 and TurboMind in the **target** model and does not touch
`qwen3_dflash.py`. The nearest issue, #478 (*DFlash2 acceptance=0 on V100
TP4*), is an acceptance problem on a running server, not a load failure.

## Test plan

New file `tests/kernels/attention/test_dflash2_context_kv_quantized.py`,
CPU-only, using a minimal fake quantization method so no checkpoint is
needed: `.weight` is packed and the dense weight is reachable only through
`quant_method.apply`.

```
$ python -m pytest tests/kernels/attention/test_dflash2_context_kv_quantized.py \
                   tests/kernels/attention/test_dflash2_context_pipeline.py -v
4 passed, 15 warnings in 2.30s

$ python -m pytest tests/v1/spec_decode/test_dflash2_alignment_rank.py \
                   tests/v1/spec_decode/test_dflash2_ngram_assist.py \
                   tests/v1/spec_decode/test_dflash2_structured_output.py -q
38 passed, 19 warnings in 28.68s
```

The complete existing `test_dflash2_context_pipeline.py` was run, not just
the new tests. With the fix reverted, the two quantized tests fail and the
unquantized one still passes, so they do catch this bug and do not constrain
the dense path.

```
$ pre-commit run --files vllm/model_executor/models/qwen3_dflash.py \
                         tests/kernels/attention/test_dflash2_context_kv_quantized.py
ruff check .. Passed / ruff format .. Passed / typos .. Passed / mypy .. Passed

$ pre-commit run mypy-3.10 --hook-stage manual --files <same files>
Run mypy for Python 3.10 .. Passed
```

## AI assistance

This change was prepared with AI assistance (Claude). I reviewed every
changed line, ran the tests and linters shown above on my own hardware, and
can defend the change end-to-end.
