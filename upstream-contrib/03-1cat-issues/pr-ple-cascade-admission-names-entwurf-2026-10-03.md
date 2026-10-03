# PR-Entwurf: PLE cascade admission under checkpoint layer names (03.10.2026)

Branch `pr-ple-cascade-admission-names` (Fork), Basis 1Cat main 62469c58f, Commits 4114c9e2c (Fix +
Regressionstest) und 8402e2a51 (Testkorrektur Umgebung). Go von Peuqui für den PR: 03.10. mittags,
nach bestandener Abnahme im echten Boot. Platzhalter <...> erst aus den Messdateien füllen.

Titel: [Bugfix][Qwen4Exp] Look up PLE cascade storage under checkpoint layer names

---

## Purpose

The disk cascade admission from #806 (`_qwen4exp_ple_cascade_requested`) asks the quantization config whether every PLE table is stored as raw E4M3. It builds the lookup prefix as `model.layers.{id}.ple.ple_embedding.ngram_embedding`, which misses in two ways:

- `ple_layer_ids` are 1-based: id L is the PLE module of decoder layer L - 1, as `check_ple_layers_on_first_pp_rank` already documents. The query names the wrong decoder layer.
- The model's `hf_to_vllm_mapper` is applied to the quantization config only when the model is built. During admission the layer metadata still uses checkpoint names (`model.language_model.layers.N...`), which the vLLM-side prefix never matches.

ModelOpt mixed-precision checkpoints declare the FP8 table per layer in `quantized_layers` instead of setting `ple_embedding_dtype`. On nvidia/Qwen3.8-Flash-Next-NVFP4 (`ple_layer_ids` [2], table under `model.language_model.layers.1.ple.ple_embedding.ngram_embedding`) the admission therefore always reports "checkpoint metadata does not provide raw E4M3 PLE storage", and the engine materializes the whole table on the first pipeline stage. On main a692497bc a PP4 start failed with

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 47.69 GiB. GPU 0 has a total capacity of 47.27 GiB
```

and TP2xPP2 stopped with "available KV cache memory (0.03 GiB)".

This PR queries the checkpoint name of decoder layer id - 1. The existing admission tests only use the forced E4M3 storage path (`ple_embedding_dtype="float8_e4m3fn"`), where the prefix is never evaluated; the new test uses ModelOpt metadata shaped like that checkpoint and also rejects a table registered under the raw id.

A second commit fixes `test_qwen4exp_ple_cascade_starts_the_offload_worker`, which fails on any host with a visible SM70 GPU, also on unchanged main: `VllmConfig()` itself applies the Flash-V100 baseline defaults (`VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE` and the SM70 GDN schedules) after the test took its environment snapshot. The snapshot is now taken after the config is built, which is what the test checks.

Not a duplicate: no open PR touches the admission's storage lookup (gh pr list / gh search for "raw E4M3 PLE storage", "ple cascade admission", "ple_layer_ids"; #702, #821 and #831 change other parts of the PLE configuration).

AI assistance was used for this change. I reviewed every line and ran the tests below.

## Test Plan

```bash
pytest tests/config/test_ple_cascade_capabilities.py
pytest tests/compile/test_sm70_decode_graph.py
pre-commit run --files vllm/config/vllm.py tests/config/test_ple_cascade_capabilities.py tests/compile/test_sm70_decode_graph.py
pre-commit run mypy-3.10 --hook-stage manual --files <same>
```

## Acceleration and benchmark contract (required for performance changes)

Not a performance change. It changes which placement the existing automatic admission selects for ModelOpt checkpoints with FP8 PLE tables.

## Test Result

Both files: 62 passed on a Tesla V100 host and with no visible GPU. On unchanged main the new test fails for the correctly registered table (mutation check), and a variant that only fixes the name, not the layer index, fails both new cases. Without the second commit, `test_qwen4exp_ple_cascade_starts_the_offload_worker` fails on the V100 host and passes without a GPU. pre-commit and mypy-3.10 pass.

Admission on the real checkpoint, building `VllmConfig` from the serving arguments without loading weights: main reports `ple_disk_cascade_active=False` with the reason above for PP4 and TP2xPP2; with this change both report `True` with no reason.

End to end on 3x Tesla V100 + 2x Quadro RTX 8000 (<Baum, Datum>): <Ergebnis PP4/TP2xPP2 aus ab_837-Log: Start, KV-Cache, Greedy gegen Produktion, Prefill/Decode>.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
