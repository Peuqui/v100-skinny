# Entwurf: Issue an 1CatAI/1Cat-vLLM — Qwen4Exp unter PP (Paket-1-Auftakt)

> Status: GEPOSTET 2026-09-03 (Peuqui-Go) als
> https://github.com/1CatAI/1Cat-vLLM/issues/479 Alle drei Fundstellen am
> 2026-09-03 abends gegen origin/main verifiziert (Zeilenstände: Loader
> skip_substrs ohne PP-Fall; gpu_model_runner.py:1409 PLE-PP-Gate;
> gpu_worker.py:212 PLE-Offload-Gate). Zahlen: FLASH-NEXT-OPERATING-POINT.md.

---

**Titel:** Qwen4Exp under pipeline parallelism: two gates and one loader
crash — PP enablement offer

Hi — following up on the pre-Ampere work from #469 and #441: we serve
Qwen3.8-Flash-Next on a heterogeneous 5-GPU box (2x RTX 8000 sm75 +
3x V100 sm70) where TP4 is not an option (mixed archs, mixed VRAM), so
pipeline parallelism is the only way to fit the model. On current main,
Qwen4Exp cannot boot under PP; we hit three separate walls
(RadixArk/Qwen3.8-Flash-Next-NVFP4, TP2/PP2, `--language-model-only`):

1. `VLLM_PLE_CPU_OFFLOAD` refuses PP outright ("Unsupported settings:
   PP=2", gpu_worker.py) — an intentional gate, noted for completeness.
2. Without offload, the PLE embedding refuses PP as well ("N-gram PLE
   embedding requires pipeline_parallel_size=1", gpu_model_runner.py) —
   also an intentional gate: non-first ranks never receive input ids.
3. Behind those gates there is a real loader bug: `hyper_connection_mixer`
   is only instantiated on the last PP rank (nvidia/model.py, `if
   get_pp_group().is_last_rank`), but `load_weights` only skips
   `hyper_connection_mixer.block_inject_weight` — the remaining mixer
   tensors crash rank 0 with "There is no module or parameter named
   'hyper_connection_mixer' in Qwen4ExpModel". Three-line fix: extend
   `skip_substrs` with `"hyper_connection_mixer."` on non-last ranks.

We have PP working for this model family in our fork (based on your
1.3.0): input-id transport to non-first ranks plus a PLE VRAM->host
cascade, serving Qwen3.8-Flash-Next at TP2/PP2 with MTP k=4 at
~52 tok/s on the box above. If PP support is something you would take,
we would upstream it as a small series against main, starting with the
loader fix (with a weight-loading test alongside your existing
tests/models/qwen4_exp/test_weight_loading.py), followed by the
input-id transport and the PLE cascade as separate reviewable steps.
Happy to adjust scope and shape to whatever fits your roadmap.

---

## Nicht behauptet (bewusst weggelassen)
- Kein Urteil ueber deren Gate-Entscheidungen (als "intentional" markiert).
- Keine 1.5.0-vs-1.3.0-Vergleichszahlen (unser Rebase laeuft noch).
