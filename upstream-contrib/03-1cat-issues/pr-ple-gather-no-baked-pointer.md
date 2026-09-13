# PR-Entwurf 1Cat: PLE-Gather ohne gebackenen Zeiger — VERÖFFENTLICHT als #622 (13.09.)

Worktree `1Cat-vLLM-pr-plegather`, Branch `ple-gather-no-baked-pointer` auf origin/main 7217bb5d.
Im Fork: work-main (uncommitted, 13.09. ~16:30), im Betrieb belegt (Flash-Next kalt/warm1/warm2).
Duplikatsprüfung 13.09.: keine offene PR/Issue zu PLE-Zeiger, pinned_gather, AOT-Absturz.
Voraussetzung im Text nennen: mit #621 (Compile-Cache an) ist dieser Fehler für jeden
Flash-Next-Nutzer sichtbar; ohne #621 nur mit VLLM_DISABLE_COMPILE_CACHE=0.

Titel: [Bugfix][Qwen4Exp] Resolve the PLE table pointer inside the gather op instead of baking it into the graph

---

## Purpose

`qwen4_exp_ple_pinned_gather` took the address of the PLE table as a Python `int`
(`weight_ptr`), the device table's `data_ptr()` or the UVA view of the pinned host
half. Inside a compiled region that integer is a constant, and Inductor writes it
verbatim into the generated code. The AOT artifact of the first pipeline stage of
Qwen3.8-Flash-Next therefore contains a literal host address:

```
torch.ops.vllm.qwen4_exp_ple_pinned_gather.default(buf1, ..., arg2_1, 136921242140672, 160)
```

A cold boot works, because the constant comes from the same process. The first warm
boot with the compile cache on (`Directly load AOT compilation`) runs the artifact in a
new process, whose pinned buffer sits at a different address. Both ranks of the stage
that holds the PLE table die in `profile_run` with `CUDA error: an illegal memory
access was encountered`; the other stage, which has no PLE layer, loads and runs. With
`VLLM_DISABLE_COMPILE_CACHE=1` the same command boots. The forced compile-cache opt-out
on the SM70 graph hid this since #403 introduced the pointer argument; #528 added the
device-table pointer the same way. #621 removes that opt-out, so this fix should land
first or together with it.

## What changes

- `qwen4_exp_ple_pinned_gather(input_ids, output, weight_scale, layer_name: str,
  use_host_table: bool, embedding_dim: int)`: the op resolves the table by name
  through `get_forward_context().no_compile_layers` and reads the pointer at run time,
  the way `qwen4_exp_compute_ple_ngram_ids` already resolves its layer. Only the string
  and the bool reach the graph; both are stable across processes.
- `Qwen4ExpPinnedHostEmbedding.__init__` registers the table under its prefix in the
  config's `static_forward_context` (duplicate prefixes raise, as for the PLE layer).
- `embedding_lookup` passes `(layer_name, use_host_table)` at its three call sites; the
  cached pointers stay where they are and keep being refreshed by
  `get_accelerator_weight` / `materialize_tables`, so CUDA-graph replay after an
  in-place reload is unchanged (existing test).
- `tests/models/qwen4_exp/test_ple.py`: `_pinned_layer` constructs under a
  `VllmConfig` context; the gather test exposes the table to the op through a
  monkeypatched forward context, mirroring the existing ngram-id test.

## Reproduction and evidence

Rig: 2x Quadro RTX 8000 (stage 0, holds the PLE table, 6 GiB of it pinned in host
memory via `VLLM_QWEN4EXP_PLE_HOST_GIB=6`) + 2x Tesla V100 (stage 1), TP2 PP2,
Qwen3.8-Flash-Next-180B-A4B-NVFP4 with MTP k=4, compile cache on, torch 2.10.0+cu128.

| boot | before this change | with this change |
|---|---|---|
| cold | ready, writes artifacts | ready, writes artifacts |
| first warm | stage-0 ranks: `Directly load AOT` → `profile_run` → `_short_conv_fallback` / `qwen4_exp_ple_pinned_gather` → illegal memory access (`CUDA_LAUNCH_BLOCKING=1` points at the gather's Triton launch) | 6 artifacts loaded, ready after 312 s, answer identical to cold (SHA-256 `e948a82ead51948c`) |
| second warm | not reached | 6 artifacts loaded, ready after 343 s, same answer (SHA-256 `e948a82ead51948c`) |

Same command with `VLLM_DISABLE_COMPILE_CACHE=1`: boots and answers before the change
(350 s), which isolates the artifact reload as the trigger.

## Test Plan

1. `pre-commit run --files vllm/models/qwen4_exp/nvidia/ple_layer.py tests/models/qwen4_exp/test_ple.py` and `pre-commit run mypy-3.10 --hook-stage manual --files <same>`.
2. `tests/models/qwen4_exp/test_ple.py` in full on a V100 (the gather test needs pinned memory and CUDA graphs).
3. The boot matrix above (cold, warm, warm) on the rig.

## Test Result

1. pre-commit: all hooks passed; mypy: passed.
2. `tests/models/qwen4_exp/test_ple.py`: 60 passed (one V100 visible), on this branch and on our fork.
3. Boot matrix: as in the table.

## Not a duplicate

`gh pr list --state open --search "PLE pointer OR pinned_gather OR data_ptr"` and
`gh issue list --state all --search "illegal memory access PLE OR AOT compile pointer"`
show nothing on this; #621 (ours) removes the opt-out that hid the bug and does not
touch the PLE layer.

AI assistance (Claude) was used to find the cause and prepare the change; I reviewed
every line and ran the boots and tests on the hardware named above.
