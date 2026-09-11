# PR-Entwurf: DFlash2-BF16-Emulation für alles unter SM80, am Worker-Gerät entschieden

Status: GESENDET 2026-09-11 abends als https://github.com/1CatAI/1Cat-vLLM/pull/599 (Commit 90c9efee). Vorher: Branch `dflash2-pre-sm80-worker-device` im
Worktree `1Cat-vLLM-pr-dflash2`, Basis origin/main fe67339d. Teil des Pakets
„gemischte Hardware" (Punkt 11, Freigabe Peuqui 2026-09-10 für die Aufbereitung;
Commit, Push und Eröffnen erst auf Ansage).

Review 2026-09-11 vormittags, Lücke: `has_device_capability(…, device_id=
torch.accelerator.current_device_index())` fragt NVML (PCI-Reihenfolge) mit
einem torch-Index (CUDA-Reihenfolge).

**Geklärt 2026-09-11 nachmittags, kein Blocker:** `NvmlCudaPlatform.
get_device_capability(device_id)` nimmt `CUDA_VISIBLE_DEVICES[device_id]` und
deutet den Eintrag als NVML-Index (`device_id_to_physical_device_id`,
`vllm/platforms/cuda.py` ~643 und `interface.py` ~227); torch zählt in
CUDA-Reihenfolge. Beides deckt sich nur mit `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
Das ist vLLM-weites Verhalten für JEDEN Aufruf mit Geräteindex, steckt
genauso im gemergten #514 und im offenen #576, und vLLM warnt beim Start
selbst (`NvmlCudaPlatform.log_warnings`, `cuda.py` ~872: „Detected different
devices in the system … make sure to set `CUDA_DEVICE_ORDER=PCI_BUS_ID`").
Folge: als Vorbedingung in den Purpose-Text (unten eingefügt), nicht als
eigener Fix. Der E2E-Beleg hängt an #572 (offen, ohne Review): messen lokal
mit #572 obendrauf (Worktree `1Cat-vLLM-e2e-572` = fe67339d + Merge
`fork/sm75-gdn-prefill-route`, Extensions aus dem Belegbau verlinkt, Skript
`scratchpad/abnahme2/e2e_11a.sh`). **Entscheidung Peuqui 11.09. nachmittags:
NICHT auf den Merge von #572 warten — sofort nach der Messung senden und
#572 im Text als Voraussetzung für den Turing-Beleg nennen.**

Nachtrag 11.09. abends: das zweite Gerät-0-Gate `_flashinfer_topk()` ist
auf Peuquis Entscheidung („mach das mit rein") in diesen PR aufgenommen —
Hunk in `qwen3_dflash2.py`, eigener Testfall
`test_flashinfer_topk_gate_follows_the_worker_device`, Mock-Anpassung im
bestehenden `test_flashinfer_topk_is_capability_gated_on_sm70`. Commit
`90c9efee` auf `dflash2-pre-sm80-worker-device`.

Titel:

    [Bugfix][Spec Decode][SM70] Gate DFlash2's BF16 emulation and FlashInfer top-k on the worker's own device

---

## Purpose

`_use_sm70_bf16_emulation` routes a BF16 DFlash2 draft checkpoint through the
range-preserving FP16 path (`dflash_sm70`). It currently fires only when
`current_platform.is_device_capability(70)` is true, which has two problems:

1. The criterion is one architecture number, but the reason for the path is a
   missing capability: native BF16 arithmetic arrives with SM80. Turing (SM75)
   runs the draft in FP16 exactly like Volta and needs the same path. Without
   it the draft's activations leave the FP16 range and acceptance collapses.
2. The call asks device 0 of the visibility list, not the device the worker
   builds on. On a node that mixes generations this answers for another card,
   in both directions: with a Turing card first, Volta workers lose the path
   too.

The gate now asks `current_platform.has_device_capability(80, device_id=...)`
for the worker's own device (`torch.accelerator.current_device_index()`, as in
#576) and emulates wherever the answer is no. Ampere and newer are unchanged;
the `VLLM_SM70_DFLASH2_BF16_EMULATION` switch is unchanged.

The same file has a second gate of the same shape: `_flashinfer_topk()`
decides once per process (`@cache`) whether the DFlash2 selector uses
FlashInfer's radix top-k or `torch.topk`, and it asked
`has_device_capability(80)` — device 0 again. With an Ampere card listed
first, a Turing or Volta worker would pick the FlashInfer kernel, which has
no pre-SM80 implementation. It now asks the worker's own device as well; the
per-process cache is per worker, so the answer stays correct.

Precondition, shared with every per-device capability query in vLLM (#514,
#576 included): the platform resolves `device_id` through
`CUDA_VISIBLE_DEVICES` to an NVML index, i.e. PCI bus order, while torch
numbers devices in CUDA order. The two agree only with
`CUDA_DEVICE_ORDER=PCI_BUS_ID`, which vLLM already asks for at startup on
nodes with mixed device names (`NvmlCudaPlatform.log_warnings`). This change
does not alter that contract; it only stops asking device 0.

The existing switch test
`test_sm70_dflash2_bf16_emulation_has_explicit_ab_switch` mocked the old
query (`is_device_capability`); on a CPU-only runner the new gate would then
reach `torch.accelerator.current_device_index()` and fail with "No CUDA GPUs
are available". Its mocks now describe a Volta worker through the new query
(`has_device_capability` and `current_device_index`), so it keeps testing the
switch and nothing else.

## Test Plan

1. `pre-commit run --files <three files>` and
   `pre-commit run mypy-3.10 --hook-stage manual --files <three files>`.
2. New CPU-only test `tests/v1/spec_decode/test_dflash2_pre_ampere_gate.py`
   (8 cases; a mocked node with Turing, Volta and Ampere at different
   indices, both gates), run with no visible GPU; counter-test with the fix
   reverted.
3. The complete existing file `tests/v1/spec_decode/test_dflash2.py` on CPU,
   plus its two CUDA cases of
   `test_sm70_dflash2_exact_rerank_matches_gathered_bmm` on an otherwise
   idle Tesla V100, with and without the fix.
4. End to end on Turing: Qwen3.8-27B NVFP4, TP2 on 2x Quadro RTX 8000,
   DFlash2 k=7 with the BF16 draft head, greedy; acceptance length and decode
   rate with and without the fix. Turing does not boot on `main` without
   #572 (open), so both sides run on `main` + #572; #572 is the prerequisite
   for the Turing evidence, not for the change itself.

## Test Result

1. All applicable hooks passed; mypy-3.10 passed.
2. `CUDA_VISIBLE_DEVICES="" pytest tests/v1/spec_decode/test_dflash2_pre_ampere_gate.py`:
   8 passed. With the fix reverted: 4 failed (the Volta worker, the Turing
   worker, the Turing worker behind an Ampere device 0, and the FlashInfer
   top-k gate for a Turing worker behind an Ampere device 0), 4 passed.
3. `CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 pytest tests/v1/spec_decode/test_dflash2.py tests/v1/spec_decode/test_dflash2_pre_ampere_gate.py`:
   166 passed, 13 skipped (the CUDA cases). With the fix reverted: 5 failed —
   the four above and the switch test whose mocks now follow the new query.
   Same result for the gate file on a clean `main` + #572 checkout (8 passed
   with the fix, 4 failed without). The three CUDA cases of
   `test_sm70_dflash2_exact_rerank_matches_gathered_bmm` on a Tesla V100
   (`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4`, `main` + #572):
   3 passed without the fix, 3 passed with it — the kernel is SM70-only and
   the gate does not touch it.
4. Turing end to end on `main` + #572 is NOT reachable today: the NVFP4
   target (`modelopt_mixed`) is refused there with "Minimum capability: 89.
   Current capability: 75" (`VllmConfig` validation, both arms, 2026-09-11).
   The Turing numbers therefore come from our fork build of the same code
   (1Cat main + this change + the fork's NVFP4 routes for Turing), Qwen3.8-27B
   NVFP4 target, `incoai/Qwen3.8-27B-DFlash2` BF16 draft, TP2 on 2x Quadro
   RTX 8000, k=7, greedy, 400 tokens, 5 runs: acceptance length **1.015 →
   3.353** and **21.35 → 69.13 tok/s** with the fix (2026-09-09); the target
   text is byte-identical in both arms, as the verifier decides every token.
   On 2x Tesla V100 the change is a no-op by construction (`is 70` and
   `< 80` agree) and the same run gives 74.09 tok/s before and after.

## Not a duplicate

Checked on 2026-09-11 against `1CatAI/1Cat-vLLM` (`gh pr list --state open
--search` for "dflash2 turing", "bf16 emulation", "dflash sm75", "pre-ampere
dflash", "_use_sm70_bf16_emulation", "worker device capability"; issues for
"dflash2 turing" and "sm75"; plus `gh pr diff --name-only` over every open PR):
no open PR modifies `vllm/model_executor/models/qwen3_dflash2.py`. The search
hits are #572 and #576 (our own, prerequisite and sibling), #435 (csrc build
fixes, no Python gate), #239 and #523 (unrelated areas). #576 uses the same
device query for the quantization gate.

AI assistance (Claude) was used to trace the failure and prepare the change;
I reviewed every line and ran the tests above.
