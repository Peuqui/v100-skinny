# PR-Entwurf: DFlash2-BF16-Emulation für alles unter SM80, am Worker-Gerät entschieden

Status: ENTWURF, nicht gesendet. Branch `dflash2-pre-sm80-worker-device` im
Worktree `1Cat-vLLM-pr-dflash2`, Basis origin/main fe67339d. Teil des Pakets
„gemischte Hardware" (Punkt 11, Freigabe Peuqui 2026-09-10 für die Aufbereitung;
Commit, Push und Eröffnen erst auf Ansage).

Review 2026-09-11, Lücke: `has_device_capability(…, device_id=
torch.accelerator.current_device_index())` fragt NVML (PCI-Reihenfolge) mit
einem torch-Index (CUDA-Reihenfolge). Ohne `CUDA_DEVICE_ORDER=PCI_BUS_ID` fragt
das Gate auf einem gemischten Rechner womöglich wieder die falsche Karte — genau
den Fehler, den der PR behebt. vLLM warnt selbst davor (`cuda.py`,
`log_warnings`). Vorschlag: Vorbedingung im Text nennen (betrifft auch #576).
Der E2E-Beleg hängt an #572 (offen, ohne Review) — zurückstellen, bis sich dort
etwas tut.

Titel:

    [Bugfix][Spec Decode][SM70] Run DFlash2's BF16 emulation on every pre-SM80 worker

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

## Test Plan

1. `pre-commit run --files <both files>` and
   `pre-commit run mypy-3.10 --hook-stage manual --files <both files>`.
2. New CPU-only test `tests/v1/spec_decode/test_dflash2_pre_ampere_gate.py`
   (7 cases; a mocked node with Turing, Volta and Ampere at different
   indices), run with no visible GPU; counter-test with the fix reverted.
3. Existing DFlash2 tests: `tests/v1/spec_decode/test_dflash2*.py`.
4. End to end on Turing: Qwen3.8-27B NVFP4, TP2 on 2x Quadro RTX 8000,
   DFlash2 k=7 with the BF16 draft head, greedy; acceptance length and decode
   rate with and without the fix. Turing needs #572 (open) to boot, so both
   sides run on `main` + #572.

## Test Result

1. All applicable hooks passed; mypy-3.10 passed.
2. `CUDA_VISIBLE_DEVICES="" pytest tests/v1/spec_decode/test_dflash2_pre_ampere_gate.py`:
   7 passed. With the fix reverted: 3 failed (the Volta worker, the Turing
   worker, and the Turing worker behind an Ampere device 0), 4 passed.
3. (auf freien GPUs nachholen: zwei CUDA-Fälle von
   `test_sm70_dflash2_exact_rerank_matches_gathered_bmm` liefen beim ersten
   Versuch neben einem belegten Modell und sind nicht aussagekräftig)
4. (Messung main + #572, mit und ohne Fix, nachtragen)

## Not a duplicate

Checked on 2026-09-10 against `1CatAI/1Cat-vLLM` (`gh pr list --state all`
for "dflash2", "bf16 emulation", "dflash sm75", "dflash turing"; issues for
"dflash2 turing"). The open DFlash2 PRs (#561, #405) and #435 do not touch
`_use_sm70_bf16_emulation` or its capability check (`gh pr diff`). #576 is our
own sibling change for the quantization gate and uses the same device query.

AI assistance (Claude) was used to trace the failure and prepare the change;
I reviewed every line and ran the tests above.
