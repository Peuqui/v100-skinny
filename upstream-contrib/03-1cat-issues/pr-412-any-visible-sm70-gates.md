# PR-Entwurf: 1CatAI/1Cat-vLLM — Fix #412 (SM70 gates probe device 0 only)

Stand 2026-09-05. Branch `pp-sm70-gates-any-visible-device`, Basis
origin/main 755baae (Tests liefen im Arbeitsbaum vor dem Branch-Wechsel; die
drei Dateien sind zwischen 755baae und e375cc3 identisch).
Status: committed als 35d2c69, gepusht nach fork Peuqui/1Cat-vLLM (Branch pp-sm70-gates-any-visible-device). PR eröffnet: https://github.com/1CatAI/1Cat-vLLM/pull/514
Diff: `pr-412-any-visible-sm70-gates.diff` (daneben) bzw. `git diff` + neue Datei im Checkout.

Geänderte Dateien (3):
- vllm/config/vllm.py — Helfer `_any_visible_device_is_capability()`, 11 SM70-Gates umgestellt (30+/13-)
- tests/config/test_prefix_anchored_swa.py — Fake-Plattform um `device_count` und `device_id` ergänzt
- tests/config/test_sm70_gates_any_visible_device.py — NEU, 9 Fälle

Scope-Entscheidung: Nur Punkt 1 des Issues (Env-Default-Gates auf Device 0).
Punkt 3 (NVFP4 get_min_capability hart 75) hat 1Cat am 28.08. in #403 selbst
gelöst (70 sobald TurboMind/Marlin/Emulation gewählt). Punkt 2 (Quant-Gate
get_device_capability() ohne device_id) bleibt unangetastet: mit Punkt 3
greift er in der Praxis nicht mehr, und ihn auf "irgendein Gerät" zu
lockern würde das Gate schwächen. Unser Fork lässt ihn ebenfalls stehen.

Vorgeschlagener PR-Titel:

    [Bugfix][SM70] Gate SM70 config defaults on any visible device (#412)

Vorgeschlagene Commit-Message:

    [Bugfix][SM70] Gate SM70 config defaults on any visible device

    VllmConfig.__post_init__ decides whether to apply the SM70 environment
    defaults (Flash-V100 baseline, FP8 MoE lanes, breakable cudagraph, GLM
    DFlash2 paths, ...) with current_platform.is_device_capability((7, 0)),
    which inspects device 0 only. On a heterogeneous pipeline-parallel rig
    with a Turing card first the whole block is skipped and the Volta stage
    runs without its tuning; with a Volta card first it is applied for both
    (#412). The same gate makes the prefix-anchored SWA contract reject a
    grid that has an SM70 device behind an SM75 one.

    Add _any_visible_device_is_capability() and gate those eleven sites on
    it. The defaults are re-gated per worker at their point of use, so
    applying them whenever an SM70 device is visible leaves the other
    stages untouched; homogeneous setups are unchanged.

    Tests build a real VllmConfig with device 0 simulated as SM75 and
    device 1 as SM70 (defaults must be applied) and with two SM75 devices
    (must stay clean), plus the helper and the prefix-anchored gate.

    Fixes #412

    Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
    Signed-off-by: Peuqui <peuqui@github.com>

---------------------------------------------------------------------------
PR-BODY
---------------------------------------------------------------------------

## Purpose

Fixes #412 (capability gates probe device 0 only), first item.

`VllmConfig.__post_init__` applies the SM70 environment defaults -- the
Flash-V100 baseline (`VLLM_SM70_GDN_*`, `VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE`,
...), the FP8 MoE lanes, breakable cudagraph, the GLM DFlash2 paths -- behind
`current_platform.is_device_capability((7, 0))`, which only inspects device 0.
On a heterogeneous pipeline-parallel deployment that is the wrong question:

- Turing card first (`CUDA_VISIBLE_DEVICES=0,2,1,3`, stage 0 = sm75): the whole
  block is skipped, the Volta stage runs without its tuning. On our rig
  (1.3.0 at the time of the issue) that showed up as 0 instead of 9
  "Auto-setting" lines and as MTP acceptance dropping from ~21 % to 2-9 %
  with incoherent output.
- Volta card first: applied for everyone, which is fine because every
  default is gated again per worker at its point of use.

The same gate makes `apply_prefix_anchored_swa_constraints` reject a grid
that has an SM70 device behind an SM75 one.

This PR adds `_any_visible_device_is_capability()` (module helper next to the
other SM70 default helpers; iterates `current_platform.device_count()` with
`is_device_capability(..., device_id=...)`) and gates the eleven
`is_device_capability((7, 0))` sites in `vllm/config/vllm.py` on it.
Homogeneous setups behave exactly as before. Sites that ask a different
question are left alone: the int-form fusion check, the Turing fp32
warning, the dense-cudagraph `cap.major == 7` check and the quantization
minimum-capability check (`get_device_capability()`), see below.

Not in this PR: item 3 of the issue (`ModelOptNvFp4Config.get_min_capability`
hard-coded to 75) was addressed in #403 (70 when a TurboMind/Marlin/emulation
backend is selected). With that, item 2 (the quantization gate reading device
0) no longer blocks a boot in practice, and widening it to "any visible
device" would weaken the check, so it stays as is.

`tests/config/test_prefix_anchored_swa.py` fakes `current_platform` with a
`SimpleNamespace`; the fake gains `device_count` and the `device_id`
parameter the helper uses. The assertions are unchanged.

Duplicate check (per AGENTS.md): `gh issue view 412 --comments` (no
follow-up), `gh pr list --state open --search "412 in:body"` (none),
`--search "is_device_capability"` and `--search "heterogeneous"` (no PR
touches these gates). Verified against origin/main 755baae today.

## Test Plan

Environment: 1Cat-vLLM 1.5.0 wheel venv (Python 3.12), checkout at
origin/main 755baae with the compiled extensions linked in, single Tesla
V100 (`CUDA_VISIBLE_DEVICES=4`), facebook/opt-125m from the HF cache.

    # every test file that exercises these gates, plus the new file
    python -m pytest tests/config/test_prefix_anchored_swa.py tests/compile/test_sm70_decode_graph.py \
        tests/engine/test_arg_utils.py tests/v1/spec_decode/test_dflash2.py \
        tests/config/test_sm70_gates_any_visible_device.py -q
    python -m pytest tests/v1/spec_decode/test_mtp.py -q

    # new file + prefix-anchored file with vllm/config/vllm.py reverted to origin/main (git stash), tests kept
    python -m pytest tests/config/test_sm70_gates_any_visible_device.py tests/config/test_prefix_anchored_swa.py -q

    # pre-commit hooks as configured in the repo (ruff, typos, mypy-local) plus the CI-only mypy stage
    pre-commit run --files <3 files>
    pre-commit run mypy-3.10 --hook-stage manual --files <3 files>
    mypy --python-version 3.10 --follow-imports skip tests/config/test_sm70_gates_any_visible_device.py tests/config/test_prefix_anchored_swa.py

## Test Result

Four affected test files on origin/main 755baae, before the change:

    264 passed, 9 skipped in 101.58s

Same four files plus the new file, with the fix (9 new cases):

    273 passed, 9 skipped in 101.33s

`tests/v1/spec_decode/test_mtp.py` (patches `is_device_capability` and builds a
VllmConfig): 8 passed before and after.

New file + prefix-anchored file, `vllm/config/vllm.py` reverted to origin/main:

    8 failed, 15 passed
    test_sm70_baseline_defaults_follow_any_visible_device[sm70-stage-behind-sm75]
        assert (set() == {'VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE', ...})   -> no defaults applied
    test_prefix_anchored_swa_accepts_sm70_behind_sm75
        ValueError: prefix_anchored_decode_window requires an NVIDIA SM70 GPU  -> grid rejected
    test_any_visible_device_is_capability[*] (6 cases)   ImportError (helper does not exist)
    [homogeneous-sm75] passes before and after: no behaviour change for homogeneous boxes.

pre-commit (`pre-commit run --files <3 files>` and `pre-commit run mypy-3.10
--hook-stage manual --files <3 files>`): ruff-check, ruff-format, typos, SPDX
headers, forbidden-imports, torch.cuda-call check, mypy-local and mypy-3.10
all Passed.
mypy on `vllm/config/vllm.py`: no issues before and after.
mypy on both test files: no issues.

On our rig (2x RTX 8000 sm75 + 3x V100 sm70) the same change has been in
production in our fork since the 1.3.0 days; with the RTX pair as PP stage 0
the Volta stage boots with all 14 SM70 defaults, and TP2xPP2 with MTP runs
coherently at the same throughput as before.

AI assistance: this change was developed with Claude (Anthropic) as a
coding assistant. Every changed line was reviewed by me and the test runs
above were executed on my hardware; I can defend the change end to end.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
