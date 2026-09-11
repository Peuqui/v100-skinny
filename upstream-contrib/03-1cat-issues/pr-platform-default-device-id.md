# PR-Entwurf: Plattform — unbestimmter Geräteindex = aktuelles Gerät des Workers

Status: GESENDET 2026-09-11 abends als https://github.com/1CatAI/1Cat-vLLM/pull/600 (Commit 2b59521a). Vorher: Branch `platform-default-device-id-current`
im Worktree `1Cat-vLLM-pr-devcap`, Basis origin/main fe67339d. Entscheidung
Peuqui 11.09. abends („Punkt 2"): der Wurzelfix für die Gerät-0-Fehlerklasse,
statt 263 Einzelstellen zu fixen. Commit, Push und Eröffnen erst auf Ansage;
Peuqui liest alle Zeilen (AGENTS.md).

Umfang: `vllm/platforms/interface.py` (+27/−8), `vllm/platforms/cuda.py`
(+51/−4), `vllm/platforms/rocm.py` (+7/−1), `vllm/platforms/xpu.py` (+1/−1),
neu `tests/platforms/test_capability_default_device.py` (11 Tests).

Review Peuqui + Selbstprüfung 11.09. abends, drei Befunde, alle eingearbeitet:
(1) torch-Pfad warf bei ungültigem Index `AssertionError`, NVML lieferte
`None` — jetzt gleicher Vertrag, mit Test und Probe auf GPU 4; (2) Kosten je
Aufruf gemessen (V100, nach `set_device`, 20.000 Aufrufe): `is(70)` 0,36 →
0,84 µs, `has(80)` 2,84 → 3,41 µs, die halbe Mikrosekunde ist
`torch.cuda.current_device()`; (3) `get_device_name/uuid/total_memory`,
`is_integrated_gpu`, `num_compute_units` behalten Vorgabe 0 — bewusst, nur
die Capability-Abfragen wählen Kernel. AGENTS.md frisch gelesen und
abgeglichen; Anker-Issue ist #412 (Peuqui, geschlossen 07.09.).

Verwandt: vLLM-Upstream #53834 (puririshi98/NVIDIA, 26.08., GESCHLOSSEN ohne
Review) hatte den torch-nach-Init-Teil für die DGX Station (GB300 + Display-
GPU, 10,7× Regression durch FASTEST_FIRST). Nicht gelandet, vLLM-main fragt
weiter NVML mit Vorgabe 0. Unser PR übernimmt die Idee und ergänzt den
Vorgabewert „aktuelles Gerät".

Hardware-Beleg: `scratchpad/abnahme2/devcap_probe.sh main|fix` auf GPU 0+1
(RTX 8000 + V100), mit und ohne `CUDA_DEVICE_ORDER` — NACH der Messkette
fahren und unten eintragen.

Bekannte, vorbestehende rote Tests (ohne Fix identisch): 3 in
`tests/config/test_sm70_gates_any_visible_device.py` (brauchen
`facebook/opt-125m` online), 3 in `tests/cuda/test_cuda_context.py` (brauchen
eine sichtbare GPU).

Titel:

    [Bugfix][Platform] Resolve an unspecified device_id to the worker's own device

---

## Purpose

Generalizes #412. That issue named three device-0 capability gates; they
were fixed one by one (#514, #403, #576) and the issue closed with the
second item left as is. This change fixes the class instead of the
instances.

Every capability query on `current_platform` — `get_device_capability`,
`has_device_capability`, `is_device_capability`,
`is_device_capability_family` — defaults to `device_id=0`, which
`NvmlCudaPlatform` maps to index 0 of `CUDA_VISIBLE_DEVICES`. On a node that
mixes card generations, every worker except the one on the first listed card
therefore gates its kernels on somebody else's capability. #514 and #576 fixed
two instances by passing the worker's device explicitly; a count on today's
`main` finds 263 call sites that still query without a device index, two of
them inside `_flashinfer_topk` and `_use_sm70_bf16_emulation` (fixed in the
sibling PR), the rest spread over quantization, MoE, GDN, attention and the
model runners. Patching them one by one is not a plan.

This change moves the decision to the one place all four queries share:

- `device_id` now defaults to `None`, meaning "the device this process runs
  on". `Platform.resolve_device_id()` turns it into a concrete index. The base
  platform keeps the historical answer (0). The CUDA platform answers with
  `torch.cuda.current_device()` once the process holds a CUDA context and 0
  before that, so the engine core, the API server and a worker before
  `set_device` behave exactly as today; a worker after `init_device` answers
  for its own card. `torch.cuda.is_initialized()` only reports and never
  creates a context.
- Explicit indices are untouched: `has_device_capability(80, device_id=i)`
  still asks device `i`.
- Once CUDA is initialized, `NvmlCudaPlatform` reads the capability from
  `torch.cuda.get_device_capability(ordinal)` instead of NVML. NVML counts in
  PCI bus order, the CUDA runtime defaults to `FASTEST_FIRST`; on a mixed node
  the two orders differ unless `CUDA_DEVICE_ORDER=PCI_BUS_ID` is set, which is
  exactly the situation `log_warnings` warns about. torch is authoritative for
  the ordinal, and reading it after initialization cannot trigger the
  initialization the NVML path exists to avoid. NVML remains the pre-init
  path. (Same approach as vllm-project/vllm#53834, which did not land.)
- The caches are keyed by the resolved index (`_torch_device_capability`,
  `_nvml_device_capability`, ROCm `_device_capability`), never by `None`, so an
  unspecified device cannot freeze one process-wide answer.

Signatures of the ROCm and XPU overrides are widened to match (`int | None`),
their behaviour is unchanged (base resolution, index 0). The torch path keeps
the NVML contract for an unusable index: it answers `None` (torch raises
`AssertionError` there), so `has_device_capability(80, device_id=7)` on a
five-card node is `False`, not an exception.

Cost: resolving the current device adds about half a microsecond per query
(Tesla V100, after `set_device`, 20,000 calls: `is_device_capability(70)`
0.36 → 0.84 µs, `has_device_capability(80)` 2.84 → 3.41 µs). The queries
that sit in forward paths run on the order of a hundred times per decode
step, i.e. well under 0.1 % of a 30 ms step.

Not covered, on purpose: `get_device_name`, `get_device_uuid`,
`get_device_total_memory`, `is_integrated_gpu` and `num_compute_units` keep
their `device_id=0` default; only the capability queries select kernels.
Module-level constants evaluated at import time in a worker before
`set_device` still see index 0; they need a different fix (lazy evaluation)
and are out of scope here.

## Test Plan

1. `pre-commit run --files vllm/platforms/{interface,cuda,rocm,xpu}.py tests/platforms/test_capability_default_device.py`
   and the same with `mypy-3.10 --hook-stage manual`, plus mypy over
   `vllm/platforms/*.py` (override compatibility).
2. New CPU-only `tests/platforms/test_capability_default_device.py`: a mocked
   node (Turing at 0, Volta at 1, Ampere at 2); resolution with and without a
   CUDA context, explicit index wins, all three predicate queries follow the
   current device, torch answers after init and NVML is left alone, the
   caches see the resolved index, an unusable index answers `None`, the base
   platform keeps index 0.
3. Existing gate tests that go through the real interface:
   `tests/config/test_sm70_gates_any_visible_device.py` (#514) and
   `tests/cuda/`.
4. Hardware probe on the mixed node (2x Quadro RTX 8000 + 3x Tesla V100):
   `CUDA_VISIBLE_DEVICES=0,1` (RTX first), one process does
   `torch.cuda.set_device(1)` like a worker with local rank 1 and then asks
   `get_device_capability()` without an index — on `main` and with this
   change, once with `CUDA_DEVICE_ORDER=PCI_BUS_ID` and once without.

## Test Result

1. All applicable hooks passed; mypy-3.10 passed on the five files and on
   `vllm/platforms/*.py`.
2. 11 passed.
3. 48 passed, 6 failed — the six fail identically without this change:
   three parametrizations of `test_sm70_baseline_defaults_follow_any_visible_device`
   need `facebook/opt-125m` from the Hub (run offline), three
   `TestSetCudaContext` cases need a visible GPU (run with none).
4. Probe on the mixed node, `CUDA_VISIBLE_DEVICES=0,1` (Quadro RTX 8000
   first, Tesla V100 second), one process, `torch.cuda.set_device(1)`:

   | | before `set_device` | after `set_device(1)`, a V100 |
   |---|---|---|
   | `main` (fe67339d + #572) | `(7, 5)` | `(7, 5)`, `has(75)=True`, `is(70)=False` |
   | this change | `(7, 5)` | `(7, 0)`, `has(75)=False`, `is(70)=True` |

   Identical with `CUDA_DEVICE_ORDER=PCI_BUS_ID` and with the default
   ordering. `main` keeps answering for the first listed card no matter which
   device the process selected; with this change the process answers for its
   own card, and before device selection nothing changes.
5. Per-call cost, Tesla V100 after `set_device`, 20,000 calls each:
   `is_device_capability(70)` 0.36 → 0.84 µs, `has_device_capability(80)`
   2.84 → 3.41 µs.

## Not a duplicate

Checked on 2026-09-11 against `1CatAI/1Cat-vLLM` (`gh issue view 412
--comments`; `gh pr list --state open --search "412 in:body"` → #572 and
#594, neither touches the platform queries; `gh pr list --state open
--search` for "device_id default", "current device capability",
"resolve_device_id", "device 0 capability", "worker device capability";
issues for "device 0 capability"; `gh pr diff --name-only` over every open
PR for `vllm/platforms/{interface,cuda}.py`). Open PRs touching `cuda.py` are
#431 (custom all-reduce on Volta) and #435 (build fixes); both add a new
`has_device_capability(70)` call and neither changes the query itself. #514
(merged) and #576 (open) are the per-site fixes this change generalizes;
their explicit `device_id` arguments keep working unchanged. Upstream:
vllm-project/vllm#53834 (closed, not merged) covers the torch-after-init part
only.

AI assistance (Claude) was used to trace the failure class, count the call
sites and prepare the change; I reviewed every line and ran the tests above.
