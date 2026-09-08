# PR-Entwurf: 1CatAI/1Cat-vLLM — Turing (sm75) lauffähig, verlustfrei und schnell

Stand 2026-09-08 abends. Branch `sm75-gdn-prefill-route`, Basis `origin/main`
**4f19ef7**. Drei Commits, fünf Dateien, 238+/22-.

| Commit | Inhalt |
|---|---|
| `688dbd5` | Fix 1 (GDN-Prefill auf Turing startfähig) + Fix 3 (fusionierter Spec-Kern) |
| `99ba6f4` | Fix 5 (Full-Forward-Wrapper nur auf Volta scharf) |
| `9c79cf4` | Fix 2 (Baseline für jedes pre-Ampere-Gerät) |

**Status:** noch kein PR eröffnet. Peuqui muss vorher jede geänderte Zeile
gelesen haben — AGENTS.md verbietet reine Agenten-PRs.

Vorgeschlagener Titel:

    [Bugfix][Perf][SM70] Make Turing (sm75) boot, compute correctly and keep its Inductor fusions

---------------------------------------------------------------------------
PR-BODY
---------------------------------------------------------------------------

## Purpose

On a Turing-only deployment (2x Quadro RTX 8000, sm75), models that use the
Qwen GDN linear-attention layer -- Qwen3.5, Qwen3.8, Qwen3.8-Flash-Next -- do
not start; once they start they answer with garbage; once they answer
correctly they lose a third of their Inductor fusions as soon as speculative
decoding is on. This PR fixes the four defects behind that, in the order a
user meets them. Other architectures and heterogeneous Volta+Turing
deployments are unaffected by the first two: a visible Volta card already
pulls the baseline in, which is why this has stayed invisible.

**1 — It does not boot.** `_resolve_gdn_prefill_backend` accepts sm75
(`capability.minor in (0, 5)`) and selects `flashqla_sm70`. That path's default
TileLang prefill requests 86016 B of dynamic shared memory per block; Turing
caps the opt-in limit at 65536 B (measured: `shared_memory_per_block_optin`).
The worker dies during engine init:

    RuntimeError: Worker failed with error 'Failed to set the allowed dynamic
    shared memory size to 86016'
    -> Worker proc VllmWorker-3 died, Engine core initialization failed

Split `is_sm70_or_sm75` into `is_sm70`/`is_sm75`, gate FlashQLA prefill on
Volta only and warn once with the reason. The Triton/FLA fallback already
exists; the FlashQLA *decode* route is untouched and keeps working on Turing.
The VLK CUDA variant of the kernel does fit into 65536 B but is slower than
Triton from 2048 tokens per chunk upwards (measured: 512 tok 1.32x faster,
2048 tok 15 % slower, 8192 tok 21 % slower), so Triton is the better choice,
not a stopgap.

**2 — It computes garbage.** Follow-up to #514. That PR taught the SM70 config
gates to look at every participating device instead of device 0; the capability
they ask for stayed exactly `(7, 0)`. A Turing-only deployment therefore never
receives the Flash-V100 baseline -- the boot log shows zero "Auto-setting
VLLM_SM70_*" lines. Unconfigured, Qwen3.8-27B-NVFP4 answers with garbage at
**every** prompt length from 19 to 13004 tokens, and **with speculative
decoding disabled as well**:

    k=0, 19-token prompt:    'Ernesto "Che" Guevara (1928-1967) was an Argentine...'
    k=0, 1543-token prompt:  '</parameter>\n</function>\n</tool_call>'
    k=0, 3071-token prompt:  ', 0x00000000, 0x00000000, 0x00000000, ...'
    k=3, 19-token prompt:    'Er is een aantal mankeleisen die\n\nErtheorieen die...'

This is therefore not a speculative-decoding problem. Bisecting the nine
baseline defaults pinned the whole difference on
`VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH`; the other eight are speed-neutral
(68.80 vs 68.83 tok/s). The baseline is a pre-Ampere tuning, not a Volta
tuning -- a heterogeneous Volta+Turing box already inherits it through the
Volta card, which is why the defect has stayed invisible. Add
`_any_participating_device_is_pre_ampere()` and gate the baseline on it. The
other eight `(7, 0)` gates in that file are untouched.

**3 — The speculative branch launches two kernels where one exists.** The
branch calls `fused_gdn_gating` followed by `fused_recurrent_gated_delta_rule`,
although the module already imports `fused_sigmoid_gating_delta_rule_update`
and uses it in the DFlash2 branch directly above. Using the fused routine there
as well drops the GDN core from 83.4 to 55.0 ms per run and matches the
reference output byte for byte where the two-kernel path produced a variant.

**4 — Speculation takes the whole layer out of Inductor.**
`QwenGatedDeltaNetAttention.forward` routes the entire GDN layer through
`torch.ops.vllm.qwen_gdn_full_forward`, which is a splitting op. Its docstring
states the intent: keep the projections around the recurrent core in strict
eager order instead of letting Inductor rewrite them. The guard is armed by
`VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH` plus "speculation active" and asks
for no device capability at all — so fix 2 above would newly arm a Volta
workaround on Turing.

On Turing the cost is about fifteen extra elementwise launches per GDN layer
and step. The gated RMSNorm alone falls back from two fused Triton kernels to
twelve native ones; nsys, decode window, device 0:

| kernel | fork (compiled) | upstream (eager) |
|---|---|---|
| `elementwise_kernel` | 10402 | 20395 |
| `vectorized_elementwise_kernel` | 3175 | 26663 |
| `unrolled_elementwise_kernel` | 783 | 10995 |
| `reduce_kernel` | 205 | 3473 |

Bind the automatic arm to Volta. The recurrent-core boundary is untouched:
`auto_sm70_qwen_gdn_full_forward` still selects
`qwen_gdn_attention_core_standard_spec`, and an explicit
`VLLM_SM70_QWEN_GDN_FULL_FORWARD=1` still forces the wrapper on any device.

The gate asks `torch.accelerator.current_device_index()`, not device 0. On a
node that mixes Volta and Turing every rank sees all devices and only
`set_device` differs, so device 0 answers for the wrong card on most ranks.

## Results

Qwen3.8-27B-NVFP4, TP2 on 2x RTX 8000, greedy with a fixed seed, 400 tokens,
five repetitions per row. Spread below 0.3 tok/s.

| configuration | tok/s | output sha256 |
|---|---:|---|
| k=0, no speculation (the reference) | 43.62 | `0106659946c064b1` |
| k=3, fixes 1+2+3 | 69.58 | `0106659946c064b1` |
| k=3, **all four fixes** | **73.42** | `0106659946c064b1` |

Qwen3.8-Flash-Next-180B, TP2xPP2 across 2x RTX 8000 and 2x V100, k=4, 300
tokens, three repetitions:

| configuration | tok/s | output sha256 |
|---|---:|---|
| wrapper forced everywhere | 57.66 | `864572d17f5fa8c4` |
| **all four fixes** | **58.83 / 59.06** | `864572d17f5fa8c4` |

Fix 4 is worth +5.4 % on the Turing-only box and +2.0 % on the mixed one,
where only half the layers sit on Turing.

**Losslessness.** Speculative decoding must reproduce the unspeculated output.
It does, before and after this PR: on the 27B, k=3 equals k=0 byte for byte at
all ten prompt lengths from 19 to 13004 tokens. Acceptance length is
bit-identical (2.963 on the 27B, 3.030 on Flash-Next), so the draft path is not
involved in the speedup.

**Per-device gate, verified on the mixed node.** Each rank writes its own
capability and guard state to a file during layer construction (INFO from
non-primary PP ranks is filtered, so a missing log line proves nothing):

    dev=0  cap=(7,5)  volta=False  auto=True  maybe=False   <- Turing: disarmed
    dev=1  cap=(7,5)  volta=False  auto=True  maybe=False   <- Turing: disarmed
    dev=2  cap=(7,0)  volta=True   auto=True  maybe=True    <- Volta: unchanged
    dev=3  cap=(7,0)  volta=True   auto=True  maybe=True    <- Volta: unchanged

`auto=True` on all four ranks shows the wrapper would have been armed
everywhere before this change. Volta behaviour is unchanged.

## Test Plan

Environment: checkout at `origin/main` 4f19ef7; unit tests and linting run
against that tree. Runtime measurements run on a 1Cat-vLLM 1.5.0 wheel
deployment with the same four changes applied by hand (see Limitations).
Hardware: 2x Quadro RTX 8000 (sm75) + 3x Tesla V100 (sm70), CUDA 12.8.

    # every test file that exercises these paths, plus the two new files
    python -m pytest tests/config/test_sm70_gates_any_visible_device.py \
        tests/config/test_prefix_anchored_swa.py \
        tests/compile/test_sm70_decode_graph.py \
        tests/engine/test_arg_utils.py \
        tests/v1/spec_decode/test_dflash2.py \
        tests/model_executor/layers/test_gdn_full_forward_device_gate.py \
        tests/model_executor/layers/test_gdn_prefill_backend_resolve.py -q

    # counter-checks: each fix reverted, tests kept
    pre-commit run --files <5 files>
    pre-commit run mypy-3.10 --hook-stage manual --files <5 files>

    # duplicate-work checks
    gh pr list --repo 1CatAI/1Cat-vLLM --state open --search "<keywords>"
    gh issue view 412 --repo 1CatAI/1Cat-vLLM --comments

## Test Result

    315 passed, 21 warnings in 120.25s

Counter-checks (the change reverted, the tests kept):

- `_sm70_current_device_is_volta` asking device 0 instead of the current
  accelerator: `test_mixed_node_answers_for_the_current_device` fails, the
  other four pass.
- the baseline gate back on exactly `(7, 0)`:
  `test_sm70_baseline_defaults_follow_any_visible_device[homogeneous-sm75-is-pre-ampere]`
  fails, the other 24 pass.
- fix 1 reverted: exactly the two Turing cases in
  `test_gdn_prefill_backend_resolve.py` fail.

pre-commit (`--files` over the five changed files): ruff check, ruff format,
typos, mypy-local, SPDX headers, root lazy imports, forbidden imports,
torch.cuda-call check, config-docstring check, attention-backend docs and the
boolean-ops check all Passed. `pre-commit run mypy-3.10 --hook-stage manual`:
Passed.

On the hardware, with fix 2 removed and no environment override, so that the
box is exactly what a Turing-only user finds today: the boot log contains zero
"Auto-setting VLLM_SM70_*" lines and the model answers with garbage at all ten
prompt lengths, at k=0 and at k=3 (excerpts under Purpose). With fix 2 the
same command line auto-sets all nine defaults, reaches 73.42 tok/s and matches
the hand-configured reference byte for byte at both k values, 10 of 10 prompt
lengths.

## Not a duplicate

Checked on 2026-09-08 against every open PR. Five touch one of our three
files:

- **#556** (draft) is the only one in the same area of
  `qwen_gdn_linear_attn.py`. It adds a `qwen_gdn_full_forward_direct` variant
  for the `output is None` case inside `forward`; this PR changes whether the
  wrapper is armed at all, in `__init__`. Different lines, orthogonal concerns.
- **#566**, **#523** touch `qwen_gdn_linear_attn.py` elsewhere and contain no
  occurrence of `maybe_sm70_qwen_gdn_full_forward` or
  `auto_sm70_qwen_gdn_full_forward`.
- **#239**, **#235** touch `vllm/config/vllm.py` and contain no occurrence of
  `sm70_flash_v100_baseline` or `_any_participating_device_is_capability`.

There is no open issue or PR mentioning Turing, sm75 or RTX 8000 anywhere in
the repository. Issue #412, whose first item became #514, is closed; fix 2 is
a follow-up to the function that PR introduced and is a different problem from
#412 itself (that one was heterogeneous device order, this one is a
Turing-only box never qualifying at all).

## Limitations

**Runtime numbers come from a 1.5.0-based deployment, not from this tree.**
The compiled extensions here are a prebuilt 1.5.0 wheel; building current main
from source on this hardware is a multi-hour CUDA build. The four changes were
applied to both trees. For the code that carries the measurement the two trees
are identical: `forward`, `_full_forward`, `qwen_gdn_full_forward`,
`_sm70_qwen_gdn_full_forward_enabled`, `_qwen_gdn_run_recurrent_core` and
`Qwen3_5GatedDeltaNet.forward_cuda` are byte-identical between them, and the
fix 4 gate is byte-identical as well. Fix 2 differs in one helper name only
(`_any_visible_device_has_capability` there, `_any_participating_device_is_pre_ampere`
here, the latter introduced by #514); in both measured topologies visible and
participating devices coincide.

**One consequence of fix 2 is untested.** On current main the widened gate also
admits, for a Turing-only box, the Qwen3.8 dual-compile lane and the hybrid PLE
defaults. Both sit behind `_is_sm70_qwen38_nomtp_dual_compile_contract`, which
requires TP=4 with PP=1, no speculation and the exact Qwen4Exp shape. We cannot
exercise that: this machine has two Turing cards, so TP=4 on Turing is not
reachable here, and the 1.5.0 base predates the lane entirely. If you would
rather keep those two Volta-only, say so and we will add the condition.

**The byte-identity claims above are made at 260 generated tokens or below.**
The 27B is reproducible well past that (the same question three times in one
server process, twice booted, all six identical at 13004 tokens of context and
1200 tokens of output), but at that length different baseline combinations
produce different -- individually correct -- texts, so a hash is no longer a
useful equality test between configurations. Flash-Next is not reproducible
there at all: k=0 twice, same code, two boots, gives three different hashes,
which is the known batch-invariance behaviour of RMSNorm, matmul and split-KV
attention kernels. The long-context quality runs (three questions of 30
sentences each behind 13004 tokens of unrelated context, including a
deliberately misspelled term as a hallucination probe) were therefore read and
judged by hand, not hashed.

AI assistance: this change was developed with Claude (Anthropic) as a coding
assistant. Every changed line was reviewed by me and the test runs above were
executed on my hardware; I can defend the change end to end.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
