# PR-Entwurf: Zwangsabschaltung des Compile-Caches für den SM70-Graph entfernen

Worktree: `1Cat-vLLM-pr-compilecache`, Branch `drop-forced-compile-cache-off`
auf `origin/main` (fe67339d). Stand 12.09. nachts: Änderung fertig,
pre-commit und mypy-3.10 grün, NICHT committet, NICHT eröffnet.

**Vor der Freigabe lesen (Peuqui):** Die Messung zeigt KEINEN
Bootzeit-Gewinn auf dem Mini. Der PR ist eine Bereinigung: er stellt das
dokumentierte Verhalten von `VLLM_DISABLE_COMPILE_CACHE` wieder her, entfernt
eine Zwangsmaßnahme, deren Ursache #536 behoben hat, und einen damit toten
Umgebungsschalter. Ob das für 1Cat einen PR wert ist, ist deine
Entscheidung; AGENTS.md verbietet Trivial-PRs, das hier sind 40 Zeilen
Workaround mit Messbeleg, kein Typo.

## Was im Fork noch falsch ist (unabhängig vom PR)

Der Overlay hat nur den Block in `config/vllm.py` entfernt. Die zweite
Erzwingung sitzt in `envs.py`, `disable_compile_cache()`: Default 1, sobald
`VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH=1`. Deshalb meldet jeder Fork- und
Produktionsboot auf diesem Pfad „vLLM's torch.compile cache is disabled".
Siehe `OVERLAY-INVENTUR.md`.

## Commit-Message

```
[Cleanup][SM70] Drop the forced compile-cache opt-out for the Flash-V100 graph

The SM70 Flash-V100 0.0.3 compile graph forced VLLM_DISABLE_COMPILE_CACHE=1
in two places (the config defaults and the envs.py default) because a
reloaded AOT artifact once reproduced greedy token drift. The cause was a
compile-cache key that ignored unregistered VLLM_ switches (#536, merged
as 53199eb8). With the key fixed, a warm boot on that graph reloads its
AOT artifacts and produces the identical greedy text, so the opt-out is
back to what the user sets. VLLM_SM70_ALLOW_COMPILE_CACHE_FOR_PROFILING
only existed to bypass the forced opt-out and goes with it.

Co-authored-by: Claude <noreply@anthropic.com>
Signed-off-by: Peuqui <peuqui@github.com>
```

## PR-Titel

`[Cleanup][SM70] Drop the forced compile-cache opt-out for the Flash-V100 graph`

## PR-Body

## Purpose

With `VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH=1`, `VllmConfig` set
`VLLM_DISABLE_COMPILE_CACHE=1` unless the user had set the variable, and
`envs.disable_compile_cache()` defaulted to `1` on the same condition. The
log line explained why: a reloaded AOT artifact had reproduced deterministic
greedy token drift. The cause of that drift was the compile-cache key, which
did not include unregistered `VLLM_` switches and therefore reused an
artifact compiled for a different kernel path (#536, merged as 53199eb8).

With the key fixed, the forced opt-out only hides the cache from users who
did not ask for that. This change removes both defaults, so
`VLLM_DISABLE_COMPILE_CACHE` means what its documentation says again, and
removes `VLLM_SM70_ALLOW_COMPILE_CACHE_FOR_PROFILING`, whose only purpose was
to bypass the forced opt-out (its `benchmark_sm70_decode.py` echo goes with
it). `VLLM_USE_AOT_COMPILE` keeps its SM70 default of `1`; that is a
separate decision and is not touched.

## Test Plan

1. `pre-commit run --files vllm/config/vllm.py vllm/envs.py benchmarks/benchmark_sm70_decode.py`
   and `pre-commit run mypy-3.10 --hook-stage manual --files <same>`.
2. `tests/config/test_sm70_gates_any_visible_device.py` (the SM70 default
   gates), with and without the change.
3. Hardware, 2× Quadro RTX 8000 (TP2), Qwen3.8-27B-NVFP4 with the DFlash2
   NVFP4 draft head, `VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH=1`, greedy,
   5 × 400 tokens per boot, compare the generated text (SHA-256), the
   acceptance length and decode throughput across: (a) cache forced off as
   on `main`, (b) cache on, empty cache root (cold), (c) cache on, same root
   (warm, artifacts reloaded).

## Test Result

1. All applicable hooks passed; mypy-3.10 passed.
2. 14 passed, 3 failed — the three fail identically without the change
   (they need `facebook/opt-125m` from the Hub; run offline).
3. | Boot | startup | text SHA-256 | acceptance length | decode |
   |---|---:|---|---:|---:|
   | (a) cache forced off (`main`) | 127 s | `0106659946c064b1` | 3.325 | 77.04 tok/s |
   | (b) cache on, cold | 447 s | `0106659946c064b1` | 3.325 | 76.72 tok/s |
   | (c) cache on, warm | 121 s | `0106659946c064b1` | 3.325 | 76.97 tok/s |

   The warm boot logs `Directly load AOT compilation from path …` for both
   ranks and both compile ranges, and the text is byte-identical to the
   cold and to the forced-off boot: no drift. Startup time is not the point
   of this change: with the Inductor FX cache warm, a forced-off boot is as
   fast as a warm cache reload on this machine; the cold AOT export costs
   about five minutes once.

## Not a duplicate

Checked on 2026-09-12 against `1CatAI/1Cat-vLLM`: `gh pr list --state open
--search` for "DISABLE_COMPILE_CACHE" and "compile cache SM70" and
`gh issue list --search "compile cache drift"` return nothing; the open
PRs touching `vllm/config/vllm.py` (#572, #579, #239, #235) and `vllm/envs.py`
(#596, #561, #547, #523, #504, #349, #348, #327, #312, #241, #238, #236,
#235) contain no `COMPILE_CACHE` hunk (checked with `gh pr diff`).

AI assistance (Claude) was used to trace the two defaults and run the
measurements; I reviewed every line and ran the tests above.
