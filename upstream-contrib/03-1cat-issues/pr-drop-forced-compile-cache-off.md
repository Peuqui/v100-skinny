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

## Kartentyp-Test und Reload-Befund (13.09., work-main = main dfef3342 + Overlays, 27B DFlash2 Produktionskopf, VLLM_DISABLE_COMPILE_CACHE=0 explizit)

| Boot | Root | Ergebnis |
|---|---|---|
| V100 kalt | frisch | kompiliert, 3 Artefakte, Schlüssel 9bb08da7…, Boot 210 s, 76,46 tok/s |
| RTX auf demselben Root, identische Umgebung | geteilt | **eigener Schlüssel b334e866…**, kompiliert, kein Fremdladen, 76,81 tok/s |
| RTX kalt | frisch | derselbe Schlüssel b334e866… wie oben → Schlüssel reproduzierbar und kartentypabhängig |
| V100 warm (1. nach kalt) | wie Zeile 1 | **Laden scheitert** („load failure … reason:" leer), kompiliert neu, speichert neu, 120 s |
| V100 warm (2.) | dito | lädt alle 6 Artefakte (2 Ränge × 3 Graphen), **70–75 s**, 76,55 / 77,14 tok/s |
| frischer Root: kalt → warm → warm | neu | reproduziert exakt: 1. Warmstart scheitert, 2. lädt |

Text-SHA in allen elf Boots `0106659946c064b1`; Annahmelänge 3,298–3,353 (Kernelwahl je Compile, „Münze").

**Ursache des ersten Fehlschlags** (Boot mit `VLLM_FORCE_AOT_LOAD=1`, Traceback in
`~/.cache/mtp-diagnostics/qual_tr_force/boot.log`): beim `finalize_loading` des
Artefakts berechnet torch den AOTAutograd-Cache-Schlüssel des Graphen
(`autograd_cache.py:493 → codecache.py:832`) und schlägt in
`torch/_higher_order_ops/triton_kernel_wrap.py:167 get_kernel` mit
`assert idx in self.id_to_kernel` fehl: die benutzerdefinierten Triton-Kernel sind
im Graphen per Index einer PROZESSLOKALEN Tabelle referenziert, und im frischen
Prozess ist dieser Index (noch) nicht vergeben. Das vom scheiternden Warmstart neu
gespeicherte Artefakt lädt in allen folgenden Prozessen. torch 2.10.0, vLLM-Wrapper
`backends.py:332` ist Upstream-Code (Dedup), nicht die Ursache. Im Netz kein
bekannter Bericht (Suche 13.09.).

**Bewertung:** Kartentyp-Risiko vom Tisch. Cache an bedeutet: je Artefakt-Generation
ein stiller Fehlschlag mit einem Extra-Compile (~40 s), danach Warmstarts in 70–75 s
statt 110–120 s (Inductor warm, kein AOT) bzw. 170–210 s kalt. Text unverändert.
Der leere Fehlergrund ist ein eigener Befund (torch-Assertion ohne Meldung).

**Nachtrag 13.09. mittags:** der Fehlschlag ist upstream behoben: pytorch/pytorch PR #173556 „[precompile] Serialize triton kernel side table for bundled AOT artifacts" (bobrenjc93, gemergt 28.01.2026, Datei `torch/_dynamo/aot_compile_types.py`, +108). Enthalten in release/2.11 und 2.12, NICHT in release/2.10 (unser und 1Cats Pin torch==2.10.0; vLLM upstream pinnt 2.13). Duplikatsuche bei PyTorch: kein offenes Issue, der Fix ist die PR. Für den 1Cat-PR-Text: „auf torch 2.10 verpufft der erste Warmstart je Artefakt-Generation (torch #173556 fehlt), ab 2.11 sollte er laden; nicht von uns geprüft, da 1Cat 2.10 pinnt."
