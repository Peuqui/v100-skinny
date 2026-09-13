# PR-Entwurf: Zwangsabschaltung des Compile-Caches für den SM70-Graph entfernen

Worktree: `1Cat-vLLM-pr-compilecache`, Branch `drop-forced-compile-cache-off`
auf `origin/main` (dfef3342, umgesetzt 13.09.). VERÖFFENTLICHT als PR #621 (13.09., Commit 399088b9; GitHub-502-Serie bei der Anlage, Wiederholung erfolgreich).

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

## PR-Body (Fassung 13.09., mit Boot-Matrix)

## Purpose

With `VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH=1`, `VllmConfig` set `VLLM_DISABLE_COMPILE_CACHE=1` unless the user had set the variable, and `envs.disable_compile_cache()` defaulted to `1` on the same condition. The log line explained why: a reloaded AOT artifact had once reproduced deterministic greedy token drift. The cause of that drift was the compile-cache key, which did not include unregistered `VLLM_` switches and therefore reused an artifact compiled for a different kernel path. That was fixed in #536.

With the key fixed, the forced opt-out only hides the cache from users who did not ask for that. This change removes both defaults, so `VLLM_DISABLE_COMPILE_CACHE` means what its documentation says again, and removes `VLLM_SM70_ALLOW_COMPILE_CACHE_FOR_PROFILING`, whose only purpose was to bypass the forced opt-out (its mirror in `benchmarks/benchmark_sm70_decode.py` goes with it). Users who want the old behaviour set `VLLM_DISABLE_COMPILE_CACHE=1` explicitly.

## What the cache does on this graph, measured

Qwen3.8-27B-NVFP4 with DFlash2 n=7, TP2, greedy, 5x400 tokens per boot, `VLLM_DISABLE_COMPILE_CACHE=0` on this branch's parent (main dfef3342 plus our open PRs), 11 boots, output text identical (same SHA-256) in every one:

- Cold boot on an empty cache root: 170-210 s to ready, three AOT artifacts written.
- Warm boot that reloads the artifacts: 70-75 s to ready ("Directly load AOT compilation" for both ranks and all three graphs). Without reload, i.e. with the opt-out, a warm boot is 110-120 s because only Inductor's own cache is warm.
- Card type is part of the artifact key: booting the RTX 8000 pair on the cache root that a V100 pair had just filled produced a different key and compiled its own artifacts; a cold RTX boot on a fresh root produced that same key again. No cross-architecture reuse.
- One caveat on torch 2.10, which requirements/cuda.txt pins: the first warm boot after a cold compile fails to load the fresh artifacts ("Compiling model again due to a load failure ..., reason:" with an empty reason), recompiles and saves again; every later warm boot loads. With `VLLM_FORCE_AOT_LOAD=1` the failure is an `AssertionError` in `torch/_higher_order_ops/triton_kernel_wrap.py:get_kernel` (`assert idx in self.id_to_kernel`) while `AOTAutogradCache` computes the key of the loaded graph: user-defined Triton kernels are referenced by an index into a process-local table that the artifact does not carry. pytorch/pytorch#173556 (merged 2026-01-28, in torch 2.11 and later) serializes that table with the artifact. torch 2.11 dropped Volta from the CUDA 12.8 wheels, so this repository stays on 2.10 and keeps the one wasted warm boot per artifact generation; it costs the same as today's behaviour, nothing more.

## Test Plan

1. `pre-commit run --files benchmarks/benchmark_sm70_decode.py vllm/config/vllm.py vllm/envs.py` and `pre-commit run mypy-3.10 --hook-stage manual --files vllm/config/vllm.py vllm/envs.py`.
2. `grep -rn ALLOW_COMPILE_CACHE_FOR_PROFILING tests/ docs/` to confirm nothing else references the removed switch.
3. The boot matrix above (scripts `cache_devtype_chain.sh`, `cache_reload_chain.sh`, `cache_trace_chain.sh` in our v100-skinny repository; logs available on request).

## Test Result

1. All applicable hooks passed; mypy-3.10 passed.
2. No references outside the removed code.
3. 11 boots as described, text identical throughout, reload verified from the second warm boot on, card type verified in the key.

## Not a duplicate

`gh pr list --state open --search` for "compile cache", "DISABLE_COMPILE_CACHE" and "ALLOW_COMPILE_CACHE_FOR_PROFILING" and `gh issue list --search "compile cache"` return nothing related; no open PR touches these lines of `vllm/config/vllm.py` or `vllm/envs.py`.

AI assistance (Claude) was used to trace the cache path, run the boot matrix and prepare the change; I reviewed every line and ran the checks above.

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
