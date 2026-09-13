# Ergänzung PR #621 — UMGESETZT 13.09. abends (Commit auf #621, Body ersetzt, Kommentar)

Zwei Änderungen am offenen PR, nach den Belegen vom 13.09.:

## A) Commit auf drop-forced-compile-cache-off: tools/torch_patches/

Neu: `tools/torch_patches/{README.md, apply.sh, 0001-aot-compile-serialize-triton-kernel-side-table.patch}`
(Backport von pytorch #173556, main dffe73e2, auf torch 2.10.0; idempotent, Hash-geprüft).

## B) PR-Body: den Absatz "One caveat on torch 2.10" ersetzen durch

> Two things on torch 2.10.0, which requirements/cuda.txt pins:
>
> 1. The first warm boot after a cold compile could not load the fresh artifacts on 2.10.0
>    (`Compiling model again due to a load failure ..., reason:` with an empty reason, then a
>    recompile and re-save; from the second warm boot on everything loads). The cause is the
>    Triton kernel side table that 2.10.0 does not serialize into the artifact; PyTorch fixed
>    it in #173556 (main dffe73e2, in 2.11+), and 2.11 drops Volta from the cu128 wheels, so
>    2.10.0 stays pinned here. This PR therefore ships the fix as a backport:
>    `tools/torch_patches/apply.sh .venv/bin/python` patches the installed 2.10.0
>    (idempotent, refuses anything but the pristine or the patched file). Measured on
>    Qwen3.8-27B-NVFP4, MTP k=3, TP2 on two V100 with the cache on, after deleting
>    `torch_aot_compile/` once: cold 476 s (writes), first warm 85 s (4 artifacts loaded,
>    no load failure), second warm 80 s; text identical (SHA-256 `a3dffc7c5e9b417a`).
> 2. Removing the opt-out exposes a bug in the Qwen4Exp PLE layer that the opt-out had
>    hidden since #403: the gather op took the table pointer as an `int` and Inductor baked
>    the address into the AOT artifact, so the first warm boot of Qwen3.8-Flash-Next died
>    with an illegal memory access on the stage that holds the PLE table. Fixed in
>    <PLE-PR-NUMMER>, which should land before or together with this PR. With both, Flash-Next
>    TP2 PP2 boots cold 340 s, first warm 312 s, second warm 343 s, six artifacts loaded
>    each time, text identical (SHA-256 `e948a82ead51948c`).

## C) Test Result ergänzen

> 4. torch backport: `tools/torch_patches/apply.sh` applied twice (second run reports
>    "already applied"); 27B cold/warm/warm as above.
> 5. Flash-Next cold/warm/warm with <PLE-PR-NUMMER>: as above.
