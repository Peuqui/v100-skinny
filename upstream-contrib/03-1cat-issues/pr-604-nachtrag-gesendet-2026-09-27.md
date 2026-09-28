#604 Nachtrag (Entwurf) — wird eingefügt, sobald der Branch sm70-ops-on-turing per Force-Push auf 2d079631 steht.

1) Im Aufzählungspunkt `sm70_turbomind.py` ersetzen:
   "call `fp8_qpn8_dispatch_sm70_out` with a cached fp16 `[K, N]` workspace."
   durch
   "register a cached fp16 `[K, N]` workspace with `sm70_layer_workspaces` (#710)
   and dispatch through `vllm::sm70_fp8_qpn8_dispatch` with the layer prefix."

2) Neuer Abschnitt vor "## Not a duplicate":

## Update 2026-09-27

Rebased onto current main (14abfc27) and stacked on #710; until #710 lands this
branch contains its commit, so please review only the last commit here.

The first version kept `workspace.data_ptr()` in the layer state and passed it
to `fp8_qpn8_dispatch_sm70_out`. TorchDynamo records that integer as a
constant, so with the AOT compile cache on every warm start of the 27B on
2x Quadro RTX 8000 failed in the profiling run with "The specified pointer
resides on host memory and is not registered with any CUDA device". The QPN8
route now registers its workspace with the module from #710 and dispatches
through `vllm::sm70_fp8_qpn8_dispatch`; cold and warm start pass (the warm
start loads the AOT artifact), greedy output identical to the cold start.

Tests on the updated branch (this PR's test plan plus #710's):
165 passed on a Tesla V100, 159 passed and 6 skipped on a Quadro RTX 8000;
pre-commit and mypy-3.10 clean.
