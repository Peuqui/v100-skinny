# PR-Entwurf: [Bugfix+Perf][SM70/SM75] Pre-Ampere QSA sparse launch profile

**Status:** Entwurf 2026-09-03, wartet auf Freigabe Peuqui (Branch + Push + PR).
**Kontext:** Nachgang zu Issue #441 und dem geschlossenen PR #455 (Dispatch-
Irrtum unsererseits — upstream lädt auf CUDA `nvidia/`, nicht `amd/`).
Maintainer yangzhuxinyzx nannte als Kandidaten: Gate `< 80` + smem-Clamp in
`nvidia/ops/qsa.py`, nach Benchmark. valentijnvenus bat in #441 um einen PR.
**Benchmark-Gate: ERFÜLLT** — `benchmarks/qsa-nvidia-ab-2026-09-03.txt`
(Harness `tools/qsa_nvidia_ab.py`, upstream-Datei von origin/main ca73a34,
Flash-Next-TP2-Geometrie 12H/1KV/D256/TOPK2048/PAGE16/bf16/KV16k).

## Befund

1. **Korrektheit (sm75/Turing):** Die GB300-Tabelle wählt für alle
   Prefill-Regime (>32 base programs) `BLOCK_N=64`. Bei D=256 überschreitet
   dieses Tile Turings 64-KiB-Shared-Memory → Triton `OutOfResources`,
   der Kernel startet NICHT. Der bestehende sm70-Retune greift auf sm75
   nicht (Gate `is_device_capability(70)`), und selbst erzwungen repariert
   er nur ≥512 Programme (N32) — 33..256 Programme (N64@W4) scheitern weiter.
2. **Performance:** N16@W4 läuft auf beiden Architekturen in allen
   Prefill-Regimen und schlägt das jeweils beste lauffähige Upstream-Profil:
   sm70 1,2–2,6× (rows 64: 1,12 vs 2,41 ms; 256: 3,54 vs 9,12; 512: 6,96
   vs 8,56; 2048: 27,4 vs 33,0), sm75 1,16–1,20× (512: 12,3 vs 14,7;
   2048: 48,7 vs 57,0). Numerik identisch (max|diff| unter Toleranz).
3. **Nicht anfassen:** Decode-Kleinstprofile (base ≤ small_profile_limit
   bzw. <32) sind mit N16 + hohen Splits bereits optimal — unser
   S8-Profil wäre bei rows=1 3× langsamer (0,231 vs 0,071 ms). Ebenso
   bleiben `sm70_single_token` (N32-Decode-Sonderfall) und der
   TP4-W2-Sonderfall unangetastet.

## Vorgeschlagener Patch (gegen origin/main, `vllm/models/qwen4_exp/nvidia/ops/qsa.py`)

Zwei Stellen:

a) Callsite: Gate von „exakt sm70" auf „pre-Ampere“ verbreitern.

```diff
     block_n, target_splits, partial_warps = _qsa_sparse_launch_profile(
         base_programs,
         block_m,
-        current_platform.is_device_capability(70),
+        _qsa_is_pre_ampere(),
     )
```

mit Helper (neben den anderen `_use_sm70_*`-Helpern):

```python
def _qsa_is_pre_ampere() -> bool:
    capability = current_platform.get_device_capability()
    return capability is not None and capability.to_int() < 80
```

b) `_qsa_sparse_launch_profile`: der Pre-Ampere-Zweig wählt das schmale Tile.

```diff
-    if is_sm70 and block_n == 64:
-        # Two warps serialize the D=256 tensor-core work on V100. Four warps
-        # restore warp-level parallelism for split and non-split prefill.
-        partial_warps = 4
-        if base_programs >= 512:
-            # A 32-column tile improves the exact 512-row and 8192-row Qwen4Exp
-            # prefill shapes without changing small-batch or non-SM70 routes.
-            block_n = 32
+    if is_pre_ampere and block_n == 64:
+        # Pre-Ampere: the 64-column tile at D=256 exceeds Turing's 64 KiB
+        # shared-memory carve-out (Triton OutOfResources -- the kernel cannot
+        # launch on SM75 at all), and two warps serialize the D=256
+        # tensor-core work on V100. A 16-column tile with four warps launches
+        # on both and measures 1.16-2.6x faster than the previous best
+        # runnable profile across the 64..2048-row prefill regimes
+        # (V100-PCIE-32GB and Quadro RTX 8000, see issue #441).
+        partial_warps = 4
+        block_n = 16
```

(Parametername `is_sm70` → `is_pre_ampere` mit umbenennen.)

**Offene Frage an Maintainer im PR-Text:** deren N32-Wahl ab 512 Programmen
war auf die exakten 512/8192-Formen getunt; unsere Messung sieht N16 auch
dort vorn (sm70 1,20×, sm75 1,17×). Falls sie N32 auf sm70 behalten wollen,
wäre der Minimalfix: Gate `< 80` + `block_n = 32` → repariert sm75 ≥512,
lässt aber 33..256 Programme auf sm75 weiter scheitern — deshalb schlagen
wir N16 durchgängig vor.

## Antwort-Entwurf für valentijnvenus (#441)

> Working on it — the missing piece was a measurement gate: we wanted the
> narrow-tile numbers confirmed against the `nvidia/ops/qsa.py` kernel
> (the file CUDA actually loads) on both of our pre-Ampere archs before
> proposing a dispatch change. That is done now: the narrow 16-column
> profile wins every prefill regime on both cards (V100 1.2-2.6x, RTX 8000
> 1.16-1.20x, identical numerics), and the measurement also surfaced that
> the current 64-column prefill tile cannot launch on SM75 at all at D=256
> (Triton OutOfResources against Turing's 64 KiB shared memory). PR follows
> with the numbers.
