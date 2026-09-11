# PR-Entwurf: Editable-Bau bricht nach dem Vollbau ab (fünf pybind11-Module)

Status: ENTWURF, nicht gesendet. Branch `editable-soabi-modules` im Worktree
`1Cat-vLLM-editable-pr`, Basis origin/main fe67339d. Freigabe Peuqui für den PR
(Punkt 8, 2026-09-10); Commit, Push und Eröffnen erst auf Ansage.

Review 2026-09-11: Mechanismus und Gegenbeleg halten (fünf Module
`PYBIND11_MODULE`, CMake `WITH_SOABI` ohne `USE_SABI`; Baum `5099866f` hatte
setup.py, CMakeLists.txt, cmake/, csrc/, flashinfer-sm70/, flash-attention-v100/
identisch mit fe67339d; Produktion wurde mit `pip install -e .` und genau dieser
Änderung gebaut, `f03a7102`). Vor dem Eröffnen noch: (1) im Text ergänzen, dass
`pip install -e .` und `build_ext --inplace` durch dieselbe setuptools-Funktion
laufen (`editable_mode` → `inplace` → `copy_extensions_to_source`); (2) den
Gegenbeleg mit Commit und Verzeichnissen präzisieren; (3) Wheel-Bau als „nicht
gebaut, am Code begründet“ kennzeichnen oder nachholen; (4) Checklistenblock der
PR-Vorlage anhängen; (5) Duplikatsprüfung am Tag des Eröffnens wiederholen;
Peuqui liest die 20 Zeilen selbst (AGENTS.md).

Titel:

    [Bugfix][Build][SM70] Declare the pybind11 SM70 extensions non-limited-API

---

## Purpose

An editable or in-place build with `TORCH_CUDA_ARCH_LIST=7.0` compiles
everything and then aborts at the copy step:

```
error: can't copy '/tmp/tmpxqiubqmn.build-lib/vllm/_sm70_exact_reduce_C.abi3.so': doesn't exist or not a regular file
```

(`pip install -e . --no-build-isolation`, after the full compile.)

It is the same symptom #319/#320 fixed for `_sm70_sampler_C`, for five more
modules: `_sm70_exact_reduce_C`, `_h3_w8a16_C`, `_h3_flashinfer_C`,
`_h3_flashattn_C` and `_sm70_sparse_attention_C`. `setup.py` declares every
`CMakeExtension` with `py_limited_api=True`, so setuptools expects
`<name>.abi3.so`, while CMake defines these five targets `WITH_SOABI` without
`USE_SABI` and emits `<name>.cpython-312-x86_64-linux-gnu.so`.

The fix from #320 does not carry over. `_sm70_sampler_C` registers its ops
through `TORCH_LIBRARY` and could move to the stable ABI; these five are
pybind11 modules (`PYBIND11_MODULE`, `pybind11::class_`, `pybind11::bytes`),
and pybind11 cannot be built against the limited API. The CMake side is
therefore right, and the declaration has to follow it: `CMakeExtension` now
accepts a `py_limited_api` override, and the five modules pass `False`. Every
other extension keeps its current declaration.

Only builds that copy extensions into the source tree are affected (`pip
install -e .`, `setup.py build_ext --inplace`). Wheel builds install the CMake
output directly, which is why the official wheels never hit this.

## Test Plan

1. `pre-commit run --files setup.py`
2. Naming, without a build: what setuptools expects per declaration, and what
   CMake emits.
3. Full in-place build of a fresh checkout of `main` (fe67339d) plus this
   change, 2x Quadro RTX 8000 + 3x Tesla V100 host, CUDA 12.8, Torch 2.10,
   Python 3.12:
   `TORCH_CUDA_ARCH_LIST=7.0 MAX_JOBS=4 python setup.py build_ext --inplace`
4. Import each of the five modules from the in-place tree.

## Test Result

1. `pre-commit run --files setup.py`: all applicable hooks passed (ruff check,
   ruff format, typos, mypy); `pre-commit run mypy-3.10 --hook-stage manual
   --files setup.py`: passed.
2. setuptools 80.10.2 expects, per declaration:

   ```
   py_limited_api=True:  _sm70_exact_reduce_C.abi3.so
   py_limited_api=False: _sm70_exact_reduce_C.cpython-312-x86_64-linux-gnu.so
   ```

   CMake emits the second form for all five targets, and `.abi3.so` for
   `_sm70_sampler_C` (which has `USE_SABI 3` since #320).
3. In-place build of fe67339d plus this change: exit 0 after 52 min
   (`MAX_JOBS=4`). The copy step now places all eleven extension modules in
   the source tree, the five pybind11 modules under their SOABI names:

   ```
   vllm/_h3_flashattn_C.cpython-312-x86_64-linux-gnu.so
   vllm/_h3_flashinfer_C.cpython-312-x86_64-linux-gnu.so
   vllm/_h3_w8a16_C.cpython-312-x86_64-linux-gnu.so
   vllm/_sm70_exact_reduce_C.cpython-312-x86_64-linux-gnu.so
   vllm/_sm70_sparse_attention_C.cpython-312-x86_64-linux-gnu.so
   ```

   The build then continues through the Flash-V100 and FlashQLA bundling. The
   optional Rust frontend is skipped on this host ("can't find Rust
   compiler"), which is unrelated and does not fail the build.
   Without the change the build aborts at exactly this step: the error quoted
   above is from `pip install -e .` on 2026-09-10, on a tree whose `setup.py`,
   `CMakeLists.txt` and native sources matched `main`. I did not repeat the
   52-minute build without the change.
4. Each module imports from the in-place tree and exposes its bindings:
   `_sm70_exact_reduce_C` (allocate, open_handle, release, run),
   `_h3_w8a16_C` (ColumnMajorGemmPlan, dequantize, gemm, prepare_fp16),
   `_h3_flashinfer_C`, `_h3_flashattn_C` and `_sm70_sparse_attention_C`
   (forward).

No kernel source, CMake target or runtime path changes; only the filename
setuptools expects for these five modules.

## Not a duplicate

Searched `1CatAI/1Cat-vLLM` issues and PRs on 2026-09-10 for `editable`,
`abi3` and `py_limited_api`: the only related work is #319/#320 (merged), which
fixed `_sm70_sampler_C` on the CMake side, a route not open to pybind11
modules. No open PR touches these declarations.

Two further editable-install gaps exist that this PR leaves alone: the
`flash_attn_v100` package is missing from the editable finder mapping, and the
prebuilt SM70 FlashQLA GDN extension is only placed in `build_lib`, so an
editable install JIT-compiles it at first use. I can follow up on both
separately if that is welcome.

AI assistance (Claude) was used to trace the failure and prepare the change;
I reviewed every line and ran the tests above.
