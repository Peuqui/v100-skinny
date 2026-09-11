# PR-Entwurf: Editable-Bau bricht nach dem Vollbau ab (fünf pybind11-Module)

Status: GESENDET 2026-09-11 abends als https://github.com/1CatAI/1Cat-vLLM/pull/601 (Commit e5e3e9af). Vorher: Branch `editable-soabi-modules` im Worktree
`1Cat-vLLM-editable-pr`, Basis origin/main fe67339d. Freigabe Peuqui für den PR
(Punkt 8, 2026-09-10); Commit, Push und Eröffnen erst auf Ansage.

Review 2026-09-11 vormittags: Mechanismus und Gegenbeleg halten (Baum
`5099866f` hatte setup.py, CMakeLists.txt, cmake/, csrc/, flashinfer-sm70/,
flash-attention-v100/ identisch mit fe67339d; Produktion wurde mit
`pip install -e .` und genau dieser Änderung gebaut, `f03a7102`).

Nachgezogen 2026-09-11 nachmittags (Belege im Text unten eingearbeitet):
- (1) setuptools-Kette, aus setuptools 80.10.2 der Produktions-venv gelesen:
  `editable_wheel._set_editable_mode` setzt `build_ext.editable_mode = True`
  (editable_wheel.py 240–247) → `build_ext.finalize_options`: `if
  self.editable_mode: self.inplace = True` (build_ext.py 221–222) →
  `build_ext.run`: `if old_inplace: self.copy_extensions_to_source()` (94–100)
  → `get_ext_filename`: `if ext.py_limited_api and abi3_suffix:` Name auf
  `.abi3.so` (159–177). `pip install -e .` und `build_ext --inplace` enden
  also in derselben Kopierfunktion mit demselben erwarteten Dateinamen.
- Alle fünf Module sind wirklich pybind11, je Quelldatei geprüft
  (`PYBIND11_MODULE(TORCH_EXTENSION_NAME`): `csrc/sm70_turbomind/ops/
  exact_row_reduce.cu`, `csrc/sm70_turbomind/ops/h3_w8a16.cu`,
  `flashinfer-sm70/csrc/h3_noncausal_sm70.cu`, `flash-attention-v100/kernel/
  h3/forward.cu`, `flash-attention-v100/kernel/h3/forward_sparse.cu`
  (Quellen aus CMakeLists.txt 758–802). Der Sampler dagegen `TORCH_LIBRARY`
  (`csrc/sm70_turbomind/ops/*.cu`), darum ging dort `USE_SABI 3`.
- Wheel-Tag: das offizielle Release-Wheel heißt
  `1cat_vllm-1.5.0-cp312-cp312-linux_x86_64.whl` (gh release view v1.5.0).
  Der Tag kommt aus `bdist_wheel.py_limited_api` (bdist_wheel.py 227/346–348),
  nicht aus den Extension-Deklarationen — die Änderung kann ihn nicht
  verschieben.

Wheel-Bau 11.09. abends ERLEDIGT (`handover/2026-09-11/scripts/abnahme2/
wheel_test.sh`, aus dem Belegbau-Baum inkrementell, `bdist_wheel`, Exit 0 nach
24 min 34 s): `1cat_vllm-1.5.1.dev945+gfe67339dd.d20260911.cu128-cp312-cp312-
linux_x86_64.whl`, 166 MB, Tag `cp312-cp312` wie das Release. `.so`-Namen
gegen das offizielle 1.5.0-Wheel: alle sieben dortigen Dateien identisch
benannt, dazu die fünf SOABI-Module, die es im 1.5.0-Release noch nicht gab
(`_h3_*`, `_sm70_exact_reduce_C`, `_sm70_sparse_attention_C` — seither
hinzugekommen). Installation in eine Kopie der Produktions-venv
(`/home/mp/vllm/venv-wheeltest`), `vllm` lädt aus `site-packages`, alle fünf
Module importieren unter ihren SOABI-Namen. `_sm70_sampler_C` lädt nicht per
`import` (kein `PyInit`, es ist `TORCH_LIBRARY` und wird in `_sm70_ops.py`
per Glob + `torch.ops.load_library` geladen) — mein Prüffehler, nicht das
Wheel. Erster Modelllauf aus dem Wheel scheiterte mit dem #592-Fehler
(`mat1 and mat2 shapes cannot be multiplied`), weil ich dem reinen main den
NVFP4-Entwurfskopf gab; Wiederholung mit dem incoai-Kopf läuft.

Noch offen vor dem Eröffnen: (4) Checklistenblock der PR-Vorlage anhängen;
(5) Duplikatsprüfung am Tag des Eröffnens wiederholen (11.09. erledigt);
Peuqui liest die 20 Zeilen selbst (AGENTS.md, erledigt 11.09.).

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
pybind11 modules (`PYBIND11_MODULE(TORCH_EXTENSION_NAME, …)` in
`csrc/sm70_turbomind/ops/exact_row_reduce.cu`, `csrc/sm70_turbomind/ops/h3_w8a16.cu`,
`flashinfer-sm70/csrc/h3_noncausal_sm70.cu`,
`flash-attention-v100/kernel/h3/forward.cu` and
`flash-attention-v100/kernel/h3/forward_sparse.cu`), and pybind11 cannot be
built against the limited API. The CMake side is therefore right, and the
declaration has to follow it: `CMakeExtension` now accepts a `py_limited_api`
override, and the five modules pass `False`. Every other extension keeps its
current declaration.

Only builds that copy extensions into the source tree are affected. In
setuptools 80.10.2, `pip install -e .` and `setup.py build_ext --inplace` end
in the same place: `editable_wheel` sets `build_ext.editable_mode`,
`build_ext.finalize_options` turns that into `inplace`, `build_ext.run` then
calls `copy_extensions_to_source`, and `get_ext_filename` derives the expected
`.abi3.so` name from `ext.py_limited_api`. A wheel build never takes that
copy step, and its ABI tag comes from `bdist_wheel.py_limited_api`, not from
the extension declarations — the official wheel is tagged `cp312-cp312`
(`1cat_vllm-1.5.0-cp312-cp312-linux_x86_64.whl`), so the change cannot
move it.

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
5. Wheel build with the change, same tree and toolchain
   (`python setup.py bdist_wheel`, incremental on the build directory of
   step 3): exit 0 after 24 min 34 s, wheel tag `cp312-cp312` like the
   release wheel. The seven `.so` files of the official
   `1cat_vllm-1.5.0` wheel carry identical names in the new wheel; the five
   modules touched here appear under their SOABI names (they postdate the
   1.5.0 release). Installed into a copy of a working venv: `vllm` imports
   from `site-packages`, and each of the five modules imports under its
   SOABI name. A model served from that wheel (Qwen3.8-27B NVFP4 target,
   `incoai/Qwen3.8-27B-DFlash2` draft, TP2 on 2x Tesla V100, k=7, greedy,
   400 tokens, 5 runs) boots in 175 s without a traceback and decodes at
   70.67 tok/s with acceptance length 3.448, coherent German output.

No kernel source, CMake target or runtime path changes; only the filename
setuptools expects for these five modules.

## Not a duplicate

Searched `1CatAI/1Cat-vLLM` issues and PRs on 2026-09-11 for `editable`,
`abi3`, `py_limited_api`, `build_ext inplace` and `pybind11 limited`: the only
related work is #319/#320 (merged), which fixed `_sm70_sampler_C` on the CMake
side, a route not open to pybind11 modules. The one open search hit, #409
(TokenSpeed MLA optional), does not touch `setup.py`. No open PR touches these
declarations.

Two further editable-install gaps exist that this PR leaves alone: the
`flash_attn_v100` package is missing from the editable finder mapping, and the
prebuilt SM70 FlashQLA GDN extension is only placed in `build_lib`, so an
editable install JIT-compiles it at first use. I can follow up on both
separately if that is welcome.

AI assistance (Claude) was used to trace the failure and prepare the change;
I reviewed every line and ran the tests above.
