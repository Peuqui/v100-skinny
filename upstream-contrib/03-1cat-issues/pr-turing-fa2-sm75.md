# PR-Entwurf 1Cat: FlashAttention-2 für Turing (Paket C, Form 1) — VERÖFFENTLICHT als #623 (13.09.)

Worktree `1Cat-vLLM-pr-fa2sm75`, Branch `sm75-fa2-pr` auf origin/main dfef3342.
FA-Fork: Peuqui/flash-attention @ 43b9d29c (Tag `sm75-1cat-2026-09-13`).
Bezug: Issue #612 (Bauform-RFC, ohne Antwort; Form 1 eingereicht, Peuqui 13.09.).

Titel: [Feature][SM75] Build and load a Turing FlashAttention-2 library next to the Volta one

---

## Purpose

Turing GPUs (RTX 8000, RTX 6000, T4, RTX 20xx) get no FlashAttention backend today: `CMakeLists.txt` fetches vllm-project/flash-attention for 8.0+ and zhinianqin/flash-attention-v100 for 7.0, so an sm75-only or a mixed 7.0;7.5 build ships no sm75 kernels. At runtime the platform offers FLASHINFER first, whose paged prefill fails with "invalid argument" on sm75, and TRITON_ATTN as the working fallback. Issue #612 asked which form you prefer for closing that gap; this PR is form 1 of the three, the one that mirrors how the sm70 build already pins a fork.

## What changes

- `cmake/external_projects/vllm_flash_attn_sm75.cmake` (new): an `ExternalProject_Add` that builds https://github.com/Peuqui/flash-attention at 43b9d29c (tag sm75-1cat-2026-09-13; vllm-project pin 28e862d plus the sm75 forward path, split-KV for paged multi-token queries, and a CMake option that names the output file) with `-DCUDA_ARCHS=7.5 -DFA2_ENABLED=ON -DFA3_ENABLED=OFF -DVLLM_FA2_OUTPUT_NAME=_vllm_fa2_C_sm75`, and installs `_vllm_fa2_C_sm75.abi3.so` into `vllm/vllm_flash_attn/` as component `_vllm_fa2_C_sm75`. It is included only when 7.5 is in `CUDA_ARCHS` and is independent of which FA2 tree the existing block fetches. A second FetchContent of the same project is not possible: both trees define the target `_vllm_fa2_C` and override global CMake functions (see the note in `vllm_flash_attn.cmake`). The fork names the interpreter `Python_EXECUTABLE`, so the file translates `VLLM_PYTHON_EXECUTABLE`.
- `CMakeLists.txt`: the include, gated on 7.5.
- `setup.py`: the extension `vllm.vllm_flash_attn._vllm_fa2_C_sm75` when `TORCH_CUDA_ARCH_LIST` contains 7.5, and the file in the precompiled-wheel extraction list.
- `vllm/vllm_flash_attn/flash_attn_interface.py`: the FA2 library is no longer imported at module import time. `load_fa2_library(device)` loads, on first use, the library built for the capability of the device the ops run on: `_vllm_fa2_C_sm75.abi3.so` for (7, 5), the regular `_vllm_fa2_C` otherwise. One library per process. `_is_fa2_supported` asks the worker's own device (`torch.accelerator.current_device_index()`), not index 0 of the visibility list, and accepts 7.5 when the matching library is installed.
- `vllm/platforms/cuda.py`: on (7, 5) the priority list is FLASH_ATTN, TRITON_ATTN, FLEX_ATTENTION (FlashInfer is not offered there); the capability for the backend choice is read from the worker's own device.
- `flash_attn_interface.ensure_fa2_library_loaded()` (new) and one call each in `vllm/v1/attention/backends/flash_attn_v100.py`, `vllm/v1/attention/ops/sm70_e4m3_long.py` and `vllm/v1/attention/ops/sm70_e4m3_scalar.py`: the SM70 backend resolves its D256 prefill, grouped long-context and scalar tail operators from `torch.ops._vllm_fa2_C` before the first attention call, and relied on the module import to have loaded the library. With loading moved to first use, those lookups now load the library for the worker's device first. Without this, Volta booted but logged "SM70 D256 exact-prefill operators are unavailable" and took its slower long-prefill fallback (found in the end-to-end run below).
- `vllm/v1/attention/backends/flash_attn.py`: capability floor 7.5; below 8.0 only fp16 is accepted (the sm75 build is fp16-only, bf16 is rejected by its entry points).
- `docs/design/attention_backends.md`: FA2 row 7.5+.
- `tests/v1/attention/test_sm70_flash_v100_policy.py::test_flash_v100_priority_is_sm70_only`: the expectation for 7.5 follows the new priority list (FLASH_ATTN, TRITON_ATTN, FLEX_ATTENTION; FlashInfer not offered).

## Measurements

2x Quadro RTX 8000 (TP2), Qwen3.8-27B-NVFP4, MTP k=3, greedy, fp16 KV cache. Because a ModelOpt NVFP4 checkpoint does not load on Turing on main without #604 (minimum capability 89 for modelopt_mixed), the runs were made on main plus #604 and #611 plus this change; the attention backend is the only variable between the two arms, and the sm75 library was the one produced by the fork build, not a hand-built file.

| | TRITON_ATTN (today's fallback) | FLASH_ATTN via _vllm_fa2_C_sm75 |
|---|---|---|
| 400-token probe, median of 5 | 70.91 tok/s | 74.19 tok/s |
| 13k-token prefix, time to first token, median of 3 | 36.26 s | 17.75 s |
| 13k-token prefix, decode of 200 tokens, median of 3 | 15.03 tok/s | 56.48 tok/s |

Output text identical in both arms (same SHA-256 for the 400-token probe and for the 13k answers). The 27B production entries of our rig have run on this library since 2026-09-06.

Volta is untouched: 2x V100 TP2 with the same checkpoint and DFlash2 gives 76.42 tok/s before and after (V100 loads `_vllm_fa2_C.abi3.so` as before; the loader log line names the file per worker).

## Build proof

Wheel built from this branch (dfef3342 plus these changes) with the sm75 tree fetched by the new ExternalProject, nothing prebuilt on the path:

```
TORCH_CUDA_ARCH_LIST=7.5 MAX_JOBS=4 CUDA_HOME=<cuda 12.8> .venv/bin/python -m pip wheel . --no-build-isolation --no-deps -w <out>
```

Result: `1cat_vllm-1.5.1.dev986+gdfef33421.d20260913.cu128-cp312-cp312-linux_x86_64.whl`, 167,220,823 bytes, containing

| file | bytes |
|---|---|
| `vllm/_C.abi3.so` | 150,129,760 |
| `vllm/_moe_C.abi3.so` | 94,325,336 |
| `vllm/vllm_flash_attn/_vllm_fa2_C_sm75.abi3.so` | 167,794,544 |

No `_vllm_fa2_C.abi3.so` is in this wheel, as expected for an arch list without 7.0 or 8.0+; `nm -D` shows no undefined `flash::` symbol in the sm75 library. Installed into a clean venv (torch 2.10.0+cu128), the library loads on a Quadro RTX 8000 and the GPU test below runs against it.

Second wheel, same tree, `TORCH_CUDA_ARCH_LIST="7.0;7.5"`, to show that the new ExternalProject sits next to the existing Volta fetch without touching it: 211,317,095 bytes, containing both FA2 libraries and the Volta-only components:

| file | bytes | `cuobjdump -lelf` |
|---|---|---|
| `vllm/_C.abi3.so` | 152,233,168 | sm_70 and sm_75 |
| `vllm/_moe_C.abi3.so` | 108,551,056 | sm_70 and sm_75 |
| `vllm/_sm70_sampler_C.abi3.so` | 11,683,168 | |
| `vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so` (zhinianqin fork, unchanged) | 21,752,184 | sm_70 only |
| `vllm/vllm_flash_attn/_vllm_fa2_C_sm75.abi3.so` | 167,794,544 | sm_75 only |

With this wheel, `load_fa2_library` on a Tesla V100 loads `_vllm_fa2_C.abi3.so` (capability 7.0) and on a Quadro RTX 8000 loads `_vllm_fa2_C_sm75.abi3.so` (7.5); the log line names the file. Volta keeps its own backend (`FLASH_ATTN_V100`, chosen by the platform on 7.0), so `_is_fa2_supported` still answers False there exactly as on main (main gates at 8.0, this change at 7.5), and `get_flash_attn_version` still falls back to None on SM70.

## Test Plan

1. `pre-commit run --files <the 12 changed and new files>` and `pre-commit run mypy-3.10 --hook-stage manual --files <the 8 python files>`.
2. New tests without a Turing device (the gates test imports `FlashAttentionBackend`, whose class body calls `get_flash_attn_version()`, so one CUDA device has to be visible once an FA2 library is installed): `tests/vllm_flash_attn/test_fa2_library_per_device.py` (library path per capability, one load per process, the current-device helper loads once, refusal without a library), `tests/v1/attention/test_flash_attn_turing_gates.py` (capability floor, fp16 gate), `tests/v1/attention/test_cuda_backend_priority_turing.py` (priority list on 7.5 and 8.0).
3. New GPU test on a 7.5 device: `tests/kernels/attention/test_fa2_sm75_forward.py` (loader picks the sm75 library, varlen forward against a torch SDPA reference for head sizes 64/128/256 and query lengths 1/8/333).
4. Existing test files that touch the changed modules, run in full on a V100: `tests/v1/attention/test_sm70_flash_v100_policy.py` (81 tests, one expectation updated as listed above) and `tests/kernels/attention/test_sm70_e4m3_scalar_fp32.py`.
5. The measurement matrix above (`fa2_chain.sh`, `fa2_probe2.sh` in our v100-skinny repository).

## Test Result

Turing rig: 2x Quadro RTX 8000 (compute capability 7.5), driver 580, CUDA 12.8, torch 2.10.0+cu128, Python 3.12.

1. `pre-commit run --files <all 12 changed and new files>`: every hook passed (including `check-torch-cuda-call`, SPDX, attention-backend docs check). `pre-commit run mypy-3.10 --hook-stage manual --files <8 python files>`: passed.
2. Tests without a Turing device, run with one V100 visible:
   - `tests/vllm_flash_attn/test_fa2_library_per_device.py`: 5 passed
   - `tests/v1/attention/test_flash_attn_turing_gates.py` and `tests/v1/attention/test_cuda_backend_priority_turing.py`: 8 passed
3. GPU test on one RTX 8000 (`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<the RTX>`): `tests/kernels/attention/test_fa2_sm75_forward.py`: 9 passed (head sizes 64/128/256 x query lengths 1/8/333, loader picked `_vllm_fa2_C_sm75.abi3.so`, output matches torch SDPA).
4. Existing test files, in full, one V100 visible: `tests/v1/attention/test_sm70_flash_v100_policy.py` and `tests/kernels/attention/test_sm70_e4m3_scalar_fp32.py`: 144 passed (both in this branch and in our fork).
   - `tests/kernels/attention/test_flash_attn.py` on the RTX 8000, as shipped: every case fails with `FlashAttention on Turing (sm75) only supports fp16 data type`, because the file parametrizes `DTYPES = [torch.bfloat16]` only and the sm75 build rejects bf16 by design. The same file with `DTYPES = [torch.float16]` and `QDTYPES = [None]` (no fp8 KV on this build): 160 passed, 160 skipped (the FA3 half, "Flash attention version 3 not supported").
   - `tests/v1/attention/test_attention_backends.py` could not run here: it needs the gated `meta-llama/Meta-Llama-3-8B` and `google/embeddinggemma-300m` configs from Hugging Face (403).
5. Measurement matrix: see the tables above.
6. End to end with the mixed 7.0;7.5 wheel (this branch plus #604 so that the NVFP4 checkpoint loads on Turing; the wheel's libraries, the branch's Python):

   - 2x Quadro RTX 8000, TP2, our production command for Qwen3.8-27B-NVFP4 (MTP k=3, prefix caching, 262k window) plus `--kv-cache-dtype float16` (the checkpoint declares FP8 KV; #613 is not in this stack): both TP workers log `Loaded FA2 library _vllm_fa2_C_sm75.abi3.so for compute capability 7.5`, backend FLASH_ATTN, ready after 451 s on an empty compile cache; a 400-token answer and a 13k-token-prefix answer, both coherent.
   - 2x Tesla V100, TP2, the same command with `--max-model-len 32768 --gpu-memory-utilization 0.90` (32 GB cards) and without the drafter's explicit `attention_backend: FLASH_ATTN` (main selects FLASH_ATTN_V100 on 7.0 itself; the explicit value is a convention of our fork): both workers log `Loaded FA2 library _vllm_fa2_C.abi3.so for compute capability 7.0`, backend FLASH_ATTN_V100, no "D256 exact-prefill operators are unavailable" line, ready after 145 s.
   - The two card pairs produce byte-identical text: SHA-256 `a3dffc7c5e9b417a` for the 400-token answer and `22da50b5c73529f5` for the 13k-prefix answer on both.
   - One model spanning both card types, PP2 with the RTX 8000 as stage 0 and the V100 as stage 1 (same command, `--tensor-parallel-size 1 --pipeline-parallel-size 2`, no speculative config): stage 0 logs `Loaded FA2 library _vllm_fa2_C_sm75.abi3.so for compute capability 7.5` and FLASH_ATTN, stage 1 logs the FLASH_ATTN_V100 decode and prefill paths active and runs its SM70 TurboMind warmup inside its own TP group; ready after 451 s (cold cache root), the same two answers with the same SHA-256 as above. Two things stay out of reach on main and are not claimed: the MTP drafter class of this model family declares no `SupportsPP` (`speculative.py` rejects the draft config under PP; the target model inherits `SupportsPP` through its VL wrapper), so the PP2 run is without speculation; and TP2 across the two card types hangs in `_warmup_fp8_dense_layers_coordinated` (`vllm/model_executor/warmup/awq_sm70_warmup.py:170` broadcasts LUT records from rank 0 inside the TP group, and a Turing rank never enters that warmup), which is a warmup assumption, not attention. Our rig runs Flash-Next TP2 PP2 with the RTX pair as stage 0 and the V100 pair as stage 1 on our fork with this loader since 2026-09-06; there every stage's TP group is homogeneous.

## Not a duplicate

Issue #612 (ours) asked for the form; #39 and #237 are usage questions about V100. `gh pr list --state open --search "sm75 OR turing OR flash attention"` shows only our own #572 (merged), #604 and #611, none touching the FA2 loader or the CMake FA2 block.

AI assistance (Claude) was used to prepare the change and run the matrix; I reviewed every line, built the wheel and measured on the hardware named above.
