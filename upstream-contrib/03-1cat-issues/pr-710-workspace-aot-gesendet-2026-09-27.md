Title: [Bugfix][SM70] Resolve SM70 linear workspace addresses on AOT reload

## Purpose

#672 fixed process-local scratch addresses in AOT artifacts for the
compressed-tensors channel-FP8 path. Other SM70 linear paths still pass a
workspace address from apply() as an int:

- fp8.py (block FP8, QPN8 and prefill-dense routes): `fp8_qpn8_dispatch_sm70_out`
  (three sites) and `fp8_gemm_sm70_prefill_dispatch_out` (two sites)
- qwen3_5.py: `fp8_qpn8_dispatch_ba_split_sm70_out`
- sm70_turbomind.py, NVFP4 QPN4 state: `nvfp4_qpn4_dispatch_sm70_out` (two sites)

TorchDynamo records the integer as a constant, so an AOT artifact of these
routes carries the compiling process's address, and a later process that
loads it hands the kernel a stale pointer.

We hit this on Turing through the QPN8 route of #604, which copied the same
pattern: Qwen3.8-27B-NVFP4 on 2x Quadro RTX 8000, AOT cache on, cold start
fine, every warm start fails in the profiling run:

```
torch.ops._C.fp8_qpn8_dispatch_sm70_out.default(buf0, 125053979066368, buf2, ...)
RuntimeError: The specified pointer resides on host memory and is not registered with any CUDA device.
```

On current main the sites above are latent in the default V100 setup: the
Flash-V100 compile graph auto-sets VLLM_DISABLE_COMPILE_CACHE=1, so no
artifact is reloaded. They become live with the compile cache on (explicit
VLLM_DISABLE_COMPILE_CACHE=0, or #621) on a route that reaches them, for
example block-FP8 ([128, 128]) layers on the QPN8 path. The same
27B on 2x V100 does not reach them: its per-tensor FP8 layers run
`fp8_gemm_sm70_out`, which takes no workspace.

New module `quantization/utils/sm70_layer_workspaces.py`: prepared layers
register their workspace under the layer prefix, which is identical in every
process, and apply() passes the prefix to four opaque custom ops
(`vllm::sm70_fp8_qpn8_dispatch`, `vllm::sm70_fp8_prefill_dispatch`,
`vllm::sm70_fp8_qpn8_dispatch_ba_split`, `vllm::sm70_nvfp4_qpn4_dispatch`)
that resolve the address at run time, as #622 does for the PLE table. A lookup
by weight size like #672's would not be enough: depending on the route a layer
holds the prefill-dense, the PP2 x TP4 or the QPN4 workspace. Registering a
second workspace for one prefix, or a layer without a prefix, raises. fp8.py
keeps `sm70_fp8_prefill_exact_dense_workspace_ptr` as the "workspace present"
marker other gates read; the QPN4 state drops its pointer field.

Not covered: the opt-in online QPN8 path (sm70_online_qpn8.py) passes its own
workspace pointer the same way (two sites) and can register with the same
module in a follow-up. #604 copies the pattern in its Turing QPN8 route;
I will update it to register its workspace here, stacked on this PR.

**Why this is not a duplicate.** #672 touches only
compressed_tensors_w8a16_fp8.py. #675 (Triton kernel side table) and #682
(autotuning) address other causes of AOT reload failures. #661 pins
Flash-V100 decode workspaces during graph capture, a different allocation in
a different file. No open PR or issue mentions these linear workspace
pointers or this error (gh pr list / gh search issues for "workspace",
"workspace_ptr", "resides on host memory", "AOT").

AI assistance was used for this change. Every line was reviewed and the tests
below were run by the submitter.

## Test Plan

```bash
pytest tests/quantization/test_sm70_fp8_workspace_aot_reload.py \
  tests/quantization/test_sm70_fp8_prefill_exact_dense.py \
  tests/model_executor/test_sm70_fp8_qpn8_pp2_tp4.py \
  tests/quantization/test_sm70_ct_fp8_cache_workspace.py \
  tests/quantization/test_sm70_turbomind_adapter.py \
  tests/model_executor/test_qwen3_5_quantization.py
```

The new test reproduces the failure mechanism on main's code: it exports a
module that calls each op, saves it, registers a different workspace, reloads,
and asserts the op now receives the new address. With an integer argument the
reloaded graph would pass the old one.

End to end: Qwen3.8-27B-NVFP4, TP2, MTP, cold start then warm start from the
AOT artifact, (a) on 2x RTX 8000 through #604 on top of this PR, (b) on
2x V100 on main with VLLM_DISABLE_COMPILE_CACHE=0, without and with this PR.

## Test Result

On this branch (rebased onto 14abfc27; extensions from a build of db292f9a,
the commits in between change TurboMind GEMM kernels and one Triton file):
71 passed on a Tesla V100 and 71 passed on a Quadro RTX 8000. pre-commit and
mypy-3.10 clean.

End to end (a): without the change every warm start fails as quoted above;
with it the warm start loads the artifact (6 AOT loads) and serves, greedy
output identical to the cold start. The artifact now reads
`torch.ops.vllm.sm70_fp8_qpn8_dispatch.default(buf0, 'language_model.model.layers.39.self_attn.o_proj', ...)`
instead of an address.
(b): cold and warm start pass both without and with the change, greedy output
identical across all four runs (this model does not reach the changed sites on
Volta, see above), so the change is neutral there.

The existing dispatch tests set a fake pointer (42, 123, 1234) on a
SimpleNamespace or a module; they now register a workspace and run the opaque
ops through a CPU kernel for the test's duration. That fixture also registers
#672's op, whose CPU-tensor test
`test_compressed_tensors_channel_fp8_qpn8_prepares_and_dispatches` in
test_sm70_fp8_prefill_exact_dense.py fails on current main with "Could not run
'vllm::sm70_ct_fp8_qpn8_dispatch' with arguments from the 'CPU' backend". The
same test still fails when it runs after test_sm70_batched_gemm_layouts.py,
on main as on this branch: another test replaces
torch.ops._C.fp8_sm70_prepare with object() and the replacement leaks into it.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
