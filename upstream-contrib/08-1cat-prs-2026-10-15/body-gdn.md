## Purpose

011a8ce1e (#1203) routes the gated RMSNorm of the Qwen GDN output projection to `RMSNormGated.forward_cuda` for every prefill with more than 64 tokens. The comment there says this is meant for the path behind the opaque Qwen GDN op. When the projection is not behind an opaque boundary, Dynamo traces this branch while compiling, reaches the FLA input guard's `torch.accelerator.device_index(...)` in vllm/model_executor/layers/fla/ops/utils.py, which is on its skip list, and engine startup fails during the profile run (traceback abridged):

```
torch._dynamo.exc.Unsupported: Attempted to inline function marked as skipped
  qwen_gdn_linear_attn.py, in _compute_output_projection: core_attn_out = self.norm.forward_cuda(core_attn_out, z)
  fla/ops/utils.py, line 110, in wrapper: ctx = torch.accelerator.device_index(tensor.device.index)
```

Both opaque boundaries are off there by default: `VLLM_SM70_QWEN_GDN_OUTPUT_PROJECTION_OP` is a diagnostic switch that defaults to 0, and the automatic full-forward wrapper is armed only for compiled MTP engines on Volta (`auto_sm70_qwen_gdn_full_forward and _sm70_current_device_is_volta()`). We hit it with Qwen3.8-Flash-Next-180B NVFP4 + MTP in PP4, where pipeline stage 0 runs on a Quadro RTX 8000: that worker failed with the traceback above on main 6f3557c97.

This PR takes the direct Triton norm only outside Dynamo tracing (`not torch.compiler.is_compiling() and num_tokens > 64`). A traced projection keeps `self.norm(...)`, as before 011a8ce1e; eager and opaque callers still take the direct route. The DFlash2 fused-norm condition is unchanged. `is_compiling()` is checked first, so tracing does not add a guard on `num_tokens`.

Not a duplicate: gh pr list --state all --search "device_index", "RMSNormGated", "is_compiling", "gdn norm", "Dynamo", "output projection" finds no PR touching this condition, and the open #1204 to #1211 do not change it.

AI assistance was used for this change. I reviewed every line and ran the tests and measurements below.

## Test Plan

```bash
pytest tests/kernels/test_gdn_prefill_norm_trace.py
pre-commit run --from-ref origin/main --to-ref HEAD
```

The new test builds a minimal GDN layer, runs the output projection eagerly for 128 tokens (expects one direct `forward_cuda` call) and then under `torch.compile(fullgraph=True, backend="eager")` (expects no direct call and the same result within 2e-3).

Serving on our fork (main 6f3557c97 plus our fork's commits and this change): Qwen3.8-Flash-Next-180B NVFP4 + MTP in PP4 (2x Quadro RTX 8000 + 2x Tesla V100) and TP2xPP2, Qwen3.8-27B NVFP4 TP2, each one cold boot plus two boots alternating with our previous build; greedy and quality prompts compared by hand.

## Path coverage

| Path / provider | Format | Key conditions (shape, TP/EP, graph, capability) | Expected selection / fallback | Evidence (source / static / executed / measured) | Unverified scope |
| --- | --- | --- | --- | --- | --- |
| Qwen GDN output projection, gated RMSNorm | FP16 | num_tokens > 64, projection traced by Dynamo (no opaque output-projection op, no full-forward wrapper) | `self.norm(...)` as before 011a8ce1e | executed: new test on V100 and RTX 8000; serving PP4 / TP2xPP2 / TP2 | none |
| Qwen GDN output projection, gated RMSNorm | FP16 | num_tokens > 64, eager or behind an opaque op | direct `RMSNormGated.forward_cuda`, unchanged | executed: new test (eager part) | opaque-op runtime not run here |
| DFlash2 fused GDN norm | FP16 | `enable_sm70_dflash2_fused_gdn_norm` | direct `forward_cuda`, unchanged | static: condition unchanged | not run |

## Test Result

- tests/kernels/test_gdn_prefill_norm_trace.py on main 6f3557c97 with this change: 1 passed on a Tesla V100 and on a Quadro RTX 8000. With the qwen_gdn_linear_attn.py change reverted, it fails on both with `torch._dynamo.exc.Unsupported` (the same error as the engine startup).
- pre-commit run --from-ref origin/main --to-ref HEAD: all hooks pass.
- Serving: all boots succeed with the change; PP4 failed at startup without it (the only one of the three configurations run without it).

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01Ly7LM3m9fa6DRJPYQjnt95
