ENTWURF 15.10.2026 früh, NICHT gesendet (PR erst auf Peuquis Freigabe). Branch pr-gdn-traced-prefill-norm bdaa680b1 im Fork =
origin/main 6f3557c97 + Cherry-Pick von aa7087485. Vor dem Senden: Duplikatsuche wiederholen, main-SHA neu abfragen, Konflikt prüfen.
Quellen: Fehlstart ab_fm1013_2026-10-14.log (fm13pp4-fm-kalt, Journal quality_2026-09-27/fm2-fm13pp4-fm-kalt-001514.journal.txt),
Abnahme mit Fix ab_fm1013b_2026-10-15.log (PP4, TP2xPP2, 27B je fm-kalt + A/B/A/B, alle Boots ok).
Duplikatsuche 15.10. ~00:30: gh pr list --state all --search "device_index" / "RMSNormGated" / "is_compiling" / "gdn norm" /
"Dynamo" / "output projection": kein PR zu dieser Stelle; die offenen #1204-#1211 enthalten die Zeile nicht (nur
torch.accelerator.set_device_index in Benchmark-Skripten).
Titel: [Bugfix][GDN] Keep the traced prefill output projection on the custom-op norm
---
## Purpose

011a8ce1e (#1203) routes the gated RMSNorm of the Qwen GDN output projection to `RMSNormGated.forward_cuda` for every prefill with more than 64 tokens. The comment there says this is meant for the path behind the opaque Qwen GDN op. With the output-projection boundary unset, which is the default (`kernel_config.gdn.projection.output_projection` is None), the projection is traced by Dynamo instead. Dynamo then reaches the FLA input guard's `torch.accelerator.device_index(...)` in vllm/model_executor/layers/fla/ops/utils.py, which is on its skip list, and engine startup fails during the profile run:

  torch._dynamo.exc.Unsupported: Attempted to inline function marked as skipped
  ... qwen_gdn_linear_attn.py, in _compute_output_projection: core_attn_out = self.norm.forward_cuda(core_attn_out, z)
  ... fla/ops/utils.py, line 110, in wrapper: ctx = torch.accelerator.device_index(tensor.device.index)

This PR takes the direct Triton norm only outside Dynamo tracing (`not torch.compiler.is_compiling() and num_tokens > 64`). A traced projection keeps `self.norm(...)` as before 011a8ce1e; eager and opaque callers still take the direct route. The DFlash2 fused-norm condition is unchanged.

Seen on Qwen3.8-Flash-Next-180B NVFP4 with MTP, PP4 (2x Quadro RTX 8000 + 2x Tesla V100): it failed to start on main 6f3557c97 with the traceback above and starts with this change. With the change, the same model in TP2xPP2 and Qwen3.8-27B NVFP4 in TP2 also start and serve normally (not run without it).

Not a duplicate: gh pr list --state all --search "device_index", "RMSNormGated", "is_compiling", "gdn norm", "Dynamo", "output projection" finds no PR touching this condition; the open #1204-#1211 do not change it.

AI assistance was used for this change. I reviewed every line and ran the tests and measurements below.

## Test Plan

```bash
pytest tests/kernels/test_gdn_prefill_norm_trace.py
pre-commit run --from-ref origin/main --to-ref HEAD
```

The new test builds a minimal GDN layer, runs the output projection eagerly for 128 tokens (expects one direct forward_cuda call) and then under torch.compile(fullgraph=True, backend="eager") (expects no direct call and the same result within 2e-3).

Serving: Qwen3.8-Flash-Next-180B NVFP4 MTP PP4 and TP2xPP2 and Qwen3.8-27B NVFP4 TP2, each one cold boot plus two boots alternating with our previous build, greedy and quality prompts compared by hand.

## Test Result

- tests/kernels/test_gdn_prefill_norm_trace.py on main 6f3557c97 with this change: 1 passed on a Tesla V100 and on a Quadro RTX 8000. With the qwen_gdn_linear_attn.py change reverted, it fails on both with torch._dynamo.exc.Unsupported (the same error as the engine startup).
- pre-commit run --from-ref origin/main --to-ref HEAD: all hooks pass.
- Serving: all boots succeed with the change; PP4 failed at startup without it (the only one of the three run without it).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
