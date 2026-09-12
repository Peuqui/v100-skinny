# PR-Entwurf 1Cat: KV-Quant-Vorgabe eines Checkpoints unterhalb Ampere ignorieren — VERÖFFENTLICHT als PR #613 (12.09. 21:30, Commit 8c98b85c)

Worktree `1Cat-vLLM-pr-kvpolicy`, Branch `kv-quant-policy-pre-ampere` auf origin/main dfef3342.
Herkunft: Overlay `utils/torch_utils.py` + `layers/attention/attention.py` (OVERLAY-INVENTUR Zeile 126),
für den PR neu geschnitten: Entscheidung über die TEILNEHMENDEN Geräte (bestehender Helfer
`_any_participating_device_is_pre_ampere` aus #579), kein Gerät-0-Gate, kein try/except-Fallback,
kein Roh-Env-Schalter (die explizite `--kv-cache-dtype` ist der Override).

Titel: [Bugfix][SM70/SM75] Honor a checkpoint's KV-cache quantization directive only on Ampere and newer

---

With --kv-cache-dtype auto (the default), vLLM resolves the cache dtype from the checkpoint: a ModelOpt config with kv_cache_quant_algo FP8, or a compressed-tensors config with a kv_cache_scheme, turns the KV cache into FP8 without the user asking for it. That metadata describes how the weights were produced. On Volta and Turing there is no FP8 hardware, and honoring it there is a loss both ways:

On Volta the FP8 cache is unpacked in software and decode attention leaves the tensor-core route. Measured on 4x V100 with Qwen3.8-27B (results in v100-skinny, mixed_regression_closed_20260818): with the checkpoint's FP8 cache the decode attention route changes from xqa_tc to scalar_paged and a decode round costs 4.82 ms more, 4.5x the cost of the FP8 weights the checkpoint ships alongside. New measurement on this branch, 2x V100 TP2, same checkpoint, --kv-cache-dtype unset: 60.71 tok/s with the directive ignored (fp16 cache) against 58.74 tok/s on main with the directive honored (fp8_e4m3 cache), identical 200-token output text in both runs (short prompt, MTP k=3, greedy; the 4.82 ms figure above was taken at long context on TP4, where the attention share is larger).

On Turing the boot does not survive at all: the FlashAttention backend rejects an fp8 cache below FA3, and the Triton path fails in the compiled cache write with "type fp8e4nv not supported in this architecture" (torch.compile Inductor kernel, sm75). Today a Turing user with unsloth's or RadixArk's Qwen3.8-27B-NVFP4 has to know to pass --kv-cache-dtype float16; nothing tells them.

This change honors the checkpoint directive only when every participating CUDA device is Ampere or newer. It reuses the participation logic from #579 (_participating_cuda_device_ids / _any_participating_device_is_pre_ampere), so a Turing card that is visible but not part of the engine does not change the decision, and a mixed rig decides once for all stages. An explicit --kv-cache-dtype is never touched: it does not go through the "auto" resolution and the new marker on CacheConfig is only set for a request of "auto".

Changes:
- vllm/config/cache.py: CacheConfig.cache_dtype_from_checkpoint (derived, init=False) and the cache_dtype docstring.
- vllm/engine/arg_utils.py: set the marker where "auto" is resolved.
- vllm/config/vllm.py: checkpoint_kv_quant_allowed(cfg) next to the existing pre-Ampere helper; VllmConfig.__post_init__ drops a checkpoint-resolved dtype back to "auto" on pre-Ampere devices, with one info line naming the ignored value and the override. The "Using fp8_e4m3 data type to store kv cache" line from CacheConfig's validator still precedes it, because the dtype is resolved before the parallel config (and with it the device participation) exists; the ignore line names the value so the two read together.
- vllm/model_executor/layers/attention/attention.py: the compressed-tensors re-apply path (kv_cache_scheme with cache_dtype "auto") follows the same policy, so it cannot re-quantize what the resolve path just refused.
- tests/config/test_checkpoint_kv_quant_policy.py: 10 tests (policy by participating devices incl. visible-but-unused Turing, drop on pre-Ampere, keep on Ampere, explicit request untouched).

Tests: tests/config/test_checkpoint_kv_quant_policy.py 10 passed; tests/config/test_sm70_gates_any_visible_device.py + tests/test_config.py 220 passed, 3 failed on main and on this branch alike (test_rope_customization, test_is_encoder_decoder, test_eagle_draft_model_config: HuggingFace download in an offline environment). pre-commit and mypy-3.10 clean on the changed files.

Duplicate check: issue 489 and PRs 450/49 concern the explicit fp8_e5m2 request with a compressed-tensors kv_cache_scheme on the Flash-V100 unit-scale path; this change is about the implicit "auto" resolution and leaves explicit requests alone. No open PR touches resolve_kv_cache_dtype_string or the attention.py re-apply block.

This work was done with AI assistance (Claude); the change is reviewed, tested and measured by me on the hardware named above.

## Belege (12.09. 21:00-21:20)

| 2x V100 (GPU 1,3) TP2, 27B-NVFP4, MTP k=3, greedy, --kv-cache-dtype unset | KV-Typ | tok/s (Median 3x200) | SHA |
|---|---|---|---|
| Branch mit Politik (`1Cat-vLLM-pr-kvpolicy`) | auto → fp16, Log "Ignoring … (fp8_e4m3)" | 60,71 | c613080edbcf8967 |
| main dfef3342 ohne Politik (`1Cat-vLLM-pr-fa2sm75`, FA2-Diffs wirken auf V100 nicht) | fp8_e4m3 | 58,74 | c613080edbcf8967 |

RTX-Paar mit Politik (`kvpolicy_patch`): Log-Zeile greift, Boot scheitert danach an
`modelopt_mixed` Mindest-Capability 89 (das behebt #604), also kein Turing-E2E auf reinem Main.
Turing-Kontrolle ohne Politik: `~/.cache/mtp-diagnostics/qual_e1t_rtx_mtp/boot.log` (fp8e4nv-Absturz).
Rohdaten: `~/.cache/mtp-diagnostics/kvpolicy_{v100_patch,v100_control,patch}/boot.log`,
Skripte `handover/2026-09-12/kvpolicy_probe.sh`, `kvpolicy_chain.sh`.
