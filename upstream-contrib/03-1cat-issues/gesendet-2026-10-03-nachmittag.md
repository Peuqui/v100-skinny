# Gesendet 03.10.2026 nachmittags (Live == Entwurf jeweils geprüft)

- PR #856 https://github.com/1CatAI/1Cat-vLLM/pull/856 (Branch pr-ple-cascade-admission-names, Kopf 8b1dacdd1, Basis main 8002bc107)
- PR #857 https://github.com/1CatAI/1Cat-vLLM/pull/857 (Branch pr-dsv4-sm70-sparse-policy-at-init, Kopf 0557cce46, Basis main 8002bc107)
- Kommentar #837 https://github.com/1CatAI/1Cat-vLLM/pull/837#issuecomment-5970301676
- #674 Text aktualisiert (Update 2026-10-03, afternoon)

## PR #856

## Purpose

The disk cascade admission from #806 (`_qwen4exp_ple_cascade_requested`) asks the quantization config whether every PLE table is stored as raw E4M3. It builds the lookup prefix as `model.layers.{id}.ple.ple_embedding.ngram_embedding`, which misses in two ways:

- `ple_layer_ids` are 1-based: id L is the PLE module of decoder layer L - 1, as `check_ple_layers_on_first_pp_rank` already documents. The query names the wrong decoder layer.
- The model's `hf_to_vllm_mapper` is applied to the quantization config only when the model is built. During admission the layer metadata still uses checkpoint names (`model.language_model.layers.N...`), which the vLLM-side prefix never matches.

ModelOpt mixed-precision checkpoints declare the FP8 table per layer in `quantized_layers` instead of setting `ple_embedding_dtype`. On nvidia/Qwen3.8-Flash-Next-NVFP4 (`ple_layer_ids` [2], table under `model.language_model.layers.1.ple.ple_embedding.ngram_embedding`) the admission therefore always reports "checkpoint metadata does not provide raw E4M3 PLE storage", and the engine materializes the whole table on the first pipeline stage. On main a692497bc a PP4 start failed with

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 47.69 GiB. GPU 0 has a total capacity of 47.27 GiB
```

and TP2xPP2 stopped with "available KV cache memory (0.03 GiB)".

This PR queries the checkpoint name of decoder layer id - 1. The existing admission tests only use the forced E4M3 storage path (`ple_embedding_dtype="float8_e4m3fn"`), where the prefix is never evaluated; the new test uses ModelOpt metadata shaped like that checkpoint and also rejects a table registered under the raw id.

A second commit fixes `test_qwen4exp_ple_cascade_starts_the_offload_worker`, which fails on any host with a visible SM70 GPU, also on unchanged main: `VllmConfig()` itself applies the Flash-V100 baseline defaults (`VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE` and the SM70 GDN schedules) after the test took its environment snapshot. The snapshot is now taken after the config is built, which is what the test checks.

Not a duplicate: no open PR touches the admission's storage lookup (gh pr list / gh search for "raw E4M3 PLE storage", "ple cascade admission", "ple_layer_ids"; #702, #821 and #831 change other parts of the PLE configuration and leave this function unchanged).

AI assistance was used for this change. I reviewed every line and ran the tests below.

## Test Plan

```bash
pytest tests/config/test_ple_cascade_capabilities.py
pytest tests/compile/test_sm70_decode_graph.py
pre-commit run --files vllm/config/vllm.py tests/config/test_ple_cascade_capabilities.py tests/compile/test_sm70_decode_graph.py
pre-commit run mypy-3.10 --hook-stage manual --files <same>
```

## Acceleration and benchmark contract (required for performance changes)

Not a performance change. It changes which placement the existing automatic admission selects for ModelOpt checkpoints with FP8 PLE tables.

## Test Result

Both files: 62 passed on a Tesla V100 host and with no visible GPU. On unchanged main the new test fails for the correctly registered table (mutation check), and a variant that only fixes the name, not the layer index, fails both new cases. Without the second commit, `test_qwen4exp_ple_cascade_starts_the_offload_worker` fails on the V100 host and passes without a GPU. pre-commit and mypy-3.10 pass.

Admission on the real checkpoint, building `VllmConfig` from the serving arguments without loading weights: main reports `ple_disk_cascade_active=False` with the reason above for PP4 and TP2xPP2; with this change both report `True` with no reason.

End to end on 3x Tesla V100 + 2x Quadro RTX 8000, with this change on top of the #837 head (035be3644, extensions built from source): Qwen3.8-Flash-Next-NVFP4 starts at PP4 and at TP2xPP2. At PP4, stage 0 keeps 17.35 GiB of the table and the cascade worker serves the remaining 30.34 GiB from the checkpoint mapping; the KV cache holds 514,527 to 520,104 tokens. Over three warm boots a 29k-token prompt took 13.9 to 16.0 s to the first token at PP4 and 18.7 to 18.8 s at TP2xPP2. Greedy outputs and our eight reference questions, read by hand, agree in substance with our production build (one of three greedy prompts identical, two equivalent rewordings).

🤖 Generated with [Claude Code](https://claude.com/claude-code)

## PR #857

## Purpose

The sparse batched-matmul admission (`_bmm_blocker` in `vllm/models/deepseek_v4/sm70/sparse.py`) and the SM70 indexer decode (`sm70_indexer_decode_logits` in `vllm/models/deepseek_v4/sm70/indexer.py`), both from #816, read `KernelConfig.sm70_sparse` through `get_current_vllm_config()` inside the forward pass. The worker sets that context while it builds, loads and captures the model (`load_model`, `initialize_from_config`, `profile_cudagraph_memory`), but not for the memory profile in `determine_available_memory` or for `execute_model`. On DeepSeek-V4-Flash at PP5 on 3x V100 + 2x Quadro RTX 8000, every start on main stopped in `profile_run` on a V100 stage:

```
AssertionError: Current vLLM config is not set. This typically means get_current_vllm_config() was called outside of a set_current_vllm_config() context, ...
  File "vllm/models/deepseek_v4/sm70/sparse.py", line 59, in _bmm_blocker
    policy = get_current_vllm_config().kernel_config.sm70_sparse
```

An eager decode reaches the same kind of lookup in the indexer. The existing tests call these functions inside `set_current_vllm_config()` (autouse fixtures in test_deepseek_v4_sm70_routes.py and test_deepseek_v4_sm70_indexer.py), so they did not see it.

This PR keeps the per-engine policy with modules that are built while the config is set, the same way #838 keeps the auxiliary GEMV policy: `DeepseekV4MLAAttention` and the indexer K caches (`DeepseekV4IndexerCache`, and `DeepseekV32IndexerCache`, which the GLM-5.3 kpool cache extends) store `kernel_config.sm70_sparse`. `_bmm_blocker` takes the layer's policy, and the two indexer ops pass the setting of the cache registered under their layer name (`no_compile_layers[k_cache_prefix]`). Nothing changes when the context is set.

Not a duplicate: no open PR touches these lookups (gh pr list / gh search for "sm70_sparse", "_bmm_blocker", "get_current_vllm_config forward", "Current vLLM config is not set"). #838 and #847 recently changed attention.py for the auxiliary GEMV; neither touches these two lookups.

AI assistance was used for this change. I reviewed every line and ran the tests below.

## Test Plan

```bash
pytest tests/models/test_deepseek_v4_sm70_sparse_policy.py   # new, runs without a config context
pytest tests/models/test_deepseek_v4_sm70_routes.py
pytest tests/kernels/test_deepseek_v4_sm70_indexer.py
pytest tests/kernels/attention/test_dsv4_sm70_sparse_bmm.py tests/kernels/attention/test_dsv4_sparse_prefill_bmm.py
pytest tests/models/glm5next/test_sm70_sparse.py
pre-commit run --files vllm/models/deepseek_v4/sm70/sparse.py vllm/models/deepseek_v4/sm70/indexer.py vllm/models/deepseek_v4/attention.py vllm/model_executor/models/deepseek_v2.py vllm/model_executor/layers/sparse_attn_indexer.py vllm/model_executor/layers/sparse_attn_indexer_kpool.py tests/kernels/test_deepseek_v4_sm70_indexer.py tests/models/test_deepseek_v4_sm70_routes.py tests/models/test_deepseek_v4_sm70_sparse_policy.py
pre-commit run mypy-3.10 --hook-stage manual --files <same>
```

## Acceleration and benchmark contract (required for performance changes)

Not a performance change. The admission decisions are the same; only where the policy is read from moves.

## Test Result

The new test fails on main (3 of 3: `_bmm_blocker` raises the assertion above, and the caches have no policy) and passes with this change. On a Tesla V100 all six files pass (96 passed); on a Quadro RTX 8000 78 passed and 18 skipped, all by their stated Volta-only conditions. Routes and the new test also pass without a visible GPU (32 passed). pre-commit and mypy-3.10 pass.

End to end on 3x V100 + 2x Quadro RTX 8000, with this change on top of the #837 head (035be3644, extensions built from source; the touched code is identical on current main): DeepSeek-V4-Flash at PP5 with the DSpark drafter started in both boots, logged "DeepSeek V4 SM70 sparse MLA decode: batched matmul", and gave one of three greedy prompts identical to our production build and two equivalent ones (a different correct prime routine, the same train-speed derivation with fractions typeset differently). Without the change the same entry failed to start in both attempts with the assertion above.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

## Kommentar #837

Native results for this integration on our rig (3x Tesla V100-PCIE-32GB + 2x Quadro RTX 8000, CUDA 12.8, Torch 2.10.0+cu128). The extensions were built from source at 035be3644 through the repository's CMake target with TORCH_CUDA_ARCH_LIST=7.0, the sm_70-only build we deploy; the RTX 8000 runs those sm_70 cubins. The skinny MoE files are identical between 035be3644 and the merged head 61db7c02b, and current main (8002bc107) differs from them only by the compile-hash bookkeeping of #847.

SM75 operator cases: tests/kernels/moe/test_skinny_sm70_moe.py and tests/kernels/moe/test_unquantized_backend_selection.py with one RTX 8000 and one V100 visible (CUDA_DEVICE_ORDER=PCI_BUS_ID, CUDA_VISIBLE_DEVICES=0,1): 91 passed, 1 skipped (the ROCm selection case). All 13 sm75 cases pass on the RTX 8000 and all 13 sm70 cases on the V100.

Complete models, on 035be3644 plus two Python-only fixes this rig needs on main: #856 (PLE cascade admission for ModelOpt checkpoints) and #857 (sparse policy read outside the config context). With moe_backend left at auto the oracle chose the skinny experts by itself. Stage 0 is a Quadro RTX 8000 in every layout and logged "Using 'SM70_SKINNY' NvFp4 MoE backend" for Qwen3.8-Flash-Next-NVFP4 (512 experts, one scale per 16 codes) and "Using 'SM70_SKINNY' Mxfp4 MoE backend" for DeepSeek-V4-Flash (256 experts, one scale per 32 codes). The other ranks do not log the choice, since info_once is local-rank only.

Paired runs on the same build, Qwen3.8-Flash-Next-NVFP4 at TP2xPP2 (stage 0 two RTX 8000, stage 1 two V100, MTP with 4 draft tokens), default against --kernel-config '{"sm70_skinny_moe":false}'. The first boot of each arm and the first long request after every boot are left out; two warm boots with skinny, one without. With the setting off, stage 0 logged "Using 'MARLIN' NvFp4 MoE backend".
- 29k-token prompt, time to first token (first send of each prompt): 18.7 to 18.8 s with skinny, 25.6 to 26.3 s without.
- Decode step at 29k tokens: 59 to 60 ms with skinny, 60 to 61 ms without; short prompts 50 to 53 ms against 51 to 54 ms.
- KV cache: 804,558 tokens with skinny, 698,585 without.

At PP4 (RTX, RTX, V100, V100) with the MTP drafter, the run with the setting off did not start in three of three boots: stage 3 ran out of memory while stacking the FP8 MTP experts for TurboMind (fp8_sm70_moe.py, torch.stack(w13_tm_weights): "Tried to allocate 1.56 GiB. GPU 3 has a total capacity of 31.73 GiB of which 1.03 GiB is free"). With the default it starts and serves.

Quality: against our fork, which runs the original #742 path with our tuned split-K settings, the three greedy prompts gave one identical output and two equivalent rewordings at PP4, at TP2xPP2 and for DeepSeek-V4 at PP5, the same in every boot. Our eight reference questions, read by hand, were equivalent; on Qwen3.8-Flash-Next the train question reached the 600-token limit on the same correct derivation in several runs of both builds.

DeepSeek-V4-Flash at PP5 with the DSpark drafter, for reference: the 35.7k-token prompt took 17.1 s to the first token and the decode step at that length was 125 to 127 ms, against 15.3 s and 85 to 86 ms on our fork in the same session. These runs differ from our fork in more than the MoE path; for example the startup log shows the indexer on the paged Triton route under full graphs ("the cuBLAS route needs a live key bound rather than a fixed full-graph bucket"), where our fork takes the cuBLAS route. We have not separated the MoE share of that difference yet and will report back.

Method: each arm was booted through our serving setup with our production arguments minus our fork-only switches and without --moe-backend; four 29k-token prompts, each sent twice, then three short decode prompts and the reference questions; the PLE checkpoint file was dropped from the page cache before every boot. AI assistance was used for running and analysing these measurements.

## #674 (vollständiger neuer Text)

Update 2026-10-03, afternoon: #742 is merged through #837, thank you. Native results for it are posted in #837: the 13 SM75 cases on a Quadro RTX 8000, complete-model runs, and a paired skinny on/off comparison. Two new fixes came out of those runs. #856: the PLE cascade never admitted ModelOpt checkpoints such as nvidia/Qwen3.8-Flash-Next-NVFP4, because the admission looked up the wrong layer under the wrong name, so a PP4 start tried to put the whole 47.69 GiB table on stage 0. #857: DeepSeek-V4 could not start on our V100/RTX 8000 rig, because the sparse policy was read through get_current_vllm_config() inside the forward pass, where the worker does not set it. Four PRs are open, and all of them merge cleanly into current main (8002bc107).

Update 2026-10-03, morning: thank you for the overnight round. #791 and #793 went in through #795, #623 through #797, #747 through #798, #604 through #804, #750 through #801, #740 through #808, #646 and #717 through #806, and #714 and #716 through #816. GPU results for #795, #797 and #798 are posted there. Three PRs are still open, and all of them merge cleanly into current main (3efe512d).

Update 2026-10-02, evening: thank you for the integration round. #743, #710, #749 and #726 went in through #773, #720 through #776, #744 and #745 through #777, #667 through #765, and #711 is merged. #646 and #717 are rebased onto main after the hybrid PLE change of #786, and #747 and #750 onto main after #777 and #782; #750 now registers its switch in the env metadata of #782. #750 also got a test fix I had missed: its PP2xTP4 tests built Fp8LinearMethod without use_qpn8. Two new fixes: #791 for the DFlash/DSpark start that #748 broke, and #793, which settles the hybrid block size before the PLE KV estimate; with the cascade of #646, Qwen3.8-Flash-Next needs it to start. 14 PRs are still open, and all of them merge cleanly into current main (a3498d46).

Update 2026-10-02: #621 and #723 are merged, and #725, #741 and #752 went in through #757, #758 and #762, thank you. 21 PRs are still open, and all of them merge cleanly into current main (e0f3fedd).

Update 2026-10-01, evening: one more, #752. It asserts that each request with drafts has at least draft_len + 1 query rows, so a scheduler/runner mismatch fails where it starts instead of verifying drafts against the preceding request's hidden states. One comparison per request. It has been in our fork since 2026-09-06 and has not fired in the logs we keep (back to 2026-09-21).

Update 2026-10-01, afternoon: two more. #750 lets every [128, 128] block-FP8 linear on Volta and Turing use the native QPN8 operators behind an opt-in switch (DeepSeek-V4 at PP5: prefill 16.7 -> 15.2 s, decode step 100 -> 85 ms, and no TurboMind tuning stall at new prompt lengths); stacked on #747. #749 checks the token ids the last pipeline stage hands to the others, so an out-of-vocabulary id is named instead of ending in an anonymous device-side assert. #744 now covers six test modules. Main plus our open PRs, with the #716 and #750 switches set, now produces output identical to our fork on DeepSeek-V4 (greedy and eight reference answers).

Update 2026-10-01, morning: three more PRs, all about DeepSeek-V4 before Ampere. #747 lets Turing take main's SM70 route (one capability-family predicate, so V100 and RTX stages of one pipeline declare the same backend) and gives wo_a a grouped fp16 path where Marlin serves block FP8. #745 decodes FP8 in software on every card without FP8 units, and #744 brings five DeepSeek-V4 test modules in line on pre-Ampere cards. #716 got a fix: its batched-matmul prefill read stale index slots past each token's length and hit a device assert on the first long prompt. Our fork now runs #747 + #716 instead of its own pre-Hopper path; on the same build that was 4 to 6 % faster (tables in #747). All 25 open PRs merge cleanly into current main (d3046986).

Update 2026-09-30, evening: #742 is the MoE part of the v100-skinny proposal in #718, and #743 fixes a memory peak in the SM70 MXFP4 repack that kept DeepSeek-V4 off our V100 stages on the default path. Earlier that day: #740, #741 and the bug report #739. #727 is closed, since #733 covers the same ground more broadly. 22 of the PRs below are merged, thank you for going through them.

Still open

- #856 Look up PLE cascade storage under checkpoint layer names. With it, the cascade admits ModelOpt mixed-precision checkpoints that declare the FP8 PLE table in quantized_layers; on nvidia/Qwen3.8-Flash-Next-NVFP4 PP4 and TP2xPP2 start again. A second commit fixes a cascade test that fails on any host with a visible SM70 GPU.
- #857 Read the SM70 sparse policy when the layers are built. Without it, DeepSeek-V4 at PP5 on our rig stopped in the memory profile with "Current vLLM config is not set".
- #715 Keep hc_head finite under float16 for attention-sink rows. Follow-up to #658, which left the fused hc_head kernel out; its store now saturates like the others.
- #611 Block-pack the activations of the NVFP4 QPN2 kernels (being integrated in the draft #822)

Bug report: #739, two tests fail on current main (a stale QSA E4M3 expectation, and the PLE offload config with a single visible GPU).

#718 proposed bringing the v100-skinny kernels into csrc; its MoE part, #742, is merged through #837. Block FP8 needs no port: main's native QPN8 kernels already match or beat it on DeepSeek-V4's shapes. A correction to what this overview said before: main does run DeepSeek-V4's MXFP4 MoE on Turing, through Marlin.

Merged

- #742 Skinny QPN MoE backend for NVFP4 and MXFP4 on SM70 and SM75 (merged as part of #837)
- #604 Run ModelOpt NVFP4 and FP8 linears on Turing through the SM70 QPN kernels (merged as part of #804)
- #750 Block FP8 through the native QPN8 operators on SM70/SM75 (merged as part of #801)
- #740 Add a direct I/O safetensors load strategy (merged as part of #808)
- #646 PLE overflow cascade: device, pinned host, disk (merged as part of #806)
- #717 Allow the PLE CPU offload under pipeline parallelism (merged as part of #806)
- #714 Take the DeepSeek-V4 cuBLAS indexer decode under speculative decoding (merged as part of #816)
- #716 DeepSeek-V4 sparse MLA as gather + batched matmul (merged as part of #816)
- #791 Copy the DFlash/DSpark draft config with vLLM's pydantic-aware replace (merged as part of #795)
- #793 Settle the hybrid block size before estimating the PLE KV budget (merged as part of #795)
- #623 Build and load a Turing FlashAttention-2 library next to the Volta one (merged as part of #797)
- #747 Run the DeepSeek-V4 SM70 route on Turing (merged as part of #798)
- #743 Repack DeepSeek-V4 MXFP4 experts for TurboMind without a second copy (merged as part of #773)
- #710 Resolve SM70 linear workspace addresses on AOT reload (merged as part of #773)
- #749 Check the token ids the last pipeline stage hands over (merged as part of #773, which extends the check to the whole payload)
- #726 Bring stale core and SM70 tests in line with current main (merged as part of #773)
- #711 Reject new VLLM_* environment reads that bypass envs.py
- #720 Size the ROCm SWA ragged copy from the actual row width (merged as part of #776)
- #744 Run the DeepSeek-V4 test modules on pre-Ampere cards (merged as part of #777)
- #745 Decode FP8 in software on every card without FP8 units (merged as part of #777)
- #667 Reuse a sliding window's dead blocks before other requests' cached ones (merged as part of #765)
- #752 Assert enough query rows for each request's drafts (merged as part of #762)
- #725 Keep the SM70 cudagraph size cap inside max_num_batched_tokens (merged as part of #757, together with #733)
- #741 Do not take shared anonymous memory for a PLE disk shard (merged as part of #758)
- #723 Keep recovered tokens inside the vocabulary (backport of vllm-project/vllm#44744)
- #621 Drop the forced compile-cache opt-out for the Flash-V100 graph
- #670 Keep dummy-run positions inside max_model_len
- #662 Ship the speculative round state to non-last PP ranks
- #574 Trim the optimistic spec-decode tokens on every pipeline rank (now also proposed upstream as vllm-project/vllm#58926)
- #636 Keep the MTP drafter stage-local under pipeline parallelism (Qwen3.5)
- #603 Align the SWA decode threshold with the sparse MLA builder
- #658 Keep mHC finite under float16 for attention-sink rows
- #613 Honor a checkpoint's KV-cache quantization directive only on Ampere and newer
- #665 Order SimpleCPUOffloadConnector stores behind the compute stream
- #592 DFlash: fuse context K/V through quant_method so a quantized draft head loads
- #600 Resolve an unspecified device_id to the worker's own device
- #576 Read the quantization SM70 gate from the worker's own device
- #618 Gate the SM70 graph tunings on the worker's own pre-Ampere device
- #599 Gate DFlash2's BF16 emulation and FlashInfer top-k on the worker's own device
- #657 Backport vllm-project/vllm#44082: cache the EAGLE/MTP lookahead block in the SWA prefix-cache mask
- #669 Skip checkpoint tensors a model does not load before reading them
- #659 Apply min_p in the rejection sampler
- #622 Resolve the PLE table pointer inside the gather op instead of baking it into the graph
- #640 Detect the FP8 PLE table from ModelOpt mixed-precision configs
- #639 Allow checkpoint FP8 MTP experts under pipeline parallelism
- #654 PLE offload: send registration inputs by file descriptor
- #619 Pass --distributed-timeout-seconds to the NCCL subgroups
- #683 Let an explicit --moe-backend win over the TurboMind MoE gate

Background, from the original overview: all of these come from running 1Cat-vLLM on a mixed rig (3x V100 + 2x Quadro RTX 8000): DeepSeek-V4-Flash + DSpark with pipeline parallel over all five cards, Qwen3.8-Flash-Next with pipeline parallel over four cards, and Qwen3.8-27B with TP2 on the two RTX cards.

AI assistance was used to prepare this overview. The merge state of each open PR was checked against current main with git merge-tree.

🤖 Generated with [Claude Code](https://claude.com/claude-code)











