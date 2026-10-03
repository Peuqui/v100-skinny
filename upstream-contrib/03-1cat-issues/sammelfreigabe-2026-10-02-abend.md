Status ~20:55: Teil 5 (#674) in Runde 2 aufgegangen und gesendet, siehe sammelfreigabe-2026-10-02-runde2.md. Vorher: Status 19:46: Teile 1–4 GESENDET (#750-Kommentar, PR #791, #773, #777); Teil 5 (#674) zurückgehalten wegen PLE-Problem #646/#717. — Sammelentwurf 2026-10-02 abends. FREIGABE Peuqui für alle fünf Teile erteilt (ca. 19:00); senden nach der Flash-Next/27B-Abnahme, vorher RTX-Tests #747 und PR-Branch auf aktuelles main + Tests.
Reihenfolge beim Senden: (1) Force-Push #747/#750 + Kommentare, (2) neuen PR eröffnen,
(3) Kommentare #773/#777, (4) #674 mit PR-Nummer und main-Hash. Absätze beim Senden zu Zeilen
zusammenfügen. Belege: Testläufe dieser Sitzung (V100 GPU 3/4, RTX GPU 0/2), DSv4-Läufe
ab_next_ds_2026-10-02.log / ab_next_ds_nextonly_2026-10-02.log, range-diffs.

==================== 1a. Force-Push #747 und #750 ====================
#747: pr-dsv4-turing 4c9bc2718 -> 240bb663a (ein Commit, unverändert, jetzt direkt auf main b9c003bc)
#750: pr-fp8-block-qpn8 7a4c665cf -> 36f17b418 (338d142d3 + 36f17b418; Schalter im env_var-Format
von #782, Zeile in docs/configuration/env_var_reference.md)
  git push --force-with-lease=pr-dsv4-turing:4c9bc2718... fork pr-dsv4-turing
  git push --force-with-lease=pr-fp8-block-qpn8:7a4c665cf... fork pr-fp8-block-qpn8

==================== 1b. Kommentar in #750 ====================
Rebased #747 and #750 onto current main (b9c003bc). #744 and #745 went in through #777, so #747
is now one commit on main, unchanged, and #750 sits on it. #782 moved the environment variables to
env_var() metadata; VLLM_SM70_FP8_BLOCK_QPN8 is now registered that way (experimental, default
False, acceleration path "FP8 QPN8"), and docs/configuration/env_var_reference.md is regenerated,
with that one row added. Tests on a Tesla V100, one pytest process per file:
test_sm70_qpn8_block_fp8 27, test_sm70_fp8_qpn8_pp2_tp4 13, test_deepseek_v4_sm70_routes 13,
test_fp8_grouped_bmm_dequant 6, test_deepseek_v4_sm70_indexer 8,
test_deepseek_v4_sm70_qnorm_rope_kv_insert 2, test_cudagraph_dispatch 46, test_envs 58 and
test_check_env_registration 8 passed. pre-commit and mypy-3.10 pass.

==================== 2. Neuer PR (Branch pr-acceleration-report-followups -> umbenennen in
pr-dflash-draft-config-replace, Commit 7214ee7be auf main 24994ba9; vor dem Senden auf aktuelles
main setzen und Tests wiederholen) ====================
Titel: [Bugfix][Spec Decode] Copy the DFlash/DSpark draft config with vLLM's pydantic-aware replace

## Purpose

Since #748, every DFlash and DSpark start fails while the proposer loads its drafter:

```
pydantic_core._pydantic_core.ValidationError: 1 validation error for VllmConfig
sm70_acceleration_report
  Unexpected keyword argument [type=unexpected_keyword_argument, ...]
```

#748 added sm70_acceleration_report to VllmConfig as Field(default_factory=dict, init=False). For
the standard library the field is still an init field, so dataclasses.replace passes its current
value to the constructor, and pydantic rejects it. DFlashProposer._create_draft_vllm_config copies
the target config with dataclasses.replace (imported in dflash.py since #403), while
LLMBaseProposer already uses vllm.config.utils.replace, which skips non-init fields. This PR makes
dflash.py use the same helper. It is a one-line import change; the three replace calls there all
copy vLLM config classes (VllmConfig, CacheConfig, AttentionConfig), which that helper is meant for.

Not a duplicate: no open PR touches vllm/v1/spec_decode/dflash.py, and no issue or PR mentions
this error (gh pr list / gh search issues for "sm70_acceleration_report", "Unexpected keyword
argument", "dflash replace").

AI assistance was used for this change. I reviewed every line and ran the tests below.

## Test Plan

```bash
pytest tests/v1/spec_decode/test_dflash_config_replace.py tests/v1/spec_decode/test_dspark.py \
  tests/v1/spec_decode/test_ddtree_config.py tests/v1/spec_decode/test_dflash2.py \
  tests/v1/spec_decode/test_dflash2_pre_ampere_gate.py
pre-commit run --files vllm/v1/spec_decode/dflash.py tests/v1/spec_decode/test_dflash_config_replace.py
pre-commit run mypy-3.10 --hook-stage manual --files <same>
```

## Acceleration and benchmark contract (required for performance changes)

Not a performance change.

## Test Result

The new test copies a VllmConfig with a filled sm70_acceleration_report through the replace that
dflash.py uses: it passes with this change and fails on main, both without a GPU and on a Tesla
V100. test_dspark 14, test_ddtree_config 5, test_dflash2 177 and test_dflash2_pre_ampere_gate 8
passed. pre-commit and mypy-3.10 pass.

End to end on our fork (main 24994ba9 plus our open PRs, 3x V100 + 2x Quadro RTX 8000):
DeepSeek-V4-Flash with DSpark at PP5 failed to start with the error above; with this change it
starts and serves (greedy, first-token probe, two 35.7k-token prompts, short prompts and eight
quality prompts).

==================== 3. Kommentar in #773 ====================

GPU results for this merge on our rig (3x Tesla V100 + 2x Quadro RTX 8000), on main 3e6d0a3c
with native extensions from our fork's build (main d3046986 plus our open PRs; main's csrc is
unchanged since then), one pytest process per file.

The modules this PR touches pass natively on both cards: test_pp_token_check 17,
test_sm70_fp8_workspace_aot_reload 6, test_sm70_fp8_prefill_exact_dense 24,
test_sm70_fp8_qpn8_pp2_tp4 13, test_sm70_turbomind_adapter 13, test_qwen3_5_quantization 14,
test_qsa_xqa_page4_workspace 5 and its CPU variant 7, test_sm70_compressed_tensors_fastpaths 5,
test_sm70_ct_fp8_cache_workspace 1, test_spec_token_aliasing 2 and test_envs 58 passed.
test_sm70_mxfp4_moe passed 33 on the V100, including the repack allocation bound (28 passed and
the 5 V100-only cases skipped on the RTX 8000); test_sm70_flash_v100_multihead passed 70 on the
V100 (31 passed, 39 SM70-only skipped on the RTX 8000).

Cold/hot AOT: the workspace commit is unchanged from #710, and our fork has carried it since
2026-09-27. Since then Qwen3.8-27B-NVFP4 (TP2 with MTP, compile cache on) started warm from its
AOT artifacts on 2x Quadro RTX 8000 and served requests 19 times without the stale-pointer
error; earlier that day, before the change, four warm starts failed with "The specified pointer
resides on host memory and is not registered with any CUDA device." Its FP8 layers reach these
sites on Turing through the QPN8 route of #604, so those runs are our fork with #604 on top.

PP token checks under real multi-rank serving: on our fork (main 24994ba9 plus our open PRs),
DeepSeek-V4-Flash with DSpark (k=5) at PP5 across all five cards served 24 requests (greedy,
first-token probe, two 35.7k-token prompts, short prompts, eight quality prompts) without a single
"PP spec decode: invalid" error, at the same speed as before the merge (86 ms per decode step at
35.7k context, 15.4 s to first token).

==================== 4. Kommentar in #777 (gemergt; Bestätigung nach dem Merge) ====================

Native run of this branch (344bb9eb) on a Tesla V100 and a Quadro RTX 8000, one pytest process
per file, against main 3e6d0a3c, with the same extensions as above:

- test_compressor_kv_cache: 32 passed, 6 skipped on both cards; main fails 21 on the V100 and
  35 on the RTX 8000.
- test_deepseek_v4_fp8_capability: 16 passed on both.
- test_deepseek_v4_mega_moe: 3 passed, 1 skipped on both; main fails 1.
- test_fused_deepseek_v4_qnorm_rope_kv_insert, test_fused_indexer_q_rope_quant and
  test_fused_inv_rope_fp8_quant: all 139, 40 and 50 cases skip on both cards with their stated
  reasons (the fused op refuses pre-Ampere; Triton's fp8e4nv needs sm_89); main fails all of
  them.

git range-diff shows the #744/#745 commits unchanged apart from your sign-off.

==================== 5. #674 neu (Platzhalter [NEUER PR #NNN] und [MAIN] beim Senden füllen) ====================
Update 2026-10-02, evening: thank you for the integration round. #743, #710, #749 and #726 went in through #773, #720 through #776, #744 and #745 through #777, #667 through #765, and #711 is merged. #646 and #717 are rebased onto the PLE changes of #766, and #747 and #750 onto main after #777 and #782; #750 now registers its switch in the env metadata of #782. #750 also got a test fix I had missed: its PP2xTP4 tests built Fp8LinearMethod without use_qpn8. [NEUER PR #NNN] 12 PRs are still open, and all of them merge cleanly into current main ([MAIN]).

Update 2026-10-02: #621 and #723 are merged, and #725, #741 and #752 went in through #757, #758 and #762, thank you. 21 PRs are still open, and all of them merge cleanly into current main (e0f3fedd).

Update 2026-10-01, evening: one more, #752. It asserts that each request with drafts has at least draft_len + 1 query rows, so a scheduler/runner mismatch fails where it starts instead of verifying drafts against the preceding request's hidden states. One comparison per request. It has been in our fork since 2026-09-06 and has not fired in the logs we keep (back to 2026-09-21).

Update 2026-10-01, afternoon: two more. #750 lets every [128, 128] block-FP8 linear on Volta and Turing use the native QPN8 operators behind an opt-in switch (DeepSeek-V4 at PP5: prefill 16.7 -> 15.2 s, decode step 100 -> 85 ms, and no TurboMind tuning stall at new prompt lengths); stacked on #747. #749 checks the token ids the last pipeline stage hands to the others, so an out-of-vocabulary id is named instead of ending in an anonymous device-side assert. #744 now covers six test modules. Main plus our open PRs, with the #716 and #750 switches set, now produces output identical to our fork on DeepSeek-V4 (greedy and eight reference answers).

Update 2026-10-01, morning: three more PRs, all about DeepSeek-V4 before Ampere. #747 lets Turing take main's SM70 route (one capability-family predicate, so V100 and RTX stages of one pipeline declare the same backend) and gives wo_a a grouped fp16 path where Marlin serves block FP8. #745 decodes FP8 in software on every card without FP8 units, and #744 brings five DeepSeek-V4 test modules in line on pre-Ampere cards. #716 got a fix: its batched-matmul prefill read stale index slots past each token's length and hit a device assert on the first long prompt. Our fork now runs #747 + #716 instead of its own pre-Hopper path; on the same build that was 4 to 6 % faster (tables in #747). All 25 open PRs merge cleanly into current main (d3046986).

Update 2026-09-30, evening: #742 is the MoE part of the v100-skinny proposal in #718, and #743 fixes a memory peak in the SM70 MXFP4 repack that kept DeepSeek-V4 off our V100 stages on the default path. Earlier that day: #740, #741 and the bug report #739. #727 is closed, since #733 covers the same ground more broadly. 22 of the PRs below are merged, thank you for going through them.

Still open

- #750 Block FP8 through the native QPN8 operators on SM70/SM75 (new, opt-in). Stacked on #747.
- #747 Run the DeepSeek-V4 SM70 route on Turing (new). Now directly on main, since #744 and #745 went in through #777. Mixed V100 + Turing pipelines start and run it end to end; with #716 a 35.8k prompt prefills in 16.7 s on our five cards.
- #742 Skinny QPN MoE backend for NVFP4 and MXFP4 on SM70 and SM75 (new, opt-in). `--moe-backend sm70_skinny` runs the MoE through the v100-skinny kernels with the checkpoint's weight layout. On DeepSeek-V4-Flash across our five cards (no drafter) prefill 18.9 s instead of 31.9 s and decode 15.5 instead of 6.8 tok/s against the default path; with the DSpark drafter only this path loads here. Tables for Qwen3.8-Flash-Next in the PR.
- #740 Add a direct I/O safetensors load strategy (new, opt-in). `--safetensors-load-strategy direct` reads the decoder layers with O_DIRECT, once per tensor-parallel group, and releases the page cache of everything else once it is loaded. On our rig the boots got 18 to 33 % faster, and the swap growth during the load went from 5 to 10 GiB to zero for DeepSeek-V4 and Qwen3.8-27B and to 0 to 4 GiB for Qwen3.8-Flash-Next (PP4). It touches one import line of `nvidia/ple_layer.py` that #646 and #717 also touch; whichever lands second, I will rebase.
- #714 Take the DeepSeek-V4 cuBLAS indexer decode under speculative decoding. The opt-in cuBLAS route only accepted a single block-table row, so with DSpark or MTP it never ran; on our rig it keeps the decode step at 90 ms at 62k context instead of 116 ms. Default stays off.
- #715 Keep hc_head finite under float16 for attention-sink rows. Follow-up to #658, which left the fused hc_head kernel out; its store now saturates like the others.
- #716 DeepSeek-V4 sparse MLA as gather + batched matmul (opt-in). Decode wins from about 16 heads per rank with speculative decoding, prefill is faster at every size measured; tables in the PR. Updated 2026-10-01 with a fix for stale prefill indices (see the comment there).
- #717 Allow the PLE CPU offload under pipeline parallelism. Stacked on #646; drops the PP entry from the unsupported list.
- #604 Run ModelOpt NVFP4 and FP8 linears on Turing through the SM70 QPN kernels. Its Turing FP8 route had the same pattern and crashed on every warm start; it now registers its workspace through the module #710 added, which is in main through #773.
- #623 Build and load a Turing FlashAttention-2 library next to the Volta one
- #611 Block-pack the activations of the NVFP4 QPN2 kernels
- #646 PLE overflow cascade: device, pinned host, disk

Bug report: #739, two tests fail on current main (a stale QSA E4M3 expectation, and the PLE offload config with a single visible GPU).

#718 proposed bringing the v100-skinny kernels into csrc; #742 is its MoE part. Block FP8 needs no port: main's native QPN8 kernels already match or beat it on DeepSeek-V4's shapes. A correction to what this overview said before: main does run DeepSeek-V4's MXFP4 MoE on Turing, through Marlin.

Merged

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








