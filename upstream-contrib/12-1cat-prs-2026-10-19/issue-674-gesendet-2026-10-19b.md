Update 2026-10-19, afternoon: #1237 fixes the rest of #739. With one visible device, the PLE offload process built its VllmConfig before reducing it to DP1 and asked for the device of its inherited local DP rank. The second test in #739 failed only after the first one, because that test restored local DP rank 1 as an envs attribute. 19 PRs are open, and all of them merge cleanly into current main (6f3557c97).

Update 2026-10-19: thank you for #856 and #857 (merged on 2026-10-03), #873 and #874 (2026-10-04), and #1004 and #1005 (2026-10-09). #611 is closed in favor of #822, which keeps its history. I closed #1006 in favor of #1187: main had added the typed field itself, so only the default change was left. 18 PRs are open, listed below, and all of them merge cleanly into current main (6f3557c97). Among themselves, #1217 builds on #1214, and #1216, #1217 and #1235 overlap only in the generated docs/design/architecture/runtime_reference.md (the line anchors of KernelConfig, which tools/generate_architecture_reference.py --write regenerates). A correction to #1226, #1228 and #1229: the boot-to-boot differences of the Flash-Next greedy texts mentioned there came from the bug that #1234 fixes. With #1234 on top (our fork commit 6bbcd40be, which also contains those three changes), three boots per topology give byte-identical greedy texts.

Update 2026-10-03, afternoon: #742 is merged through #837, thank you. Native results for it are posted in #837: the 13 SM75 cases on a Quadro RTX 8000, complete-model runs, and a paired skinny on/off comparison. Two new fixes came out of those runs. #856: the PLE cascade never admitted ModelOpt checkpoints such as nvidia/Qwen3.8-Flash-Next-NVFP4, because the admission looked up the wrong layer under the wrong name, so a PP4 start tried to put the whole 47.69 GiB table on stage 0. #857: DeepSeek-V4 could not start on our V100/RTX 8000 rig, because the sparse policy was read through get_current_vllm_config() inside the forward pass, where the worker does not set it. Four PRs are open, and all of them merge cleanly into current main (8002bc107).

Update 2026-10-03, morning: thank you for the overnight round. #791 and #793 went in through #795, #623 through #797, #747 through #798, #604 through #804, #750 through #801, #740 through #808, #646 and #717 through #806, and #714 and #716 through #816. GPU results for #795, #797 and #798 are posted there. Three PRs are still open, and all of them merge cleanly into current main (3efe512d).

Update 2026-10-02, evening: thank you for the integration round. #743, #710, #749 and #726 went in through #773, #720 through #776, #744 and #745 through #777, #667 through #765, and #711 is merged. #646 and #717 are rebased onto main after the hybrid PLE change of #786, and #747 and #750 onto main after #777 and #782; #750 now registers its switch in the env metadata of #782. #750 also got a test fix I had missed: its PP2xTP4 tests built Fp8LinearMethod without use_qpn8. Two new fixes: #791 for the DFlash/DSpark start that #748 broke, and #793, which settles the hybrid block size before the PLE KV estimate; with the cascade of #646, Qwen3.8-Flash-Next needs it to start. 14 PRs are still open, and all of them merge cleanly into current main (a3498d46).

Update 2026-10-02: #621 and #723 are merged, and #725, #741 and #752 went in through #757, #758 and #762, thank you. 21 PRs are still open, and all of them merge cleanly into current main (e0f3fedd).

Update 2026-10-01, evening: one more, #752. It asserts that each request with drafts has at least draft_len + 1 query rows, so a scheduler/runner mismatch fails where it starts instead of verifying drafts against the preceding request's hidden states. One comparison per request. It has been in our fork since 2026-09-06 and has not fired in the logs we keep (back to 2026-09-21).

Update 2026-10-01, afternoon: two more. #750 lets every [128, 128] block-FP8 linear on Volta and Turing use the native QPN8 operators behind an opt-in switch (DeepSeek-V4 at PP5: prefill 16.7 -> 15.2 s, decode step 100 -> 85 ms, and no TurboMind tuning stall at new prompt lengths); stacked on #747. #749 checks the token ids the last pipeline stage hands to the others, so an out-of-vocabulary id is named instead of ending in an anonymous device-side assert. #744 now covers six test modules. Main plus our open PRs, with the #716 and #750 switches set, now produces output identical to our fork on DeepSeek-V4 (greedy and eight reference answers).

Update 2026-10-01, morning: three more PRs, all about DeepSeek-V4 before Ampere. #747 lets Turing take main's SM70 route (one capability-family predicate, so V100 and RTX stages of one pipeline declare the same backend) and gives wo_a a grouped fp16 path where Marlin serves block FP8. #745 decodes FP8 in software on every card without FP8 units, and #744 brings five DeepSeek-V4 test modules in line on pre-Ampere cards. #716 got a fix: its batched-matmul prefill read stale index slots past each token's length and hit a device assert on the first long prompt. Our fork now runs #747 + #716 instead of its own pre-Hopper path; on the same build that was 4 to 6 % faster (tables in #747). All 25 open PRs merge cleanly into current main (d3046986).

Update 2026-09-30, evening: #742 is the MoE part of the v100-skinny proposal in #718, and #743 fixes a memory peak in the SM70 MXFP4 repack that kept DeepSeek-V4 off our V100 stages on the default path. Earlier that day: #740, #741 and the bug report #739. #727 is closed, since #733 covers the same ground more broadly. 22 of the PRs below are merged, thank you for going through them.

Still open

- #715 Keep hc_head finite under float16 for attention-sink rows. Follow-up to #658, which left the fused hc_head kernel out; its store now saturates like the others.
- #1183 Restore DSpark decode on SM70 after the per-engine policy migration. Since #1137, DeepSeek-V4-Flash with DSpark on SM70 accepted almost no drafts: at PP5, main be3722bc0 had a mean acceptance length of 1.01-1.07 at 13.9-14.1 tok/s, with the three commits 2.95-3.19 at 39.0-42.6 tok/s.
- #1187 Default the FP8 small-shape runtime tuning to off. With the tuning on, four of six distinct short prompts took 2.27-4.54 s to the first token on their first request; with it off, every request took 0.41-0.57 s (DeepSeek-V4-Flash PP5, with #1183 applied). Replaces #1006.
- #1213 Keep the traced prefill output projection on the custom-op norm. Since 011a8ce1e (#1203), engine startup fails in the profile run when the Qwen GDN output projection is not behind the opaque GDN op, because Dynamo traces into the FLA input guard.
- #1214 Repair stale fakes in PLE offload and SM70 decode-graph tests: 24 test cases that fail on main 6f3557c97. Test-only.
- #1216 Two-rank FP16 all-reduce through pinned host memory for GPUs without P2P, adapted from mzen17's skinny_ar.cu. Qwen3.8-27B TP2 on two Quadro RTX 8000: prose decode 37 ms per step = 66.3-66.4 tok/s, with it 33 ms = 73.7-73.8 tok/s (about +11 %).
- #1217 Overlap the PLE gather of the next prefill chunk, opt-in through kernel_config.ple_prefill_early_request. It shortens the first cold prompt of a boot by 0.6-1.1 s (Flash-Next PP4). Builds on #1214.
- #1218 Reduced draft vocabulary for MTP, the fixed-list part of vllm-project/vllm#59740 by stecasta, opt-in through speculative_config.draft_token_map. Flash-Next PP4 prose decode 59.0 ms per step = 41.8 tok/s, with it 54.2 ms = 45.7 tok/s, at unchanged acceptance per round.
- #1219 Verify P2P before provisioning compact collectives without custom all-reduce. On our two Tesla V100-PCIE-32GB the driver reports peer access that vLLM's IPC write test rejects, and with --disable-custom-all-reduce the engine hung; with the check the logits processor takes its existing all-gather.
- #1221 DSpark speculator (port of vllm-project/vllm#46995 by benchislett) and compressed-index context buckets for Model Runner V2. DeepSeek-V4-Flash PP5, V1 against V2: prose decode 70 ms per step = 42.9-43.0 tok/s against 63 ms = 48.2-48.3 tok/s.
- #1222 Keep the cuBLAS decode indexer above 131,072 tokens of context. DeepSeek-V4-Flash PP5 at 287,077 tokens: 241.4-242.4 ms per step = 12.6-13.1 tok/s, with it 93.1-93.5 ms = 33.2-34.3 tok/s.
- #1226 Let the delta_h autotuner lift the ptxas register cap, which holds the GDN state recurrence to 64 registers per thread on Turing. 6.91 -> 3.01 ms per call, bit-identical; Qwen3.8-27B TP2 on two Quadro RTX 8000, ~29k-token prefill 41.2-41.3 -> 38.5-38.6 s.
- #1227 Pack GQA heads for FA2 multi-token decode verification. Qwen3.8-27B with MTP on two Quadro RTX 8000 at 249,950 tokens: 182.5 ms per step = 14.2 tok/s, with it 63.2 ms = 41.5-41.6 tok/s.
- #1228 Run the FlashQLA TileLang prefill kernel on Turing, through an sm_75 variant that fits into the 65,536 B of shared memory per block. Qwen3.8-27B TP2 on two Quadro RTX 8000: ~29k-token prefill 38.5-38.6 -> 35.3-35.4 s.
- #1229 Admit the cuBLAS QSA indexer and exact top-k on Turing. Together with #1228, Flash-Next TP2xPP2 prefill of 184,134 tokens 178.9-194.1 -> 126.3-129.2 s.
- #1234 Decide the PLE cascade merge when the layer is built. With the PLE disk cascade, on by default for admitted engines since #806, the Qwen4Exp n-gram embedding was computed without its resident rows. Mean NLL 1.4115 and 1.4023 in two passes -> 1.3609 in both; three boots per topology give byte-identical greedy texts.
- #1235 Bound the page prefetch of the mapped PLE row gather (kernel_config.ple_disk_prefetch_window_kib, default 4096 KiB). Flash-Next PP4, time to first token for 184,134 tokens 131.8 / 183.0 s -> 69.4 / 63.6 s.
- #1236 Run the native page4 QSA prefill routes for 12/24-head ranks, several KV heads and Turing. Flash-Next PP4, ~29k tokens 10.33 / 10.20 s (2,826 / 2,863 tok/s) -> 7.67 / 7.70 s (3,809 / 3,793 tok/s). It applies with and without #1229; its conflicts with other open PRs (#828, #903, the draft series #1179-#1198) are described there.
- #1237 Reduce the PLE offload config to DP1 before it is built. With one visible device, the PLE offload process of a GPU worker that passes down local DP rank 1 asked for device 1 and failed with IndexError before it set DP1; the regression test also stops leaving local DP rank 1 behind for later tests.

Bug report #739: the first test was fixed on main by 91701504c, the second is fixed by #1237.

#718 proposed bringing the v100-skinny kernels into csrc; its MoE part, #742, is merged through #837. Block FP8 needs no port: main's native QPN8 kernels already match or beat it on DeepSeek-V4's shapes. A correction to what this overview said before: main does run DeepSeek-V4's MXFP4 MoE on Turing, through Marlin.

Merged

- #1004 Read the PLE row-gather policy when the layer is built
- #1005 Repair twelve SM70 test cases broken by recent main changes
- #873 Start block QPN8 on Volta: warm up the layout it holds, keep the TurboMind copy only when decode can use it
- #874 Bucket speculative decode graphs by context so the sparse indexer can take cuBLAS
- #856 Look up PLE cascade storage under checkpoint layer names
- #857 Read the SM70 sparse policy when the layers are built
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

AI assistance was used to prepare this overview. The merge state of each open PR was checked against current main with git merge-tree, and the two tests of #739 were run again on current main.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01YaG8VjXn5eN3Aq7eZrA3dp
