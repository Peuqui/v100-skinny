Title: Overview of my open PRs: what each fixes, dependencies, and an offer to bundle

Update 2026-09-30: two new PRs (#740, #741) and a bug report (#739). #727 is closed, since #733 covers the same ground more broadly. 22 of the PRs below are merged, thank you for going through them. All eighteen open ones merge cleanly into current main (d3046986).

Still open

- #740 Add a direct I/O safetensors load strategy (new, opt-in). `--safetensors-load-strategy direct` reads the decoder layers with O_DIRECT, once per tensor-parallel group, and releases the page cache of everything else once it is loaded. On our rig the boots got 18 to 33 % faster, and the swap growth during the load went from 5 to 10 GiB to zero for DeepSeek-V4 and Qwen3.8-27B and to 0 to 4 GiB for Qwen3.8-Flash-Next (PP4). It touches one import line of `nvidia/ple_layer.py` that #646 and #717 also touch; whichever lands second, I will rebase.
- #741 Do not take shared anonymous memory for a PLE disk shard (new). The disk-tier guard accepted a `/dev/zero` mapping as a file-backed shard.
- #714 Take the DeepSeek-V4 cuBLAS indexer decode under speculative decoding. The opt-in cuBLAS route only accepted a single block-table row, so with DSpark or MTP it never ran; on our rig it keeps the decode step at 90 ms at 62k context instead of 116 ms. Default stays off.
- #715 Keep hc_head finite under float16 for attention-sink rows. Follow-up to #658, which left the fused hc_head kernel out; its store now saturates like the others.
- #716 DeepSeek-V4 sparse MLA as gather + batched matmul (opt-in). Decode wins from about 16 heads per rank with speculative decoding, prefill is faster at every size measured; tables in the PR.
- #717 Allow the PLE CPU offload under pipeline parallelism. Stacked on #646; drops the PP entry from the unsupported list.
- #720 Size the ROCm SWA ragged copy from the actual row width. With DSpark every decode step on ROCm fails in that copy; found through our fork, which runs the same builder on CUDA.
- #723 Keep recovered tokens inside the vocabulary. Backport of vllm-project/vllm#44744 (GHSA-8wr5-jm2h-8r4f); on 1Cat main an all-NaN target row under DSpark/MTP/EAGLE still yields token id vocab_size, the device-side assert from #658.
- #725 Keep the SM70 cudagraph size cap inside max_num_batched_tokens. On a V100 with speculative decoding and a token budget below 64 the config failed validation.
- #726 Bring stale core and SM70 tests in line with current main. Tests only: scheduler, spec-token aliasing, an order-dependent QPN8 test, and SM70-only multihead cases that failed on Turing.
- #710 Resolve SM70 linear workspace addresses on AOT reload. Several SM70 linear paths pass a scratch workspace address from apply() as an integer, which ends up baked into AOT artifacts; the same pattern #672 fixed for compressed-tensors. Latent on main in the default V100 setup, where the compile cache is off.
- #604 Run ModelOpt NVFP4 and FP8 linears on Turing through the SM70 QPN kernels. Its Turing FP8 route had the same pattern and crashed on every warm start; it now registers its workspace through #710 and is stacked on it.
- #711 Reject new VLLM_* environment reads that bypass envs.py. A pre-commit check so that switches outside envs.py, which the compile cache key does not see, stop growing; the 137 existing ones stay as a baseline.
- #623 Build and load a Turing FlashAttention-2 library next to the Volta one
- #611 Block-pack the activations of the NVFP4 QPN2 kernels
- #667 Reuse a sliding window's dead blocks before other requests' cached ones
- #646 PLE overflow cascade: device, pinned host, disk
- #621 Drop the forced compile-cache opt-out for the Flash-V100 graph

Bug report: #739, two tests fail on current main (a stale QSA E4M3 expectation, and the PLE offload config with a single visible GPU).

Proposal before a larger PR: #718 asks whether you want the v100-skinny NVFP4/MXFP4 kernels in csrc. They are what lets the Turing stages run DeepSeek-V4's MXFP4 MoE on our rig, and they speed up the NVFP4 MoE prefill.

#667 is smaller now: the first half of it, recycling blocks without a hash first, reached main through 1cbaa2c5. The remaining part still matters on DeepSeek-V4: a 23k prefix after an unrelated 30k request costs 10.6 s on main and 0.7 s with the PR. The PR description has the details.

Merged

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








---
Status: gepostet 2026-09-30 14:27 als neuer Text von https://github.com/1CatAI/1Cat-vLLM/issues/674 (Freigabe Peuqui)
