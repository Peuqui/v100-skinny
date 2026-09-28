Update 2026-09-27, evening: two new PRs and a fix to #604. 22 of the PRs below are merged, thank you for going through them. All eight open ones merge cleanly into current main (357d07bc).

Still open

- #710 Resolve SM70 linear workspace addresses on AOT reload (new). Several SM70 linear paths pass a scratch workspace address from apply() as an integer, which ends up baked into AOT artifacts; the same pattern #672 fixed for compressed-tensors. Latent on main in the default V100 setup, where the compile cache is off.
- #604 Run ModelOpt NVFP4 and FP8 linears on Turing through the SM70 QPN kernels. Its Turing FP8 route had the same pattern and crashed on every warm start; it now registers its workspace through #710 and is stacked on it.
- #711 Reject new VLLM_* environment reads that bypass envs.py (new). A pre-commit check so that switches outside envs.py, which the compile cache key does not see, stop growing; the 137 existing ones stay as a baseline.
- #623 Build and load a Turing FlashAttention-2 library next to the Volta one
- #611 Block-pack the activations of the NVFP4 QPN2 kernels
- #667 Reuse a sliding window's dead blocks before other requests' cached ones
- #646 PLE overflow cascade: device, pinned host, disk
- #621 Drop the forced compile-cache opt-out for the Flash-V100 graph

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

