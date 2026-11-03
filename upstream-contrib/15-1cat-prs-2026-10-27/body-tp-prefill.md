## Purpose

Tensor-parallel prefill without peer access spends about half its time in all-reduces: on two Quadro RTX 8000 (no P2P, NCCL through host memory at about 2 GB/s) the all-reduces took 96 of 208 s of a 150k-token Qwen3.8-27B prefill, and the remaining compute ran FA2's Turing attention at about a quarter of the tensor-core rate. This PR overlaps the all-reduces with compute, moves them off the SMs, and gives Turing its own prefill attention. Six commits, stacked on the Turing kernel PR (#TURING_PR), #1216 (two-rank host all-reduce, which the copy-engine mode extends) and #1227 (GQA packing for FA2 verification, same attention backend file); only the last six commits are new here:

1. Turing paged prefill attention for head dim 256 (82bd95f77). FA2's Turing split-KV kernel keeps Q in shared memory next to K and V, so a CTA fills the SM's 64 KB and every tile load stalls the math (no cp.async before SM80). The new kernel holds 128 GQA-packed query rows per CTA in registers, double-buffers 32-key K/V tiles and fetches the next tile in two register-staged halves while the current one computes. The FLASH_ATTN backend takes it on SM75 (capability of the layer's own device) for causal head-dim-256 attention over an fp16 paged cache once a request brings at least attention_config.flash_attn_turing_prefill_min_query_len (65) query tokens; below that FA2's context split stays faster.
2. Split-KV for that kernel by waves (dbdde81b6). One CTA fills a Turing SM, so a 1600-token chunk (150 CTAs on 72 SMs) took three waves for 2.1 waves of work. Each CTA's key range can be split into whole-tile ranges, merged by log-sum-exp; the launcher picks the split count (up to 4) that finishes in the fewest full-CTA times on the device.
3. Overlap of TP prefill all-reduces with compute (5ac5f0071), behind kernel_config.tp_prefill_overlap_min_tokens (0 = off, the default). Eager prefill batches of at least that many tokens run as two interleaved token-range halves on one compute stream in two threads, after the DBO micro-batch mechanics; each all-reduce runs on a comm stream while the other half computes its layer. A request cut at the split continues in the second half as a following chunk would, so the second half's attention reads the first half's KV writes; in mamba align mode the split moves so that both halves use the recurrent state slot of the chunk end (801 + 799 tokens for a 1600-token chunk of 800-token blocks). Needs TP without PP and DP; LoRA, KV transfer, PLE offload and mamba cache mode "all" are rejected at startup.
4. Copy-engine mode of the host all-reduce (95393bf4f), behind kernel_config host_reduce.copy_engine_max_bytes (0 = off, the default). NCCL's LL protocol and the host reduce's kernel mode poll host memory with SMs, which slowed the kernels running beside them by 5 to 37 % and ate most of the overlap's gain. The new mode moves the tensors through the host slots with the copy engines in up to 16 pieces, ordered by cuStreamWriteValue32/cuStreamWaitValue32, then one FP16 add gives the same bits as the kernel mode and NCCL's two-rank sum. It refuses graph capture; decode payloads stay on the kernel mode.
5. Turing tile for the NVFP4 prefill dequantization (a16601c7d): 16 groups and 8 warps instead of Volta's 64 groups, chosen per device (2.2x faster on the RTX 8000, bit-identical; Volta unchanged).
6. Fix (eda4889e1): MLA prefill backend selection asked flash_v100_dense_prefill_lse_available() while the model was built, and the check bound the Flash-V100 runtime, which raised "Flash-V100 resources must be bound before execution" on every V100 stage (GLM-5.3-Flash). Availability now only loads; execution still binds.

Not a duplicate: I searched the open PRs for all-reduce, allreduce, overlap, "host memory", "copy engine", P2P, "prefill attention", Turing, SM75, split-KV and NVFP4 dequant, and compared the changed files with those of every open PR. SEARCH_PLACEHOLDER

AI assistance was used for this change. I reviewed every line and ran the tests and measurements below.

## Test Plan

Native build of this branch (in-place).
python -m pytest TESTFILES_PLACEHOLDER, one process per file, on a Quadro RTX 8000 pair (tests/distributed/test_host_reduce.py with NCCL_P2P_DISABLE=1), a single Quadro RTX 8000 and a Tesla V100.
pre-commit run --from-ref origin/main --to-ref HEAD; pre-commit run mypy-3.10 --hook-stage manual --from-ref origin/main --to-ref HEAD

## Path coverage

| Path / provider | Format | Key conditions (shape, TP/EP, graph, capability) | Expected selection / fallback | Evidence (source / static / executed / measured) | Unverified scope |
| --- | --- | --- | --- | --- | --- |
| FLASH_ATTN prefill on SM75 | fp16 paged KV, head dim 256, causal | at least 65 query tokens in a request; no window, softcap or sinks | paged_prefill_attention_sm75; otherwise FA2 | executed: test_paged_prefill_attention_sm75.py (fp32 reference, 0/1/2/4 splits); measured: kernel and model runs | other head dims, fp8 KV, SM8x+ (never selected there) |
| TP prefill overlap | any | TP > 1, no PP/DP, eager prefill batch >= tp_prefill_overlap_min_tokens | two halves with comm-stream all-reduces; off by default | executed: test_prefill_overlap.py (bit-identical to the whole batch on a GPU); measured: Qwen3.8-27B TP=2 A/B | TP > 2 not measured |
| host all-reduce copy-engine mode | FP16 | two ranks without P2P, payload above max_bytes and up to copy_engine_max_bytes, not capturing | copy engines + stream memory operations; off by default | executed: test_host_reduce.py on the RTX pair (bit-identical to NCCL 0.3-21 MB, capture refused); measured: model A/B | more than two ranks (mode is two-rank only, like #1216) |
| NVFP4 prefill dequantization | NVFP4 (QPN2) | SM75 | 16 groups, 8 warps; SM70 keeps 64 groups | executed: test_nvfp4_qpn2_dequant.py on RTX 8000 and V100 | none |
| Flash-V100 dense prefill availability | MLA | model build before runtime binding | availability loads without binding | executed: test_flash_v100_dense_prefill_availability.py | none |

## Acceleration and benchmark contract (required for performance changes)

- Default enabled or opt-in: the Turing prefill attention, its split-KV choice and the NVFP4 tile are on by default on the matching capability (per device). The overlap (tp_prefill_overlap_min_tokens) and the copy-engine mode (host_reduce.copy_engine_max_bytes) are opt-in, 0 by default.
- Required CLI options and environment switches: model runs used --kernel-config '{"tp_prefill_overlap_min_tokens":1024,"host_reduce":{"copy_engine_max_bytes":20971520}}' in the new arm. Entry: Qwen3.8-27B NVFP4, TP=2 on the two Quadro RTX 8000, MTP (3 speculative tokens), VLLM_USE_V2_MODEL_RUNNER=1.
- KV cache dtype used for the benchmark: auto (fp16).
- Wheel SHA or source commit: fork builds (github.com/Peuqui/1Cat-vLLM); old arm = fork production at that time, new arm = the same plus these changes. The commits are rebased onto the Turing kernel PR and main 6a143197e; the branch itself was built and unit-tested, not booted with a full model.
- PYTHONPATH, source overlays, or external native libraries used: both arms ran the fork's production entry, which also uses the tensor-core Triton routes of a later PR of this series (identical in both arms; the changes here do not use Triton).
- User-entry route-hit, speed, and output-quality evidence: see Test Result.
- Existing kernel/backend registry and capability-rejection reasons: the Turing prefill attention is chosen inside the FLASH_ATTN backend by capability, head dim, cache dtype, causality and query length; the overlap rejects unsupported configurations at startup with an error.
- Per-engine configuration and deprecated-variable compatibility: three new fields (attention_config.flash_attn_turing_prefill_min_query_len, kernel_config.tp_prefill_overlap_min_tokens, kernel_config.host_reduce.copy_engine_max_bytes); no environment variables.
- New/changed environment metadata and generated reference check: REFERENCE_PLACEHOLDER
- Route snapshot command and intentional changed rows: none.
- SM70 registered variables / unregistered reads / model-parameter restrictions, before -> after: no change.

## Test Result

Model A/B/A/B, Qwen3.8-27B NVFP4 TP=2 on two Quadro RTX 8000, one boot per arm, the first boot after a compile change not counted. Prefill time in s, lower is better (old a / new a / old b / new b):
- 29k-token prompts (four prompts per arm): 33.0-33.1 / 19.1-19.3 / 33.0-33.1 / 19.1-19.3 (-42 %)
- 184k tokens: 269.05 / 174.34 / 269.21 / 174.38 (-35 %)
- 250k tokens: 402.27 / 269.06 / 402.47 / 269.29 (-33 %)
Decode unchanged, tok/s (higher is better) old / new: prose 74.6-75.0 / 74.8-75.2, code 114.3-115.1 / 115.4-115.9, edit 119.1-119.9 / 119.8-120.1; at 184k 54.3 ms per step (50.7 tok/s) and at 250k 62.2-62.4 ms in both. Greedy output identical to the reference in all five boots. Eight quality answers: seven byte-identical between the arms, one (a speed calculation) worded differently from character 909 on, both on the same correct derivation and stopped at the 600-token limit; each arm byte-identical across its two boots. Peak memory per RTX 8000 48,285 MiB against 47,979 MiB (+306).

Contribution of the steps (single boots on the same entry, not A/B), prefill time in s, lower is better, old / + overlap / + copy-engine mode / + split-KV / + NVFP4 tile (the A/B arm):
- 29k: 33.0-33.1 / 28.5-29.0 / 21.7-21.9 / 21.1-21.2 / 19.1-19.3
- 184k: 269.05 / 260.28 / 216.81 / 185.25 / 174.34
- 250k: 402.27 / 400.14 / 341.66 / 283.50 / 269.06

Kernel microbenchmarks (CUDA graphs, ms per call, lower is better), Quadro RTX 8000, Qwen3.8-27B per-rank shapes:
- Prefill attention, FA2 -> new kernel: 2048 tokens after 32k context 35.3 -> 13.1, after 150k 165.8 -> 61.5 (61 TFLOPS); 512 tokens after 150k 52.5 -> 20.0; 96 tokens 24.6 -> 19.8; 64 tokens 5.3 (FA2) against 19.9, hence the threshold of 65. Errors against an fp32 reference equal FA2's in every case.
- Split-KV, one split / automatic: 1600 tokens after 30k 12.40 / 9.71, after 100k 40.42 / 31.75, after 184k 75.59 / 58.47; 801 tokens after 100k 27.08 / 17.14, after 184k 49.65 / 32.07.
- Copy-engine all-reduce, 8 MiB per call: 3.15 ms (2.60 GB/s) against NCCL's 3.93 ms (2.08 GB/s); the time of a GEMM loop running beside it changed by -2.8 % against +18.2 % beside NCCL, that of a memory-bound loop by +1.8 % against +6.0 %.
- NVFP4 dequantization: 17408 x 5120 1.090 -> 0.496, 5120 x 8704 0.580 -> 0.250, 5120 x 2560 0.166 -> 0.075, 8192 x 5120 0.525 -> 0.230.

TESTS_PLACEHOLDER

🤖 Generated with Claude Code (https://claude.com/claude-code)

https://claude.ai/code/session_011k7FsBAcZ8Cw91wisiZxMf
