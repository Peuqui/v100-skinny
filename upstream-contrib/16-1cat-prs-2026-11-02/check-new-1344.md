## Purpose

The driver reserves the largest stack of any kernel it has launched for every thread of the card, until the process ends. For DeepSeek-V4 head dim 512, ptxas gives the 4-warp build of _fused_kv_compress_norm_rope_insert_sparse_attn 32 registers on Volta and 64 on Turing, and the kernel spills 11,744 B per thread on a Tesla V100 and 9,088 B on a Quadro RTX 8000. That costs about 2 GiB on each V100 and 0.7 GiB on each RTX 8000 outside the torch allocator. It shows up at graph capture: the capture took 0.64 / 1.76 / 1.76 / 1.76 / 0.68 GiB on the five pipeline stages, whatever the capture sizes, while PyTorch's own allocation grew by 42 MiB.

This PR launches that one kernel with 16 warps and maxnreg=128, what 512 threads may hold. Its stack falls to at most 1,392 B on Volta and 1,320 B on Turing. maxnreg alone does not do it: with 4 warps the kernel keeps up to 12 KB. The other two compress kernels keep num_warps=1 and no maxnreg.

The arithmetic is not bit-identical to the 4-warp build. The warp count changes single rounding results: in a probe without a model, 1 of 215,040 FP8 bytes on a V100 and 2 of 215,040 on an RTX 8000 differ, each by one code. Registers do not: 4 warps with maxnreg 255 are byte-identical to 4 warps without it. Greedy answers then take another path, see Test Result; answer quality is the same.

Not a duplicate: searched the open PRs of this repository (86 at the time of writing) for compress, maxnreg, stack and DeepSeek-V4 kernel; none changes this kernel's launch.

AI assistance was used for this change. I reviewed every line and ran the tests and measurements below.

## Test Plan

python -m pytest tests/kernels/test_compressor_kv_cache.py, on a Tesla V100 and on a Quadro RTX 8000
pre-commit run --from-ref origin/main --to-ref HEAD
pre-commit run mypy-3.10 --all-files --hook-stage manual

## Path coverage

| Path / provider | Format | Key conditions (shape, TP/EP, graph, capability) | Expected selection / fallback | Evidence (source / static / executed / measured) | Unverified scope |
| --- | --- | --- | --- | --- | --- |
| compress_norm_rope_store_triton, sparse attention KV | head_dim 512 | DeepSeek-V4 compressed KV on SM70 and SM75 | same kernel, num_warps 16, maxnreg 128 | measured: model A/B below; executed: kernel tests on both cards | other cards than V100 and RTX 8000 |
| same, indexer KV | fp4 and fp8 indexer cache | head_dim other than 512 | unchanged: num_warps 1, no maxnreg | source; executed: kernel tests | none |

## Acceleration and benchmark contract (required for performance changes)

- Default enabled or opt-in: default enabled, no switch.
- Required CLI options and environment switches: none for the change. Benchmark entry: --pipeline-parallel-size 5 --kv-cache-dtype fp8 --max-num-seqs 1 --max-num-batched-tokens 1024 --max-model-len 343000 --num-gpu-blocks-override 3600, speculative method dspark with 5 speculative tokens, VLLM_USE_V2_MODEL_RUNNER=1.
- KV cache dtype used for the benchmark: fp8.
- Wheel SHA or source commit: fork build, reference tag verified-2026-10-31-prod-v5b, changed f1b54cf46, three commits on that tag (a build pin of the Turing library, which both arms used in the same build; the V100 verify change, not on this path; this change), branch dsv4-compress-maxnreg in github.com/Peuqui/1Cat-vLLM. This branch carries the same commit on main.
- PYTHONPATH, source overlays, or external native libraries used: the fork tree and a patched Triton 3.6 build (sm_70/sm_75 tensor cores) on PYTHONPATH in both arms; native libraries identical in both arms. The kernel tests of this PR ran with the same Triton build and with main's native build.
- User-entry route-hit, speed, and output-quality evidence: OpenAI API server entry, series below.
- Existing kernel/backend registry and capability-rejection reasons: not touched.
- Per-engine configuration and deprecated-variable compatibility: no new or changed setting.
- New/changed environment metadata and generated reference check: none.
- Route snapshot command and intentional changed rows: none (no route changes).
- SM70 registered variables / unregistered reads / model-parameter restrictions, before -> after: no change.

## Test Result

DeepSeek-V4-Flash, PP=5 on 3 Tesla V100 + 2 Quadro RTX 8000 (stage 0 and stage 4 on the RTX 8000), runs T T P T P through the same server entry, counted T#2 + T#4 against P#3 + P#5; T#2's warmup and P#3's boot each coincided with a disk stall on the host, so the 33k first token and the boot time do not count.

Memory peak per card (MiB, lower is better):
- Tesla V100: 30,992 changed, 32,488 reference.
- RTX 8000, last stage: 47,241 changed, 47,789 reference.
- RTX 8000, first stage, boot peak: 39,605 changed, 39,681 reference.

The memory does not reach the KV pool by itself: --num-gpu-blocks-override held the pool at 349,900 tokens in both arms. Graph capture took 0.10 / 0.18 / 0.18 / 0.18 / 0.15 GiB per stage with the change.

Speed (first token: seconds, lower is better; rounds: ms per decode round, lower is better; the tok/s of a round follow the acceptance of the run, so the ms per round are the number to compare):
- Prefill of an 8,921-token prompt: 2.29 / 2.28 s (3,903 / 3,907 tok/s) changed, 2.31 / 2.31 s (3,869 / 3,857 tok/s) reference.
- First token at 178,480 tokens: 41.07 / 41.14 s changed, 44.14 / 43.73 s reference.
- First token at 287,039 tokens: 73.54 / 73.44 s changed, 76.05 / 75.92 s reference.
- First token at 336,216 tokens: 89.85 / 89.87 s changed, 92.29 / 92.18 s reference.
- Decode round at 178,480 tokens: 64.0 / 64.4 ms changed, 64.4 / 64.8 ms reference.
- Decode round at 287,039 tokens: 64.4 / 64.2 ms changed, 64.8 / 64.1 ms reference.
- Decode round at 336,216 tokens: 64.7 / 64.7 ms (52.3 / 52.3 tok/s) changed, 65.1 / 65.1 ms (48.6 / 51.9 tok/s) reference.
- Short decode: 55.91 tok/s (17.9 ms per token) changed, 56.62 tok/s (17.7 ms per token) reference. The rounds take the same time, the difference follows the token path of the greedy answer.

Output: GSM8K-30 29/30 in all arms. Within one build every answer is the same; between the builds 23 of 30 answers differ, through the rounding described above (12,285 against 12,445 tokens, acceptance 4.103 against 4.118). Each build misses one question (12 and 2), both at the 1,024-token limit. Of eight fixed questions, seven answers are byte-identical; the code review answer has 104 instead of 175 tokens and reports the same bug and the same fix.

Tests on this branch (main 4d57473db plus this commit; it merges cleanly into main 6db9c816c), main's native build: tests/kernels/test_compressor_kv_cache.py 32 passed, 6 skipped on a Quadro RTX 8000 and the same on a Tesla V100. pre-commit run --from-ref origin/main --to-ref HEAD: every hook passes except check-layering, which reports vllm/model_executor/layers/fused_moe/runner/moe_runner.py (platform coupling 37 -> 39) exactly as main 4d57473db does. pre-commit run mypy-3.10 --all-files --hook-stage manual: no issues.

🤖 Generated with Claude Code (https://claude.com/claude-code)

https://claude.ai/code/session_0172gtUkBtN8FSrthHLCZe8W

