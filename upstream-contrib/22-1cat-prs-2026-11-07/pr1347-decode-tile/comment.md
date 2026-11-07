I added a third commit to this PR, 6968a8b40 (perf(deepseek-v4): decode score tile for the indexer verifier rows). It changes only the tile that the fused decode scorer of the first commit uses.

The fused multi-row decode route scored a request's verifier rows with the prefill tile of _prefill_index_scores_kernel. That tile spans 1,024 (Volta) or 2,048 (Turing) keys per program, so a decode over the keys of a context bucket starts few programs, and Volta's four tokens per program leave a quarter of the MMA rows empty for six verifier rows. _decode_score_tile now picks the tile from the key count for 64 index heads; other head counts keep the prefill tile. Turing keeps its tile shape with 256 keys per program up to 16k keys and 512 above, so its scores stay bit-identical. Volta takes two tokens per program (block_k 128 and 8 warps up to 16k keys, block_k 64 and 4 warps above). The callers pass the tile to _prefill_scores_triton.

Kernel level, one call (gather + score) captured in a CUDA graph, 6 rows of one request, 64 index heads, at the C4 key counts of the decode buckets; ms per call, lower is better, old tile -> new tile:
4,096 keys: Tesla V100 0.093 -> 0.017, Quadro RTX 8000 0.122 -> 0.021
16,384 keys: 0.094 -> 0.045, 0.101 -> 0.039
32,768 keys: 0.104 -> 0.087, 0.121 -> 0.081
65,536 keys: 0.198 -> 0.152, 0.161 -> 0.138
124,992 keys: 0.382 -> 0.280, 0.299 -> 0.258
Against the old tile, the top-512 sets of every row were equal at all 15 checked key counts from 1,024 to 124,992. The Turing scores are bit-identical; the Volta scores differ by at most 4.1e-2 relative (at 57,344 keys, relative to the old score clamped at 1e-3).

Model A/B on our fork (github.com/Peuqui/1Cat-vLLM, branch fork-main-v6 at 31d5473d0, which is this commit on top of our build, against the same build without it): DeepSeek-V4-Flash, PP=5 on 3 Tesla V100 + 2 Quadro RTX 8000, the server entry of the description with --max-model-len 500000 and --num-gpu-blocks-override 4220 (500,000 tokens need #1344), the same in both arms. Both arms also carry what only our fork has so far: DSpark and the context buckets of #1221 with a finer list of 16 buckets up to 500,000 tokens, and prompt-length-dependent prefill chunks (--short-prompt-chunk-tokens 512 --short-prompt-threshold 6144). Runs T T P T P, counted T#2 + T#4 against P#3 + P#5. Decode round in ms, lower is better, changed / reference:
32,668 tokens: 56.0 / 55.9 against 57.6 / 57.5
91,790 tokens: 57.6 / 57.4 against 59.0 / 58.7
178,480 tokens: 60.8 / 60.7 against 62.7 / 62.1
248,418 tokens: 61.4 / 61.3 against 62.1 / 62.3
287,039 tokens: 65.1 / 64.8 against 65.0 / 64.9
323,199 tokens: 65.1 / 65.1 against 66.3 / 65.7
456,810 tokens: 67.0 / 66.9 against 67.9 / 67.8
486,316 tokens: 67.5 / 67.1 against 68.9 / 69.5
Unchanged: prefill of an 8,921-token prompt 2.28 / 2.27 s (3,917 / 3,926 tok/s) changed, 2.28 / 2.27 s (3,914 / 3,926 tok/s) reference; short decode 55.88 / 55.87 tok/s changed, 55.90 / 55.89 tok/s reference (17.9 ms per token); first token at 486,316 tokens 146.15 / 146.79 s changed, 146.62 / 146.78 s reference. Output: GSM8K-30 29/30 in all four counted runs with the same acceptance length (4.103); the eight fixed questions token-identical between the arms. Memory peak per card identical in all counted runs (39,885 / 30,822 / 47,133 / 30,862 / 30,862 MiB).

Tests on this branch (6968a8b40), with a Triton build that lowers tl.dot to tensor cores on sm_70/sm_75 and with stock Triton 3.6:
[FILL GPU]
pre-commit run --from-ref [FILL MAIN] --to-ref HEAD: [FILL PRECOMMIT]

I updated the first paragraph of the description: the PR's own diff is now the last three commits (git log e9147089c..6968a8b40).
