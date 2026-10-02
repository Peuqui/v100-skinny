# Draft token list for the MTP drafter (`draft_token_map`)

Plan Oct 2026, item 2 — see STAND.md, section "Arbeitsplan Oktober 2026".

The MTP drafter of Qwen3.8-Flash-Next projects every draft token onto the full
248,320-row lm_head (1.88 ms per draft token on a V100). With
`speculative_config.draft_token_map` (fork backport of vllm#59740, branch
`fork-next`) it projects only onto a list of token ids; the target still
verifies with the full vocabulary, so output is unchanged.

- `qwen38-flash-next-de-en-code-98304.json` — the list in use (98,304 ids).
  Coverage on held-out AIfred sessions 99.8 %, on unseen code 99.4–99.9 %
  (C++, CUDA, Rust, SQL, YAML, JS, shell).
- `build_draft_vocab.py` — builds it. Ranking: (1) frequency in model-generated
  text (`corpus-flashnext-2026-10-02.jsonl`, from `draftvocab_corpus.py`),
  (2) German running text (AIfred docs/prompts), (3) Python sources,
  (4) `/usr/share/dict/ngerman`, then BPE order; special tokens always.
  Paths are this machine's; for another language swap the text and dictionary.
- `lookup_bench.py` / `lookup_bench_greedy.py` — prose, code and a copy-heavy
  edit, at production sampling or greedy (greedy = identical output across
  lossless variants, so only step time differs).
- `prompt_lookup_sim.py` — offline estimate of prompt-lookup drafting gains.
