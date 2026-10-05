Titel: [Perf][Qwen4Exp] PLE disk tier: read a step's pages together, release only their page tables, keep hot rows in a row cache

## Purpose

With the PLE disk cascade from #806, Qwen3.8-Flash-Next at PP4 keeps 17.35 GiB of the FP8 PLE table on stage 0 and reads the remaining 30.34 GiB from the mapped checkpoint in the offload worker. On a host whose RAM cannot hold that part in the page cache (ours: 30 GiB, llama-swap service with MemoryHigh=16G), every decode step reads about 50 rows from disk, and three things made that slow:

1. The rows are copied one page fault after the other, each waiting for the previous one (the offload worker took 930,000 major faults in one benchmark run).
2. With `kernel_config.ple_disk_release_pages`, every touched shard is unmapped as a whole after each gather: an MADV_DONTNEED over about 400 MB of page tables per shard and step.
3. The table is hash-addressed, so the hot rows (frequent n-grams) are scattered over all of it and share each 4 KiB page with rows nobody reads. The page cache holds them 25 times less densely than they need, and under memory pressure the kernel drops their clean pages first: 600 to 730 major faults per second in decode, PLE worker only.

This PR, in three commits:

- Profiles the cascade's disk tier like the whole-table disk lane (`VLLM_PLE_DISK_OFFLOAD_PROFILE`): both lookups share one profiled entry point. No behavior change with the switch off.
- Asks for all pages of a request with MADV_WILLNEED before copying, so the disk serves a step's misses together (prefill does the same per shard). With `ple_disk_release_pages`, it releases the page tables the rows lie in, clipped to the row's shard, instead of whole shards. Releasing only the rows' own pages is not enough: a read fault maps cached neighbours along (fault-around) or a whole large folio, on kernel 7.0 up to 2 MiB for one row, and the page-exact release left 32 of 36 KiB mapped. A fault never maps beyond its page table, so releasing those leaves nothing behind. madvise goes through one cached libc handle instead of a new CDLL per call.
- Adds `kernel_config.ple_disk_row_cache_gib` (default 0, today's behavior): a fixed-size, set-associative row cache in the offload worker with small use counts. A hit raises its row's count; a row read from disk replaces the coldest way of its set only once that count has decayed to zero, otherwise it ages the set by one. Rows that keep coming back stay, while the one-off n-grams of a long prompt pass through. It serves the whole-table disk lane and the cascade's disk tier. The cache is host memory on top of the ranks' pinned share: the start-up check (`check_ple_host_share`) refuses a cache the host cannot hold beside the reserve, and the derived host budget leaves room for it. The field stays out of the compilation hash.

Outputs are unchanged: served rows are the checkpoint's bytes, only how pages are faulted in and released differs. On our production fork, which carries the same disk tier code under environment switches, we compared 3.28 million rows served through the row cache against the checkpoint on the full model: no difference.

Not a duplicate: no open PR touches the disk gather's page handling or adds a hot-row cache (gh pr list --state open --search "PLE disk", "ple_disk", "row cache", "MADV_WILLNEED", "ple cascade", "PLE offload"). Related: Draft #821 adds a bounded draft-prefix prefetch cache (`ple_draft_prefetch`, 8 MiB) in the same `_gather_mapped_rows`. That cache holds rows predicted for the next MTP round; this PR does not prefetch, its row cache keeps rows by reuse. The two can coexist, but they touch the same functions and conflict textually; I will rebase this PR onto #821 if that lands first.

AI assistance was used for this change. I reviewed every line and ran the tests and measurements below.

## Test Plan

```bash
pytest tests/models/qwen4_exp/test_ple.py tests/models/qwen4_exp/test_ple_row_cache.py tests/config/test_ple_cascade_capabilities.py tests/compile/test_sm70_decode_graph.py
pre-commit run --files vllm/config/kernel.py vllm/engine/arg_utils.py vllm/models/qwen4_exp/common/ple.py vllm/models/qwen4_exp/common/ple_row_cache.py vllm/models/qwen4_exp/nvidia/ple_layer.py tests/models/qwen4_exp/test_ple.py tests/models/qwen4_exp/test_ple_row_cache.py
pre-commit run mypy-3.10 --hook-stage manual --files <same>
```

15 new tests: the gather asks for every page before reading, releases page tables rather than whole shards and unmaps what fault-around mapped along; page spans merge and follow page crossings; the cascade's disk tier is profiled per gather; the row cache serves exact rows, takes one row per set and call, keeps recurring rows while one-off rows pass through, frees rows nobody reads again, refuses a budget without one set, and counts against the host share; the cascade worker keeps a row cache only when configured.

## Acceleration and benchmark contract (required for performance changes)

Qwen3.8-Flash-Next-NVFP4 at PP4 on 3x Tesla V100-PCIE-32GB + 2x Quadro RTX 8000, MTP with 4 speculative tokens, `--max-num-seqs 1`, cascade placement as above (17.35 GiB on stage 0, 30.34 GiB on the disk tier), host 30 GiB RAM, llama-swap service with MemoryHigh=16G. Both arms on one build: main f551e0afe plus #873, #874, #715, our draft vocabulary backport and a per-device FLA shared-memory check, native extensions built from source (sm_70); the baseline arm reverts this PR and runs main's defaults (no kernel-config entry), the other arm sets `--kernel-config '{"ple_disk_release_pages":true,"ple_disk_row_cache_gib":0.5}'`. PLE file dropped from the page cache before every boot, a cold boot of the baseline first, then A/B/A/B; the first long request after a boot not counted. Four 29k-token prompts and three short prompts per run.

| PP4 | without this PR | with this PR |
| --- | ---: | ---: |
| decode ms per step at 29k context (lower is better) | 66-69 | 61-64 |
| decode ms per step, 33-token prose prompt | 57-61 | 52-54 |
| decode ms per step, 1,460-token edit prompt | 54 | 54 |
| prefill of the 29k prompts, s | 13.8-13.9 | 13.8-13.9 |
| major page faults per run (system) | 178,080-179,681 | 534-683 |
| I/O stall per run, ms (llama-swap cgroup) | 1,991-2,060 | 1,057-1,077 |

The host had little memory pressure in this run (104 to 159 memory.high events per run in both arms). The gain grows with memory pressure. On our production fork (the same disk tier code under environment switches; its previous build released whole shards after each gather), in a series with heavier pressure (1,059 memory.high events in that run) the previous path took 18.3 s for the 29k prefill and 70-71 ms per step at 29k context and read 4,055 MiB per run, while the fast path in the same series stayed at 13.5-13.8 s, 61-66 ms and 736-785 MiB.

A second series on the same build isolates the two switches (this PR's page requests active in both arms): with them, prefill of the 29k prompts 13.8-14.0 s against 14.4-14.6 s without, decode per step unchanged (61-65 against 60-62 ms at 29k context, 53-55 ms short).

## Test Result

- Tesla V100-PCIE-32GB: 180 passed. Quadro RTX 8000: 177 passed, 3 skipped (existing cases that require an exact SM70 device).
- The two changed test files on main without this change: both fail at collection (they import the new `PLERowCache`).
- pre-commit and mypy-3.10 pass.
- Greedy and quality prompts (8) read by hand in both arms: equivalent, the unknown-person prompt correctly declined. Greedy text differs from our production reference in two of three prompts in both arms alike, as expected across builds.
- The measured build carries this PR's commits before a pre-commit import-order fix in `vllm/config/kernel.py`; no other difference.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
