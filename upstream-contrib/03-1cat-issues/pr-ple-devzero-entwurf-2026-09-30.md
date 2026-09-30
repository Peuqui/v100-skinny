Title: [Bugfix][Qwen4Exp] Do not take shared anonymous memory for a PLE disk shard

## Purpose

`_advise_random_file_access` requires the PLE disk-tier shards to be
file-backed: it looks the tensor's address up in `/proc/self/maps` and
accepts the mapping if the line names a path. Shared anonymous memory, for
example a buffer from `mmap.mmap(-1, n)` with its default flags, is listed
there as `/dev/zero (deleted)`. It passed the check as a file, so the disk
tier kept the table in shared memory that only swap can evict while it
reported file-backed shards, and it recorded `/dev/zero` as the shard's file.
With a release of mapped pages via `MADV_DONTNEED` (as the page release in
#646 adds) such a shard would even lose its contents. The check now rejects
`/dev/zero`.

Found while testing a loader that read shards into anonymous buffers: the
guard should have refused them and did not.

Why this is not a duplicate. `gh pr list --state open --search` for
"dev/zero" and "_advise_random_file_access" finds nothing related. With
`git merge-tree` the branch merges cleanly with main and with every open PR
that touches the same files and merges with main itself (#717, #707, #646);
#689 already conflicts with main. The test lives in its own file because #646
and #717 rework `tests/models/qwen4_exp/test_ple.py`.

AI assistance was used for this change. Every line was reviewed and the test
below was run by the submitter.

## Test Plan

```bash
pytest tests/models/qwen4_exp/test_ple_disk_shard_guard.py
pytest tests/models/qwen4_exp/test_ple.py
# counter-check: ple_layer.py reverted to main, new test kept
pre-commit run --files <changed files>
pre-commit run mypy-3.10 --hook-stage manual --files <changed files>
```

## Test Result

On this branch (main d3046986), both files in one run:

- Tesla V100: 79 passed (the new test and the 78 of `test_ple.py`)
- Quadro RTX 8000: 76 passed, 3 skipped (they need an SM70 device)

With main's `ple_layer.py` the new test fails, because the anonymous buffer
is accepted. pre-commit clean, mypy-3.10 passed.

---
Status: gesendet 2026-09-30 als https://github.com/1CatAI/1Cat-vLLM/pull/741 (Zweig pr-ple-devzero, 3d40ec09)
