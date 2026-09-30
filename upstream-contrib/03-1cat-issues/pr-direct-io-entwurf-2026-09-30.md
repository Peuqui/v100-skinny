Title: [Model Loader] Add a direct I/O safetensors load strategy

## Purpose

Loading memory-maps every checkpoint shard, so every byte a worker touches
lands in the page cache and stays there. With a checkpoint larger than host
RAM (123 GiB for Qwen3.8-Flash-Next, ~160 GiB for DeepSeek-V4-Flash, on a
machine with 30 GiB) the cache pushes other processes into swap while the
model loads, and under a cgroup memory limit it pushes vLLM's own workers out.

`--safetensors-load-strategy direct` reads the decoder-layer tensors, the bulk
of a checkpoint, with `O_DIRECT` into private buffers, in coalesced runs in
file order (`vllm/model_executor/model_loader/direct_io.py`). Under pipeline
parallelism each stage reads only the decoder layers it owns, taken from the
`start_layer`/`end_layer` of its decoder stack; for a multimodal model that is
the stack of its language model, since an encoder tower built with
`make_layers` carries a range of its own. All other tensors stay
memory-mapped, so a stage reads only the embeddings, heads, vision tower or
MTP layers it actually uses, and each one's pages are released from the page
cache once the loader has consumed it (`madvise` then `posix_fadvise`
`DONTNEED`, whole pages inside the tensor only). A model can keep tensors
mapped with `map_checkpoint_weight`; Qwen4Exp does so for the PLE embedding
shards, which each rank reads only in part and the disk tier serves from the
mapped checkpoint. The strategy is opt-in; nothing changes without the flag.

Tensor-parallel ranks of one pipeline stage select the same decoder tensors,
and without a shared page cache each would read the whole stage from disk.
The group's first rank reads each run and broadcasts it over the group's CPU
(gloo) group (about 4 GiB/s here, against ~0.9 GB/s from the SSD). The ranks
first check that they select the same runs, and for every run each rank
reports whether it has its part (the leader the data, the others a receive
buffer), so the group fails together instead of leaving a rank waiting in a
broadcast. Under expert parallelism, where ranks keep different experts, each
still reads its own.

Two loader passes read far more than they use. Direct I/O makes this visible,
so they are fixed here as well:

- Handing out a memory-mapped tensor already reads it: `safe_open.get_tensor`
  touches its first pages, and readahead over tensors that lie next to each
  other reads almost the whole file. The PLE offload worker streamed the whole
  checkpoint and dropped everything but the PLE tensors afterwards, which read
  all 123 GiB of Flash-Next through the page cache next to what the stages
  read themselves. `get_all_weights` takes an optional `skip_weight` now; the
  worker passes its name filter there and always loads lazily.
- The Qwen3.5 and Qwen4Exp MTP drafters ship inside their target's checkpoint
  and read it a second time for `load_weights` to keep only the MTP tensors.
  `skip_checkpoint_weight`, as the DSpark drafter already has, leaves the rest
  out before it is read; it uses the same remapping as `load_weights`.

The PLE shard name check moves to `qwen4_exp/common/ple.py`, so the NVIDIA and
AMD Qwen4Exp classes share it, and both `ple_layer.py` files take the prefix
from there. The AMD path gets the same `map_checkpoint_weight`; I have no AMD
hardware, so that path is untested.

Why this is not a duplicate. `gh pr list --state open --search` for "direct
io", "O_DIRECT", "safetensors load strategy", "skip_checkpoint_weight", "MTP
drafter load" and "page cache" finds nothing that reads weights this way. With
`git merge-tree` against every open PR that touches the same files, the branch
merges cleanly with main and with #707, #702, #696, #664 and #312. It
conflicts with my own #646 and #717 in one import line of
`nvidia/ple_layer.py` (each adds a name at the top of the same import list);
whichever lands second, I will rebase it. #709, #700, #693, #690, #689, #687,
#682, #629, #547, #523, #504, #349, #348 and #327 already conflict with main.

AI assistance was used for this change. Every line was reviewed and the tests
below were run by the submitter.

## Test Plan

```bash
pytest tests/model_executor/model_loader/test_direct_io.py
pytest tests/models/test_mtp_skip_checkpoint_weight.py
pytest tests/v1/worker/test_ple_offload_worker.py tests/models/qwen4_exp/test_ple.py
pytest tests/models/qwen3_5/test_mtp_stage_local.py tests/models/qwen4_exp/test_mtp_stage_local.py
pre-commit run --files <changed files>
pre-commit run mypy-3.10 --hook-stage manual --files <changed files>
```

End to end: each model booted through llama-swap with the default load and
with `--safetensors-load-strategy direct`, the page cache, swap and memory
pressure (PSI) logged every 5 s, then three greedy prompts (200 tokens each)
compared with the other load. Mixed machine: 2x Quadro RTX 8000 and 3x Tesla
V100, 30 GiB host RAM, checkpoints on a USB SSD (~930 MB/s), vLLM under a
16 GiB cgroup `MemoryHigh`.

## Test Result

The new tests check bitwise equality with `safe_open` (one coalesced run and
one run per tensor), the stage filter, the stage range of a multimodal model
and that two partial ranges raise, that non-decoder tensors and the ones a
model asks for stay file-backed and still read correctly after their pages
were released, that both drafters skip exactly what `load_weights` drops, and
that the offload worker touches only the PLE tensors. With two gloo processes
they check that a follower receives every run without reading the disk, and
that the group fails together when the ranks select different tensors, when
the leader cannot open or read the shard, and when a follower cannot map its
buffer, without leaking the shard's file descriptor.

On this branch (main d3046986), all listed files in one run:

- Tesla V100: 146 passed, 1 skipped
- Quadro RTX 8000: 143 passed, 4 skipped (three of them need an SM70 device)

With a single visible device `test_offload_distributed_sets_config_only_for_model_parallel`
fails on main as well when `test_ple_offload_worker.py` runs on its own
(#739); in the combined run above it passed. pre-commit clean,
mypy-3.10 passed.

Time from the first request, which starts the server, to the end of its
200-token answer, swap growth during the load, and the highest memory pressure
(PSI full avg10), each
from the same boot:

| Model | default | direct | swap growth default / direct | memory PSI max default / direct |
|---|---|---|---|---|
| DeepSeek-V4-Flash (PP5, 5 GPUs) | 321 s | 252 s | +9 GiB / 0 | 26 % / 0.3 % |
| Qwen3.8-27B NVFP4 (TP2) | 202 s | 152 s | +5 GiB / 0 | 21 % / 0.2 % |
| Qwen3.8-Flash-Next NVFP4 (PP4, PLE disk tier) | 282 s | 200-230 s | +9 GiB / 0-4 GiB | 24 % / 0-11 % |
| Qwen3.8-Flash-Next NVFP4 (TP2xPP2, 2 GiB PLE host tier) | 571 s | 384 s | +10 GiB / +7 GiB | 18 % / 33 % |

The Flash-Next PP4 range covers two direct boots, each after a direct boot; a
direct boot right after a default boot in the same service took 309 s. The
direct column predates the shared read, so each tensor-parallel rank read for
itself. With it, and otherwise the same code, Qwen3.8-27B (TP2) loads its
weights in 34 s instead of 43 s and gives its first answer after 142 s
instead of 152 s. For Flash-Next TP2xPP2 with the PLE tables on device and
disk only (no host tier), the weights went from 220 s to 165 s and the time
until the server is ready from 301 s to 236 s, with no swap growth. The
remaining swap in the host-tier row comes from its pinned host share, which
the kernel cannot reclaim.

Greedy outputs are identical between the default load and direct I/O for all
three models, and with the shared read for both tensor-parallel setups. With
the offload-worker fix
alone and the default load, Flash-Next stays at 295 s and +8 GiB, since the
stages then touch every tensor themselves; the two changes only pay off
together.

---
Status: gesendet 2026-09-30 als https://github.com/1CatAI/1Cat-vLLM/pull/740 (Zweig pr-safetensors-direct-io, 65ec7ad9)
