# PR-Entwurf B: Drafter liest den Checkpoint nicht mehr doppelt

Fork: `f639adf6` (Branch `qwen4exp-ple-tier-cascade`), im Betrieb seit 21.09.
PR-Branch: `drafter-skip-checkpoint-weights` auf `origin/main` (949728e8),
Commit `8ebd1c4e`, Worktree `1Cat-vLLM-pr-drafterskip`. Gepusht 21.09.2026.

Unterschied zum Fork-Commit: nur Lader + DSpark-Regel + Tests. NICHT dabei
(Fork-only): `embed.weight`-Zuordnung (auf main gibt es sie nicht, dort liest
der Drafter damit ausschließlich `mtp.*`), `torch.accelerator`-Umstellung,
`method-assign`-Ignore, Test-Attrappe für `main_norm.weight`.

ERÖFFNET 21.09.2026 als https://github.com/1CatAI/1Cat-vLLM/pull/669 (Peuquis Go „Push der PRs“),
Branch gepusht nach fork/drafter-skip-checkpoint-weights.

## Duplikatsprüfung (21.09.2026, AGENTS.md)

- 1Cat, `gh pr list --state open --search` mit "skip checkpoint weight",
  "drafter load weights", "dspark load", "safetensors skip before reading",
  "load only mtp weights": nur fachfremde Treffer (#640, #604, #577, #519,
  #662, #576, #636, #603, #553, #664).
- Kein Treffer zum doppelten Lesen des Checkpoints durch einen Drafter.

---

Titel: [Perf][Loader] Skip checkpoint tensors a model does not load before reading them

## Purpose

A DSpark drafter ships inside its target's checkpoint, so `SpeculativeConfig`
points the drafter at the target path (`self.model = self.target_model_config.model`)
and the default loader iterates every shard of the target a second time — only
for `load_weights` to drop everything that is not `mtp.*`.

On DeepSeek-V4-Flash-NVFP4 with DSpark (165 GB, 46 shards, host RAM far
smaller than the checkpoint) that second pass cannot come from the page cache.
With pipeline parallelism it runs on the last stage after all stages have
finished their own weights, so every other stage waits for it: 255 s of a
10:25 min boot.

This lets a model name the checkpoint tensors it does not load. A model can
offer `skip_checkpoint_weight(name) -> bool`; `DefaultModelLoader` passes it to
`safetensors_weights_iterator` through a new `Source.skip_weight` field (next to
`allow_patterns_overrides`, which is already read from the model the same way),
and the iterator drops accepted names before `get_tensor`, i.e. before the
bytes are read. It sits next to the existing index filter and EP filter; the
three checks now go through one helper, `_keep_weight`, instead of being
repeated in each load strategy.

DSpark implements the rule with the name mapping it already uses in
`load_weights`: `_remap_dspark_name(name) is None` means "not mine". Models that
do not offer the method are unaffected.

The multithreaded iterator is left out on purpose: it reads each file with
`load_file` before any filter could apply, so a name filter there would not
save I/O (the EP filter has the same gap upstream, vllm-project/vllm#48891).

AI assistance was used for this change. Every line was reviewed and the tests
below were run by the submitter.

## Test Plan

```bash
CUDA_VISIBLE_DEVICES="" pytest \
  tests/model_executor/model_loader/test_ep_weight_filter.py \
  tests/model_executor/model_loader/test_safetensors_index_filter.py \
  tests/v1/spec_decode/test_dspark.py
```

End-to-end on 3x V100 + 2x RTX 8000, PP5, DeepSeek-V4-Flash-NVFP4 with DSpark
(`--speculative-config '{"method":"dspark","num_speculative_tokens":5}'`),
cold boot, stage timestamps from the log and bytes read from `/proc/diskstats`.

## Test Result

60 passed. New: `test_model_skip_rule_drops_tensors_before_reading` (skipped
names never reach `get_tensor`) and `test_loader_applies_the_model_skip_rule`
in the existing EP filter suite, and
`test_dspark_skips_every_checkpoint_weight_it_does_not_load`. Both loader tests
fail on `main` without the change.

| Boot phase | before | after |
|---|---|---|
| target weights, all stages (`Loading weights took`) | 276 s | 286 s |
| drafter on the last stage | 255 s | 17 s |
| boot to `Application startup complete` | 10:25 min | 6:03 min |

Disk read during the boot after the change: 177 GiB, i.e. the checkpoint once
plus the drafter's own file. DSpark mean acceptance length afterwards 2.6–3.5,
answers unchanged.

## Not a duplicate

Open PRs searched for "skip checkpoint weight", "drafter load weights",
"dspark load", "safetensors skip before reading" and "load only mtp weights";
none touches how a drafter reads a shared checkpoint.
