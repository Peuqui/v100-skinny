# Entwurf: PR an 1Cat — PLE-Shards ohne Zwischenkopie auf die Karte kopieren

Status: VERWORFEN als eigener PR (24.09.). Die Messung ohne Kaskade
(STAND 43) zeigt keinen belegbaren KV-Gewinn; die KV-Zahl schwankt zwischen
Boots mit dem Compile-Cache. Belegt sind nur der Allokator-Effekt und der OOM
auf der gefüllten Store-Karte — beides Kaskaden-Themen. Die Korrektur geht
als Teil von #646 mit. Text bleibt als Vorlage.

Titel:

[Bugfix][Qwen4Exp] Copy PLE checkpoint shards to the device without a staging copy

---

## Purpose

`copy_ple_embedding_shard_` (vllm/models/qwen4_exp/common/ple.py, from #538)
moves every checkpoint slice with `source.to(device=..., dtype=...)` and then
copies it into the table:

```python
target.copy_(source.to(device=target.device, dtype=target.dtype))
```

The staging tensor is freed right away, but PyTorch's caching allocator keeps
its block reserved on the device. For `nvidia/Qwen3.8-Flash-Next-NVFP4` one
checkpoint shard of the PLE table is 2,500,012 rows of 160 bytes, 0.37 GiB, and
every rank that holds rows of the table in device memory loads it shard by
shard. `copy_` converts device and dtype itself, so the staging step can go:

```python
target.copy_(source)
```

## Effect

- Measured in isolation on a Quadro RTX 8000, one 0.37 GiB shard copied in four
  slices into a 1.5 GiB table: 382 MiB of device memory kept after loading with
  the staging copy, 0 MiB with `copy_` directly (`torch.cuda.memory_reserved()`
  equal to the table).
- On a full model the retained blocks count against the KV budget of the stage
  that owns the table. LÜCKE-KV: Flash-Next, TP2 x PP2, pinned-host split with
  `VLLM_QWEN4EXP_PLE_HOST_GIB=3`, `Available KV cache memory` of the first stage
  before / after: … GiB / … GiB.
- Output is unchanged: the same bytes land in the same rows.

## Duplicate check

`gh pr list --state all --search "copy_ple_embedding_shard_"`, `"PLE staging
copy"`, `"ple shard copy device"` and `"caching allocator PLE"`, plus the same
searches over issues: no open or closed PR or issue about this. #646 (PLE
overflow cascade, open, mine) will carry the same change once this is merged;
it is proposed separately because it affects every PLE deployment, not only the
cascade.

## Test plan

```bash
pytest tests/models/qwen4_exp/test_ple.py -q
pre-commit run --files vllm/models/qwen4_exp/common/ple.py tests/models/qwen4_exp/test_ple.py
```

## Test result

- New test `test_shard_copy_to_the_device_keeps_no_staging_memory`: the
  allocator reserves nothing beyond the table after a shard copy; it fails on
  the old line (reserved grows by the slice size).
- `tests/models/qwen4_exp/test_ple.py`: … passed (fill in on the PR branch).
- pre-commit: … (fill in).

## AI assistance

Written with Claude (Anthropic) as a coding assistant: it found the retained
block while debugging an out-of-memory error, wrote the fix and the test, and
ran the measurements above. I reviewed every changed line and can defend the
change end to end.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
