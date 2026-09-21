# PR-Entwurf A: Präfix-Cache überlebt keine fremde Anfrage

Im Fork enthalten als `901bf307` (Branch `qwen4exp-ple-tier-cascade`).
PR-Branch: `fork/prefix-cache-reuse-dead-window-blocks`, Commit `acf9a648`,
gepusht 21.09.2026.

**HÄNGT AN PR #657.** Der Branch sitzt NICHT auf `origin/main` auf, sondern auf
`fork/swa-prefix-cache-eagle-mask` (456a62f5, = PR #657). Grund:
`reachable_block_mask` — die Funktion, auf der die ganze Argumentation ruht,
weil sie festlegt, welche Blöcke einer Fenstergruppe überhaupt gecacht werden —
existiert auf `origin/main` nicht; sie kam mit unserem Backport von
vllm-project/vllm#44082. Gegen main allein lässt sich der Patch nicht anwenden
(Konflikte in `kv_cache_coordinator.py` und `single_type_kv_cache_manager.py`).
Wird #657 gemerged, rebast dieser PR darauf; wird #657 abgelehnt, muss der Fix
neu gedacht werden, weil dann auch die Eintragungsmaske eine andere ist.

Testlage auf dem PR-Branch: 133 grün. Ein Fehlschlag,
`test_deepseek_v4_tuple_width_minimizes_physical_pool_pages`, ist VORBESTEHEND
(scheitert auf 456a62f5 ohne unsere Änderung, Mock ohne `max_in_flight_tokens`)
und gehört nicht in diesen PR.

ERÖFFNET 21.09.2026 als https://github.com/1CatAI/1Cat-vLLM/pull/667 (Peuquis Go),
mit Abhängigkeitshinweis auf #657; CI grün nach ruff-format-Nachbesserung.

## Duplikatsprüfung (AGENTS.md Pflicht, gelaufen 21.09.2026)

- `gh pr list --state open --search "prefix cache"` → #617 (Mamba-Retention),
  #657 (unser EAGLE-Masken-Backport), #603, #598, #665, #622. Keiner berührt
  die Verdrängungsreihenfolge in `free_blocks`.
- `gh pr list --state open --search "eviction free blocks"` → #239, #617.
  Kein Überlapp.
- `gh issue list --state open --search "prefix cache evict"` → #632 (native
  KV-Auslagerung), #490 (TP2-Decode), #205 (EAGLE-Maske, von uns als #657
  beantwortet). Kein bestehender Bericht zu diesem Fehler.

## PR-Titel

`[Bugfix][Core] Reuse a sliding window's dead blocks before other requests' cached ones`

## PR-Body

## Purpose

A cached prefix does not survive an unrelated request, even when the pool is
nearly empty. On DeepSeek-V4-Flash (262k window, 2400 blocks, chunk 512) a 22k
prompt answers in 0.8 s when repeated, but pays the full 14.4 s prefill again
as soon as any request above roughly 5k tokens ran in between. The MLA group
holds 89 of 2400 blocks at that point, so this is not capacity pressure.

Sliding-window groups cache only the short run of blocks at the end of each
alignment segment — that is what `reachable_block_mask` registers, because a
hit can only end on such a boundary and needs just the last `sliding_window`
tokens before it. With block size 4 and an 8-token window that is 3 of 64
blocks, so about 95 % of what such a group releases mid-prefill carries no
hash and can never serve a hit.

`free_blocks` appends all of it to the back of the free queue, behind other
requests' cached blocks. A 30k prefill therefore takes 13730 blocks from the
front of the queue, evicting foreign prefixes as it goes, although it only
ever touches 1227 distinct blocks and could have reused what it had just
released. Upstream vLLM splits the two cases in `free_blocks` (blocks without
a hash are prepended and reused first, cached ones are appended for LRU);
`prepend_n` does not exist in this repository at all, so the split cannot be
expressed.

This restores that split and adds the part upstream does not need: a sliding
window's released blocks are dead even when they still carry a hash, so those
groups opt into `reuse_first` explicitly. The one exception is the run at the
last alignment boundary of the prompt, which is exactly what a repeat of that
prompt looks up — it stays allocated until the request finishes and is then
released with everything else. The held range is derived with the same formula
as `reachable_block_mask`, including the EAGLE shift, and the coordinator hands
each manager its `lcm_block_size` the way it already hands out `use_eagle`.

`test_prefill`, `test_prefill_plp` and `test_evict` pinned the old all-FIFO
order; they now carry upstream's expectations and its "partial blocks (without
hash) at head" comment.

This builds on #657: `reachable_block_mask`, which decides what a
sliding-window group caches in the first place, arrives with that backport and
is what the held range here is derived from. The branch is therefore based on
#657 rather than on `main`, and should be rebased once that one lands.

AI assistance was used for this change. Every line was reviewed and the tests
below were run by the submitter.

## Test Plan

```bash
pytest tests/v1/core/test_kv_cache_utils.py tests/v1/core/test_prefix_caching.py
```

End-to-end on 3x V100 + 2x RTX 8000, PP5, DeepSeek-V4-Flash-NVFP4 with DSpark
(262144 context, 2400 blocks, chunk 512): prompt A of 22914 tokens, repeated,
then an unrelated 30k request, then A again, then a 125k request, then A again.

## Test Result

134 passed, including two new tests: `test_free_kv_cache_block_queue_prepend_n`
and `test_free_blocks_reuses_uncached_blocks_before_cached_ones`.

| Measurement | before | after |
|---|---|---|
| A cold | 14.4 s | 14.4 s |
| A repeated | 0.8 s | 0.8 s |
| A after a 30k request | 14.4 s | 0.8 s |
| A after a 125k request | 14.4 s | 0.8 s |
| decode step | 89-90 ms | 89-90 ms |
| V100 VRAM | 31776 MiB | 31776 MiB |

125k needle test 4/4 in both cases. Instrumented counters for the 30k request:
13730 block allocations over 1227 distinct blocks, 13578 of 13730 releases now
returned to the front of the queue; the following request for A allocates 203
blocks instead of 9983.
