# PR-Entwurf: SimpleCPUOffloadConnector ordnet das Wegschreiben hinter den Rechen-Stream

Stand 20.09.2026: Änderung fertig und im Produktions-Checkout abgenommen,
NOCH NICHT committet, kein Worktree, nichts gepusht, kein PR eröffnet.
Geändert: 3 Dateien (+21/−5) plus eine neue Testdatei (4 Fälle).

Vor dem Absenden von Peuqui zu lesen: der Diff und dieser Text. AGENTS.md
verlangt, dass ein Mensch jede Zeile vertreten kann.

Ehrlich vorweg, gehört so in den PR: Es ist ein Wettlauf zweier Streams. Ein
Test, der ihn ohne den Fix zuverlässig rot zeigt, existiert nicht und kann
nicht existieren. Der Beleg ist die Messreihe — gefunden, gemessen,
nachvollzogen, danach reproduzierbar weg. Der beigefügte Test sichert die
Struktur (Barriere vorhanden, Quellordnung je Richtung richtig), nicht das
Zeitverhalten.

## PR-Titel

`[Bugfix][KV Offload] Order SimpleCPUOffloadConnector stores behind the compute stream`

## PR-Body

### Purpose

`SimpleCPUOffloadWorker.get_finished()` issues the GPU→CPU copy on its own
transfer stream, from a background thread, without ever ordering that stream
behind the compute stream. On top of that, `build_params()` sets
`srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_ANY` for **both** directions.

The store reads the live KV cache while the compute stream is still writing
into it. Nothing connects the two, so the copy can capture a block halfway
through being written. The block's hash is stamped regardless, so the damaged
copy enters the CPU prefix cache as valid data. Nothing fails, nothing is
logged. The next request whose prefix matches restores the block and answers
from a poisoned context.

The sibling path in this repository already handles this correctly —
`vllm/v1/kv_offload/cpu/gpu_worker.py` does
`stream.wait_stream(torch.cuda.current_stream())` before offloading and keeps
STREAM source ordering for GPU→CPU, with a comment stating why: the compute
stream keeps writing the source. `SimpleCPUOffloadConnector` does neither.

Both halves are required. The barrier alone is not enough, because with
`ANY` the driver is explicitly permitted to pull the source reads ahead of
it; `ANY` remains correct for CPU→GPU, whose source is host pinned memory
that no GPU stream writes.

Measured on DeepSeek-V4-Flash (5-stage pipeline parallel, 2x RTX 8000 +
3x V100, FP8 KV cache, `cpu_bytes_to_use` 5 GiB). Sequence: two 18k requests,
a 30k needle, then a 62k needle that shares a ~2k-token prefix with the 30k
one, so its opening blocks come back from the CPU pool.

Without the fix, with the offloaded blocks re-read and compared against
their GPU source right after the transfer event completed (the scheduler
still holds a reference on those GPU blocks, so the source cannot have been
reused):

```
PP1: 21504 block/tensor pairs compared, 12 differ
PP2: 21504 compared,  9 differ
PP3: 21504 compared,  8 differ
PP0, PP4: 0 differ

event=0  model.layers.17.attn               gpu_block=28  cpu_block=1     583 of  1728 bytes differ
event=0  model.layers.18.attn.indexer.k_cache gpu_block=28 cpu_block=1    129 of  8640 bytes differ
event=89 model.layers.16.attn               gpu_block=598 cpu_block=2489 2255 of 37440 bytes differ
```

Partial differences in the middle of a block — a mix of old and new content,
which is what a copy looks like when it races the writer. The end-to-end
effect: the 62k needle returns 0 of 4 facts and degenerates into token salad.

With the fix, same sequence, same comparison: **0 differing pairs on all five
ranks**, needle 4 of 4, twice in a row. Step time is unchanged at 82/84 ms
against 81/83 ms before, so the barrier costs nothing measurable here.

The fault is timing-dependent: the same sequence produced a clean 4-of-4 run
and a corrupted one under identical conditions. Reproducing it through output
quality alone is unreliable; the block comparison is what makes it visible.

### Test Plan

- New `tests/v1/simple_kv_offload/test_store_stream_ordering.py`: the store
  queues the barrier before the copy, the load does not take one, and
  `build_params` selects STREAM ordering for stores and ANY for loads.
- Existing `tests/v1/simple_kv_offload/` suite.
- End-to-end on the hardware above: the sequence described in Purpose, run
  twice, plus a restore probe (30k needle, 62k evictor with a different seed,
  identical 30k needle again) checking that the answer is unchanged.

### Test Result

- `tests/v1/simple_kv_offload/`: 19 passed, 8 skipped.
- Against the unpatched sources, `test_store_is_ordered_behind_the_compute_stream`
  fails: `assert 'store_stream.wait_stream' in ['backend.launch_copy']`.
- End-to-end: 0 differing pairs, needle 4 of 4 in both runs, restore probe
  returns a character-identical answer, step time unchanged.

---

## Messbelege (nicht Teil des PR-Bodies)

- Repro und Buchführung je Anfrage: scratchpad `repro_measured.py`
  (cached tokens gegen Delta der GPU-Präfix-Zähler; die 62k-Anfrage holt
  2.048 Token aus dem CPU-Pool bei 0 GPU-Treffern).
- Rückhol-Probe: scratchpad `cpu_load_probe.py`.
- Gegenprobe ohne Offload aus der Vorgängersitzung: scratchpad
  `indexer_cublas.log`, gleiche Kette, 4 von 4.
- Die Vergleichs-Instrumentierung lag temporär in `worker.py` hinter
  `VLLM_OFFLOAD_VERIFY_STORES=1` und ist vor dem Commit entfernt worden.

## Offen

- Worktree anlegen, committen, pre-commit und mypy-3.10 laufen lassen.
- Der zweite Kandidat bleibt bewusst unangetastet: `manager.py` nimmt
  `num_computed_tokens - num_output_placeholders` als „fertig gerechnet" und
  ignoriert `num_in_flight_tokens`, das der Scheduler an anderer Stelle genau
  dafür verwendet. Die Barriere wartet auf den gesamten Rechen-Stream und
  deckt diesen Fall mit ab; die Messung zeigt null Abweichungen. Erst wieder
  aufgreifen, wenn sie das nicht mehr tut.
