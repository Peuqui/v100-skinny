# PR-Entwurf D: Spec-Decode-Zustand an die nicht-letzten PP-Stufen (Folge zu #511, Kontext #439)

Worktree: `vllm-research/1Cat-vLLM-pr-ppspec`, Branch `pp-spec-state-transport-pr`
auf `origin/main` (b711d530), Commits cf0e5d6c (DSpark-Embedding) und d788bbd1
(Transport). Stand 19.09.2026 abends: Tests, Gegenprobe, pre-commit, mypy-3.10 und
ein echter Serverlauf mit Negativkontrolle sind grün. NICHT gepusht, NICHT eröffnet.

Das ist die PRODUKTIONSVARIANTE aus dem Fork (gloo cpu_group,
`_count_contiguous_spec_tokens`), nicht der alte NCCL-Entwurf vom 06.09.
(`pr-439-pp-spec-state-transport.md`, lokal noch als Branch
`ppspec-old-nccl-variant`). Bewusst nicht enthalten: `_pp_check_token_ids`
(Fork-Diagnose vom 19.09.) und der sm75-GDN-Zweig in
`_pp_receive_spec_decode_state` (auf 1Cats main nicht sauber testbar).

Vor dem Absenden von Peuqui zu lesen: `git -C vllm-research/1Cat-vLLM-pr-ppspec
diff origin/main HEAD` (3 Dateien + 1 neuer Test) und dieser Text.

ACHTUNG Abgrenzung #539 (DSYZayn, DSv4-Flash TP4×PP2): dessen Log bricht im
Custom-All-Reduce ab (`custom_all_reduce.cuh:1735`, illegal memory access im
Speicher-Probelauf). Das behebt dieser PR NICHT. Nicht als Fix für #539 ausgeben.

## PR-Titel

`[Bugfix][Spec Decode] Ship the speculative round state to non-last PP ranks`

## PR-Body

## Purpose

Follow-up to #511 (context: #439). With pipeline parallelism, a drafter and
async scheduling, the non-last ranks never learn what the last rank sampled.
`_pp_broadcast_prev_sampled_token_ids` asserts a `[num_reqs, 1]` tensor, which
the speculative sampler does not produce, and the draft token ids that the next
step scatters into `input_ids` exist on the last rank only. On `main` with #574
and #636 applied (Qwen3.8 MTP needs both to get this far under PP) the first
speculative step ends on rank 0 with

```
RuntimeError: Speculative decode scheduled draft input slots, but the worker
has no draft token tensor to scatter.
```

and the request hangs.

This change ships the round state:

- The last rank broadcasts the sampled matrix, padded with -1 to the static
  shape `[num_reqs, num_spec_tokens + 1]` (the sampler emits fewer columns in
  rounds with fewer or no scheduled drafts), and this step's draft token ids
  `[num_reqs, num_spec_tokens]`. List-form drafts (ngram) travel as zeros; the
  scheduler schedules no GPU-resident spec slots from those, so they are never
  read.
- A non-last rank derives the next token ids and the accepted counts from the
  matrix with `_count_contiguous_spec_tokens`, hands them to
  `_copy_valid_sampled_token_count`, and runs
  `_update_states_after_model_execute` on the `scheduler_output` it stashed
  when it returned its intermediate tensors. A missing stash raises instead of
  silently skipping the hybrid-state update.
- Both payloads go over the gloo `cpu_group`. An NCCL broadcast on the
  `device_group` shares the communicator with the pipeline's send/recv. On a
  five-stage pipeline the two interleaved and the first request hung, ranks
  0-2 in the broadcast and ranks 3-4 in `irecv`. A CPU rendezvous has no
  stream ordering to violate, and the payloads are a few dozen int32. The
  non-speculative `[num_reqs, 1]` path is unchanged and stays on the
  `device_group`.
- The sender now asserts that its row count equals `input_batch.num_reqs`, the
  number the receiver sizes its buffer from, so a divergence raises instead of
  hanging every rank in an unmatched collective.

The second commit lets the DSpark drafter load its own embedding table from
`embed.weight`. With PP=1 the proposer replaces it by the shared target
embedding afterwards. Under pipeline parallelism the target embedding lives on
the first stage and the drafter on the last, so the drafter ran on an
uninitialized table; a checkpoint without `embed.weight` now fails loudly under
PP.

Not covered: #539 stops earlier, in `custom_all_reduce.cuh` during the memory
profile run, which this does not touch. Qwen3.5-family MTP under PP also needs
#636, and #574 trims the optimistic tokens on every rank; both are independent
of this change and merge cleanly with it.

## Test Plan

```
pytest tests/v1/worker/test_gpu_model_runner_pp_spec.py \
       tests/v1/worker/test_gpu_model_runner.py \
       tests/v1/spec_decode/test_dspark.py -q
pre-commit run --files <4 files>
pre-commit run mypy-3.10 --hook-stage manual --files <3 files>
```

The new test runs two CPU processes over gloo as the last and a non-last rank
of a PP=2 deployment: a speculative round with a narrower sampler output, a
round without a stashed `scheduler_output` (must raise after both broadcasts
were consumed, otherwise the last rank would hang on the next collective),
list-form drafts, and the plain `[num_reqs, 1]` path.

Server run: Qwen3.8-27B-NVFP4 with MTP (`num_speculative_tokens=3`), pipeline
parallel over two Tesla V100-PCIE-32GB, `--enforce-eager`, async scheduling on
(the default), this branch merged with #574 and #636.

## Test Result

Tesla V100-PCIE-32GB, `CUDA_DEVICE_ORDER=PCI_BUS_ID`: `53 passed, 2 skipped`.
The new test against the unmodified runner of `main` (b711d530): `1 failed`.
pre-commit: all hooks passed; mypy-3.10 manual stage passed.

Server run with this change: four greedy fact prompts correct, a 320-token
greedy generation at 39.9 tok/s, every greedy prompt repeated token for token,
mean acceptance length 2.9 to 3.2 of 4, and 90 short requests, three in
flight, greedy and temperature 1.0 mixed, without an error. The same stack
with this change reverted: the first request hangs with the RuntimeError
quoted above on `Worker_PP0`.

A fork of this repository has carried the same transport since early
September on a five-stage pipeline (two RTX 8000, three V100) with
DeepSeek-V4-Flash and DSpark, `num_speculative_tokens=5`; that is where the
NCCL hang was seen and the gloo path came from.

## Not a duplicate

```
gh pr list -R 1CatAI/1Cat-vLLM --state open --search "pipeline parallel speculative"
gh pr list -R 1CatAI/1Cat-vLLM --state open --search "PP spec decode"
gh pr list -R 1CatAI/1Cat-vLLM --state open --search "draft token"
gh pr list -R 1CatAI/1Cat-vLLM --state open --search "async scheduling pipeline"
gh issue list -R 1CatAI/1Cat-vLLM --state open --search "pipeline parallel"
gh issue list -R 1CatAI/1Cat-vLLM --state open --search "draft token tensor"
```

The hits are my own #574, #636 and #639 (different defects, see above) and
#625, which rejects out-of-range sampled ids in the async output path and does
not move data between ranks.

## AI assistance

AI assistance (Claude) was used to port the change from the fork, to write the
test and to draft this text. I have read every changed line and ran the tests
and the server runs above on my own hardware.

---

## Offene Punkte für Peuqui vor dem Absenden

- Diff lesen (klein: 120 Zeilen Runner, 19 Zeilen dspark.py, Test). Erst danach
  stimmt der Satz „I have read every changed line".
- Entscheidung: die Negativkontrolle zitiert eine RuntimeError-Meldung aus 1Cats
  eigenem Code — gut als Beleg, so belassen.
- Danach: `git -C vllm-research/1Cat-vLLM-pr-ppspec push fork
  pp-spec-state-transport-pr`, PR gegen `1CatAI/1Cat-vLLM:main`, Body wie oben
  plus die Checkliste aus der PR-Vorlage (wie bei #657–#659).
- Optional danach in #539 kommentieren: nur, dass dort ein ANDERER Fehler vorliegt
  (Custom-All-Reduce) und `--disable-custom-all-reduce` ein Versuch wäre — ungeprüft,
  deshalb vorsichtig formulieren oder lassen.
