# PR-Entwurf D (vLLM upstream): Drafter liest den Checkpoint nicht doppelt

Gegenstück zu Entwurf B (1Cat). PR-Branch `skip-checkpoint-weights-before-read`
auf `upstream/main` (382970ee, 21.09.2026 abends), Commit `84f4e51c`,
Worktree `vllm-upstream-pr-drafterskip`, Remote `origin` = Peuqui/vllm.
Gepusht 21.09.2026.

Unterschiede zur 1Cat-Fassung: upstream hat keinen Index-Filter, `_keep_weight`
bündelt dort EP-Filter und Modell-Regel. `_remap_dspark_name` ist upstream
eine Instanzmethode (liest `self.model.confidence_head`); die Regel greift
trotzdem, weil `_duplicate_context_wkv_weights` seine zusätzlichen Namen nur
aus `mtp.N.attn.wkv.*` erzeugt, und die bleiben erhalten.

EHRLICH ZU BENENNEN: gemessen wurde auf dem 1Cat-basierten Fork (V100/Turing,
upstream läuft dort nicht). Der DSpark-Test (`test_dspark_mla.py`) lief lokal
NICHT — upstream verlangt humming-kernels 0.1.15, unsere venv hat 1Cats 0.1.2
(Peuqui 21.09.: kein Update); er ist `cpu_test` und läuft in der CI.

ERÖFFNET 21.09.2026 als https://github.com/vllm-project/vllm/pull/58016 (Peuquis Go „Push der PRs“),
Branch gepusht nach origin/skip-checkpoint-weights-before-read (Peuqui/vllm).

## Duplikatsprüfung (21.09.2026, AGENTS.md)

- `gh pr list --repo vllm-project/vllm --state open --search` mit "skip
  checkpoint weight", "drafter load weights", "dspark load", "safetensors skip
  before reading", "load only mtp weights": kein Treffer zum Thema.
- Nahe, aber anders: #48023 (Drafter erbt `model_weights` bei RunAI/S3 —
  bestätigt, dass mtp/dspark vom Target-Pfad gebaut werden), #40068 (Lesereihe
  der Dateien je Rang versetzen — gleiche zwei Funktionen, anderer Fix,
  möglicher Merge-Konflikt), #48891 (EP-Filter im Multithread-Lader),
  #57733 (Post-Load-Finalisierung).

---

Titel: [Perf][Model Loader] Skip checkpoint tensors a model does not load before reading them

## Purpose

Drafters that ship inside their target's checkpoint are built from the target
path (`SpeculativeConfig`: `self.model = self.target_model_config.model` for
`dspark`, and likewise for MTP). The default loader then iterates every shard
of the target a second time, only for the drafter's `load_weights` to drop
everything that is not `mtp.*`.

When the checkpoint is larger than the host's page cache, that second pass is
a second full read from disk. On DeepSeek-V4-Flash with DSpark (165 GB, 46
shards, pipeline parallel over five GPUs) it runs on the last stage after all
stages have loaded their own weights, and took 255 s of a 10:25 min boot.

The EP weight filter already skips tensors by name before `get_tensor`. This
generalises the same idea to the model: a model can offer
`skip_checkpoint_weight(name) -> bool`, `DefaultModelLoader` reads it the way
it already reads `allow_patterns_overrides` and passes it through a new
`Source.skip_weight` field to `safetensors_weights_iterator`, which drops
accepted names before reading them. EP filter and model rule go through one
helper, `_keep_weight`, instead of being repeated per load strategy. Models
that do not offer the method are unaffected.

The DeepSeek-V4 DSpark drafter implements the rule with the mapping it already
uses in `load_weights`: `_remap_dspark_name(name) is None`. Other drafters that
load from their target's checkpoint (MTP variants, the other DSpark models)
can opt in the same way; this PR keeps to the one that was measured.

The multithreaded iterator is left as is: it reads whole files with
`load_file` before any filter can apply, so a name rule there would not save
I/O (#48891 addresses the same gap for the EP filter).

AI assistance was used for this change. Every line was reviewed and the tests
below were run by the submitter.

## Test Plan

```bash
pytest tests/model_executor/model_loader/test_ep_weight_filter.py
pre-commit run mypy-3.12 --hook-stage manual --files <changed files>
```

`tests/models/test_dspark_mla.py::test_deepseek_v4_dspark_skips_every_checkpoint_weight_it_does_not_load`
is added as a `cpu_test`; it could not be run locally (see Test Result).

End-to-end measurement on 3x V100 + 2x RTX 8000 with the same loader change on
a V100/Turing fork of vLLM (1Cat-vLLM), DeepSeek-V4-Flash-NVFP4 with DSpark,
PP5, cold boot.

## Test Result

39 passed, including the two new tests in the existing EP filter suite
(`test_model_skip_rule_drops_tensors_before_reading`: skipped names never reach
`get_tensor`; `test_loader_applies_the_model_skip_rule`). Both fail on `main`
without the change. mypy-3.12 passes. The new DSpark test did not run locally:
this machine's environment pins an older `humming-kernels` than main requires,
so the DeepSeek-V4 DSpark module does not import there; CI covers it.

| Boot phase (V100/Turing fork, same loader change) | before | after |
|---|---|---|
| target weights, all stages | 276 s | 286 s |
| drafter on the last stage | 255 s | 17 s |
| boot to `Application startup complete` | 10:25 min | 6:03 min |

Disk read during the boot after the change: 177 GiB (checkpoint once plus the
drafter's file). DSpark mean acceptance length afterwards 2.6–3.5.

## Not a duplicate

Open PRs searched for "skip checkpoint weight", "drafter load weights",
"dspark load", "safetensors skip before reading" and "load only mtp weights".
Related but different: #48023 (drafter inherits `model_weights` for object
storage), #40068 (staggers file order per rank in the same two functions),
#48891 (EP filter in the multithreaded loader).
