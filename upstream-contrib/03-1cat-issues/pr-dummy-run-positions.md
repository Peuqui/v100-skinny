# PR-Entwurf C: Dummy-Positionen bleiben innerhalb von max_model_len

Fork: `30251e6e` (Branch `qwen4exp-ple-tier-cascade`).
PR-Branch: `dummy-run-positions-in-range` auf `origin/main` (949728e8),
Commit `653f39d9`, Worktree `1Cat-vLLM-pr-dummypos`. Gepusht 21.09.2026.
Die betroffene Stelle in `_dummy_run` ist auf main wortgleich mit dem Fork vor
dem Fix (per diff geprüft); die Tests liefen im Fork, weil GPU-Tests die
kompilierten Module brauchen, die nur dort liegen.

Nur 1Cat: Upstream vLLM schreibt beim Profilieren keine aufsteigenden
Positionen (Puffer bleibt null), ist also nicht betroffen.

ERÖFFNET 21.09.2026 als https://github.com/1CatAI/1Cat-vLLM/pull/670 (Peuquis Go „Push der PRs“),
Branch gepusht nach fork/dummy-run-positions-in-range.

## Duplikatsprüfung (21.09.2026, AGENTS.md)

- `gh pr list --state open --search` mit "dummy run positions",
  "profile_run positions", "max_model_len positions dummy",
  "index out of bounds profile", "opt-125m": keine passenden Treffer (#590
  Dependabot, #600 Geräte-Index).

---

Titel: [Bugfix][Worker] Keep dummy-run positions inside max_model_len

## Purpose

`_dummy_run` fills the positions buffer with `arange(num_tokens_padded)`. A
dummy batch can hold more tokens than a single sequence may
(`max_num_batched_tokens > max_model_len`), so those positions run past
`max_model_len`, and for a model with a fixed learned position table past the
end of that table.

`facebook/opt-125m` (learned positions, 2050 entries) with `max_model_len=1024`
and chunked prefill disabled hits a device-side assert in `profile_run`
(`index out of bounds: 0 <= tmp14 < 2050` in the compiled position lookup) and
the engine never starts. This is what
`test_deserialized_encrypted_vllm_model_has_same_outputs` and
`test_deserialized_hf_model_has_same_outputs` in
`tests/model_executor/model_loader/tensorizer_loader/test_tensorizer.py` run
into; it happens on V100 and on RTX 8000 alike.

The fix keeps the ascending positions but wraps them modulo `max_model_len`,
which every position table is validated to cover at startup. Models whose
dummy batch fits into `max_model_len` see no change.

The second change is in the test only: `test_serialize_and_serve_entrypoints`
started `vllm serve` on the default port 8000 and failed with "Address already
in use" wherever that port is taken; it now asks for a free port with
`get_open_port()`, like other tests that spawn a server.

AI assistance was used for this change. Every line was reviewed and the tests
below were run by the submitter.

## Test Plan

```bash
PATH=<venv>/bin:$PATH pytest \
  tests/model_executor/model_loader/tensorizer_loader/test_tensorizer.py \
  -k "not with_tp and not tp_path"
```

## Test Result

Before: both `test_deserialized_*_has_same_outputs` tests fail in `profile_run`
with the device-side assert above; `test_serialize_and_serve_entrypoints`
fails with `IncompleteReadError` (the server exits on the taken port).

After: 13 passed, 2 deselected (11:29 min on 3x V100 + 2x RTX 8000). The run
used a fork carrying this change, because the GPU tests need the compiled
extensions built there; the `_dummy_run` lines this touches are identical to
`main`. The two tensor-parallel cases
(`test_tensorizer_with_tp_path_without_template`,
`test_deserialized_encrypted_vllm_model_with_tp_has_same_outputs`) hang right
after NCCL initialisation on this machine before and after the change; they
are unrelated and excluded above.

## Not a duplicate

Open PRs searched for "dummy run positions", "profile_run positions",
"max_model_len positions dummy", "index out of bounds profile" and
"opt-125m"; none addresses the dummy-run positions.
