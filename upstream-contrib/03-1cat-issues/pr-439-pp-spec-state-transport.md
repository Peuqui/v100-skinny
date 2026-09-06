# Test-Branch / PR-Entwurf: 1CatAI/1Cat-vLLM — PP + Spec Decode Transport (#439, Folge zu #511)

Stand 2026-09-06 vormittags. Branch `pp-spec-state-transport`, Basis origin/main 755baae,
darauf der #511-Commit (cherry-pick 89aa181 → 898a770) und zwei neue Commits.
Zweck: Test-Branch für DSYZayn (DeepSeek-V4-Flash, DSpark, TP4×PP2) und spätere PR.
Status: gepusht 2026-09-06 nach fork Peuqui/1Cat-vLLM (Freigabe Peuqui); PR erst nach DSYZayn-Test / Transport-Antwort.
Diff: `pr-439-pp-spec-state-transport.diff` (daneben; gegen 898a770, also ohne #511).

Geänderte Dateien (8):
- vllm/v1/worker/gpu_model_runner.py — Stash des scheduler_output auf Nicht-Last-Ranks,
  `_pad_spec_sampled_token_ids`, `_pp_broadcast_prev_sampled_token_ids` (Spec-Zweig),
  NEU `_pp_broadcast_draft_token_ids`, NEU `_pp_receive_spec_decode_state`, Dispatch in
  `_pp_receive_prev_sampled_token_ids_to_input_batch`, Backup-Token-Puffer im __init__
- vllm/v1/spec_decode/utils.py — NEU `fill_backup_next_token_ids`, `prepare_next_token_ids_padded`
  (SSOT um den bestehenden Kernel `eagle_prepare_next_token_padded_kernel`)
- vllm/v1/spec_decode/llm_base_proposer.py — `prepare_next_token_ids_padded` delegiert (27-/7+)
- vllm/v1/spec_decode/extract_hidden_states.py — Backup-Fill über den Helfer
- vllm/models/deepseek_v4/nvidia/dspark.py — `embed.weight` → Drafter-Tabelle; lauter Fehler unter PP
- tests/v1/worker/test_gpu_model_runner_pp_spec.py — NEU, 2-Prozess-gloo-Round-Trip (CPU)
- tests/v1/spec_decode/test_prepare_next_token_ids_padded.py — NEU, Kernel gegen Torch-Referenz (GPU)
- tests/v1/spec_decode/test_dspark.py — Mapping-Assertion `embed.weight`

Abweichungen zum Fork (fork_patches_150):
- Transport über `pp.device_group` (NCCL) wie der bestehende Broadcast, NICHT gloo. Unser
  gloo-Umweg ist ein Befund unserer 5-Stufen-USB4-Pipeline (#439-Kommentar 04.09.); für PP2
  lief NCCL bei uns (24.08., V100-Paar k=7). Wird im PR-Text als Alternative genannt.
- Ableitung der Next-Token-IDs auf der Nicht-Last-Stufe mit DEMSELBEN Kernel wie der Drafter
  (Discard-Maske, Backup-Token), nicht mit der vereinfachten Fork-Ableitung.
- Kein sm75-Sonderzweig (Fork-eigene GDN-Layer).
- Kein `SupportsPP` für den DSpark-Drafter: main setzt `draft_parallel_config.pipeline_parallel_size`
  für dspark bereits auf 1 (tests/v1/spec_decode/test_dspark.py::test_dspark_draft_is_local_to_last_pipeline_stage).

Bekannte Grenzen (im PR-Text nennen):
- Nur unter async scheduling (Default für Eagle/MTP/DFlash/DSpark). Im Sync-Modus bleibt PP+Spec
  für hybride Modelle ohne State-Update (unverändert zum Stand vor dem PR).
- TP4×PP2 nicht von uns gefahren; unsere Läufe: TP1×PP2 und TP2×PP2 (Qwen3.8-27B, MTP), TP1×PP5
  heterogen (DSV4-Flash, DSpark) — mit gloo-Transport auf 1.5.0.

Duplikats-Check 2026-09-06 (gh pr list --state open --search): "439 in:body" → nur #511 (unser);
"pipeline parallel speculative" → #511, #517 (DFlash2-Präzision, andere Baustelle); "dspark embed" → leer;
"non-last rank" → #511, #512 (unsere), #519/#521/#522 (AWQ-Kernel, andere Baustelle).

Tests (Wheel-venv /home/mp/vllm/venv-150, PYTHONPATH=Checkout, V100 = PCI-Index 4):
    PYTHONPATH=$PWD python -m pytest tests/v1/worker/test_gpu_model_runner_pp_spec.py tests/v1/spec_decode/test_dspark.py
      → 14 passed
    Negativ-Beweis: Runner-Datei per git stash auf den Stand ohne Transport → Round-Trip-Test rot
      ("PP+async expects sampled_token_ids to have shape [num_reqs, 1]", exitcodes [1, 1])
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 PYTHONPATH=$PWD python -m pytest
      tests/v1/spec_decode/test_prepare_next_token_ids_padded.py tests/v1/worker/test_gpu_model_runner.py -k "matches_reference or non_last_pp_rank"
      → 7 passed (Kernel gegen Referenz ×3, #511-Init auf Nicht-Last-Rank ×3, Round-Trip)
    tests/v1/spec_decode/test_backup_token_async_spec.py + test_extract_hidden_states.py + test_eagle.py -k "prepare_next_token or backup"
      → 12 passed, 1 failed: test_eagle::test_prepare_next_token_ids braucht das gesperrte HF-Repo
        meta-llama/Llama-3.1-8B-Instruct (403) — Umgebung, nicht die Änderung.
    Vollsuite der berührten Dateien: siehe unten (Nachtrag).
    pre-commit run --files <8 Dateien> → alle Hooks Passed; pre-commit run mypy-3.10 --hook-stage manual → Passed.

Commit 2 (vorgeschlagen):

    [Core][Spec Decode] Ship the speculative round state to non-last PP ranks

    Under async scheduling the last pipeline rank broadcasts the sampled
    token ids to the other ranks after every step, but only as a
    [num_reqs, 1] column: with speculative decoding the sampler returns the
    full [num_reqs, num_spec_tokens + 1] matrix and the broadcast asserts.
    The non-last ranks also never see the proposed draft ids (the scheduler
    carries placeholder slots only) nor the accepted counts that drive the
    hybrid-state update, so speculative decoding under pipeline parallelism
    stops in the first decode round once the ranks survive profiling
    (#439, #511).

    Broadcast the round state instead: the sampled matrix, padded to the
    static wire shape with the rejection sampler's -1, and this step's
    draft ids. The receiving rank derives the next-token ids and accepted
    counts with the drafter's own kernel, moved into shared helpers
    (prepare_next_token_ids_padded / fill_backup_next_token_ids in
    spec_decode/utils.py, delegated to by the eagle-family and
    extract-hidden-states proposers), and runs the hybrid-state update on
    the scheduler_output it stashed in execute_model. Both broadcasts use
    the PP device group like the existing one.

Commit 3 (vorgeschlagen):

    [Bugfix][DeepSeekV4] Load the DSpark drafter's own embedding under PP

    DSparkDeepseekV4ForCausalLM has no embedding of its own in the
    checkpoint (has_own_embed_tokens = False): with PP=1 the proposer
    shares the target's embed_tokens. Under pipeline parallelism it does
    not (_maybe_share_embeddings: "will be loaded separately"), because the
    target embedding lives on the first stage and the drafter on the last;
    but _remap_dspark_name drops every key outside mtp.*, embed.weight
    included, so the drafter's VocabParallelEmbedding kept its random
    initialisation. The boot succeeds and acceptance collapses to a few
    percent.

    Map embed.weight onto the drafter's table (with PP=1 the shared target
    embedding replaces it afterwards, as before) and fail loudly under PP
    when the checkpoint did not supply it.

Vorgeschlagener PR-Text (später, nach Test durch DSYZayn / Maintainer-Antwort zur Transportfrage):

    ## Purpose

    Follow-up to #511 for #439. With the drafter bound on every rank the
    non-last PP ranks survive profiling, but speculative decoding still
    stops in the first decode round: under async scheduling the last rank
    broadcasts a [num_reqs, 1] column of sampled ids and asserts on the
    [num_reqs, k + 1] matrix the sampler returns with speculation; the
    other ranks never receive the draft ids or the accepted counts either.

    This PR broadcasts the round state (padded sampled matrix and draft
    ids) from the last rank and lets the other ranks derive next-token ids
    and accepted counts with the drafter's own kernel, moved into a shared
    helper. The hybrid-state update runs on the receiving rank's stashed
    scheduler_output. A second commit fixes the DSpark drafter under PP,
    whose own embedding table was never loaded (acceptance collapsed to a
    few percent while the boot looked fine).

    Transport: both broadcasts use the PP device group like the existing
    sampled-id broadcast. On our five-stage USB4 rig that NCCL broadcast
    interleaves with the pipeline send/recv and deadlocks, and we run the
    same payloads over the gloo cpu_group there; for two stages the device
    group worked for us. Happy to switch if you prefer the CPU rendezvous.

    Not covered: synchronous scheduling (unchanged), TP4xPP2 (not on our
    hardware; the wire format does not depend on the TP size).

    ## Test Plan
    (Kommandos + Ergebnisse von oben, englisch)

    ## Duplicate check / AI assistance
    gh pr list --state open --search "439 in:body" / "pipeline parallel speculative" /
    "dspark embed": only #511 (ours) and unrelated kernel PRs. Written with AI
    assistance (Claude); every line reviewed and tested by me on 2x RTX 8000 + 3x V100.

Nachtrag (Vollsuite, V100 PCI-Index 4): tests/v1/worker/test_gpu_model_runner.py + test_gpu_model_runner_pp_spec.py,
tests/v1/spec_decode/test_dspark.py + test_extract_hidden_states.py + test_backup_token_async_spec.py +
test_sm70_mtp_safety.py + test_prepare_next_token_ids_padded.py + test_mtp.py → 95 passed, 2 skipped (1:20 min).
Committed lokal: Branch pp-spec-state-transport = 755baae + 898a770 (#511) + 2 Commits (siehe git log). Nicht gepusht.
