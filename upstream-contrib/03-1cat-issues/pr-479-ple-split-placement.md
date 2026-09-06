# PR-Entwurf: 1CatAI/1Cat-vLLM — Qwen4Exp PLE: Aufteilung Device/Host (#479, dritte Scheibe)

Stand 2026-09-06 mittags. Branch `qwen4exp-ple-split-placement`, Basis origin/main 755baae, zwei Commits
(493a958 Split-Platzierung, cfd8a5f Capability-Entscheid auf dem Worker-Device). Gepusht; PR #528 eroeffnet 2026-09-06: https://github.com/1CatAI/1Cat-vLLM/pull/528
Diff: `pr-479-ple-split-placement.diff` (daneben; Stand vor dem Rebase auf main 95205a2 am 06.09. ~10:30 — Konflikt common/ple.py Import-Kopf, Inhalt unveraendert).

Geänderte Dateien (4):
- vllm/envs.py — `VLLM_QWEN4EXP_PLE_HOST_GIB` (float|None, "auto"), `VLLM_QWEN4EXP_PLE_VRAM_RESERVE_GIB`
- vllm/models/qwen4_exp/common/ple.py — `PLEPlacement`, `plan_ple_placement`, `copy_ple_embedding_shard_split_`,
  `kv_cache_bytes_for_max_model_len`, `auto_ple_host_budget_bytes`, `available_host_bytes`
- vllm/models/qwen4_exp/nvidia/ple_layer.py — `Qwen4ExpPinnedHostEmbedding` mit Platzhalter, lazy
  `materialize_tables()`, `load_shard()`, Dual-Gather; Loader-Routing; `_should_use_pinned_host_ple` worker-lokal, < 8
- tests/models/qwen4_exp/test_ple.py — 3 Lebenszyklus-Tests (CUDA), Split-Gather auf SM70 ×3 (exakt V100),
  4 CPU-Helfertests, Capability-Entscheid ×4; die zwei alten Pinned-Host-Tests ersetzt (neuer Vertrag)

Abweichungen zum Fork: Env über vllm.envs (lazy) statt os.getenv je Aufruf; kein `_vllm_config_ref` im
__init__ (Config erst beim Auto-Budget); `get_accelerator_weight` materialisiert selbst (Dummy-Weights-Pfad);
`split_ple_embedding_lookup` (Torch-Referenz) nicht mitgenommen — im Fork ungenutzt.

Upstream-Kontext (Duplikats-Check 2026-09-06, gh pr list --state all --search): #345/#374 (pinned host,
ganze Shard im Host), #471 (plattengestützte Tabelle, mmap) — beide gemerged, keine Aufteilung Device/Host;
"PLE placement"/"PLE_HOST" → nichts. Unser Ansatz ergänzt: Tabelle bleibt im VRAM, nur der Überhang
gepinnt; #471 bleibt der Weg ohne RAM und ohne VRAM.

Rig-Beleg: FLASH-NEXT-OPERATING-POINT.md (28.08., Fork auf 1.3.0): TP2/PP2 auf 0,2,1,4, PLE_HOST_GIB=6,
51,9 tok/s MTP k=4, Kohärenz 8/8; dieselbe Platzierung liegt im 1.5.0-Overlay (fork_patches_150). Auf
main selbst NICHT bootbar hier (kein main-Build für sm70/sm75).

Tests (Wheel-venv venv-150, PYTHONPATH=Checkout, V100 = PCI 4):
    tests/models/qwen4_exp/test_ple.py → 34 passed (davon 3 SM70-exakt)
    Negativ-Beweis: alte ple_layer.py mit den neuen Tests → 6 failed (die 6 Pinned-Host-Tests), 24 passed
    tests/models/qwen4_exp/ + tests/v1/worker/test_ple_offload_worker.py + tests/v1/spec_decode/test_qwen4_exp.py
      → siehe Nachtrag
    pre-commit alle Hooks (inkl. torch.cuda-Hook, mypy-local) Passed; mypy-3.10 Passed

Bekannte Grenzen (im PR nennen): Auto-Budget schätzt Aktivierungsspitze + Graph-Pool über eine Reserve
(8 % / max 4 GiB, per Env); der sichere Weg ist das explizite Host-Budget. Gather liest immer beide Hälften.

Vorgeschlagener PR-Titel:

    [Model][Qwen4Exp] Split the pinned-host PLE table between device and host (#479)

PR-Text (Entwurf):

    ## Purpose

    Third slice of the PP enablement offered in #479, and the placement we
    run Qwen3.8 Flash Next with. The pinned-host PLE path (#345/#374) keeps
    the whole FP8 n-gram shard in pinned host memory: slow where the card
    has room beside its layers, and not bootable on a box with little host
    RAM (51 GB table, 25 GB per TP2 rank). #471 answers that with a
    disk-backed table. This PR adds the option in between: rows stay in
    device memory as long as the stage's weights, the KV cache of the
    requested max_model_len and a reserve still fit; only the remainder is
    pinned on the host. Budget derived from the real headroom on the first
    checkpoint shard, or fixed per rank with VLLM_QWEN4EXP_PLE_HOST_GIB.
    Second commit: the pinned-host decision asks for the worker's own
    device capability (device 0 of the visible list is the wrong card on
    a heterogeneous pipeline) and covers every pre-Ampere card.

    ## Test Plan
    (Kommandos + Ergebnisse von oben, englisch)

    ## Duplicate check / AI assistance
    gh pr list --state all --search "PLE host" / "pinned host PLE" / "PLE placement": #345, #374, #471
    (merged, whole-shard host or disk; no device/host split). Written with AI
    assistance (Claude); every line reviewed and tested by me on 2x RTX 8000 + 3x V100.

Nachtrag (Regression, V100 PCI 4): tests/models/qwen4_exp/ (ohne test_qsa_amd.py: nvidia- und amd-Tree
registrieren im selben Prozess denselben Custom-Op, vorbestehend) + tests/v1/worker/test_ple_offload_worker.py
+ tests/v1/spec_decode/test_qwen4_exp.py → 234 passed, 3 failed (test_qsa_reference.py: side/circular/
compressed metadata). Die drei sind auf unverändertem 755baae identisch rot (Gegenprobe mit den drei
Dateien auf 755baae: 3 failed, 41 passed) — nicht von dieser Änderung.
