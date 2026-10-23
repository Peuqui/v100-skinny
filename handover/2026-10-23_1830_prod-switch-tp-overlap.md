# Übergabe 23.10.2026, ~18:30 — Produktion auf a1ac0c89d umgestellt; Plan „alle Punkte“; Start TP-Prefill-Überlappung (27B)

Ausführlicher Stand: `v100-skinny/STAND.md` (Nachträge 23.10. 14:30 / 17:30 / 18:05 oben). Gedächtnis:
`project_dsv4_stage_levers_2026-10-23.md`, `project_dsv4_decode_profile_2026-10-23.md`, `project_tp_prefill_allreduce_overlap.md`
(Abschnitt „Profil 23.10. abends“), `reference_dsv4_mini_checkpoint_tests.md`, `reference_bench_cublas_worker_matmul_policy.md`.

## Ziel (Peuqui 23.10. abends)
„Rausholen, was irgend geht, am besten in Triton“; alle Punkte des Plans angehen, auch mit hohem Aufwand:
1. 27B: TP2-Prefill-Allreduce überlappen (läuft gerade an, s. u.)
2. DSv4-Decode-Hebel (Profil liegt vor)
3. Neuer MoE-Kern in Triton (Gewichte einmal je Experte und Chunk lesen; RTX zuerst, dann V100)
4. S4-Speicher: Drafter-Vokabular-Einbettung (~1 GiB) in Host-Speicher, dann fp16-Prefill-Kopie der QPN8-Gewichte auf Turing
5. GLM-5.3-Flash produktionsreif
6. Upstream-PRs an 1Cat (del logits, PP-Send-Überlappung, Testreparaturen; Triton-Arbeit gesammelt später)

## Stand
- **Produktion seit 23.10. 17:30 = a1ac0c89d** (Zweig `triton-sm7x-routes`), Tag `verified-2026-10-23-routes` (gepusht).
  Worktree `1Cat-vLLM-work` prod-fork-main (lokal), 12 native Libs aus `1Cat-vLLM-fork-main` kopiert (alte:
  `~/.cache/prod-switch-2026-09-29/so-vor-2026-10-23-routes/`), Triton-Kopie `~/.cache/triton-sm7x-prod-2026-10-23`,
  llama-swap DSv4 (Haupt + Coding-K7) + 27B: PYTHONPATH=Triton-Kopie, `--kernel-config '{"triton_sm7x_tensor_cores":true}'`,
  Caches `torchinductor-/triton-cache-routes-2026-10-23`, `vllm-routes1024-2026-10-23`; DSv4 1024er-Chunks, Pool 3500,
  `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64`. Config-Sicherung `config.yaml.vor-2026-10-23-routes`, Switch-Skript
  `~/.cache/bench-scripts/prod_switch_2026-10-23.sh`. Rücksprung: `verified-2026-10-21-moegrid` + Libs + Config.
- **Schlussreihe** `~/.cache/bench-scripts/ab_final_2026-10-23.log`: DSv4-Prefill 35,7k 12,5 → 7,0 s, 178k 79,2 → 41,1 s,
  287k 144,7 → 73,6 s; Decode-Schritt −2…−3 %; 27B 184k 358,6 → 269,1 s, 250k 565,9 → 402,4 s, Decode gleich; 8/8.
  Kontrolle `prodcheck_2026-10-23_routes.log`: wortgleich zum Fork-Arm. `greedy_refs.sh` REF_DS ersetzt.
- **Heute committet im Zweig** (alle gepusht): 00d06e31c/1f9de3af9 Triton-MoE Lastausgleich + Blockfindung im Kern,
  d77726c11 Testreparatur, cb8272ee1 Indexer `del logits`, 993f69f5d Testfall, c80e83f8a PP-Send-Überlappung,
  a1ac0c89d Sparse-Prefill auf Turing ein Durchgang je Chunk.
- **Verworfen/zurückgestellt** (Details im Gedächtnis): RTX-Triton-MoE (10 % langsamer), Handkern-Blockliste/RB8/L2/Band,
  FP8-QPN8-Triton-GEMM (Prototyp `triton-sm7x/qpn8_gemm_triton_draft.py`), fp16-Prefill-Kopie Turing (Diff
  `triton-sm7x/qpn8_turing_fp16_copy_zurueckgestellt.diff`; Mini −2…−3 %, braucht ~195 MiB je RTX-Schicht, S4 zu voll),
  Indexer-Kachel 1020 (nur −1,4 %).

## Aktuelle Arbeit: TP-Prefill-Überlappung (Zweig `tp-prefill-overlap` in `1Cat-vLLM-fork-main`, ab a1ac0c89d, noch leer)
- 27B-Profil 23.10. (`~/.cache/bench-scripts/nsys/q27-routes-1023/`, Log `profile_q27_routes_2026-10-23.log`): 150k-Prefill
  208 s, NCCL-Allreduce 96,4 s (46 %), Turing-Prefill-Attention 46,7 s, cuBLAS ~43 s, NVFP4-Entpacken 10 s. Ziel ~125–130 s.
- Entwurf `v100-skinny/docs/entwurf-tp-prefill-allreduce-overlap-2026-10-17.md`: zwei verschränkte Mikro-Batches im V2-Runner
  (`vllm/v1/worker/gpu/model_runner.py` execute_model: `input_batch` → `model_state.prepare_attn` → `set_forward_context` →
  `self.model(**model_inputs)`), Threads/Events aus `vllm/v1/worker/ubatching.py` (UBatchContext, make_ubatch_contexts),
  Vorlage V1 `vllm/v1/worker/gpu_ubatch_wrapper.py` (`_run_ubatches`, `_make_ubatch_metadata`), Split-Hilfen
  `vllm/v1/worker/ubatch_utils.py` (`maybe_create_ubatch_slices`, `_make_metadata_with_slice`/`split_attn_metadata`),
  Umschaltpunkt im TP-Allreduce (`vllm/distributed/parallel_state.py` all_reduce / Custom-Op `vllm.all_reduce`):
  compute_done aufzeichnen → Comm-Stream → Allreduce → yield → beim Fortsetzen auf comm_done warten. Schalter als
  KernelConfig-Feld (Arbeitstitel `tp_prefill_overlap` enabled/min_tokens).
- **Erkenntnisse heute zur GDN-Frage (Align-Modus, 27B hat Präfix-Cache + MTP):** Scheduler legt Chunk-Enden auf das
  Mamba-Zustandsraster (`scheduler.py` `_mamba_block_aligned_split`, 816er-Zustandsblock laut Kommentar, Log „attention
  block size 800“); GDN-Prefill nutzt je Sequenz EINEN Zustandsplatz zum Lesen und Schreiben
  (`gdn_attn.py` GDNAttentionMetadata.non_spec_state_indices_tensor), Align-Vorkopie (`vllm/v1/worker/gpu/mamba_align.py`)
  füllt ihn vor dem Schritt → Hälfte A schreibt ihren Endzustand in denselben Platz, Hälfte B liest/schreibt dort weiter.
  B braucht `has_initial_state=True` und eigene Chunk-/Conv-Metadaten (chunk_indices/offsets, conv1d nums_dict). Prüfen:
  GDN-/conv-Kerne lesen Anfangszustand vollständig vor dem Schreiben (in-place-fest). Full Attention: B wie Folge-Chunk.
- Nächste Schritte: (1) `_make_metadata_with_slice`/`split_attn_metadata` lesen (teilt es EINE Anfrage in der Mitte?),
  (2) im V2-Runner reine Prefill-Batches ≥ min_tokens in zwei Hälften teilen (Teilungspunkt mittig; auf Zustandsraster
  nicht nötig, da gleicher Zustandsplatz), je Hälfte Metadaten über die Builder, zwei Threads, Ausgaben zusammenfügen,
  MTP-Kopf ungeteilt danach; (3) Allreduce-Umschaltpunkt; (4) Mini-Test? — 27B hat kein Mini; Test über Kerntests + 27B-Boot,
  Abnahme laut Entwurf: Greedy/Qualität/lange Prompts, Zahlentest ≤ Abweichung 2048 vs 1024 (Wortgleichheit keine Pflicht,
  Peuqui 17.10.); (5) danach Allreduce über Kopier-Engines (gemessen 2,75 statt 2,09 GB/s).

## Danach (Reihenfolge)
- **DSv4-Decode** (`project_dsv4_decode_profile_2026-10-23.md`): Schritt = Summe der 5 Stufen. Hebel 1: V100-wo_a läuft je
  Schicht als 8 TurboMind-`gemm_kernel` à 25,8 µs (~206 µs, Gewichte nur ~36 µs) — gruppierter Decode-Pfad existiert nur für
  M = 1 (`qpn/fp8.py` `sm70_fp8_bmm_grouped_decode`), DSpark prüft 6 Token → gruppiert für M ≤ 8 (Triton-Kandidat),
  ≈ −4 ms/Schritt (~−5 %). Hebel 2: lm_head S4 2 × 1,77 ms (Bandbreite) → nur reduziertes Entwurfsvokabular (PR #1218-Backport).
- **MoE-Triton-Neubau**: Handkern RTX 9,1 ms je Schicht/Chunk (1020 Token Zipf 0,8) gegen 4,7 ms Bandbreiten-Untergrenze;
  Kosten folgen den Durchgängen (Gewichtskachel je 32 Zeilen neu aus DRAM). Bench `triton-sm7x/bench_moe_skew.py`
  (+ `moe_bench_data.py`, `sweep_moe_tiles.py`), Decode-Check `bench_moe_decode_order.py`.
- **S4-Speicher**: Drafter-Einbettung (`vllm/models/deepseek_v4/nvidia/dspark.py` embed_tokens, „loaded separately“) in
  Host-Speicher (UVA, PLE-Mechanik), Kosten nur Decode ~48 KB/Schritt (Prefill nutzt „context-only“-Pfad ohne Einbettung,
  `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py`); dann fp16-Kopie (Diff s. o.) auf beiden RTX.
- GLM-5.3, Upstream-PRs.

## Werkzeuge
`dsv4-mini/run_mini_ab.sh` (ENTRIES/MEASURE/SIZES/MEMSNAP_GPU, `eintrag@variante` mit SWAP_PATH/SWAP_NAME als Listen),
`dsv4-mini/variants/` (gpu_worker.diag.py = SIGUSR1-Speichermitschnitt), `nsys/stage_ops.py <sqlite> <devs>` (inkl.
NCCL-Überlappung), `ab_table.py <log> <präfix->`, `profile_stages_nsys_run_2026-10-23.sh <eintrag> <tag> [tokens]`
(Prefill + Decode lang/kurz), `run_ab_final_2026-10-23.sh` + `start_ab_final_2026-10-23.sh` (Schlussreihe),
`mhc/sweep_sparse_tc.py` (PATTERN=shared!), `triton-sm7x/sweep_idx_tiles.py` (NUM_Q/HEADS), `watch_job.sh`.
Regeln von heute: Code-Baum nicht ändern, während ein Lauf daraus bootet; cuBLAS-Grundlinien mit
`set_high_precision_cuda_matmul_defaults()`; Sparse-Sweeps mit realistischem Indexmuster.

## Peers
Mini:AIfred-Intelligence weiß von der Umstellung (Einträge geändert), braucht keine GPUs; AIfred-Dienst aus. Vor Messfenstern
nur aktive Sitzungen fragen.
