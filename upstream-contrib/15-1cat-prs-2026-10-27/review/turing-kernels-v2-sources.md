# Quellen zu body-turing-kernels.v2.md (27.10.2026)

Logs unter `~/.cache/bench-scripts/` (kurz: B/), Antworten unter `B/quality_2026-09-27/` (kurz: Q/). Textvergleich mit
`scratchpad/v2body/cmp_runs.py <Q> <Präfixe…>` (Sitzung b66b4495; vergleicht Greedy-Hashes aus `*.greedy.txt` und SHA1 der
Antworttexte aus `*-ds.json`). Branch = Worktree `1Cat-vLLM-pr-turing-kernels`, `git log --abbrev=9 origin/main..HEAD`.

## Purpose

- Neun Commits, Hashes 35caa5fc1 a1d080efb 3917b3997 c55064416 f5b53b058 2ecf9355d 8d4f2e978 b2652b399 e83d76471 — `git log --abbrev=9 --format='%h %s' origin/main..HEAD`.
- „mma.m8n8k4 … half its tensor rate“ — Commit 8d4f2e978 (1. Satz), Header mma_fragments_sm70.cuh Z. 6–8.
- „w13 … 8 instead of 16 warps; w2 already used 8“ — main: `git show origin/main:vllm/model_executor/layers/fused_moe/experts/skinny_sm70_moe.py` Z. 645 `grouped_splitk(…, 16)` (w13), Z. 664 `grouped_splitk(inter_dim, 8)` (w2); HEAD: `grouped_splitk` Präferenz (8, 16, 10) für M8N8K4.
- „tiles alone change no bit … regroups“ — Commit b2652b399: „With the same split-K the tile counts change no bit; the new split regroups the reduction.“
- GEMM-Beschreibung (bit-identical to dense reconstruction, ldmatrix, row tiles in groups, fixed-order FP32 reduction, no second layout) — Commit 35caa5fc1.
- „M decision stays inside the opaque op“ — fp8_block.py `_qpn8_native_linear` (HEAD Z. 84–140): M ≤ 8 → fp8_qpn8_gemm_sm70_out, sm75_mma und M ≤ 512 → fp8_qpn8_mma_sm75_out, sonst TurboMind-Fallback oder Prefill.
- „faster on 10 of 13 shapes“ — Commit 35caa5fc1 (1.01–1.28x auf 10 von 13); Journal STAND.md 20.10. 17:20 (KORREKTUR, `qpn8proto/graph_eval.py`).
- „Volta's routes unchanged“ — Diff fp8_block.py: nur `capability.to_int() == 75` setzt `_qpn8_sm75_mma`; `fp8_qpn8_occupancy_split_k` gibt bei 2048 Threads/SM (ctas_per_sm_at[16] = 4 > 2) die Tabelle zurück.
- „capability of the weight's own device (it was the current device)“ — `git diff origin/main..HEAD -- …/fp8_block.py`: `get_device_capability()` → `get_device_capability(codes.device.index)`.
- Split-K-Mechanik (1024 Threads/SM, N = 5120 Schwanzwelle, split 8 eine Welle; nur M ≤ 8; split_k/decode_split_k/split_k_for; M=17..32 nimmt 12 oder 16; launch_config liefert Tabelle) — Commit 3917b3997; Code sm70_turbomind.py (FP8QPN8LinearState.split_k_for, fp8_qpn8_launch_config-Docstring); Test test_qpn8_per_tensor_occupancy_split_only_up_to_m8.
- „split-K 32 … measured on DeepSeek-V4 and not adopted“ — Commit 3917b3997; Zahlen dort gegen B/ab_qpn8split_ds_2026-10-20.log (Schritt 62/65/70 beide, 250k 222,25–222,28 vs 222,16–222,37 s) und B/ab_qpn8split_accept{,2}_2026-10-20.log (8 von 8 Vergleichen niedriger, 0,1–1,5 Punkte) geprüft.
- Grid min(slots, experts), bit-identisch — Commit c55064416.
- „backend admits capability 7.0 and 7.5 only“ — skinny_sm70_moe.py `_supports_current_device` (HEAD Z. 512–523, `for cap in ((7, 0), (7, 5))`); Fragmentordnung `qpn_fragment_order` Z. 57–59 (`has_device_capability(75)`).
- Packen beim Laden, Permutation, gleiche Größe, A/B-Operand-Tausch, „decoded weights equal … bit for bit“ — Commit 8d4f2e978.
- „16 or 10 only where 8 does not divide K/16“ — `grouped_splitk`: `(k // 16) % split == 0` (HEAD Z. 78).
- „two-accumulator variant … leaves the kernel and the op signature“ — Commit b2652b399.
- e2m1-Decoder-Herkunft (TurboMind cvt_f16x8_e2m1, Apache-2.0) — mma_fragments_sm70.cuh Z. 19–20.
- „replaces the e2m1 decoder copies in nvfp4_qpn4_sm70.cu and skinny_moe_qpn_sm70.cu“ — `git show 8d4f2e978 -- …/nvfp4_qpn4_sm70.cu` (fp4x8_to_half2x4 entfernt); origin/main skinny_moe_qpn_sm70.cu Z. 68–84 (`dequant8_tm`, „derived from TurboMind's cvt_f16x8_e2m1“), in 8d4f2e978 entfernt; fp8_qpn8_mma_sm75.cu gibt es auf main nicht (neu in 35caa5fc1), seine ldmatrix/mma-Kopien entfernt 8d4f2e978.
- „helpers trap below SM75“ — mma_fragments_sm70.cuh Z. 42–64 (`__trap()` im `#else`).
- Schiefes Routing im Test — Commit e83d76471.
- SM70_FAMILY_ARCHS „7.0;7.5“, fp8_qpn8_mma_sm75.cu nur „7.5“, skinny_moe_qpn_sm70.cu in VLLM_SM70_TURBOMIND_SRCS — `git show HEAD:CMakeLists.txt` Z. 376, ~Z. 455–462, Liste ab ~Z. 388.
- sm70_884 nur [700, 750), leere sm_75-Körper, Registry lehnt auf SM75 ab — arch.h Z. 17 (`struct Sm70: Arch<700, 750>`), gemm_universal.h Z. 241–242 (`if constexpr (Kernel::Arch::is_compatible(__CUDA_ARCH__))`), tm_registry_sm70.cu Z. 22 (`is_arch_compatible`).
- Extensions mit exakter SM70-Prüfung (router, QSA indexer/device history, ring collectives, exact row reduce, H3, sparse attention) — Review-Befund 5 (Datei:Zeile-Liste: sm70_router.cu:28, qsa_indexer_sm70.cu:179, sm70_ring_collectives.cu:118, qsa_device_history_sm70.cu:878, exact_row_reduce.cu:143, h3_w8a16.cu:126, h3_noncausal_sm70.cu:334, flash-attention-v100 h3/forward.cu:120, forward_sparse.cu:89); von mir nicht einzeln nachgelesen.
- TORCH_CUDA_ARCH_LIST/Marlin, SM75-Zweige in TurboMind-Headern — Commit f5b53b058.
- Unquoted „70;75“, compute_70;75, H3 scheiterte — Commit 2ecf9355d; STAND.md 21.10. 10:35 (`build_sm75_2026-10-21b.out`).

## Not a duplicate

- 80 offene PRs — `gh pr list -R 1CatAI/1Cat-vLLM --state open --limit 200 --json number,headRefOid` (80 Zeilen); alle Heads lokal vorhanden; je PR `git merge-tree --write-tree --name-only --no-messages origin/main <head>` und `… HEAD <head>`; neue Konflikte nur #1243, #1216, #1031, #1030, #822.
- #1069, #1157: Autor kezboard233, OPEN, MERGEABLE, je 1 Datei (fp8_qpn8_sm70.cu) — `gh pr view N --json …`; Inhalt (M=9..32 chunked über M ≤ 8; nativer M=17..32-Kern; beide im `fp8_qpn8_dispatch_sm70_out`; V100-TP4-DFlash2-Decode mit mehreren 8-Token-Zeilen) — PR-Texte und `gh pr diff 1069` (`if (m > 8 && m <= 32 …)` im Dispatch); kein Konflikt (merge-tree leer).
- „only moves two decode helpers out of that file“ — `git diff origin/main..HEAD -- csrc/sm70_turbomind/ops/fp8_qpn8_sm70.cu` (fp8x8_to_half2x4_fast, qpn8_col_from_lane nach fp8_qpn8_codes_sm70.cuh).
- Blockkern ruft den Dispatch nicht — fp8_block.py `_qpn8_native_linear` (siehe oben).
- #1030/#1031/#1243: Konflikte mit main 1 / 4 / 4 Dateien, neu durch den Zweig je skinny_moe_qpn_sm70.cu + skinny_sm70_moe.py — merge-tree wie oben; #1031/#1243 tragen dieselbe Änderung an beiden Dateien (diff --stat je 35/31 Zeilen, gleiche TORCH_CHECK-Änderungen) wie #1030; #1030 nennt `nacc == 2` für split 8/16 — `git diff $(git merge-base origin/main 19f25c48ee95) 19f25c48ee95 -- …/skinny_moe_qpn_sm70.cu`; #1031 = „DeepSeek-V4.1-Flash on V100“, #1243 „stacks on #1031“ — Titel aus gh.
- #822: CONFLICTING mit main, neuer Konflikt nur tests/quantization/test_turing_qpn_kernel_selection.py — merge-tree.
- #1216: neuer Konflikt CMakeLists.txt; #1216 hängt `csrc/host_reduce.cu` an die ring-collectives-Zeilen — `git diff $(git merge-base origin/main 130ff00e6984) 130ff00e6984 -- CMakeLists.txt`.
- Suchen — `gh pr list -R 1CatAI/1Cat-vLLM --state open --search "<Begriff>"` für m16n8k8, QPN8, "QPN8 Turing", "Turing tensor", sm_75, SM75, "skinny MoE", split-K, SM70_FAMILY; QPN8-Treffer #1069 #953 #1157 #1021 #1273; #953/#1021 ohne Dateiüberschneidung (`gh api --paginate repos/1CatAI/1Cat-vLLM/pulls/N/files`).

## Test Plan

- Testdateien existieren — `git ls-files | grep …`: tests/quantization/{test_sm70_native_policy,test_block_fp8_qpn8_capabilities,test_sm70_moe_mainflow,test_sm70_fp4_mainflow}.py, tests/tools/test_sm70_path_explanations.py, tools/sm70_route_snapshot.py (`--category` moe/fp8, `--output`, `--check`, Z. 333–349); Ablauf vorher/nachher laut docs/contributing/1cat-development.md Z. 58–62.

## Path coverage / Contract

- Testnamen test_turing_decode_keeps_the_packed_gemm (test_sm70_qpn8_block_fp8.py:226), test_qpn8_split_k_follows_device_occupancy (:349), test_qpn8_per_tensor_occupancy_split_only_up_to_m8 (:360) — `git grep` auf HEAD.
- „K/16 divisible by 8“ als Bedingung der Split-Regel — fp8_qpn8_occupancy_split_k: `groups % 8 == 0`, `groups = k // QPN2_GROUP_SIZE`.
- Modellkonfigurationen (PP=5, Stufen 0 und 4 auf RTX: CUDA_VISIBLE_DEVICES=0,1,4,3,2 mit PCI_BUS_ID, GPU 0/2 = RTX 8000; DSpark 5; --max-num-batched-tokens 512; --dtype half; --kv-cache-dtype fp8; 27B TP=2 auf 0,2, MTP 3, 2048, float16, ohne kv-cache-dtype = auto; VLLM_USE_V2_MODEL_RUNNER=1; NCCL_P2P_DISABLE=1) — `~/.cache/prod-switch-2026-09-29/config.yaml.vor-ab-sm75-2026-10-21` (+ `…vor-ab-qpn8turing-ds-…`, `…vor-ab-qpn8split-…`), FP8-27B: `B/q27_fp8_test_entry_2026-10-20.yaml`.
- „changed arm = same entry + PYTHONPATH + own compile cache“ — B/run_ab_sm75_2026-10-21.sh Z. 80–105 (DROP-Liste greift nicht: keine der Variablen in den Referenzeinträgen, keine `--kernel-config`, kein `--moe-backend`), B/run_ab_q27fp8_turing_2026-10-20.sh Z. 47.
- „stock Triton 3.6.0“ — `/home/mp/vllm/venv/lib/python3.12/site-packages/triton-3.6.0.dist-info`; Run-Skripte setzen nur PYTHONPATH=<Baum>; sm7x-Triton erst ab 23.10. (STAND.md 23.10. 08:15).
- Route-Hit-Logzeile — Q/fm2-qt2-ds-fm-a-*.journal.txt und Q/fm2-s75-ds-fm-a-*.journal.txt („Block FP8 QPN8 on SM75: M=1..8 uses the packed decode GEMM, M=9..512 …“); 27B: „Selected TuringQpn8Fp8LinearKernel for ModelOptFp8LinearMethod“ in B/ab_qpn8split_2026-10-20.log.
- Fehlender Op als Ablehnungsgrund — fp8_block.py can_implement (HEAD-Diff: `required.append("fp8_qpn8_mma_sm75_out")` bei capability 75).

## Test Result: Mikrobenchmarks (nur Commit-Nachrichten, nicht nachmessbar)

- 0.16–0.60x M=16..64, 1.01–1.28x auf 10/13, 0.78–1.02x auf 3, 20 Aufrufe je Graph — Commit 35caa5fc1.
- BM128 7–12 %, BM16/32/64 ±2 %, 840 Konfigurationen — Commit a1d080efb.
- 0.78–0.90 / 0.73–0.87, V100 1.18–1.24 — Commit 3917b3997.
- 7070→6506, 3546→3250, 5524→5004, 3004→2442, T6 unverändert, sm_70-SASS, synthetisch, gleichverteilt — Commit c55064416.
- 6555→4303, 3271→2238, 8566→5316, 4205→2760, −1,3 %..+1,0 %, 6,3e-4 — Commit 8d4f2e978 (auch STAND.md 21.10. 13:30).
- 4979→4374, 2441→2395, 6878→5954, 3567→3218, 0,80–0,97 — Commit b2652b399 (auch STAND.md 21.10. 15:40).

## Test Result: Modell-A/B (alle Zahlen aus den Logs nachgerechnet)

- Fork-Bauten auf github.com/Peuqui/1Cat-vLLM: 23e34b03c 7f343a791 4da0bcfa4 6c0ef8efb e1cb50fd9 79e07bfef 5028a3fc4 7b8fb5387 — `git branch -r --contains <h> | grep fork/` > 0 für alle.
- Unterschiede je Paar — `git log 23e34b03c..7f343a791` (1 Commit), `23e34b03c..4da0bcfa4` (3), `23e34b03c..6c0ef8efb` (4), `e1cb50fd9..79e07bfef` (1), `79e07bfef..5028a3fc4` (2: 2f70712b0, 5028a3fc4), `79e07bfef..7b8fb5387` (13, darunter 039c1b6af und 7ee89582f → „eleven other commits“).
- Ablauf „uncounted cold boot, then reference, changed, reference, changed“ — Bootfolge in jedem Log (`##### …-fm-kalt`, prod-a, fm-a, prod-b, fm-b).
- Routing-A/B (B/ab_qpn8turing_ds2_2026-10-20.log): Kopf Z. 1 „fork-main 4da0bcfa4“; Referenz 23e34b03c laut STAND.md 20.10. 17:20 und Run-Skript-Kopf; Prefill Z. 82–84/169–171 (14,77/14,76; 113,4/113,41; 223,59/223,59) gegen Z. 125–127/212–214 (14,63/14,66; 112,74/112,71; 222,42/222,42) → −0,81/−0,60/−0,52 %; Schritte/tok/s aus den Zeilen „prosa:/code:/edit:“ der Boots prod-a/b, fm-a/b; Annahme 4,78 → 4,41 dort; Texte: cmp_runs.py fm2-qt2-ds-{prod-a,prod-b,fm-kalt,fm-a,fm-b} → Greedy 1/3, Antworten 1/8 gleich zwischen Armen, je Arm 3/3 und 8/8; „read and judged correct and equivalent“ — STAND.md 20.10. 17:20 („händisch gelesen alle richtig und gleichwertig“).
- Annahme gesampelt (B/ab_qpn8turing_accept_2026-10-20.log): 35,4→37,1; 38,2→37,6; 72,0→71,6; 93,5→93,2 %; Sampling temperature 1.0, top_k 40, top_p 1.0, min_p 0.1, feste Seeds — B/sampled_accept_bench.py Z. 25, 103.
- 27B-FP8-Zwischenstand (B/ab_q27fp8turing_2026-10-20.log, Kopf Z. 1 „Produktion 23e34b03c, Worktree 7f343a791“): tok/s/Schritte aus prosa/code/edit-Zeilen; 184k ms Z. 83/179 (67,1) gegen Z. 131/227 (66,9/66,8); Prefill 351,74/351,73 gegen 351,62/351,66; „M=1..512 uses the tensor-core GEMM“ in Q/fm2-q27t-fm-a-*.journal.txt; Texte: cmp_runs.py fm2-q27t-* → fm-a/fm-b 3/3 und 8/8 gleich zu prod-a (fm-kalt 7/8, ungezählt).
- Split-Regel (B/ab_qpn8split_2026-10-20.log, Kopf „fork-main 6c0ef8efb“): Prosa prod 74,4/74,5/74,2/74,3, fm 75,6/75,6/75,7/75,7 → Mittel +1,75 %; Code 114,7/114,9/114,3/114,4 vs 116,4/116,5/116,7/116,9 → +1,79 %; Edit 119,0/119,1/118,8/118,8 vs 120,7/120,6/121,0/121,0 → +1,60 %; Schritt 33–34 → 32–33; 184k Z. 72/148 (54,6/54,7) vs Z. 110/186 (54,2/54,2); Prefill 250k 565,86/565,86 vs 565,82/566,04; Texte cmp_runs.py fm2-qs-q27-* alle 3/3 und 8/8. „no block-FP8 layers, per-tensor“ — Log „Selected TuringQpn8Fp8LinearKernel for ModelOptFp8LinearMethod“, kein QPN8Fp8BlockScaled-Eintrag.
- Grid (B/ab_dsv4sm_2026-10-21.log Kopf „e1cb50fd9“, B/ab_moegrid_ds_2026-10-21.log Kopf „79e07bfef“): fm-Prefill Z. 126–128/214–216 beider Logs → 12,94/12,94 → 12,32/12,33; 82,20/82,14 → 79,03/79,00; 149,73/149,72 → 144,76/144,25 → −4,75/−3,84/−3,49 %; Referenzarm 4da0bcfa4 in beiden Reihen (STAND.md 21.10.), Z. 82–84/170–172: Abweichung ≤ 0,06 %; Decode-Zeilen; Texte cmp_runs.py fm2-sm-ds-fm-* gegen fm2-mg-ds-fm-* alle 3/3 und 8/8.
- sm_75-Bau (B/ab_sm75_2026-10-21.log, Kopf „fork-main 5028a3fc4“, Referenz 79e07bfef laut Run-Skript-Kopf und STAND.md 21.10. 10:35): DSv4 Prefill Z. 82–84/170–172 vs 126–128/214–216 → 12,33 → 12,30 (−0,24 %), 79,11–79,12 → 78,92–78,97 (−0,21 % Mittel), 144,52–144,56 → 144,18–144,47 (−0,15 % Mittel); Schritte prod 61/63–64/69, fm 61/63–64/68; tok/s aus den Zeilen; 178k ms 83,9/83,7 vs 84,0/84,4; Texte cmp_runs.py fm2-s75-ds-* alle fünf Boots 3/3 und 8/8 (Review-Aussage bestätigt). 27B: Boots fm-kalt Z. 222, prod-a Z. 261, fm-a Z. 300, prod-b Z. 339, „Beendet“ Z. 367 (prod-b ohne summary/english, ohne Langkontext; STAND.md 21.10. 13:30: „sm_75-A/B abgebrochen“); fm-a 74,5/74,6, 114,8/115,1, 119,2/119,4 vs prod 75,4/75,6/75,3/75,3, 116,1/116,5/116,2/115,7, 120,6/120,8/120,2/119,2 → −1,13/−1,01/−0,75 %; Schritt prod 33, fm-a 33/34 (edit); 184k Z. 293 (54,3) vs Z. 332 (54,8); Prefill 358,77 vs 358,68; fm-kalt 74,5/74,5, 114,3/115,4, 119,7/119,7; Texte: Greedy 4/4 gleich, Antworten fm-kalt/prod-a/fm-a 8/8.
- Gebündelter Lauf (Volta-/Turing-MoE): B/ab_routes_2026-10-23.log Kopf „fork-main 3b20a9b82“, Baum identisch mit 7b8fb5387 (`git rev-parse 3b20a9b82^{tree}` = `7b8fb5387^{tree}`; 3b20a9b82 selbst nicht auf dem Fork, 7b8fb5387 ja); Texte cmp_runs.py fm2-rt-ds-*: Greedy 0/3, Antworten 3/8 gleich zur Referenz; „read and judged correct“ — STAND.md 23.10. 08:15 („Qualität: 8/8 in beiden Fork-Läufen von Hand richtig“); sm7x-Triton dort laut STAND.md.

## Test Result: bekannte Fehler

- test_sm70_native_policy auf main 1 failed (test_packaged_native_policy_schemas_parse_before_build) wegen gguf_grouped_prefill_gemm_sm70_out — PR #1273 Text (Z. 10 und 70) und `git diff … ad925b18073e -- vllm/_sm70/policy.py`; fp8_qpn8_mma_sm75_out in CONFIGURED_OPERATORS — HEAD-Diff policy.py.
- V100-Intermittenz auf früherem Bau (Baum ff7685bf8): 2 failed bzw. 1 failed/1 passed, dann 3 × 2 passed — `/tmp/claude-1000/-home-mp-Projekte-vllm-research/9d4efe47-e611-49c0-91a8-bf6af11410d7/scratchpad/qpn8_pr_v100_{1,2,notune_1,notune_2,notune_3}/summary.log`. Dass die drei grünen Läufe mit ausgeschaltetem Tuning liefen, steht nur im Verzeichnisnamen („notune“) und im alten Text, nicht im Log.
- „fp8_tune_small_shapes unset = tuning on“ — PR #1187 Text („With fp8_tune_small_shapes unset, the native FP8 dense and MoE selectors time the kernel candidates …“).

## Nicht vollständig gelöste Befunde und offene Punkte

1. Befund 13: Lauf des V100-Tests auf einem origin/main-Bau steht aus; Satz im Text endet mit „[TO BE COMPLETED: number of runs and result]“.
2. Befund 7: Testergebnisse stehen aus (TESTS_PLACEHOLDER, PRECOMMIT_PLACEHOLDER). Zu erwarten laut früheren Läufen: test_sm70_native_policy 1 failed (main-Fehler, #1273). Am 23.10. hatte test_sm70_fp4_mainflow.py auf 1Cat main dbc1f4f8f 119 Fehlschläge (eingefrorenes Fixture mit `_DFLASH_NVFP4_TRACE_ARMED`, STAND.md 23.10. 08:15) — auf 6a143197e nicht geprüft; falls er wieder rot ist, gegen origin/main gegenprüfen und als main-Fehler nennen.
3. Befund 5 (Code): Der CMake-Kommentar auf HEAD (CMakeLists.txt Z. 371–373) nennt weiter die alte Begründung „Native sm_75 code is a few percent faster … (skinny MoE on an RTX 8000: 1-4 %, 10 % in one decode shape)“. Der Text widerspricht dem jetzt; der Kommentar müsste im Zweig angepasst werden (nicht meine Aufgabe, kein Commit gemacht).
4. Befund 5 (neu aus dem Log): Der 27B-Teil des sm_75-A/B zeigt −1,0 bis −1,1 % Decode mit dem sm_75-Bau (nur ein gezählter Boot, Reihe abgebrochen). Das betrifft die Hand-Kerne in _C, nicht die toten Erweiterungen; eine Beschränkung der Liste auf „laufende“ Quellen würde es nicht beheben. Möglich wäre, nur skinny_moe_qpn_sm70.cu (und fp8_qpn8_mma_sm75.cu) für 7.5 zu bauen. Entscheidung offen; der Text berichtet den Befund.
5. Befund 4 (Code, nicht nachgemessen): `state.split_k_for(reshaped_x.shape[0])` entscheidet in Python VOR dem opaken Op `torch.ops.vllm.turing_fp8_qpn8_linear` (sm70_turbomind.py apply_prepared_fp8_qpn8_linear). Unter torch.compile wird dieser Zweig einmal getract und als Literal eingebrannt (STAND.md 20.10. 21:20: AOT-Artefakte enthalten `turing_fp8_qpn8_linear.default(…, 5120, 8, 2, False)`); der Kommentar in fp8_block.py warnt genau davor. Beim 27B (kompiliert) gilt dann vermutlich ein Split für alle M — wird bei großem M getract, ist die Regel im Decode wirkungslos. Vorschlag: beide Splits an den Op geben und M innerhalb wählen (wie `_qpn8_native_linear`). Der Text sagt nur „scoping to M <= 8 not measured on a model“.
6. Cubin-Zahl (72 sm_70 / 55 sm_75) aus dem Text genommen: `vllm/_C.abi3.so` im Worktree ist von 09:39, älter als der Umbau (Commits 14:20). Nach dem Neubau mit `cuobjdump --list-elf` zählen und bei TESTS_PLACEHOLDER eintragen.
7. Befund 2: kein isoliertes Modell-A/B der Volta-Kacheln (und des m16n8k8-MoE); Text sagt das offen. Die Qualitätsaussage zum gebündelten Lauf stammt aus dem Journal (Handlesung), nicht aus einem Log.
8. NIT (Code, offen): veralteter Kommentar skinny_sm70_moe.py Z. 515 („The kernels are written for SM70 (mma.m8n8k4)“); kein Test für den SM75-Zweig „missing op“ in can_implement; can_implement fragt `current_platform.get_device_capability()` ohne Geräteindex (Standardgerät), nicht das Gerät der Schicht.
9. Historie (kosmetisch): c55064416 hat eine Leerzeile vor Signed-off-by, dadurch sind Co-Authored-By/Claude-Session dort keine Trailer (`git interpret-trailers --parse`). Commit-Nachrichten weichen vom Text ab, wo der Text genauer ist: 3917b3997 „1.4-1.8 %“ (Text: Mittel +1,75/+1,79/+1,60 %), b2652b399 „where 8 does not divide K“ (Text: K/16), 8d4f2e978 „replacing the copies in nvfp4_qpn4_sm70.cu and fp8_qpn8_mma_sm75.cu“ (Text: Decoder-Kopien in nvfp4_qpn4_sm70.cu und skinny_moe_qpn_sm70.cu), f5b53b058 nennt nur das DSv4-A/B, nicht den 27B-Teil.
10. Docstring fp8_qpn8_occupancy_split_k nennt „3-27 % faster“, die Commit-Nachricht 0,73–0,90 der Zeit (10–27 %); die 3–27 % stammen aus dem breiteren Prüfsweep (STAND.md 20.10. 18:20). Kein Widerspruch im Text, nur uneinheitlich.
