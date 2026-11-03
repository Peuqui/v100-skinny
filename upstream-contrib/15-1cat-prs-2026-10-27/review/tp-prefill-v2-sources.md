# Quellen zu body-tp-prefill.v2.md (27.10.2026)

Logs unter `~/.cache/bench-scripts/` (kurz B/), Antworten unter `B/quality_2026-09-27/` (Q/), nsys-Datenbanken unter `B/nsys/<tag>/*.1.sqlite`
(Abfragen per sqlite3 auf `CUPTI_ACTIVITY_KIND_KERNEL` / `_MEMCPY`, Summe `end-start` je deviceId). Hilfsdateien dieser Prüfung
(open.json, files/N.txt, conflicts.py) in `/tmp/claude-1000/-home-mp-Projekte-vllm-research/b66b4495-69d0-4f76-83bc-cc61ac9e6906/scratchpad/tpv2/`.
Zweig = Worktree `1Cat-vLLM-pr-tp-prefill`, Kopf 5bd8a8d21; origin/main = 899e9ed33 (`gh api repos/1CatAI/1Cat-vLLM/commits/main` = 899e9ed3319a…).

## Stapel und Hashes

- Eigene sechs Commits (`git log --abbrev=9 pr-turing-kernels..HEAD`): 78d8c2b18 Attention, 24fb9a581 Flash-V100, bfd35e7b5 Überlappung,
  cacae3976 Copy-Engine, 7c9951aa2 Split-KV, 5bd8a8d21 NVFP4-Kachel. Darunter 1732a24ea + 3c981c0f0 (#1227), 3c0f7b3f5 (#1216), pr-turing-kernels = 001b73e3e (#1278, 7 Commits).
- #1278 live = 001b73e3e (MERGEABLE/UNSTABLE), #1227 live bd41a3194 (MERGEABLE/CLEAN), `git range-diff` gegen 3c981c0f0/1732a24ea: „=“ beide.
- #1216 live 130ff00e6: `gh pr view 1216` → CONFLICTING/DIRTY gegen main. `git range-diff 130ff00e6^! 3c0f7b3f5^!`: Unterschied nur in
  runtime_reference.md-Ankern (L619/L1061/L632 → L662/L1146/L675); in kernel.py nur Kontextzeilen anders. Wenn #1216 vor dem Senden
  force-gepusht wird, den Satz „#1216 as opened conflicts with main …“ anpassen (der Stapelcommit 3c0f7b3f5 selbst liegt auf #1278, nicht auf main).
- Fork-Bauten (alle auf fork/ vorhanden, `git branch -r --contains <h> | grep -c fork/` > 0): 4da0bcfa4, f9ac81b6b, 79e07bfef, a1ac0c89d, 88d25b8bd, c1816294c, afe87abd0, f21d1c564.
- `git log a1ac0c89d..f21d1c564` = 88d25b8bd, c1816294c, afe87abd0, f21d1c564 (4). `merge-base --is-ancestor f9ac81b6b a1ac0c89d` = ja;
  3856a7b67 (Fork-Fassung von 24fb9a581) in keinem Arm. a1ac0c89d enthält 33c79f2b0 (#1216), df08fe7a9 + 9a41a8e67 (#1227).
- `git log 4da0bcfa4..f9ac81b6b` = 9 Commits (f9ac81b6b + 039c1b6af, 5028a3fc4, 2f70712b0, 79e07bfef, e1cb50fd9, 5669da106, 0314585ac, 6c0ef8efb).
- `git log 79e07bfef..a1ac0c89d` = 21 Commits, enthält f9ac81b6b.

## Purpose

- 150 033 Token, 108 s FA2, 96 s NCCL, 268 s: nsys `q27-stages-1021` (Kopf B/profile_stages_q27_2026-10-21.log „Produktion 4da0bcfa4“, Prefill 267,91 s):
  FA2 (`%flash_fwd%`) 108,10/107,99 s, 1995 Aufrufe; NCCL 95,77/95,71 s; Rest 63,58/63,75 s; Spanne 267,68 s.
- 208 s, Attention 47 s, NCCL 96 s: nsys `q27-stages-turing-1021` (B/profile_q27_turing_2026-10-21.log „fm-Baum f9ac81b6b“, Prefill 207,87 s):
  paged_prefill_attention 46,84/46,99 s (1615 Aufrufe), FA2 0,1 s (380), NCCL 96,43/96,28 s, Rest 64,02/64,01 s, Spanne 207,62 s.
  Review nannte 95,95 s NCCL; meine Summe ist 96,3–96,4 s, „96“ stimmt in beiden Fällen.
- 2,08 GB/s, 8 192 000 B: nur Commit cacae3976 (3,93 ms NCCL); 8 192 000 B / 3,93 ms = 2,08 GB/s, 8 MiB ergäbe 2,13 GB/s → Größe 800 × 5120 × 2 B.
- FA2 „about 23 TFLOPS“: 61 TFLOPS × 61,5/165,8 = 22,6 (Commit 78d8c2b18); exakt 4·12·256·(2048·150 000 + 2048·2049/2) FLOP → neu 61,8, FA2 22,9.
  STAND.md 21.10. 13:30 nennt 22,5 TFLOPS aus `scratchpad/turing/fa_bench.py` (Skript nicht mehr auffindbar).
- Kernbeschreibung Attention (128 Zeilen, 8 Warps, 32 Schlüssel, zwei Hälften, 32-Bit, Seitengröße als Template) — Commit 78d8c2b18 und STAND.md 21.10. 14:05.
- Auswahlbedingungen — flash_attn.py `_use_turing_prefill` (HEAD): `0 < min <= max_query_len`, head 256, causal, fp16, FA-Version 2, Fenster (-1,-1),
  ohne ALiBi/Sinks/Softcap, nicht quantisiert, nicht batch-invariant, `key_cache.size(1) in (16, 32, 64)`, größter Offset ≤ INT32_MAX,
  `is_device_capability((7, 5), device_id=query.device.index)`. sm_75-only: CMakeLists `VLLM_SM75_TURBOMIND_SRCS` (CUDA_ARCHS "7.5").
- Flash-V100-Fix — `git show 24fb9a581` (ops.py `_load_flash_dense_forward`, `flash_dense_forward_available`; runtime.py:38 wirft den Fehler).
  „first GLM-5.3-Flash boot“ — STAND.md 21.10. 15:40 „GLM erster Start: Fehlerkette … (1) 3856a7b67“. „every V100 stage“ nicht belegbar → gestrichen.
- Überlappung — prefill_overlap.py (HEAD): `__init__` (TP<2/PP/DP, LoRA/KV-Transfer, DCP, Encoder-Decoder), `check_model` (FusedMoE; `deepstack_num_level > 0`),
  `split_token` (CUDAGraphMode.NONE, ≥ min, keine Draft-Token, ohne Padding), `token_range_model_inputs` (NotImplementedError);
  model_runner.py (PLE-Offload-ValueError; `check_model` in load_model nach LoRA; dummy_run → kein Split); mamba_hybrid.py (Modus „all“ ValueError; `align_ubatch_split`);
  parallel_state.py (`_all_reduce_out_place` und `all_reduce_sum2` über `dbo_run_collective_overlapped`; `_reject_overlapped_fused_norm` in den SM70-TP2/TP4-Gemma-RMSNorm-Ops);
  ubatching.py (abort_event, `_raise_if_aborted` in `__enter__` und `_cpu_yield`, `MicroBatchAborted`); kernel.py (Feld außerhalb compute_hash).
  MoE-Begründung „asserts outside DBO when prepare/finalize is not async“ — Review (modular_kernel.py:1260/1486), nicht selbst nachgelesen.
- 801 + 799, Builder 0/1/2 — Commit bfd35e7b5, Test test_align_mode_split_keeps_the_last_state_block.
- Copy-Engine — host_reduce.py (`rejection_reason`: `copy_engine_mode_not_capturable` bei `nbytes > max_bytes` und `is_current_stream_capturing()`),
  kernel.py `HostReduceConfig.__post_init__` (0 < copy ≤ max → ValueError), host_reduce.cu (kCopyChunks 16; TORCH_CHECK „cannot be captured“ Z. 353–355),
  Docstring „pins four slots“. max_bytes 262 144 B = 25,6 Token × 5120 × 2 B.
- Split-KV — cu: kMaxSplits 4, `splits_by_waves(ctas, SMs × occupancy)`; 150/76 CTAs = ceil(1600·6/128)·2 bzw. ceil(801·6/128)·2; 72 SMs RTX 8000.
- NVFP4 — `git show 5bd8a8d21` (`_groups_per_block`: 16 auf (7,5), sonst 64); 2,2–2,3× = 1,090/0,496 … 0,525/0,230 → 2,20/2,32/2,21/2,28; „about 200 GB/s“ Commit.
- Überlappung allein 13 % / 3 % / 0,5 %: B/ab_tpo_2026-10-23.log Block 88d25b8bd (29k 29,0/28,8/28,5/28,6 → Mittel 28,73 gegen Referenzmittel 33,04;
  184k 260,28 gegen 269,05; 250k 400,14 gegen 402,27).

## Not a duplicate

- 81 offene PRs (`gh pr list -R 1CatAI/1Cat-vLLM --state open --limit 300` → open.json); Dateilisten je PR per `gh api --paginate …/pulls/N/files` (files/N.txt).
- Schnittmenge mit den Dateien der sechs Commits (`git diff --name-only 1732a24ea..HEAD`): #1020 #1092 #1102 #1125 #1167 #1216 #1217 #1218 #1221 #1227 #1232
  #1235 #1243 #1245 #1260 #1261 #1264 #1271 #1273 #1276 #1278 #822 #828 #890 #894 #897 #903 #953 #983.
- Symbolsuche in `git diff $(merge-base origin/main head) head` aller PRs: Treffer nur #1260 (get_flash_dense_forward, dense_prefill_lse_available),
  #1217 (cuStreamWriteValue32, PLE-Gather), #1216 (host_reduce), #1208 (all_reduce_sum2 nur in Tests des P2P-Custom-AR), #822 (Pfad …/utils/nvfp4_qpn2_dequant.py, ohne Kachel).
- #1260: `git diff -w` auf ops.py: `prepare_attention_runtime()` vor `bind_attention_operation` in get_flash_dense_forward (Kommentar „MLA can load the LSE entry …
  Bind each engine's owner“); CONFLICTING mit main (gh). Damit behebt #1260 denselben Fehler anders → im Text offen benannt.
- Dreiwege-Merge (lesend, `git merge-tree <base> <a> <b>` alte Form, schreibt nichts) je PR gegen 1732a24ea und gegen HEAD (conflicts.py): neue Konflikte nur
  #1216 (host_reduce.cu, test_host_reduce.py, host_reduce.py), #1227 (runtime_reference.md, attention.py, flash_attn.py) — beide gestapelt — und #1260 (flash_v100/ops.py, model_runner.py).
- #1245: Titel/Text (`gh pr view 1245`): Push-All-Reduce für NVLink/PCIe-Peer-Topologien, Tabelle 5 KiB je Rang, Token-Generierung; CONFLICTING/DIRTY.
- Stichwortsuchen (`gh pr list --search`): all-reduce, allreduce, overlap, host memory, copy engine, P2P, prefill attention, Turing, SM75, split-KV, NVFP4 dequant,
  micro-batch, DBO, ubatch, Flash-V100, bound before execution; Treffermengen geprüft, inhaltlich nur die oben genannten relevant.

## Test Plan — Testdateien (alle auf HEAD vorhanden)

- tests/v1/worker/test_prefill_overlap.py (neu, bfd35e7b5)
- tests/v1/worker/test_gpu_model_runner_v2.py (angepasst, bfd35e7b5)
- tests/kernels/attention/test_paged_prefill_attention_sm75.py (neu, 78d8c2b18/7c9951aa2)
- tests/distributed/test_host_reduce.py (#1216, erweitert in cacae3976; Paar mit NCCL_P2P_DISABLE=1)
- tests/kernels/quantization/test_nvfp4_qpn2_dequant.py (bestehend, unverändert; deckt die Kachel je Gerät)
- tests/v1/attention/test_flash_v100_dense_prefill_availability.py (neu, 24fb9a581)
- tests/kernels/attention/test_flash_attn_packed_gqa_verify.py (#1227; im Lauf 18:39/18:40 als Modul übersprungen: „1 skipped“ auf RTX und V100 –
  Ursache nicht ausgegeben, entweder ImportError aus vllm.vllm_flash_attn oder `is_fa_version_supported(2)` falsch. Entweder klären oder aus dem Test Plan nehmen.)
- Indirekt per pre-commit: tests/tools/test_architecture_reference.py, test_architecture_doc_urls.py, test_architecture_check_scope.py (Hook test-architecture-reference).
- Nicht sinnvoll: tests/v1/distributed/test_dbo.py (braucht deep_ep).
- Kein Test für die Prüfung copy_engine_max_bytes > max_bytes (HostReduceConfig.__post_init__).

Bereits vorhandene Läufe auf 5bd8a8d21 (`B/prtests_2026-10-27/`, nur zur Orientierung, Platzhalter bleiben):
- RTX (tp-final-rtx 18:39): paged_prefill 21 passed; nvfp4 11 passed; flash_v100 2 passed; runner_v2 5 passed, 2 failed
  (test_sm70_v2_mtp_profile_reports_from_last_pp_stage[pp2-last-stage-tp1], [pp2-first-stage]); test_prefill_overlap 2 failed / 12 passed —
  Fehler `InputBatch.make_dummy() got an unexpected keyword argument 'context_len'`, im heutigen Stand der Datei gibt es dieses Argument nicht
  (Z. 26 `make_dummy(2, 12, buffers)`); Wiederholung tp-overlap-rtx 18:43: 14 passed, tp-overlap-cpu 18:43: 11 passed, 3 skipped.
  Vermutlich lief 18:39 mit einer ungespeicherten Änderung im Arbeitsbaum.
- V100 (tp-final-v100 18:40): paged_prefill 5 passed, 16 skipped; nvfp4 11 passed; flash_v100 2 passed; runner_v2 und overlap wie RTX.
- Paar (tp-final-pair 18:41): test_host_reduce.py 2 passed.
- runner_v2 auf einem main-nahen Baum (tp-main-runnerv2-rtx 18:43, 1Cat-vLLM-pr-mla-codec f4d5a4e6a = 899e9ed33 + #1272, nur kv_cache_interface): dieselben 2 failed.

## Path coverage / Contract

- Eintrag 27B: `~/.cache/prod-switch-2026-09-29/config.yaml.vor-tpo-2026-10-23` Z. 83–99: --dtype float16, --disable-custom-all-reduce, --enable-prefix-caching,
  TP 2, --block-size 16, --max-model-len 262144, --max-num-batched-tokens 2048, MTP 3, cudagraph_capture_sizes [1,2,4,8], NCCL_P2P_DISABLE=1,
  CUDA_VISIBLE_DEVICES=0,2, VLLM_USE_V2_MODEL_RUNNER=1, PYTHONPATH=Triton-sm7x-Kopie, --kernel-config '{"triton_sm7x_tensor_cores":true}'.
  KV-Cache auto: Journal Q/fm2-tpo-q27-fm-a-203501.journal.txt `kv_cache_dtype=auto`.
- Kernel-Config des Fork-Arms: B/start_ab_tpo_2026-10-23.sh (KC_NEW), STAND.md 23.10. 21:25; Journale fm-a/fm-b: `HostReduceConfig(enabled=True, max_bytes=262144,
  copy_engine_max_bytes=20971520), tp_prefill_overlap_min_tokens=1024`, `triton_sm7x_tensor_cores=True`; Referenz-Journal ohne beide Felder, mit triton_sm7x.
  Einzelboots: 18:43 nur `tp_prefill_overlap_min_tokens=1024`; 19:24 und 19:56 zusätzlich `copy_engine_max_bytes=20971520`.
- `triton_sm7x_tensor_cores` auf diesem Zweig und auf main: `git grep` leer.
- Route-Hit-Zeile „TP prefill overlap: prefill batches of at least 1024 tokens run as two interleaved halves.“ — je 1× in Q/fm2-tpo-q27-fm-a-203501 und fm-b-210620.
- Compile-Caches: start_ab_tpo (Kopien `*-routes-2026-10-23` → `*-tpo-2026-10-23`, Kommentar „same compile artifacts“).
- Commit-1-Profile mit Standard-Triton: Config-Sicherungen `config.yaml.vor-nsys-q27-stages{,-turing}-1021`, 27B-Eintrag ohne Triton-PYTHONPATH;
  sm7x-Triton erst ab 23.10. (STAND.md 23.10. 18:05). Unit-Tests: run_pytest_per_file.sh setzt PYTHONPATH=<Baum>, kein Triton-Overlay; venv triton-3.6.0.
- „captured batches ≤ 8 Token = 81 920 B“: 8 × 5120 × 2; Capture-Test 5120 × 64 × 2 = 655 360 B (test_host_reduce.py Z. 127).
- runtime_reference.md auf HEAD: HostReduceConfig __post_init__ #L681, Klasse #L662; KernelConfig compute_hash #L1169, Klasse #L692; AttentionConfig #L111/#L16 —
  stimmt mit `grep -n` in kernel.py/attention.py überein. Generator nicht ausgeführt (Hook im pre-commit-Lauf).

## Test Result

- A/B: B/ab_tpo_2026-10-23.log ab Z. „fork-main f21d1c564, Beginn 20:14:18“: fm-kalt (nur Boot+Greedy), prod-a, fm-a, prod-b, fm-b.
  29k (je Boot vier Prompts 6171/6173/6185/6187 mit 29 191/29 201/29 167/29 243 Token): prod 33,0/33,0/33,1/33,0 und 33,1/33,0/33,1/33,0; fm 19,1/19,2/19,2/19,3 zweimal
  → Mittel 33,04 → 19,20 = −41,9 %; tok/s 881–886 → 1515–1528 (Token/ttft je Prompt).
  184k: 269,05/174,34/269,21/174,38 → −35,2 %; 684,4/684,0 → 1056,2/1055,9 tok/s. 250k: 402,27/269,06/402,47/269,29 → −33,1 %; 621,5/621,1 → 929,1/928,3 tok/s.
- Decode: prosa prod 74,6/74,9/74,6/75,0, fm 75,0/75,2/74,9/74,8; code prod 114,8/115,0/115,1/114,3, fm 115,6/115,9/115,4/115,8; edit prod 119,6/119,4/119,1/119,9,
  fm 119,9/119,8/119,9/120,1; Schritt 33 ms (prod-b edit einmal 34); Annahme 2,47/3,82/3,99 überall. 184k ms/tok/s: prod 54,3/50,7 und 54,4/50,6, fm 54,3/50,7 zweimal;
  250k: prod 62,3/40,8 zweimal, fm 62,2/41,1 und 62,4/41,0. (STAND.md nennt „Code 114,8–115,6“ – gemischt; Log ist maßgeblich.)
- Greedy: „greedy vs reference: identical ×3“ in allen fünf Boots. Antworten: Q/fm2-tpo-q27-{prod-a,fm-a,prod-b,fm-b}-ds.json, Vergleich der Felder text:
  7/8 gleich, „zug“ erste Abweichung Index 909, beide `tokens` 600 `finish` length, beide bei „v = 287 · 12/41 km/h“ abgeschnitten (Endergebnis nicht erreicht);
  je Arm a = b für alle 8. Lange Prompts: nicht textlich verglichen (Log zeigt bei 6171 im ersten Durchgang 95 Token Referenz gegen 121 Token geändert).
- Speicher: „Messung 0=47979 … 2=47979“ (prod-a/b), „0=48285 … 2=48285“ (fm-a/b); Boot+Greedy 47 867 überall; „GPU KV cache size: 1,038,640 tokens“ in allen Boots.
  Erklärung Profillauf: model_runner `if self.prefill_overlap is not None and not dummy_run` → Speicherprofil ohne Hälften; Ursache der +306 MiB nicht weiter zerlegt.
- Einzelboots: gleiche Log-Datei, Blöcke 88d25b8bd (18:43), c1816294c (19:24), afe87abd0 (19:56): 29k 29,0/28,8/28,5/28,6 | 21,7/21,8/21,8/21,9 | 21,1/21,1/21,1/21,2;
  184k 260,28 | 216,81 | 185,25; 250k 400,14 | 341,66 | 283,5. Jeweils erster Boot des Baus (Boot 226 / 139 / 116 s).
- Commit 1 gebündelt: B/ab_final_2026-10-23.log (Kopf „fork-main a1ac0c89d“), 27B-Boots Z. 191–348, Greedy identisch; Q/fm2-fin-q27-*-ds.json 8/8 gleich zwischen
  allen vier Boots (selbst verglichen). Referenzarm 79e07bfef mit Standard-Triton, Fork-Arm mit sm7x-Triton (STAND.md 23.10. 08:15/17:30).
  Prompt-Längen (Tokenizer des Modells, Chat-Template mit Systemtext „Reasoning effort …“): Qualität 65–132 Token, Greedy (ohne Template) 18/20/52 Token.
  Der Systemtext ist allen Qualitätsfragen gemeinsam → Präfix-Cache-Treffer möglich, deshalb „not traced“.
- CE-Profile: `q27-routes-1023` (Kopf „Produktion a1ac0c89d“, 150k), `q27-tpo-1023` (100k, 118,22 s), `q27-tpo-ce-1023` (100k, 95,85 s). Bau der beiden Überlappungsprofile
  aus der Zeitfolge erschlossen (Profil 19:05 zwischen Einzelboot 88d25b8bd bis 19:02 und c1816294c ab 19:24; Profil 19:41 nach c1816294c, vor afe87abd0 19:56), nicht im Log vermerkt.
  Dequant je Aufruf Gerät 0 (µs): `_nvfp4_qpn2_dequant_kernel` Grid 544×5 1081,4/1412,0/1138,3, 160×9 569,9/611,2/631,7; `fp8_qpn8_dequantize_sm70_kernel` 10240 285,0/389,9/316,1,
  8960 280,4/377,4/291,0, 3840 118,0/139,5/123,4 → +31/+7/+37/+35/+18 % bzw. +5,3/+10,8/+10,9/+3,8/+4,6 %.
  GEMM `turing_fp16_s1688gemm_fp16_256x128_ldg8_f2f_tn` Grid 68×13 3347,9 µs (/13 = 257,5) gegen 68×7 2195,8 (/7 = 313,7, +21,8 %) bzw. 1807,8 (/7 = 258,3, +0,3 %).
  Andere GEMM-Kacheln haben zwischen den Profilen andere Formen; „5 to 22 %“ (Commit-Text) nur für 22 % nachgerechnet → im Text nur „largest GEMM“.
  Route-Hit: NCCL-Kerne Gerät 0 16 832 (75,58 s) → 448 (0,34 s); MEMCPY copyKind 1/2 ≥ 64 KiB 260 166 / 260 164, je 135,21 GB; add_half8_kernel 16 384.
- Mikrobenchmarks: nur Commit-Nachrichten (78d8c2b18, 7c9951aa2, cacae3976, 5bd8a8d21), nicht nachmessbar; Attention zusätzlich STAND.md 21.10. 14:05 (gleiche Zahlen).
  CUDA-Graph nur bei Split-KV belegt (Commit „CUDA graph“); CE eager zwingend; Attention/NVFP4 ohne Angabe → „not recorded“.

## Review-Befunde: Stand

1. A/B-Zuordnung — erledigt im Text: A/B = Commits 3–6 (Referenz a1ac0c89d, geändert f21d1c564), Commit 1 eigene Belege (nsys 4da0bcfa4 → f9ac81b6b, 267,91 → 207,87 s,
   Kernzeiten; gebündelte Qualität 79e07bfef → a1ac0c89d), Commit 2 in keinem Arm. „96 of 208 s“ jetzt als Profil MIT neuem Kern, FA2-Zustand 268 s / 108 s / 96 s.
2. Capture — Code erledigt (host_reduce.py rejection_reason), Test im Copy-Engine-Worker (Capture 655 360 B). Text: Pfadtabelle und Purpose.
3. Modellprüfung — Code: FusedMoE, deepstack_num_level, sum2 über Comm-Stream, fusionierte AR+RMSNorm wirft (erst zur Laufzeit, nicht beim Start), Test
   test_overlap_rejects_moe_and_deepstack_models. Text „Format: any“ → „dense models (measured: NVFP4)“. **Offen: Granite 4 Vision**
   (granite4_vision.py: Deepstack-Merkmale in `_ds_buffers`, gelesen als `[:n]` mit n = inputs_embeds.size(0), Z. 893; Attribut `_ds_layer_indices`, kein
   `deepstack_num_level`; Sprachmodell dichtes GraniteModel) wird von check_model nicht erkannt → Hälfte 2 liest stilles Falsches wie im Review für Qwen3-VL beschrieben.
   Vorschlag: check_model zusätzlich auf `_ds_layer_indices` (nicht leer) prüfen oder generisch Modelle mit `get_multimodal_embeddings`/Seitenpuffern ablehnen; Test ergänzen.
   Im Text vorerst als „Unverified scope“ benannt.
4. Turing-Prädikat — Code erledigt (Seiten 16/32/64, Offset), Test test_selection_keeps_unsupported_caches_on_fa2 (128, 800, > 2^31).
5. runtime_reference.md — Anker stimmen auf HEAD (siehe oben); Hook-Ergebnis steht aus (PRECOMMIT_PLACEHOLDER).
6. Abbruch — Code erledigt (abort_event), Test test_a_failing_half_stops_both_and_reports_its_error (erste/zweite Hälfte), gilt auch für DBO (gemeinsame Kontexte).
7. „does not use Triton“ — Text korrigiert: Commit 6 ist eine Triton-Kachel; beide Arme mit gepatchtem Triton; Sweep-Triton unbekannt.
8. CUDA-Graph-Angaben — je Punkt: Split-KV graph, CE eager, Attention/NVFP4 „not recorded“.
9. DCP/Encoder-Decoder — Code erledigt (`__init__`), Test test_overlap_rejects_configurations_it_cannot_split.
- NITs: kernel-config-Feld erklärt; 184k 54,3–54,4 ms; 250k tok/s 40,8 → 41,0–41,1; Split-KV-Spalte „without split-KV“; 8 192 000 B; zwei gezählte Boots je Arm + ein
  ungezählter; TFLOPS als Zahlen; 96/64 „after 150k“; Speicher „changed 48,285 / reference 47,979“; #1245 im Duplikatsabsatz; #1216-Konfliktzustand genannt;
  Profillauf ohne Überlappung (+306 MiB, KV-Cache gleich) erwähnt; copy_engine_max_bytes > max_bytes geprüft (Code, ohne Test).
- Nicht erledigt (Code-NITs, nicht Teil des Auftrags): Stream-Memory-Ops werden beim Öffnen des Kanals nicht geprüft (im Text als Unverified scope);
  `cudaFuncSetAttribute` + Occupancy-Abfrage je Aufruf (paged_prefill_attention_sm75.cu Z. 475–480), nicht je Gerät zwischengespeichert.
- Scope-NIT 24fb9a581 (gehört eher in einen MLA-PR): im Text als unabhängig markiert, Auslagern angeboten; zusätzlich jetzt #1260 als konkurrierende Fassung.

## Weitere Auffälligkeiten

- Code-Texte sagen weiter „about a quarter of the tensor-core rate“ bzw. „of the m16n8k8 rate“ (vllm/config/attention.py Docstring von
  flash_attn_turing_prefill_min_query_len; flash_attn.py `_use_turing_prefill`-Docstring). Gemessen sind ≈ 23 gegen 61 TFLOPS (RTX-8000-Spitze FP16-Tensor laut Datenblatt
  rund 130 TFLOPS, nicht gemessen → 17 %); „Viertel“ passt nur zur eigenen Rechengrenze des Kerns (83 TFLOPS, STAND.md 21.10. 14:05). Nicht geändert (Code), nur gemeldet.
- 5bd8a8d21: Sweep-Triton (Standard oder sm7x mit anderem Registerbudget) nicht dokumentiert; die Begründung „past Triton's register budget“ hängt davon ab.
- 27B-Langprompt-Ausgaben unterscheiden sich zwischen den Armen mindestens einmal (6171: 95 gegen 121 Token) – erwartbar (Split-KV ändert die Summationsreihenfolge),
  aber im Text nur als „not compared“ vermerkt.
