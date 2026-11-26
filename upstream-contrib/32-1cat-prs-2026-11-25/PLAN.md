# PR-Runde an 1Cat, 25.11.2026

Auftrag Peuqui 25.11.: alle echten Verbesserungen seit der letzten Einreichung einreichen. Jeder PR wird vor dem Senden von einem
unabhängigen Agenten geprüft. Alle Zweige werden auf das aktuelle 1Cat main gehoben, unsere PRs kommen obendrauf. 1Cat mergt in Schüben.

Basis: 1Cat main 7db135fa0 (#1556). Zweig `pr-base-7db135fa0`, Baum `1Cat-vLLM-prmain`, nativer Bau
`~/.cache/bench-scripts/build_prmain_2026-11-25.sh` (Turing-FA2 aus v16). Bestandsaufnahme aller 288 Commits origin/main..prod-v16:
`inventar_origin-main..prod-v16.tsv`.

Seit der Bestandsaufnahme von 1Cat gemergt: #1571 (DSML nicht-String), #1609 (KV-Zeroer virtuelle Seiten), #1610 (kpool-Tail-Block).

## Neue PRs auf main

| Zweig | Inhalt (Fork-Commits) | Hängt an | Stand |
|---|---|---|---|
| pr-adaptive-verify | variable Prüflänge c504ce162 90c9def2c c9edcaa35 75fc73769 | – | Rebase (Agent) |
| pr-pp-pack-handover | gepackte PP-Übergaben a43e966b1 df927238c | pr-adaptive-verify (Testdatei) | Rebase (Agent) |
| pr-glm5next-kda-decode-glue | KDA-Kleber b817aa185 381a4eac7; Umbau: Kern als `FusedRMSNormGated.forward_cuda`, opt-in `+fused_rms_norm_gated` statt Umgehung von #1165 (`kda-dispatch-analyse.md`); danach zieht der Fork nach (GLM-Eintrag `+fused_rms_norm_gated`) | – | Umbau (Agent) |
| pr-uva-shared-host-memory | UVA exakt pinnen, Einbettung teilen 1f17e135f (+ uva-Teil 33a070e99) | – | Rebase (Agent) |
| pr-cutlass-int8-build-query-v2 | e398ceb14 (Duplikat-/Erledigt-Prüfung) | – | Rebase (Agent) |

## Gestapelte neue PRs (nach dem Rebase ihrer Basis)

| Zweig | Inhalt | Basis-PR |
|---|---|---|
| wo_a Turing aus FP8 | 6b8d41945 e336f616e | #1278 |
| kv_b-Skinny-Kern | 48a92aa11 | #1282 |
| MoE-Kernpaket (Sechs-Bit, 3 CTAs Turing, Volta-Ring) | cab7b545d 358108041 dcd2b9953 de1405f55 (+ Benchmark 341d5120c) | #1286 |
| Routing als Sortierung | ca2c1b65a | #1286/#1348 |

## Bestehende offene PRs

- #1536: Schichtenprüfung rot („model coupling 31 → 32“ in `kv_cache_utils.py`, Docstring nannte DFlash2). Docstring ohne Modell,
  Zweig `pr-kv-pool-max-model-len-v2` auf der neuen Basis (Baum `1Cat-vLLM-pr-kvpool`).
- #1535, #1611: sauber, nur auf Konflikte mit main prüfen.
- #1221, #1278, #1282, #1286, #1347, #1348: Konflikte mit main, Rebase der Stapel #1278 → #1282 → #1347 und #1286 → #1348, #1221 allein.
  Nativer Bau je Stapel als Kettenschritt.

## Nicht eingereicht

- FLA Shared Memory je Gerät (a6d6d3e82/969dfdea0): wirkt auf unseren Karten nicht (V100 96 KB, RTX 64 KB, Grenze 100 KiB), kein Gewinn.
- Sync-PP mit Spec Decode (fc51a3be1/7096dc040/f29a2bf00): auf main überholt durch #1371 (bb03140ee: Engine wendet Drafts erst nach
  update_from_output an, `_pp_prepend_accepted_drafts`) und den Negativ-Wächter im Scheduler; unser Nullfall-Test grün auf main, der
  andere prüft eine Reihenfolge, die mains Engine nicht mehr erzeugt. **Fork-Aufräumen beim nächsten Merge:** unsere Scheduler-Wächter
  und `_pp_sync_prev_drafts` liegen dort neben mains Lösung (doppelt).

- Host-Bereinigung 3b06bda1a/508ac9f8f: A/B gleich (`ab_hpo_{glm,dsv4}_2026-11-24a.log`), kein messbarer Gewinn.
- Schmaler FP16-Linear-Kern (Router-Kette), wo_a-Überlappung: verworfen (31-…/ROUTER_KETTE.md, WOA_TURING.md).
- PLE-Plattenstufe: main hat eigenen Zeilencache, Flash-Next gelöscht (PLAN 02.11.).
- Fork-interne Commits (Doku, Diagnose, Merge-Kleber) laut Bestandsaufnahme.

## Ablauf je PR

Rebase und statische Prüfung → Tests CPU und GPU (V100 und RTX) als Kettenschritt mit Hold → PR-Text, jede Zahl mit Quelle →
unabhängige Prüfung (Diff, Text, Messdateien, AGENTS.md) → Befunde nachprüfen und beheben → senden → Live-Text prüfen → #674 nachtragen.

## Fork zieht nach (nach der Runde, als eigenes Bündel)

- UVA: Fork-Code (eigener memfd, 1f17e135f) durch die PR-Fassung ersetzen (SharedHostArena + fileno() + registration_guard, 51a8bb14c).
- KDA: Kern als `FusedRMSNormGated.forward_cuda`, GLM-Eintrag `+fused_rms_norm_gated` (Prüfstufe 1, bitgleich).
- Sync-PP: doppelte Wächter/`_pp_sync_prev_drafts` neben mains #1371-Lösung entfernen.
- Indexer-Epilog-Kommentar (fp32-Kachel) wie im Rebase.

## Stand 25.11. 07:00 — Karten an AIfred (GLM) abgegeben, Fortsetzung offen

- Hold 06:5x freigegeben (AIfred forderte das Ende an). Test- und Baukette gestoppt.
- Getestet: pr-kvpool (75/75 CPU/V100/RTX), pr-adaptive-verify, pr-pp-pack (CPU-Fehler = main, Name für Name; GPU-Abweichungen
  test_config.py V100 15 / RTX 6 statt 9 und pr-pp-pack test_sm70_long_attention_graphs V100 1 Fall: Gegenprobe auf main mit GPU offen),
  pr-r-dspark (grün), KDA-Stand 2e4e44a80 (94/94 bitgleich, Benchmarks).
- Gebaut: pr-cutlass-int8, pr-r-turing, pr-r-triton. Offen: Bauten pr-stack (#1347), pr-r-glm53, pr-r-glm53ph; Tests pr-cutlass-int8 (abgebrochen),
  pr-r-turing, pr-r-triton, pr-stack, pr-r-glm53, pr-r-glm53ph, pr-uva (51a8bb14c).
- Fortsetzung: Warteschlangen neu schreiben (erledigte Zeilen raus), Ketten neu starten.
- KDA: Kern umgebaut (4 Zeilen je Programm, [ROWS,128]-Ladung), uncommittet im Baum; Endregel und Endstand-Tests auf GPU offen; Logs in
  `~/.cache/bench-scripts/kdanorm_v2_logs_2026-11-25/` nachträglich aus Terminal geschrieben → Belege neu mit tee messen.
- Nebenbefund: `rms_norm_gated(..., out_dtype=None)` schreibt das Ergebnis in x zurück (`fla/ops/kda.py` layer_norm_gated_fwd, y = x);
  prüfen, ob 1Cats forward_cuda-Aufrufer das erwarten.
- #1536: Update 159cce181 geprüft (Urteil senden), Force-Push vom Filter gesperrt → Freigabe Peuqui; Antworttext im Gesprächsverlauf.

## Nachtrag 25.11. 07:15

- #1536: Update 159cce181 gepusht (Freigabe Peuqui), Antwort und Hinweis gepostet (live = Entwurf); Schichtenprüfung grün, pre-commit rot nur
  wegen mains #1556 (test_processor_text_only_fallback.py, processor.py:1208), main selbst ebenso rot.
- #1278-Rebase: fehlender Turing-Wächter vor mains Gated-SiLU-Weg ergänzt (cc4751cf0, Test rot ohne/grün mit), alle abhängigen Zweige
  neu aufgesetzt (range-diff alle `=`): #1282 b4cc63c10, #1347 fbc1b58e5, #1286 8b85abcc2, #1348 de68acd7c, kv_b b7e3e8480, wo_a d20326a7e,
  MoE c6b63b32f, Routing a348b5440.
- Neue gestapelte Zweige: pr-dsv4-wo-a-turing-fp8 (Hand-Merge im .cu, nie kompiliert → Bau), pr-mla-kv-b-skinny, pr-sm70-moe-kernel-pack
  (.cu per clang-format zusammengeführt), pr-sm70-moe-routing-sort-v2 (CPU-Test aus dem alten Zweig). Vor dem Senden auf neuestes main:
  wo_a muss dann `_block_qpn8_prefill_dense` (#1535) nutzen; wo_a konfliktiert mit #1286 (RowGroupScales).
- Auf main rot, nicht von uns: tests/quantization/test_sm70_fp8_prefill_exact_dense.py zwei compressed-tensors-Tests (#1557).

## Stand 25.11. 09:20 — Karten wieder an AIfred (Hold-Ende 09:16)

- Getestet (Kette 2, 07:46–09:16): GPU-Gegenprobe main (prmain), pr-uva, pr-cutlass-int8, #1278, #1282, #1347, #1286, #1348. Rote Dateien
  werden gegen main abgeglichen → `testauswertung-2026-11-25.md` (Agent).
- Offen: pr-n-kvb (abgebrochen), pr-n-route, pr-n-woa, pr-n-moe → `chain_prtest3_2026-11-25.sh` (Warteschlange test_queue3, alle Bauten fertig).
- KDA fertig: bb34e7637 (Tests grün beide Karten, Belege per tee in `~/.cache/bench-scripts/kdanorm_v2_logs_2026-11-25/*_final_*`).

## Nachtrag 25.11. ~10:00 — Stapel-Fixes und Fork-Abgleich (`fork-abgleich-2026-11-25.md`, `testauswertung-2026-11-25.md`)

- Fixups in den verursachenden Commits, Stapel neu aufgesetzt (Commitzahlen gleich): #1278 b66f59e70, #1282 d76b999db, #1347 35ed910b0,
  #1286 00ad05cb9, #1348 a88d81165, kv_b c70171be2, wo_a 153eccb0c, moe 8fde65a1b, route 82d4f95db. Alte Köpfe: scratchpad/w/oldheads.txt.
- Verlorene Fork-Merge-Auflösungen (iii), behoben: prefill_overlap.py max_seq_len_np (Absturz der TP-Prefill-Überlappung), parallel_state
  `_reject_overlapped_fused_norm` im registrierten Gemma-Norm-Op, Indexer-Kommentar fp32-Kachel, Indexer-Test-Keyword, Turing-Gated-Test,
  current_device_index ohne CUDA-Kontext (a1048ae98-Muster), CPU-Skips in GLM-Tests.
- GPU-Nachprüfung offen: test_deepseek_v4_sm70_indexer.py (V100/RTX), test_sm70_qpn8_block_fp8.py (RTX), dazu Gegenproben auf main
  test_cpu_offload.py (-k test_mrv2_weight_offloading) und test_cutlass_scaled_mm.py.
- Fork zieht nach (Ergänzung): skinny_sm70_moe.py `is_supported_config` ruft `current_device_index()` (legt CUDA-Kontext an) → Fassung aus
  dem Stapel übernehmen (`qpn_fragment_order(device_id=None)`).
- Nach den Fixups müssen die Python-Tests der Stapel (CPU/GPU) erneut laufen; Bauten bleiben gültig (nur .py geändert) — vor dem Wiederholen
  prüfen: `git diff --stat <alt> <neu> -- csrc cmake CMakeLists.txt` leer.

## Nachtrag 25.11. ~15:20 — Stapel auf main 28f321c2c (zweiter Rebase, Agent), von Opus geprüft

- Neue Köpfe: #1278 103afc1e0, #1282 54a2d5a35, #1347 e6c182681, #1286 8158afb52, #1348 cd06fac94, kv_b c4c48a9e7, wo_a 55b3a5e5a,
  moe 04782e959, route e67e11134 (alte: scratchpad/w/oldheads2.txt). Kein nativer Unterschied, kein Bau nötig; pre-commit + mypy grün;
  CPU-Fehlschläge nur main-gleich. Fragen und Entscheidungen: `stack-rebase2-fragen.md`.
- Geprüft und angenommen: (1) #1548 doppelt → mains Fassung bleibt, unser Commit wird reiner Test (main #1548 hat keinen Test);
  (2) Turing-Arbeitsbereich in #1278 Zeile für Zeile wie prod-v16; (3) `if quant_algo and quant_algo != "MXFP8"` in #1286 (Lader
  dequantisiert den MXFP8-Kopf, Kommentar sagt es), #1348 wieder mains Zeile; mains Test `test_preserves_unquantized_controls` erwartet
  für den Kopf jetzt `None` → im #1286-Update-Text nennen; (4) zwei verlorene Fork-Testanpassungen nachgezogen.
- Im Update-Text #1282 nennen: Flash-V100-Fix entfällt zugunsten #1548, Test bleibt.
- GPU-Nachprüfung (Liste im Agentenbericht): test_sm70_qpn8_block_fp8 (V100/RTX), qpn8_grouped_bmm (RTX), qpn8_mxfp8_linear, glm5next/*,
  modelopt/mxfp8-Quantisierungstests, model_runner-Tests, flash_v100_dense_prefill_availability, dazu test_deepseek_v4_sm70_indexer und
  main-Gegenproben test_cpu_offload / test_cutlass_scaled_mm / test_mla_backends (RTX). Neuer main-Baum `1Cat-vLLM-prmain2` (28f321c2c).

## Nachtrag 25.11. ~15:40 — unabhängige Zweige auf 28f321c2c fertig, main inzwischen 73da3445d (#1616 Inkling 2 Mini)

- Unabhängige Köpfe auf 28f321c2c (Agent, range-diff alle `=`, pre-commit/mypy grün, CPU-Fehler main-gleich): adaptive-verify 4018ddb54,
  pp-pack 21bf07f67, uva 18d5cdcdc, cutlass-int8 ac55e3b70, KDA e9fb3052d, #1221 f01e1c581. Offene Textpunkte: KDA-Text „#1549 RECHECK“
  erledigt; KDA-Sources „test_kda_glue 94 skipped“ ist falsch (124); Duplikatprüfungen nennen noch 7db135fa0.
- #1616 (73da3445d, 14:56) bringt einen Inkling2-DSpark-Speculator unter `spec_decode/dspark/speculator.py` und adaptive DSpark-Breiten im
  AsyncScheduler. Probe-Merge: 13 Zweige sauber, Konflikte nur #1221 und adaptive-verify.
- Entscheidungen (Opus): #1221: mains Dateien bleiben, unser generischer DSparkSpeculator zieht nach `dspark/block_drafter.py`,
  init_speculator wählt nach Entwurfstyp (Inkling2 → mains Klasse); mains `and not self._is_inkling2_dspark()` an den zwei Stellen in
  `_validate_v2_model_runner` wird durch unser allgemeines "dspark" überflüssig und entfällt (im Text nennen). adaptive-verify: mains
  Inkling2-Logik bleibt im Verhalten exakt und hat Vorrang, unser Controller ersetzt nur den EntMTP-Teil. Im PR-Text: Inkling2-Breiten sind
  modellspezifisch, unser Controller generisch (Abgrenzung, kein Duplikat).
- Zwei Agenten: Konfliktzweige (rebase3-fragen.md) und 13 saubere Zweige (rebase3b-fragen.md).

## Nachtrag 25.11. ~16:30 — Konfliktzweige auf 73da3445d (Agent + Opus)

- #1221 pr-mrv2-dspark-r1125 → 5e7fa7ca5 (3 Commits): unser Speculator in `dspark/block_drafter.py` (bytegleich zu prod-v16
  dspark/speculator.py), Weiche in init_speculator über `_is_inkling2_dspark()`, mains zwei Inkling2-Ausnahmen in
  `_validate_v2_model_runner` entfallen. Opus-Fixup: mains `_add_dspark_width_graphs` nur noch für den Inkling2-Entwurf (sonst hätten
  unsere DeepSeek-DSpark-Einträge Zielgraphen für nie genutzte schmalere Breiten aufgenommen); Test
  `test_narrower_width_graphs_only_for_the_drafter_that_narrows` rot ohne / grün mit. Commit-Texte 1 und 3 ergänzt.
- adaptive-verify → aeeac556d (4 Commits): mains Inkling2-Breiten mit Vorrang, Controller ersetzt nur EntMTP; `resolve_adaptive_verify`
  liefert für dspark None, beide schließen sich aus.
- pre-commit auf main 73da3445d selbst rot (Layering durch #1616-Dateien, runtime_reference.md veraltet, mypy 49 Zeilen); unsere Zweige
  bringen nichts Neues hinzu (Zähler gleich main bzw. schon in unserer Baseline). Keine Reparatur in unseren PRs; im Text erwähnen, falls
  CI rot ist.
- Fork zieht nach: Speculator-Datei umbenennen wie hier, Breitengraphen-Begrenzung.

## Nachtrag 25.11. 17:25 — alle 15 Zweige auf main 73da3445d, GPU-Runde 5 läuft

- 13 saubere Zweige (Agent): range-diff 87/87 `=`, Rücksetz-Check 0, nativ leer, pre-commit nur main-Rot, CPU 0 neue Fehlschläge.
- Befund (Agent, von Opus nachgeprüft und behoben): #1286-Commit „feat(glm53): load GLM-5.3-Flash-FP4-Local on SM70 and SM75“ liest in
  `ModelOptNvFp4FusedMoE.create_weights` `self.csf_scales`; `ModelOptNvFp4SM70MoEMethod.__init__` überspringt den Eltern-__init__ und
  setzt es nie → AttributeError beim Laden über den TurboMind-MoE-Weg (Inkling2 mit #1616 sofort, sonst latent). Fix: `self.csf_scales =
  False` im SM70-__init__, Test `test_nvfp4_sm70_moe_creates_uncompressed_expert_scales` (rot ohne / grün mit), Fixup im Commit, #1348/moe/
  route neu aufgesetzt (range-diff nur Commit 4 `!`). **Fork zieht nach: derselbe latente Fehler steckt in prod-v16.**
- Köpfe: #1278 4ab593ca3, #1282 e0f93d68f, #1347 f546c6761, #1286 918a96748, #1348 c7fcf8b50, kv_b 9b195aa1f, wo_a 0fffc9d8d, moe 68ba93b25,
  route 3ca101284, adaptive-verify aeeac556d, pp-pack 3548c2349, uva 5f58dd8e9, cutlass 459f937b4, KDA 7f9ae424e, #1221 5e7fa7ca5.
- GPU-Runde 5: `chain_prtest5_2026-11-25.sh` (Unit prtest5-chain, Hold), Listen `prtest_2026-11-25/lists5/` (alte Listen + Überschneidungsdateien
  zu #1616 und den main-Commits), main-Baum prmain2 mit der Vereinigung (77 Dateien), Logs `runs5/<baum>/{cpu,v100,rtx}/`.

## Nachtrag 26.11. 00:25 — GPU-Runde 5 fertig (main 73da3445d), alle 15 Zweige ohne neue Fehlschläge

- Kette 5 (17:21–19:01, Abbruch für GLM; Fortsetzung 20:01–00:20, Hold von Agent-Orc gewährt). Vergleich je Baum und Gerät Name für Name
  gegen main prmain2 (`prtest_2026-11-25/cmp_fail5.py`, Ergebnis `cmp5_result.txt`): 0 neue Fehlschläge auf V100 und RTX in allen 15
  Bäumen, 0 Hänger. CPU: CUTLASS 1 (unser neuer Test braucht CUDA, auf beiden GPUs grün: 3 statt 1 passed), #1282/#1347/#1286/#1348 je 17 in
  test_dsv4_sparse_prefill_bmm.py = dieselben 17 CPU-Fälle wie main („No CUDA GPUs“), nur mit dem zusätzlichen tensor_cores-Parameter im Namen.
- #1221: Opus-Fixup hatte 259 Fälle in test_sm70_long_attention_graphs.py rot gemacht (Attrappen ohne `_is_inkling2_dspark`); behoben, indem
  mains `spec is None or spec.method != "dspark"` vorn bleibt; neuer Kopf 7730427a0, Runde 5 auf diesem Kopf grün (Lauf vor dem Fix:
  runs5/pr-r-dspark.vor-fix2).
- test_cutlass_scaled_mm: auf main wie im Zweig 405 failed (schlanker Bau ohne INT8-Kerne) — im CUTLASS-Text so erklären.

## Nachtrag 26.11. 07:00 — #1627 (CUTLASS) und #1628 (KDA) gesendet; adaptive-verify und pp-pack nach Review korrigiert

- Gesendet nach zwei unabhängigen Prüfungen (Erstprüfung, Delta-Prüfung) und Live-Abgleich (Titel und Text gleich dem Entwurf):
  #1627 CUTLASS INT8 Build-Abfrage (Zweig pr-cutlass-int8-build-query-v2 ef43e2f70; Tests auf diesem Kopf, runs5/pr-cutlass-int8),
  #1628 KDA-Decode-Kleber (Zweig pr-glm5next-kda-decode-glue d2a0ff732; Baum = e05e84b10, Layer-Test 128 passed auf V100/RTX,
  runs5/pr-glm5next-kda-glue-e05e84b10; 8-Dateien-Lauf auf 7f9ae424e).
- adaptive-verify (Prüfung 26.11.): Code geändert. Feld `max_tokens` entfernt (wirkungslos, erzeugte nie genutzte Graphen),
  `verify_cost_ratio` ohne Platzhalter-Standard (Durchsatzregel verlangt Messwert), Fehler bei ngram_assist und bei synchronem Scheduler
  (`check_adaptive_verify_scheduling` in VllmConfig), Tests der Scheduler-Verdrahtung, Kommentare ohne Fork-Namen/DESIGN.md. Kopf 6ddaf6b53
  + Fixup (Kommentar output.py ohne Modellnamen, ruff-Format) → noch einsortieren. Die 4 mypy-Fehler der #1616-DSpark-Zeilen in
  async_scheduler.py bleiben (auch auf main, 1Cats Code).
- pp-pack (Prüfung 26.11.): neuer Commit 88707c696: unzulässige pp_pack_handover-Einstellung (PP1, TP>1, DP>1, V1, VLLM_V41_PP_RANK0)
  ist ein ValueError beim Worker-Start statt still ignoriert. Textbefunde B1/B2/S2–S5/K1–K8 offen.
- Gestapelte neue PRs (wo_a, kv_b, moe, route): Texte fertig (Agent), noch nicht unabhängig geprüft; kv_b-Numerik mit sm7x-Triton läuft
  (`chain_pp_kvb.sh`); Senden erst nach den Force-Pushes der Basis-PRs.
- Peuqui 26.11. ~07:45: Sobald alle PRs raus und alles abgeschlossen ist, die Platte gründlich aufräumen (Arbeitsbäume gesendeter PRs, Testläufe, Scratch; Zweige im Fork bleiben, readlink -f vor jedem Löschen, keine Produktionsbäume/-caches).

## Nachtrag 26.11. 09:45 — sechs Updates gesendet (#1278, #1282, #1347, #1286, #1348, #1221)

- Force-Push mit Lease (alt → neu): pr-turing-kernels c15d1446f → 4ab593ca3, pr-triton-sm7x b9fbe637c → e0f93d68f, pr-dsv4-longctx-decode
  6968a8b40 → f546c6761, pr-glm53 75fd75d70 → 918a96748, pr-glm53-prefill-head ceceb328e → c7fcf8b50, pr-mrv2-dspark 450e65c52 → 7730427a0.
  Beschreibungen (update-<nr>-body.txt) und Kommentare (update-<nr>-comment.txt) live zeichengleich geprüft; alle MERGEABLE.
- Prüfung: unabhängiger Agent (Push-Sicherheit aller sechs belegt; Textbefunde A1–A6 und je PR) → Korrektur-Agent (Runde 2 in update-<nr>.changes.txt).
- sm7x-Triton-Lauf aller Stapelzweige (`chain_sm7x_triton.sh`, runs5/<baum>-sm7x-triton): keine Fehlschläge.
- Offen: adaptive-verify (Kopf 8bd4a4f2c, Endlauf V100/RTX wartet in der Kette), danach pp-pack (Platzhalter ADAPTIVE_PR), dann die vier gestapelten
  PRs (Code-Fixes: moe-Benchmark ohne eigene Pfade 71fc439bd, moe-Kommentar 63–67 %, wo_a-Kommentar 780f8f535, kv_b ab zwei Zeilen cfb6bf382; GPU-Lücken
  `chain_stack_gaps.sh`, `chain_stack_after_fix.sh`; Textbefunde des Prüfers offen; moe-DeepSeek-Frage offen).

## Nachtrag 26.11. 10:10 — #1629 (adaptive-verify) und #1630 (pp-pack) gesendet

- #1629: Zweig pr-adaptive-verify 8bd4a4f2c (nach Nachprüfung: toter V1-Zweig raus, VllmConfig-Test, Kommentare ohne Modellnamen); Endlauf
  runs5/pr-adaptive-verify-final (6 Dateien × CPU/V100/RTX, 0 neu gegen main), übrige 13 Dateien auf 6ddaf6b53; pre-commit belege/pc6_adaptive-verify.log.
  Text live gleich, Querverweis auf #1630 nachgetragen (gh pr edit, live gleich).
- #1630: Zweig pr-pp-pack-handover 5b3f76805 (drei Commits; Startprüfung im Worker, Tests 23 auf CPU/V100/RTX); Text live gleich.
- Merge-Sonden beider Endköpfe gegen alle 74 offenen Köpfe und die eigenen: belege/probe_all_open_2026-11-26b.json.
- Offen: vier gestapelte PRs (Texte beim Agenten, danach Delta-Prüfung, dann senden), Issue #674, Aufräumen der Platte, STAND/Übergabe/Memory.
- Nach der Runde (Peuqui 26.11.): GLM-Präfix-Cache untersuchen — trifft nur 54.528 von 62.372 Token (TTFT 4,04 s gegen DeepSeek 0,64 s),
  vermutlich KDA-Schnappschussgrenzen; Peuqui bevorzugt GLM für Recherche. Optional danach deutscher Qualitätsvergleich DeepSeek gegen GLM.

## Nachtrag 26.11. 11:20 — vier gestapelte PRs gesendet: #1631 (wo_a, 780f8f535), #1632 (kv_b, 4a26d56d7), #1633 (moe, 50c215daf), #1634 (route, b42b25b8d)

- Nach Erst- und Delta-Prüfung; Code-Fixes: moe-Benchmark ohne eigene Pfade (Pflichtargumente, Histogrammformat, Beispiel), Kommentar 63–67 %;
  wo_a-Kommentar mit Messwerten; kv_b Dispatch erst ab zwei Zeilen (KV_B_SKINNY_MIN_ROWS) + Kommentar/Docstring; route nur Trailer.
- GPU-Lücken geschlossen: wo_a-Aufrufer (runs5/pr-n-woa-callers = main), route test_skinny_sm70_moe (runs5/pr-n-route-moelayer), kv_b mit sm7x-Triton
  37/37 (runs5/pr-n-kvb-final-sm7x), sm7x-Kette aller Stapelzweige ohne Fehlschlag. pre-commit Endköpfe: belege/pc6_n-woa, pc9_n-kvb, pc9_n-moe.
- Alle Texte live gleich dem Entwurf. Damit ist die PR-Runde gesendet: neu #1627–#1634, aktualisiert #1278 #1282 #1347 #1286 #1348 #1221.
- Offen: Issue #674 (Entwurf beim Agenten), Black-Week-Auswertung (GLM ×2 gegen DeepSeek), Fork-Nachzug, GLM-Präfix-Cache, Aufräumen, STAND/Übergabe.

## Nachtrag 26.11. 12:10 — Issue #674 neu gesetzt, Runde abgeschlossen

- Übersicht #674 neu gesetzt (Text `issue674-body.txt`, Quellen `issue674.sources.txt`), live gleich dem Entwurf.
- CI aller PRs rot = dieselben Hooks, an denen 1Cat main selbst seit #1616 scheitert. Peuqui 26.11.: 1Cat repariert sein main selbst,
  wir legen nur offen, dass main gleich rot ist.
- Offen nach der Runde: Fork-Nachzug (Abschnitt „Fork zieht nach“, dazu csf_scales-Fix aus #1286 — latent in prod-v16 —, kv_b ab zwei Zeilen,
  DSpark block_drafter-Umbenennung + Breitengraph-Gate, adaptive-verify-Endfassung, pp-pack-Startprüfung, UVA-Paket mit B4 nbytes==0,
  B5 Finalizer, B6 created-Flag, GLM-Boot), GLM-Präfix-Cache, Aufräumen der Platte.
