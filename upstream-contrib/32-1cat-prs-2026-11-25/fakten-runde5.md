# Faktenblatt Runde 5 (Stand 26.11.2026 00:45) — Grundlage für alle PR-Texte und Update-Kommentare

Repo: Remote `origin` = 1CatAI/1Cat-vLLM, `fork` = Peuqui/1Cat-vLLM. Haupt-Checkout /home/mp/Projekte/vllm-research/1Cat-vLLM (Zweige
dort sichtbar), Arbeitsbäume /home/mp/Projekte/vllm-research/1Cat-vLLM-<baum>.

## Basis

- Alle Zweige liegen auf 1Cat main 73da3445d (#1616 "Port Inkling 2 Mini to Volta ..."). main steht inzwischen auf 44708bdb9
  (+ #1625 Inkling2-Kerne, #1624 Qwen4-QSA); diese zwei Commits berühren keine Datei unserer Zweige (geprüft je Zweig mit
  `comm` der Dateilisten) und mergen sauber (`git merge-tree`). Kein weiterer Rebase.
- Nativ (csrc, cmake, CMakeLists.txt, setup.py) hat main seit 7db135fa0 nichts geändert.

## Zweige, Köpfe, Bäume, PR

| Zweig (lokal) | Kopf | Baum | PR | Fork-Zweig des PR (Head) | Commits |
|---|---|---|---|---|---|
| pr-turing-kernels-r1125 | 4ab593ca3 | pr-r-turing | #1278 | pr-turing-kernels (c15d1446f) | 8 auf main |
| pr-triton-sm7x-r1125 | e0f93d68f | pr-r-triton | #1282 | pr-triton-sm7x (b9fbe637c) | 28 auf #1278 |
| pr-dsv4-longctx-decode-r1125 | f546c6761 | pr-stack | #1347 | pr-dsv4-longctx-decode (6968a8b40) | 3 auf #1282~2 |
| pr-glm53-r1125 | 918a96748 | pr-r-glm53 | #1286 | pr-glm53 (75fd75d70) | 31 auf #1282~2 |
| pr-glm53-prefill-head-r1125 | c7fcf8b50 | pr-r-glm53ph | #1348 | pr-glm53-prefill-head (ceceb328e) | 3 auf #1286~3 |
| pr-mrv2-dspark-r1125 | 7730427a0 | pr-r-dspark | #1221 | pr-mrv2-dspark (450e65c52) | 3 auf main |
| pr-mla-kv-b-skinny | 9b195aa1f | pr-n-kvb | neu | – | 1 auf #1282 |
| pr-dsv4-wo-a-turing-fp8 | 0fffc9d8d | pr-n-woa | neu | – | 2 auf #1278 |
| pr-sm70-moe-kernel-pack | 68ba93b25 | pr-n-moe | neu | – | 5 auf #1286 |
| pr-sm70-moe-routing-sort-v2 | 3ca101284 | pr-n-route | neu | – | 1 auf #1286 |
| pr-adaptive-verify | aeeac556d | pr-adaptive-verify | neu | – | 4 auf main |
| pr-pp-pack-handover | 3548c2349 | pr-pp-pack | neu | – | 2 auf main |
| pr-uva-shared-host-memory | 5f58dd8e9 | pr-uva | neu | – | 1 auf main |
| pr-cutlass-int8-build-query-v2 | 459f937b4 | pr-cutlass-int8 | neu | – | 1 auf main |
| pr-glm5next-kda-decode-glue | 7f9ae424e | pr-glm5next-kda-glue | neu | – | 1 auf main |

Gestapelte neue PRs zeigen in GitHub die Commits ihrer Basis mit; der Text sagt, auf welchem offenen PR sie liegen und welche Commits
die eigenen sind.

## Was sich seit dem letzten Push der offenen PRs geändert hat (für die Update-Kommentare)

Gemeinsam: Rebase auf main 73da3445d (vorher 9. bzw. 23.11.), Autor überall Peuqui noreply. Range-diff alt (Fork-Head) gegen neu selbst
erzeugen (`git range-diff <alt-basis>..<fork-head> origin/main..<neu>` bzw. mit der jeweiligen Stapelbasis) und jede `!`-Abweichung
benennen. Bekannte inhaltliche Änderungen:
- #1278: Turing-Zweig legt den Block-FP8-Prefill-Arbeitsbereich nach mains #1535 nur bei aufgenommenem Prefill an
  (`_graphs_capture_dense_prefill(_SM75_MMA_MAX_M)`); Turing-Wächter vor mains Gated-SiLU-Weg (#1278-Test
  `test_block_qpn8_gate_up_fuses_the_gated_route_except_on_turing`); `qpn8_native_linear_out` hat `sm75_mma` (mains Direktausgabe).
- #1282: unser Fix "check the dense prefill entry without binding it" ist durch mains #1548 überholt; mains Fassung bleibt, von unserem
  Commit bleibt nur der Test (`test(flash_v100): cover the dense prefill availability check before the runtime is bound`). Indexer-Epilog
  (BLOCK_K 512 / 2 Warps) jetzt auf mains fp32-Score-Kachel: nachgemessen, Zahlen in `stack-rebase-fragen.md` Entscheidung 1
  (Logs ~/.cache/bench-scripts/bench_dsv4_idx_epilogue_fp32_2026-11-25_gpu{2,3}.log selbst öffnen und jede Zahl daraus nehmen).
  Verlorene Merge-Auflösungen wiederhergestellt (prefill_overlap.py max_seq_len_np, `_reject_overlapped_fused_norm` im registrierten
  Gemma-Norm-Op, current_device_index ohne CUDA-Kontext); Details `fork-abgleich-2026-11-25.md`.
- #1347: nur Rebase (range-diff `=`), prüfen.
- #1286: mains #1541 wirft bei unbekanntem quant_algo; #1286 nimmt MXFP8 davon aus (`if quant_algo and quant_algo != "MXFP8"`, der Lader
  dequantisiert einen MXFP8-LM-Kopf), mains Test `test_preserves_unquantized_controls` erwartet für den LM-Kopf jetzt `None` (eigene
  UnquantizedEmbeddingMethod); `stack-rebase2-fragen.md` Punkt 3. Neu: `ModelOptNvFp4SM70MoEMethod.__init__` setzt `csf_scales = False`
  (vorher AttributeError in `create_weights` auf dem TurboMind-MoE-Weg, mit #1616 für Inkling 2 sofort), Test
  `test_nvfp4_sm70_moe_creates_uncompressed_expert_scales`. kda.py: mains #1549-Bindung bleibt, unsere doppelte Zeile entfällt.
- #1348: in diesem PR bedient der MXFP8-Zweig alle Schichten inklusive LM-Kopf, dort wieder mains `if quant_algo:`.
- #1221: mains #1616 hat einen Inkling2-DSpark-Speculator unter `spec_decode/dspark/speculator.py`. Unser generischer DSparkSpeculator liegt
  jetzt in `dspark/block_drafter.py`, init_speculator wählt nach Entwurf (Inkling2-Entwurf → mains Klasse), zwei Inkling2-Ausnahmen in
  `_validate_v2_model_runner` entfallen (durch allgemeines "dspark" überflüssig), mains `_add_dspark_width_graphs` bleibt beim
  Inkling2-Entwurf (unser Speculator hat eine feste Breite; sonst nie abgespielte Graphen), Test
  `test_narrower_width_graphs_only_for_the_drafter_that_narrows`. Details `rebase3-fragen.md`.
- adaptive-verify (neu): mains Inkling2-DSpark-Breiten im AsyncScheduler bleiben unverändert und haben Vorrang; unser Controller ersetzt
  nur den EntMTP-Async-Zustand; für method "dspark" gibt `resolve_adaptive_verify` None (beide schließen sich aus). Abgrenzung im Text:
  mains Breiten sind Inkling2-spezifisch, unser Controller generisch.

## Testläufe auf genau diesen Köpfen (Runde 5, 25./26.11.2026)

- Skript ~/.cache/bench-scripts/pr_test5_2026-11-25.sh, Kette chain_prtest5_2026-11-25.sh; Listen
  ~/.cache/bench-scripts/prtest_2026-11-25/lists5/<baum>.txt; Ergebnisse runs5/<baum>/{cpu,v100,rtx}/summary.log und je Datei
  `<pfad_mit_unterstrichen>.log`. Geräte: cpu = CUDA_VISIBLE_DEVICES leer; v100 = Tesla V100-PCIE-32GB (PCI-Index 1, sm_70); rtx = Quadro RTX
  8000 (PCI-Index 0, sm_75). Eine pytest-Instanz je Datei, Zeitlimit 25 min, 0 Hänger.
- Köpfe beim Lauf: Kettenlogs prtest_2026-11-25/chain5.log und chain5b.log (Zeile "##### <baum> <kopf> ..."). Für pr-r-dspark gilt
  runs5/pr-r-dspark (Kopf 7730427a0); runs5/pr-r-dspark.vor-fix2 ist ein überholter Lauf.
- main-Vergleich: runs5/prmain2 (main 73da3445d, alle drei Geräte) und runs5/prmain2-indep/cpu (zusätzliche CPU-Dateien).
- Vergleich Name für Name: prtest_2026-11-25/cmp_fail5.py, Ergebnis cmp5_result.txt: auf V100 und RTX 0 neue Fehlschläge in allen 15
  Bäumen. CPU: CUTLASS 1 (unser neuer Test braucht CUDA; auf V100 und RTX bestanden: test_cutlass_scaled_mm 3 passed statt 1 auf main,
  405 failed auf main und Zweig gleich, weil der Bau keine INT8-CUTLASS-Kerne enthält — selbst im Log prüfen und genau so belegen);
  #1282/#1347/#1286/#1348 je 17 in test_dsv4_sparse_prefill_bmm.py = dieselben 17 CPU-Fälle wie main ("No CUDA GPUs are available"),
  nur mit zusätzlichem tensor_cores-Parameter im Testnamen.
- Jede Zahl im Text (passed/failed/skipped je Datei und Gerät) direkt aus dem jeweiligen Datei-Log nehmen. Fehlschläge, die main
  genauso hat, als "same on main 73da3445d" mit Ursache aus dem Log nennen (z. B. "No CUDA GPUs are available", World size > GPUs).

## pre-commit

- main 73da3445d ist selbst rot: check-layering (33 Dateien aus #1616), check-architecture-reference / test-architecture-reference
  (runtime_reference.md auf main veraltet), mypy (model_runner.py `try_sample_target`, async_scheduler.py, u. a.), ruff/markdownlint an
  #1616-Dateien. Unsere Zweige fügen keinen Befund hinzu (Logs im Scratchpad
  /tmp/claude-1000/-home-mp-Projekte-vllm-research/d820eb98-aa5b-4b03-9979-706846eee66f/scratchpad/w/pc3b_*.log, pccmp3b_result.txt,
  mypy_*.txt, pc_b2.txt). Im Text ehrlich: welche Hooks liefen, dass die roten auf main genauso rot sind, und dass der Zweig keine
  zusätzliche Meldung bringt.

## Regeln für die Texte (AGENTS.md von 1Cat und Peuquis Regeln)

- Englisch, ohne Markdown-Auszeichnung (keine Überschriften mit #, keine **, keine Backticks, keine Tabellen mit |). Einfache Zeilen mit
  "- " sind erlaubt.
- Pflicht: warum kein Duplikat (gh-Suche auf 1CatAI/1Cat-vLLM frisch ausführen, offene UND gemergte PRs, Suchbegriffe nennen), welche
  Test-Kommandos mit welchen Ergebnissen, klarer KI-Hinweis (AI assistance: Claude Code; Peuqui reviewed and tested).
- Jede Zahl aus einer Messdatei, mit Quelle in der .sources.txt (Datei:Zeile). Keine Schätzungen. Was nicht gemessen ist, nicht behaupten.
- Keine Verkaufs- oder Gewinnangaben.
- Am Ende jeder neuen PR-Beschreibung (wie bei #1536) genau diese zwei Zeilen, durch eine Leerzeile getrennt:
  "🤖 Generated with [Claude Code](https://claude.com/claude-code)" und "https://claude.ai/code/session_0172gtUkqNsK9nE7L151CUUE".
  Update-Kommentare zu bestehenden PRs enden mit dem KI-Hinweis, ohne diese zwei Zeilen.
- Commit-Trailer der neuen Commits sind schon gesetzt; nichts an Commits ändern.
