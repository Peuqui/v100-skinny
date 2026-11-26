# Abgleich der PR-Zweige gegen den Fork prod-v16 (25.11.2026)

Stand: nach den Korrekturen und dem Neuaufsetzen vom 25.11.2026. Basis der Zweige: `pr-base-7db135fa0` (1Cat main 7db135fa0). Fork: `prod-v16` e336f616e (liegt auf 1Cat main 2d218ab03, 16 main-Commits dahinter). Gelesen wurde je Zweig und Datei `git diff -U0 prod-v16 <zweig> -- <datei>` für jede Datei, die der Zweig gegenüber der Basis ändert.

## Verfahren und Einordnung

Ein Hunk ist ein Block von `git diff -U0` (ohne Kontextzeilen). Je Hunk wurden die Fork-Zeilen mit `git blame prod-v16` und die Zweig-Zeilen mit `git blame <zweig> ^pr-base-7db135fa0` auf Commits zurückgeführt; Fork-Commits wurden anhand ihrer hinzugefügten Zeilen den Stapel-Commits zugeordnet (Zwillinge, 63 von 110 per Betreff, weitere per Zeileninhalt; die übrigen Stapel-Commits haben keinen Fork-Zwilling: sie sind neuer als der Fork). Zeilen, die im Hunk auf der einen Seite stehen und anderswo derselben Datei auf der anderen, gelten als verschoben. Zusätzlich wurden alle Auflösungs-Diffs der 16 Fork-Merges (`git show --remerge-diff`) gegen die Stapel-Dateien gelesen und alle Fork-Commits, die Stapel-Dateien anfassen, einzeln eingeordnet (Abschnitt unten).

- (i) Fork-Kontext außerhalb dieses Zweigs: Inhalt eines anderen PR des Stapels (Fork-Zwilling dessen Commits) oder ein reiner Fork-Commit; Beleg: Fork-Commits in der Spalte Beleg.
- (ii) main-Änderung seit 2d218ab03, die der Fork noch nicht hat. Nur sechs Stapel-Dateien sind von den 16 main-Commits berührt: `CMakeLists.txt`, `tests/v1/worker/test_qwen4_exp_v2.py`, `vllm/_custom_ops.py`, `vllm/v1/attention/backends/flash_v100/dense_prefill.py`, `.../flash_v100/ops.py`, `vllm/v1/worker/gpu/attn_utils.py` (dort 0 Hunks).
- (iii) verlorene Fork-Auflösung: der Fork hat eine Anpassung an main, die dem Zweig fehlte. Alle Fälle stehen unten und sind behoben; die Spalte zeigt, was nach der Korrektur noch als (iii) zählt (nur die Testform in `test_sm70_qpn8_block_fp8.py`, wo der Stapel jetzt prüft statt wie der Fork zu überspringen).
- (iv) sonstiges: Stapel-Commit ohne Fork-Zwilling oder neuere Stapel-Fassung desselben Commits (Review-Korrekturen, clang-format, Wortlaut, `torch.accelerator`), verschobene Blöcke, generierte Doku.

## Summe je Zweig (Hunks, nach der Korrektur)

Einheit: Hunks (`-U0`). Gezählt: alle Dateien des Zweigs; Dateien ohne Hunk sind byte-gleich zum Fork.

| Zweig | neuer Kopf | alter Kopf | Dateien | davon identisch zum Fork | Hunks | (i) | (ii) | (iii) | (iv) |
|---|---|---|---|---|---|---|---|---|---|
| pr-turing-kernels-r1125 (#1278, Baum pr-r-turing) | b66f59e70 | cc4751cf0 | 20 | 5 | 185 | 173 | 1 | 2 | 9 |
| pr-triton-sm7x-r1125 (#1282, pr-r-triton) | d76b999db | b4cc63c10 | 75 | 36 | 301 | 263 | 6 | 2 | 30 |
| pr-dsv4-longctx-decode-r1125 (#1347, pr-stack) | 35ed910b0 | fbc1b58e5 | 73 | 41 | 259 | 226 | 6 | 2 | 25 |
| pr-glm53-r1125 (#1286, pr-r-glm53) | 00ad05cb9 | 8b85abcc2 | 114 | 52 | 389 | 316 | 7 | 2 | 64 |
| pr-glm53-prefill-head-r1125 (#1348, pr-r-glm53ph) | a88d81165 | de68acd7c | 115 | 54 | 385 | 283 | 7 | 2 | 93 |
| pr-mla-kv-b-skinny (pr-n-kvb) | c70171be2 | b7e3e8480 | 79 | 38 | 308 | 264 | 6 | 2 | 36 |
| pr-dsv4-wo-a-turing-fp8 (pr-n-woa) | 153eccb0c | d20326a7e | 25 | 8 | 170 | 143 | 1 | 2 | 24 |
| pr-sm70-moe-kernel-pack (pr-n-moe) | 8fde65a1b | c6b63b32f | 115 | 55 | 296 | 202 | 7 | 2 | 85 |
| pr-sm70-moe-routing-sort-v2 (pr-n-route) | 82d4f95db | a348b5440 | 115 | 53 | 385 | 309 | 7 | 2 | 67 |

## Fälle (iii), vollständig

| Nr | Datei | Befund | Fork-Beleg | Behoben in |
|---|---|---|---|---|
| 1 | `vllm/v1/worker/gpu/prefill_overlap.py` | `max_seq_len_np=requests(input_batch.max_seq_len_np)` im `InputBatch`-Aufbau; das Feld gibt es auf main nicht mehr | Fork c483c516c (Backport vllm#54436) entfernt Feld und Zeile | #1282, fixup in 4ec816be4 (Teil 1a) |
| 2 | `vllm/distributed/parallel_state.py` | `all_reduce_gemma_rms_norm` (von main 119cae740 eingeführter registrierter Op) erreicht die fused Op ohne `group._reject_overlapped_fused_norm()`; die Out-of-place-Wrapper haben den Wächter, dieser Weg nicht, die TP-Prefill-Überlappung könnte dort mit der fused Norm um den Kommunikator rennen | Fork-Merge 8018abc2d (Auflösung fügt die Zeile ein) | #1282, fixup in 4ec816be4 (neu gefunden) |
| 3 | `vllm/models/deepseek_v4/sm70/indexer.py` | Kommentar zu `_PREFILL_EPILOGUE_BLOCK_K` beschreibt eine fp16-Score-Kachel, main (1Cat #1328) schreibt sie in fp32 | Fork d52068f23 (»only its comment changes«) | #1282, fixup in 72124bf81 (neu gefunden; nur Kommentar) |
| 4 | `tests/kernels/test_deepseek_v4_sm70_indexer.py` | `sm70_indexer_prefill_logits(...)` ohne Pflicht-Keyword `triton_tensor_cores` im Test, den main erst nach der Zusammenführung brachte | Fork-Merge 8018abc2d und d52068f23 (`triton_tensor_cores=False`) | #1282, fixup in 6c0a85448 (Teil 1b) |
| 5 | `tests/kernels/quantization/test_sm70_qpn8_block_fp8.py` | Gated-Test verlangt die Gated-Route auch auf Turing | Fork-Merge 8018abc2d (Auflösung: Skip auf Turing) | #1278, fixup in cc4751cf0 (Teil 1c); der Stapel prüft statt zu überspringen |
| 6 | `vllm/model_executor/kernels/linear/qpn/fp8_block.py`, `tests/quantization/test_block_fp8_qpn8_capabilities.py` | `can_implement` fragt `torch.accelerator.current_device_index()` und legt damit einen CUDA-Kontext an; Test stubbte das | Fork a1048ae98 | #1278, fixup in f6876432d (Teil 1d) |
| 7 | `vllm/models/glm5next/nvidia/kda.py` | gleiche Ursache im KDA-Layer | Fork a1048ae98 | #1286 fixup in 792eed4ab, #1348 fixup in a8d404cdd (Teil 1d) |
| 8 | `vllm/model_executor/layers/fused_moe/experts/skinny_sm70_moe.py` | `is_supported_config` fragt `current_device_index()`; 12 CPU-Fälle rot. Der Fork hat dieselbe Stelle unverändert (auch dort CPU rot), also kein Fork-Vorbild; gelöst nach dem Muster von a1048ae98 | (a1048ae98 Muster) | #1278, fixup in 99d9f05c4 (Teil 1d) |

Zusätzlich Teil 1e (reine CPU-Hygiene, nicht im Fork): `tests/models/glm5next/test_sm70_sparse.py` (zwei Tests, fixups in 666cc386f/a2c21a594 und den #1348-Kopien 8476e6394/1189d0206) und `tests/models/glm5next/test_vision_row_chunks.py` (fixup in 8b85abcc2, nur #1286).

## Merge-Auflösungen des Forks (`git log --merges 2d218ab03^..prod-v16`, 16 Merges)

| Merge | Auflösung berührt Stapel-Dateien | Ergebnis |
|---|---|---|
| 1ddebbbd7 GLM way B auf Turing-Zweig | `test_csf32_scales.py`, `skinny_sm70_moe.py`, `csf32_expand.py` | `test_csf32_scales.py`, `csf32_expand.py`: 0 Hunks (gleich). `skinny_sm70_moe.py` (Quelle und Test): (i) für Inhalt von #1286/moe, (iv) für die kontextfreie Fassung von `qpn_fragment_order()` und Stapel-Wortlaut |
| 4d8ee343d Produktionsbasis in GLM-Decode | `skinny_sm70_moe.py` (Import) | Import: (i) (Fork-Inhalt von #1286/moe); kein (iii) |
| 624de3acd main 6a143197e | `runtime_reference.md`, `vllm/config/kernel.py`, `mamba_hybrid.py` | `mamba_hybrid.py`: 0 Hunks; `kernel.py`: nur (i) (adaptive_verify, pp_pack, PLE) und ein (iv)-Wortlaut; Referenz je Zweig geprüft |
| 012c9758b main b2b6b3bb9 | `test_deepseek_v4_sm70_indexer.py`, `test_mla_sparse_profile_budget.py`, `config/attention.py`, `mla_attention.py`, `flash_attn.py` | `attention.py`, `flash_attn.py`, `test_mla_sparse_profile_budget.py`: 0 Hunks; `mla_attention.py`: nur (i); Indexer-Test siehe Fall 4 |
| 8018abc2d main 119cae740 | `torch_bindings.cpp`, `runtime_reference.md`, `test_sm70_qpn8_block_fp8.py`, `test_deepseek_v4_sm70_indexer.py`, `parallel_state.py`, `fp8_block.py`, `indexer.py`, `deepseek_v41/attention_ops.py`, `glm5next/nvidia/multimodal.py` | Fälle 2, 3, 4, 5 (Wächter, Kommentar, Testkeyword, Turing-Skip); `attention_ops.py`: 0 Hunks; `torch_bindings.cpp`: nur (i); `fp8_block.py`: Turing-Wächter und `sm75_mma`-Pfade gleich, Rest (i)/(iv); `multimodal.py`: verschoben/(i) |
| c19c50908 main 4f565bae3 (Audit-Batch #1527) | `host_reduce.cu`, `host_reduce.py`, `test_host_reduce.py`, `vllm/config/kernel.py`, `fp8_block.py`, `runtime_reference.md` | `host_reduce.cu`, `test_host_reduce.py`: 0 Hunks; `host_reduce.py`: (i) (Fork-Log 58415430e); `kernel.py`: s. o.; `fp8_block.py`: (i) |
| 37e7797db main b7baf7771 | `runtime_reference.md` | (iv), Referenz grün |
| 31f2937aa main 2d218ab03 | `attn_utils.py`, `model_states/default.py`, `model_states/mamba_hybrid.py`, `runtime_reference.md` | `attn_utils.py`, `mamba_hybrid.py`: 0 Hunks (ubatch_id und main-Argumente beide da); `default.py`: ein (i)-Hunk (Fork 62b127eab) |
| 4b995ea8d Stale-Tests #1506 | `test_sm70_glm53_modelopt_nvfp4.py` | nur ein (iv)-Hunk (Stapel-Test 792eed4ab) |
| 608e9b19b moe-kernel-pack in v15 | `benchmark_glm_skinny_moe_routing.py` (nur #moe) | (iv) (clang-format, torch.accelerator) |
| 859833396, 0991b1656, 2a32f5573, 869c50ad8, cb63a9f83, e51b3b877 | keine Stapel-Dateien | nichts zu prüfen |

## Fork-Commits, die Stapel-Dateien anfassen und keine Zwillinge sind (einzeln eingeordnet)

- Eigene Fork-/PR-Arbeit, anderer Inhalt: c504ce162 (adaptive_verify), a43e966b1 und df927238c (pp-pack), c071cad10 (PLE-Zeilencache), 58415430e (host_reduce-Log), 0f039a5f8 (DSpark), 62b127eab (Kontext-Buckets), 508ac9f8f (Index-Kopie je Schritt), b817aa185 und 0ef69f4e6 (GLM-KDA, Router), 6597ec462 und 0a694f29f (FP8-Arbeitsbereich), b6583fb19 (GLM-Bildcache), 2f70712b0 (CMake-Kommentar) → (i) bzw. (iv).
- Anpassungen an main: d39f2b85b (DSv4.1-Aufrufer, ubatching-Docstring, ops.h): `attention_ops.py` und `ubatching.py` im Stapel gleich bzw. Wortlaut, `ops.h` Format; a1048ae98 → Fälle 6 bis 7; d52068f23 → Fälle 3, 4; c483c516c → Fall 1; c36116de9 → (iv).
- Jeder dieser Commits erscheint in den Tabellen unten als Beleg bei den Hunks, die er erzeugt.

## Je Datei und Zweig

Spalten: Hunks gesamt; (i)/(ii)/(iii)/(iv); Beleg. Beleg nennt die Fork-Commits (kurze SHA und Betreff) der (i)-Hunks, bei (iv) den Grund.

### pr-turing-kernels-r1125 (#1278, Baum pr-r-turing) (Kopf b66f59e70)

| Datei | Hunks | i | ii | iii | iv | Beleg |
|---|---|---|---|---|---|---|
| `CMakeLists.txt` | 4 | 2 | 1 | 0 | 1 | db216c2 perf(glm53): expand csf_32x2_v1 expert scales... · f9ac81b perf(attention): Turing paged prefill attenti... · 7d83b70 perf(dsv4): fused lightning-indexer prefill s... · ii: main-Änderung seit 2d218ab03 · iv: CMake-Kommentar (Fork 8515f1cc1/2f70712b0), kein Verhalten |
| `csrc/ops.h` | 1 | 1 | 0 | 0 | 0 | f9ac81b perf(attention): Turing paged prefill attenti... · afe87ab perf(turing): split-KV in the paged prefill a... · 7d83b70 perf(dsv4): fused lightning-indexer prefill s... |
| `csrc/sm70_turbomind/ops/fp8_qpn8_codes_sm70.cuh` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `csrc/sm70_turbomind/ops/fp8_qpn8_mma_sm75.cu` | 14 | 14 | 0 | 0 | 0 | c213a33 perf(sm75): Tensorkern-GEMM für MXFP8-Zeileng... |
| `csrc/sm70_turbomind/ops/fp8_qpn8_sm70.cu` | 45 | 45 | 0 | 0 | 0 | 768d1a3 feat(sm70): QPN8-Skalenmodus je Spalte und 32... · 8cbf9c1 style(sm70): clang-format our lines in fp8_qp... · 6b8d419 perf(qpn8): DeepSeek-V4 wo_a on Turing straig... |
| `csrc/sm70_turbomind/ops/mma_fragments_sm70.cuh` | 1 | 1 | 0 | 0 | 0 | f9ac81b perf(attention): Turing paged prefill attenti... |
| `csrc/sm70_turbomind/ops/nvfp4_qpn4_sm70.cu` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `csrc/sm70_turbomind/ops/skinny_moe_qpn_sm70.cu` | 38 | 38 | 0 | 0 | 0 | cab7b54 feat(moe): six-bit expert scale planes for th... · e6483cf perf(glm53): decode reads the expert scales a... · de1405f perf(moe): explicit load ring in the Volta de... |
| `csrc/torch_bindings.cpp` | 2 | 2 | 0 | 0 | 0 | cab7b54 feat(moe): six-bit expert scale planes for th... · db216c2 perf(glm53): expand csf_32x2_v1 expert scales... · 31f9613 perf(glm53): write the CSF exception records ... |
| `tests/kernels/moe/test_skinny_sm70_moe.py` | 9 | 9 | 0 | 0 | 0 | 64cca01 fix(glm53): no csf entries needed once every ... · cab7b54 feat(moe): six-bit expert scale planes for th... · e6483cf perf(glm53): decode reads the expert scales a... |
| `tests/kernels/quantization/test_sm70_qpn8_block_fp8.py` | 9 | 5 | 0 | 2 | 2 | iii: Fork-Auflösung 8018abc2d (Turing-Skip des Gated-Tests); im Stapel als Turing-Prüfung ersetzt (fixup cc4751cf0) · iii: Turing-Prüfung des Gated-Tests (fixup cc4751cf0), ersetzt den Fork-Skip · 6597ec4 [Bugfix][SM70] Keep the block-FP8 prefill wor... · 0a694f2 [Bugfix][SM70] Hold the block-FP8 prefill wor... · iv: Block verschoben, Text gleich; Test verschoben (Position), Text gleich |
| `tests/quantization/test_sm70_fp8_prefill_exact_dense.py` | 1 | 0 | 0 | 0 | 1 | iv: Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `tests/quantization/test_turing_qpn_kernel_selection.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/_sm70/linear.py` | 2 | 2 | 0 | 0 | 0 | 768d1a3 feat(sm70): QPN8-Skalenmodus je Spalte und 32... |
| `vllm/_sm70/policy.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/_sm70_ops.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/model_executor/kernels/linear/qpn/fp8_block.py` | 16 | 14 | 0 | 0 | 2 | 0a694f2 [Bugfix][SM70] Hold the block-FP8 prefill wor... · e336f61 perf(qpn8): grouped wo_a prefill as one batch... · 4392c58 <older than 2d218ab03> · iv: Block verschoben, Text gleich; Modul-Docstring (Fork-Auflösung beschreibt Fork-Logik) |
| `vllm/model_executor/kernels/linear/qpn/pre_ampere.py` | 11 | 11 | 0 | 0 | 0 | 4417a4c feat(glm53): load GLM-5.3-Flash-FP4-Local on ... · c213a33 perf(sm75): Tensorkern-GEMM für MXFP8-Zeileng... |
| `vllm/model_executor/layers/fused_moe/experts/skinny_sm70_moe.py` | 25 | 22 | 0 | 0 | 3 | 5fef3ad perf(glm53): prefill reads the five-bit scale... · 64009b8 perf(moe): Volta skinny MoE on a Triton group... · 7ea6a68 perf(moe): reduce skinny expert rows with the... · iv: Stapel-Commit ohne Fork-Zwilling ersetzt Hauptlinien-Zeilen; Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut) |
| `vllm/model_executor/layers/quantization/sm70_turbomind.py` | 7 | 7 | 0 | 0 | 0 | c213a33 perf(sm75): Tensorkern-GEMM für MXFP8-Zeileng... · 4417a4c feat(glm53): load GLM-5.3-Flash-FP4-Local on ... · 768d1a3 feat(sm70): QPN8-Skalenmodus je Spalte und 32... |

### pr-triton-sm7x-r1125 (#1282, pr-r-triton) (Kopf d76b999db)

| Datei | Hunks | i | ii | iii | iv | Beleg |
|---|---|---|---|---|---|---|
| `CMakeLists.txt` | 3 | 1 | 1 | 0 | 1 | db216c2 perf(glm53): expand csf_32x2_v1 expert scales... · ii: main-Änderung seit 2d218ab03 · iv: CMake-Kommentar (Fork 8515f1cc1/2f70712b0), kein Verhalten |
| `csrc/host_reduce.cu` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `csrc/ops.h` | 1 | 1 | 0 | 0 | 0 | db216c2 perf(glm53): expand csf_32x2_v1 expert scales... · 31f9613 perf(glm53): write the CSF exception records ... · d39f2b8 fix(merge): adapt main's DeepSeek-V4.1 caller... |
| `csrc/sm70_turbomind/ops/fp8_qpn8_codes_sm70.cuh` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `csrc/sm70_turbomind/ops/fp8_qpn8_mma_sm75.cu` | 14 | 14 | 0 | 0 | 0 | c213a33 perf(sm75): Tensorkern-GEMM für MXFP8-Zeileng... |
| `csrc/sm70_turbomind/ops/fp8_qpn8_sm70.cu` | 45 | 45 | 0 | 0 | 0 | 768d1a3 feat(sm70): QPN8-Skalenmodus je Spalte und 32... · 8cbf9c1 style(sm70): clang-format our lines in fp8_qp... · 6b8d419 perf(qpn8): DeepSeek-V4 wo_a on Turing straig... |
| `csrc/sm70_turbomind/ops/indexer_prefill_scores_sm75.cu` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `csrc/sm70_turbomind/ops/mma_fragments_sm70.cuh` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `csrc/sm70_turbomind/ops/nvfp4_qpn4_sm70.cu` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `csrc/sm70_turbomind/ops/paged_prefill_attention_sm75.cu` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `csrc/sm70_turbomind/ops/skinny_moe_qpn_sm70.cu` | 38 | 38 | 0 | 0 | 0 | cab7b54 feat(moe): six-bit expert scale planes for th... · e6483cf perf(glm53): decode reads the expert scales a... · de1405f perf(moe): explicit load ring in the Volta de... |
| `csrc/torch_bindings.cpp` | 2 | 2 | 0 | 0 | 0 | cab7b54 feat(moe): six-bit expert scale planes for th... · db216c2 perf(glm53): expand csf_32x2_v1 expert scales... · 31f9613 perf(glm53): write the CSF exception records ... |
| `docs/design/architecture/runtime_reference.md` | 1 | 0 | 0 | 0 | 1 | iv: generierte Referenz; check-architecture-reference je Zweig grün |
| `docs/design/sm70_triton_tensor_cores.md` | – | – | – | – | – | Datei fehlt im Fork (nur im Stapel; (iv) Stapel-Commit ohne Fork-Zwilling) |
| `tests/distributed/test_host_reduce.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/kernels/attention/test_dsv4_sparse_prefill_bmm.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/kernels/attention/test_paged_prefill_attention_sm75.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/kernels/moe/test_skinny_moe_triton.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/kernels/moe/test_skinny_sm70_moe.py` | 8 | 8 | 0 | 0 | 0 | 64cca01 fix(glm53): no csf entries needed once every ... · cab7b54 feat(moe): six-bit expert scale planes for th... · e6483cf perf(glm53): decode reads the expert scales a... |
| `tests/kernels/quantization/test_sm70_qpn8_block_fp8.py` | 9 | 5 | 0 | 2 | 2 | iii: Fork-Auflösung 8018abc2d (Turing-Skip des Gated-Tests); im Stapel als Turing-Prüfung ersetzt (fixup cc4751cf0) · iii: Turing-Prüfung des Gated-Tests (fixup cc4751cf0), ersetzt den Fork-Skip · 6597ec4 [Bugfix][SM70] Keep the block-FP8 prefill wor... · 0a694f2 [Bugfix][SM70] Hold the block-FP8 prefill wor... · iv: Block verschoben, Text gleich; Test verschoben (Position), Text gleich |
| `tests/kernels/test_deepseek_v4_sm70_indexer.py` | 17 | 12 | 0 | 0 | 5 | 9bc45c6 perf(deepseek-v4): score decode index keys wi... · 8018abc merge: 1Cat main 119cae740 into fork-main-v6 · 31d5473 [Perf][SM70] Decode score tile for the DSv4 i... · iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut); Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `tests/kernels/test_kernel_config_triton_sm7x.py` | 2 | 0 | 0 | 0 | 2 | iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut); Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `tests/kernels/test_mhc_prenorm_tensor_cores.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/kernels/utils.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/models/test_deepseek_v4_sm70_routes.py` | 2 | 2 | 0 | 0 | 0 | 772cbe8 perf(deepseek-v4): halve the sparse prefill w... |
| `tests/models/test_deepseek_v4_sm70_sparse_policy.py` | 1 | 0 | 0 | 0 | 1 | iv: Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `tests/quantization/test_sm70_fp8_prefill_exact_dense.py` | 1 | 0 | 0 | 0 | 1 | iv: Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `tests/quantization/test_turing_qpn_kernel_selection.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/v1/attention/test_flash_v100_dense_prefill_availability.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/v1/worker/test_gpu_model_runner_v2.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/v1/worker/test_gpu_model_runner_v2_eplb.py` | 1 | 0 | 0 | 0 | 1 | iv: Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `tests/v1/worker/test_prefill_overlap.py` | 1 | 0 | 0 | 0 | 1 | iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut) |
| `tests/v1/worker/test_qwen4_exp_v2.py` | 4 | 0 | 1 | 0 | 3 | ii: main-Änderung seit 2d218ab03 · iv: Block verschoben, Text gleich; Testattrappe: Stapel setzt prefill_overlap=None (ee27e4904), Fork anders formuliert |
| `vllm/_sm70/linear.py` | 2 | 2 | 0 | 0 | 0 | 768d1a3 feat(sm70): QPN8-Skalenmodus je Spalte und 32... |
| `vllm/_sm70/policy.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/_sm70_ops.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/config/attention.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/config/kernel.py` | 18 | 17 | 0 | 0 | 1 | c071cad [Perf][Qwen4Exp] Keep the PLE disk tier's hot... · c504ce1 [Feature][Spec Decode] Add the adaptive_verif... · a43e966 feat(pp): send decode/verify handovers as one... · iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut) |
| `vllm/config/sm70_sparse.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/distributed/device_communicators/host_reduce.py` | 5 | 4 | 0 | 0 | 1 | 5841543 [Misc][Distributed] Log the channel mapping w... · iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut) |
| `vllm/distributed/parallel_state.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/model_executor/kernels/linear/qpn/fp8_block.py` | 16 | 14 | 0 | 0 | 2 | 0a694f2 [Bugfix][SM70] Hold the block-FP8 prefill wor... · e336f61 perf(qpn8): grouped wo_a prefill as one batch... · 4392c58 <older than 2d218ab03> · iv: Block verschoben, Text gleich; Modul-Docstring (Fork-Auflösung beschreibt Fork-Logik) |
| `vllm/model_executor/kernels/linear/qpn/nvfp4_dequant.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/model_executor/kernels/linear/qpn/pre_ampere.py` | 11 | 11 | 0 | 0 | 0 | 4417a4c feat(glm53): load GLM-5.3-Flash-FP4-Local on ... · c213a33 perf(sm75): Tensorkern-GEMM für MXFP8-Zeileng... |
| `vllm/model_executor/kernels/mhc/tilelang.py` | 7 | 7 | 0 | 0 | 0 | 4417a4c feat(glm53): load GLM-5.3-Flash-FP4-Local on ... |
| `vllm/model_executor/kernels/mhc/triton.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/model_executor/layers/fused_moe/experts/skinny_moe_triton.py` | 1 | 0 | 0 | 0 | 1 | iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut) |
| `vllm/model_executor/layers/fused_moe/experts/skinny_sm70_moe.py` | 21 | 19 | 0 | 0 | 2 | 5fef3ad perf(glm53): prefill reads the five-bit scale... · 9712e0e perf(sm70): Experten-Gruppierung des Skinny-M... · 4417a4c feat(glm53): load GLM-5.3-Flash-FP4-Local on ... · iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut) |
| `vllm/model_executor/layers/fused_moe/sm70/reduction.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/model_executor/layers/mhc.py` | 2 | 2 | 0 | 0 | 0 | 4417a4c feat(glm53): load GLM-5.3-Flash-FP4-Local on ... |
| `vllm/model_executor/layers/quantization/sm70_turbomind.py` | 7 | 7 | 0 | 0 | 0 | c213a33 perf(sm75): Tensorkern-GEMM für MXFP8-Zeileng... · 4417a4c feat(glm53): load GLM-5.3-Flash-FP4-Local on ... · 768d1a3 feat(sm70): QPN8-Skalenmodus je Spalte und 32... |
| `vllm/model_executor/layers/sparse_attn_indexer.py` | 1 | 1 | 0 | 0 | 0 | 9bc45c6 perf(deepseek-v4): score decode index keys wi... |
| `vllm/model_executor/layers/sparse_attn_indexer_kpool.py` | 5 | 5 | 0 | 0 | 0 | 4417a4c feat(glm53): load GLM-5.3-Flash-FP4-Local on ... · 9bc45c6 perf(deepseek-v4): score decode index keys wi... |
| `vllm/models/deepseek_v4/attention.py` | 1 | 1 | 0 | 0 | 0 | 772cbe8 perf(deepseek-v4): halve the sparse prefill w... |
| `vllm/models/deepseek_v4/common/ops/__init__.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/models/deepseek_v4/common/ops/cache_utils.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/models/deepseek_v4/common/ops/sparse_decode_bmm.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/models/deepseek_v4/common/ops/sparse_prefill_bmm.py` | 1 | 1 | 0 | 0 | 0 | 772cbe8 perf(deepseek-v4): halve the sparse prefill w... |
| `vllm/models/deepseek_v4/nvidia/model.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/models/deepseek_v4/sm70/indexer.py` | 21 | 17 | 0 | 0 | 4 | 9bc45c6 perf(deepseek-v4): score decode index keys wi... · 31d5473 [Perf][SM70] Decode score tile for the DSv4 i... · iv: Block verschoben, Text gleich; Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut); Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `vllm/models/deepseek_v4/sm70/sparse.py` | 3 | 3 | 0 | 0 | 0 | 772cbe8 perf(deepseek-v4): halve the sparse prefill w... |
| `vllm/models/deepseek_v41/attention_ops.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/triton_utils/mma_probe.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/v1/attention/backends/flash_attn.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/v1/attention/backends/flash_v100/dense_prefill.py` | 2 | 0 | 2 | 0 | 0 | ii: main-Änderung seit 2d218ab03 |
| `vllm/v1/attention/backends/flash_v100/ops.py` | 2 | 0 | 2 | 0 | 0 | ii: main-Änderung seit 2d218ab03 |
| `vllm/v1/worker/gpu/attn_utils.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/v1/worker/gpu/model_runner.py` | 23 | 23 | 0 | 0 | 0 | a43e966 feat(pp): send decode/verify handovers as one... · 508ac9f perf(runner): one copy for the step's index a... · df92723 feat(pp): verify packed handover layouts at s... |
| `vllm/v1/worker/gpu/model_states/default.py` | 1 | 1 | 0 | 0 | 0 | 62b127e [Perf][MRV2] Compressed-index decode graphs a... |
| `vllm/v1/worker/gpu/model_states/interface.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/v1/worker/gpu/model_states/mamba_hybrid.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/v1/worker/gpu/model_states/whisper.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/v1/worker/gpu/prefill_overlap.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/v1/worker/gpu_ubatch_wrapper.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/v1/worker/ubatching.py` | 1 | 0 | 0 | 0 | 1 | iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut) |

### pr-dsv4-longctx-decode-r1125 (#1347, pr-stack) (Kopf 35ed910b0)

| Datei | Hunks | i | ii | iii | iv | Beleg |
|---|---|---|---|---|---|---|
| `CMakeLists.txt` | 3 | 1 | 1 | 0 | 1 | db216c2 perf(glm53): expand csf_32x2_v1 expert scales... · ii: main-Änderung seit 2d218ab03 · iv: CMake-Kommentar (Fork 8515f1cc1/2f70712b0), kein Verhalten |
| `csrc/host_reduce.cu` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `csrc/ops.h` | 1 | 1 | 0 | 0 | 0 | db216c2 perf(glm53): expand csf_32x2_v1 expert scales... · 31f9613 perf(glm53): write the CSF exception records ... · d39f2b8 fix(merge): adapt main's DeepSeek-V4.1 caller... |
| `csrc/sm70_turbomind/ops/fp8_qpn8_codes_sm70.cuh` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `csrc/sm70_turbomind/ops/fp8_qpn8_mma_sm75.cu` | 14 | 14 | 0 | 0 | 0 | c213a33 perf(sm75): Tensorkern-GEMM für MXFP8-Zeileng... |
| `csrc/sm70_turbomind/ops/fp8_qpn8_sm70.cu` | 45 | 45 | 0 | 0 | 0 | 768d1a3 feat(sm70): QPN8-Skalenmodus je Spalte und 32... · 8cbf9c1 style(sm70): clang-format our lines in fp8_qp... · 6b8d419 perf(qpn8): DeepSeek-V4 wo_a on Turing straig... |
| `csrc/sm70_turbomind/ops/indexer_prefill_scores_sm75.cu` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `csrc/sm70_turbomind/ops/mma_fragments_sm70.cuh` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `csrc/sm70_turbomind/ops/nvfp4_qpn4_sm70.cu` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `csrc/sm70_turbomind/ops/paged_prefill_attention_sm75.cu` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `csrc/sm70_turbomind/ops/skinny_moe_qpn_sm70.cu` | 38 | 38 | 0 | 0 | 0 | cab7b54 feat(moe): six-bit expert scale planes for th... · e6483cf perf(glm53): decode reads the expert scales a... · de1405f perf(moe): explicit load ring in the Volta de... |
| `csrc/torch_bindings.cpp` | 2 | 2 | 0 | 0 | 0 | cab7b54 feat(moe): six-bit expert scale planes for th... · db216c2 perf(glm53): expand csf_32x2_v1 expert scales... · 31f9613 perf(glm53): write the CSF exception records ... |
| `docs/design/architecture/runtime_reference.md` | 1 | 0 | 0 | 0 | 1 | iv: generierte Referenz; check-architecture-reference je Zweig grün |
| `tests/distributed/test_host_reduce.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/kernels/attention/test_dsv4_sparse_prefill_bmm.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/kernels/attention/test_paged_prefill_attention_sm75.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/kernels/moe/test_skinny_moe_triton.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/kernels/moe/test_skinny_sm70_moe.py` | 8 | 8 | 0 | 0 | 0 | 64cca01 fix(glm53): no csf entries needed once every ... · cab7b54 feat(moe): six-bit expert scale planes for th... · e6483cf perf(glm53): decode reads the expert scales a... |
| `tests/kernels/quantization/test_sm70_qpn8_block_fp8.py` | 9 | 5 | 0 | 2 | 2 | iii: Fork-Auflösung 8018abc2d (Turing-Skip des Gated-Tests); im Stapel als Turing-Prüfung ersetzt (fixup cc4751cf0) · iii: Turing-Prüfung des Gated-Tests (fixup cc4751cf0), ersetzt den Fork-Skip · 6597ec4 [Bugfix][SM70] Keep the block-FP8 prefill wor... · 0a694f2 [Bugfix][SM70] Hold the block-FP8 prefill wor... · iv: Block verschoben, Text gleich; Test verschoben (Position), Text gleich |
| `tests/kernels/test_deepseek_v4_sm70_indexer.py` | 6 | 0 | 0 | 0 | 6 | iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut); Wortlaut (Stapel neuer); Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `tests/kernels/test_kernel_config_triton_sm7x.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/kernels/test_mhc_prenorm_tensor_cores.py` | 3 | 3 | 0 | 0 | 0 | 0c5cbfe perf(mhc): let the mHC layer ops take the sm7... |
| `tests/kernels/utils.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/models/test_deepseek_v4_sm70_routes.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/models/test_deepseek_v4_sm70_sparse_policy.py` | 1 | 0 | 0 | 0 | 1 | iv: Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `tests/quantization/test_sm70_fp8_prefill_exact_dense.py` | 1 | 0 | 0 | 0 | 1 | iv: Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `tests/quantization/test_turing_qpn_kernel_selection.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/v1/attention/test_flash_v100_dense_prefill_availability.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/v1/worker/test_gpu_model_runner_v2.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/v1/worker/test_gpu_model_runner_v2_eplb.py` | 1 | 0 | 0 | 0 | 1 | iv: Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `tests/v1/worker/test_prefill_overlap.py` | 1 | 0 | 0 | 0 | 1 | iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut) |
| `tests/v1/worker/test_qwen4_exp_v2.py` | 4 | 0 | 1 | 0 | 3 | ii: main-Änderung seit 2d218ab03 · iv: Block verschoben, Text gleich; Testattrappe: Stapel setzt prefill_overlap=None (ee27e4904), Fork anders formuliert |
| `vllm/_sm70/linear.py` | 2 | 2 | 0 | 0 | 0 | 768d1a3 feat(sm70): QPN8-Skalenmodus je Spalte und 32... |
| `vllm/_sm70/policy.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/_sm70_ops.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/config/attention.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/config/kernel.py` | 17 | 17 | 0 | 0 | 0 | c071cad [Perf][Qwen4Exp] Keep the PLE disk tier's hot... · c504ce1 [Feature][Spec Decode] Add the adaptive_verif... · a43e966 feat(pp): send decode/verify handovers as one... |
| `vllm/config/sm70_sparse.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/distributed/device_communicators/host_reduce.py` | 5 | 4 | 0 | 0 | 1 | 5841543 [Misc][Distributed] Log the channel mapping w... · iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut) |
| `vllm/distributed/parallel_state.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/model_executor/kernels/linear/qpn/fp8_block.py` | 16 | 14 | 0 | 0 | 2 | 0a694f2 [Bugfix][SM70] Hold the block-FP8 prefill wor... · e336f61 perf(qpn8): grouped wo_a prefill as one batch... · 4392c58 <older than 2d218ab03> · iv: Block verschoben, Text gleich; Modul-Docstring (Fork-Auflösung beschreibt Fork-Logik) |
| `vllm/model_executor/kernels/linear/qpn/nvfp4_dequant.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/model_executor/kernels/linear/qpn/pre_ampere.py` | 11 | 11 | 0 | 0 | 0 | 4417a4c feat(glm53): load GLM-5.3-Flash-FP4-Local on ... · c213a33 perf(sm75): Tensorkern-GEMM für MXFP8-Zeileng... |
| `vllm/model_executor/kernels/mhc/tilelang.py` | 7 | 7 | 0 | 0 | 0 | 4417a4c feat(glm53): load GLM-5.3-Flash-FP4-Local on ... |
| `vllm/model_executor/kernels/mhc/triton.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/model_executor/layers/fused_moe/experts/skinny_moe_triton.py` | 1 | 0 | 0 | 0 | 1 | iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut) |
| `vllm/model_executor/layers/fused_moe/experts/skinny_sm70_moe.py` | 21 | 19 | 0 | 0 | 2 | 5fef3ad perf(glm53): prefill reads the five-bit scale... · 9712e0e perf(sm70): Experten-Gruppierung des Skinny-M... · 4417a4c feat(glm53): load GLM-5.3-Flash-FP4-Local on ... · iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut) |
| `vllm/model_executor/layers/fused_moe/sm70/reduction.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/model_executor/layers/quantization/sm70_turbomind.py` | 7 | 7 | 0 | 0 | 0 | c213a33 perf(sm75): Tensorkern-GEMM für MXFP8-Zeileng... · 4417a4c feat(glm53): load GLM-5.3-Flash-FP4-Local on ... · 768d1a3 feat(sm70): QPN8-Skalenmodus je Spalte und 32... |
| `vllm/model_executor/layers/sparse_attn_indexer.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/model_executor/layers/sparse_attn_indexer_kpool.py` | 4 | 4 | 0 | 0 | 0 | 4417a4c feat(glm53): load GLM-5.3-Flash-FP4-Local on ... |
| `vllm/models/deepseek_v4/attention.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/models/deepseek_v4/common/ops/__init__.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/models/deepseek_v4/common/ops/cache_utils.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/models/deepseek_v4/common/ops/sparse_decode_bmm.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/models/deepseek_v4/common/ops/sparse_prefill_bmm.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/models/deepseek_v4/nvidia/model.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/models/deepseek_v4/sm70/indexer.py` | 1 | 0 | 0 | 0 | 1 | iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut) |
| `vllm/models/deepseek_v4/sm70/sparse.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/models/deepseek_v41/attention_ops.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/triton_utils/mma_probe.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/v1/attention/backends/flash_attn.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/v1/attention/backends/flash_v100/dense_prefill.py` | 2 | 0 | 2 | 0 | 0 | ii: main-Änderung seit 2d218ab03 |
| `vllm/v1/attention/backends/flash_v100/ops.py` | 2 | 0 | 2 | 0 | 0 | ii: main-Änderung seit 2d218ab03 |
| `vllm/v1/worker/gpu/attn_utils.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/v1/worker/gpu/model_runner.py` | 23 | 23 | 0 | 0 | 0 | a43e966 feat(pp): send decode/verify handovers as one... · 508ac9f perf(runner): one copy for the step's index a... · df92723 feat(pp): verify packed handover layouts at s... |
| `vllm/v1/worker/gpu/model_states/default.py` | 1 | 1 | 0 | 0 | 0 | 62b127e [Perf][MRV2] Compressed-index decode graphs a... |
| `vllm/v1/worker/gpu/model_states/interface.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/v1/worker/gpu/model_states/mamba_hybrid.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/v1/worker/gpu/model_states/whisper.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/v1/worker/gpu/prefill_overlap.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/v1/worker/gpu_ubatch_wrapper.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/v1/worker/ubatching.py` | 1 | 0 | 0 | 0 | 1 | iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut) |

### pr-glm53-r1125 (#1286, pr-r-glm53) (Kopf 00ad05cb9)

| Datei | Hunks | i | ii | iii | iv | Beleg |
|---|---|---|---|---|---|---|
| `CMakeLists.txt` | 2 | 0 | 1 | 0 | 1 | ii: main-Änderung seit 2d218ab03 · iv: CMake-Kommentar (Fork 8515f1cc1/2f70712b0), kein Verhalten |
| `csrc/host_reduce.cu` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `csrc/ops.h` | 1 | 1 | 0 | 0 | 0 | cab7b54 feat(moe): six-bit expert scale planes for th... |
| `csrc/sm70_turbomind/ops/csf32_expand_sm70.cu` | 1 | 0 | 0 | 0 | 1 | iv: Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `csrc/sm70_turbomind/ops/fp8_qpn8_codes_sm70.cuh` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `csrc/sm70_turbomind/ops/fp8_qpn8_mma_sm75.cu` | 3 | 0 | 0 | 0 | 3 | iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut) |
| `csrc/sm70_turbomind/ops/fp8_qpn8_sm70.cu` | 28 | 28 | 0 | 0 | 0 | 6b8d419 perf(qpn8): DeepSeek-V4 wo_a on Turing straig... · e336f61 perf(qpn8): grouped wo_a prefill as one batch... |
| `csrc/sm70_turbomind/ops/indexer_prefill_scores_sm75.cu` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `csrc/sm70_turbomind/ops/mma_fragments_sm70.cuh` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `csrc/sm70_turbomind/ops/nvfp4_qpn4_sm70.cu` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `csrc/sm70_turbomind/ops/paged_prefill_attention_sm75.cu` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `csrc/sm70_turbomind/ops/skinny_moe_qpn_sm70.cu` | 52 | 45 | 0 | 0 | 7 | cab7b54 feat(moe): six-bit expert scale planes for th... · de1405f perf(moe): explicit load ring in the Volta de... · dcd2b99 perf(moe): three CTAs per SM for the Turing d... · iv: Stapel-Commit ohne Fork-Zwilling ersetzt Hauptlinien-Zeilen; Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut) |
| `csrc/torch_bindings.cpp` | 1 | 1 | 0 | 0 | 0 | cab7b54 feat(moe): six-bit expert scale planes for th... |
| `docs/design/architecture/runtime_reference.md` | 1 | 0 | 0 | 0 | 1 | iv: generierte Referenz; check-architecture-reference je Zweig grün |
| `docs/design/sm70_glm53_pp5_dflash2_mixed.md` | 7 | 7 | 0 | 0 | 0 | ea42af1 docs(glm53): PP5 with DFlash2 on three V100 a... |
| `tests/distributed/test_host_reduce.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/kernels/attention/test_dsv4_sparse_prefill_bmm.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/kernels/attention/test_paged_prefill_attention_sm75.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/kernels/moe/test_skinny_moe_triton.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/kernels/moe/test_skinny_sm70_moe.py` | 13 | 13 | 0 | 0 | 0 | cab7b54 feat(moe): six-bit expert scale planes for th... · 3581080 test(moe): six-bit scale test data that fp16 ... · de1405f perf(moe): explicit load ring in the Volta de... |
| `tests/kernels/moe/test_sm70_group_routing.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/kernels/quantization/test_csf32_scales.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/kernels/quantization/test_s5_scales.py` | 12 | 12 | 0 | 0 | 0 | cab7b54 feat(moe): six-bit expert scale planes for th... |
| `tests/kernels/quantization/test_sm70_qpn8_block_fp8.py` | 9 | 5 | 0 | 2 | 2 | iii: Fork-Auflösung 8018abc2d (Turing-Skip des Gated-Tests); im Stapel als Turing-Prüfung ersetzt (fixup cc4751cf0) · iii: Turing-Prüfung des Gated-Tests (fixup cc4751cf0), ersetzt den Fork-Skip · 6597ec4 [Bugfix][SM70] Keep the block-FP8 prefill wor... · 0a694f2 [Bugfix][SM70] Hold the block-FP8 prefill wor... · iv: Block verschoben, Text gleich; Test verschoben (Position), Text gleich |
| `tests/kernels/quantization/test_sm70_qpn8_mxfp8_linear.py` | 1 | 0 | 0 | 0 | 1 | iv: Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `tests/kernels/test_deepseek_v4_sm70_indexer.py` | 17 | 12 | 0 | 0 | 5 | 9bc45c6 perf(deepseek-v4): score decode index keys wi... · 8018abc merge: 1Cat main 119cae740 into fork-main-v6 · 31d5473 [Perf][SM70] Decode score tile for the DSv4 i... · iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut); Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `tests/kernels/test_kernel_config_triton_sm7x.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/kernels/test_mhc_prenorm_tensor_cores.py` | 3 | 3 | 0 | 0 | 0 | 0c5cbfe perf(mhc): let the mHC layer ops take the sm7... |
| `tests/kernels/test_mhc_sm70_fp16.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/kernels/utils.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/models/glm5next/test_fp8_attention_loader.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/models/glm5next/test_mxfp8_held_modules.py` | 1 | 1 | 0 | 0 | 0 | 0181fdc perf(glm53): serve the MXFP8 LM head with the... |
| `tests/models/glm5next/test_pp.py` | 1 | 0 | 0 | 0 | 1 | iv: Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `tests/models/glm5next/test_sm70_sparse.py` | 2 | 1 | 0 | 0 | 1 | 4221842 perf(glm53): Sparse-MLA-Prefill mit FP8-KV üb... · iv: Skip-Bedingung: Stapel strenger (CPU-Hygiene, Teil 1e) |
| `tests/models/glm5next/test_vision_row_chunks.py` | 1 | 0 | 0 | 0 | 1 | iv: Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `tests/models/test_deepseek_v4_sm70_routes.py` | 2 | 2 | 0 | 0 | 0 | 772cbe8 perf(deepseek-v4): halve the sparse prefill w... |
| `tests/models/test_deepseek_v4_sm70_sparse_policy.py` | 1 | 0 | 0 | 0 | 1 | iv: Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `tests/quantization/test_modelopt.py` | 2 | 2 | 0 | 0 | 0 | 0181fdc perf(glm53): serve the MXFP8 LM head with the... |
| `tests/quantization/test_mxfp8_weight_pairs.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/quantization/test_sm70_fp8_prefill_exact_dense.py` | 1 | 0 | 0 | 0 | 1 | iv: Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `tests/quantization/test_sm70_glm53_modelopt_nvfp4.py` | 1 | 0 | 0 | 0 | 1 | iv: Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `tests/quantization/test_sm70_modelopt_mixed_nvfp4.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/quantization/test_sm72_volta_admission.py` | 2 | 0 | 0 | 0 | 2 | iv: Stapel-Commit ohne Fork-Zwilling ersetzt Hauptlinien-Zeilen; Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `tests/quantization/test_turing_qpn_kernel_selection.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/v1/attention/test_flash_v100_dense_prefill_availability.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/v1/attention/test_mla_sparse_profile_budget.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/v1/worker/test_gpu_model_runner_v2.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/v1/worker/test_gpu_model_runner_v2_eplb.py` | 1 | 0 | 0 | 0 | 1 | iv: Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `tests/v1/worker/test_prefill_overlap.py` | 1 | 0 | 0 | 0 | 1 | iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut) |
| `tests/v1/worker/test_qwen4_exp_v2.py` | 4 | 0 | 1 | 0 | 3 | ii: main-Änderung seit 2d218ab03 · iv: Block verschoben, Text gleich; Testattrappe: Stapel setzt prefill_overlap=None (ee27e4904), Fork anders formuliert |
| `vllm/_custom_ops.py` | 1 | 0 | 1 | 0 | 0 | ii: main-Änderung seit 2d218ab03 |
| `vllm/_sm70/linear.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/_sm70/policy.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/_sm70_ops.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/config/attention.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/config/kernel.py` | 17 | 17 | 0 | 0 | 0 | c071cad [Perf][Qwen4Exp] Keep the PLE disk tier's hot... · c504ce1 [Feature][Spec Decode] Add the adaptive_verif... · a43e966 feat(pp): send decode/verify handovers as one... |
| `vllm/config/sm70_sparse.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/distributed/device_communicators/host_reduce.py` | 5 | 4 | 0 | 0 | 1 | 5841543 [Misc][Distributed] Log the channel mapping w... · iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut) |
| `vllm/distributed/parallel_state.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/model_executor/kernels/linear/__init__.py` | 2 | 0 | 0 | 0 | 2 | iv: Stapel-Commit ohne Fork-Zwilling ersetzt Hauptlinien-Zeilen |
| `vllm/model_executor/kernels/linear/mxfp8/Mxfp8LinearKernel.py` | 2 | 0 | 0 | 0 | 2 | iv: Stapel-Commit ohne Fork-Zwilling ersetzt Hauptlinien-Zeilen |
| `vllm/model_executor/kernels/linear/qpn/fp8_block.py` | 16 | 14 | 0 | 0 | 2 | 0a694f2 [Bugfix][SM70] Hold the block-FP8 prefill wor... · e336f61 perf(qpn8): grouped wo_a prefill as one batch... · 4392c58 <older than 2d218ab03> · iv: Block verschoben, Text gleich; Modul-Docstring (Fork-Auflösung beschreibt Fork-Logik) |
| `vllm/model_executor/kernels/linear/qpn/nvfp4_dequant.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/model_executor/kernels/linear/qpn/pre_ampere.py` | 1 | 1 | 0 | 0 | 0 | 4417a4c feat(glm53): load GLM-5.3-Flash-FP4-Local on ... |
| `vllm/model_executor/kernels/mhc/tilelang.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/model_executor/kernels/mhc/triton.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/model_executor/layers/attention/mla_attention.py` | 6 | 6 | 0 | 0 | 0 | 48a92aa perf(mla): skinny Triton kernel for the two k... |
| `vllm/model_executor/layers/fused_moe/experts/skinny_moe_triton.py` | 1 | 0 | 0 | 0 | 1 | iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut) |
| `vllm/model_executor/layers/fused_moe/experts/skinny_sm70_moe.py` | 26 | 23 | 0 | 0 | 3 | cab7b54 feat(moe): six-bit expert scale planes for th... · iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut); Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `vllm/model_executor/layers/fused_moe/sm70/reduction.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/model_executor/layers/fused_moe/sm70/routing.py` | 8 | 7 | 0 | 0 | 1 | ca2c1b6 perf(sm70-moe): sort-based routing-group kern... · iv: Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `vllm/model_executor/layers/mhc.py` | 5 | 5 | 0 | 0 | 0 | 0c5cbfe perf(mhc): let the mHC layer ops take the sm7... |
| `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_mxfp8.py` | 3 | 1 | 0 | 0 | 2 | 187b932 <older than 2d218ab03> · iv: Stapel-Commit ohne Fork-Zwilling ersetzt Hauptlinien-Zeilen; Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `vllm/model_executor/layers/quantization/modelopt.py` | 10 | 7 | 0 | 0 | 3 | 0181fdc perf(glm53): serve the MXFP8 LM head with the... · 187b932 <older than 2d218ab03> · iv: Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `vllm/model_executor/layers/quantization/online/mxfp8.py` | 3 | 1 | 0 | 0 | 2 | 187b932 <older than 2d218ab03> · iv: Stapel-Commit ohne Fork-Zwilling ersetzt Hauptlinien-Zeilen; Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `vllm/model_executor/layers/quantization/sm70_turbomind.py` | 4 | 0 | 0 | 0 | 4 | iv: Stapel-Commit ohne Fork-Zwilling ersetzt Hauptlinien-Zeilen; Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `vllm/model_executor/layers/quantization/utils/csf32_expand.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/model_executor/layers/quantization/utils/csf32_scales.py` | 1 | 1 | 0 | 0 | 0 | 4417a4c feat(glm53): load GLM-5.3-Flash-FP4-Local on ... |
| `vllm/model_executor/layers/quantization/utils/mxfp8_utils.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/model_executor/layers/quantization/utils/s5_scales.py` | 20 | 20 | 0 | 0 | 0 | cab7b54 feat(moe): six-bit expert scale planes for th... · e6483cf perf(glm53): decode reads the expert scales a... |
| `vllm/model_executor/layers/sparse_attn_indexer.py` | 1 | 1 | 0 | 0 | 0 | 9bc45c6 perf(deepseek-v4): score decode index keys wi... |
| `vllm/model_executor/layers/sparse_attn_indexer_kpool.py` | 1 | 1 | 0 | 0 | 0 | 9bc45c6 perf(deepseek-v4): score decode index keys wi... |
| `vllm/models/deepseek_v4/attention.py` | 1 | 1 | 0 | 0 | 0 | 772cbe8 perf(deepseek-v4): halve the sparse prefill w... |
| `vllm/models/deepseek_v4/common/ops/__init__.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/models/deepseek_v4/common/ops/cache_utils.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/models/deepseek_v4/common/ops/sparse_decode_bmm.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/models/deepseek_v4/common/ops/sparse_prefill_bmm.py` | 1 | 1 | 0 | 0 | 0 | 772cbe8 perf(deepseek-v4): halve the sparse prefill w... |
| `vllm/models/deepseek_v4/nvidia/model.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/models/deepseek_v4/sm70/indexer.py` | 21 | 17 | 0 | 0 | 4 | 9bc45c6 perf(deepseek-v4): score decode index keys wi... · 31d5473 [Perf][SM70] Decode score tile for the DSv4 i... · iv: Block verschoben, Text gleich; Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut); Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `vllm/models/deepseek_v4/sm70/sparse.py` | 3 | 3 | 0 | 0 | 0 | 772cbe8 perf(deepseek-v4): halve the sparse prefill w... |
| `vllm/models/deepseek_v41/attention_ops.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/models/glm5next/nvidia/attention.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/models/glm5next/nvidia/kda.py` | 13 | 13 | 0 | 0 | 0 | b817aa1 perf(glm5next): fuse the KDA decode glue kern... · 3f6bdcc perf(glm53): tune the chunked KDA prefill ker... |
| `vllm/models/glm5next/nvidia/model.py` | 8 | 8 | 0 | 0 | 0 | 0ef69f4 fix(glm5next): compute the MoE router GEMM on... · 16d886e perf(glm53): Einbettungstabelle über den UVA-... · df65c5a feat(glm53): MXFP8-Projektionen als 8 Bit im ... |
| `vllm/models/glm5next/nvidia/mtp.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/models/glm5next/nvidia/multimodal.py` | 2 | 1 | 0 | 0 | 1 | b6583fb [Bugfix] Fix GLM image cache size (backport o... · iv: Block verschoben, Text gleich |
| `vllm/models/glm5next/nvidia/ops/kpool_compress.py` | 6 | 5 | 0 | 0 | 1 | c37a884 <older than 2d218ab03> · 4417a4c feat(glm53): load GLM-5.3-Flash-FP4-Local on ... · a1048ae fix(sm70): ask the own device's capability wi... · iv: Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `vllm/models/glm5next/nvidia/quant.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/models/glm5next/sm70/fp8_kv.py` | 1 | 1 | 0 | 0 | 0 | 4221842 perf(glm53): Sparse-MLA-Prefill mit FP8-KV üb... |
| `vllm/models/glm5next/sm70/sparse.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/platforms/cuda.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/triton_utils/mma_probe.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/v1/attention/backends/flash_attn.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/v1/attention/backends/flash_v100/dense_prefill.py` | 2 | 0 | 2 | 0 | 0 | ii: main-Änderung seit 2d218ab03 |
| `vllm/v1/attention/backends/flash_v100/ops.py` | 2 | 0 | 2 | 0 | 0 | ii: main-Änderung seit 2d218ab03 |
| `vllm/v1/worker/gpu/attn_utils.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/v1/worker/gpu/model_runner.py` | 23 | 23 | 0 | 0 | 0 | a43e966 feat(pp): send decode/verify handovers as one... · 508ac9f perf(runner): one copy for the step's index a... · df92723 feat(pp): verify packed handover layouts at s... |
| `vllm/v1/worker/gpu/model_states/default.py` | 1 | 1 | 0 | 0 | 0 | 62b127e [Perf][MRV2] Compressed-index decode graphs a... |
| `vllm/v1/worker/gpu/model_states/interface.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/v1/worker/gpu/model_states/mamba_hybrid.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/v1/worker/gpu/model_states/whisper.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/v1/worker/gpu/prefill_overlap.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/v1/worker/gpu_ubatch_wrapper.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/v1/worker/ubatching.py` | 1 | 0 | 0 | 0 | 1 | iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut) |

### pr-glm53-prefill-head-r1125 (#1348, pr-r-glm53ph) (Kopf a88d81165)

| Datei | Hunks | i | ii | iii | iv | Beleg |
|---|---|---|---|---|---|---|
| `CMakeLists.txt` | 2 | 0 | 1 | 0 | 1 | ii: main-Änderung seit 2d218ab03 · iv: CMake-Kommentar (Fork 8515f1cc1/2f70712b0), kein Verhalten |
| `csrc/host_reduce.cu` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `csrc/ops.h` | 1 | 1 | 0 | 0 | 0 | cab7b54 feat(moe): six-bit expert scale planes for th... |
| `csrc/sm70_turbomind/ops/csf32_expand_sm70.cu` | 1 | 0 | 0 | 0 | 1 | iv: Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `csrc/sm70_turbomind/ops/fp8_qpn8_codes_sm70.cuh` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `csrc/sm70_turbomind/ops/fp8_qpn8_mma_sm75.cu` | 3 | 2 | 0 | 0 | 1 | c213a33 perf(sm75): Tensorkern-GEMM für MXFP8-Zeileng... · iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut) |
| `csrc/sm70_turbomind/ops/fp8_qpn8_sm70.cu` | 28 | 28 | 0 | 0 | 0 | 6b8d419 perf(qpn8): DeepSeek-V4 wo_a on Turing straig... · e336f61 perf(qpn8): grouped wo_a prefill as one batch... |
| `csrc/sm70_turbomind/ops/indexer_prefill_scores_sm75.cu` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `csrc/sm70_turbomind/ops/mma_fragments_sm70.cuh` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `csrc/sm70_turbomind/ops/nvfp4_qpn4_sm70.cu` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `csrc/sm70_turbomind/ops/paged_prefill_attention_sm75.cu` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `csrc/sm70_turbomind/ops/skinny_moe_qpn_sm70.cu` | 52 | 42 | 0 | 0 | 10 | cab7b54 feat(moe): six-bit expert scale planes for th... · de1405f perf(moe): explicit load ring in the Volta de... · dcd2b99 perf(moe): three CTAs per SM for the Turing d... · iv: Stapel-Commit ohne Fork-Zwilling ersetzt Hauptlinien-Zeilen; Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut) |
| `csrc/torch_bindings.cpp` | 1 | 1 | 0 | 0 | 0 | cab7b54 feat(moe): six-bit expert scale planes for th... |
| `docs/design/architecture/runtime_reference.md` | 1 | 0 | 0 | 0 | 1 | iv: generierte Referenz; check-architecture-reference je Zweig grün |
| `docs/design/sm70_glm53_pp5_dflash2_mixed.md` | 8 | 0 | 0 | 0 | 8 | iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut); Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `tests/distributed/test_host_reduce.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/kernels/attention/test_dsv4_sparse_prefill_bmm.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/kernels/attention/test_paged_prefill_attention_sm75.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/kernels/moe/test_skinny_moe_triton.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/kernels/moe/test_skinny_sm70_moe.py` | 13 | 13 | 0 | 0 | 0 | cab7b54 feat(moe): six-bit expert scale planes for th... · 3581080 test(moe): six-bit scale test data that fp16 ... · de1405f perf(moe): explicit load ring in the Volta de... |
| `tests/kernels/moe/test_sm70_group_routing.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/kernels/quantization/test_csf32_scales.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/kernels/quantization/test_s5_scales.py` | 12 | 12 | 0 | 0 | 0 | cab7b54 feat(moe): six-bit expert scale planes for th... |
| `tests/kernels/quantization/test_sm70_qpn8_block_fp8.py` | 9 | 5 | 0 | 2 | 2 | iii: Fork-Auflösung 8018abc2d (Turing-Skip des Gated-Tests); im Stapel als Turing-Prüfung ersetzt (fixup cc4751cf0) · iii: Turing-Prüfung des Gated-Tests (fixup cc4751cf0), ersetzt den Fork-Skip · 6597ec4 [Bugfix][SM70] Keep the block-FP8 prefill wor... · 0a694f2 [Bugfix][SM70] Hold the block-FP8 prefill wor... · iv: Block verschoben, Text gleich; Test verschoben (Position), Text gleich |
| `tests/kernels/quantization/test_sm70_qpn8_mxfp8_linear.py` | 1 | 0 | 0 | 0 | 1 | iv: Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `tests/kernels/test_deepseek_v4_sm70_indexer.py` | 17 | 12 | 0 | 0 | 5 | 9bc45c6 perf(deepseek-v4): score decode index keys wi... · 8018abc merge: 1Cat main 119cae740 into fork-main-v6 · 31d5473 [Perf][SM70] Decode score tile for the DSv4 i... · iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut); Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `tests/kernels/test_kernel_config_triton_sm7x.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/kernels/test_mhc_prenorm_tensor_cores.py` | 3 | 3 | 0 | 0 | 0 | 0c5cbfe perf(mhc): let the mHC layer ops take the sm7... |
| `tests/kernels/test_mhc_sm70_fp16.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/kernels/utils.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/models/glm5next/test_fp8_attention_loader.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/models/glm5next/test_kda_prefill_warmup.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/models/glm5next/test_mxfp8_held_modules.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/models/glm5next/test_pp.py` | 1 | 0 | 0 | 0 | 1 | iv: Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `tests/models/glm5next/test_sm70_sparse.py` | 2 | 0 | 0 | 0 | 2 | iv: Skip-Bedingung: Stapel strenger (CPU-Hygiene, Teil 1e); Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut) |
| `tests/models/test_deepseek_v4_sm70_routes.py` | 2 | 2 | 0 | 0 | 0 | 772cbe8 perf(deepseek-v4): halve the sparse prefill w... |
| `tests/models/test_deepseek_v4_sm70_sparse_policy.py` | 1 | 0 | 0 | 0 | 1 | iv: Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `tests/quantization/test_modelopt.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/quantization/test_mxfp8_padded_lm_head.py` | – | – | – | – | – | Datei fehlt im Fork (nur im Stapel; (iv) Stapel-Commit ohne Fork-Zwilling) |
| `tests/quantization/test_mxfp8_weight_pairs.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/quantization/test_sm70_fp8_prefill_exact_dense.py` | 1 | 0 | 0 | 0 | 1 | iv: Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `tests/quantization/test_sm70_glm53_modelopt_nvfp4.py` | 1 | 0 | 0 | 0 | 1 | iv: Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `tests/quantization/test_sm70_modelopt_mixed_nvfp4.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/quantization/test_sm72_volta_admission.py` | 2 | 0 | 0 | 0 | 2 | iv: Stapel-Commit ohne Fork-Zwilling ersetzt Hauptlinien-Zeilen; Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `tests/quantization/test_turing_qpn_kernel_selection.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/v1/attention/test_flash_v100_dense_prefill_availability.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/v1/attention/test_mla_sparse_profile_budget.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/v1/worker/test_gpu_model_runner_v2.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/v1/worker/test_gpu_model_runner_v2_eplb.py` | 1 | 0 | 0 | 0 | 1 | iv: Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `tests/v1/worker/test_prefill_overlap.py` | 1 | 0 | 0 | 0 | 1 | iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut) |
| `tests/v1/worker/test_qwen4_exp_v2.py` | 4 | 0 | 1 | 0 | 3 | ii: main-Änderung seit 2d218ab03 · iv: Block verschoben, Text gleich; Testattrappe: Stapel setzt prefill_overlap=None (ee27e4904), Fork anders formuliert |
| `vllm/_custom_ops.py` | 1 | 0 | 1 | 0 | 0 | ii: main-Änderung seit 2d218ab03 |
| `vllm/_sm70/linear.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/_sm70/policy.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/_sm70_ops.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/config/attention.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/config/kernel.py` | 17 | 17 | 0 | 0 | 0 | c071cad [Perf][Qwen4Exp] Keep the PLE disk tier's hot... · c504ce1 [Feature][Spec Decode] Add the adaptive_verif... · a43e966 feat(pp): send decode/verify handovers as one... |
| `vllm/config/sm70_sparse.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/distributed/device_communicators/host_reduce.py` | 5 | 4 | 0 | 0 | 1 | 5841543 [Misc][Distributed] Log the channel mapping w... · iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut) |
| `vllm/distributed/parallel_state.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/model_executor/kernels/linear/__init__.py` | 2 | 0 | 0 | 0 | 2 | iv: Stapel-Commit ohne Fork-Zwilling ersetzt Hauptlinien-Zeilen |
| `vllm/model_executor/kernels/linear/mxfp8/Mxfp8LinearKernel.py` | 2 | 0 | 0 | 0 | 2 | iv: Stapel-Commit ohne Fork-Zwilling ersetzt Hauptlinien-Zeilen |
| `vllm/model_executor/kernels/linear/qpn/fp8_block.py` | 16 | 14 | 0 | 0 | 2 | 0a694f2 [Bugfix][SM70] Hold the block-FP8 prefill wor... · e336f61 perf(qpn8): grouped wo_a prefill as one batch... · 4392c58 <older than 2d218ab03> · iv: Block verschoben, Text gleich; Modul-Docstring (Fork-Auflösung beschreibt Fork-Logik) |
| `vllm/model_executor/kernels/linear/qpn/nvfp4_dequant.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/model_executor/kernels/linear/qpn/pre_ampere.py` | 1 | 0 | 0 | 0 | 1 | iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut) |
| `vllm/model_executor/kernels/mhc/tilelang.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/model_executor/kernels/mhc/triton.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/model_executor/layers/attention/mla_attention.py` | 6 | 6 | 0 | 0 | 0 | 48a92aa perf(mla): skinny Triton kernel for the two k... |
| `vllm/model_executor/layers/fused_moe/experts/skinny_moe_triton.py` | 1 | 0 | 0 | 0 | 1 | iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut) |
| `vllm/model_executor/layers/fused_moe/experts/skinny_sm70_moe.py` | 26 | 23 | 0 | 0 | 3 | cab7b54 feat(moe): six-bit expert scale planes for th... · iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut); Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `vllm/model_executor/layers/fused_moe/sm70/reduction.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/model_executor/layers/fused_moe/sm70/routing.py` | 8 | 7 | 0 | 0 | 1 | ca2c1b6 perf(sm70-moe): sort-based routing-group kern... · iv: Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `vllm/model_executor/layers/mhc.py` | 5 | 5 | 0 | 0 | 0 | 0c5cbfe perf(mhc): let the mHC layer ops take the sm7... |
| `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_mxfp8.py` | 6 | 1 | 0 | 0 | 5 | 187b932 <older than 2d218ab03> · iv: Stapel-Commit ohne Fork-Zwilling ersetzt Hauptlinien-Zeilen; Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `vllm/model_executor/layers/quantization/modelopt.py` | 10 | 1 | 0 | 0 | 9 | 187b932 <older than 2d218ab03> · iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut); Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `vllm/model_executor/layers/quantization/online/mxfp8.py` | 3 | 1 | 0 | 0 | 2 | 187b932 <older than 2d218ab03> · iv: Stapel-Commit ohne Fork-Zwilling ersetzt Hauptlinien-Zeilen; Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `vllm/model_executor/layers/quantization/sm70_turbomind.py` | 4 | 0 | 0 | 0 | 4 | iv: Stapel-Commit ohne Fork-Zwilling ersetzt Hauptlinien-Zeilen; Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `vllm/model_executor/layers/quantization/utils/csf32_expand.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/model_executor/layers/quantization/utils/csf32_scales.py` | 1 | 0 | 0 | 0 | 1 | iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut) |
| `vllm/model_executor/layers/quantization/utils/mxfp8_utils.py` | 1 | 0 | 0 | 0 | 1 | iv: Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `vllm/model_executor/layers/quantization/utils/s5_scales.py` | 20 | 19 | 0 | 0 | 1 | cab7b54 feat(moe): six-bit expert scale planes for th... · iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut) |
| `vllm/model_executor/layers/sparse_attn_indexer.py` | 1 | 1 | 0 | 0 | 0 | 9bc45c6 perf(deepseek-v4): score decode index keys wi... |
| `vllm/model_executor/layers/sparse_attn_indexer_kpool.py` | 1 | 1 | 0 | 0 | 0 | 9bc45c6 perf(deepseek-v4): score decode index keys wi... |
| `vllm/model_executor/layers/vocab_parallel_embedding.py` | 1 | 0 | 0 | 0 | 1 | iv: Stapel-Commit ohne Fork-Zwilling ersetzt Hauptlinien-Zeilen |
| `vllm/models/deepseek_v4/attention.py` | 1 | 1 | 0 | 0 | 0 | 772cbe8 perf(deepseek-v4): halve the sparse prefill w... |
| `vllm/models/deepseek_v4/common/ops/__init__.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/models/deepseek_v4/common/ops/cache_utils.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/models/deepseek_v4/common/ops/sparse_decode_bmm.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/models/deepseek_v4/common/ops/sparse_prefill_bmm.py` | 1 | 1 | 0 | 0 | 0 | 772cbe8 perf(deepseek-v4): halve the sparse prefill w... |
| `vllm/models/deepseek_v4/nvidia/model.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/models/deepseek_v4/sm70/indexer.py` | 21 | 17 | 0 | 0 | 4 | 9bc45c6 perf(deepseek-v4): score decode index keys wi... · 31d5473 [Perf][SM70] Decode score tile for the DSv4 i... · iv: Block verschoben, Text gleich; Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut); Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `vllm/models/deepseek_v4/sm70/sparse.py` | 3 | 3 | 0 | 0 | 0 | 772cbe8 perf(deepseek-v4): halve the sparse prefill w... |
| `vllm/models/deepseek_v41/attention_ops.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/models/glm5next/nvidia/attention.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/models/glm5next/nvidia/kda.py` | 11 | 11 | 0 | 0 | 0 | b817aa1 perf(glm5next): fuse the KDA decode glue kern... |
| `vllm/models/glm5next/nvidia/model.py` | 6 | 3 | 0 | 0 | 3 | 0ef69f4 fix(glm5next): compute the MoE router GEMM on... · bbce30a [Bugfix][GLM-5.3] Build the vision tower with... · iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut) |
| `vllm/models/glm5next/nvidia/mtp.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/models/glm5next/nvidia/ops/kpool_compress.py` | 6 | 2 | 0 | 0 | 4 | c37a884 <older than 2d218ab03> · 4417a4c feat(glm53): load GLM-5.3-Flash-FP4-Local on ... · a1048ae fix(sm70): ask the own device's capability wi... · iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut); Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `vllm/models/glm5next/nvidia/quant.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/models/glm5next/sm70/fp8_kv.py` | 1 | 0 | 0 | 0 | 1 | iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut) |
| `vllm/models/glm5next/sm70/sparse.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/platforms/cuda.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/triton_utils/mma_probe.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/v1/attention/backends/flash_attn.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/v1/attention/backends/flash_v100/dense_prefill.py` | 2 | 0 | 2 | 0 | 0 | ii: main-Änderung seit 2d218ab03 |
| `vllm/v1/attention/backends/flash_v100/ops.py` | 2 | 0 | 2 | 0 | 0 | ii: main-Änderung seit 2d218ab03 |
| `vllm/v1/worker/gpu/attn_utils.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/v1/worker/gpu/model_runner.py` | 23 | 23 | 0 | 0 | 0 | a43e966 feat(pp): send decode/verify handovers as one... · 508ac9f perf(runner): one copy for the step's index a... · df92723 feat(pp): verify packed handover layouts at s... |
| `vllm/v1/worker/gpu/model_states/default.py` | 1 | 1 | 0 | 0 | 0 | 62b127e [Perf][MRV2] Compressed-index decode graphs a... |
| `vllm/v1/worker/gpu/model_states/interface.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/v1/worker/gpu/model_states/mamba_hybrid.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/v1/worker/gpu/model_states/whisper.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/v1/worker/gpu/prefill_overlap.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/v1/worker/gpu_ubatch_wrapper.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/v1/worker/ubatching.py` | 1 | 0 | 0 | 0 | 1 | iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut) |

### pr-mla-kv-b-skinny (pr-n-kvb) (Kopf c70171be2)

| Datei | Hunks | i | ii | iii | iv | Beleg |
|---|---|---|---|---|---|---|
| `CMakeLists.txt` | 3 | 1 | 1 | 0 | 1 | db216c2 perf(glm53): expand csf_32x2_v1 expert scales... · ii: main-Änderung seit 2d218ab03 · iv: CMake-Kommentar (Fork 8515f1cc1/2f70712b0), kein Verhalten |
| `benchmarks/kernels/benchmark_mla_kv_b_skinny.py` | 6 | 0 | 0 | 0 | 6 | iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut); Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `csrc/host_reduce.cu` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `csrc/ops.h` | 1 | 1 | 0 | 0 | 0 | db216c2 perf(glm53): expand csf_32x2_v1 expert scales... · 31f9613 perf(glm53): write the CSF exception records ... · d39f2b8 fix(merge): adapt main's DeepSeek-V4.1 caller... |
| `csrc/sm70_turbomind/ops/fp8_qpn8_codes_sm70.cuh` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `csrc/sm70_turbomind/ops/fp8_qpn8_mma_sm75.cu` | 14 | 14 | 0 | 0 | 0 | c213a33 perf(sm75): Tensorkern-GEMM für MXFP8-Zeileng... |
| `csrc/sm70_turbomind/ops/fp8_qpn8_sm70.cu` | 45 | 45 | 0 | 0 | 0 | 768d1a3 feat(sm70): QPN8-Skalenmodus je Spalte und 32... · 8cbf9c1 style(sm70): clang-format our lines in fp8_qp... · 6b8d419 perf(qpn8): DeepSeek-V4 wo_a on Turing straig... |
| `csrc/sm70_turbomind/ops/indexer_prefill_scores_sm75.cu` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `csrc/sm70_turbomind/ops/mma_fragments_sm70.cuh` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `csrc/sm70_turbomind/ops/nvfp4_qpn4_sm70.cu` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `csrc/sm70_turbomind/ops/paged_prefill_attention_sm75.cu` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `csrc/sm70_turbomind/ops/skinny_moe_qpn_sm70.cu` | 38 | 38 | 0 | 0 | 0 | cab7b54 feat(moe): six-bit expert scale planes for th... · e6483cf perf(glm53): decode reads the expert scales a... · de1405f perf(moe): explicit load ring in the Volta de... |
| `csrc/torch_bindings.cpp` | 2 | 2 | 0 | 0 | 0 | cab7b54 feat(moe): six-bit expert scale planes for th... · db216c2 perf(glm53): expand csf_32x2_v1 expert scales... · 31f9613 perf(glm53): write the CSF exception records ... |
| `docs/design/architecture/runtime_reference.md` | 1 | 0 | 0 | 0 | 1 | iv: generierte Referenz; check-architecture-reference je Zweig grün |
| `docs/design/sm70_triton_tensor_cores.md` | – | – | – | – | – | Datei fehlt im Fork (nur im Stapel; (iv) Stapel-Commit ohne Fork-Zwilling) |
| `tests/distributed/test_host_reduce.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/kernels/attention/test_dsv4_sparse_prefill_bmm.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/kernels/attention/test_paged_prefill_attention_sm75.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/kernels/moe/test_skinny_moe_triton.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/kernels/moe/test_skinny_sm70_moe.py` | 8 | 8 | 0 | 0 | 0 | 64cca01 fix(glm53): no csf entries needed once every ... · cab7b54 feat(moe): six-bit expert scale planes for th... · e6483cf perf(glm53): decode reads the expert scales a... |
| `tests/kernels/quantization/test_sm70_qpn8_block_fp8.py` | 9 | 5 | 0 | 2 | 2 | iii: Fork-Auflösung 8018abc2d (Turing-Skip des Gated-Tests); im Stapel als Turing-Prüfung ersetzt (fixup cc4751cf0) · iii: Turing-Prüfung des Gated-Tests (fixup cc4751cf0), ersetzt den Fork-Skip · 6597ec4 [Bugfix][SM70] Keep the block-FP8 prefill wor... · 0a694f2 [Bugfix][SM70] Hold the block-FP8 prefill wor... · iv: Block verschoben, Text gleich; Test verschoben (Position), Text gleich |
| `tests/kernels/test_deepseek_v4_sm70_indexer.py` | 17 | 12 | 0 | 0 | 5 | 9bc45c6 perf(deepseek-v4): score decode index keys wi... · 8018abc merge: 1Cat main 119cae740 into fork-main-v6 · 31d5473 [Perf][SM70] Decode score tile for the DSv4 i... · iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut); Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `tests/kernels/test_kernel_config_triton_sm7x.py` | 2 | 0 | 0 | 0 | 2 | iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut); Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `tests/kernels/test_mhc_prenorm_tensor_cores.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/kernels/test_mla_kv_b_skinny.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/kernels/utils.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/models/test_deepseek_v4_sm70_routes.py` | 2 | 2 | 0 | 0 | 0 | 772cbe8 perf(deepseek-v4): halve the sparse prefill w... |
| `tests/models/test_deepseek_v4_sm70_sparse_policy.py` | 1 | 0 | 0 | 0 | 1 | iv: Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `tests/quantization/test_sm70_fp8_prefill_exact_dense.py` | 1 | 0 | 0 | 0 | 1 | iv: Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `tests/quantization/test_turing_qpn_kernel_selection.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/v1/attention/test_flash_v100_dense_prefill_availability.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/v1/worker/test_gpu_model_runner_v2.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/v1/worker/test_gpu_model_runner_v2_eplb.py` | 1 | 0 | 0 | 0 | 1 | iv: Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `tests/v1/worker/test_prefill_overlap.py` | 1 | 0 | 0 | 0 | 1 | iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut) |
| `tests/v1/worker/test_qwen4_exp_v2.py` | 4 | 0 | 1 | 0 | 3 | ii: main-Änderung seit 2d218ab03 · iv: Block verschoben, Text gleich; Testattrappe: Stapel setzt prefill_overlap=None (ee27e4904), Fork anders formuliert |
| `vllm/_sm70/linear.py` | 2 | 2 | 0 | 0 | 0 | 768d1a3 feat(sm70): QPN8-Skalenmodus je Spalte und 32... |
| `vllm/_sm70/policy.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/_sm70_ops.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/config/attention.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/config/kernel.py` | 17 | 16 | 0 | 0 | 1 | c071cad [Perf][Qwen4Exp] Keep the PLE disk tier's hot... · c504ce1 [Feature][Spec Decode] Add the adaptive_verif... · a43e966 feat(pp): send decode/verify handovers as one... · iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut) |
| `vllm/config/sm70_sparse.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/distributed/device_communicators/host_reduce.py` | 5 | 4 | 0 | 0 | 1 | 5841543 [Misc][Distributed] Log the channel mapping w... · iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut) |
| `vllm/distributed/parallel_state.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/model_executor/kernels/linear/qpn/fp8_block.py` | 16 | 14 | 0 | 0 | 2 | 0a694f2 [Bugfix][SM70] Hold the block-FP8 prefill wor... · e336f61 perf(qpn8): grouped wo_a prefill as one batch... · 4392c58 <older than 2d218ab03> · iv: Block verschoben, Text gleich; Modul-Docstring (Fork-Auflösung beschreibt Fork-Logik) |
| `vllm/model_executor/kernels/linear/qpn/nvfp4_dequant.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/model_executor/kernels/linear/qpn/pre_ampere.py` | 11 | 11 | 0 | 0 | 0 | 4417a4c feat(glm53): load GLM-5.3-Flash-FP4-Local on ... · c213a33 perf(sm75): Tensorkern-GEMM für MXFP8-Zeileng... |
| `vllm/model_executor/kernels/mhc/tilelang.py` | 7 | 7 | 0 | 0 | 0 | 4417a4c feat(glm53): load GLM-5.3-Flash-FP4-Local on ... |
| `vllm/model_executor/kernels/mhc/triton.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/model_executor/layers/attention/mla_attention.py` | 2 | 2 | 0 | 0 | 0 | 4342ad6 fix(mla): Profil-Lauf reserviert für Sparse-M... |
| `vllm/model_executor/layers/attention/mla_kv_b_skinny.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/model_executor/layers/fused_moe/experts/skinny_moe_triton.py` | 1 | 0 | 0 | 0 | 1 | iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut) |
| `vllm/model_executor/layers/fused_moe/experts/skinny_sm70_moe.py` | 21 | 19 | 0 | 0 | 2 | 5fef3ad perf(glm53): prefill reads the five-bit scale... · 9712e0e perf(sm70): Experten-Gruppierung des Skinny-M... · 4417a4c feat(glm53): load GLM-5.3-Flash-FP4-Local on ... · iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut) |
| `vllm/model_executor/layers/fused_moe/sm70/reduction.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/model_executor/layers/mhc.py` | 2 | 2 | 0 | 0 | 0 | 4417a4c feat(glm53): load GLM-5.3-Flash-FP4-Local on ... |
| `vllm/model_executor/layers/quantization/sm70_turbomind.py` | 7 | 7 | 0 | 0 | 0 | c213a33 perf(sm75): Tensorkern-GEMM für MXFP8-Zeileng... · 4417a4c feat(glm53): load GLM-5.3-Flash-FP4-Local on ... · 768d1a3 feat(sm70): QPN8-Skalenmodus je Spalte und 32... |
| `vllm/model_executor/layers/sparse_attn_indexer.py` | 1 | 1 | 0 | 0 | 0 | 9bc45c6 perf(deepseek-v4): score decode index keys wi... |
| `vllm/model_executor/layers/sparse_attn_indexer_kpool.py` | 5 | 5 | 0 | 0 | 0 | 4417a4c feat(glm53): load GLM-5.3-Flash-FP4-Local on ... · 9bc45c6 perf(deepseek-v4): score decode index keys wi... |
| `vllm/models/deepseek_v4/attention.py` | 1 | 1 | 0 | 0 | 0 | 772cbe8 perf(deepseek-v4): halve the sparse prefill w... |
| `vllm/models/deepseek_v4/common/ops/__init__.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/models/deepseek_v4/common/ops/cache_utils.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/models/deepseek_v4/common/ops/sparse_decode_bmm.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/models/deepseek_v4/common/ops/sparse_prefill_bmm.py` | 1 | 1 | 0 | 0 | 0 | 772cbe8 perf(deepseek-v4): halve the sparse prefill w... |
| `vllm/models/deepseek_v4/nvidia/model.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/models/deepseek_v4/sm70/indexer.py` | 21 | 17 | 0 | 0 | 4 | 9bc45c6 perf(deepseek-v4): score decode index keys wi... · 31d5473 [Perf][SM70] Decode score tile for the DSv4 i... · iv: Block verschoben, Text gleich; Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut); Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `vllm/models/deepseek_v4/sm70/sparse.py` | 3 | 3 | 0 | 0 | 0 | 772cbe8 perf(deepseek-v4): halve the sparse prefill w... |
| `vllm/models/deepseek_v41/attention_ops.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/triton_utils/mma_probe.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/v1/attention/backends/flash_attn.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/v1/attention/backends/flash_v100/dense_prefill.py` | 2 | 0 | 2 | 0 | 0 | ii: main-Änderung seit 2d218ab03 |
| `vllm/v1/attention/backends/flash_v100/ops.py` | 2 | 0 | 2 | 0 | 0 | ii: main-Änderung seit 2d218ab03 |
| `vllm/v1/worker/gpu/attn_utils.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/v1/worker/gpu/model_runner.py` | 23 | 23 | 0 | 0 | 0 | a43e966 feat(pp): send decode/verify handovers as one... · 508ac9f perf(runner): one copy for the step's index a... · df92723 feat(pp): verify packed handover layouts at s... |
| `vllm/v1/worker/gpu/model_states/default.py` | 1 | 1 | 0 | 0 | 0 | 62b127e [Perf][MRV2] Compressed-index decode graphs a... |
| `vllm/v1/worker/gpu/model_states/interface.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/v1/worker/gpu/model_states/mamba_hybrid.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/v1/worker/gpu/model_states/whisper.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/v1/worker/gpu/prefill_overlap.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/v1/worker/gpu_ubatch_wrapper.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/v1/worker/ubatching.py` | 1 | 0 | 0 | 0 | 1 | iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut) |

### pr-dsv4-wo-a-turing-fp8 (pr-n-woa) (Kopf 153eccb0c)

| Datei | Hunks | i | ii | iii | iv | Beleg |
|---|---|---|---|---|---|---|
| `CMakeLists.txt` | 4 | 2 | 1 | 0 | 1 | db216c2 perf(glm53): expand csf_32x2_v1 expert scales... · f9ac81b perf(attention): Turing paged prefill attenti... · 7d83b70 perf(dsv4): fused lightning-indexer prefill s... · ii: main-Änderung seit 2d218ab03 · iv: CMake-Kommentar (Fork 8515f1cc1/2f70712b0), kein Verhalten |
| `benchmarks/kernels/benchmark_dsv4_wo_a_grouped.py` | 5 | 0 | 0 | 0 | 5 | iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut) |
| `csrc/ops.h` | 1 | 1 | 0 | 0 | 0 | f9ac81b perf(attention): Turing paged prefill attenti... · afe87ab perf(turing): split-KV in the paged prefill a... · 7d83b70 perf(dsv4): fused lightning-indexer prefill s... |
| `csrc/sm70_turbomind/ops/fp8_qpn8_codes_sm70.cuh` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `csrc/sm70_turbomind/ops/fp8_qpn8_mma_sm75.cu` | 14 | 14 | 0 | 0 | 0 | c213a33 perf(sm75): Tensorkern-GEMM für MXFP8-Zeileng... |
| `csrc/sm70_turbomind/ops/fp8_qpn8_sm70.cu` | 29 | 23 | 0 | 0 | 6 | 768d1a3 feat(sm70): QPN8-Skalenmodus je Spalte und 32... · 8cbf9c1 style(sm70): clang-format our lines in fp8_qp... · iv: Stapel-Commit ohne Fork-Zwilling ersetzt Hauptlinien-Zeilen; Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut) |
| `csrc/sm70_turbomind/ops/mma_fragments_sm70.cuh` | 1 | 1 | 0 | 0 | 0 | f9ac81b perf(attention): Turing paged prefill attenti... |
| `csrc/sm70_turbomind/ops/nvfp4_qpn4_sm70.cu` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `csrc/sm70_turbomind/ops/skinny_moe_qpn_sm70.cu` | 38 | 38 | 0 | 0 | 0 | cab7b54 feat(moe): six-bit expert scale planes for th... · e6483cf perf(glm53): decode reads the expert scales a... · de1405f perf(moe): explicit load ring in the Volta de... |
| `csrc/torch_bindings.cpp` | 2 | 2 | 0 | 0 | 0 | cab7b54 feat(moe): six-bit expert scale planes for th... · db216c2 perf(glm53): expand csf_32x2_v1 expert scales... · 31f9613 perf(glm53): write the CSF exception records ... |
| `tests/kernels/moe/test_skinny_sm70_moe.py` | 9 | 9 | 0 | 0 | 0 | 64cca01 fix(glm53): no csf entries needed once every ... · cab7b54 feat(moe): six-bit expert scale planes for th... · e6483cf perf(glm53): decode reads the expert scales a... |
| `tests/kernels/quantization/test_sm70_qpn8_block_fp8.py` | 9 | 5 | 0 | 2 | 2 | iii: Fork-Auflösung 8018abc2d (Turing-Skip des Gated-Tests); im Stapel als Turing-Prüfung ersetzt (fixup cc4751cf0) · iii: Turing-Prüfung des Gated-Tests (fixup cc4751cf0), ersetzt den Fork-Skip · 6597ec4 [Bugfix][SM70] Keep the block-FP8 prefill wor... · 0a694f2 [Bugfix][SM70] Hold the block-FP8 prefill wor... · iv: Block verschoben, Text gleich; Test verschoben (Position), Text gleich |
| `tests/kernels/quantization/test_sm70_qpn8_grouped_bmm.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/quantization/test_fp8_marlin_bmm_dequant.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/quantization/test_qpn8_grouped_bmm.py` | 1 | 0 | 0 | 0 | 1 | iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut) |
| `tests/quantization/test_sm70_fp8_prefill_exact_dense.py` | 1 | 0 | 0 | 0 | 1 | iv: Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `tests/quantization/test_turing_qpn_kernel_selection.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/_sm70/linear.py` | 2 | 2 | 0 | 0 | 0 | 768d1a3 feat(sm70): QPN8-Skalenmodus je Spalte und 32... |
| `vllm/_sm70/policy.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/_sm70_ops.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/model_executor/kernels/linear/qpn/fp8_block.py` | 11 | 6 | 0 | 0 | 5 | 0a694f2 [Bugfix][SM70] Hold the block-FP8 prefill wor... · 6597ec4 [Bugfix][SM70] Keep the block-FP8 prefill wor... · iv: Block verschoben, Text gleich; Modul-Docstring (Fork-Auflösung beschreibt Fork-Logik); Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut) |
| `vllm/model_executor/kernels/linear/qpn/pre_ampere.py` | 11 | 11 | 0 | 0 | 0 | 4417a4c feat(glm53): load GLM-5.3-Flash-FP4-Local on ... · c213a33 perf(sm75): Tensorkern-GEMM für MXFP8-Zeileng... |
| `vllm/model_executor/layers/fused_moe/experts/skinny_sm70_moe.py` | 25 | 22 | 0 | 0 | 3 | 5fef3ad perf(glm53): prefill reads the five-bit scale... · 64009b8 perf(moe): Volta skinny MoE on a Triton group... · 7ea6a68 perf(moe): reduce skinny expert rows with the... · iv: Stapel-Commit ohne Fork-Zwilling ersetzt Hauptlinien-Zeilen; Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut) |
| `vllm/model_executor/layers/quantization/fp8.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/model_executor/layers/quantization/sm70_turbomind.py` | 7 | 7 | 0 | 0 | 0 | c213a33 perf(sm75): Tensorkern-GEMM für MXFP8-Zeileng... · 4417a4c feat(glm53): load GLM-5.3-Flash-FP4-Local on ... · 768d1a3 feat(sm70): QPN8-Skalenmodus je Spalte und 32... |

### pr-sm70-moe-kernel-pack (pr-n-moe) (Kopf 8fde65a1b)

| Datei | Hunks | i | ii | iii | iv | Beleg |
|---|---|---|---|---|---|---|
| `CMakeLists.txt` | 2 | 0 | 1 | 0 | 1 | ii: main-Änderung seit 2d218ab03 · iv: CMake-Kommentar (Fork 8515f1cc1/2f70712b0), kein Verhalten |
| `benchmarks/kernels/benchmark_glm_skinny_moe_routing.py` | 3 | 0 | 0 | 0 | 3 | iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut); Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `csrc/host_reduce.cu` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `csrc/ops.h` | 1 | 0 | 0 | 0 | 1 | iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut) |
| `csrc/sm70_turbomind/ops/csf32_expand_sm70.cu` | 1 | 0 | 0 | 0 | 1 | iv: Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `csrc/sm70_turbomind/ops/fp8_qpn8_codes_sm70.cuh` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `csrc/sm70_turbomind/ops/fp8_qpn8_mma_sm75.cu` | 3 | 0 | 0 | 0 | 3 | iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut) |
| `csrc/sm70_turbomind/ops/fp8_qpn8_sm70.cu` | 28 | 28 | 0 | 0 | 0 | 6b8d419 perf(qpn8): DeepSeek-V4 wo_a on Turing straig... · e336f61 perf(qpn8): grouped wo_a prefill as one batch... |
| `csrc/sm70_turbomind/ops/indexer_prefill_scores_sm75.cu` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `csrc/sm70_turbomind/ops/mma_fragments_sm70.cuh` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `csrc/sm70_turbomind/ops/nvfp4_qpn4_sm70.cu` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `csrc/sm70_turbomind/ops/paged_prefill_attention_sm75.cu` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `csrc/sm70_turbomind/ops/skinny_moe_qpn_sm70.cu` | 25 | 1 | 0 | 0 | 24 | e6483cf perf(glm53): decode reads the expert scales a... · iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut) |
| `csrc/torch_bindings.cpp` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `docs/design/architecture/runtime_reference.md` | 1 | 0 | 0 | 0 | 1 | iv: generierte Referenz; check-architecture-reference je Zweig grün |
| `docs/design/sm70_glm53_pp5_dflash2_mixed.md` | 7 | 7 | 0 | 0 | 0 | ea42af1 docs(glm53): PP5 with DFlash2 on three V100 a... |
| `tests/distributed/test_host_reduce.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/kernels/attention/test_dsv4_sparse_prefill_bmm.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/kernels/attention/test_paged_prefill_attention_sm75.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/kernels/moe/test_skinny_moe_triton.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/kernels/moe/test_skinny_sm70_moe.py` | 1 | 0 | 0 | 0 | 1 | iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut) |
| `tests/kernels/moe/test_sm70_group_routing.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/kernels/quantization/test_csf32_scales.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/kernels/quantization/test_s5_scales.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/kernels/quantization/test_sm70_qpn8_block_fp8.py` | 9 | 5 | 0 | 2 | 2 | iii: Fork-Auflösung 8018abc2d (Turing-Skip des Gated-Tests); im Stapel als Turing-Prüfung ersetzt (fixup cc4751cf0) · iii: Turing-Prüfung des Gated-Tests (fixup cc4751cf0), ersetzt den Fork-Skip · 6597ec4 [Bugfix][SM70] Keep the block-FP8 prefill wor... · 0a694f2 [Bugfix][SM70] Hold the block-FP8 prefill wor... · iv: Block verschoben, Text gleich; Test verschoben (Position), Text gleich |
| `tests/kernels/quantization/test_sm70_qpn8_mxfp8_linear.py` | 1 | 0 | 0 | 0 | 1 | iv: Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `tests/kernels/test_deepseek_v4_sm70_indexer.py` | 17 | 12 | 0 | 0 | 5 | 9bc45c6 perf(deepseek-v4): score decode index keys wi... · 8018abc merge: 1Cat main 119cae740 into fork-main-v6 · 31d5473 [Perf][SM70] Decode score tile for the DSv4 i... · iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut); Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `tests/kernels/test_kernel_config_triton_sm7x.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/kernels/test_mhc_prenorm_tensor_cores.py` | 3 | 3 | 0 | 0 | 0 | 0c5cbfe perf(mhc): let the mHC layer ops take the sm7... |
| `tests/kernels/test_mhc_sm70_fp16.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/kernels/utils.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/models/glm5next/test_fp8_attention_loader.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/models/glm5next/test_mxfp8_held_modules.py` | 1 | 1 | 0 | 0 | 0 | 0181fdc perf(glm53): serve the MXFP8 LM head with the... |
| `tests/models/glm5next/test_pp.py` | 1 | 0 | 0 | 0 | 1 | iv: Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `tests/models/glm5next/test_sm70_sparse.py` | 2 | 1 | 0 | 0 | 1 | 4221842 perf(glm53): Sparse-MLA-Prefill mit FP8-KV üb... · iv: Skip-Bedingung: Stapel strenger (CPU-Hygiene, Teil 1e) |
| `tests/models/glm5next/test_vision_row_chunks.py` | 1 | 0 | 0 | 0 | 1 | iv: Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `tests/models/test_deepseek_v4_sm70_routes.py` | 2 | 2 | 0 | 0 | 0 | 772cbe8 perf(deepseek-v4): halve the sparse prefill w... |
| `tests/models/test_deepseek_v4_sm70_sparse_policy.py` | 1 | 0 | 0 | 0 | 1 | iv: Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `tests/quantization/test_modelopt.py` | 2 | 2 | 0 | 0 | 0 | 0181fdc perf(glm53): serve the MXFP8 LM head with the... |
| `tests/quantization/test_mxfp8_weight_pairs.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/quantization/test_sm70_fp8_prefill_exact_dense.py` | 1 | 0 | 0 | 0 | 1 | iv: Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `tests/quantization/test_sm70_glm53_modelopt_nvfp4.py` | 1 | 0 | 0 | 0 | 1 | iv: Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `tests/quantization/test_sm70_modelopt_mixed_nvfp4.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/quantization/test_sm72_volta_admission.py` | 2 | 0 | 0 | 0 | 2 | iv: Stapel-Commit ohne Fork-Zwilling ersetzt Hauptlinien-Zeilen; Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `tests/quantization/test_turing_qpn_kernel_selection.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/v1/attention/test_flash_v100_dense_prefill_availability.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/v1/attention/test_mla_sparse_profile_budget.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/v1/worker/test_gpu_model_runner_v2.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/v1/worker/test_gpu_model_runner_v2_eplb.py` | 1 | 0 | 0 | 0 | 1 | iv: Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `tests/v1/worker/test_prefill_overlap.py` | 1 | 0 | 0 | 0 | 1 | iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut) |
| `tests/v1/worker/test_qwen4_exp_v2.py` | 4 | 0 | 1 | 0 | 3 | ii: main-Änderung seit 2d218ab03 · iv: Block verschoben, Text gleich; Testattrappe: Stapel setzt prefill_overlap=None (ee27e4904), Fork anders formuliert |
| `vllm/_custom_ops.py` | 1 | 0 | 1 | 0 | 0 | ii: main-Änderung seit 2d218ab03 |
| `vllm/_sm70/linear.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/_sm70/policy.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/_sm70_ops.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/config/attention.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/config/kernel.py` | 17 | 17 | 0 | 0 | 0 | c071cad [Perf][Qwen4Exp] Keep the PLE disk tier's hot... · c504ce1 [Feature][Spec Decode] Add the adaptive_verif... · a43e966 feat(pp): send decode/verify handovers as one... |
| `vllm/config/sm70_sparse.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/distributed/device_communicators/host_reduce.py` | 5 | 4 | 0 | 0 | 1 | 5841543 [Misc][Distributed] Log the channel mapping w... · iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut) |
| `vllm/distributed/parallel_state.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/model_executor/kernels/linear/__init__.py` | 2 | 0 | 0 | 0 | 2 | iv: Stapel-Commit ohne Fork-Zwilling ersetzt Hauptlinien-Zeilen |
| `vllm/model_executor/kernels/linear/mxfp8/Mxfp8LinearKernel.py` | 2 | 0 | 0 | 0 | 2 | iv: Stapel-Commit ohne Fork-Zwilling ersetzt Hauptlinien-Zeilen |
| `vllm/model_executor/kernels/linear/qpn/fp8_block.py` | 16 | 14 | 0 | 0 | 2 | 0a694f2 [Bugfix][SM70] Hold the block-FP8 prefill wor... · e336f61 perf(qpn8): grouped wo_a prefill as one batch... · 4392c58 <older than 2d218ab03> · iv: Block verschoben, Text gleich; Modul-Docstring (Fork-Auflösung beschreibt Fork-Logik) |
| `vllm/model_executor/kernels/linear/qpn/nvfp4_dequant.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/model_executor/kernels/linear/qpn/pre_ampere.py` | 1 | 1 | 0 | 0 | 0 | 4417a4c feat(glm53): load GLM-5.3-Flash-FP4-Local on ... |
| `vllm/model_executor/kernels/mhc/tilelang.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/model_executor/kernels/mhc/triton.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/model_executor/layers/attention/mla_attention.py` | 6 | 6 | 0 | 0 | 0 | 48a92aa perf(mla): skinny Triton kernel for the two k... |
| `vllm/model_executor/layers/fused_moe/experts/skinny_moe_triton.py` | 1 | 0 | 0 | 0 | 1 | iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut) |
| `vllm/model_executor/layers/fused_moe/experts/skinny_sm70_moe.py` | 2 | 0 | 0 | 0 | 2 | iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut) |
| `vllm/model_executor/layers/fused_moe/sm70/reduction.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/model_executor/layers/fused_moe/sm70/routing.py` | 8 | 7 | 0 | 0 | 1 | ca2c1b6 perf(sm70-moe): sort-based routing-group kern... · iv: Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `vllm/model_executor/layers/mhc.py` | 5 | 5 | 0 | 0 | 0 | 0c5cbfe perf(mhc): let the mHC layer ops take the sm7... |
| `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_mxfp8.py` | 3 | 1 | 0 | 0 | 2 | 187b932 <older than 2d218ab03> · iv: Stapel-Commit ohne Fork-Zwilling ersetzt Hauptlinien-Zeilen; Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `vllm/model_executor/layers/quantization/modelopt.py` | 10 | 7 | 0 | 0 | 3 | 0181fdc perf(glm53): serve the MXFP8 LM head with the... · 187b932 <older than 2d218ab03> · iv: Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `vllm/model_executor/layers/quantization/online/mxfp8.py` | 3 | 1 | 0 | 0 | 2 | 187b932 <older than 2d218ab03> · iv: Stapel-Commit ohne Fork-Zwilling ersetzt Hauptlinien-Zeilen; Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `vllm/model_executor/layers/quantization/sm70_turbomind.py` | 4 | 0 | 0 | 0 | 4 | iv: Stapel-Commit ohne Fork-Zwilling ersetzt Hauptlinien-Zeilen; Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `vllm/model_executor/layers/quantization/utils/csf32_expand.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/model_executor/layers/quantization/utils/csf32_scales.py` | 1 | 1 | 0 | 0 | 0 | 4417a4c feat(glm53): load GLM-5.3-Flash-FP4-Local on ... |
| `vllm/model_executor/layers/quantization/utils/mxfp8_utils.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/model_executor/layers/quantization/utils/s5_scales.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/model_executor/layers/sparse_attn_indexer.py` | 1 | 1 | 0 | 0 | 0 | 9bc45c6 perf(deepseek-v4): score decode index keys wi... |
| `vllm/model_executor/layers/sparse_attn_indexer_kpool.py` | 1 | 1 | 0 | 0 | 0 | 9bc45c6 perf(deepseek-v4): score decode index keys wi... |
| `vllm/models/deepseek_v4/attention.py` | 1 | 1 | 0 | 0 | 0 | 772cbe8 perf(deepseek-v4): halve the sparse prefill w... |
| `vllm/models/deepseek_v4/common/ops/__init__.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/models/deepseek_v4/common/ops/cache_utils.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/models/deepseek_v4/common/ops/sparse_decode_bmm.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/models/deepseek_v4/common/ops/sparse_prefill_bmm.py` | 1 | 1 | 0 | 0 | 0 | 772cbe8 perf(deepseek-v4): halve the sparse prefill w... |
| `vllm/models/deepseek_v4/nvidia/model.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/models/deepseek_v4/sm70/indexer.py` | 21 | 17 | 0 | 0 | 4 | 9bc45c6 perf(deepseek-v4): score decode index keys wi... · 31d5473 [Perf][SM70] Decode score tile for the DSv4 i... · iv: Block verschoben, Text gleich; Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut); Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `vllm/models/deepseek_v4/sm70/sparse.py` | 3 | 3 | 0 | 0 | 0 | 772cbe8 perf(deepseek-v4): halve the sparse prefill w... |
| `vllm/models/deepseek_v41/attention_ops.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/models/glm5next/nvidia/attention.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/models/glm5next/nvidia/kda.py` | 13 | 13 | 0 | 0 | 0 | b817aa1 perf(glm5next): fuse the KDA decode glue kern... · 3f6bdcc perf(glm53): tune the chunked KDA prefill ker... |
| `vllm/models/glm5next/nvidia/model.py` | 8 | 8 | 0 | 0 | 0 | 0ef69f4 fix(glm5next): compute the MoE router GEMM on... · 16d886e perf(glm53): Einbettungstabelle über den UVA-... · df65c5a feat(glm53): MXFP8-Projektionen als 8 Bit im ... |
| `vllm/models/glm5next/nvidia/mtp.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/models/glm5next/nvidia/multimodal.py` | 2 | 1 | 0 | 0 | 1 | b6583fb [Bugfix] Fix GLM image cache size (backport o... · iv: Block verschoben, Text gleich |
| `vllm/models/glm5next/nvidia/ops/kpool_compress.py` | 6 | 5 | 0 | 0 | 1 | c37a884 <older than 2d218ab03> · 4417a4c feat(glm53): load GLM-5.3-Flash-FP4-Local on ... · a1048ae fix(sm70): ask the own device's capability wi... · iv: Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `vllm/models/glm5next/nvidia/quant.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/models/glm5next/sm70/fp8_kv.py` | 1 | 1 | 0 | 0 | 0 | 4221842 perf(glm53): Sparse-MLA-Prefill mit FP8-KV üb... |
| `vllm/models/glm5next/sm70/sparse.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/platforms/cuda.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/triton_utils/mma_probe.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/v1/attention/backends/flash_attn.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/v1/attention/backends/flash_v100/dense_prefill.py` | 2 | 0 | 2 | 0 | 0 | ii: main-Änderung seit 2d218ab03 |
| `vllm/v1/attention/backends/flash_v100/ops.py` | 2 | 0 | 2 | 0 | 0 | ii: main-Änderung seit 2d218ab03 |
| `vllm/v1/worker/gpu/attn_utils.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/v1/worker/gpu/model_runner.py` | 23 | 23 | 0 | 0 | 0 | a43e966 feat(pp): send decode/verify handovers as one... · 508ac9f perf(runner): one copy for the step's index a... · df92723 feat(pp): verify packed handover layouts at s... |
| `vllm/v1/worker/gpu/model_states/default.py` | 1 | 1 | 0 | 0 | 0 | 62b127e [Perf][MRV2] Compressed-index decode graphs a... |
| `vllm/v1/worker/gpu/model_states/interface.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/v1/worker/gpu/model_states/mamba_hybrid.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/v1/worker/gpu/model_states/whisper.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/v1/worker/gpu/prefill_overlap.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/v1/worker/gpu_ubatch_wrapper.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/v1/worker/ubatching.py` | 1 | 0 | 0 | 0 | 1 | iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut) |

### pr-sm70-moe-routing-sort-v2 (pr-n-route) (Kopf 82d4f95db)

| Datei | Hunks | i | ii | iii | iv | Beleg |
|---|---|---|---|---|---|---|
| `CMakeLists.txt` | 2 | 0 | 1 | 0 | 1 | ii: main-Änderung seit 2d218ab03 · iv: CMake-Kommentar (Fork 8515f1cc1/2f70712b0), kein Verhalten |
| `csrc/host_reduce.cu` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `csrc/ops.h` | 1 | 1 | 0 | 0 | 0 | cab7b54 feat(moe): six-bit expert scale planes for th... |
| `csrc/sm70_turbomind/ops/csf32_expand_sm70.cu` | 1 | 0 | 0 | 0 | 1 | iv: Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `csrc/sm70_turbomind/ops/fp8_qpn8_codes_sm70.cuh` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `csrc/sm70_turbomind/ops/fp8_qpn8_mma_sm75.cu` | 3 | 0 | 0 | 0 | 3 | iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut) |
| `csrc/sm70_turbomind/ops/fp8_qpn8_sm70.cu` | 28 | 28 | 0 | 0 | 0 | 6b8d419 perf(qpn8): DeepSeek-V4 wo_a on Turing straig... · e336f61 perf(qpn8): grouped wo_a prefill as one batch... |
| `csrc/sm70_turbomind/ops/indexer_prefill_scores_sm75.cu` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `csrc/sm70_turbomind/ops/mma_fragments_sm70.cuh` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `csrc/sm70_turbomind/ops/nvfp4_qpn4_sm70.cu` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `csrc/sm70_turbomind/ops/paged_prefill_attention_sm75.cu` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `csrc/sm70_turbomind/ops/skinny_moe_qpn_sm70.cu` | 52 | 45 | 0 | 0 | 7 | cab7b54 feat(moe): six-bit expert scale planes for th... · de1405f perf(moe): explicit load ring in the Volta de... · dcd2b99 perf(moe): three CTAs per SM for the Turing d... · iv: Stapel-Commit ohne Fork-Zwilling ersetzt Hauptlinien-Zeilen; Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut) |
| `csrc/torch_bindings.cpp` | 1 | 1 | 0 | 0 | 0 | cab7b54 feat(moe): six-bit expert scale planes for th... |
| `docs/design/architecture/runtime_reference.md` | 1 | 0 | 0 | 0 | 1 | iv: generierte Referenz; check-architecture-reference je Zweig grün |
| `docs/design/sm70_glm53_pp5_dflash2_mixed.md` | 7 | 7 | 0 | 0 | 0 | ea42af1 docs(glm53): PP5 with DFlash2 on three V100 a... |
| `tests/distributed/test_host_reduce.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/kernels/attention/test_dsv4_sparse_prefill_bmm.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/kernels/attention/test_paged_prefill_attention_sm75.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/kernels/moe/test_skinny_moe_triton.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/kernels/moe/test_skinny_sm70_moe.py` | 13 | 13 | 0 | 0 | 0 | cab7b54 feat(moe): six-bit expert scale planes for th... · 3581080 test(moe): six-bit scale test data that fp16 ... · de1405f perf(moe): explicit load ring in the Volta de... |
| `tests/kernels/moe/test_sm70_group_routing.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/kernels/moe/test_sm70_group_routing_cpu.py` | 4 | 0 | 0 | 0 | 4 | iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut) |
| `tests/kernels/quantization/test_csf32_scales.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/kernels/quantization/test_s5_scales.py` | 12 | 12 | 0 | 0 | 0 | cab7b54 feat(moe): six-bit expert scale planes for th... |
| `tests/kernels/quantization/test_sm70_qpn8_block_fp8.py` | 9 | 5 | 0 | 2 | 2 | iii: Fork-Auflösung 8018abc2d (Turing-Skip des Gated-Tests); im Stapel als Turing-Prüfung ersetzt (fixup cc4751cf0) · iii: Turing-Prüfung des Gated-Tests (fixup cc4751cf0), ersetzt den Fork-Skip · 6597ec4 [Bugfix][SM70] Keep the block-FP8 prefill wor... · 0a694f2 [Bugfix][SM70] Hold the block-FP8 prefill wor... · iv: Block verschoben, Text gleich; Test verschoben (Position), Text gleich |
| `tests/kernels/quantization/test_sm70_qpn8_mxfp8_linear.py` | 1 | 0 | 0 | 0 | 1 | iv: Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `tests/kernels/test_deepseek_v4_sm70_indexer.py` | 17 | 12 | 0 | 0 | 5 | 9bc45c6 perf(deepseek-v4): score decode index keys wi... · 8018abc merge: 1Cat main 119cae740 into fork-main-v6 · 31d5473 [Perf][SM70] Decode score tile for the DSv4 i... · iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut); Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `tests/kernels/test_kernel_config_triton_sm7x.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/kernels/test_mhc_prenorm_tensor_cores.py` | 3 | 3 | 0 | 0 | 0 | 0c5cbfe perf(mhc): let the mHC layer ops take the sm7... |
| `tests/kernels/test_mhc_sm70_fp16.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/kernels/utils.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/models/glm5next/test_fp8_attention_loader.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/models/glm5next/test_mxfp8_held_modules.py` | 1 | 1 | 0 | 0 | 0 | 0181fdc perf(glm53): serve the MXFP8 LM head with the... |
| `tests/models/glm5next/test_pp.py` | 1 | 0 | 0 | 0 | 1 | iv: Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `tests/models/glm5next/test_sm70_sparse.py` | 2 | 1 | 0 | 0 | 1 | 4221842 perf(glm53): Sparse-MLA-Prefill mit FP8-KV üb... · iv: Skip-Bedingung: Stapel strenger (CPU-Hygiene, Teil 1e) |
| `tests/models/glm5next/test_vision_row_chunks.py` | 1 | 0 | 0 | 0 | 1 | iv: Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `tests/models/test_deepseek_v4_sm70_routes.py` | 2 | 2 | 0 | 0 | 0 | 772cbe8 perf(deepseek-v4): halve the sparse prefill w... |
| `tests/models/test_deepseek_v4_sm70_sparse_policy.py` | 1 | 0 | 0 | 0 | 1 | iv: Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `tests/quantization/test_modelopt.py` | 2 | 2 | 0 | 0 | 0 | 0181fdc perf(glm53): serve the MXFP8 LM head with the... |
| `tests/quantization/test_mxfp8_weight_pairs.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/quantization/test_sm70_fp8_prefill_exact_dense.py` | 1 | 0 | 0 | 0 | 1 | iv: Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `tests/quantization/test_sm70_glm53_modelopt_nvfp4.py` | 1 | 0 | 0 | 0 | 1 | iv: Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `tests/quantization/test_sm70_modelopt_mixed_nvfp4.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/quantization/test_sm72_volta_admission.py` | 2 | 0 | 0 | 0 | 2 | iv: Stapel-Commit ohne Fork-Zwilling ersetzt Hauptlinien-Zeilen; Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `tests/quantization/test_turing_qpn_kernel_selection.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/v1/attention/test_flash_v100_dense_prefill_availability.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/v1/attention/test_mla_sparse_profile_budget.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/v1/worker/test_gpu_model_runner_v2.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `tests/v1/worker/test_gpu_model_runner_v2_eplb.py` | 1 | 0 | 0 | 0 | 1 | iv: Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `tests/v1/worker/test_prefill_overlap.py` | 1 | 0 | 0 | 0 | 1 | iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut) |
| `tests/v1/worker/test_qwen4_exp_v2.py` | 4 | 0 | 1 | 0 | 3 | ii: main-Änderung seit 2d218ab03 · iv: Block verschoben, Text gleich; Testattrappe: Stapel setzt prefill_overlap=None (ee27e4904), Fork anders formuliert |
| `vllm/_custom_ops.py` | 1 | 0 | 1 | 0 | 0 | ii: main-Änderung seit 2d218ab03 |
| `vllm/_sm70/linear.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/_sm70/policy.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/_sm70_ops.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/config/attention.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/config/kernel.py` | 17 | 17 | 0 | 0 | 0 | c071cad [Perf][Qwen4Exp] Keep the PLE disk tier's hot... · c504ce1 [Feature][Spec Decode] Add the adaptive_verif... · a43e966 feat(pp): send decode/verify handovers as one... |
| `vllm/config/sm70_sparse.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/distributed/device_communicators/host_reduce.py` | 5 | 4 | 0 | 0 | 1 | 5841543 [Misc][Distributed] Log the channel mapping w... · iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut) |
| `vllm/distributed/parallel_state.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/model_executor/kernels/linear/__init__.py` | 2 | 0 | 0 | 0 | 2 | iv: Stapel-Commit ohne Fork-Zwilling ersetzt Hauptlinien-Zeilen |
| `vllm/model_executor/kernels/linear/mxfp8/Mxfp8LinearKernel.py` | 2 | 0 | 0 | 0 | 2 | iv: Stapel-Commit ohne Fork-Zwilling ersetzt Hauptlinien-Zeilen |
| `vllm/model_executor/kernels/linear/qpn/fp8_block.py` | 16 | 14 | 0 | 0 | 2 | 0a694f2 [Bugfix][SM70] Hold the block-FP8 prefill wor... · e336f61 perf(qpn8): grouped wo_a prefill as one batch... · 4392c58 <older than 2d218ab03> · iv: Block verschoben, Text gleich; Modul-Docstring (Fork-Auflösung beschreibt Fork-Logik) |
| `vllm/model_executor/kernels/linear/qpn/nvfp4_dequant.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/model_executor/kernels/linear/qpn/pre_ampere.py` | 1 | 1 | 0 | 0 | 0 | 4417a4c feat(glm53): load GLM-5.3-Flash-FP4-Local on ... |
| `vllm/model_executor/kernels/mhc/tilelang.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/model_executor/kernels/mhc/triton.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/model_executor/layers/attention/mla_attention.py` | 6 | 6 | 0 | 0 | 0 | 48a92aa perf(mla): skinny Triton kernel for the two k... |
| `vllm/model_executor/layers/fused_moe/experts/skinny_moe_triton.py` | 1 | 0 | 0 | 0 | 1 | iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut) |
| `vllm/model_executor/layers/fused_moe/experts/skinny_sm70_moe.py` | 26 | 23 | 0 | 0 | 3 | cab7b54 feat(moe): six-bit expert scale planes for th... · iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut); Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `vllm/model_executor/layers/fused_moe/sm70/reduction.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/model_executor/layers/fused_moe/sm70/routing.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/model_executor/layers/mhc.py` | 5 | 5 | 0 | 0 | 0 | 0c5cbfe perf(mhc): let the mHC layer ops take the sm7... |
| `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_mxfp8.py` | 3 | 1 | 0 | 0 | 2 | 187b932 <older than 2d218ab03> · iv: Stapel-Commit ohne Fork-Zwilling ersetzt Hauptlinien-Zeilen; Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `vllm/model_executor/layers/quantization/modelopt.py` | 10 | 7 | 0 | 0 | 3 | 0181fdc perf(glm53): serve the MXFP8 LM head with the... · 187b932 <older than 2d218ab03> · iv: Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `vllm/model_executor/layers/quantization/online/mxfp8.py` | 3 | 1 | 0 | 0 | 2 | 187b932 <older than 2d218ab03> · iv: Stapel-Commit ohne Fork-Zwilling ersetzt Hauptlinien-Zeilen; Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `vllm/model_executor/layers/quantization/sm70_turbomind.py` | 4 | 0 | 0 | 0 | 4 | iv: Stapel-Commit ohne Fork-Zwilling ersetzt Hauptlinien-Zeilen; Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `vllm/model_executor/layers/quantization/utils/csf32_expand.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/model_executor/layers/quantization/utils/csf32_scales.py` | 1 | 1 | 0 | 0 | 0 | 4417a4c feat(glm53): load GLM-5.3-Flash-FP4-Local on ... |
| `vllm/model_executor/layers/quantization/utils/mxfp8_utils.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/model_executor/layers/quantization/utils/s5_scales.py` | 20 | 20 | 0 | 0 | 0 | cab7b54 feat(moe): six-bit expert scale planes for th... · e6483cf perf(glm53): decode reads the expert scales a... |
| `vllm/model_executor/layers/sparse_attn_indexer.py` | 1 | 1 | 0 | 0 | 0 | 9bc45c6 perf(deepseek-v4): score decode index keys wi... |
| `vllm/model_executor/layers/sparse_attn_indexer_kpool.py` | 1 | 1 | 0 | 0 | 0 | 9bc45c6 perf(deepseek-v4): score decode index keys wi... |
| `vllm/models/deepseek_v4/attention.py` | 1 | 1 | 0 | 0 | 0 | 772cbe8 perf(deepseek-v4): halve the sparse prefill w... |
| `vllm/models/deepseek_v4/common/ops/__init__.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/models/deepseek_v4/common/ops/cache_utils.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/models/deepseek_v4/common/ops/sparse_decode_bmm.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/models/deepseek_v4/common/ops/sparse_prefill_bmm.py` | 1 | 1 | 0 | 0 | 0 | 772cbe8 perf(deepseek-v4): halve the sparse prefill w... |
| `vllm/models/deepseek_v4/nvidia/model.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/models/deepseek_v4/sm70/indexer.py` | 21 | 17 | 0 | 0 | 4 | 9bc45c6 perf(deepseek-v4): score decode index keys wi... · 31d5473 [Perf][SM70] Decode score tile for the DSv4 i... · iv: Block verschoben, Text gleich; Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut); Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `vllm/models/deepseek_v4/sm70/sparse.py` | 3 | 3 | 0 | 0 | 0 | 772cbe8 perf(deepseek-v4): halve the sparse prefill w... |
| `vllm/models/deepseek_v41/attention_ops.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/models/glm5next/nvidia/attention.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/models/glm5next/nvidia/kda.py` | 13 | 13 | 0 | 0 | 0 | b817aa1 perf(glm5next): fuse the KDA decode glue kern... · 3f6bdcc perf(glm53): tune the chunked KDA prefill ker... |
| `vllm/models/glm5next/nvidia/model.py` | 8 | 8 | 0 | 0 | 0 | 0ef69f4 fix(glm5next): compute the MoE router GEMM on... · 16d886e perf(glm53): Einbettungstabelle über den UVA-... · df65c5a feat(glm53): MXFP8-Projektionen als 8 Bit im ... |
| `vllm/models/glm5next/nvidia/mtp.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/models/glm5next/nvidia/multimodal.py` | 2 | 1 | 0 | 0 | 1 | b6583fb [Bugfix] Fix GLM image cache size (backport o... · iv: Block verschoben, Text gleich |
| `vllm/models/glm5next/nvidia/ops/kpool_compress.py` | 6 | 5 | 0 | 0 | 1 | c37a884 <older than 2d218ab03> · 4417a4c feat(glm53): load GLM-5.3-Flash-FP4-Local on ... · a1048ae fix(sm70): ask the own device's capability wi... · iv: Zeilen aus Stapel-Commit, die dem Fork-Zwilling fehlen (Stapel neuer) |
| `vllm/models/glm5next/nvidia/quant.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/models/glm5next/sm70/fp8_kv.py` | 1 | 1 | 0 | 0 | 0 | 4221842 perf(glm53): Sparse-MLA-Prefill mit FP8-KV üb... |
| `vllm/models/glm5next/sm70/sparse.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/platforms/cuda.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/triton_utils/mma_probe.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/v1/attention/backends/flash_attn.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/v1/attention/backends/flash_v100/dense_prefill.py` | 2 | 0 | 2 | 0 | 0 | ii: main-Änderung seit 2d218ab03 |
| `vllm/v1/attention/backends/flash_v100/ops.py` | 2 | 0 | 2 | 0 | 0 | ii: main-Änderung seit 2d218ab03 |
| `vllm/v1/worker/gpu/attn_utils.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/v1/worker/gpu/model_runner.py` | 23 | 23 | 0 | 0 | 0 | a43e966 feat(pp): send decode/verify handovers as one... · 508ac9f perf(runner): one copy for the step's index a... · df92723 feat(pp): verify packed handover layouts at s... |
| `vllm/v1/worker/gpu/model_states/default.py` | 1 | 1 | 0 | 0 | 0 | 62b127e [Perf][MRV2] Compressed-index decode graphs a... |
| `vllm/v1/worker/gpu/model_states/interface.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/v1/worker/gpu/model_states/mamba_hybrid.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/v1/worker/gpu/model_states/whisper.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/v1/worker/gpu/prefill_overlap.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/v1/worker/gpu_ubatch_wrapper.py` | 0 | 0 | 0 | 0 | 0 | identisch zum Fork |
| `vllm/v1/worker/ubatching.py` | 1 | 0 | 0 | 0 | 1 | iv: Stapel-Fassung neuer als Fork-Zwilling (Review, Format, Wortlaut) |

