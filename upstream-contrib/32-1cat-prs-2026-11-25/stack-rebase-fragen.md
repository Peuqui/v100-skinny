# Offene Fragen aus dem Stapel-Rebase auf main 7db135fa0

Nichts davon blockiert einen Zweig; alle sechs Zweige sind vollständig rebaset. Die Punkte brauchen eine Entscheidung oder
eine GPU-Messung, bevor die PRs aktualisiert werden.

1. `vllm/models/deepseek_v4/sm70/indexer.py:59-64` und `:655-680` (#1282, #1347, #1286, #1348; Commit "vectorize the sm70
   indexer prefill epilogue loads"). main hat die Prefill-Scores von fp16 auf fp32 umgestellt (`torch.mm(..., out_dtype=torch.float32)`).
   Unser Commit begründet BLOCK_K 512 über 2 Warps mit fp16-Scores (16-Byte-Ladevorgänge, 562/782 GB/s gemessen) und der
   Kommentar an `_PREFILL_EPILOGUE_BLOCK_K` sagt "The prefill epilogue reads fp16 scores". Aufgelöst wurde mechanisch (main's fp32 plus
   unsere Blockgröße). Mit fp32 sind es 32 Byte je Thread; ob die Beschleunigung bleibt, ist nicht gemessen (keine GPU erlaubt).
   Vorschlag: Mikrobenchmark mit fp32-Scores auf RTX und V100 nachholen, danach Kommentar und Commit-Text anpassen
   (bei Ausbleiben des Gewinns den Commit streichen).
2. `vllm/models/deepseek_v41/attention_ops.py:1122-1160` (main, 1Cats `_sink_softmax`/`_attend_fused` für DeepSeek-V4.1) und
   `vllm/models/deepseek_v4/common/ops/sparse_prefill_bmm.py:142-270` (unser `_sink_softmax_kernel`, Commit "fuse the sparse
   MLA softmax and zero unused keys"): zwei Implementierungen derselben fusionierten Sink-Softmax. Aufgelöst ohne Änderung an
   beiden: V4.1 behält `_ATTN_FUSED`/`_ATTN_DEC`, der Nicht-Fused-Weg geht durch unsere neue `attend_gathered_keys`
   (Aufrufer nullen die unbenutzten Slots selbst, Zeilen 1460 und 1576). Vorschlag: nach der Abnahme `_attend_fused` auf unsere
   Funktion zurückführen (SSOT) oder bewusst nebeneinander lassen und im PR erwähnen.
3. Information, keine Entscheidung nötig: Turing-Tensorkern-GEMM (#1278, `fp8_block.py:115-137, 251-323, 760, 778`) ist jetzt
   auch durch `qpn8_native_linear_out` (Direktausgabe, neu in main) und `_qpn8_native_linear_write` geführt; die Op
   `sm70_fp8::qpn8_native_linear_out` bekommt dafür ein weiteres Argument `bool sm75_mma=False` (Schema und Fake). Wenn Sie das
   lieber getrennt halten wollen (Direktausgabe auf Turing bleibt beim gepackten GEMM), müsste der Parameter dort wieder raus.
4. Information: #1347/#1286/#1348 lagen im alten Stand auf #1282 ohne dessen letzte zwei Commits (`docs(triton): where to get the sm7x
   Triton ...` 294882b45 und `perf(mhc): let the mHC layer ops take the sm7x tensor-core route` b9fbe637c). Die Stapel
   hängen deshalb an `pr-triton-sm7x-r1125~2`, nicht an dessen Spitze. Soll #1347/#1286/#1348 auf die Spitze von #1282 aufsetzen,
   enthalten sie diese zwei Commits zusätzlich (Inhaltsänderung der PRs).
5. `vllm/v1/worker/gpu/spec_decode/dflash/utils.py:83` (#1221): `share_runtime_resources` läuft jetzt in
   `get_parallel_draft_model`, also auch für den DSpark-Drafter (main hatte es nur im DFlash-Pfad). Gewollte Folge der Fassung
   von main (derselbe Zweck: Laufzeitressourcen des Ziels teilen); bei DSpark nicht getestet (keine GPU).
6. Ruff-Version: das zentrale ruff ist 0.16.10, das Projekt pinnt v0.14.0 (`.pre-commit-config.yaml`). Auf main selbst melden
   `vllm/models/deepseek_v4/sm70/indexer.py` (ISC004 Zeilen 1018/1024 dieses Stands, plus Format) und
   `tests/models/test_deepseek_v4_sm70_routes.py` (Format) Befunde; die Zeilen sind unverändert aus main (Formatdiff Byte-gleich
   zu dem von main), nicht von uns angefasst.
7. Die Layering-Basis auf main ist veraltet (check_layering meldet "7 Dateien haben Kopplung verringert", Basis nicht per
   `--update` nachgezogen); gehört nicht in unsere PRs.

## Entscheidungen (Opus, 25.11.2026 06:40)

1. Indexer-Epilog: nachgemessen auf der fp32-Kachel (`~/.cache/bench-scripts/bench_dsv4_idx_epilogue_fp32_2026-11-25.py`, Logs
   `bench_dsv4_idx_epilogue_fp32_2026-11-25_gpu{2,3}.log`, CUDA-Graph, µs je Aufruf, weniger ist besser; Kern im v16-Baum identisch zum
   rebasten Zweig): RTX 8000 BLOCK_K 256/4 Warps gegen 512/2: 473,4 → 462,8 (1024×1024), 940,0 → 929,9 (1024×2048), 942,7 → 930,5
   (512×4096), also −1,1 bis −2,2 %, 567–571 → 577–580 GB/s; V100 337,1 → 330,0, 665,4 → 659,6, 669,0 → 651,8, −0,9 bis −2,6 %,
   796–807 → 813–824 GB/s. Nicht bytegleich (Summationsreihenfolge über die Köpfe). Der große Gewinn vom 21.10. war die fp16-Kachel
   (2 Keys je Thread = 4 Byte). Commit bleibt (in Produktion, kostet nichts), neuer Kommentar des Rebases stimmt; im Update-Text von
   #1282 die neuen Zahlen nennen.
2. Sink-Softmax doppelt: beide Fassungen sind so in Produktion (v16 = Zweig, 0 Zeilen Unterschied in attention_ops.py); bleibt, kein
   Umbau in dieser Runde.
3. `sm75_mma` durch `qpn8_native_linear_out`: so auch im Fork (prod-v16 Schema mit `sm75_mma` an `_out`), bleibt.
4. Stapel auf `pr-triton-sm7x-r1125~2` wie bisher, keine Inhaltsänderung der PRs.
5. `share_runtime_resources` in `get_parallel_draft_model`: prod-v16 hat genau diese Fassung (kein Unterschied in `dflash/utils.py`),
   DSpark läuft in Produktion damit.
6. ruff: pre-commit mit der gepinnten Version entscheidet (Kettenschritt), nicht das zentrale ruff.
7. Layering-Basis von main: nicht unser PR.
