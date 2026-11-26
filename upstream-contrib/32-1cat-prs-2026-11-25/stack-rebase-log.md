# Stapel-Rebase auf 1Cat main 7db135fa0 (Zweig `pr-base-7db135fa0`), Protokoll der Konfliktlösungen

Arbeitsbaum `1Cat-vLLM-pr-stack`, Stand 25.11.2026. Neue Zweige (nur lokal, nicht gepusht):

| PR | Zweig | SHA | Commits alt/neu |
| --- | --- | --- | --- |
| #1278 | pr-turing-kernels-r1125 | 05ddf2af3707905b7503f909aa82abaaf073e692 | 7 / 7 |
| #1282 | pr-triton-sm7x-r1125 | bfc82098d8b2c7bd3538b2c6f4e2e41cd21aabb2 | 36 / 35 |
| #1347 | pr-dsv4-longctx-decode-r1125 | e5c4cf7c418ee5c563e1c7a48780d7c33ab3e039 | 37 / 36 |
| #1286 | pr-glm53-r1125 | 38bf1751011a5072782fe3917b6a82bbd8ef7882 | 65 / 64 |
| #1348 | pr-glm53-prefill-head-r1125 | c224c23db5e6ed3176a97b42744c83b13e35aeb9 | 65 / 64 |
| #1221 | pr-mrv2-dspark-r1125 | cb1065c20ee2008e10a98948cad0ebdf243e7fe1 | 3 / 3 |

Stapelstruktur (aus den Patch-IDs): #1278 = Commits 1-7. #1282 = diese 7 + d3aaa4eb7 (#1216) + 26 gemeinsame + 2 eigene
(docs(triton), mHC-Route). #1347, #1286, #1348 tragen die 7 + d3aaa4eb7 + die 26 gemeinsamen Commits (= #1282 ohne seine
letzten zwei) und setzen darauf auf. #1286 und #1348 teilen außerdem 28 weitere Commits (bis `test(glm53): the GLM-5 next
sparse backend admits SM75 as well`), danach hat #1286 drei Vision-Commits und #1348 drei andere (KDA-Tuning, MXFP8-LM-Kopf,
Skalenpolster). Neu ist es ein echter Git-Stapel: `pr-turing-kernels-r1125` ist Vorfahr von `pr-triton-sm7x-r1125`; dessen
drittletzter Commit (`test(dsv4): give the sparse policy test layer its tensor-core flag`) ist Vorfahr von
`pr-dsv4-longctx-decode-r1125` und `pr-glm53-r1125`; der viertletzte Commit von `pr-glm53-r1125` (`test(glm53): the GLM-5 next
sparse backend admits SM75 as well`) ist Vorfahr von `pr-glm53-prefill-head-r1125`.

Weggelassener Commit: d3aaa4eb7 `[Kernel] Two-rank FP16 all-reduce through pinned host memory ...` (#1216) in #1282, #1347,
#1286, #1348. Beleg: main enthält ihn als 01f1c7d2b (gesquasht, gleicher Autor/Zeitpunkt, "cherry picked from commit
f53a668eb"). `csrc/host_reduce.cu`, `tests/distributed/test_host_reduce.py` und
`vllm/distributed/device_communicators/host_reduce.py` sind in main Byte für Byte der Stand von d3aaa4eb7
(`git diff d3aaa4eb7 pr-base-7db135fa0 -- <datei>` leer). Die +/- Zeilen des Commits gegen 01f1c7d2b: bis auf zwei Zeilen
der generierten `runtime_reference.md` (Zeilennummern) identisch.

Regeln beim Rebase: rerere je Befehl (`-c rerere.enabled=true -c rerere.autoupdate=true`). `rr-cache` ist repoweit
gemeinsam: rerere hat bereits früher aufgezeichnete Auflösungen für zwei Konfliktgruppen stillschweigend eingespielt
(`707e11939`: attn_utils/default/mamba_hybrid; `48fe4e404`: zwei Testdateien); beide wurden nachträglich Zeile für Zeile
geprüft (siehe unten). `runtime_reference.md` ist generiert: bei jedem Konflikt wurde die main-Fassung genommen und mit
`python -m tools.generate_architecture_reference --write` neu erzeugt; danach lief ein `rebase --exec` über alle Zweige, das
jeden Commit mit veralteter Datei amendet (3 Commits wurden dabei zusätzlich angepasst: `707e11939`, `1f250afc9`,
`a6dbf69f8`, jeweils nur diese Datei). Auf jedem Zweigende meldet der Generator `--check` "match".

## #1278 pr-turing-kernels

### c78b18d91 feat(sm75): tensor-core GEMM on QPN8 block-FP8 codes ...
- `tests/kernels/quantization/test_sm70_qpn8_block_fp8.py`: main hat an derselben Stelle zwei neue Tests angehängt
  (`fused_gated_silu`, Row-Block/Graph-Workspace); wir hängen zwei Turing-Tests an. Beides behalten (main zuerst).
- `vllm/model_executor/kernels/linear/qpn/fp8_block.py`:
  - Konstanten: main hat `_QPN8_BATCH_NATIVE_MAX_M`, `_QPN8_ROW_BLOCK(S)_MAX` und Workspace-Pool ergänzt; wir `_SM75_MMA_MAX_M`. Beides.
  - main hat `_qpn8_native_linear` in `_qpn8_native_linear_write` (gemeinsamer Körper) plus die Op `qpn8_native_linear_out`
    (Direktausgabe) zerlegt. Git hat unseren `sm75_mma`-Zweig deshalb in `_qpn8_native_gated_silu` einsortiert (Konflikt).
    Lösung: dort verworfen (main-Seite), stattdessen in `_qpn8_native_linear_write` direkt nach dem M<=8-Zweig und vor
    `fallback_weight` eingefügt (gleiche Stelle wie im Original); Parameter `sm75_mma` durch `_write`, `_qpn8_native_linear`
    (Schema war schon durch Auto-Merge ergänzt), `_qpn8_native_linear_out` (Schema, Signatur, Fake) und den Direktausgabe-Aufruf
    in `apply_weights` durchgereicht. Ohne das hätte der Direktausgabe-Pfad auf Turing die Tensorkern-GEMM übersprungen.
    Siehe Fragen-Datei, Punkt 3 (Information).

### fd0266731 perf(sm75): fit the QPN8 decode split-K to Turing's occupancy
- `fp8_block.py`: Importe (main: `sm70_layer_workspaces`, wir: `fp8_qpn8_occupancy_split_k`): beide, isort-Reihenfolge.
  `layer._qpn8_cfg`: main setzt `_qpn8_cfg = _sm70_fp8_qpn8_config(...)` und hängt danach den `_qpn8_gated_cfg`-Block an;
  wir packen `fp8_qpn8_occupancy_split_k` um den Split. Lösung: unser Tupel für `_qpn8_cfg`, main-Block für Gated-Pair danach
  unverändert (Gated-Pair ist Volta-only; die Occupancy-Funktion lässt Volta unverändert).

## #1282 pr-triton-sm7x (eigene Commits auf dem neuen #1278-Zweig; #1216 weggelassen)

- 9f9a98f92 perf(attention): Turing paged prefill attention: `runtime_reference.md` (generiert): neu erzeugt.
- 707e11939 perf(v2): overlap TP prefill all-reduces: `vllm/v1/worker/gpu/attn_utils.py`, `model_states/default.py`,
  `model_states/mamba_hybrid.py`: main hat `build_attn_metadata` um `block_tables_cpu` und `is_all_prefilling` erweitert und
  die Aufrufstellen angepasst; wir ergänzen `ubatch_id`. rerere-Auflösung (früher aufgezeichnet), geprüft: der +/- Inhalt
  des neuen Commits für diese drei Dateien ist identisch zum Original (`ubatch_id` neben den neuen main-Parametern,
  `num_metadata_builders`, `align_ubatch_split`, mamba-Prüfung). Außerdem `runtime_reference.md` neu erzeugt.
- 1f250afc9 perf(host_reduce): copy-engine mode: `runtime_reference.md` neu erzeugt.
- 18b9096a0 perf(dsv4): vectorize the sm70 indexer prefill epilogue loads: `vllm/models/deepseek_v4/sm70/indexer.py`:
  main hat `torch.mm(...)` auf `out_dtype=torch.float32` umgestellt (Überlauf zu inf/NaN, Kommentar dazu); wir ersetzen die
  Epilog-Blockgröße (`_PREFILL_EPILOGUE_BLOCK_K`, Warps). Lösung: main-Zeilen (fp32-Ausgabe samt Kommentar) plus unser
  Kernelaufruf mit `_PREFILL_EPILOGUE_BLOCK_K`. Offen: die Messung des Commits (fp16-Scores, 16-Byte-Ladevorgänge) gilt für
  fp32 nicht mehr unbedingt; Fragen-Datei, Punkt 1.
- 61fb11707 perf(dsv4): fuse the sparse MLA softmax ...: `vllm/models/deepseek_v41/attention_ops.py` (`attend_draft_latents`):
  main hat vor `_ATTN_FUSED` einen `_ATTN_DEC`-Zweig (`v41sk.attn_dec`) eingeschoben, der selbst unbenutzte Zeilen nullt; wir
  ziehen das `keys.masked_fill_(unused, 0)` vor den `_ATTN_FUSED`-Zweig und entfernen den logits-Puffer. Lösung: main-Zweig
  bleibt zuerst (kehrt vor dem masked_fill zurück), danach unser `masked_fill_`. Auto-gemergte Stellen
  (`attend_paged_latents`, Aufruf von `attend_gathered_keys` mit `[T,H,W]`-Wahrscheinlichkeiten) unverändert. Fragen-Datei,
  Punkt 2 (zwei fusionierte Softmax).
- 6adb09cc6 perf(dsv4): fused lightning-indexer prefill scores: `runtime_reference.md` neu erzeugt.
- 4df715087 feat(triton): one setting for tensor-core Triton: `runtime_reference.md` neu erzeugt.
- a6dbf69f8 perf(dsv4): Triton prefill index scorer for the SM70-family relu route: `runtime_reference.md` neu erzeugt;
  `indexer.py` Kommentarblock über `_RELU_BLOCK_N`: main hat "fp16" -> "fp32 score tile / 512 KiB per key" (passend zu
  seiner fp32-Umstellung) geändert, wir haben die ersten beiden Zeilen (Routing-Beschreibung) umformuliert. Lösung: unsere
  zwei Zeilen + main-Zeilen mit fp32 und 512 KiB.
- Alle übrigen Commits (`=` im Range-Diff) ohne Konflikt.

## #1347 pr-dsv4-longctx-decode (3 eigene Commits auf `test(dsv4): give the sparse policy test layer its tensor-core flag`)

- 6faf94bfb perf(deepseek-v4): score decode index keys in one tensor-core kernel: `tests/kernels/test_deepseek_v4_sm70_indexer.py`:
  main hat die Testfunktion `test_decode_cublas_keeps_the_fp32_topk_set_and_masks_graph_tail` um den Parameter
  `graph_width` ([2048, 8192], `live_seq_len = graph_width // 2 - 7`) erweitert; wir fügen `fused` hinzu. Beides
  (Dekoratoren und Signatur).
- 6968a8b40 perf(deepseek-v4): decode score tile for the indexer verifier rows: selbe Testfunktion: wir ersetzen die feste
  Form durch das Paar `live_seq_len,graph_width` ([(1017, 2048), (16437, 20480)]), um 16k Schlüssel zu kreuzen. Lösung: ein
  Paar-Parameter mit allen drei Formen, damit main's 8192er-Fall erhalten bleibt: (1017, 2048), (4089, 8192), (16437, 20480);
  main's eigener `graph_width`-Dekorator und `live_seq_len`-Zeile entfallen (ersetzt).

## #1286 pr-glm53 (31 eigene Commits auf demselben Unterbau wie #1347)

- 48fe4e404 test: align the GLM PP and ModelOpt NVFP4 tests with the current code: `tests/quantization/test_sm70_glm53_modelopt_nvfp4.py`,
  `test_sm70_modelopt_nvfp4.py`: rerere-Auflösung (früher aufgezeichnet). Geprüft: main enthält die Korrekturen schon
  (`skinny_backend_admitted`-Patch Zeile 121, `is_exact_sm70_cuda`/`is_turing_cuda_platform` in `test_sm70_modelopt_nvfp4.py`
  Zeilen 138-219); der Commit ändert deshalb nur noch `tests/models/glm5next/test_pp.py`. Alle Aussagen des Commit-Textes
  zu den beiden Testdateien sind damit durch main erfüllt.
- 3315505d2 [Perf][GLM-5.3] Bound the vision tower's activation memory: `vllm/models/glm5next/nvidia/multimodal.py`:
  main hat einen `TYPE_CHECKING`-Block (`_MediaProcessor`), wir `_ROWS_PER_CHUNK`/`_row_chunks`. Beides.
- Auto-gemergte Commits mit geänderten Kontextzeilen (kein Konflikt, geprüft über Range-Diff): `0829aad9d` (Kommentar "M=9..32" ->
  main "M=9..40" im Kontext), `d20f9733c`, `c9252bc01`, `dfb18ebbf` (CMake-Kontext), `3315505d2` (Kontext).

## #1348 pr-glm53-prefill-head (3 eigene Commits auf `test(glm53): the GLM-5 next sparse backend admits SM75 as well` des neuen #1286-Zweigs)
- Keine Konflikte; alle drei Commits Range-Diff `=`.

## #1221 pr-mrv2-dspark

- e3614d9d4 [Spec Decode][MRV2] DSpark speculator: `vllm/v1/worker/gpu/spec_decode/dflash/utils.py`: main ruft in
  `load_dflash_model` jetzt `share_runtime_resources(draft_vllm_config, vllm_config)` vor `get_model` auf; wir ziehen den
  Laden-Teil in `get_parallel_draft_model(vllm_config, model_tag)`. Lösung: Aufruf von `share_runtime_resources` dort,
  vor `get_model` (damit gilt er für DFlash und DSpark). Fragen-Datei, Punkt 5.
- 966659632 [Perf][MRV2] Compressed-index decode graphs at context buckets: `tools/pre_commit/layering_baseline.json`:
  Eintrag `vllm/v1/worker/gpu/cudagraph_utils.py` platform: Basis 33, main 30, unser Commit +1. Lösung: Wert 31, abgesichert
  mit `tools/pre_commit/check_layering.py` (kein Zuwachs gemeldet; "7 Dateien haben Kopplung verringert" betrifft main).
