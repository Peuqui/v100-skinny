# Übergabe: sm75-Koexistenz im v100-skinny-Fork

> **SCHNELLEINSTIEG — Stand 2026-09-03 10:15 (gilt vor allem, was darunter steht).**
> Das Dokument ist ein chronologisches Logbuch ab 2026-09-01; die Abschnitte
> darunter beschreiben teils überholte Zwischenstände. Für den aktuellen Stand
> reicht dieser Block plus die letzten zwei Abschnitte („nsys-Per-Kernel-Profil"
> und „mHC fp16 freigeschaltet").

**Erreicht:** DeepSeek-V4-Flash (NVFP4, 43 Layer) läuft unter vLLM auf allen
5 Karten (2× RTX 8000 sm75 + 3× V100 sm70) mit DSpark-Spekulation K=5,
CUDA-Graphs (breakable capture), grouped NVFP4-MoE-Kernel und seit 10:15 dem
freigeschalteten mHC-fp16-Kernelpfad (Upstream-Backport statt Torch-Generic).
Kohärenz 8/8 gegen die Referenz (5/8 bitidentisch — gleiche Quote wie alle
akzeptierten Läufe). **Essay 21,3 tok/s, Code 26,7 tok/s** (llama.cpp auf
derselben Maschine: 21 / 38 — Prosa ist eingeholt). Start der Arbeit war 3,6.

**So startet man es:** `scripts/serve-deepseek-het-graphs.sh` (Port 19998,
Modell `dsv4-manual`). Topologie darin: `CUDA_VISIBLE_DEVICES=0,1,4,3,2`
(RTX, V100, V100, V100, RTX), `VLLM_PP_LAYER_PARTITION=11,8,8,8,8`, Drafter
(10,7 GiB) auf der letzten RTX, util 0.95, `--num-gpu-blocks-override 512`,
`VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64`, batched 64, Capture-Größen [6],
`VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`. Boot ~9 min. Kohärenzprobe:
`scripts/deepseek_coherence.py --url http://127.0.0.1:19998 --model dsv4-manual`.
Referenz: `results-coherence-nvidia-base.json`.

**Code-Stand:** Fork-Branch `pp-mtp-merge`, committed bis `8e27308`
(davor `dda7ad2`, `0b5093a`); danach nur Doku/Benchmark-Dateien. Alle
Patches liegen in `fork_patches/` (Deploy über `scripts/bootstrap-sm70.sh`,
Tabelle in `fork_patches/README.md`); der MoE-Kernel in
`kernels/skinny_kernels.cu` (`moe_simt`, Test
`scripts/nvfp4_skinny_moe_grouped_test.py`). Das venv `.venv-sm70-130`
entspricht dem deployten Stand.

**Nächste Hebel (Hebel 1 = mHC ist ERLEDIGT, siehe 10:15-Abschnitt):**
1. ~~mHC-Pfad unter fp16~~ ERLEDIGT 10:15 — Upstream-Backport, 13/20 →
   21,3/26,7 tok/s. Das nsys-Profil ist damit veraltet; vor weiteren
   Hebeln NEU profilieren (`QnxKernelTrace` war mutmaßlich Teil des
   Torch-Pfads und könnte mit verschwunden sein).
2. Sparse-Decode: ROCm-Triton-Impl (`amd/rocm.py`) gegen `sm70/sparse_kernels.py`
   messen (war 0,65 ms/Layer).
3. MoE w13/w2 + Aktivierung in einen Launch (war ~0,3 ms).

**Geschlossene Fragen (nicht neu aufrollen):** Drafter-Akzeptanz ist auf
llama.cpp-Niveau (32 vs 28 % Prosa, 64 vs 69 % Code) — kein 8-Bit-Drafter.
Die 21 „exakt sm70"-Weichen sind worker-lokal bzw. strukturell gegated.
PR #455 (QSA) ist zu Recht geschlossen (Dispatch-Irrtum, siehe 06:50).

**Werkzeug-Fallen:** Boots mit `setsid nohup … & disown` in EIGENEM Aufruf
starten, in einem SEPARATEN Aufruf warten (Tool-Timeout killt sonst die
Prozessgruppe); Prozesse über `nvidia-smi --query-compute-apps=pid` und
`pgrep -f "[a]pi_server"` beenden, nie `pkill -f`; bei „Deadlock" zuerst
`grep ERROR` je Worker; nach Interface-Patches an Modellklassen
`~/.cache/vllm/modelinfos/*.json` löschen; Triton-A/B je Karte mit eigenem
`CUDA_VISIBLE_DEVICES`; ModelOpt `weight_scale_2` ist der Dequant-
Multiplikator; `nsys`/`ncu` liegen unter /usr/bin (Rezept: 06:50-Abschnitt).

---

> Stand 2026-09-01 nachts. Ziel der nächsten Instanz: den Fork so
> ertüchtigen, dass **gemischte Karten** (2× RTX 8000 sm75 + 3× Tesla V100
> sm70) zusammenarbeiten. Auslöser war der gescheiterte Versuch,
> DeepSeek-V4-Flash unter vLLM zu fahren.

## Kernbefund

Der Fork ist durchgängig für eine **homogene V100-Maschine** geschrieben.
An 21 Stellen fragt er `current_platform.is_device_capability((7, 0))` ab
— also **genau** Volta. Auf unseren beiden RTX 8000 (sm75) greift keine
davon, und die Worker auf diesen Karten laufen in Pfade, die für sie nicht
gedacht sind.

Das ist kein Randproblem: bei Pipeline-Parallelität ist jede Karte ein
eigener Worker mit eigener Kernel-Übersetzung. Stirbt eine Stufe, warten
alle anderen — gemessen mit py-spy: `broadcast()` und `recv()`, fünf
Karten bei 0 % Auslastung.

## Die 21 Fundstellen, klassifiziert

Pfad relativ zu `.venv-sm70-130/lib/python3.12/site-packages/vllm/`.

### A — Software-Rückfallebene für fehlende FP8-Einheiten (sicher erweiterbar)

Native FP8 gibt es erst ab Ada (sm89). Weder Volta noch Turing haben
welche, die Bedingung ist also schlicht zu eng. Der Software-Zweig
existiert bereits und funktioniert.

| Datei | Zeile |
|---|---|
| `models/deepseek_v4/common/ops/cache_utils.py` | 202, 365 (**bereits gepatcht**) |
| `models/deepseek_v4/common/ops/cache_utils.py` | 398 |
| `models/deepseek_v4/common/ops/fused_compress_quant_cache.py` | 107 |
| `models/deepseek_v4/common/ops/fused_indexer_q.py` | 391 |

Richtige Bedingung: `not current_platform.has_device_capability((8, 9))`.

### B — Kernel-Auswahl (einzeln prüfen)

Hier wird ein sm70-spezifischer Kernel gewählt. Ob er auf sm75 läuft oder
ob es dort einen besseren gibt, muss je Stelle beurteilt werden.

| Datei | Zeile |
|---|---|
| `models/deepseek_v4/attention.py` | 132 |
| `models/deepseek_v4/sm70/gemv.py` | 47 |
| `models/deepseek_v4/nvidia/model.py` | 1324 |
| `model_executor/layers/sparse_attn_indexer.py` | 96 |
| `model_executor/layers/quantization/sm70_turbomind.py` | 57 |
| `model_executor/layers/quantization/utils/marlin_utils_fp4.py` | 32 |
| `v1/attention/backends/mla/flashmla_sparse.py` | 674 |
| `v1/attention/backends/mla/sparse_swa.py` | 513 |

### C — Numerik und Konfiguration (NICHT blind ändern)

| Datei | Zeile | Was |
|---|---|---|
| `models/deepseek_v4/nvidia/dspark.py` | 109 | Skalierungsfaktor `2.0**-6` statt `1.0` |
| `models/deepseek_v4/nvidia/dspark.py` | 226 | Verzweigung im Draft-Kopf |
| `v1/cudagraph_dispatcher.py` | 52, 92 | Graph-Erfassung |
| `v1/worker/gpu/cudagraph_utils.py` | 244 | Graph-Erfassung |
| `config/vllm.py` | 1192, 1211, 1361, 1532, 1587 | Konfigurations-Zwänge |

Eine falsche Änderung in C liefert *stille* Fehlberechnungen statt eines
Absturzes. Hier ist Kohärenzprüfung nach jeder Änderung Pflicht.

## Bereits angewendeter Patch

`cache_utils.py`, zwei Stellen (202 und 365), Bedingung auf
`not has_device_capability((8, 9))` geändert.
Sicherung: `backups/2026-09-01-dsv4-softfp8/cache_utils.py.orig`.
**Noch nicht in `fork_patches/`** — gehört dorthin, sobald es sich bewährt.
Bei einem venv-Neubau ist er sonst weg.

## Reproduktion

Startskript: `backups/2026-09-01-dsv4-softfp8/boot_ds.sh`. Es enthält das
erprobte Rezept aus `scripts/serve-deepseek-mini.sh` plus die heute
gefundenen Ergänzungen.

Erforderlich für DeepSeek-V4-Flash auf dieser Maschine:

| Parameter | Grund |
|---|---|
| `--kv-cache-dtype fp8` | Architektur-Zwang, andere Formate werden abgelehnt |
| `VLLM_PP_LAYER_PARTITION=11,11,7,7,7` | 43 Schichten, zwei 48-GB- und drei 32-GB-Karten |
| **keine** `--block-size` | vLLM muss sie für `fp8_ds_mla` selbst wählen |
| `NCCL_P2P_DISABLE=1` | Kommunikation über den USB4-Tunnel |
| `--enforce-eager` | Graph-Erfassung stirbt an `cudaErrorStreamCaptureInvalidated` |

Sackgassen, nicht erneut probieren:
`--disable-hybrid-kv-cache-manager` führt zu „No common block size" — die
Schnittmenge der von MLA- und SWA-Backend unterstützten Blockgrößen ist
leer. Weder 16 noch 64 hilft.

## Was funktioniert hat

* Die 164,0 GiB passen auf fünf Karten (belegt: 173 GB), mit der
  Gewichtung 11:11:7:7:7. Auf vier Karten (156,3 GiB) passen sie nicht.
* Die Skinny-Kernel greifen: `Using 'SM70_SKINNY' NvFp4 MoE backend`,
  `Nvfp4SkinnySm70Experts`.
* Der Server erreicht `Application startup complete`, KV-Cache 68.048
  Token bei 16k Kontext.
* Der **DSpark-Proposer existiert** (`v1/spec_decode/dspark.py`,
  `class DSparkProposer(DFlashProposer)`) und ist eine **Eigenentwicklung
  des Forks** — im Upstream-Klon `vllm-upstream-pr` gibt es nur
  `dflash.py`. Die Notiz vom 26.08., vLLM bräuchte erst einen Proposer,
  ist damit überholt.
* `dflash.py` enthält `warmup_sm70_dflash_hotpath_kernels` — die
  Fork-Autoren haben Spekulation für Volta ausdrücklich vorbereitet.

## Messmethodik

CPU-Last richtig einordnen (drei Fehldiagnosen an einem Abend):

```bash
PYSPY=.venv-sm70-130/bin/py-spy
$PYSPY dump --pid $(pgrep -f "VLLM::Worker_PP" | head -1)
```

Die Phasen unterscheiden sich deutlich:

| Phase | Kennzeichen |
|---|---|
| Vorbereitung | 97 % CPU je Worker, VRAM 0, Stack in `_load_w13` |
| Laden | 25–33 % CPU, Platte 852 MB/s, VRAM bereits voll (Vorabreservierung!) |
| Hänger | 0 % GPU, 0 MB/s Platte, Stack in `broadcast()`/`recv()` |

**Voller VRAM heißt nicht „geladen"** — vLLM reserviert gemäß
`--gpu-memory-utilization` vorab in einem Rutsch.

## Randbedingungen der Maschine

* **30 GB RAM**, davon 8 GB bereits ausgelagert. Auslagern von
  Modellgewichten in den Hauptspeicher ist praktisch ausgeschlossen.
* Boot-NVMe über USB 3.2 Gen 2, gemessen **852 MB/s**. Eine SSD über die
  zweite 2,5-GbE-Buchse wäre mit ~290 MB/s dreimal langsamer — kein Weg.
* Ladezeit für das 165-GB-Modell: rund vier bis fünf Minuten.

## Messlatte

llama.cpp fährt DeepSeek-V4-Flash mit dspark bei **40,4 tok/s** und hat am
2026-09-01 als eines von nur zwei Modellen die Coandă-Fangfrage gelöst,
sogar auf der niedrigsten Denkstufe. vLLM ohne Spekulation lag im August
bei 3,8–4,1 tok/s. Ein vLLM-Betrieb lohnt nur, wenn die Spekulation über
den DSpark-Proposer greift **und** die sm75-Stufen mitrechnen.

## Empfohlene Reihenfolge

1. Gruppe A vollständig patchen, in `fork_patches/` sichern.
2. Boot wiederholen. Kommt eine Antwort heraus? Kohärenz prüfen.
3. Erst dann Gruppe B einzeln durchgehen, jeweils mit Kohärenzprüfung.
4. Gruppe C nur mit Messung gegen eine bekannte Referenz anfassen.
5. Zuletzt `--speculative-config` mit dspark, und gegen die 40,4 tok/s
   messen.

Bricht es in Schritt 2 oder 3 erneut, ist der ehrliche Schluss: DeepSeek
bleibt unter llama.cpp, und der Fork bleibt für die homogene V100-Hälfte
der Maschine reserviert.

---

## Arbeitsprotokoll 2026-09-01 nachts (Fortsetzung, autonome Session)

### Schritt 1 — Gruppe A vollständig + persistiert ✓

Alle vier Gruppe-A-Stellen tragen jetzt die `not has_device_capability((8, 9))`-
Bedingung (cache_utils.py 202/365 waren schon gepatcht; neu: cache_utils.py 398
`use_cutedsl`, fused_compress_quant_cache.py 107, fused_indexer_q.py 391).
Persistiert als `fork_patches/dsv4_cache_utils.py`,
`dsv4_fused_compress_quant_cache.py`, `dsv4_fused_indexer_q.py` — mit
Apache-§4(b)-Headern, in der Bootstrap-Deploy-Liste und der README-Tabelle.
Originale: `backups/2026-09-01-dsv4-softfp8/*.orig`.

### Erkenntnis: die B-Stellen sind überwiegend Device-0-Kopplungs-Bugs

`current_platform.is_device_capability((7, 0))` fragt Device 0 der
Sichtbarkeitsliste — bei unserem `CUDA_VISIBLE_DEVICES=0,2,1,4,3` eine
RTX 8000. Damit liefern die "exakt sm70"-Weichen auf ALLEN Workern False,
auch auf den V100-Stufen: die verlieren ihren SM70-Pfad, nicht die RTX ihren.
Das ist exakt die Wurzel, die das Merge-Projekt (Session 4) für Qwen3.8
gefunden und in `fork_patches/vllm_config.py` (ANY-device-Baseline) und
`fork_patches/cuda.py` (Worker-lokale Backend-Wahl) gefixt hat.
Das validierte Muster für Worker-lokale Weichen steht in
`fork_patches/qwen3_5.py:346`:
`torch.cuda.get_device_capability(torch.cuda.current_device())`.

Klassifikation der B-Stellen nach Lektüre:

| Stelle | Wirkung auf heterogenem Boot | Risiko |
|---|---|---|
| `models/deepseek_v4/attention.py:132` (`_is_exact_sm70_cuda`, 3 Nutzungen) | Impl-Wahl: pre-Hopper nimmt ohnehin den Triton-Pfad (has_device_capability(90)-Gate davor); aber `_use_sm70_path` (Z. 223) steuert Layer-Verhalten und ist Device-0-gekoppelt → V100-Stufen laufen ohne SM70-Pfad | HOCH |
| `sparse_attn_indexer.py:96` (Z. 207 `sm70_fp16_indexer`, Z. 564) | dito, Worker-lokal nötig | HOCH |
| `flashmla_sparse.py:674` (`fixed_row_stride`) | LAYOUT-Parameter Device-0-gekoppelt → potenziell stille Fehlberechnung der V100-Stufen | HÖCHSTES |
| `sparse_swa.py:513` | sm70-Ausnahme beim FlashMLA-SchedMeta-Stub; Device-0-gekoppelt | MITTEL |
| `sm70_turbomind.py:57` | Turbomind-Auswahl; bei uns per Env aus (`VLLM_SM70_NVFP4_TURBOMIND=0`) | NIEDRIG (derzeit) |
| `sm70/gemv.py:47` | Opt-in-Fastpath (`VLLM_SM70_DSV4_FP16_GEMV`); ohne ihn nur langsamer, nicht falsch | NIEDRIG |
| `nvidia/model.py:1324` | Nur ein Dtype-Guard (fp16-Zwang); Boot nutzt --dtype half → erfüllt | KEINES |
| `marlin_utils_fp4.py:32` | `has_device_capability(75)` = True über Device 0; V100-Worker könnten den sm75-Marlin-Pfad wählen. Boot-Log beobachten | MITTEL |

### Gruppe C (config/vllm.py 1192/1211/1361/1532/1587): für DIESEN Boot inaktiv

1192/1211/1587 gelten nur bei `quantization == "fp8"` (Checkpoint ist NVFP4),
1361/1532 nur im Compile-/Graph-Pfad (Boot ist --enforce-eager). Bleiben
unangetastet, bis eager-Kohärenz steht — wie vom Handover verlangt.

### Schritt 2 — Boot + Kohärenz mit Gruppe A: BESTANDEN ✓ (23:05)

Heterogener 5-Karten-Boot (`boot_ds.sh`, K=0, eager) erreicht startup,
und die Kohärenzprobe (`scripts/deepseek_coherence.py`, 8 Prompts,
greedy) ist **vollständig deckungsgleich mit der nvidia-base-Referenz**:
Paris / 1591 / 10 / Mercury / `s[::-1]` / exakte longctx-Liste; Prosa
kohärent. Ergebnis: `results-coherence-het-groupA.json`.

**Wichtige Korrektur der Ausgangsthese:** Die B-Stellen sind für die
KORREKTHEIT dieses Setups nicht nötig. Die Device-0-Kopplung setzt alle
Worker einheitlich auf den generischen (Nicht-SM70-)Pfad — einheitlich
generisch rechnet richtig, verschenkt aber die SM70-Optimierungen der
V100-Stufen. B ist damit ein TEMPO-Thema, kein Korrektheitsthema.
Vorsicht beim späteren Worker-lokal-Machen: divergierende Layouts
zwischen Stufen (fixed_row_stride!) können die Kohärenz brechen —
einzeln, mit Kohärenzprobe, wie gehabt.

Baseline-Durchsatz K=0 eager: 501 Token Prosa in 121,3 s = **4,13 tok/s**
(deckt sich mit den 3,8–4,1 vom August).

### Schritt 3 (vorgezogen: Spekulation statt B) — DSpark K=5

Begründung der Umreihung: Die Messlatte (40,4 tok/s) ist nur über
Spekulation erreichbar; B bringt einstellige Prozente auf den
V100-Stufen. Also erst DSpark, B danach gezielt nach Profil.

Rezept: `--speculative-config '{"method": "dspark",
"num_speculative_tokens": 5}'` — Minimum ist `dspark_block_size` 5 aus
der Checkpoint-Config (`dspark_markov_rank` 256, target_layers 40–42);
Draft-Modell = derselbe Checkpoint (`speculative.py:650`). Skript:
`backups/2026-09-01-dsv4-softfp8/boot_ds_k5.sh`. Auffällig:
`speculative_config.enforce_eager` defaultet auf False — der Drafter
versucht CUDA-Graphs trotz eager-Hauptmodell.

### Bug gefunden (betrifft auch homogene V100-Boots!): hf_config_override nicht idempotent

Der erste K=5-Boot starb mit „Pipeline parallelism is not supported for
this model" — obwohl `DeepSeekV4MTP` das `SupportsPP` des Forks trägt.
Ursache: `SpeculativeConfig.hf_config_override` (config/speculative.py
~360) wird auf dem Draft-Config-Pfad mehrfach angewandt. Erste Anwendung:
model_type deepseek_v4 → deepseek_mtp, Architektur `DeepSeekV4MTPModel`
(so steht es auch im Boot-Log, „Resolved architecture"). Zweite
Anwendung: model_type ist bereits deepseek_mtp → der V3-Zweig
überschreibt die Architektur mit `DeepSeekMTPModel` — und das V3-MTP hat
kein SupportsPP. Repro: Override 2× auf dieselbe Config anwenden.

Fix (fork_patches/speculative.py + venv, synchron): der
deepseek_mtp-Zweig greift nicht mehr, wenn `initial_architecture`
bereits `DeepSeekV4MTPModel` ist. Nachweis: 1×/2×/3× Anwendung liefern
jetzt stabil DeepSeekV4MTPModel; der v3-Pfad ist unverändert
(DeepSeekMTPModel, idempotent). Kein Turing-Thema — jeder
DSpark-PP-Boot, auch rein auf V100s, lief in diesen Fehler.

**Korrektur zur Crash-Ursache:** Der Idempotenz-Bug ist real (Repro:
2× Anwendung → DeepSeekMTPModel) und der Fix bleibt drin — aber der
K=5-Crash kam von etwas anderem: dspark schreibt die Draft-Architektur
auf `DSparkDraftModel` → `DSparkDeepseekV4ForCausalLM`
(nvidia/dspark.py:256) um, und DIESER Klasse fehlte `SupportsPP` —
dieselbe Lücke, die die Fork-Autoren bei `DeepSeekV4MTP` schon einmal
geschlossen haben („Fork addition: SupportsPP"). Fix nach exakt deren
Muster: Basisklasse `SupportsPP`, `make_empty_intermediate_tensors`-
Factory im `__init__`, `intermediate_tensors`-Keyword im `forward`
(akzeptiert und ignoriert — der Drafter läuft komplett auf der letzten
Stufe). KEINE Numerik-Änderung; die Gruppe-C-Warnung zu dspark.py
109/226 bleibt unberührt. Persistiert als `fork_patches/dsv4_dspark.py`
+ Deploy + README.

**Stolperfalle Model-Info-Cache:** `~/.cache/vllm/modelinfos/*.json`
cached die Interface-Inspektion; der Staleness-Hash bemerkt einen Patch
an `nvidia/dspark.py` NICHT (Registry-Mapping zeigt aufs Paket
`vllm.models.deepseek_v4`). Nach jedem Interface-Patch die betreffende
JSON löschen, sonst prüft der Boot gegen die alte Antwort.

### Drafter-Gewicht und Partition (K=5)

OOM-Analyse über die Safetensors-Header (Bytes je Tensor): alle 43
Hauptlayer uniform ~3,52 GiB (Summe 151,4), `embed` 0,99, `head` 0,99 —
und der **DSpark-Drafter (`mtp.*`) wiegt 10,68 GiB**. Die letzte Stufe
trägt Drafter + head, neben denen nur 4 Hauptlayer auf eine 32-GiB-V100
passen (14,1 + 11,7 ≈ 25,8). Neue Partition für Spekulation:
`VLLM_PP_LAYER_PARTITION=12,12,8,7,4` bei `--gpu-memory-utilization
0.94` → Stufen-Gewichte 43,2 / 42,3 / 28,2 / 24,6 / 25,8 GiB
(RTX-Budget 45,4, V100-Budget 29,8). Die K=0-Partition 11,11,7,7,7
bleibt für spekulationsfreie Boots die richtige.

### K=5-Bring-up: die Zwiebel (Boots 4–17, Nacht 02:00–02:30)

Nach dem SupportsPP-Fix schälte sich Schicht um Schicht:

1. **OOM letzte Stufe** → Partition muss den Drafter einpreisen (s.o.).
2. **SWA-Ragged-Kopie zu schmal** (`models/deepseek_v4/amd/rocm.py`,
   `_copy_ragged_to_graph_buffers`-Aufrufer): der Drafting-Pfad baut
   SWA-Zeilen BREITER als `window_size` (Block-Overlap; 5 Draft-Zeilen
   × 256 statt × 128) — der Slice kappte die Kopie („size of tensor a
   (640) must match … (1280)"). Fix: Slice-Breite aus der echten
   dense-Zeilenbreite (`dense_swa.shape[1]`), nicht aus der Annahme.
3. **dspark `_insert_context_kv`**: der sm70-Software-Zweig hing an
   `is_device_capability((7,0))` = Device 0 = RTX → die V100-Stufe
   rannte in den fused Op („requires sm_80+; got sm_70"). Fix nach dem
   validierten Muster: Worker-lokale Capability, Grenze < (8,0). KEINE
   Numerik-Änderung (die Gruppe-C-Warnung betrifft andere Zeilen).
4. **Scheinbarer PP-Deadlock** (PP0-2 im Spec-Broadcast, PP3-4 in
   irecv): drei Fehlfährten — --no-async-scheduling (bringt eigenen
   Bruch: Scheduler kennt Drafts nicht, assert num_scheduled >=
   draft_len+1), Batch-Queue-Cap (VLLM_SM70_ASYNC_SCHEDULING_QUEUE_DEPTH
   wirkt jetzt auch bei PP>1 — als Option behalten), Spec-Broadcasts auf
   gloo/cpu_group (Transport war unschuldig; Umstellung bleibt drin,
   schadet nicht, Payloads sind winzig). Die WAHRHEIT fand erst der
   Seam-Trace (`VLLM_PP_SEAM_TRACE=1`, neue Diagnose in
   parallel_state.py): PP0→PP1→PP2 lief, **PP2 sendete nie an PP3** —
   sein execute_model warf „Triton Error [CUDA]: out of memory", der
   worker_busy_loop loggt das nur und nimmt den nächsten RPC an, die
   Pipeline verklemmt OHNE dass die Engine stirbt. Merke: bei
   „Deadlock"-Symptomen IMMER erst `grep ERROR` über alle Worker.
5. **JIT-Launch-OOM auf der 8-Layer-V100-Stufe** (28,2 GiB Gewichte):
   weder util 0.96/0.97 noch batched 512 + expandable_segments retten
   sie — eine V100 verträgt im DSpark-Betrieb KEINE 8 Hauptlayer.
   Konsequenz: Partition 12,12,7,7,5 (PP0 wagt 12+embed, PP4 5+Drafter)
   ist die einzige Verteilung ohne 8er-V100.

Env-Stand des K=5-Skripts (boot_ds_k5.sh): QUEUE_DEPTH=2, SEAM_TRACE=1,
PYTORCH_ALLOC_CONF=expandable_segments:True, batched 512, util 0.96,
max-model-len 8192.

### DURCHBRUCH 03:40 — DSpark K=5 läuft heterogen, Drafter auf der RTX 8000

Finale Boot-Konfiguration (boot_ds_k5.sh):
`CUDA_VISIBLE_DEVICES=0,1,4,3,2` (RTX, V100, V100, V100, RTX),
`VLLM_PP_LAYER_PARTITION=11,8,8,8,8`, util 0.97,
`--num-gpu-blocks-override 512`, batched 256, max-model-len 4096,
max-num-seqs 1, QUEUE_DEPTH=2, expandable_segments.

Der Speicher-Zielkonflikt (KV-Profiling will hohen util-Pool, das
Serving-JIT braucht physischen Nicht-Pool-Raum) löst sich durch
**util hoch + `--num-gpu-blocks-override` klein**: der Check rechnet
mit dem util-Budget, die reale KV-Allokation bleibt winzig (MLA-KV
≈ 0,3 MB/Block). Alle fünf Stufen positiv (1,5/1,5/1,5/2,5/3,7 GiB).
Der Drafter (10,7 GiB mtp.*) erstickt auf jeder V100 — auf der
zweiten RTX 8000 hat er Luft. Kohärenzprobe K=5: **alle 8 Prompts
deckungsgleich mit nvidia-base** (results-coherence-het-k5.json).

### Offen: DSpark-Akzeptanz kollabiert (~5–6 %) — Tempo bleibt ~3,7–4,5 tok/s

Draft-Dumps (`VLLM_SM70_MTP_DUMP_STEP_DIR` + scripts/analyze_drafts.py):
Position 0 trifft oft (25 %), ab Position 1 deterministischer Müll
(' parallelogram', ' cryptocur'). Verdächtige geprüft:
* mhc_post/hc_head-TileLang: nehmen bei fp16 den Torch-Generic-Pfad — raus.
* main_proj_input_scale 2^-6: Device-0-Bug gefixt (worker-lokal, < (8,9)) —
  Akzeptanz unverändert, Fix bleibt (sachlich richtig).
* sm75-Layer-Mathe generell: durch die kohärenten 8 Hauptlayer auf
  derselben RTX entlastet.

Nächste Verdächtige (Reihenfolge für die nächste Session):
1. **Markov-Head-Laden verifizieren** (mtp.2.markov_head.* →
   model.markov_head.* — Remap sieht korrekt aus, aber prüfen ob die
   Params wirklich geladen werden; Pos-0-gut/Rest-Müll passt exakt zu
   fehlendem Markov-Bias). Auch: DSparkProposer wurde hier zum ERSTEN
   Mal überhaupt end-to-end gefahren — der Bug kann hardware-unabhängig
   im Fork-DSpark-Pfad liegen.
2. `sm70_qnorm_rope_kv_fp8_insert` auf sm75 gegen V100 diffen (A/B-Skript
   liegt im Scratchpad-Ansatz vor; scheiterte an belegtem VRAM).
3. dflash.py:159 Warmup-Gate ist noch Device-0-gekoppelt (nur Warmup,
   keine Numerik — aber gleiche Bug-Klasse, fixen).
4. Aux-Hidden-States (dspark_target_layer_ids 40–42, alle auf PP4) und
   der nicht-kausale Block-Attention-Pfad (SWA-Drafting-Metadata).

### Persistiert (fork_patches + Deploy + README, alles kompiliert)

dsv4_cache_utils / dsv4_fused_compress_quant_cache / dsv4_fused_indexer_q
(Gruppe A), dsv4_dspark (SupportsPP + worker-lokaler Dispatch + Scale),
dsv4_amd_rocm (SWA-Breite), speculative (hf_config_override idempotent),
gpu_model_runner (Spec-Transport gloo), multiproc_executor (Queue-Cap),
gpu_worker (per-Rank-KV-Log + Trace), parallel_state (Seam-Trace),
kv_cache_utils (Fehlermeldung mit GiB-Zahl).
**Uncommitted** — Commit auf Ansage des Users.

### Sonstiges
* Model-Info-Cache-Falle (s.o.): nach Interface-Patches an Modellklassen
  die JSON unter ~/.cache/vllm/modelinfos/ löschen.
* boot_ds_k5_dump.sh = Variante mit Draft-Dumps ins Scratchpad.
* K=0-Referenz unverändert: boot_ds.sh (11,11,7,7,7, util 0.92) →
  4,13 tok/s, kohärent.

### Akzeptanz-Forensik (04:00–05:20) — NaN-Wurzel gefunden und gefixt, Rest offen

Werkzeugkette der Diagnose (alles env-gated, bleibt im Fork):
`VLLM_DSPARK_DIAG=1` aktiviert Prints in
- dspark.py: Kontext-Insert (Layer, Token, Slots), Block-forward
  (Positionen, input_ids), combine (aux/proj-Statistik inkl.
  badrows/Layer-Drittel), Proposer (Anker + base-argmax)
- amd/rocm.py: Drafting-SWA (rows, width, lens, row0-Indices)
Skript-Variante: `boot_ds_k5_diag.sh`; Draft-Dumps weiterhin über
`boot_ds_k5_dump.sh` + scripts/analyze_drafts.py.

Entlastet (bewiesen):
* Drafting-SWA-Metadata: Slots/lens exakt konsistent mit dem Insert
  (Kontext 192–210 + Block 211–215, lens=24 ✓ nicht-kausal korrekt)
* `sm70_qnorm_rope_kv_fp8_insert` auf sm75: bit-identisch zur V100
  (A/B im Scratchpad, ab_kernels.py — Achtung: je Karte mit eigenem
  CUDA_VISIBLE_DEVICES laufen lassen, sonst kompiliert Triton für
  Device 0 und wirft „no kernel image")
* Positionen/Noise-Token des Blocks: [ctx..ctx+4], [anchor, 4×128799] ✓
* mhc/hc_head-TileLang: bei fp16 laufen die Torch-Generic-Pfade
* Markov-Bias: funktioniert (er allein hob Position 0 auf 25 %)

**Gefundene Wurzel:** Die Aux-Hidden-States der **BOS-Zeile** (Attention-
Sink) sprengen unter `--dtype half` den FP16-Bereich → inf/NaN → ein
einziger vergifteter Kontext-KV-Slot macht via Softmax ALLE
base_logits des Drafters zu Müll (belegt: base-argmax vor Fix
' cryptocur/' parallelogram' selbst auf Position 0). Die bf16-DSpark-
Referenz kennt das Problem nicht. Fix in dsv4_dspark.py
(combine_hidden_states): nan_to_num-Sättigung auf ±65504 — wie eine
saturierende bf16→fp16-Konvertierung; die RMSNorm normalisiert die
Sink-Zeile danach ohnehin (mainx absmax 0,39 statt NaN). Wirkung
belegt: base[0] trifft jetzt (' fox' nach 'The', ' the' nach ' over').
Vermutlich betrifft dieser Bug auch einen HOMOGENEN V100-dspark-Boot —
der Pfad lief vor dieser Nacht nie bis zur Akzeptanzmessung.

**Weiter offen:** Akzeptanz bleibt ~5 % (70/1420 im Essay-Lauf), Tempo
~3,6–4,5 tok/s ≈ K=0-Baseline. base[1..4] (die Noise-Positionen)
liefern kontext-nahe, aber positionsblinde Token (' frog',
' parallelogram'). Nächste Schritte:
1. Markov-Gewichts-Laden verifizieren (Werte gegen Checkpoint diffen,
   nicht nur Remap-Logik lesen).
2. Prüfen, wie llama.cpp dspark implementiert (Block-Attention-Maske!
  sieht Noise-Position i dort die Positionen <i des Blocks? Unsere
  lens=24 GLEICH für alle 5 Zeilen = voll nicht-kausal — wenn die
  Referenz eine Stufen-Maske nutzt, wäre unsere Attention-Maske falsch
  und genau das Positionsblind-Muster erklärt).
3. Akzeptanz-Referenz je Position aus llama.cpp ziehen (dort läuft
   dspark mit 40,4 tok/s — Akzeptanzprofil vergleichen).

Endzustand 05:25: alle GPUs frei, kein Server. K=5-Boot reproduzierbar
über backups/2026-09-01-dsv4-softfp8/boot_ds_k5.sh (Diag-Varianten
daneben). Alle Fixes in fork_patches + Bootstrap-Deploy + README,
Repo uncommitted — Commit auf Ansage.

### 2026-09-02 17:00 — Akzeptanz-Wurzel Nr. 2: Drafter lief auf ZUFALLS-Embeddings (PP-Bug)

`_maybe_share_embeddings` (llm_base_proposer.py) teilt die Target-
Embeddings nur bei `pp_world_size == 1`; bei PP heißt es „will be loaded
separately" — aber `_remap_dspark_name` verwarf jeden Nicht-`mtp.*`-Key,
also auch `embed.weight`. Der Drafter (has_own_embed_tokens=False, sitzt
auf der LETZTEN Stufe, die Target-Embeddings liegen auf der ERSTEN) rechnete
seinen Block-Forward auf zufällig initialisierten Embeddings; lm_head war
geteilt (deshalb traf Position 0). Beleg: Boot-Log „vocab embedding will be
loaded separately" auf PP4. llama.cpp-Vergleich (src/models/dflash.cpp,
common/speculative.cpp): Sampling-Pfad (Anker-Start, Markov-Kette, voll
nicht-kausale Attention, RoPE ab Kontextende) ist äquivalent — der Fehler
lag allein im Laden.

Fix (dsv4_dspark.py): `embed.weight` → `model.embed_tokens.weight` im
Remap + harter RuntimeError, falls es nicht geladen wurde. Bei PP=1
ersetzt das Sharing die Tabelle danach ohnehin.

**Wirkung:** Akzeptanz 5 % → 36 % (Essay, 239/660), Profil je Position
78/51/33/21/13 %; Tempo 3,6 → **7,97 tok/s** (Essay 371 Token / 46,6 s),
seq-Prompt 10 tok/s. Kohärenz weiterhin 8/8
(results-coherence-het-k5-embedfix.json). Hardware-unabhängiger Bug —
jeder PP-DSpark-Boot war betroffen.

Nebenbefund aus llama.cpp: der Confidence-Head (`dspark_conf_proj`, im
vLLM-Remap verworfen) dient dort NUR der Draft-Kürzung (p_min) — bei
uns werden alle 5 Draft-Token verifiziert, auch wenn Pos 3–4 nur 21/13 %
treffen; der Verify skaliert auf Volta linear mit q (Issue #441, Teil 2).
Kandidat für den nächsten Tempo-Hebel.

### 2026-09-02 18:00 — Gruppe B worker-lokal: korrekt, aber kein Tempogewinn

Die sechs „exakt sm70"-Weichen des DeepSeek-Pfads (attention.py-Helper,
sparse_attn_indexer-Helper, sm70_turbomind-Helper, gemv, flashmla_sparse
fixed_row_stride, sparse_swa) fragen jetzt das Worker-Device
(Semantik bleibt „exakt Volta"; Skript: Scratchpad
apply_group_b_worker_local.py, Ergebnis in fork_patches). Stolperstein:
der grouped SM70-O-Projektionspfad setzt TurboMind-präpariertes wo_a
voraus — mit der QPN8-blk-is_bmm-Route ist wo_a fp16-dequantisiert
(auch im homogenen V100-Lauf vom 26.08., dessen Log 3–4 Tracebacks hat)
und der Referenz-Einsum ist die passende Implementierung. Der Zweig
hängt jetzt strukturell an `wo_a._qpn8_dequant16`.

Ergebnis: Kohärenz 8/8, Akzeptanz 32 %, **7,56 tok/s — unverändert**.
Stufenlatenz (Seam-Zeitstempel, ein Decode-Schritt mit 6 Token):
V100-Stufen ~71 ms je 8 Layer, PP4 (RTX, 8 Layer + Drafter) ~100 ms,
Schritt gesamt ~330 ms ⇒ ~9 ms/Layer. llama.cpp: 43 Layer in ~80 ms
⇒ ~2 ms/Layer. Die Kernel-Wahl ist nicht der Engpass — der Eager-Modus
(hunderte Launches pro Layer, keine CUDA-Graphs) ist der Hauptverdächtige.
Nächster Hebel: `--enforce-eager` fallen lassen (Capture-Crash aus dem
Handover war evtl. ein Symptom der inzwischen gefixten Bugs).

### 2026-09-02 18:45 — Weg zu CUDA-Graphs: der Skinny-MoE ist die Bruchstelle

Ohne `--enforce-eager` stirbt das Capture in `nvfp4_skinny_moe.py`
(`topk_ids.cpu()` — host-getriebene Experten-Schleife, „operation not
permitted when stream is capturing"). Kontext: Der Fork aktiviert für
DeepSeek V4 automatisch `VLLM_USE_BREAKABLE_CUDAGRAPH` (Segment-Capture
mit Eager-Breaks an den Attention-Ops, `compilation/breakable_cudagraph.py`);
MARLIN-MoE (capture-sicher, so lief Flash-Next k=0 mit Graphs bei 28–29
tok/s) scheidet für DeepSeek aus, weil nur TRTLLM/EMULATION/SM70_SKINNY
den `swiglu_limit`-Clamp können. Umsetzung: `@eager_break_during_capture`
auf `Nvfp4SkinnySm70Experts.apply` — exakt der Mechanismus der
Attention-Ops; Adressvertrag erfüllt, weil output/hidden_states/topk_*
in gecapturten Segmenten alloziert und in place beschrieben werden.

Microbench (Scratchpad moe_bench.py, echte Layer-5-Experten, eager):
M=1: 0,8 ms/Layer; M=6 (27 aktive Experten): **3,5 ms/Layer** auf RTX 8000
wie V100. Bei ~9 ms/Layer je Schritt ist der MoE ~40 %; der Rest ist
Graph-Kandidat. Realistische Erwartung mit Graphs + Eager-MoE: ~5 ms/Layer
⇒ ~13 tok/s. Für 40 tok/s müsste der MoE selbst unter 1 ms/Layer —
d. h. ein grouped/device-seitiger Skinny-MoE-Kernel statt der
Per-Experten-Schleife (27 Experten × 2 GEMMs + Index-Ops je Layer).
Das ist die eigentliche Tempo-Baustelle nach den Graphs.

Werkzeug-Lektion: Boots IMMER mit `setsid nohup … & disown` in einem
eigenen kurzen Aufruf starten und in einem SEPARATEN Aufruf warten — der
10-Minuten-Timeout des Tools killt sonst die Prozessgruppe samt Boot
(Boot 41 starb so um 18:18, exakt 10 min nach Start, ohne Traceback).

### 2026-09-02 20:00 — Graph-Capture: drei Brecher beseitigt, vierter ist Speicher

Boots 41–48 ohne `--enforce-eager` (Drafter eager, `VLLM_DISABLE_SHARED_
EXPERTS_STREAM=1`), Capture-Brecher in Reihenfolge:
1. Skinny-MoE `topk_ids.cpu()` → `@eager_break_during_capture(ignore_full_
   mode=True)` auf `Nvfp4SkinnySm70Experts.apply` (Decorator um die Option
   erweitert: der Fork-Bypass für FULL-Runtime-Modus ist für capture-sichere
   Attention-Ops richtig, der host-getriebene MoE muss immer brechen).
2. Shared Experts auf Aux-Stream über die Segmentgrenze → Env-Schalter.
3. Indexer auf den RTX-Stufen im torch-reference-Pfad (Host-Sync): Gate
   `< (8,0)` UND Produzenten-Fix in attention.py — `has_deep_gemm()` ist auf
   beiden Karten False, der fp16-Indexer war damit bisher UNERREICHBAR
   (Produzent lieferte immer fp8); pre-Ampere nutzt jetzt den fused
   Software-Produzenten (fp16 + gefaltete Gewichte). RTX-Ränge capturen.
4. V100-8-Layer-Stufen: Triton-Launch-OOM im Capture (auch mit
   Capture-Größen [1,6,8], util 0.95). Physik: 28,2 GiB Gewichte auf 31,7,
   Graph-Pools brauchen mehr als die ~1,5 GiB Rest. Layer-Arithmetik ohne
   8er-V100: RTX-Paar max 23, V100-Trio (Drafter auf einer V100) max 18
   ⇒ 41 < 43. Graphs auf DIESER 5-Stufen-Topologie brauchen noch einen
   echten Speicherhebel (Workspaces, batched-tokens) — sonst bleibt
   DeepSeek eager und der grouped MoE-Kernel ist der Weg.

### 2026-09-02 20:10 — CUDA-Graphs laufen heterogen (Boot 49)

Letzter Hebel für die V100-8-Layer-Stufen: `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64`
(Default 512 MB je Rang; kappt nur die Prefill-Chunk-Größe des Indexers),
`--max-num-batched-tokens 64`, Capture-Größen `[6]` (1 Sequenz × K=5), util
0.95. „Graph capturing finished in 1 secs, took 0.05 GiB". Rezept:
`scripts/serve-deepseek-het-graphs.sh` (Drafter eager, Shared Experts inline).

Ergebnis: Kohärenz **8/8** (results-coherence-het-k5-graphs.json), Essay
**8,76 tok/s** (eager 7,56), Code/Sequenz/Liste 10–12 tok/s, Akzeptanz
32 %. Gewinn kleiner als erhofft: der MoE bleibt eager (~3,5 ms × 8 Layer je
V100-Stufe) und die 5-Stufen-PP-Latenz (USB4, NCCL_P2P_DISABLE, gloo-Metadaten
je Naht) bleibt. Nächste Hebel in dieser Reihenfolge:
1. Stufenlatenz mit Graphs messen (SEAM+CPU-Trace-Boot) — wie viel ist PP-Naht,
   wie viel Rechnen.
2. Grouped/device-seitiger Skinny-MoE (routing ohne Host-Sync, ein Launch je
   Layer statt 2×Experten) — der einzige Weg unter ~2 ms/Layer.
3. Drafter mit Graphs (speculative enforce_eager=false) — ~20 ms/Schritt.
4. Akzeptanz 32 → 60 % (llama.cpp-Niveau): BOS-Sättigung durch fp32-Umweg
   ersetzen, fp16-Numerik des Drafters auf sm75 prüfen.

### 2026-09-02 20:30 — Stufenlatenz MIT Graphs (Seam-Trace, ein Decode-Schritt, 6 Token)

PP0 RTX 11 Layer+embed: **75 ms** · PP1/2/3 V100 8 Layer: **49/47/45 ms**
(eager waren es 71) · PP4 RTX 8 Layer+Drafter(eager 20 ms)+Sampling: **77 ms**
· Summe 293 ms ≈ Engine-Schritt 286 ms. Pro Layer: V100 ~5,9 ms, RTX ~6,9 ms
(GDDR6 672 GB/s gegen HBM2 900 GB/s — Decode ist bandbreitengebunden, die
V100 ist die schnellere Karte). Davon MoE eager 3,5 ms × 43 Layer ≈ **150 ms
je Schritt** — die Hälfte. Der grouped Skinny-MoE ist damit belegt der
nächste große Hebel (Ziel ≤ 1 ms/Layer ⇒ Schritt ~180 ms ⇒ ~15 tok/s; mit
Akzeptanz 60 % und Drafter-Graphs Richtung 25–30 tok/s).
Boot-Rezept mit Traces: backups/2026-09-01-dsv4-softfp8/boot_ds_k5_graphs_trace.sh.

### 2026-09-02 21:20 — Punkt 3 (Drafter mit Graphs): neutral

`speculative_config.enforce_eager` weggelassen (Boot 51,
backups/…/boot_ds_k5_graphs_draftgraph.sh): Capture läuft durch, Essay
8,73 tok/s (vorher 8,76), Akzeptanz 32 % — kein messbarer Effekt, der
Drafter-Anteil (~20 ms) ist vom eager MoE-Break dominiert. Bleibt als
Default (weniger Sonderfälle). Commit dda7ad2 enthält den Stand davor.

### 2026-09-02 22:10 — Punkt 2 (Akzeptanz): fp32-Aux-Abgriff statt Sättigung

`_mhc_post_torch_generic` rechnet fp32 und castet am Ende nach fp16 — dort
lief die BOS-Zeile über. Neu: `mhc_post_fp32` (gleiche Mathematik, kein
Cast) für den Aux-Abgriff in model.py; der Drafter skaliert in fp32 mit
2^-6 und castet erst dann auf den Aktivierungs-dtype (main_norm.weight);
die nan_to_num-Sättigung ist raus. Boot 52 (Graphs, Drafter mit Graphs):
Kohärenz 8/8, Essay **9,36 tok/s**, Akzeptanz **36 %** (Profil
75/50/34/22/14 % je Position — praktisch wie vorher).

Befund zur restlichen Lücke (36 gegen 61–65 % bei llama.cpp): llama.cpp
fährt den DSpark-Drafter als **Q8_0**; unser Checkpoint trägt ihn mit
NVFP4-Experten (U8-gepackt) und FP8-Attention (mtp-quant-transplant,
bewusst wegen VRAM/Bandbreite). Ein 4-Bit-Drafter rät schlechter als ein
8-Bit-Drafter — ein Teil der Lücke ist damit Checkpoint, nicht Code.
Nächster Akzeptanz-Schritt wäre ein 8-Bit-Drafter-Transplantat (Projekt
mtp-quant-transplant), kein Kernel-Thema. Damit ist Punkt 2 im Rahmen
des Forks ausgereizt; weiter mit Punkt 1 (grouped MoE-Kernel).

### 2026-09-02 22:45 — Punkt 1: grouped NVFP4-MoE-Kernel (moe_simt) — 13–20 tok/s

Neuer Kernel `skinny_nvfp4_moe_simt` in kernels/skinny_kernels.cu (Binding
`moe_simt`): ein Launch je Gewichtsmatrix und Layer, Grid (N/8 × Experten),
Routing device-seitig (argsort + scatter_add + cumsum — KEIN bincount, das
synchronisiert), Blöcke inaktiver Experten beenden sich ohne Gewichtszugriff,
x-Zeilen per Slot-Permutation gegathert (token-major für w13, slot-major für
w2), Zeilen-Dispatch blockuniform 1/2/4/8, Mehrpass falls Hash-Routing einem
Experten mehr als 8 Zeilen gibt (das war der „misaligned address"-Crash
von Boot 54: smem-Zeilenarray mit 8 Einträgen bei cnt=12). Python-Seite
(nvfp4_skinny_moe.py): Batches ≤ 8 Token → grouped, capture-sicher ohne
Eager-Break; Prefill → alte Schleife mit Break.

Standalone (scripts/nvfp4_skinny_moe_grouped_test.py, echte Layer-5-
Gewichte, RTX 8000 und V100): max|diff| 1,2–2,4e-4 (fp16-Rundung), T=6:
1,0 ms statt 3,7–4,0 ms je Layer (3,5–4×), T=1: 0,4 statt 0,8 ms.
Achtung Loader: ModelOpt-`weight_scale_2` ist der Dequant-Multiplikator
(compressed-tensors speichert den Kehrwert).

Server (Boot 56, Graphs, Drafter-Graphs, fp32-Aux): Kohärenz **8/8**;
Essay **13,1 tok/s** (Akzeptanz 32 %), Code-Prompt **20,4 tok/s**
(Akzeptanz 64 %), Sequenz/Liste 15–16 tok/s. Tag gesamt: 3,6 → 13–20 tok/s.
Offen: Stufenlatenz-Trace mit grouped MoE (welcher Anteil bleibt: Attention/
Indexer eager-Breaks, PP-Nähte), Akzeptanz bei Prosa (Q8-Drafter-Frage).

### 2026-09-03 01:30 — Punkt 2 abgeschlossen: KEINE Akzeptanz-Lücke zu llama.cpp

Direkter Vergleich, gleiche Prompts, temperature 0, gleicher DSpark-Drafter
(llama.cpp: dspark-…-Q8_0.gguf = Q8 nur für Attention, Experten MXFP4 —
also ebenfalls 4-Bit-Experten, nicht „8-Bit-Drafter"):

| Prompt | llama.cpp Akzeptanz | vLLM (unser) | llama.cpp tok/s | vLLM tok/s |
|---|---:|---:|---:|---:|
| Essay (Prosa) | 27,9 % (208/745) | 32 % | 21,2 | 13,1 |
| Code (CSV-Parser) | 68,8 % (231/336) | 64 % | 37,8 | 20,4 |

Die 61–65 % der Performance-History waren anderer Inhalt. Der Drafter-Pfad
ist damit auf Augenhöhe; der 8-Bit-Drafter entfällt (und passte
speichermäßig ohnehin nicht: FP8-Experten ≈ 19 GiB auf PP4).
**Die gesamte Restlücke ist Schrittlatenz:** llama.cpp ~110 ms je
6-Token-Verify-Schritt (≈2,5 ms/Layer über 5 Karten), wir ~200 ms
(≈4,2 ms/Layer; GPU-Zeit laut PP4-Sample-Trace 180 ms, Drafter 5 ms).
Nächster Schritt: Per-Stufen-Rechenzeit unter Graphs (SEAM-Trace mit
CUDA_LAUNCH_BLOCKING, sonst laufen die Zeitstempel der GPU voraus).

### 2026-09-03 02:00 — Per-Stufen-Rechenzeit unter Graphs (CUDA_LAUNCH_BLOCKING, Boot 58)

Ein Decode-Schritt (6 Token), Seam-Zeitstempel = GPU-Zeit dank LAUNCH_BLOCKING
(leicht überhöht, ~234 statt ~200 ms): PP0 RTX 11 Layer+embed **54 ms**,
PP1/2/3 V100 8 Layer **38/38/37 ms**, PP4 RTX 8 Layer+Drafter+Sampling
**67 ms**. ⇒ ~4,5–4,9 ms je Layer auf ALLEN Stufen; kein Ausreißer.
Zum Vergleich llama.cpp ~2,5 ms/Layer. Die Restlücke steckt in den Kernel-
Kosten je Layer (MoE grouped ~1,0 ms, Rest Attention/Indexer/mHC/Norms/
Shared-Experts). Nächster Schritt: Per-Kernel-Profil (nsys, Boot 59).

### 2026-09-03 06:50 — PR #455 (QSA) von 1Cat geschlossen: Dispatch-Irrtum unsererseits

Maintainer yangzhuxinyzx (und Leonccaa im Review): upstream-main dispatcht
`qwen4_exp/__init__.py` rein nach `is_rocm()`; CUDA (auch V100/RTX) lädt
den `nvidia/`-Baum. Die Weiche „Capability < 80 → `amd/`" ist eine
Ergänzung UNSERES Ports (weil `nvidia/` `cute_dsl.skinny_gemm` braucht,
das im 1.3.0-Wheel fehlt; `amd/` hat dafür einen No-Op). Unser
`tools/qsa_bench.py` und alle Produktionslogs (32× `amd/ops/qsa`) messen
den amd-Baum — die 19,3×/4,2× aus #441 gelten für die dort untunte
GB300-Tabelle; `nvidia/ops/qsa.py` hat bereits einen sm70-Retune
(4 Warps, BLOCK_N 32 ab 512 Programmen). Antwort mit Eingeständnis
gepostet (issuecomment-5520570087). Was upstream real bleibt: Turing
(sm75) ist vom `is_sm70`-Retune ausgenommen, und das 80-KiB-Tile kann
auf 64 KiB nicht starten → Kandidat für einen NEUEN PR gegen
`nvidia/ops/qsa.py` (Gate `< 80` + smem-Clamp), aber erst nach Benchmark
N16 gegen deren N32/N64@W4 auf V100 und RTX 8000 mit der nvidia-Datei.
Lektion: vor Upstream-PRs den Dispatch auf UPSTREAM-main verifizieren,
nicht auf dem Fork.

### 2026-09-03 07:00 — nsys-Per-Kernel-Profil (Boot 59; benchmarks/nsys-dsv4-het-2026-09-03.txt)

Letzte 60 s (Essay 200 + Code 200 Token, ~160 Decode-Schritte). Die
NCCL-SendRecv-Zeit auf den Stufen 1–4 (6–14,5 s von 60) ist Pipeline-
Wartezeit (GPU spinnt im Recv), keine Rechenzeit. Rechenzeit je Layer und
Schritt auf PP0 (RTX, 11 Layer, ~3,7 ms/Layer):

| Kernel | ms/Layer·Schritt | Launches/Layer·Schritt |
|---|---:|---:|
| `skinny_nvfp4_moe_simt` (grouped MoE) | 1,3 | 2 |
| `_sparse_attn_decode_ragged_kernel` (ROCm-Triton-Sparse-Decode) | 0,65 | 1 |
| `reduce_kernel` (torch-Reduktionen: RMSNorm/mean/amax) | 0,43 | ~68 |
| `Kernel2` (TileLang-generiert) | 0,32 | ~6 |
| `elementwise_kernel` + vectorized/unrolled | 0,43 | ~125 |
| `QnxKernelTrace` (Tracing-Artefakt, im Code nicht auffindbar) | 0,20 | ~85 |
| cuBLAS turing_fp16 GEMMs (wo_a/wo_b dequant16 u. a.) | 0,23 | 2 |

Befund: neben MoE (35 %) und Sparse-Decode (18 %) gehen ~1,2 ms/Layer in
**~280 Kleinst-Launches** der pre-Hopper-Torch-Referenzpfade (mHC generic
`_mhc_post_torch_generic`/`_hc_head_torch_generic`, Norms, Indexer-Glue) —
auch im Graph kostet jeder Knoten 2–5 µs. Die TileLang-mHC-Kernel werden
unter fp16 bewusst umgangen (`if residual.dtype == torch.float16: generic`).

Hebel in Reihenfolge des Ertrags:
1. mHC-Pfad fusionieren (ein Triton-Kernel für post+pre statt ~30 Torch-Ops
   je Layer) oder die TileLang-Kernel für fp16 freischalten — ~0,8 ms/Layer.
2. `QnxKernelTrace` identifizieren/abschalten — 0,2 ms/Layer geschenkt.
3. Sparse-Decode-Kernel: ROCm-Triton-Impl gegen die sm70-Triton-Sparse-Kernel
   (`sm70/sparse_kernels.py`, jetzt worker-lokal aktiv) messen — 0,65 ms.
4. MoE: w13/w2-Launch fusionieren (Aktivierung in-Kernel) — ~0,3 ms.
Ziel ~2,5 ms/Layer (llama.cpp-Niveau) ⇒ ~110 ms/Schritt ⇒ ~22 tok/s Prosa,
~40 tok/s Code.

Endzustand 07:00: GPUs frei, kein Server. Fork: alles committed bis
8e27308, danach nur Doku/Benchmark-Dateien (uncommitted).

### 2026-09-03 10:15 — Hebel 1 ERLEDIGT: mHC fp16 freigeschaltet (Upstream-Backport) — 21,3/26,7 tok/s

**Kernbefund:** Der Fork-Patch `mhc_tilelang.py` stammte aus der Zeit vor dem
1.3.0-Rebase („TileLang-Kernel sind hart bf16") und bog an allen vier Public-
Entries fp16 in die Torch-Generic-Pfade ab — dabei sind die TileLang-Kernel
im 1.3.0-Wheel längst dtype-parametrisiert (`use_fp16`), und Upstream-HEAD
(187b932) hat zusätzlich eine reine sm70-Triton-Dekoderoute samt FP32-Stage
(`docs/design/sm70_deepseek_v4_mhc_decode.md`, dort bitweise verifiziert,
+7 % TPOT auf V100). Die Bypässe maskierten das alles.

**Umsetzung (importiert, nicht nachgebaut):** `fork_patches/mhc_tilelang.py`
neu = Upstream-HEAD-Datei + Fork-Block (Torch-Referenzen + `mhc_post_fp32`
für die DSpark-Aux-Extraktion), Bypässe raus. NEU `fork_patches/mhc_triton.py`
= Upstream-HEAD `kernels/mhc/triton.py` (sm70_mhc_prenorm_staging /
sm70_mhc_post / sm70_mhc_pre_norm_from_staging; das Wheel hat nur den
partiellen hc_head_triton). Drei nötige Anpassungen:
1. Capability-Checks worker-lokal (`torch.cuda.get_device_capability(
   torch.cuda.current_device())`) statt Device 0 — sonst verlieren die
   V100-Stufen im Het-Boot ihre sm70-Route (bekannte Device-0-Falle).
2. M=1: Upstream verlangt hart den nativen Op `_C.sm70_glm_mhc_pre_norm_out`
   (im 1.3.0-Wheel nicht vorhanden, kam nach dem Wheel-Build). Fork: M=1
   läuft denselben Triton-Finalstage wie M>1 (Grid (1,), gleiche Mathematik).
3. hc_head: TileLang-Codegen CRASHT auf pre-Ampere (sm70 UND sm75 verifiziert,
   tvm-ffi-Exception segfaultet beim Traceback-Rendern) → auf < sm80 bleibt
   die Torch-Referenz. Kostet nichts: läuft 1× pro Step auf der letzten Stufe.

**Werkzeug-Falle (neu):** TileLang-JIT außerhalb des Serve-Skripts braucht
`CUDA_HOME=…/.cuda-nvcc-deb/usr/local/cuda-12.8` — das System-nvcc (12.0)
bricht bei sm70 an bf16-Template-Intrinsics in tl_templates/cuda/common.h.
Und: Exceptions aus dem tvm-ffi-Callback (Compile-Fehler) reißen den Prozess
per Segfault — Kernel einzeln in Repros testen, stdout ungepuffert.

**Verifikation:** A/B gegen die Torch-Referenz auf V100 UND RTX 8000, alle
Entries (fused/post/pre/broadcast/hc_head), M=1/6/64: max rel ≤ 4e-4
(fp16-Rauschen), keine NaN. Server Boot 60 (boot-dsv4-k5-mhc.out): Kohärenz
8/8, davon 5/8 bitidentisch — exakt die Quote der akzeptierten Läufe
(grouped, graphs). Akzeptanz gesamt 51 % (1002/1960 über Kohärenz+Bench,
Positionsprofil 86/67/45/36 %). **Essay 21,3 tok/s (war 13,1), Code 26,7
(war 20,4)**; llama.cpp-Referenz 21/38 — Prosa eingeholt.

**Deploy:** Beide Dateien nach site-packages kopiert; bootstrap-sm70.sh um
`deploy mhc_triton.py` ergänzt; fork_patches/README.md um beide Zeilen
ergänzt (Tabelle war unvollständig). Alles UNCOMMITTED.

**Offen/nächstes:** nsys-Profil neu ziehen (das 07:00-Profil ist überholt;
`QnxKernelTrace` war vermutlich Teil des Torch-mHC-Pfads), dann Hebel
Sparse-Decode-A/B und MoE-w13/w2-Fusion neu bewerten. Code (26,7 vs 38)
bleibt das Ziel.

### 2026-09-03 10:20 — Re-Profil nach mHC-Backport (Boot 61; benchmarks/nsys-dsv4-het-mhc-2026-09-03.txt)

nsys via `nsys launch --session-new=… --trace=cuda --cuda-graph-trace=node
--trace-fork-before-exec=true bash scripts/serve-…sh`, dann `nsys start/stop`
um das Benchfenster (Rezept jetzt reproduzierbar; qdstrm →
/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter → `nsys export --type
sqlite` → scratchpad/nsys_kernsum.py). Unter Profiler: 19,7/24,6 tok/s.

**mHC-Hebel bestätigt abgeräumt:** `QnxKernelTrace` ist KOMPLETT verschwunden
(war also Teil des Torch-mHC-Pfads), `reduce_kernel`/`elementwise` auf den
V100-Stufen um Größenordnungen runter (91k → 21k Launches). mHC kostet je
V100-Layer·Step jetzt ~0,1 ms (pre_norm 0,04×2 + dot-stage 0,01 + Rest)
statt ~0,8. Auf den RTX-Stufen: `mhc_fused_tilelang_kernel` 0,04 ms/Call.

**Neue Rangliste Rechenzeit je V100-Layer·Step (346 Steps im Fenster):**
MoE grouped 1,04 ms > Sparse-Decode 0,77 ms > wo-GEMMs 0,18 >
skinny_fp8_qpn8 0,14 > Kernel2 0,08 > mHC 0,1 > Elementwise-Rest ~0,13
⇒ ~2,5 ms Compute/Layer — das llama.cpp-Niveau ist auf Kernelebene
praktisch erreicht; die verbleibende Tempolücke bei Code (26,7 vs 38)
steckt jetzt in Schrittfixkosten (Drafter+Sampling auf PP4, PP-Nähte),
nicht mehr in den Layer-Kernels. Hebel-Reihenfolge unverändert sinnvoll:
MoE-w13/w2-Fusion (~0,3), Sparse-Decode-A/B (~0,2–0,3 realistisch).

Endzustand 10:25: GPUs frei, kein Server. mHC-Backport + Doku UNCOMMITTED.

### 2026-09-03 11:00 — QSA-PR-Gate ERFÜLLT: N16 gewinnt auf beiden Archs, sm75 ist upstream sogar KAPUTT

Anlass: SabaTech-dev-Kommentar in #441 (haben unsere QSA-Tiles adoptiert,
wollen das flash_attn_v100-64x80-Diff) + valentijnvenus bittet um PR.
Das Benchmark-Gate aus dem 06:50-Abschnitt (N16 vs N32/N64@W4 mit der
NVIDIA-Datei auf beiden Karten) ist jetzt erledigt: Harness
`tools/qsa_nvidia_ab.py` lädt die origin/main-Datei (ca73a34) per importlib
und übersteuert NUR `_qsa_sparse_launch_profile` je Messpunkt; Ergebnisse
`benchmarks/qsa-nvidia-ab-2026-09-03.txt`. Kernbefunde:
1. **sm75-Korrektheitsbug upstream:** GB300-Tabelle wählt N64 für alle
   Prefill-Regime; bei D=256 → Triton OutOfResources auf Turings 64 KiB —
   der Kernel startet auf sm75 GAR NICHT (alle rows ≥ 64). Deren
   sm70-Retune, auf sm75 erzwungen, repariert nur ≥512 Programme.
2. **N16@W4 läuft überall und gewinnt:** sm70 1,2–2,6×, sm75 1,16–1,20×
   gegen das beste lauffähige Upstream-Profil, Numerik identisch.
3. **Decode-Kleinstprofile nicht anfassen** (GB300 dort bereits optimal;
   unser S8 wäre bei rows=1 3× langsamer).
PR-Entwurf (Patch, Zahlen, Maintainer-Frage N32-vs-N16) liegt in
`upstream-contrib/03-1cat-issues/pr-qsa-pre-ampere-tiles.md`; Antwort-
Entwurf für SabaTech (64x80-Diff, sauber aus `git diff v1.3.0` im
1Cat-Clone extrahiert) in `reply-441-sabatech-64x80.md`. BEIDES wartet
auf Freigabe Peuqui (Posten/Branch/Push = outward-facing).

Endzustand 11:00: GPUs frei, kein Server. Uncommitted: mHC-Backport
(fork_patches + bootstrap + README), tools/qsa_nvidia_ab.py, zwei
Benchmark-Dateien, zwei upstream-contrib-Entwürfe, Handover.

### 2026-09-03 11:30 — Alles committed/gepusht, #441-Antworten raus, QSA-PR #469 eingereicht

Commits dd8e012 (mHC-Backport), 934a3f5 (QSA-Gate), 3847a7c (Status) auf
`pp-mtp-merge` gepusht; AIfred-Repo: EN-Message-Hub-Doku (82b9d69d).
Issue-#441-Kommentare gepostet: SabaTech (64x80-Diff,
issuecomment-5523207244), valentijnvenus (issuecomment-5523207906).
**PR https://github.com/1CatAI/1Cat-vLLM/pull/469** eingereicht (Freigabe
Peuqui): Branch `qsa-pre-ampere-launch-profile` im Fork, EIN Commit
b2663602 auf ca73a34, drei Dateien (qsa.py + beide Testdateien — Upstream
hat Tests auf `_qsa_sparse_launch_profile`, via GitHub-Code-Suche
gefunden). Verifikation vor dem Einreichen: alle 10 angepassten
Test-Assertions grün gegen die gepatchte Datei; A/B-Harness gegen die
GEPATCHTE Datei auf beiden Karten (Default-Dispatch reproduziert die
N16-Zeiten, Decode-Kleinstprofile unverändert); Capability-Gate
`not has_device_capability(80)` auf RTX und V100 direkt geprüft (beide
True). PR-Tabelle exakt aus benchmarks/qsa-nvidia-ab-2026-09-03.txt
(Reproduktionslauf; sm75-Ratio dort 1,16-1,17x — der erste Lauf hatte
1,20x, beide real gemessen, PR zitiert konservativ die committete Datei).
Peuqui-Regeln ab heute im Memory: äußerst penibel bei Außenwirkung,
täglicher Upstream-Check beim Projektstart. GPUs frei, kein Server.
