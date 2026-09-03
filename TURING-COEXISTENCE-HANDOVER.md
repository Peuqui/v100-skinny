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

### 2026-09-03 12:00 — Hebel Sparse-Decode: GESCHLOSSEN (negativ), plus zwei Strukturbefunde

**Befund 1 — sm70-Sparse-Impl ist im PP-Verbund nicht aktivierbar:**
`_select_v4_sparse_impl` (attention.py) wählt durch den Prä-Hopper-
Catch-all IMMER den ROCm-Impl; der exakt-sm70-Zweig ist tot. Der Versuch,
ihn per Reorder worker-lokal zu aktivieren (Boot 62,
boot-dsv4-k5-sm70sparse.out), bootet zwar, stirbt aber beim ERSTEN
Request: die Stufen fahren dann VERSCHIEDENE Attention-Backends
(V4_SM70_TRITON_SPARSE auf V100 vs ROCM_V4_FLASHMLA_SPARSE auf RTX) und
der Cross-Stage-Metadata-Vertrag bricht — PP0-Prefill verliert seine
topk_indices (`assert topk_indices is not None`, amd/rocm.py:795).
Reorder ZURÜCKGEDREHT; das Warum steht jetzt als NOTE-Kommentar direkt
im Dispatch (fork_patches/deepseek_v4_attention.py). Isoliert ist die
Selektion korrekt (Repro: alle 5 Ranks richtig).

**Befund 2 — Standalone-A/B sagt: Wechsel lohnt ohnehin nicht.**
tools/sparse_decode_ab.py (V100, 64H/D512, SWA128+TopK512, identische
Eingaben, benchmarks/sparse-decode-ab-2026-09-03.txt): Ausgaben
BITIDENTISCH (max|diff|=0) — gleiche Dekodierung. Tempo am
Produktionspunkt b=6: sm70-Kernel 0,88× (12 % LANGSAMER als ragged);
nur b=1 wäre 1,12×, den Fall fahren wir nicht. Der splitk-Pfad der
sm70-Kernel liefert NICHT-FINITE Ausgaben (schlummernder Code, nie
produktiv; nicht weiter debuggt). ⇒ Ragged bleibt zu Recht Produktion.
Hinweis: Absolutniveau des Harness (≈3,8 ms) liegt über dem
Produktionsprofil (0,77 ms/Layer·Step) — Cache-Residenz/Kontextlage
unterscheiden sich; der Relativvergleich ist davon unberührt.

**Verbleibende Hebel:** MoE-w13/w2-Fusion (kernels/skinny_kernels.cu)
und PP4-Schrittfixkosten (Drafter+Sampling, 67 vs 38 ms je Stufe).
Endzustand 12:00: GPUs frei, kein Server. Uncommitted seit c4ca076:
attention.py-NOTE, tools/sparse_decode_ab.py, Benchmark-Datei, Handover.

### 2026-09-03 13:00 — Step-Anatomie komplett + MoE-Mikro-Tuning negativ + NEUER Top-Hebel: RTX-Tiny-GEMMs

**Step-Anatomie aus dem Boot-61-SQLite** (tools/pp4_attribution.py,
tools/nsys_kernsum.py; Timeline-Zerlegung, ~158 ms/Step unter nsys):
PP0-Phase 38,5 ms Spanne (11L+embed, fragmentiert; die Mini-Bursts auf
dev1-4 währenddessen sind harmlose Metadata-Vorbereitung à ~0,24 ms) →
Verify-Welle dev1/2/3 je ~22,6 ms mit NUR ~0,2 ms Naht-Übergaben (die
früher vermuteten „33 ms Nähte" sind WIDERLEGT) → dev4 27,1 ms
(8L+Verify-Tail) → Broadcast-Flurry 0,3 → **Drafter-SOLO auf dev4
14,2 ms, alle anderen GPUs idle** → Turnaround 1,7. Serialisiertes
Compute gesamt 140,7 ms/Step (PP0 37,3 / V100 je ~22 / PP4 36,8).

**MoE-Mikro-Tuning: VIER Hypothesen getestet, alle verworfen** (Harness
scripts/nvfp4_skinny_moe_grouped_test.py mit VLLM_SKINNY_NVFP4_SRC auf
Scratchpad-Kopie; Baseline T=6 ≈ 1,00-1,04 ms, 2,4× Traffic-Floor
390 MB/0,43 ms; ncu ohne sudo nicht möglich, ERR_NVGPUCTRPERM):
KC=2048 → 1,13 (Occupancy-Verlust); Akku-Ketten-Splitting → 1,11
(nicht FMA-latenzgebunden); uint4-Loads → 1,03 (neutral, nicht
Load-Issue-gebunden); Double-Buffering → 1,36 (Registerdruck).
Fazit: Der Kernel ist gegen Mikro-Tuning robust; die Restlücke zum
Floor braucht ein Redesign (Tensor-Core-Pfad + Layout-Prepack) —
Tagesprojekt, nicht angefangen. Numerik in allen Tests OK.

**NEUER Top-Hebel — RTX-Stufen verbrennen ~12 ms/Step in
cuBLAS/CUTLASS-Tiny-GEMMs** (nsys „Kernel2" =
`cutlass::Kernel2<cutlass_75_wmma_tensorop_f16_s161616gemm_f16_16x16_
128x2_tn>`, 16×16-Tiles à ~0,106 ms): dev0 4,3 ms/Step (64+14
Launches), dev4 7,7 ms/Step (68+11, inkl. Drafter-Linears ~4,5).
Die V100 zahlt für dieselben logischen Matmuls nur 0,65 ms/Step —
dort laufen sie über skinny_fp8_qpn8 (40 Launches/Step); auf sm75
gehen die Gewichte dequant16 → cuBLAS (is_bmm-Route). **Potential
~10 ms/Step (~6-7 %).** Ansatzpunkte: gemm_qpn8_blk_wmma (Tensor-Core-
Tiles aus gepacktem Layout, bisher „prefill band") für Decode-M≤8 auf
sm75 ertüchtigen, oder Turing-Skinny-GEMM für die dequant16-Matmuls
(+ Drafter-Linears getrennt betrachten). Zweiter Hebel: Drafter-Solo
14,2 ms (Drafter-Quantisierung wäre der Weg, größeres Projekt).

**Präzisierung nach Code-Lektüre:** `maybe_sm70_dsv4_fp16_gemv` ist NICHT
der Unterschied — der Kernel ist auf `x.shape == (1, 4096)` (M=1) gegated
und feuert bei K=5-Spekulation (M=6) auch auf V100 nie. Die V100-
Ersparnis kommt aus der QPN8-Route (skinny_fp8_qpn8, M≤8); auf sm75
laufen dieselben Gewichte fp16-dequantisiert durch cuBLAS. **Nächster
konkreter Schritt:** Re-Profil mit `nsys … --trace=cuda,nvtx` (die
NVTX-Phasen-Brackets aus dem gpu_model_runner-Fork-Patch aktivieren),
um die 64-78 Tiny-GEMM-Launches/Step den Modulen zuzuordnen (Kandidaten:
wo_a-is_bmm-Einsum, wq_b/q-lora, Indexer-Projektionen, Drafter-Linears);
danach entscheiden: gemm_qpn8_blk_wmma (WMMA, läuft nativ auf sm75) für
Decode-M≤8 ertüchtigen ODER die dequant16-Entscheidung auf sm75 kippen
(Gewichte gepackt lassen). Boot-61-Profil hat KEIN NVTX (nur cuda-Trace).

Endzustand 13:00: GPUs frei, kein Server. Uncommitted zusätzlich:
tools/pp4_attribution.py, tools/nsys_kernsum.py, Handover.

### 2026-09-03 13:30 — PR #469 GEMERGED (Upstream-Erstkontakt erfolgreich)

yangzhuxinyzx hat #469 um 11:15 APPROVED und sofort gemergt
(Merge-Commit 65d25c1): „Validated against main after #466: merge is
conflict-free; targeted QSA suites report 20 passed, 1 skipped; targeted
pre-commit passes. The change remains a pre-Ampere fallback/profile
improvement and does not override the grouped Page4 fast path."
Damit ist der sm75-OutOfResources-Bug upstream gefixt und unser
N16-Profil offiziell drin. Für UNSEREN Fork/Boots ändert sich nichts
(wir fahren den amd/-Zweig); beim nächsten Wheel-/Rebase-Zyklus kommt
der Fix von selbst mit. #441 bleibt offen (Sammelthread); SabaTech-
und TianHengZhuang-Antworten stehen noch aus.

### 2026-09-03 14:00 — Tiny-GEMM-Hebel VOLLSTÄNDIG identifiziert (ohne Boot, aus Grid-Formen)

Methode: Grid-Dimensionen der cutlass-Launches aus dem Boot-61-SQLite
gruppiert (gridX/Y/Z stehen in CUPTI_ACTIVITY_KIND_KERNEL) und die
Formen empirisch per torch.profiler auf der RTX gematcht — `x @ W.t()`
(Linear, tn-Layout) wählt auf sm75 exakt den
`cutlass_75_wmma_…16x16_128x2_tn`-Kernel. Kein NVTX/Boot nötig.

**Befund 1 — der Monster-Launch ist der lm_head:** grid=(8,1010,1) ⇔
vocab 129280/128 = 1010. 2 Launches/Step à 1,78 ms = **3,6 ms/Step auf
PP4** (mutmaßlich 1× Verify-Logits M=6, 1× Draft-Logits M=5 gebündelt).
`head.weight` liegt UNQUANTISIERT als BF16 (129280×4096 = 1,06 GB) im
Checkpoint — der GEMM läuft mit ~595 GB/s = 89 % der RTX-Bandbreite
memory-OPTIMAL. Hebel wäre NUR Gewichtsformat: head auf NVFP4/FP8
quantisieren (→ 0,27–0,53 GB ⇒ ~1,3–2,7 ms/Step) — ABER das berührt
direkt die Token-Wahl (Argmax bei knappen Logits) ⇒
QUALITÄTSENTSCHEIDUNG PEUQUI, nicht autonom machen.

**Befund 2 — die Layer-weisen Tiny-GEMMs sind die dequant16-Linears:**
je RTX-Layer ~4-5 tn-GEMMs (wkv-, wo_b-, wq-Klasse; Grids (8,32,5/6),
(8,12,1), (8,32,1), (8,64,1)) ≈ 3,5 ms/Step auf dev0 + ~3,3 auf dev4.
Im Checkpoint SIND diese Gewichte quantisiert; die sm75-Route
dequantisiert sie beim Laden zu fp16 (QPN8-blk-is_bmm-Entscheidung) und
zahlt dann 2-4× Traffic. **Konkretes nächstes Paket:** Gewichte auf sm75
gepackt halten und für Decode-M≤8 über `gemm_qpn8_blk_wmma` (WMMA läuft
nativ auf sm75, bisher nur „prefill band") servieren — gleiche
Dequant-Mathematik, nur in-Kernel ⇒ kein Qualitätsthema, ~3-4 ms/Step
Potential. Einstieg: fork_patches/modelopt.py (Route/Census) +
marlin.py-Routenkarte; Standalone-A/B vor jedem Boot.

Hebel-Rangliste damit: (1) QPN8-blk-WMMA-Decode auf sm75 ~3-4 ms,
(2) lm_head-Quantisierung ~1,3-2,7 ms (NUR mit Peuqui-Entscheid +
Qualitäts-Eval), (3) Drafter-Solo 14,2 ms (Drafter-Quant, groß),
(4) MoE-Redesign (Tagesprojekt).

### 2026-09-03 15:00 — Hebel 1 UMGESETZT: QPN8-Route auch auf sm75 (Boot 63) — BEHALTEN

**Überraschung im Standalone-Bench** (benchmarks/fp8_blk_sm75_decode_bench
.py, RTX 8000): der „Volta"-m8n8k4-Kernel `gemm_qpn8_blk` schlägt
fp16-cuBLAS bei M≤8 um ~2× (0,060 vs 0,118-0,126 ms auf wq_b/wo_a/wo_b)
— bei Decode-M ist alles memory-bound, die Turing-mma-Skepsis greift
nicht; der WMMA-Kandidat war sogar am langsamsten. Numerik 4,3e-4 rel
(= fp16-Rundung).

**Umsetzung = EIN Gate** (fork_patches/qpn8_blk.py): die Bedingung
`capability == (7,5)` vor dem persistenten fp16-Dequant gestrichen —
sm75 behält die Gewichte gepackt und fährt dieselbe M-Band-Dispatch wie
V100 (blk ≤8, mt2 ≤16, wmma ≤64, transienter Dequant + cuBLAS darüber).
`is_bmm` (wo_a-Einsum) bleibt auf BEIDEN Archs dequant16 — strukturell.
Nebeneffekt: gepackte Gewichte = halber VRAM für diese Linears auf RTX.

**Validierung Boot 63** (boot-dsv4-k5-qpn8sm75.out): Census 0×
sm75-dequant, Linears gepackt (fused_wqa_wkv/wq_b/wo_b/shared/main_proj),
46× is_bmm-dequant korrekt; 0 ERRORs. Kohärenz **5/8 bitidentisch zur
nvidia-Referenz = exakt die akzeptierte Quote**. Tempo: der EINZIGE
bitidentische Lang-Workload (longctx) **+3,7 %**; prose +5,6 %, code
+10,9 % (Probe) bzw. Bench-200-Token code 27,2 (+0,5); essay 20,4
(−0,9) — Essay-Delle ist Akzeptanz-Drift (38,8 % vs ~42) durch die
4e-4-Logit-Verschiebung, kein Kernel-Thema: ms/Step gemessen 144
(essay) / 146,5 (code). Akzeptanz-Lotterie pro Prompt überdeckt bei
Einzelprompts den ~2-3-ms-Struktur-Gewinn. ENTSCHEIDUNG: behalten
(strukturell richtig, Kohärenz-Quote unverändert, VRAM-Gewinn).

Endzustand 15:00: GPUs frei, kein Server. Commits 66fe3a7 (Tagesstand)
+ Folgecommit db9470a (qpn8_blk-Umstellung) auf pp-mtp-merge.

### 2026-09-03 15:30 — lm_head-FP8: Vorab-Messung NEGATIV, Idee VERWORFEN

Gate-Messung vor jeder Code-Änderung (offline, echter head.weight auf
V100): e4m3-Blockquant [128,128] des BF16-Heads hat **2,64e-2 rel
Gewichtsfehler** — e4m3 hat nur 3 Mantissenbits, der ~2-3-%-Fehler ist
formatinhärent und ließe sich auch mit feineren Blöcken nicht unter
~1e-2 drücken (nicht vergleichbar mit den 4e-4 der schon-quantisierten
FP8-Layer — dort ist der Quantisierungsfehler im Checkpoint „eingebaut",
wir rechnen nur exakt auf den vorhandenen Zahlen). Proxy-Flip-Messung
(RMS-normierte synthetische Hiddens, 2560 Positionen): **5,27 %
Argmax-Flips**, Logit-Rauschen median 1,8 bei Top1-Top2-Marge median
3,6 — meilenwert vom vereinbarten Promille-Gate entfernt. Reale Hiddens
hätten größere Margen, aber Rauschen/Marge ≈ 0,5 ist indiskutabel.
llama.cpps Q8_0-Head ist INT8 (~7-8 effektive Bits, Fehler ~4e-3) —
unsere QPN8-Kernel sind e4m3; ein präzisionsgleicher Head bräuchte
einen neuen INT8-Pfad. VERWORFEN; Ersparnis wäre eh nur ~1,3 %.
Der Head bleibt BF16 — Genauigkeit vor Tempo, wie vereinbart.

### 2026-09-03 16:00 — MoE-Redesign UMGESETZT: moe_qpn (Tensor-Core + QPN-Prepack) — Standalone 1,3-1,45×

**Design (importiert statt neu erfunden):** Neuer Kernel `skinny_nvfp4_moe_qpn`
(kernels/skinny_kernels.cu, Binding `moe_qpn`) = moe_simt-Routing-Skelett
(device-seitig perm/offsets, inaktive Experten exiten vor Gewichtszugriff,
Multipass bei cnt>8, capture-sicher) × QPN2-Compute-Dataflow (mma.m8n8k4,
SPLITK/NACC-Templates). Gewichte liegen dafür PER EXPERTE im QPN-Fragment-
Layout ([tile N/32][group K/16][lane 32]×8B, dieselbe `_qpn_prepack`-
Permutation wie die dense Route). `__launch_bounds__(32*SPLITK)` noetig
(erster Launch starb an cudaErrorLaunchOutOfResources — der Pass-Loop kostet
Register gegenueber dense qpn2).

**Upstream-Check vorab (Tagespflicht):** 1Cat main hat v1.5.0 getaggt, unser
QSA-Fix ist als 4a69044 drin; deren `nvfp4_sm70_moe.py` (TurboMind compact
grouped) ist auf feste Contracts gegated (Qwen3.6/3.8, GLM-5.3 — DSv4 NICHT
dabei) → kein Backport-Kandidat, Eigenbau aus Fork-Bausteinen war der Weg.

**Standalone (scripts/nvfp4_skinny_moe_qpn_test.py, echte Layer-5-Gewichte):**
Kernel T=6: w13 0,53→0,38 ms, w2 0,34→0,20 ms ⇒ Kernel-Summe 0,58 ms =
**1,35× Traffic-Floor** (moe_simt: 2,3×). Full Layer 1,40-1,46× auf V100,
1,27-1,36× auf RTX 8000. Sieger-Configs auf BEIDEN Archs identisch:
w13 (16,1), w2 (8,1) — kein Turing-Sonderfall. max|diff| ≤2,4e-4 gegen die
Checkpoint-Layout-Referenz, Multipass-Haertefall OK.

**Route (fork_patches/nvfp4_skinny_moe.py, deployed):** Prepack IN-PLACE in
`process_weights_after_loading` (Byte-gleiche Permutation von Gewichten UND
Scale-Rastern; Shapes/Footprint unveraendert, kein VRAM-Doppel; laeuft in
der Load-Phase vor der KV-Reservierung). Decode/Verify → `moe_qpn` (Configs
via VLLM_SM70_NVFP4_MOE_QPN_CFG, Default "16,1,8,1"); Prefill-Loop →
`gemm_qpn` M≤16 + 16er-Chunks (statt gemm_simt/gemm_wmma, die das alte
Layout braeuchten). Beide Testskripte auf das neue Layout angepasst
(grouped_test: moe_simt bleibt auf Checkpoint-Layout als layout-
unabhaengiger Referenz-Anker).

**BEFUND nebenbei — gemm_qpn_simt hat einen latenten Prepack-Mismatch:**
rechnet auf echten Expert-Bytes strukturell falsch (max|diff| ~1e0),
waehrend gemm_qpn auf denselben Packs bei M 1..16 korrekt ist. Produktiv
unerreichbar (dense Route nutzt ihn nur unter VLLM_SKINNY_DROP_CT=1,
Default 0) — die MoE-Route meidet ihn; wer DROP_CT je aktiviert, muss das
erst fixen.

**Erwartung Server:** MoE war 1,04 ms/Layer·Step (nsys V100) → ~0,29 ms
weniger × 43 Layer ≈ 12,5 ms/Step ⇒ ~+9 % (Prognose ~23 Essay / ~29 Code).
Boot + Kohaerenz + Bench: naechster Abschnitt.

### 2026-09-03 17:00 — MoE-Redesign VALIDIERT im Server (Boot 64): 112-115 ms/Step (war 144-147)

Boot ueber scripts/serve-deepseek-het-graphs.sh (unveraendert; die Extension
baut den neuen Kernel automatisch aus dem Repo-Source). 0 ERRORs, Graphs
capturen, QPN-Prepack laeuft im Weight-Load jeder Stufe.

**Kohaerenz: 8/8, davon 5/8 bitidentisch zur nvidia-base-Referenz — exakt
die akzeptierte Quote** (results-coherence-het-k5-moeqpn.json; Abweichler
sind die bekannten fp16-Drift-Freitexte).

**Struktur-Messgroesse ms/Step (wall/drafts via /metrics, 200-Token-Laeufe):
essay 112,8 / code 114,5 ms — vor dem Umbau 144/146,5 ⇒ −21 %.** Damit ist
llama.cpps Schrittlatenz (~110 ms je 6-Token-Schritt) praktisch erreicht.
Der Gewinn liegt UEBER der 12,5-ms-Prognose aus dem Layer-Anteil — der
Drafter-Block und die Verify-Pfade fahren dieselbe MoE-Route mit.
tok/s (Scratchpad dsv4_bench.py, eigene Prompts, Akzeptanz-Lotterie
beachten): essay 25,2 (Akz. 38 %), code 28,1 (Akz. 46 %).

Endzustand 17:00: GPUs frei, kein Server. UNCOMMITTED: kernels/
skinny_kernels.cu (moe_qpn), fork_patches/nvfp4_skinny_moe.py (Route +
in-place-Prepack), scripts/nvfp4_skinny_moe_qpn_test.py (neu),
scripts/nvfp4_skinny_moe_grouped_test.py (Pack-Anpassung), Kohaerenz-JSON,
Handover. venv ist deployt (cp), bootstrap-Zeile existierte schon.

**Verbleibende Hebel:** Drafter-Solo-Phase neu vermessen (14,2 ms-Zahl ist
nach diesem Umbau veraltet — der Drafter-MoE wurde mitbeschleunigt), dann
Drafter-Quantisierung bewerten; RTX-Tiny-GEMM-Rest (lm_head bleibt BF16,
VERWORFEN-Notiz 15:30 beachten).

### 2026-09-03 17:45 — Drafter-Solo NEU vermessen (Boot 65, nsys): Drafter-Quantisierung ZURÜCKGESTELLT

Frisches nsys-Profil nach dem MoE-Umbau (Rezept wie 10:20; unter Profiler
23,5/26,0 tok/s; SQLite nsys-dsv4-het-moeqpn.sqlite, Attribution
benchmarks/nsys-dsv4-het-moeqpn-2026-09-03.txt, 403 Steps im 60-s-Fenster).

**Timeline-Befund: Batch-1-Decode ist strikt seriell** — jede Stufe laeuft
solo (dev4-Solo-Zeit == dev4-Busy-Zeit). Kernelzeit je Step: dev0 26,3 /
V100 16,3-16,7 / dev4 27,3 ms (Summe ~103 unter Profiler).
`moe_qpn` kostet auf V100 0,50 ms/Layer·Step (moe_simt: 1,04) — der
MoE-Anteil hat sich im Serving mehr als halbiert.

**Drafter+Sampling+Heads: nur noch ~8,3 ms/Step** (27,3 − ~19 fuer 8
RTX-Layer; alte Zahl 14,2 — Drafter-MoE und FP8-Linears reiten die neuen
Routen mit). Quantisierbar bleibt der BF16-Rest: Kernel2-Tiny-GEMMs
3,4 ms (Drafter-Linears/Heads) + Anteil der Draft-Logits am
turing-1688-GEMM (1,49 gesamt) ⇒ realistisch **~2-3 ms/Step ≈ +2 %** —
gegen ein Checkpoint-Transplantat-Projekt (mtp-quant) mit Akzeptanz-
Risiko + Qualitaets-Eval. ENTSCHEIDUNG: zurueckgestellt (schwaches
Kosten-Nutzen-Verhaeltnis nach dem MoE-Umbau); Wiedervorlage nur, wenn
ein 8-Bit-Drafter-Transplantat aus anderem Grund entsteht.

**Kernel-Rangliste V100-Stufe jetzt:** Sparse-Decode 0,74 ms/Layer (Top,
A/B negativ = geschlossen) > moe_qpn 0,50 > Rest klein. Auf Kernelebene
ist die Kiste nahe am Optimum; die Restluecke zu llama.cpps Code-38
haengt an Akzeptanz je Prompt, nicht mehr an Schrittlatenz (113 vs
110 ms). Commit-Stand: 10f3d27 (MoE-Redesign) auf pp-mtp-merge.
Endzustand 17:45: GPUs frei, kein Server.

### 2026-09-03 18:15 — Gewichts-Beleg fuer das 1Cat-Angebot: Flash-Next-Geometrie 2,5-3,4×

Vor dem #441-Kommentar (Entwurf comment-441-grouped-moe-offer.md) zwei
Absicherungen gemessen:
1. **Shape-Generik belegt** (scripts/nvfp4_skinny_moe_qpn_synth_fn.py):
   moe_qpn auf der Qwen3.8-Flash-Next-TP1-Geometrie (E=512, topk=10,
   w13 1280x2560, w2 2560x640, synthetische NVFP4-Bytes, Anker moe_simt
   auf denselben Bytes): T=1 **3,2-3,4x**, T=6-8 2,5-2,9x vs moe_simt,
   Numerik identisch auf V100 und RTX 8000. LIMITATION gefunden: K%64-
   Anforderung schliesst den FN-TP4-w2-Shard (K=160) aus; TP1/TP2 ok —
   steht jetzt ehrlich im Entwurf.
2. **1Cat-Referenzzahlen gesichtet:** PR #361 (FN-Decode 82 tok/s TP4)
   ist End-to-End, kein Kernel-A/B — deren per-Layer-MoE-Kosten kennen
   nur sie; der Kommentar liefert unsere ms/Layer, die sie selbst
   einordnen koennen. Echter A/B gegen deren compact-grouped braeuchte
   deren 1.5.0-Build (Installation, Peuqui-Freigabe) ODER einen zweiten
   echten NVFP4-MoE-Checkpoint (lokal existiert nur DSv4; Qwen3.5-122B-
   NVFP4-Cache ist leer, ~70 GB Download, Peuqui-Entscheid).

Endzustand 18:15: GPUs frei, kein Server. Uncommitted seit 10f3d27:
Handover-Nachtraege, benchmarks/nsys-dsv4-het-moeqpn-2026-09-03.txt,
Kommentar-Entwurf, scripts/nvfp4_skinny_moe_qpn_synth_fn.py; die grossen
nsys-Binaerdateien (nsys-dsv4-het-moeqpn.*) bleiben unversioniert.

### 2026-09-03 19:00 — Echter A/B gegen 1Cats compact-grouped-Route: differenziertes Ergebnis

Freigabe Peuqui: eigene venv `.venv-sm70-150` (Release-Wheel 1cat_vllm-
1.5.0, torch 2.10.0+cu128, Wheel unter .wheels/) + Qwen3.6-35B-A3B-NVFP4-
Shard 1 (~7,4 GB, /home/mp/models/Qwen3.6-35B-A3B-NVFP4/). Harness
tools/moe_ab_1cat.py: ruft 1Cats `nvfp4_moe_dense_stage_sm70_out` mit
compact 1-Zeile/Gruppe-Offsets (exakt deren Decode-Staging, prepare via
`nvfp4_sm70_prepare`+`awq_moe_build_strided_ptrs`) gegen unser moe_qpn;
identisches vorbereitetes Routing, Korrektheit beider gegen fp32-Dequant-
Referenz. Ergebnisse: benchmarks/moe-ab-1cat-2026-09-03.txt.

**Kernbefund (V100, echte Bytes):** Qwen3.6 (Mini-Experten 1,25 MiB):
1Cat gewinnt T=1-2 (wir 0,67-0,76x), Paritaet ab T=4. DSv4 (12-MiB-
Experten, Sharing): WIR gewinnen durchgehend — T=1 1,18x, T=6 1,34x
(Verify-Punkt), T=8 1,45x. Mechanik: deren 1-Zeile/Gruppe liest geteilte
Experten je Slot erneut; moe_qpn buendelt bis 8 Zeilen je Lesung.
**RTX 8000: deren TurboMind-Op bricht ab** („No feasible kernel found …
sm75_f16_e2m1k16…") — deren Route ist real sm70-only. Numerik beider
Routen in der fp16-Rundungsklasse (1e-4/2-4e-4).

Harness-Falle (gefixt): moe_qpn adressiert slot-major ueber die ORIGINAL-
Slot-Nummer; wer die Zwischenaktivierung in 1Cat-Sortierordnung uebergibt,
misst Muell (w2-diff 1,8e-1) — mid vor dem Aufruf zurueckstreuen.

Der #441-Entwurf (comment-441-grouped-moe-offer.md) traegt jetzt die
A/B-Zahlen inkl. des ehrlichen 1Cat-Siegs im Mini-Shape-T=1-Regime und
den sm75-Befund; Framing „complementary lanes". WARTET AUF FREIGABE.
Endzustand 19:00: GPUs frei, kein Server.

### 2026-09-03 19:20 — KORREKTUR (Peuqui-Einwand): TP ueber den Interconnect geht doch — 2x2-Gitter ist ein offener Latenz-Hebel

Meine Aussage „TP kann unser Interconnect nicht" war zu pauschal. Fakten
(reference_gpu_setup_5_cards): nur 1 von 5 Karten haengt am USB4-Tunnel
(~5 % langsamer, kein Flaschenhals); das echte Limit ist fehlendes
P2P-DMA zwischen Root-Ports (GEM10) → NCCL_P2P_DISABLE=1, Host-Bounce.
Decode-AllReduces sind aber winzig (~10-50 KB): TP2 laeuft produktiv
(-vllm-speed, 2xV100, 60,5 tok/s k=5). Offen aus dem Merge-Projekt bleibt
das 2x2-Gitter (TP2 je Kartenklasse + PP darueber): fuer DSv4 hiesse das
RTX-TP2 (~23 L) → V100-TP2 (~15 L) → V100 solo (5 L + Drafter) = 3
serielle Stationen statt 5, TP2-Stationen mit doppelter Gewichtsband-
breite; Groessenordnung 10-25 % Schrittlatenz, gegenzurechnen ~86
Host-Bounce-AllReduces/Step. Mehrtaegiges Experiment (Blocker von damals:
heterogener Graph-Capture, K>0-Spec im Upstream-Builder — SSOT
MERGE-PROJECT-HANDOVER.md); als Kandidat NACH der 1Cat-Entscheidung.

### 2026-09-03 20:00 — Compact-Grid-Fix: moe_qpn schlaegt 1Cats Route jetzt in JEDEM Regime

Peuqui-Frage „koennen wir bei T=1 aufholen?" beantwortet mit Umbau:
moe_qpn-Grid laeuft jetzt ueber slot-count-many kompakte Gruppen
(gids/goff statt offsets[E+1]; statische Obergrenze S = T*topk, bleibt
graph-tauglich; Padding-Gruppen exiten). Die leeren Experten-Bloecke
waren bei Mini-Experten der GESAMTE Rueckstand: Qwen3.6 T=1 von 0,67x
auf **1,32x** gedreht; alle Regime jetzt vorn (Qwen3.6 1,05-1,32x,
DSv4 1,22-1,38x — benchmarks/moe-ab-1cat-2026-09-03.txt NACHTRAG).
FN-Geometrie: T=1 jetzt 7-9x vs moe_simt (E=512-Grid entfaellt).
Route/Kernel/alle 4 Skripte auf das compact-Format umgestellt
(NO LEGACY: altes offsets[E+1]-Format ersatzlos raus; moe_simt behaelt
seins als Checkpoint-Layout-Referenzanker — bewusst NICHT geloescht,
er ist der layout-unabhaengige Korrektheitsanker der Tests).

Serving Boot 66: 0 ERRORs, Kohaerenz 8/8 (5/8 bitidentisch = Quote),
**113,8/114,5 ms/Step** (Boot 64: 112,8/114,5 — fuer DSv4 neutral wie
prognostiziert, ~2-3 % gehen in der Streuung unter); essay 25,1 /
code 28,6 tok/s. Entwurf comment-441-grouped-moe-offer.md auf die
neuen Zahlen umgeschrieben (inkl. ehrlicher Notiz, dass die fruehere
Grid-Version die T=1-Lane verlor und der Fix sie drehte).
Endzustand 20:00: GPUs frei, kein Server.

### 2026-09-03 20:15 — Commits a1ac62c gepusht (Remote `fork`!), #441-Angebot GEPOSTET

Push-Falle: `origin` zeigt auf dnv2003/v100-skinny (403) — Pushes gehen
an das Remote **fork** (git@github.com:Peuqui/v100-skinny.git). Commits
10f3d27 + a1ac62c auf pp-mtp-merge gepusht. Vor dem Posten letzter
Gegenlese-Pass: drei veraltete Zahlen im Entwurf korrigiert (Full-Layer
1,35-1,37x/1,24-1,29x nach compact-Umbau; FN-Synth 7-9x bei T=1;
DSv4-A/B-Spanne 1,03-1,38x inkl. T=2). Kommentar gepostet:
https://github.com/1CatAI/1Cat-vLLM/issues/441#issuecomment-5527353132
— auf Maintainer-Antwort achten (Port-PR-Angebot steht).

**NÄCHSTE STATION (Peuqui-Beschluss):** 2x2-Gitter fuer DSv4 (TP2 je
Kartenklasse + PP darueber, 3 serielle Stationen statt 5; Ausgangspunkte
MERGE-PROJECT-HANDOVER.md — Blocker von damals: heterogener
Graph-Capture, K>0-Spec im Upstream-Builder). Davor ggf. Doku-Nachcommit.

### 2026-09-03 20:45 — Schrittfixkosten-Gap-Analyse: kein Einzelblocker, sondern 3500 Mikro-Luecken

Aus der Boot-65-SQLite (Union aller Nicht-NCCL-Kernel ueber alle 5 GPUs,
60-s-Fenster, 403 Steps, unter Profiler): Compute 101,8 ms/Step;
Intra-Lauf-Idle (alle GPUs kernel-still) 20,3 ms/Step verteilt auf
**~3.482 Luecken pro Step** — 5,3 ms in <10-µs-, 8,0 ms in 10-50-µs-,
4,8 ms in 50-200-µs-, 2,2 ms in 200µs-1ms-Luecken. (Inter-Request-Pausen
sauber ausgeklammert.) Es gibt KEINEN einzelnen Fixkosten-Blocker: der
Rest ist Launch-/Segment-Konfetti (breakable-capture-Segmentgrenzen,
PP-Naht-Metadaten via gloo, Drafter-Iterationen). Der einzige Hebel
waere Segment-Konsolidierung (weniger Graph-Breaks) — Fork-Infrastruktur,
Ertrag real geschaetzt einstellige ms, Aufwand hoch. FAZIT: das
Kernel-/Schrittlatenz-Kapitel fuer DSv4-PP5 ist AUSOPTIMIERT; weitere
Tempoarbeit lohnt erst wieder bei Flash-Next (2x2-Gitter steht bereit,
MERGE-Handover Session 4) bzw. nach 1Cat-Antwort (Port-PR).

### 2026-09-03 21:00 — Branch-/Veroeffentlichungspolitik (Peuqui-Entscheid)

Arbeitsmodus ab jetzt: **Branch `work`** (taeglicher Stand, Push auf fork
erlaubt — eigener Fork, Push-Freigabe pauschal fuer work). `pp-mtp-merge`
ist EINGEFROREN auf dem letzten kuratierten Stand (3552c6e) und wird nur
noch auf Peuqui-Freigabe aktualisiert; **PR dnv2003#7 ist auf DRAFT**
(wuchs vorher mit jedem Push live mit — Befund 20:30).
Veroeffentlichung an 1Cat als PIPELINE fokussierter Pakete (Peuqui-GO):
(1) Bugfix-Serie "Qwen4Exp unter PP" (hyper_connection_mixer-Loader u.a.),
(2) moe_qpn-Port (laeuft, #441-Antwort abwarten), (3) PP-Enablement/PLE-
Kaskade nach 1.5.0-Rebase, (4) sm75/Device-0-Serie. Jedes Paket einzeln
wasserdicht (Tests, A/B, Kohaerenz); kein Big-Bang-PR.

### 2026-09-03 21:30 — Paket-1-Auftakt GEPOSTET: 1Cat-Issue #479 (Qwen4Exp unter PP)

https://github.com/1CatAI/1Cat-vLLM/issues/479 — drei Blocker (2 bewusste
PLE-Gates + hyper_connection_mixer-Loader-Bug, alle gegen origin/main
verifiziert), Beleg ~52 tok/s PP-Betrieb auf unserer Kiste, Angebot einer
PR-Serie (Loader-Fix + Test zuerst, dann input-id-Transport, dann
PLE-Kaskade). Beobachten: Antworten auf #479 UND #441 (MoE-Angebot,
noch unbeantwortet) beim taeglichen Upstream-Check.

### 2026-09-03 21:45 — WICHTIG: .venv-sm70-Loeschung hatte Nebenwirkung (CUDA-Deb-Symlinks)

Die FlashInfer-JIT-Fehler der Kontrollboots 1-4 („unsupported GNU
version", /usr/include-12.0-Header) hatten eine selbstverschuldete
Wurzel: `.cuda-nvcc-deb/cuda-12.8` bezog 83 Runtime-Header/Libs als
SYMLINKS aus der pip-nvidia-Struktur der heute geloeschten `.venv-sm70`
(1.2.2). Nach der Loeschung fiel gcc auf die System-CUDA-12.0-Header
zurueck, deren host_config gcc>12 ablehnt (System-gcc ist inzwischen
13.3 — frueher lief der 12.0-Fallback zufaellig durch). FIX: alle 83
Symlinks auf `.venv-sm70-130` umgebogen (gleiche pip-Pakete);
nvcc-Standalone-Test gruen. **MERKE: .venv-sm70-130 darf NICHT
geloescht werden, solange .cuda-nvcc-deb dranhaengt** — beim
1.5.0-Umzug die Symlinks auf -150 mitziehen oder das Deb um echte
Kopien ergaenzen.

### 2026-09-03 22:15 — 1.3.0-Kontrollboot GRUEN: Baseline bestaetigt, Zerfall-Sonde differenziert nicht

Kontrollboot 5 (RadixArk-FN roh, k=0, TP2/PP2, 24/24, PLE_HOST_GIB=6,
CUDA_HOME+PATH aufs 12.8-Deb) nach Symlink-Reparatur: startup, Pruefstand
(scratchpad/smoke_check_150.py, kopiert nach benchmarks/fn-130-control-
2026-09-03.txt): **Tempo 32,3 tok/s** — die 28.08.-Baseline (32,2) gilt
EXAKT weiter; die September-Patches (moe_qpn etc.) beschleunigen
DSv4, aber nicht den FN-k=0-Pfad (FN-MoE laeuft weiter MARLIN-Route).
Kohaerenz 3/3 (Reasoning-Modell: Antworten hinter <think>-Block,
Pruefstand braucht max_tokens >=220). Sprachzerfall-NAEHERUNG (7,3k-
Token-Kunstkontext, 3 Turns DE + EN): NULL Zeichenlecks — die Sonde
reproduziert den 30.08.-Zerfall NICHT (deckt sich mit der Doku-Warnung:
nur die echte AIfred-Persona zeigt ihn). Konsequenz: Der Checkpoint-
Qualitaetstest fuer Kandidaten muss spaeter UEBER AIFRED laufen
(inject-API + echte Persona), die Sonde taugt nur als Grobfilter.
Pruefstand-Fallen: Token-Schaetzung Deutsch ~85 tok/Filler (Faktor 2
ueber Gefuehl), 16k-Fenster in Turn 3 knapp.
Endzustand 22:15: GPUs frei, kein Server. NAECHSTER BLOCK: Phase 2 —
die 6 ernsten Konflikt-Patches (mhc_tilelang, modelopt,
gpu_model_runner, worker_mamba_utils, worker_utils, utils) auf 1.5.0.

### 2026-09-03 23:00 — Phase 2 KOMPLETT: alle 22 Rebase-Konflikte aufgeloest

fork_patches_150/ traegt jetzt 71 Patches + STATUS.txt (alle py_compile-
gruen); von 83 Deploy-Zielen sind NEUN Patches upstream OBSOLET geworden:
mhc_triton (unser eigener Backport), worker_utils (Mixed-Page-Zeroing
generischer geloest), worker_mamba_utils (natives Qwen4Exp), model_runner,
interfaces, vocab_parallel_embedding (FP8-int8-AllReduce uebernommen!),
short_conv_attn, kv_cache_coordinator, compilation. Der komplette
CSA-Linear-Cache-Komplex (unser DSv4-Layout!) ist upstream GELANDET —
teils wortgleiche Kommentare; unsere PR-#7-/Issue-Beitraege sind sichtbar
eingeflossen. Kombiniert statt ersetzt: modelopt (QPN8+TurboMind
koexistieren), utils (Stacked-Shard: deren Typform + unsere
renaming/shard-Durchreichung), gpu_model_runner (PLE-Partition-Pruefung
statt PP-Verbot = Issue-#479-Kern, PP-Spec-Gate is_last_rank, E5-Hooks;
6 Bloecke obsolet: deren _sync_mamba_accepted_token_state,
IntermediateTensors-Unwrap, compressed_kernel_block_size).
NAECHSTER BLOCK: bootstrap-Skript auf -150/fork_patches_150 portieren
(Deploy-Liste um die 9 Obsoleten kuerzen, qwen4_exp-Eigenbauten raus),
dann Boot-Gates: erst 27B-Regression (2x2, Gates gruen?), dann Flash-Next
(PLE-Partition!), dann DSv4-PP5. Kohaerenz + Bench je Modell gegen die
frischen Baselines (FN k=0: 32,2; DSv4: 113 ms/Step).

### 2026-09-03 23:15 — KORREKTUR zur CSA-Herkunft (Peuqui-Nachfrage, git-verifiziert)

Die 23:00-Formulierung „unser DSv4-Layout ist upstream gelandet" war
FALSCH HERUM: _CSALinearCacheTuple, orig_to_new_stacked und der
FP8-int8-AllReduce kamen alle mit 1Cat-PR #403 (Merge 187b932,
29.08. 16:45) in deren main — UNSERE Fassungen waren Backports/
Adaptionen von deren HEAD in unsere 1.3.0-Basis (eigener Patch-Kommentar
„Upstream writes this layout..."). Beim Rebase kehren die Originale
zurueck; Uebernahme UNSERER Arbeit ist fuer diese drei nicht belegbar
(deren Branch startete Stunden nach unseren 28.08.-Posts — Konvergenz
plausibel, Kausalitaet offen). Echt von uns bleiben: PP-Transport,
PLE-Kaskade, Device-0-Fixklasse, QPN-Kernelfamilie, moe_qpn (#441/#479).

### 2026-09-04 00:15 — 27B-Gate-Kampagne: Basis GRUEN, K>0-Spec degradiert (Verdaechtiger identifiziert)

Sieben Gate-Boots auf 1.5.0+Patches, fehlergetrieben gefixt: (1) FLA-Baum
fehlte im 150er-Deploy (Verzeichnis-Deploys ergaenzt, qwen4_exp_models
bewusst nicht), (2) qwen3_next-Merge-Chimaere (mein K1-Fehler: theirs-Kopf
in UNSERE _project_qkv_gate-Splitmethode gepflanzt) — Datei ist jetzt
OBSOLET/pures 1.5.0, (3) FA2-Trio zurueckgestellt (sm75-Enablement haengt
am Drop-in-Build, der nur in -130 installiert ist; RTX faellt auf
TRITON_ATTN), (4) SECHS neue Device-0-(7,0)-Gates in 1.5.0-vllm_config
auf _any_visible_device_has_capability umgestellt (Fixklasse #412 —
Upstream produziert sie nach, homogene Referenz).

STAND: elf Boot-Gates PASS, **k=0 via Chat VOLL KOHAERENT** (sauberer
Denkblock, korrekte Antworten), Tempo k=7 math 75,9 / code 53,7
(Referenz 85,0/56,3 — Luecke plausibel durchs zurueckgestellte FA2-Trio).
**OFFEN: K=7-Spec degradiert das Denken** (Repro: "capital of France"
→ Denkblock 165 Zeichen, "4. 4."-Duplikate, terminiert ohne Antwort;
k=0 identisch sauber). PRUEFSTAND-LEKTION: Kohaerenz-Kurzproben bei
Instruct-Modellen NUR ueber /v1/chat (Raw-Completions erzeugte
Phantom-Salat auf BEIDEN venvs) und reasoning_content mitlesen.
NAECHSTER SCHRITT: K2-Rollback-Experiment in gpu_model_runner (unser
Async-Align-Block statt theirs' _sync_mamba_accepted_token_state) —
1 Datei, 1 Boot; danach ggf. Verifier-Pfad (XQA/compile 1.5.0).

PAKET 1a VORBEREITET: Branch qwen4exp-pp-mixer-loader-skip im 1Cat-Klon
(Fix in nvidia+amd model.py, neuer Test in test_weight_loading.py —
Lauf blockiert auf pytest-Freigabe fuer -150). Klon-Notiz: Sparse-
Checkout (nur FA-Baeume materialisiert; qwen4_exp+tests hinzugefuegt),
Stash "fa-v100-64x80-arbeit-2026-09-03" gesichert.

### 2026-09-04 01:30 — K>1-Spec-Degeneration: Bisektions-Matrix komplett, Suchraum final eingegrenzt

REPRO (deterministisch, greedy): 27B, 1.5.0+Patches, TP2/PP2, K>=2:
"capital of France"-Chatprompt → Denkblock degeneriert bit-identisch
('**Identify the user is "France?"', "4. 4."-Duplikate, terminiert ohne
content); 1591/Mercury-Prompts OK; K=0 und K=1 VOLL sauber (3/3).
-130-Referenz: gleicher Prompt K=7 sauber → echter 150er-Regress.

BISEKTIONS-MATRIX (je 1 Boot, K=7-Chatprobe):
- K2-Rollback (unser 1.3.0-Async-Align statt deren Sync): SCHLECHTER
  (1/3, Token-Duplikate "37 * 37*43") → deren _sync ist KORREKT,
  Producer-Kontrakt hat sich mitgeaendert. Zurueckgenommen.
- gdn_attn Stock: unveraendert 2/3 → unschuldig (Patch vorerst Stock,
  Optimierung -1,4ms spaeter neu aufsetzen).
- llm_base_proposer Stock: unveraendert → unschuldig (Stock belassen).
- gpu_model_runner Stock: BOOTET NICHT ("no attribute 'drafter'") →
  **1.5.0 hat weiterhin KEINEN PP-Spec-Support; unsere Runner-Hunks
  sind zwingend** (Beleg fuer Paket 3!).
- Scheduling-Trio (sched_scheduler/scheduler/multiproc_executor) Stock:
  unveraendert → unschuldig (wieder gepatcht).
- Verifier-Backend SPEC_ATTN=TRITON_ATTN statt XQA: unveraendert →
  Wheel-XQA-Op unschuldig.

FAZIT: Text bit-identisch ueber ALLE getauschten Schichten → Fehler
sitzt in den SPEC-VERIFY-HUNKS des gemergten gpu_model_runner
(Patch-6/7-Region: Draft-Broadcast/Accept-Ableitung/Bonus-Token gegen
die umgebaute 1.5.0-Verify-Logik) oder qwen3_5_mtp-Fluss. NAECHSTER
SCHRITT (statisch, kein Boot): unsere Spec-Hunks in fork_patches_150/
gpu_model_runner.py Zeile fuer Zeile gegen (a) 1.3.0-Fassung und
(b) 1.5.0-Umgebung diffen — Verdacht doppelte/versetzte Anwendung der
akzeptierten Tokens im Bonus-Pfad. Diag-Hebel danach:
VLLM_DSPARK_DIAG-artige Drafts-Dumps am 27B.

Paket-1a-Stand: Branch qwen4exp-pp-mixer-loader-skip, Commit fb44f10
(nvidia+amd+Test), isolierter Test PASS gegen Wheel+Fix (pytest+tblib
in -150 installiert, Peuqui-Pauschalfreigabe); PR wartet auf Peuquis
SICHTUNG vor dem Online-Stellen (explizite Ansage 2026-09-04).
Endzustand 01:30: GPUs frei, kein Server.

### 2026-09-04 01:50 — PAKET 1a GESTELLT: PR 1CatAI/1Cat-vLLM#485

https://github.com/1CatAI/1Cat-vLLM/pull/485 (Peuqui-Go nach Sichtung
von Diff+Text): [Bugfix][SM70] Skip final mixer weights on non-last PP
ranks — Branch Peuqui:qwen4exp-pp-mixer-loader-skip (fb44f10), 48
Zeilen (nvidia+amd symmetrisch + Test), TP-unabhaengig (Bedingung nur
PP-Rank). Referenziert #479. Beobachten: #441, #479, #485 beim
taeglichen Upstream-Check. Unabhaengig von der offenen K>=2-Regression
(eigener Codepfad, Stock erreicht Spec nie).
