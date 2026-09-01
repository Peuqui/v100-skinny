# Übergabe: sm75-Koexistenz im v100-skinny-Fork

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
