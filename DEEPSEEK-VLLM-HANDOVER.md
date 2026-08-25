# Übergabe: DeepSeek-V4-Flash unter vLLM + Entscheidung Block-FP8-Kernel

Stand: 2026-08-25 · Vorgeschichte: [MERGE-PROJECT-HANDOVER.md](MERGE-PROJECT-HANDOVER.md)
(PP×TP-Merge, heterogenes 2×2) und [../vllm-bench/RESULTS.md](../vllm-bench/RESULTS.md)
(alle Messungen mit Methodik).

## Auftrag

Peuqui will **DeepSeek-V4-Flash wenigstens einmal in der schnellsten verfügbaren
Version unter vLLM testen**. Wenn dafür ein blockskalierter FP8-Kernel nötig ist,
darf der gebaut werden — aber als eigenes Projekt in eigener Sitzung, nicht
nebenbei.

## Ausgangslage in einem Absatz

Das heterogene 2×2-Gitter läuft: TP=2 auf den zwei RTX 8000 als Stufe 0, TP=2 auf
zwei V100 als Stufe 1, PP über die Generationsgrenze, MTP k=7, CUDA-Graphen an —
85,0 tok/s (math) bei allen elf Boot-Gates grün. Die Wurzel der letzten Blockade
war eine device-0-gekoppelte Konfigurationsentscheidung (SM70-Baseline in
`config/vllm.py`), gefixt und committet als 9e3b658 auf Branch `pp-mtp-merge`.

## Messlage (Qwen3.8-27B, 2× V100 TP=2, identische Methodik)

| Engine / Format | Bits | math | code | Bemerkung |
|---|---|---:|---:|---|
| vLLM NVFP4 k=7 | ~4,5 | **88,2** | **62,2** | schnellstes im Haus |
| vLLM FP8 k=7 | 8 | 44,8 | 37,2 | über Marlin-Weight-only |
| llama.cpp Q8 n=3 | ~8,3 | 41,1 | 31,5 | n=7 bringt nichts (40,8) |
| vLLM AWQ-INT4 k=7 | ~4,25 | 37,0 | 30,0 | auf Volta ohne INT4-Tensorcores |

Dieselbe Matrix auf 2× RTX 8000 (TP=2, Fork): NVFP4 43,2 (K=0) / **79,1** (k=7),
AWQ 12,2 / 38,5. **AWQ hat auf dieser Maschine keine Nische** — im Fork auf beiden
Generationen gleich langsam (Fork bedient den Pfad nicht), und selbst upstream-vLLM
mit 67,6 auf den RTX bleibt unter den 79,1, die NVFP4 dort im Fork liefert. Ohne MTP
ist Turing vorn (43,2 vs 40,5), mit MTP Volta (88,2 vs 79,1) — die V100-Stufe bekommt
den XQA-Verifier, die RTX-Stufe verifiziert mit TRITON_ATTN. Verfügbarkeit spricht
ebenfalls für NVFP4: bei fünf geprüften Großmodellen durchweg mehr NVFP4- als
AWQ-Checkpoints (z. B. DeepSeek-V4-Flash 10 gegen 1).

Qualitätsabstand gegen FP8 (Teacher Forcing, KL in nats): NVFP4 0,043 / 0,003 /
0,018 (Prosa/Code/Fakten), AWQ 0,030 / 0,002 / 0,014. **4 Bit ist messbar, aber
klein von 8 Bit entfernt; bei Code nicht unterscheidbar.** Format-Empfehlung für
diese Kiste: **NVFP4** — AWQ ist minimal genauer, aber 2,4× langsamer, sobald
V100s beteiligt sind.

## Bestand

vLLM (HF-Cache): `Qwen/Qwen3.8-27B-FP8` (28,8 GB), `RadixArk/Qwen3.8-27B-NVFP4`
(20,4 GB), `cyankiwi/Qwen3.8-27B-AWQ-INT4` (19,6 GB). **Nichts über 27B.**
llama.cpp: DeepSeek-V4-Flash-284B-A13B UD-Q4_K_XL (145 GB) + dspark-Draft Q8_0
(11 GB), Qwen3.8-27B-Q8_K_XL (30 GB), Qwen3VL-4B, bge-m3. Platte: 324 GB frei.

## Der DeepSeek-Fall: was zu klären ist, in dieser Reihenfolge

**1. Die Latte messen — kostenlos, kein Download.** Was liefert llama.cpp heute mit
DeepSeek? Die llama-swap-Konfiguration steht (`DeepSeek-V4-Flash-0731-UD-Q4_K_XL`,
5 GPUs Layer-Split 11,12,8,7,5, `-c 193536`, `--spec-type draft-dspark
--spec-draft-n-max 5`, Draft auf CUDA4). Mit `bench.py` messen, Zellen math+code,
Seeds 1001/2002. **Ohne diese Zahl ist jede vLLM-Messung wieder wertlos** — genau
der Fehler, den wir heute zweimal gemacht haben.

**2. Zwei ungeprüfte Voraussetzungen klären — auch ohne Download.**
- *Läuft PP=5 im Fork?* Bewiesen ist PP=2. Mit der 27B über alle fünf Karten
  testen (TP=1 PP=5, `VLLM_PP_LAYER_PARTITION` für die ungleiche Kartengröße).
  Nebeneffekt: liefert die Topologie-Kosten von PP=5 gegen das 2×2.
- *Trägt der Fork die DeepSeek-V4-Architektur?* Steht seit der ersten Übergabe als
  offener Punkt. Lässt sich an der Modell-Registry und den Ops prüfen, bevor
  164 GB heruntergeladen werden.

**3. Kapazität — es passt, aber knapp.** Frühere Aussage „passt nicht" war falsch.
192 GB gesamt, Checkpoint 164 GB (RedHatAI) ⇒ 28 GB für KV, Aktivierungen und
Puffer. Empirischer Anker: llama.cpp trägt heute 145 GB Gewichte **plus 193.536
Token Kontext** in denselben 192 GB, also rund 260 Byte/Token. 28 GB minus ~10 GB
Overhead lassen also grob **70.000 Token Kontext**. Bedingungen: `gpu_memory_
utilization` von 0,88 auf ~0,95 anheben (der Wert stammt aus einem 16-GB-Kontext),
GPU 3 muss das Vigilantia-VLM räumen, und die PP-Partition muss so gewählt sein,
dass keine Stufe überläuft (Kapazitäten 48/48/32/32/32).

**4. Checkpoint-Wahl — und hier ist der Haken.**

| Variante | Größe | MTP | Passung |
|---|---:|---|---|
| RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8 | 164 GB | nein | knapp, ~70k Kontext |
| nvidia/DeepSeek-V4-Flash-NVFP4 | 168 GB | nein | sehr knapp |
| canada-quant/…-NVFP4-FP8-MTP | 184 GB | **ja** | **passt nicht** (8 GB Rest) |

Die *schnellste* Variante wäre die mit MTP-Block — Spekulation brachte auf der 27B
Faktor 2,2 bis 3,4. Sie ist mit 184 GB aber nicht unterzubringen. Ohne MTP fällt
vLLM auf die K=0-Klasse zurück, während llama.cpp seinen dspark-Draft behält. Das
ist der zentrale Zielkonflikt dieses Tests.

## Warum der Kernel ins Spiel kommt

Der NVFP4-Checkpoint ist **nicht durchgängig 4 Bit**. Seine Config sagt
`quant_algo: MIXED_PRECISION`, `producer: modelopt/dsv4-nvfp4-experts`,
`quantized_layers: layers.N.ffn.experts`, `ignore: ['*.attn.*',
'*.ffn.shared_experts.*', 'head']` über einer Basis `quant_method: fp8` mit
`weight_block_size: [128,128]`. Nur die gerouteten Experten sind NVFP4; Attention,
Shared Experts und Head bleiben **blockweises FP8**.

Auf dieser Hardware landet blockweises FP8 in vLLMs generischer `Fp8Config` und
damit auf `MarlinFP8ScaledMMLinearKernel` (Weight-only) — gemessen 14,8 tok/s auf
der 27B. Bei einem A13B-MoE feuern Attention und Shared Experts bei **jedem**
Token, während von den Experten nur eine Handvoll aktiv ist. Der ständig laufende
Teil liefe also über den langsamen Pfad.

### Was der Kernel wäre

Der schnelle Pfad des Forks (QPN8, `mma.sync.m8n8k4`) erwartet **einen Skalar pro
Matrix** (`weight_scale`, Form `[]`). Blockskalierte Checkpoints tragen ein
**Raster** (`weight_scale_inv`, z. B. Form `[136, 40]` = 128×128-Blöcke). Der
Kernel müsste im innersten Schleifenkörper der Kachelung ein Skalenraster
mitführen. Das ist Kernel-Arbeit im Herzstück des Forks, **keine
Dispatch-Änderung** — ein Umbiegen der Config-Klasse würde nur den falschen Kernel
mit den falschen Skalen rechnen lassen.

### Erfolgskriterium, damit das Projekt nicht ins Leere läuft

Schätzung aus den Messungen: FP8 über Marlin 14,8 tok/s bei K=0; ein QPN8-artiger
Blockkernel sollte auf 25–30 kommen (Bandbreitenverhältnis zu NVFP4s 40,5), mit dem
gemessenen MTP-Faktor ~3,0 also **60–75 tok/s**. Der Kernel lohnt sich, wenn
mindestens eines zutrifft:

- DeepSeek unter vLLM schlägt damit die llama.cpp-Latte aus Schritt 1, oder
- ein Zielmodell existiert nur als blockskalierter Checkpoint.

**Er lohnt sich NICHT als Qualitätsprojekt.** Die Messung von heute zeigt, dass
4 Bit nur 0,02–0,04 nats von 8 Bit entfernt liegt — dafür die halbe Decode-Rate zu
zahlen wäre ein schlechtes Geschäft.

## Umgebung und Repro

```bash
cd /home/mp/Projekte/v100-skinny            # Branch pp-mtp-merge
SNAP=$(ls -d /home/mp/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4/snapshots/*/)
TP=2 PP=2 PORT=8020 K=7 ATTN_BACKEND=AUTO \
VLLM_QWEN35_MTP_SHARE_IO_WEIGHTS=0 \
DISABLE_CAR=1 NCCL_P2P_DISABLE=1 ASYNC_SCHED=1 \
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,2,1,4 \
CUDA_HOME=$PWD/.cuda-nvcc-deb/usr/local/cuda-12.8 \
LOG=$PWD/serve-2x2.log bash scripts/serve-qwen38-mini.sh "$SNAP"
```

Pflicht auf dieser Kiste: `NCCL_P2P_DISABLE=1` + `DISABLE_CAR=1` (Ryzen-Root-Port-P2P
defekt), `CUDA_DEVICE_ORDER=PCI_BUS_ID` mit numerischen Indizes, `CUDA_HOME` auf die
entpackten nvcc-12.8-debs. **Kein `--enforce-eager`** — das kostet die V100-Stufe den
XQA-Pfad des MTP-Verifiers, der bei q>1 als einziger korrekt rechnet.

Schalter im Serve-Script: `SPEC_ATTN` (Drafter-Backend separat, der Drafter lebt auf
der letzten Stufe), `CUDAGRAPH_MODE` (trennt compile von Graph-Aufzeichnung),
`GATE_SOFT=1` (durchgefallener Boot bleibt zur Diagnose stehen — Messwerte daraus
sind unzitierbar).

Messwerkzeuge in `/home/mp/Projekte/vllm-bench`: `bench.py` (Durchsatz),
`quality_kl.py` (Qualitätsabstand per Teacher Forcing), `run-format-matrix.sh` und
`run-quality-matrix.sh` (serielle Treiber mit Boot/Messung/Teardown).

## Fallstricke, heute selbst hineingetreten

- **`pkill -f` mit einem Muster, das auf die eigene Kommandozeile passt**, killt die
  eigene Shell. Zweimal passiert. Nur über explizite PIDs aufräumen.
- **Teardown über die PID-Datei ist unzuverlässig:** bei `GATE_SOFT` beendet sich der
  Launcher selbst, die Worker leben weiter und blockieren die GPUs des nächsten Laufs.
  Die Treiber räumen deshalb über `nvidia-smi --query-compute-apps` auf und verschonen
  den produktiven `llama-server` explizit.
- **Vergleiche über Engine- *und* Formatgrenzen sind wertlos**, solange nicht beide
  Seiten auf derselben Hardware mit ihrer besten Konfiguration gemessen wurden.
- **Boot-Gates, die an Startvariablen hängen statt am beobachteten Verhalten**, gehen
  still durch. Das XQA-Gate war so gebaut und hat einen falsch rechnenden Server
  passieren lassen.
