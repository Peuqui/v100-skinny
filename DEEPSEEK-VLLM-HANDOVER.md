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

### Stand 2026-08-25 nachmittags: Kernel GEBAUT und validiert

`skinny_fp8_qpn8` und `…_mt2` tragen jetzt einen `BLOCKED`-Template-Zweig:
die Skala wandert vom Epilog an den Decode-Punkt (ein warp-uniformer `__ldg`
pro 16er-Gruppe + 8 `HMUL2`, das NVFP4-Muster mit gröberer Granularität).
Zwei Geometrie-Glücksfälle machen das billig: ein N=32-Tile liegt immer in
EINEM BN-Block (Skala pro Tile×K-Block ist ein Skalar, kein Lane-Raster)
und eine 16er-Gruppe immer in EINEM BK-Block (ein Lookup pro Gruppe).
Neue Einstiegspunkte `gemm_qpn8_blk` / `gemm_qpn8_blk_mt2` nehmen das
Checkpoint-Raster `[ceil(N/bn), ceil(K/bk)]` fp32 direkt; gleiche
Dispatch-Keys (splitk×10+nacc, +2 = Fast-Decoder) wie der Per-Tile-Pfad.

Validierung (`scripts/qpn8_blk_test.py`, V100): 23 Fälle OK bei rel_err
~3,9e-4 (Per-Tile-Referenz: 2,75e-4; leicht höher, weil die Gewichte jetzt
auf wahrer Magnitude in fp16 gerundet werden — Toleranz 3e-3), inkl.
M=1..16, Partial-K-Block (K=5184), 64×64-Raster, beide Decoder. Regression
`qpn8_test.py` unverändert grün. Kosten des Rasters: **+2,1 % (M=8) bis
+4,1 % (M=1)** gegen den Per-Tile-Kernel (671 vs. 685 GB/s bei M=8) — die
25–30-tok/s-Schätzung unten bleibt damit stehen.

**Verdrahtung (gleicher Tag, nachmittags): FERTIG und End-to-End belegt.**
Neue Kernel-Klasse `QPN8Fp8BlockScaledMMLinearKernel`
(`fork_patches/qpn8_blk.py` → `kernels/linear/scaled_mm/qpn8_blk.py`) an
der Spitze von `_POSSIBLE_FP8_BLOCK_KERNELS` (`fork_patches/linear_init.py`);
gated sich per `is_supported` aufs LOKALE Worker-Device (Session-4-Lektion).
Weight-only: `apply_input_quant=False`, Aktivierungen bleiben fp16. Stash
prepackt beim Laden (Inversions-Assert pro Shape) und verwirft das
Original-Gewicht; M-Routing im Custom-Op `sm70_fp8.qpn8_blk_linear`
(≤8 nativ, ≤16 MT2, ≤96 chunked, darüber Prefill-Reconstruct).
`fork_patches/fp8.py`: `get_min_capability` öffnet das 70er-Gate auch für
diesen Pfad. Env-Schalter: `VLLM_SM70_QPN8_BLK` (default 1),
`VLLM_SM70_QPN8_BLK_CFG` (default 16,3), `…_CHUNK_MAX` (default 96).

**Wichtige Korrektur der Ausgangsannahme:** blockweises FP8 landet im Fork
NICHT auf Marlin, sondern auf einem eigenen **TurboMind-W8A16-Pfad** in
fp8.py (Vorrang via `VLLM_SM70_FP8_TURBOMIND`). Der QPN8-Pfad greift, wenn
`VLLM_SM70_FP8_TURBOMIND=0 VLLM_SM70_FP8_DEQUANT_FALLBACK=0` gesetzt sind —
Default-Vorrang zugunsten QPN8 umzudrehen ist eine offene Entscheidung.

**A/B (Qwen/Qwen3-0.6B-FP8, block [128,128], 1× V100, eager, greedy,
200 Token, textidentischer Output auf allen drei Pfaden):**
TurboMind 63,2 tok/s · **QPN8-blk 99,9 tok/s (+58 %)** · Dequant-fp16
105,5 tok/s (Obergrenze, doppelter Gewichts-Footprint). QPN8-blk liefert
95 % der fp16-Decke bei halbem Speicher — auf den kleinen 0.6B-GEMMs;
auf DeepSeek-Attention-Shapes sollte der Abstand größer ausfallen.

**Fallweises Routing (Peuquis Frage; „GB tun weh" ⇒ kein Dual-Format):**
Die Backend-Matrix (benchmarks/fp8_blk_backend_bench.py, echte
DeepSeek-Attention-Shapes wkv/wq_a/wq_b/wo_a/wo_b) zeigte einen sauberen
Kreuzungspunkt: QPN8 gewinnt M≤8 überall (1,08–1,79×), TurboMind ab M≈12
(Prefill bis 10×, weil Chunked/Reconstruct das Band nicht bedienten). Statt
TurboMind-Format zusätzlich resident zu halten (~4,6 GB, ≈18k Token
Kontextverlust), wurden ZWEI neue Kernel gebaut, die das VORHANDENE
gepackte QPN8-Layout lesen:
- `skinny_fp8_wmma_blk` (`gemm_qpn8_blk_wmma`): Tensor-Core-Tiles direkt
  aus Fragment-Order (col-Map/korder invertiert im Loader), Mittelband.
- `skinny_fp8_dequant_blk` (`qpn8_blk_dequant`): gepackt→fp16 [N,K] in
  einem Streaming-Pass (~5× schneller als der Torch-Indexing-Unpack),
  danach cuBLAS hgemm; transienter Workspace, nie persistiert.
Gemessene interne Frontier (Sweep in der Handover-Session): nativ ≤8,
MT2 ≤16, WMMA ≤256, Dequant+hgemm darüber (Kreuzung 256–512, Margen <20 %).
So verdrahtet in `sm70_fp8.qpn8_blk_linear` (`VLLM_SM70_QPN8_BLK_WMMA_MAX`).
Ergebnis gegen TurboMind: M≤8 gewinnt QPN8 (bis 1,79×), M=2048 Parität bis
+35 % (dequant+hgemm), nur das seltene Mittelband M≈12–512 bleibt
TurboMind-Territorium (typ. 1,2–2×). Bei MTP-k7-Betrieb dominiert M=8 die
Laufzeit ⇒ QPN8-Format als Einzellformat ist die richtige Wahl; der
Default-Vorrang in fp8.py steht aber weiterhin auf TurboMind
(Umschalten: `VLLM_SM70_FP8_TURBOMIND=0 VLLM_SM70_FP8_DEQUANT_FALLBACK=0`).
Default-Flip = offene Entscheidung.

**Bestand aktualisiert:** `Qwen/Qwen3-0.6B-FP8` (Testvehikel, ~1 GB) und
`RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8` (153 GB, 4 Shards) liegen im
HF-Cache. ACHTUNG: der RedHatAI-Export ist **compressed-tensors**
(mixed-precision), nicht ModelOpt: block-FP8 [128,128] NUR auf den
Attention-Projektionen (`fused_wqa_wkv|wq_b|wo_a|wo_b`), ALLE
ffn.(gate|up|down)_proj inkl. Shared Experts sind NVFP4 (group 16,
fp8-Skalen). `DeepseekV4ForCausalLM` ist in der Fork-Registry vorhanden.

**Nächstes Paket für den DeepSeek-Boot:** der compressed-tensors-Pfad —
`CompressedTensorsW8A8Fp8.get_min_capability()` ist 89 (SM70 fällt durchs
Gate), und die NVFP4-Experten kommen als `nvfp4-pack-quantized` statt
ModelOpt. Ob dessen Block-FP8-Scheme durch `init_fp8_linear_kernel` läuft
(dann greift die neue Klasse automatisch), ist zu prüfen. Davor llama.cpp-
Latte messen (Schritt 1 oben) und PP=5 klären (Schritt 2).

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


## Sitzung 2026-08-25 (nachmittags, Fortsetzung): DeepSeek-Bring-up auf Volta — Stand

**Latte gemessen** (bench.py, math+code, Seeds 1001/2002, Prod-Config mit
dspark-Draft): **math 40,4 ± 1,1 · code 42,8 ± 0,4 tok/s.** In RESULTS.md.

**PP=5 bewiesen** (27B, Partition 16,16,11,11,10, K=7, elf Gates grün):
66,8/42,9 tok/s — ~22 % unter dem 2×2, aber kohärent. In RESULTS.md.

**Default-Flip vollzogen:** QPN8-blk ist Standard für Block-FP8 (fp8.py;
`VLLM_SM70_QPN8_BLK=0` = Rollback). Graph-Mode verifiziert (PIECEWISE+FULL
Capture, Output identisch, 329 tok/s auf dem 0.6B).

**DeepSeek-Bring-up: 9 Boots, 8 Fixes — die gesamte Architektur läuft auf
Volta bis einschließlich Attention; die Wand sind die gerouteten Experten.**
Alle Fixes als fork_patches deployt + in bootstrap-sm70.sh eingetragen:
1. `deepseek_v4_nvidia_model.py`: `scale_fmt` per `.get()` (ModelOpt-Key,
   leserlose Zuweisung; compressed-tensors hat ihn nicht).
2. Launcher `scripts/serve-deepseek-mini.sh`: `--kv-cache-dtype fp8` Pflicht,
   MNBT/MML/GMU-Knöpfe, Modulstart (1cat-Wheel hat keine vllm-Dist-Metadata).
3. `compressed_tensors.py`: Block-FP8-W8A8 auf SM70 zugelassen
   (`_CompressedTensorsW8A8Fp8Sm70Block`, Gate 70, nur Block-Strategie) →
   Attention-Linears laufen über QPN8-blk (V100-Stufen: Census+Prepack-Assert
   grün) bzw. fp16-Dequant (RTX-Stufen, sm75-Zweig in qpn8_blk.py; auch
   `is_bmm`-Layer wie wo_a auf BEIDEN Archs, das Modell liest deren .weight
   direkt im Gruppen-einsum).
4. `nvfp4_moe_oracle.py`: EMULATION in die SwiGLU-Clamp-Liste (TritonExperts
   implementiert den Clamp längst; nur der Filter schloss es aus — Modelle
   mit swiglu_limit hatten auf pre-Hopper GAR KEIN Backend). Upstream-würdig.
5. `sparse_attn_indexer.py`: Torch-Referenz für die zwei DeepGEMM-Logit-
   Kernel (Lightning-Indexer: ReLU über Head-Dots, gewichtete Summe; fp16-
   Matmuls, Bounds macht weiter der Bounded-TopK). DeepGEMM ist SM90+.
6. `mhc_tilelang.py`: fp16-Torch-Pfad für ALLE vier mHC-Einstiege (pre,
   post, fused=post∘pre, hc_head) — TileLang-Kernel sind hart bf16, Volta
   hat kein bf16. Intern fp32, Verträge exakt gespiegelt.
7. `deepseek_v4_attention.py`: Referenz-O-Pfad ohne DeepGEMM (inverse RoPE
   torch + fp16-Gruppen-einsum über das dequantisierte wo_a) und Torch-
   Q-Quant für den Indexer (Numerik exakt inkl. bf16-Rundungs-Roundtrip,
   Scale-Fold in weights). Triton kann auf sm70 kein fp8e4nv.
8. Speicherprofil: MNBT 2048, MML 8192, expandable_segments.

**Die Wand: `Nvfp4QuantizationEmulationTritonExperts` dequantisiert pro
Forward die KOMPLETTEN Expertengewichte einer Layer** (~13 GB fp16 bei 256
Experten) → OOM auf 47-GB-Karten, und selbst passend wäre der Traffic
absurd. Die Emulation ist für kleine MoEs gebaut.

**Optionen (Entscheidung Peuqui):**
- **A: Triton-fused-MoE mit NVFP4-Dequant im Kernel** (e2m1-LUT + fp8-
  Skalen-Decode per Bit-Mathe, kein fp8-Cast nötig — analog use_int4_w4a16).
  Mittlerer Aufwand. Traffic-Abschätzung: ~0,2 GB packed/Layer/Step → Decke
  ~80 tok/s, real auf Volta-Triton eher 15–25 → Latte (40) fraglich.
- **B: Skinny-NVFP4-Grouped-MoE-CUDA-Kernel** (QPN-Codec existiert, neu ist
  der Grouped-Expert-Dataflow). Großes Projekt, einzige realistische Chance
  die Latte zu schlagen.
- **C: Hier schneiden.** Attention-Stack + QPN8-blk sind bewiesen und
  committen-reif; DeepSeek-vLLM als eigenes Folgeprojekt mit B.

Repro aktueller Stand: `bash scripts/serve-deepseek-mini.sh <snap>` mit
`EXTRA_ARGS=--enforce-eager MML=8192 PYTORCH_ALLOC_CONF=expandable_segments:True`
→ scheitert reproducible im ersten MoE-Forward (OOM, moe_runner→
nvfp4_emulation_moe.dequantize_to_dtype).

## ENTSCHIEDENER PLAN (Peuqui, 25.08. abends): Zwei Schritte

Die Wand ist lösbar — fehlende Software, keine Hardware-Grenze. Zwei Belege:
llama.cpp fährt 4-Bit-Experten mit Dequant-im-Kernel auf DENSELBEN fünf
Karten mit 40 tok/s, und der Skinny-NVFP4-Codec (dequant8_tm & Co.) dekodiert
dasselbe Format bereits im Kernel (27B: 88 tok/s). Es fehlt allein der
Grouped-Experten-Datenfluss auf Volta.

**Schritt 1 (klein, ZUERST): Korrektheit des gesamten Volta-Ports beweisen,
bevor der große Kernel entsteht.** Die Emulation chunkbar machen: in
`Nvfp4QuantizationEmulationTritonExperts.apply`
(`fused_moe/experts/nvfp4_emulation_moe.py`) werden w1/w2 heute KOMPLETT
dequantisiert (~13 GB/Layer). Umbau: Experten häppchenweise in einen
wiederverwendeten Workspace dequantisieren und die Expert-GEMMs pro Chunk
fahren (oder nur die AKTIVEN Experten aus topk_ids dequantisieren — bei
Decode 8 von 256). Ziel ist NICHT Tempo (grob ~1 s/Token ist ok), sondern:
DeepSeek bootet, generiert 30–50 Token, Kohärenzprobe + Mathe-Check gegen
llama.cpp-Ausgaben. Damit ist alles Heutige (QPN8-Attention, Indexer-Torch,
mHC-fp16, Referenz-O-Pfad) end-to-end verifiziert. Fork-Patch + Deploy-Zeile
nicht vergessen.

**Schritt 2 (groß, eigene Session(s)): Grouped-NVFP4-MoE-Kernel für SM70**
im Stil der Skinny-Familie (`kernels/skinny_kernels.cu`). Schlüssel-Insight:
Beim Decode sieht jeder aktive Experte nur 1–8 Token — exakt das
Skinny-M-Band der QPN-Kernel. Bausteine: Token→Expert-Gather liefert die
vLLM-Modular-Kernel-Infrastruktur (siehe TritonExperts als Strukturvorlage),
neu ist ein Kernel, der pro (Experte, Token-Gruppe) die NVFP4-Codes des
Experten dekodiert und die gate_up→SwiGLU(clamp 10.0!)→down-Kette rechnet;
Prefill-Band analog WMMA. Integration als eigene FusedMoEExperts-Klasse in
der Backend-Kette (`fused_moe/oracle/nvfp4.py`, vor EMULATION). Traffic-
Decke ~80 tok/s ⇒ die 40er-Latte ist realistisch schlagbar. Qualitätsrisiko
null — identische Mathematik wie die Emulation, ohne fp16-Materialisierung.

### Betriebsnotizen für die nächste Instanz
- Branch `pp-mtp-merge`, ALLES UNCOMMITTED (kernels/skinny_kernels.cu,
  scripts/, benchmarks/, 10 fork_patches, dieses Dokument, RESULTS.md).
  Deploy-Stand in `.venv-sm70` entspricht den fork_patches; Bootstrap-
  Deploy-Liste ist vollständig nachgeführt.
- Serve: `scripts/serve-deepseek-mini.sh` (TP=1 PP=5, Partition 11,11,7,7,7,
  Reihenfolge 0,2,1,4,3 = RTX,RTX,V100,V100,V100; fp8-KV Pflicht; eager —
  die Indexer/mHC-Torch-Pfade sind noch nicht graph-tauglich).
- GPU 3 gehört produktiv dem Vigilantia-VLM (lädt on demand) — vor Läufen
  `nvidia-smi` prüfen; llama-swap-Modelle mit `curl :11435/unload` räumen.
- Messwerkzeuge: `vllm-bench/bench.py` (Seeds 1001/2002, Zellen math+code),
  Qualität via `vllm-bench/quality.py` auf BEIDEN Engines (quality_kl.py
  geht nicht — kein zweiter vLLM-DeepSeek-Anker für Teacher Forcing).
- Latte: llama.cpp math 40,4 ± 1,1 / code 42,8 ± 0,4 tok/s (RESULTS.md).

## Sitzung 2026-08-25 (abends): Schritt 1 ausgeführt — Wand weg, neue Wand sichtbar

### Was erledigt ist

**Die Emulation ist gechunkt, und die MoE-Wand ist damit gefallen.**
`fork_patches/nvfp4_emulation_moe.py` (deployt nach
`vllm/model_executor/layers/fused_moe/experts/nvfp4_emulation_moe.py`,
Bootstrap-Zeile ergänzt) dequantisiert nur noch die vom Router
ausgewählten Experten (`torch.unique(topk_ids)` — beim Decode 6 von 256)
und das in Chunks zu je N Experten (`VLLM_SM70_NVFP4_EMU_CHUNK`,
Default 4). Jeder Chunk läuft als eigener Expert-Parallel-Shard: seine
Experten sind die lokalen, alle anderen sind über `expert_map = -1`
maskiert (der Triton-Kernel schreibt für `expert_id == -1` Nullen,
`fused_moe.py:161`), die Partialsummen werden addiert. Kein EP-Fallback:
`expert_map is not None` wirft `NotImplementedError`.

Vorab isoliert verifiziert (`scripts/nvfp4_emu_chunk_test.py`, V100):
die Chunk-Summe gegen den ungechunkten `fused_experts`-Lauf, **8/8 grün**,
rel_err ≤ 9,2e-4 (fp16-Akkumulationsrauschen; exakt 0 wo ein Chunk
reicht). Fälle: Einzeltoken-Decode, MTP-Batch, Chunk=1, 205 von 256
Experten aktiv, Prefill-Form, ragged Tail.

**DeepSeek-V4-Flash bootet damit unter vLLM auf der 5-GPU-Kiste.**
`MML=4096 MNBT=512 GMU=0.96 EXTRA_ARGS=--enforce-eager
PYTORCH_ALLOC_CONF=expandable_segments:True VLLM_SM70_NVFP4_EMU_CHUNK=2`
→ `Application startup complete`, KV-Cache 180k–213k Token pro Stufe.
Log: `deepseek-chunked2.log`. Mit MML=8192/MNBT=2048/GMU=0.94 scheitert
es dagegen an der Kapazität (Stufe 0 nur 0,87 GiB KV frei, mindestens
eine Stufe ≤ 0) — die Knöpfe sind also eng zu fahren.

### Die neue Wand: die Attention-KERNEL sind auf Volta nie gelaufen

Der erste echte Request stirbt sofort:

```
RuntimeError: fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert
requires sm_80+ (Ampere or newer); got sm_75
```

**Korrektur der Ausgangsannahme der Vorsitzung.** „Die gesamte
Architektur läuft auf Volta bis einschließlich Attention" ist NICHT
belegt — der Schluss stammte aus dem Profiling-Lauf, und der umgeht
beide Attention-Kernel per Konstruktion:

- `attention.py:_fused_qnorm_rope_kv_insert` steigt bei
  `not isinstance(attn_metadata, dict)` vorzeitig aus („Profile run:
  kernel doesn't fire") und gibt nur ein passend gepolstertes q zurück.
- `nvidia/flashmla.py:forward_mqa` macht bei `attn_metadata is None`
  ein `output.zero_()` und kehrt zurück.

Im Profiling war die Attention-Ausgabe also **konstant Null**. Belegt
sind damit die GEMMs drumherum (QPN8-blk), der Torch-Indexer, mHC-fp16
und die MoE — nicht die Attention-Rechnung selbst.

Zwei Bausteine fehlen für Volta, beide NVIDIA-seitig:

1. **`fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert`** (C++-Op mit
   hartem sm80-Gate) — die Schreibseite: Q-seitig gewichtsloses
   Per-Head-RMSNorm + GPT-J-RoPE + Zero-Pad auf `padded_heads`,
   KV-seitig GPT-J-RoPE + UE8M0-FP8-Quant + Paged-Cache-Insert.
   Cache-Eintrag ist `head_bytes = nope(448 fp8) + rope*2(128 bf16) +
   nope/64(7 Skalen) + 1 Pad = 584 B`. Aufrufstelle ist bereits
   gepatcht-fähig (`fork_patches/deepseek_v4_attention.py`), und
   `self.q_head_norm` (RMSNorm, `has_weight=False`) existiert schon als
   Modul.
2. **`flash_mla_with_kvcache`** (FlashMLA, SM90+) — die Rechenseite.
   `_select_v4_sparse_impl()` wählt auf CUDA immer
   `DeepseekV4FlashMLASparseImpl`; es gibt keinen pre-Hopper-Zweig.

**Portierungsvorlage vorhanden:** `models/deepseek_v4/amd/rocm.py`
(856 Zeilen, reines Triton) plus `v1/attention/ops/rocm_aiter_mla_sparse.py`
(~1700 Zeilen Triton + Torch-Referenzen: `rocm_sparse_attn_prefill/decode`,
`fp8_paged_mqa_logits_torch`, `_apply_gptj_inv_rope_ref`,
`_decode_e8m0_scales`, `indexer_k_quant_and_cache_triton`). Der
Plattform-Zweig sitzt an genau EINER Stelle (`_select_v4_sparse_impl`).
Bekannte Hürde: Triton kann auf sm70 kein `fp8e4nv` — dieselbe
Bit-Mathe-Umgehung wie bei den bisherigen Fixes wäre nötig.
Offen zu prüfen: wie die ROCm-Seite die SWA-Cache-Schreibseite bedient —
`_fused_qnorm_rope_kv_insert` wird dort unverzweigt aufgerufen, ROCm
hätte den Op also ebenso wenig.

### Landkarte statt Neubau (gleiche Sitzung, nach Peuquis Einwand)

Peuqui: „wir sollten soviel guten getesteten Code (im-)portieren wie
möglich — es gibt ja fertige Implementationen für Volta und Turing und
NVFP4, nur nicht als Komplex." Zutreffend. Die folgende Inventur ersetzt
die Annahme „Attention-Port ≈ Größe des MoE-Kernels".

**Muster:** JEDE Stelle, die auf Volta bricht, hat bereits einen zweiten,
getesteten Zweig — nur hängt der an `is_rocm()` statt an einer
Capability-Abfrage.

| Blocker | Vorhandener Ersatz |
|---|---|
| `flash_mla_with_kvcache` (SM90+) | `DeepseekV4ROCMAiterMLASparseImpl` (`models/deepseek_v4/amd/rocm.py`) — **keine einzige AMD-Abhängigkeit**, reines Triton, erbt von den NVIDIA-Basisklassen und benutzt deren Metadata-Builder |
| CuteDSL-Compressor (`compressor.py:338`, head_dim 512) | `compress_norm_rope_store_triton` — im selben Dispatch, AMD-Zweig |
| `fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` (C++, sm80-Gate) | KV-Hälfte: Triton-Zwilling `_fused_kv_compress_norm_rope_insert_sparse_attn` (`common/ops/fused_compress_quant_cache.py`), schreibt laut Docstring „head=512, nope=448 FP8 + rope=64 bf16" = dasselbe 584-B-Layout. Q-Hälfte: `self.q_head_norm` (RMSNorm, has_weight=False) und `rotary_emb.forward_native` existieren beide |

**Die eine echte Hürde — und auch die ist gelöstes Terrain.**
Empirisch geprüft (`scratchpad/triton_e4b15_probe.py`) auf V100 **und**
RTX 8000: `type fp8e4nv not supported in this architecture. The supported
fp8 dtypes are ('fp8e4b15', 'fp8e5')`. Also KEIN fp8e4nv auf sm70 UND
sm75 — die Hürde gilt für alle fünf Karten.

`fp8e4b15` ist e4m3 mit Bias 15 statt 7, gleiche 1-4-3-Bitaufteilung.
Erschöpfend über alle 256 Bytes getestet:
- **Decode** (`bitcast to fp8e4b15 → fp32 → ×256`): 254/256 Muster
  identisch zu e4m3fn. Die zwei Abweichungen sind `0x7F`/`0xFF`, die
  NaN-Kodierungen — die ein korrekter Quantisierer nie erzeugt (Clamp bei
  ±448 = `0x7E`). Für regulär geschriebene Caches also exakt.
- **Encode** (`×1/256 → cast to fp8e4b15 → bitcast to uint8`): 254/254
  endliche Werte exakt.
- Beide Richtungen, beide Architekturen.

Der ×256-Faktor ist nötig, weil die Cache-Bytes echtes e4m3fn bleiben
sollen (es gibt weitere Leser). vLLMs eigener Turboquant-Pfad lässt ihn
weg, weil dort derselbe Code schreibt und liest und der Faktor kürzt —
`triton_turboquant_store.py:192` bzw. `triton_turboquant_decode.py:165`.
**Das Capability-Gate existiert schon fertig:** `_use_fp8_e4b15(device)`
in `triton_turboquant_decode.py:26` („Return 1 if device needs fp8e4b15,
SM < 8.9"), per-Device gecacht.

Betroffen sind **9 Stellen in 2 Idiomen**, alle mechanisch:
- Encode (7): `cache_utils.py:116`, `fused_compress_quant_cache.py:254,466`,
  `fused_indexer_q.py:141,146,150`, `fused_inv_rope_fp8_quant.py:108`
- Decode (3): `cache_utils.py:277`, `rocm_aiter_mla_sparse.py:1233,1301`
  — die zwei letzten haben die Verzweigung SCHON, nur an `IS_FNUZ`
  gehängt statt an der Capability.

**Schritt 2 (NVFP4-MoE) ist ebenfalls kleiner als angenommen.**
Der Fork hat ein geteiltes SM70-MoE-Gerüst: `sm70_moe_router.py`
(`select_sm70_quantized_moe_route`) — Docstring: „This only chooses the
compute order. AWQ and FP8 keep their own **thin adapters** because their
packed weights and scale descriptors differ." Bestand in `_sm70_ops.py`:
- **AWQ-INT4: 13 MoE-Ops** (`awq_moe_gemm_sm70_out`, per-expert-dispatch,
  dense/active-dense stages, Single-Token-Schnellpfade, weighted reduce,
  strided-ptr-Bau) — die vollständige Familie, plus 1704 Zeilen Adapter
  in `awq_sm70_moe.py`.
- **FP8: 9 MoE-Ops** plus 1368 Zeilen `fp8_sm70_moe.py`.
- **NVFP4: 0 MoE-Ops** — nur Linear-Ebene (`nvfp4_gemm_sm70_out`, drei
  GEMV-Varianten, `nvfp4_sm70_prepare`).

Der Grouped-NVFP4-Kernel fehlt also wirklich — aber Dataflow, Routing,
Adapterstruktur und die Single-Token-Schnellpfade (das „Skinny-M-Band"
der Übergabe) sind zweimal vorexerziert. Aufwand eher „dritter Adapter
plus Kernel nach zwei Vorlagen" als „Neuentwurf".

**Noch offen in der Landkarte** (vor dem ersten Patch zu klären):
1. Kompilieren die ROCm-Triton-Kernel auf sm70 auch jenseits von fp8
   (andere nicht unterstützte Ops)?
2. Kann der Compressor-Triton-Kernel den SWA-Insert direkt bedienen, oder
   braucht es eine Variante ohne Kompression?
3. Hat die Q-Hälfte des sm80-Ops (Per-Head-RMSNorm + RoPE + Zero-Pad auf
   `padded_heads`) ein exaktes vorhandenes Gegenstück?

## SCHRITT 1 ERREICHT (2026-08-25 abends): DeepSeek-V4-Flash generiert unter vLLM auf Volta/Turing

**Belegt.** Erste kohärente Generierung auf den fünf Karten, Antwort korrekt
(`Paris`). Damit ist der GESAMTE Volta-Port erstmals end-to-end durch einen
echten Forward mit echtem KV-Cache gelaufen — QPN8-blk-Attention,
Torch-Lightning-Indexer, fp16-mHC, Referenz-O-Pfad, der neue Referenz-
KV-Schreiber, der importierte Triton-Sparse-Attention-Pfad und die gechunkte
NVFP4-Expertenemulation.

### Repro

```bash
cd /home/mp/Projekte/v100-skinny
SNAP=$(ls -d /home/mp/.cache/huggingface/hub/models--RedHatAI--DeepSeek-V4-Flash-NVFP4-FP8/snapshots/*/)
EXTRA_ARGS=--enforce-eager MML=4096 MNBT=256 GMU=0.90 \
PYTORCH_ALLOC_CONF=expandable_segments:True VLLM_SM70_NVFP4_EMU_CHUNK=1 \
bash scripts/serve-deepseek-mini.sh "$SNAP"
```

`GMU` MUSS runter (0,96 → 0,90): weil im Profiling-Lauf ALLE Attention-Kernel
aussteigen, dimensioniert vLLM den KV-Cache ohne deren Arbeitsspeicher — bei
0,96 bekommt der Cache 180.000 Token und die Kernel gehen leer aus (OOM auf
PP4 beim Compressor-Kernel-Start). Bei 0,90 bleiben 48.668 Token, weit mehr
als die gefahrenen 4.096.

### Kohärenzprobe gegen llama.cpp (`scripts/deepseek_coherence.py`)

8 Prompts, greedy, max 32 Token, beide Engines dieselbe Datei.
**7 von 8 stimmen semantisch überein.**

| Prompt | vLLM-Port | llama.cpp |
|---|---|---|
| capital | Paris | Paris |
| arith (37×43) | **1591** | **1591** |
| count ("strawberry") | 7 | **10** |
| seq | "doubling the previous one" | "multiplied by 2" |
| recall | Mercury | Mercury |
| prose | sinngleich, unschärfer | präziser |
| code | `s = s[::-1]` | `s[::-1]` |
| longctx (8er-Liste) | exakt, in Reihenfolge | exakt |

Die einzige Abweichung ist die Buchstabenzählung — klassische
Tokenisierungsfalle, und die Quantisierungen unterscheiden sich ohnehin
(NVFP4-Experten + Block-FP8 vs. UD-Q4_K_XL), Tokenidentität war nie zu
erwarten. Die aussagekräftigen Prüfungen sind sauber: exakte Multiplikation,
exakte geordnete Achtwortliste (geht nur mit funktionierender Attention über
den KV-Cache), lauffähiger Code, grammatische Prosa. Ein Portschaden sähe
anders aus.

**Tempo: 0,07–0,36 tok/s** gegen llama.cpps 21–26 tok/s auf denselben
Prompts (Faktor ~70). Erwartet — `EMU_CHUNK=1`, eager, Torch/Triton-
Referenzen überall. Schritt 1 war der Korrektheitsbeweis, nicht Tempo.

### Die sechs Blocker und ihre Behebung

Fünf von sechs waren dasselbe Muster: **die Bedingung fragte nach der
Plattform, gemeint war die Fähigkeit.**

| # | Blocker | Behebung | Datei |
|---|---|---|---|
| 1 | `fused_..._qnorm_rope_kv_rope_quant_insert` sm80-Gate | Referenz aus `q_head_norm` + `rotary_emb.forward_native` + `quantize_and_insert_k_cache` | `deepseek_v4_attention.py` |
| 2 | FlashMLA-Impl (SM90+) | `_select_v4_sparse_impl`: Pre-Hopper nimmt die Triton-Impl | `deepseek_v4_attention.py` |
| 3 | `KeyError: 'sm_75'` in CuteDSL | `has_cutedsl()` verlangt zusätzlich Capability 80 | `import_utils.py` |
| 4 | `tl.dot`: fp16 gegen bf16 | Cache-Seite folgt dem Query-Dtype (6 Stellen) | `rocm_aiter_mla_sparse.py` |
| 5 | Tile-Scheduler ruft FlashMLA | `_needs_triton_sparse_swa()` in beiden Gates | `sparse_swa.py` |
| 6 | 70.656 B Shared Memory | `block_h` nach gemessenem Gerätelimit (V100 98.304 B, RTX 8000 **65.536 B**) | `rocm_aiter_mla_sparse.py` |

Dazu die neun→sechs fp8-Stellen (`float8e4nv` → `float8e4b15` × 256), siehe
Landkarte oben.

### Schlüsselerkenntnis für alles Weitere

`models/deepseek_v4/amd/rocm.py` trägt **keine einzige AMD-Abhängigkeit** —
reines Triton, erbt von den NVIDIA-Basisklassen. Der Name führt in die Irre.
Wer hier weiterarbeitet, sollte zuerst prüfen, ob der gesuchte Pfad schon als
„ROCm"-Variante existiert, bevor er etwas schreibt.

### Offen

1. **Schritt 2** (Tempo): Grouped-NVFP4-MoE auf SM70. Der Fork hat das
   Gerüst schon — `sm70_moe_router.py` ist laut Docstring
   quantisierungsunabhängig („AWQ and FP8 keep their own thin adapters"),
   **AWQ-INT4 hat 13 MoE-Ops**, **FP8 hat 9**, **NVFP4 hat 0** (nur Linear:
   `nvfp4_gemm_sm70_out`, drei GEMV-Varianten, `nvfp4_sm70_prepare`). Also:
   dritter Adapter nach zwei Vorlagen plus der fehlende Grouped-Kernel.
2. **Kontext**: die 180k/48k aus den Boot-Logs sind KV-Obergrenzen OHNE die
   Attention-Workspaces. Der real fahrbare Kontext liegt darunter; für die
   Messphase gegen die 40,4/42,8-Latte erst schrittweise hochfahren.
3. **CUDA-Graphen**: alles läuft `--enforce-eager`. Die Torch-Referenzpfade
   (Indexer, mHC, O-Pfad, KV-Schreiber) sind nicht graph-tauglich.
4. **`quality.py`-Duplikat**: `scripts/deepseek_coherence.py` hat einen
   eigenen HTTP-Chat-Helfer, den vierten in `vllm-bench` (bench.py,
   quality.py, quality_kl.py). Konsolidieren.

## SCHRITT 2 NEU BEWERTET (2026-08-25 nachts): kein Grouped-Kernel noetig

Die Uebergabe plante einen Grouped-NVFP4-MoE-CUDA-Kernel als „grosses
Projekt, einzige realistische Chance die Latte zu schlagen". **Das ist fuer
das Decode-Band nicht noetig.** Der vorhandene `skinny_gemm_qpn` bedient
einen Experten direkt.

**Belegt** (`scripts/nvfp4_expert_gemm_test.py`, echte Checkpoint-Bytes aus
`layers.5.ffn.experts.0`, V100): 10/10 Faelle, rel_err ~6e-4 gegen
`dequantize_to_dtype` + matmul. Geprueft bei M = 1, 2, 6, 8, 16 fuer w1
(N=2048, K=4096) und w2 (N=4096, K=2048).

### Warum es passt

Das Checkpoint-Layout pro Experte IST die Kernel-Signatur:

| Kernel erwartet | Checkpoint liefert |
|---|---|
| `codes` uint8 [N][K/2] | `w1.weight_packed` [2048, 2048] uint8 |
| `scales` uint8 [N][K/16] fp8-e4m3 | `w1.weight_scale` [2048, 256] F8_E4M3 |
| `gscale` ein Skalar | `w1.weight_global_scale` [1] fp32 |
| M 1..16 | Decode: jeder aktive Experte sieht 1-8 Token |

Ein Expertenschnitt ist zusammenhaengend, also ohne Kopie nutzbar.

### Zwei Fallstricke, beide gekostet

1. **`weight_global_scale` ist der KEHRWERT.** Der Checkpoint speichert
   21504.0; dequantisiert wird mit 1/21504 = 4,65e-5. Mit dem Rohwert
   laeuft der Gewichts-Absmax auf `inf` und alles wird NaN.
2. **`gemm_qpn` will die Fragment-Reihenfolge, nicht die rohen Bytes.** Der
   Dateikopf von `skinny_kernels.cu` beschreibt das LOGISCHE Format; der
   Kernel erwartet die Permutation aus `_qpn_prepack`
   (`kernels/linear/nvfp4/marlin.py:87`). Ohne Prepack: rel_err ~1,3.
   Der Prepack ist eine reine Permutation — **kein Speicherzuwachs**.

### Was daraus folgt

Schritt 2 ist ein **Per-Experten-Dispatch-Adapter**, kein Kernel-Projekt:
pro aktivem Experten `gemm_qpn` fuer gate/up, SwiGLU (clamp 10.0!), dann
`gemm_qpn` fuer down, Ergebnis gewichtet aufaddieren. Als eigene
`FusedMoEExperts`-Klasse vor EMULATION in `fused_moe/oracle/nvfp4.py`.

Grobe Rechnung: 6 aktive Experten x 2 GEMMs x 43 Layer = 516 Kernel-Starts
pro Token, verteilt auf 5 PP-Stufen. Traffic ~3,6 GB/Token gegen ~700 GB/s
=> einige ms; Startkosten bei ~10 us dominieren. Groessenordnung 30-60
tok/s, also die 40er-Latte in Reichweite — ohne eine Zeile CUDA.

### Offen fuer die Umsetzung

1. **Prepack-Kosten beim Laden**: 256 Experten x 3 Matrizen x 43 Layer =
   ~33.000 `_qpn_prepack`-Aufrufe. Der Prepack chunkt intern schon gegen
   int64-Spitzen; Laufzeit messen, ggf. ueber die Experten-Dimension
   vektorisieren.
2. **Prefill-Band**: `gemm_qpn` deckt M<=16. Darueber `gemm_wmma` (M<=64)
   oder Token-Chunking; alternativ Prefill weiter ueber die Emulation.
3. **SwiGLU-Clamp 10.0** nicht vergessen (swiglu_limit im Config).
4. **gate/up sind getrennte Tensoren** (`w1`/`w3`), nicht fusioniert —
   entweder zwei GEMM-Aufrufe oder beim Laden konkatenieren (`gemm_qpn`
   hat kein `gated_silu`-Flag; `nvfp4_gemm_sm70_out` haette eines).

## SCHRITT 2 UMGESETZT (2026-08-26): Per-Experten-Skinny-MoE — 11x, jetzt CPU-bound

### Was gebaut wurde (alles Kleben, kein neuer Kernel, kein Prepack)

Die Neubewertung oben plante `gemm_qpn` + `_qpn_prepack`. Beim Umsetzen
zeigte sich eine noch kleinere Loesung: **`gemm_simt` (M<=7) und
`gemm_wmma` (M<=64) lesen das Checkpoint-Layout DIREKT** — genau die
Slices, die der Expertentest als layoutgleich bewiesen hat. Damit
entfallen Prepack (offener Punkt 1) und jede Layout-Mutation ersatzlos;
die Gewichte bleiben wie geladen.

- **`fork_patches/nvfp4_skinny_moe.py`** (deployt nach
  `fused_moe/experts/nvfp4_skinny_moe.py`, Bootstrap-Zeile ergaenzt):
  `Nvfp4SkinnySm70Experts` erbt von der gechunkten Emulationsklasse,
  ueberschreibt nur `apply`. Kern ist die modulglobale, testbare Funktion
  `skinny_moe_forward`: ein Host-Sync pro Layer (`topk_ids.cpu()`), dann
  pro aktivem Experten Gather -> gemm(w13-Slice) -> geerbter
  `activation()`-Helper (SwiGLU-Clamp 10.0 inklusive, offener Punkt 3)
  -> gemm(w2-Slice) -> gewichteter `index_add_`. M-Dispatch nach der
  vermessenen Linear-Frontier: simt <=7, wmma <=64, darueber 64er-Chunks
  (offener Punkt 2). Punkt 4 (w1/w3 getrennt) erledigt vLLM selbst: der
  Loader fusioniert zu w13; bei abweichenden w1/w3-Globalscales gilt
  dieselbe [:,0]-Reduktion (+Warnung) wie ueberall.
- **Aktivierungen bleiben fp16 (w4a16)** — bewusste Abweichung von der
  Emulations-QDQ, konsistent mit allen NVFP4-Linears des Forks.
- **Oracle** (`nvfp4_moe_oracle.py`): Backend `SM70_SKINNY` vor
  EMULATION, in der Clamp-Liste, `moe_backend="sm70_skinny"` explizit
  waehlbar, Rollback `VLLM_SM70_NVFP4_MOE_SKINNY=0`. Der
  Konvertierungs-Branch ist bewusst leer (Checkpoint-Layout ==
  Kernel-Layout); Quant-Config = w4a16-Bauart + `gemm1_clamp_limit`.
- Gate auf lokales Worker-Device (Session-4-Lektion), sm70 UND sm75 —
  die Skinny-NVFP4-Kernel sind RTX-solo-verifiziert.

### Validierung

- `scripts/nvfp4_skinny_moe_test.py` (echte Checkpoint-Bytes, Layer 5,
  12 Experten): **7/7 auf V100 UND RTX 8000**, rel_err <= 2,1e-3 —
  Decode, MTP-Batch, Prefill-Baender, erzwungenes m_e=400-Chunking,
  Doppel-Slot-Routing.
- Boot: Oracle waehlt `SM70_SKINNY`; Kohaerenzprobe **identisch zur
  Schritt-1-Tabelle** (7/8, gleiche count-Abweichung).

### Tempo-Befund: MoE ist nicht mehr der Engpass

- Serving: **1,3-4,1 tok/s** (vorher 0,07-0,36 — Faktor ~11).
- MoE-only-Mikrobench (M=1, topk=8, echte Bytes): 1,08 ms/Layer ->
  46 ms/Token -> 21,5 tok/s waeren MoE-seitig drin. Davon nur ~0,15 ms
  Kernel — der Rest Python-/Launch-Overhead des eager-Loops.
- Torch-Profil (`--profiler-config.profiler=torch`, Traces in
  `profiles/`): auf den V100-Stufen war der groesste GPU-Einzelposten
  der **Shared-Experts-Pfad ueber TurboMind** (`nvfp4_gemm_sm70_out`,
  ~1,15 ms Self-CUDA je GEMM ~ 4-5 ms/Layer), waehrend die RTX-Stufen
  dieselben GEMMs ueber `sm70_marlin` in 129 us fahren (sm70_turbomind.py
  waehlt TurboMind nur auf exakt sm70).
- **`VLLM_SM70_QUANT_BACKEND=marlin`** (vorhandener Schalter) beseitigt
  den Posten — Kohaerenz identisch, **Tokenrate unveraendert ~4 tok/s**.
  Beleg: das System ist CPU-Dispatch-bound. Die NCCL-Broadcast-Kernel
  „spinnen" 87-95 % der CUDA-Zeit (Pipeline-Warten), und die Breite aus
  tausenden 1-5-us-Kerneln der Torch-Referenzpfade (Lightning-Indexer,
  mHC-fp16, Referenz-O-Pfad, Referenz-KV-Schreiber) plus deren
  Python-Dispatch dominiert die 244 ms/Token auf jeder Stufe.

### Naechstes Paket (gross, eigene Session): Referenzpfade eindampfen

Die 40er-Latte faellt nicht durch MoE-Feintuning (brachte rechnerisch
4,1 -> 4,4), sondern nur durch Angriff auf die Breite:

1. **Torch-Referenzpfade durch kompakte Kernel ersetzen — wieder erst
   Landkarte:** die ROCm-Triton-Seite (`rocm_aiter_mla_sparse.py`,
   `fused_compress_quant_cache.py`) hat fusionierte Kernel fuer Indexer
   und KV-Schreibseite, die die importierte Sparse-Attention schon
   nutzt; pruefen, welche der 2026-08-25-Referenzpfade (Indexer-Q-Quant,
   mHC, O-Pfad) dort ebenfalls fertige Zwillinge haben.
2. **CUDA-Graphen**: eager kostet auf JEDER Stufe Dispatch; die
   Referenzpfade sind nicht graph-tauglich (Handover-Punkt 3) — nach 1.
   neu bewerten. Der MoE-Loop selbst ist wegen dynamischem Routing +
   Host-Sync nicht graph-faehig; fuer Decode (M=1, topk fix) waere eine
   graph-taugliche Variante ueber einen Grouped-Dispatch denkbar —
   NACH 1. messen, ob noetig.
3. Danach erst bench.py gegen die Latte (40,4/42,8); die 4 tok/s sind
   indikativ aus der Kohaerenzprobe, KEINE Kampagnen-Messung, deshalb
   nicht in RESULTS.md.

Betriebsnotiz: Server-Repro unveraendert (serve-deepseek-mini.sh, GMU
0,90, eager); optional `VLLM_SM70_QUANT_BACKEND=marlin` (Shared-Experts/
Dense-Linears auf V100 ueber Marlin statt TurboMind — GPU-Zeit runter,
tok/s bei eager unveraendert). Logs: deepseek-skinny-moe.log,
deepseek-skinny-marlin.log; Profile: profiles/.
