# Entwurf: TP-Prefill – Allreduce mit der Rechnung überlappen (17.10.2026)

Status: Entwurf, nichts gebaut. Plan Peuqui 17.10.: (1) A/B `turing-prefill`, (2) Mikrobench große Host-Übertragungen,
(3) dieser Entwurf, (4) PP2 und FP8-Allreduce bleiben gestrichen. Der Mikrobench (`~/.cache/bench-scripts/
host_allreduce_bench.py`) läuft erst nach dem A/B; Punkte, die von ihm abhängen, sind unten als offen markiert.

## Befund

Gemessen:

- Profil 06.10. (STAND „Nachtrag 06.10. früh“), 27B TP2 auf den RTX, 9,7k Prompt: 13,0 s Kernelzeit je GPU, davon
  6,2 s (47,8 %) NCCL-Allreduce, cuBLAS-GEMMs ~20 %, NVFP4-Entpacker ~7 %.
- NCCL RTX↔RTX bei großen Nachrichten ~2,1 GB/s (alle Protokolle und Puffergrößen), Senden in eine Richtung
  2,9–3,1 GB/s (29.09.). Alle fünf Karten hängen an PCIe Gen3 x4 (`nvidia-smi`), kein P2P (`NCCL_P2P_DISABLE=1`,
  Weg über Host-SHM).
- 27B-Prefill 29k: Produktion 38,5–38,7 s = 754–758 tok/s (mehr ist besser); Decode bleibt bei 39 ms = 56,0 tok/s.
- Geräteattribute 17.10. (Treiber 580.178.04, CUDA 13.0, abgefragt ohne Kontext): alle fünf Karten melden
  `CAN_USE_64_BIT_STREAM_MEM_OPS=1`, `CAN_USE_STREAM_WAIT_VALUE_NOR=1`, `CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM=1`,
  `HOST_REGISTER_SUPPORTED=1`, `CAN_FLUSH_REMOTE_WRITES=0`. Kopier-Engines: RTX 8000 3, V100 7. Damit gehen
  `cuStreamWaitValue32`/`cuStreamWriteValue32` auf Volta und Turing (wie beim PLE-Offload schon im Einsatz).

Gerechnet, nicht gemessen:

- Je 2048er-Häppchen 128 Allreduces (64 Schichten × 2) zu 2048 × 5120 × 2 B = 20 MiB. Bei 29k: ~38 GB, ~18 s von
  38,7 s; die Rechnung ist damit ~21 s.
- Pro Schicht und Rang tragen die zeilenparallelen GEMMs (o_proj/out_proj und down_proj) ~31 % der GEMM-MACs
  (GDN-Schicht: 60,3 von 191,3 M MACs je Token).

## Warum die vLLM-Mittel nicht greifen

- `enable_sp` und `enable_dbo` stehen in `vllm/config/vllm.py` auf der Liste „unsupported“ des V2-Runners.
- Async TP (`fuse_gemm_comms`, `collective_fusion.py`) braucht torch-Symmetric-Memory, also P2P.
- DBO ist in `vllm/config/vllm.py` zusätzlich an DeepEP-all2all gebunden (`use_ubatching` verlangt
  `deepep_low_latency`/`deepep_high_throughput`); die Umschaltpunkte sitzen nur im MoE-all2all.
- 1Cats `awq_tile_overlap` (`linear.py` `_maybe_sm70_awq_mlp_down_tile_gemm_reduce`) kachelt down_proj mit dem
  Allreduce, läuft aber über den Custom-Allreduce (`ca_comm`, IPC/P2P) und nur für AWQ auf SM70.

## Drei Wege und ihre Obergrenzen

1. **Schneller übertragen** (Kopier-Engines statt NCCL-SHM): Beide Wege bewegen je Richtung und Karte einmal die
   Nachrichtengröße; gewinnen kann der CE-Weg nur, wenn DMA je Richtung mehr schafft als NCCLs SM-Kopien in den
   gemappten Host-Speicher. **Gemessen 17.10.** (20 MiB fp16, bitgenau): NCCL 10,0 ms = 2,09 GB/s, Kopier-Engines
   (16 Stücke, Addition auf eigenem Stream) 7,63 ms = 2,75 GB/s (V100-Paar 2,07 gegen 2,70) → Allreduce-Anteil ×0,76,
   bei 29k rechnerisch ~34 s. `NCCL_SHM_USE_CUDA_MEMCPY=1` hängt.
2. **GEMM-Kacheln mit dem Allreduce überlappen** (Prinzip wie `awq_tile_overlap`, aber ohne P2P und
   quantisierungsunabhängig): versteckt nur hinter der GEMM, die die Nachricht erzeugt. Das sind ~31 % der GEMMs, die
   wiederum ~20 % der Zeit sind; bei 9,7k also ≤ ~0,8 s von 13 s (~6 %). Als Hauptweg verworfen.
3. **Zwei Mikro-Batches verschränkt** (DBO-Prinzip): Während Hälfte A ihren Allreduce überträgt, rechnet Hälfte B die
   gleiche Schicht, und umgekehrt. Das versteckt die Übertragung hinter der *ganzen* Rechnung (Attention, GDN,
   Entpacker, GEMMs). Obergrenze bei 9,7k: max(6,8; 6,2) s plus Blasen statt 13,0 s, also gut −40 %. Bei 29k etwa
   Rechenzeit (~21 s) plus Blasen und doppeltem Entpacken statt 38,7 s.

Überlappung gemessen (zehn GEMMs der 27B-down_proj-Form neben Allreduces, „verdeckt“ = Anteil der kürzeren Seite):
NCCL selbst 80 % (V100 72 %), Kopier-Engines 97 % (V100 81 %). NCCL ist damit kein Hindernis für Weg 3; der
Prototyp startet mit NCCL, die Kopier-Engines kommen als zweiter Schritt dazu.

Empfehlung: Weg 3 als Hauptweg. Weg 1 kommt dazu, der Mikrobench trägt ihn. Läuft die Übertragung ohne SMs, wird
die Rechnung neben dem Allreduce nicht langsamer, und eine schnellere Übertragung verkleinert die Blasen.

## Weg 3 im Einzelnen

### Warum es auch bei einer einzigen Anfrage stimmt

Wir haben fast immer genau eine Anfrage im Prefill. Die Hälften sind deshalb die Token 0–1023 und 1024–2047 *derselben*
Anfrage. Hälfte A ist in jeder Schicht vor Hälfte B dran: Beide teilen einen Rechen-Stream, und A gibt erst am
Allreduce ab.

- Full Attention: B liest A's KV derselben Schicht aus dem Cache. Das ist dieselbe Lage wie beim Chunked Prefill mit
  1024er-Häppchen.
- GDN: B startet mit dem Endzustand von A derselben Schicht. Der GDN-Builder leitet `has_initial_state` aus
  `seq_lens − query_lens` ab (`compute_num_computed_tokens`, `gdn_attn.py` ~1191/1592). Mit den geteilten Metadaten ist
  das für B größer null.
- Numerik: nicht bitgleich zur Produktion. GDN-Zustand und Attention verhalten sich wie bei
  `--max-num-batched-tokens 1024`; das ist eine unterstützte Einstellung, keine Näherung. Die GEMMs laufen mit M=1024
  statt 2048. Abnahme also über Greedy-Vergleich, Qualitätsfragen und lange Prompts, nicht über Wortgleichheit.
- **Entscheidung Peuqui 17.10.:** Wortgleichheit ist keine Pflicht, Abweichungen werden aber beobachtet und
  ausgewiesen. Maßgeblich ist, dass die Antworten inhaltlich korrekt sind. Zusätzlich der Zahlentest: Die Abweichung
  zur Produktion darf nicht größer sein als die zwischen Häppchengröße 2048 und 1024.

### Vorhanden, wird wiederverwendet

- `vllm/v1/worker/ubatching.py`: `UBatchContext`, `make_ubatch_contexts`. Zwei Threads, ein gemeinsamer Rechen- und
  ein gemeinsamer Kommunikations-Stream, GPU-Events, CPU-Übergabe. Unabhängig vom Runner.
- `vllm/v1/worker/ubatch_utils.py`: `maybe_create_ubatch_slices` und `split_attn_metadata` teilen eine Anfrage in der
  Mitte (`query_start_loc`, `seq_lens`, `slot_mapping`, `block_table`).
- Signal ohne SMs, falls Weg 1 kommt: `CpuGpuSemaphore` in `ple_offload_layer.py`
  (`cuStreamWriteValue32`/`cuStreamWaitValue32`) und die Prüfung `host_result_capability` in
  `kernels/ple/host_result.py`.
- Host-Allreduce `HostReduceCommunicator` (`host_reduce.py`, `csrc/host_reduce.cu`, ≤ 256 KiB, Decode): bekommt die
  große Betriebsart, es entsteht kein neues Backend.

### Neu zu bauen

a) **V2-Runner** (`vllm/v1/worker/gpu/model_runner.py`): Reine Prefill-Batches über einer Tokenschwelle werden in zwei
   Hälften geteilt. Für jede KV-Cache-Gruppe entstehen Metadaten je Hälfte, dazu je Hälfte ein ForwardContext. Zwei
   Threads führen das Modell aus, die Ausgaben werden zusammengefügt; der MTP-Entwurfskopf läuft danach ungeteilt
   (eine Schicht). Decode, Spec-Verify und die CUDA-Graph-Größen
   (27B: 1/2/4/8) bleiben unberührt. Schalter als KernelConfig-Feld (Arbeitstitel `tp_prefill_overlap` mit `enabled`
   und `min_tokens`), keine `VLLM_*`-Env. Die Sperre „dual batch overlap“ in `vllm/config/vllm.py` bleibt stehen, weil
   das ein eigener Schalter ist und nicht DBO.
b) **Umschaltpunkt im TP-Allreduce** (`GroupCoordinator.all_reduce` bzw. Custom-Op `vllm.all_reduce`): Bei aktivem
   Mikro-Batch wird `compute_done` aufgezeichnet, dann auf den Kommunikations-Stream gewechselt, der Allreduce
   gestartet und `comm_done` aufgezeichnet. Danach wird an die andere Hälfte abgegeben; beim Weiterlaufen wartet der
   Rechen-Stream auf `comm_done`. Das gilt generisch für jedes Modell, das über `RowParallelLinear` bzw.
   `tensor_model_parallel_all_reduce` reduziert, auch für den MoE-Allreduce von Flash-Next TP2×PP2.
c) **Align-Modus** (Präfix-Cache + MTP setzen `mamba_cache_mode="align"`, `models/config.py` ~526): Hälfte B muss den
   GDN- und Conv-Zustand dort lesen, wohin A geschrieben hat, nicht im Eingangsslot des Schritts. Bs Lese-Slot ist
   also der Schreib-Slot des Schritts. Das ist genau zu prüfen (`gdn_attn.py`, `v1/worker/gpu/mamba_align.py`).
   Ebenso die Prefill-Metadaten von FA2 und `causal_conv1d`.
d) **Lebensdauer über zwei Streams**: Allreduce out-of-place, Ein- und Ausgabe mit `record_stream` bzw. so allokiert
   wie bei DBO, damit der Caching-Allocator nichts früh wiederverwendet.
e) **Threads unter torch.compile**: Der 2048er-Prefill läuft stückweise kompiliert ohne Graph. DBO gibt in V1 schon
   aus einer Custom-Op mitten im kompilierten Stück ab; das muss hier genauso tragen.

### Kosten und Risiken

- NVFP4-Prefill entpackt jede Schicht je Aufruf in einen fp16-Puffer (`qpn/nvfp4_dequant.py`). Mit zwei Hälften
  geschieht das zweimal: ~7 % werden ~14 % der Rechenzeit, die aber selbst mit überlappt wird. Im Prototyp per nsys
  messen.
- NCCL-SHM belegt für seine Kanäle SMs. Die Rechnung daneben wird etwas langsamer; der Mikrobench misst das als
  „verdeckt %“ für NCCL und für den CE-Weg.
- 256 CPU-Übergaben je Häppchen (128 Allreduces × 2 Hälften). Gegen 2,7 s je Häppchen ist das klein, im Profil
  bestätigen.
- PP (TP2×PP2): Die Mikro-Batches laufen innerhalb der Stufe, die Zwischentensoren werden vor dem Senden
  zusammengefügt.
- CE-Weg (falls Weg 1): Prozessübergreifend ist das Signal ein `WriteValue32` nach der D2H-Kopie auf demselben Stream.
  Ohne `NO_MEMORY_BARRIER` sorgt die Barriere dafür, dass die Daten vor dem Flag sichtbar sind. Wiederverwendung des
  Slots über ein zweites Quittungs-Flag. `CAN_FLUSH_REMOTE_WRITES=0` ist hier egal, weil kein RDMA im Spiel ist. Der
  Mikrobench (ein Prozess, Events) deckt diesen Signalweg nicht ab; das sichert ein Test im Prototyp ab.

### Schritte

1. Mikrobench auf RTX-Paar (0,2) und V100-Paar (1,3): NCCL mit Produktions-Env und Varianten
   (`NCCL_SHM_USE_CUDA_MEMCPY=1` u. a.), CE mit 1/4/16 Stücken, jeweils mit Überlappungstest.
2. Prototyp nur für reine Prefill-Batches, 27B TP2, Align-Modus. Test gegen den heutigen Pfad: Logits-Abweichung im
   Rahmen von `--max-num-batched-tokens 1024`.
3. nsys-Profil 9,7k: Überlappung und Kosten belegen.
4. A/B 27B und Flash-Next TP2×PP2 bis 250k: Prefill als s und tok/s, Decode als ms und tok/s, Greedy und Qualität
   händisch.
5. PR an 1Cat (hilft jedem Aufbau ohne NVLink/P2P).
