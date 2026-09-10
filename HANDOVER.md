# Übergabe — Stand 10.09.2026

**Betriebsstand steht in `STAND.md`. Damit anfangen, nicht mit diesem Dokument.**
Hier steht nur, was als Nächstes ansteht und was du über den letzten Tag wissen
musst, um nicht dieselben Wege noch einmal zu gehen.

---

## Was sich geändert hat: die RTX 8000 hat die V100 überholt

**69,13 → 77,13 tok/s** auf 2× RTX 8000 bei DFlash2 k=7, in zwei Schritten,
Text-SHA in jedem Lauf unverändert `0106659946c064b1`:

| Schritt | RTX 8000 | V100 |
|---|---:|---:|
| Ausgangslage 09.09. | 69,13 | 74,09 |
| Block-Pack der Aktivierungen (`STAND.md` Punkt 8) | 72,72 | 74,02 |
| quantisierter Entwurfskopf (`STAND.md` Punkt 10) | **77,13** | **76,33** |

Die RTX liegt damit **vor** der V100, obwohl sie 34 % weniger Bandbreite hat.
Beide Änderungen sind bitgleich belegt — der Text hat sich in keinem Lauf
bewegt.

**Der Auftrag der vorigen Übergabe war falsch.** Er lautete: sm75-Variante der
Skinny-Kernel bauen, weil Turing auf `m16n8k8` ausgelegt ist und Voltas
`m8n8k4` nur ausführt. Gemessen sind beide Formen auf der RTX 8000 bei gleicher
FLOP-Zahl **ununterscheidbar** (45,2 gegen 45,0 TFLOPS). Es gibt keinen
MMA-Formnachteil auf Turing. Wer das noch einmal angeht, baut gegen eine
widerlegte These.

Die Ursache war die **Zeilenstreuung der Aktivierungen**: acht Cachezeilen für
256 verschiedene Bytes, weil `m8n8k4` einem Warp acht x-Zeilen je k-Gruppe
abverlangt und die in `x[M][K]` `K*2` Byte auseinanderliegen. Turings 96-KB-L1
leidet darunter drei- bis fünfmal stärker als Voltas 128-KB-L1. Der Fix ist ein
Block-Pack (`skinny_pack_x8`), reine Datenlage, bitgleich.

---

## Auftrag

### 1. Geometrie-Tabelle — ENTSCHIEDEN am 10.09.: bleibt, wie sie ist

`_QPN2_TABLE` im Shim trägt „graph-mode winners (qpn_graphmatrix 2026-08-17)"
— **auf der V100 vermessen, auf Turing unverändert angewandt**. Neu vermessen
am 09.09. mit dem gepackten Kernel:

| Form | ausgeliefert | besser auf beiden Karten | nur Turing besser |
|---|---|---|---|
| `1536,5120` | (16,2) | **(16,1)**: RTX 1,07× / V100 1,02× | (8,1): RTX 1,18×, V100 0,99× |
| `4352,5120` | (16,2) | **(16,1)**: RTX 1,08× / V100 1,03× | (8,1): RTX 1,16×, **V100 0,91×** |

Alle Formen mit K=5120 bestätigen ihre bisherige Einstellung (1,00–1,02×).

**Der Haken:** `nacc` UND `SPLITK` ändern beide, welche Teilsummen wo gebildet
werden. Beide Reihenfolgen sind in fp32 gleich gültig, aber das Ergebnis wäre
**nicht mehr bitgleich** — und die Bitgleichheit ist der Prüfanker dieses
ganzen Projekts.

**Entscheidung Peuqui, 10.09.: die Bitgleichheit wird nicht geopfert.** Damit
ist die Geometrie-Tabelle als Hebel vollständig vom Tisch, nicht nur der
`nacc`-Teil. Der Preis ist beziffert: die beiden Formen sind ~26 % der
Gewichtsbytes, bei ~16 % Gewinn also ~3,6 % der `qpn2`-Zeit und **rund 0,5 %
end-zu-end**. Ein bitgleicher Ersatzweg existiert nicht — mehr Arbeit je CTA
halbiert das Gitter und verschlimmert die Wellen-Quantisierung (`STAND.md`
Punkt 9). **Nicht neu aufrollen.**

### 2. Der Extension-Build ist auf jeder Karte sm_70

`_get_skinny_ext()` in `fork_patches_150/marlin.py` übergibt
`-gencode=arch=compute_70,code=sm_70` in `extra_cuda_cflags`, und damit
ignoriert `torch.utils.cpp_extension` `TORCH_CUDA_ARCH_LIST` vollständig
(`_get_cuda_arch_flags()` gibt `[]` zurück, sobald ein Extra-Flag „arch"
enthält). Die RTX 8000 fährt Volta-SASS über die Cubin-Kompatibilität
derselben Major-Version.

**Gemessen kostet das wenig** — ein echter sm_75-Build bringt bei M≤4 ein bis
sechs Prozent und bei M=8 nichts —, aber die Sache gehört geradegezogen: das
Gencode aus `torch.cuda.get_device_capability()` des Workers ableiten. Kleiner
Eingriff, ein Build je Architektur, Cache getrennt halten (der Ninja-Cache ist
nur über den Extensionsnamen verschlüsselt).

### 3. Danach: die verbleibenden Posten im Decode-Profil

Decode-Profil vom 10.09. mit gepacktem Kernel (`prof_dflash.sh packed 0,2`),
ein Rang, 5.395 ms GPU-Zeit im Fenster, aufgeschlüsselt bis 96 %:

| Posten | ms | Anteil | Stand |
|---|---:|---:|---|
| `ncclDevKernel_AllReduce` | 1.714 | 31,8 % | **größter Posten, nie untersucht** |
| `skinny_nvfp4_qpn2` | 1.303 | 24,2 % | fertig, 87–98 % der Dachlinie |
| `skinny_fp8_qpn8` | 889 | 16,5 % | geprüft, DRAM-gebunden, nicht anfassen |
| `cutlass_75_wmma…s161616gemm_f16_16x16` | 291 | 5,4 % | **unidentifiziert** |
| `turing_fp16_s1688gemm` | 273 | 5,1 % | war der fp16-Entwurfskopf, erledigt |
| `fused_sigmoid_gating_delta_rule` | 180 | 3,3 % | |
| `ncclDevKernel_AllGather` | 178 | 3,3 % | |
| `skinny_pack_x8` | **29** | 0,5 % | der Pack selbst — spart 310, kostet 29 |

Zwei Fäden sind offen:

- **Das AllReduce.** 102 µs für 80 KB Nutzlast ist latenz-, nicht
  bandbreitendominiert; ohne P2P läuft alles über Host-Staging. Billigster
  erster Versuch sind die NCCL-Schalter (`NCCL_ALGO`, `NCCL_PROTO`,
  Puffergrößen) — reine Env-Experimente, keine Codezeile. Zum Vergleich:
  unter MTP kostet dasselbe AllReduce nur 58,7 µs je Aufruf, weil k=3 die
  halbe Nutzlast bedeutet.
- **Der Cutlass-fp16-GEMM je Schicht.** In BEIDEN Profilen vorhanden (291 ms
  bei DFlash2, 351 ms bei MTP), also im Zielmodell und nicht im Entwurf; die
  Aufrufzahl entspricht **einem je Schicht und Vorwärtsschritt**. Ein kleiner
  fp16-GEMM, der an den Skinny-Kerneln vorbeiläuft — was genau, ist offen.

- **`skinny_fp8_qpn8`, 889 ms.** Geprüft und für erledigt befunden: FP8 liest
  doppelt so viele Bytes je Gewicht, der Kernel steht mit 86 % näher an
  Turings Obergrenze als der V100-Kernel an seiner (81 %). Der Block-Pack
  brachte dort 1,05× auf der RTX und kostete 4 % auf der V100 — **nicht
  noch einmal versuchen.**

---

## Wo du NICHT weitersuchen solltest

**Fünf** Erklärungen für die Turing-Lücke sind gemessen widerlegt:

- **Die MMA-Form.** Neu am 09.09.: `m8n8k4` und `m16n8k8` sind auf der RTX
  8000 bei gleicher FLOP-Zahl gleich schnell. Werkzeug `mma_probe.cu`, das
  auch beide Fragment-Layouts gegen eine CPU-Referenz verifiziert
  (max|err| = 0) — der Ersatz für das verschollene `mma8_probe.cu`.
- **QPN8-Rerank für den Kandidaten-TopK.** Kein `topk`/`sort`-Kernel in den
  Top-14 des Decode-Profils.
- **Das AllReduce als *Erklärung der Lücke*.** 1.697 gegen 1.530 ms bei
  nahezu gleicher Aufrufzahl, beide Kartenpaare hängen identisch an. Grundlast
  auf beiden Seiten (Peuqui). Als eigenständiger Posten bleibt es interessant,
  siehe oben.
- **Die Compile-Vorgaben** (`SM70TUNE=1`): +0,5 %.
- **Marlin als Skinny-Ersatz:** 37 % langsamer auf Turing.

**TurboMind ist kein Rebuild, sondern eine Portierung.** Der Quelltext hat
Turing-Instruktionen und eine `config_sm75_s16816.h`, aber es existieren nur
Volta-Kernel-Instanzen (`sm70_884_4/8/16.cu`); `vllm/_C.abi3.so` trägt 39
ELF-Einträge, alle `sm_70`, kein PTX.

**Nsight Compute läuft jetzt** (seit 10.09.). `ncu` als normaler Nutzer ist
freigeschaltet — `/etc/modprobe.d/nvidia-profiling.conf` setzt
`NVreg_RestrictProfilingToAdminUsers=0`, Kontrolle:
`grep RmProfilingAdminOnly /proc/driver/nvidia/params` muss `0` zeigen (der
Treiber führt den Parameter unter DIESEM Namen, nicht unter dem der conf).

**Die richtige Währung ist `l1tex__data_pipe_lsu_wavefronts_mem_lg.sum`, nicht
die Sektoren.** Beim Block-Pack blieben Anfragen, DRAM-Bytes und Sektoren
gleich; nur die Wavefronts fielen um Faktor 3,2, und die Zeit folgte ihnen.
Wer hier Sektoren zählt, sieht nichts.

Auch ohne Profiler bleibt der Weg gültig, der die Ursache gefunden hat:
dieselbe Quelle zweimal bauen, **einen** Faktor ändern, und eine Zelle
mitmessen, in der die Änderung wirkungslos sein MUSS. Bei M=1 lag der
Kontrollwert bei exakt 1,00× — daran hing der ganze Beweis.

---

## Offene Fäden (unverändert)

1. **QUASAR-QAT ist in diesem Stack unbrauchbar** — `STAND.md` Punkt 7.
   Gemessen wird auf RadixArk.
2. **DeepSeek-V4 nach dem GDN-Umstieg nicht nachgemessen** — `STAND.md`
   Punkt 3. Argument, keine Messung; ein Gegentest wäre billig.
3. **PLE-Überlaufkaskade** (vier Stufen) — `STAND.md` Punkt 6. Ausdrücklich
   erst anzugehen, wenn die Turing-Arbeit veröffentlicht ist.
4. **`prof_prefill.sh` ist nicht lauffähig** — `STAND.md` Punkt 11.
5. **Zwei Upstream-Änderungen beim nächsten Wheel mitziehen:**
   `custom_all_reduce.py` und `platforms/cuda.py`, Hinweise in
   `fork_patches_150/STATUS.txt`.
6. **Vier PR-Kandidaten für 1Cat:**
   - Guard in `compute_candidates` (macht DFlash2 mit jedem Checkpoint
     fahrbar, dessen LM-Head mitquantisiert ist),
   - `_use_sm70_bf16_emulation` auf „< sm80" am Gerät des Workers,
   - **neu:** der Aktivierungs-Block-Pack für die QP-N-Kernel. Er ist
     bitgleich belegt, hilft Turing 1,39–1,45× auf dem Kernel und schadet
     Volta nicht — und er behebt etwas, das jeden Nutzer dieser Kernel auf
     einer Turing-Karte trifft.
   - **ERÖFFNET als #592 (10.09.):** die fusionierte Kontext-K/V in
     `DFlashQwen3Model._build_context_kv_buffers` greift am `quant_method`
     vorbei und macht DFlash2 mit JEDEM quantisierten Entwurfskopf unfahrbar.
     Kartenunabhängig, gemessen +6,1 % RTX, +3,1 % V100, Text unverändert.
     https://github.com/1CatAI/1Cat-vLLM/pull/592

   Vor einer Meldung: `AGENTS.md`-Pflichtregeln beachten.
