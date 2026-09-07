# Jagd: MTP-Spekulations-Einbruch bei langem Kontext

Stand: 2026-08-29 ~22:30 (Nachtschicht, autonome Jagd nach Baseline-Ende)

## Befund (aus der Baseline-Kalibration, 27B TP2-RTX @262k, Langpunkt ~28,8k)

- k=0: lang 29,8 tok/s (33,5 ms/Step)
- k=1: lang 14,8 tok/s bei **86 % Akzeptanz** — Step-Kosten ~96 ms = **~3×**
  gegen k=0, obwohl: Drafter = 1 Schicht von 64, Verify = 2-Token-Pass
  (memory-bound, KV einmal gelesen). Erwartbarer Overhead: <15 %.
- Trend: je groesser k, desto schlechter lang (k=7: 16,4 → k=5: 18,2);
  kurz hilft Spekulation immer (k=1: 59,2 vs. 43 bei k=0).
- Gitter (Drafter auf V100/XQA): k=7 lang 20,1, accept 29 % — auch dort
  Verlust gegen k=0 (29,5). V100-pure-Sweep laeuft noch (Diskriminator!).
- => KOSTENPROBLEM, kein Akzeptanz-Kollaps. llama.cpp gewinnt mit
  demselben MTP-Kopf an BEIDEN Enden → Implementierungs-Artefakt.

## Hypothesen (Verdaechtige)

H1 Drafter-Attention-Metadata-Rebuild pro Draft-Step
   (build_per_group_and_layer_attn_metadata, O(blocks)=O(ctx/16) CPU?)
H2 Verify-/Sampling-Seite im Model-Runner: O(ctx)-Operation pro Step
   (Batch-Expansion, Slot-Mapping, Logits-Gather)
H3 CUDA-Graph-Miss bei langem Kontext (Graph nur fuer kleine
   Batch-Deskriptoren; lang → eager Drafter/Verify)
H4 Scheduler/async: TP2-Pfad laeuft OHNE --async-scheduling (nur PP
   setzt es) → CPU-GPU-Bubbles skalieren mit Metadata-Groesse
H5 Drafter-KV-Verwaltung: Draft-KV over full ctx neu geschrieben/kopiert

## Werkzeuge

- **VLLM_SM70_MTP_PROFILE=1** (+ _INTERVAL, Default 16): eingebautes
  Phasen-Profiling im Drafter (first_forward, loopN_forward,
  loopN_sample, metadata_cpu) — loggt Worker-seitig.
- DFLASH_DDTREE_WORKER_PROFILE nur fuer dflash_ddtree (nicht mtp).
- py-spy 0.4.2 in .venv-sm70-130/bin (22:40 installiert, Peuqui hat
  setcap cap_sys_ptrace + ptrace_scope=0 gesetzt; Attach am laufenden
  Worker verifiziert).
- tests/rtx_fa2_sweep.py-Muster fuer eigene Boots (explizite PIDs).

## Experimente (nach Baseline-Ende, GPUs frei)

E1: k=1 RTX-TP2 @65k, MTP_PROFILE=1 → Kurz-Decode vs. Lang-Decode
    (Prefix-Cache-Methodik). Vergleich der Phasen-Timings kurz/lang.
    Erwartung: EINE Phase explodiert mit ctx → Taeter-Komponente.
E2: py-spy dump --pid <Worker> waehrend Lang-Decode (CPU vs. GPU-Wartezeit).
E3: Ableitungen: enforce_eager-Drafter-Vergleich; Drafter-Backend
    FLASH_ATTN vs. TRITON; ggf. --async-scheduling auf TP2.

## Leitplanken

- Kein AIfred-Code, keine Commits/Pushes ohne Kommando.
- venv-Aenderungen nur reversibel mit Backup, hier dokumentieren.
- Prozesse nur per expliziter PID beenden; GPU-Check vor jedem Boot.

## Baseline KOMPLETT (23:30, volle Matrix: baseline-matrix-2026-08-29.txt)

Lang-Decode @28.843 (k=0-Referenz): RTX 29,8 / Gitter 29,5→29,3 / V100 30,3.

- **V100 pur: Spekulation GEWINNT bei lang** — k=2: 38,2 (+26 %),
  k=3: 37,3, k=4: 31,7, k=1: 30,8. RTX: ALLE k verlieren (Best k=2:
  20,9 = 0,70x). Gitter exakt dazwischen (k=2: 25,9).
- Akzeptanz je k ueber alle Topologien identisch (97 % bei k=2!) —
  Taeter ist der VERIFY-/Step-Kostenpfad auf sm75, nicht der Drafter-
  Inhalt. Bei k=2/97 % kostet der RTX-Spec-Step ~1,8x dessen, was er
  auf V100 kostet.
- **Hauptverdaechtiger:** `VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH=1`
  wird auf sm75 ignoriert (Log: "Ignoring ... platform is not SM70") —
  1Cat hat Graph-Kompilierung fuer den Volta-Spec-Verify gebaut; das
  FA2-sm75-Pendant fehlt → Verify auf RTX vermutlich eager/piecewise
  mit Launch-Overhead x 64 Schichten x Step.
- Zweitraetsel: k=1-Delle auf RTX/Gitter (14,8/19,1 — unter k=2 trotz
  86 % Akzeptanz); auf V100 kaum (30,8).
- Siegerregel-Praxistest bestanden: Gitter gewinnt Betriebspunkt via
  Prefill-Patt-Brecher (29,3~29,8, Prefill 836 vs. 510); Speed-Variante
  V100 k=2 38,2 @151.200 persistiert.

## TAETER GEFUNDEN (00:0x): Kein Split-KV fuer den Spec-Verify

Kette (E1+E2+Mikrobenchmark):
1. py-spy: 83 % der Worker-Zeit = GPU-Warten (synchronize) → GPU-bound,
   kein CPU-/Launch-Overhead. (Nebenbefund: _sm70_qpn8_indices ~9 % CPU.)
2. MTP-Profiler: Drafter nur 10,7 ms/propose (7,5 first_forward) —
   von ~127 ms Step frisst der VERIFY ~116 ms.
3. Kernel-Mikrobenchmark FA2 paged varlen @31k (4/1 Heads, hdim128):
   q=1: 0,134 ms — q>=2: 2,6 ms = **20x**. Ursache in flash_api.cpp:
   "Only apply split-k for decoding" — set_params_splitkv laeuft NUR im
   seqlenq_ngroups_swapped-Fall (q=1); fuer q>1 paged varlen ist
   num_splits<=1 erzwungen → 1 CTA je (batch, head) laeuft SERIELL
   ueber ~2.000 KV-Bloecke (4 CTAs auf 72 SMs).

Fix 1 (flash_api.cpp): Split-k auch fuer paged varlen mit
max_seqlen_q <= 64 (Verify; Prefill q>64 bleibt unangetastet).
→ Mikrobenchmark q>=2: 2,6 → 0,31 ms (**8,5x**).

Dabei ZWEI weitere Upstream-Bugs im Combine-Kernel aufgedeckt (der
Split-Pfad lief upstream nie mit echtem varlen):
Fix 2: O-Write nutzte batch_idx*o_batch_stride — im varlen-Fall ist
  der Stride ungesetzt (O ist ueber cu_seqlens_q gepackt): zweite
  Sequenz eines Batches → NaN/Fehlschreiben. Jetzt cu_seqlens-Offsets
  + Zeilen-Guard fuer kuerzere Sequenzen.
Fix 3: Unpadded-LSE-Write nahm uniforme Sequenzlaengen an; jetzt
  cu_seqlens-Packung. Plus Accum-Vorinitialisierung (-inf/0) fuer
  Leer-Split-Slots kuerzerer Sequenzen.

Erwartung E2E: Verify-Attention 165 → ~20 ms/Step (64 Schichten) →
k=2 lang von 20,9 Richtung ~29-35 tok/s auf RTX. Beide Funde
(Split-Enable + Combine-varlen) sind upstream-PR-Material — Ampere
duerfte am seriellen Verify ebenfalls leiden, nur mildert es dort
schnellere serielle Blockarbeit.

## E2E-BESTAETIGUNG (00:20, Fix live im 1Cat-venv, Commit 40e3746)

27B TP2-RTX @65k, Langpunkt 31.469 tok (200er-Decode / 600er-Decode):

| k | kurz | lang (200) | lang (600) | vorher lang |
|---|---:|---:|---:|---:|
| 2 | 70,6 | 32,8 | 44,7 | 20,9 |
| 3 | 75,4 | 33,2 | 43,3 | 20,5 |

Spekulation gewinnt auf RTX jetzt an BEIDEN Enden (wie llama.cpp).
RTX k=2/3 lang schlaegt die bisherige V100-Bestmarke (38,2). Der
600er-Wert liegt ueber dem 200er (Akzeptanz auf laengerem repetitivem
Auslauf + Amortisierung des Anlaufs).

KONSEQUENZ: 27B-Neukalibration unter dem Fix noetig — die komplette
k-Landschaft hat sich gedreht; die Lang-Regel wird jetzt ein k>0
kueren. Offene Nebenbaustellen: k=1-Delle, _sm70_qpn8_indices (~9 %
CPU/Step), V100-Verify koennte vom selben Split-Fix profitieren
(XQA-Pfad separat, nicht untersucht).

## Protokoll

- 22:15 Proben-OOM-Retry (neu) hat Gitter-k=6 gerettet (GMU 0,95).
- 22:40 py-spy 0.4.2 einsatzbereit (setcap + ptrace_scope=0 durch Peuqui).
- 23:31 Baseline geerntet, ki abgemeldet, Chrome beendet, GPUs frei.
- E1 startet: k=2 RTX TP2 @65k mit VLLM_SM70_MTP_PROFILE=1 + py-spy.

## AMPERE-GEGENBEWEIS (2026-08-30, RTX 3090 Ti / sm86)

Unveraenderter Upstream-Kernel (vLLM 0.13.0 aus dem Paket, torch 2.9.0,
WSL2 auf Aragon), paged varlen Attention bei 31.488 Token KV, hdim 128:

| Koepfe (Q/KV) | q=1 | q>=2 | Faktor |
|---|---:|---:|---:|
| 4 / 1 | 0,065 ms | ~1,44 ms | **22,8x** |
| 8 / 2 | 0,083 ms | ~1,38 ms | **16,5x** |
| 32 / 8 | 0,189 ms | ~1,37 ms | **7,3x** |

Die ABSOLUTE Zeit bleibt bei ~1,4 ms — unabhaengig von Kopfzahl und
q-Laenge. Signatur des seriellen KV-Durchlaufs: die Arbeit haengt nur an
der Kontextlaenge. Mehr Koepfe = mehr CTAs = kleinerer Faktor, aber der
Einbruch bleibt selbst bei 32 Q-Koepfen (Llama-8B-Geometrie) beim
Siebenfachen.

Quelltext-Pruefung am aktuellen Upstream-main (2026-08-30): die Sperre
steht unveraendert
(`} else if (paged_KV) { STD_TORCH_CHECK(num_splits <= 1, ...)`,
Kommentar "Only apply split-k for decoding").

=> Der Befund ist NICHT Turing-spezifisch. Messskript:
tools/ampere_verify_bench.py (laeuft gegen jedes installierte vLLM).

## OPTIMIERUNGSRUNDE 27B (2026-08-30 nachmittags, tools/grid_stage_test.py)

Alle Laeufe: Gitter TP2xPP2, k=2, 262.144 Kontext, Langpunkt 31.469 tok.

### E-A: Drafter-Platzierung — HYPOTHESE WIDERLEGT

| Stufenordnung | kurz | Prefill | lang |
|---|---:|---:|---:|
| RTX vorn (38,26), Drafter V100 = produktiv | 67,4 | 835 | 33,1 |
| V100 vorn (26,38), Drafter RTX | 64,7 | 810 | 32,5 |

Der Drafter auf den RTX bringt nichts (-2 bis -4 %). 1Cats XQA-Drafter
auf Volta ist gut genug; die Umkehrung verschiebt zudem die fruehen
Schichten auf die langsameren Karten. Konvention "schnellste Klasse
zuerst" bleibt — jetzt gemessen, nicht nur angenommen.

**Nebenprodukt (2 Fork-Patches, noetig damit der Test ueberhaupt bootet):**
vLLM leitet die FlashAttention-Version GLOBAL von Geraet 0 ab
(`get_flash_attn_version` -> `get_device_capability()`, Default 0) und
unser sm75-Gate ebenso (`has_device_capability(75)`). In einem
heterogenen Gitter diktiert damit die schwaechste Karte auf Position 0
den Kernel fuer ALLE Stufen; mit V100 auf 0 bricht der RTX-Worker mit
"FlashAttention version not detected". Fix: beide Stellen fragen
`torch.cuda.current_device()`. Verifiziert ohne Regression auf der
produktiven Ordnung (66,4/835/32,9 vs. 67,4/835/33,1 = Rauschen).
Dateien: fork_patches/fa_utils.py, fork_patches/flash_attn_interface.py,
Backups in backups/2026-08-30-fa-version-per-device/.

### E-B: Chunk-Groesse und Blockgroesse

| Konfiguration | kurz | Prefill | lang |
|---|---:|---:|---:|
| 2048 / Block 16 (Ausgangspunkt) | 67,4 | 835 | 33,1 |
| 4096 / Block 16 | 67,3 | 858 | 33,0 |
| 8192 / Block 16 | 66,1 | 817 | 32,8 |
| **4096 / Block 32** (Kohaerenz 3/3) | 67,5 | **863** | 33,1 |

+3,5 % Prefill, Decode unveraendert. Der Abfall bei 8192 ist kein
Messfehler: im PP-Betrieb lebt der Prefill von mehreren gleichzeitig
fliegenden Chunks — zu grosse Chunks reduzieren die Ueberlappung.
Kohaerenz bei der Bestkonfiguration 3/3 geprueft (Blockgroesse greift in
die Paged-Geometrie und den Split-KV-Pfad ein, deshalb Pflichtcheck).

Offen als grosser Hebel: Turing-Tile-Tuning (hdim 128 belegt derzeit
64 KB smem = 1 CTA/SM; 64x32 waere 32 KB = 2 CTA/SM).

### E-C: Kachel-Tuning beider Architekturen (2026-08-30 abends)

Turing (FA2-Fork, JIT-Sweep tests/jit_probe/tile_sweep.py): Dispatch
hdim128 auf 64x32 (-18 % Decode/Verify, 2 CTAs/SM), Align auf 128x64
(-37 % Chunk-Prefill, Standard-Kernel-Kachel). Numerik PASS, deployed
(Backup backups/2026-08-30-fa2-tile-tuning/). End-to-end am 27B NEUTRAL
(A/B RTX-TP2: 533/33,8 neu vs. 535/33,9 alt) — Attention ist dort nicht
der Engpass. grid_stage_test.py kann jetzt PP1 (PP-Groesse folgt der
Partition).

Volta (1Cat-Quellen jetzt lokal: ~/Projekte/1Cat-vLLM/flash-attention-v100,
Sweep tools/volta_probe/): Verify skaliert LINEAR mit q (tokens-as-batch,
0,153->0,632 fuer q=1->8); XQA-Gate nur D=256, das 27B nutzt ihn nie.
Prefill-Kachel 32x176 -> 64x80 = 14,06 statt 16,10 ms (-13 %); Smem-Bilanz
empirisch 272N + 856M + 4MN Bytes, 96-KB-Wand; M muss Vielfaches von 32
sein. Restluecke zu Turing (14 vs 4 ms) strukturell (kein Double-Buffering,
Score/Out im Smem) -> 1Cat-Kontakt. Messwerte: tools/volta_attn_bench.py.

Gitter-Konsequenz: kritischer Pfad ist die V100-Stufe; Turing-Gewinne
versickern in der Pipeline. Vollstaendige Tabellen:
docs/*/benchmarks/vllm-autokalibration (AIfred-Repo).

Nachtrag E-C: Volta-Kachel 64x80 aus v1.3.0-Quellstand neu gebaut
(1Cat-Historie geprueft: D=128 nie getunt, nur Race-Fix 13.08. + Akku-Fix
16.08., beide schon im 1.3.0-Wheel; alle Performance-Commits zielen auf
D=256). Deployed mit Backup (backups/2026-08-30-volta-tile-64x80/),
Mikrobench 14,09 ms bestaetigt. E2E-A/B Gitter: 863/33,1 vs. 864/33,0 =
NEUTRAL. Fazit: beim 27B @31k ist Attention auf KEINER Architektur der
Engpass (NVFP4/QPN8-GEMMs dominieren); Kachelgewinne zahlen erst bei
langen Kontexten und D=256 (Flash-Next) ein. Quell-Clone:
~/Projekte/1Cat-vLLM (sparse, Tag v1.3.0 fuer flash-attention-v100).

### E-D: Flash-Next-Mini-Sweep (2026-08-30 spaet)

tools/flashnext_stage_test.py (Ableger des Gitter-Harnesses: MML 16384,
Langpunkt 13k, Akzeptanz aus /metrics, GMU als 8. Argument, k=0 ohne
Spec-Config). Kernbefunde: (1) GMU 0.95 drosselte Long-Decode 28,3->12,6
(Allokator-Druck QSA/Verify-Workspace, einmal harter Triton-OOM auf der
V100-Stufe; Zombie-Worker hinterliessen 23 GiB VRAM -> kill noetig).
(2) Akzeptanz 57,9 % kurz -> 19,0 % lang 13k; extern bestaetigt
vllm#47602 (Qwen3.6-27B: 64,9->39,1 %, Speedup +129 %->-51 %). Das 27B
haelt dagegen 97 % auch lang - Kopf-Robustheit ist modellspezifisch.
(3) Spek bleibt netto positiv (28,3 vs 26,4 lang) -> kein
Laengen-Abschalt-Patch. (4) MBT4096/Blk32 schaden Flash-Next (-12 %
Prefill) - 27B-Config NICHT universell; AIfred-Kalibrations-Defaults
zurueck auf neutral. Betriebspunkt-Yaml: GMU 0.93 + long_context_13k-Meta.
Offen: QSA-Triton-Kernel (Basis-Attention Flash-Next) auf sm70/75 nie
vermessen - naechster Kernel-Kandidat nach dem Kampagnen-Muster.

### E-E: QSA-Kernel-Retune fuer Pre-Ampere (2026-08-30 nacht)

Der dritte Attention-Pfad (Flash-Next-Grundmodell, Triton, "Tuned on
GB300"): Pre-Ampere nutzt den amd/-Zweig (qwen4_exp/__init__.py,
DeepSeek-Port-Lektion — ERST den richtigen Zweig instrumentieren, die
nvidia-Zwillingsdatei ist auf sm70/75 tot!). Harness tools/qsa_bench.py
(Produktionsgeometrie H12/KV1/D256, TOPK 2048).

Mikrobench 2048er-Chunk: V100 527 -> 27,3 ms (19,3x; GB300-Profil N64/W2
blieb dank 96 KB Smem stehen); RTX 199,7 -> 47,9 ms (4,2x; Smem-Klammer
hatte N64->N32 gerettet, aber W2 behalten). Decode/Verify-Zweige
(base_programs < 32) waren bereits optimal. Patch: Pre-Ampere-Tabelle
(16er-Kacheln, 4 Warps) in amd/ops/qsa.py hinter der GB300-Tabelle;
Backup backups/2026-08-30-qsa-preampere-tiles/, fork_patches/qsa_amd_ops.py.

E2E Flash-Next (2 Laeufe): PREFILL 392-448 -> 1482-1696 tok/s (~4x,
13k in 7,7 s statt 33 s), Kohaerenz 3/3, kurz unveraendert. ABER:
Lang-Akzeptanz deterministisch 19,0 -> 13,3 % (Kachel-Numerik verschiebt
den fragilen Drafter), Long-Decode 26-29 -> 22,5-25,6. Netto pro
13k-Turn (+500 tok) trotzdem ~28 s statt ~51 s. k-Frage (4 vs 0 lang)
geht an die volle Kalibration.
