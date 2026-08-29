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

## Protokoll

- 22:15 Proben-OOM-Retry (neu) hat Gitter-k=6 gerettet (GMU 0,95).
- 22:40 py-spy 0.4.2 einsatzbereit (setcap + ptrace_scope=0 durch Peuqui).
- 23:31 Baseline geerntet, ki abgemeldet, Chrome beendet, GPUs frei.
- E1 startet: k=2 RTX TP2 @65k mit VLLM_SM70_MTP_PROFILE=1 + py-spy.
