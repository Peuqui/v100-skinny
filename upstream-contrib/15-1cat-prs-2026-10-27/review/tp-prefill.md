# Unabhängige Prüfung TP-Prefill-PR (27.10. ~12:05): NOT READY → READY AFTER FIXES

BLOCKER
1. A/B-Zuordnung: alter Arm a1ac0c89d enthielt schon f9ac81b6b (Turing-Attention); −42/−35/−33 % = nur Commits 2–5
   (88d25b8bd, c1816294c, afe87abd0, f21d1c564). Commit 6 (Flash-V100-Fix) in keinem Arm. Commit 1 eigene Belege: nsys-Einzelboots
   150 033 Token 267,91 → 207,87 s (kein A/B). „96 von 208 s“ stammt aus dem Profil MIT neuem Kern; mit FA2 ≈ 268 s (FA2 ≈ 108 s, AR ≈ 95 s).
   Text: „reference a1ac0c89d, changed f21d1c564“.
2. Copy-Engine-Modus: host_reduce.py:122-130 schickt Nutzlast > max_bytes immer zum CE-Op, der beim Capture TORCH_CHECKt
   (host_reduce.cu:355). Mit üblichen Capture-Größen (32…512) Startabbruch. Prod verdeckt (Capture 1,2,4,8). Fix: beim Capture
   (torch.cuda.is_current_stream_capturing) Nutzlast > max_bytes wie bisher an NCCL; Test mit Capture dieser Größe.
3. Überlappung ohne Modellprüfung: Qwen3-VL-Deepstack (_get_deepstack_input_embeds[:num_tokens]) liest in Hälfte 2 falsche Zeilen
   (still falsch); MoE über FusedMoEModularKernel ohne async prepare → assert not dbo_enabled() (Absturz → Hänger, s. 6);
   all_reduce_sum2 + fusionierte AR+RMSNorm laufen am Overlap-Haken vorbei (zwei Kollektive auf einem Kommunikator aus zwei Streams).
   Fix: nicht unterstützte Modelle beim Start ablehnen (MoE, Deepstack/Multimodal mit tokenindizierten Puffern), sum2 durch
   dbo_run_collective_overlapped, fusionierte AR+RMSNorm bei dbo_overlaps_collectives() Fehler; Text „Format: any“ → dichtes TP2.
SHOULD-FIX
4. Turing-Prefill-Kern: Launcher nur Seiten 16/32/64 (cu:518) + 32-Bit-Offsetgrenze (cu:463); _use_turing_prefill (flash_attn.py:711)
   prüft beides nicht → Absturz statt FA2 (z. B. --block-size 128). Prädikat + Unit-Test.
5. docs/design/architecture/runtime_reference.md veraltet (KernelConfig-Anker #L645/#L1116 statt 653/1130) → neu erzeugen.
6. Ausnahme in einer Hälfte (prefill_overlap.py:215-228) → anderer Thread hängt in _cpu_yield, Hauptthread in join → Abbruch-Flag.
7. Text: „changes here do not use Triton“ falsch (Commit 5 = Triton-Kachel; beide Arme mit sm7x-Triton).
8. Text: „CUDA graphs“ gilt nicht für CE-Allreduce (verweigert Capture) → je Punkt angeben.
9. DCP und Encoder-Decoder beim Start ablehnen.
NIT: kernel-config-String mit triton_sm7x-Feld (auf diesem Zweig nicht vorhanden) erklären; 184k 54,3–54,4 ms; 250k tok/s 40,8 / 41,0–41,1;
split-KV-Spalte „production“ statt „one split“; 8 MiB = 8 192 000 B (800×5120×2); „two boots per arm + one uncounted“; „22,5 vs 61 TFLOPS“;
96/64-Token-Zeilen „after 150k“; Speicher „new 48 285 / old 47 979“; Commit 95393bf4f Kern-Zuordnung (+37 % = fp8-dequant 10240) und
„production“→„baseline“ (auch dbdde81b6); Signed-off-by bei den sechs Commits; eda4889e1 passt besser in MLA-PR; #1245 im Duplikat-Absatz;
#1216 vorher force-pushen; copy_engine_max_bytes > max_bytes prüfen; Stream-Mem-Ops bei open_channel prüfen; cudaFuncSetAttribute je
Gerät cachen; Profillauf ohne Overlap (+306 MiB Comm-Pool) erwähnen.
Geprüft und in Ordnung: Stream-Synchronisation, Align-Split, token_range_batch, CE-Protokoll, Gerätegates, Revert-Prüfung, PR-Zustände.
