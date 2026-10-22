# Übergabe — Stand 22.10.2026, 20:50 (Projekt vllm-research; Abend-Nachtrag ganz oben)

Claude Code startet in `/home/mp/Projekte/vllm-research` (Projektregeln `CLAUDE.md`, Gedächtnis unter
`~/.claude/projects/-home-mp-Projekte-vllm-research/memory/`, AI-Connect-Peer `Mini:vllm-research`).
Betriebsstand (SSOT): `STAND.md`, Nachträge 22.10. 04:25 bis 07:20 ganz oben unter „Laufzeitumgebung“.
Vorige Übergabe (21.10. 15:55) liegt in der Git-Historie dieses Repos. Grund der Übergabe: Peuqui startet nur Claude neu
(kein Rechner-Neustart). AIfred-Dienst bleibt gestoppt, kein anderer Dienst braucht die Karten (Peuqui 06:00).

## Nachtrag 22.10. 20:50 — Abend: Volta-Compiler nach Handkern-Vorbild, Familie 2 (W8A16-Decode) läuft

**Regeln von heute Abend (alle im Gedächtnis `feedback_unify_hand_kernels_in_triton.md`):** Maximum aus den Karten, so viel
wie möglich (am liebsten alles) in Triton, aber nie langsamer; der Schnellere gewinnt je Form und Karte (Route wählt nach M,
gemessene Konstanten); innerhalb der Messstreuung = gleich; 10-%-Regel gestrichen; holt Triton einen Handkern ein, sofort
umstellen. Handkern-Algorithmik 1:1 übernehmen (Kernquelle oder eigener Compiler). Programmierteil zuerst, Modell-A/B/A/B
gesammelt am Schluss (DSv4 + 27B, GLM-5.3 wenn reif, Flash-Next nur bei eigenen Pfaden); Schnellprüfungen je Schritt
bleiben. AIfred-Dienst ist aus bis Projektende (keine 06:45-Grenze). **Befehle nur über freigegebene Wege** (`env …`,
`timeout … bash skript`, Edit/Write; nie `export`, `rm`, `bash x.sh` direkt — `feedback_avoid_approval_commands.md`).
Stack bleibt gepinnt (torch 2.10 / Triton 3.6, CUDA 13 kann kein sm_70); 1Cat zuerst anbieten, sonst eigener Fork-Name.

**Triton-Baum `triton-sm7x`, Zweig `sm7x-tensor-cores` (lokal), Commits heute Abend:**
- dcf200731d Volta row/col je Operand (kWidth 1/4 nach `getOrderForMemory`); V100-GEMM 64 → 74 TFLOPS (cuBLAS 86).
- d105df138b Volta-Kachel **8x32** für Dots ≤ 8 Zeilen (instrShape [8,32], Quadpairs nebeneinander wie QPN8), Hochzieh-
  Regel aus Linear Layout (Dekodieren im Operand-Layout nur ohne Lane-Mehrfachhaltung), Faltungsfix MMAv1 (Absturz map::at
  bei M < 4 aus dcf200731d). Budget unverändert, RTX-PTX bytegleich.
- 2ddd426b2a **Batch-Dots (3D) auf Volta** (Split-K im Block über Batch-Dimension), k-Dimension der row/col-Wahl rangfest.
  PTX aller 786 Budget-Kompilate bytegleich zu d105df138b.
- **Uncommittet in `RemoveLayoutConversions.cpp`: direkter Registerweg** (`hoistConvertDotOperand`, nur 8x32-Kachel):
  Load + Adressrechnung im Dot-Operand-Layout (kein Smem, keine Barriere), B immer, A nur als reiner Load. Nur-B-Fassung
  (Runde `directload`, 20:31): alle Prüfungen 0 Fehler. **Mit A-Erweiterung (Runde `directload2`): check_skinny_volta 72
  schlechte Fälle → zuerst klären** (Log `~/.cache/bench-scripts/triton-sm7x/check_skinny_volta_directload2.log`).

**W8A16-Decode V100 (M 6, Gewichtsrotation, Faktor Triton/QPN8, kleiner ist besser), Stand Runde `directload`:**
gepackt+1 Start: wkv 0,63, wq_a 0,75 (Triton vorn); wq_b 1,09, shared_w13 1,08, wo_b 1,15–1,20, shared_w2 1,23 (dort
schwankte die Referenz 20 % zwischen Läufen → Schlussmessung jetzt A/B/A/B verschränkt, 5 Runden, Median).
ncu wo_b: alle lesen 33,67 MB DRAM; QPN8 ≈ 26 Warps/SM (78 % DRAM), Triton ≈ 6 Warps/SM (67 %) — Engpass Parallelität.
Split-K im Block (3D, S 16 = Handkern-Struktur) war durch Smem+Barriere je K-Schritt langsam; mit direktem Weg für A und B
sollte es die Lücke schließen (darum die A-Erweiterung). Restliche Barrieren im 3D-Kern: Skalen-Load ([S, BN] = Threadzahl
→ `isExpensiveLoadOrStore` hält ihn für teuer; Kandidat `<` → `<=`, dann PTX-Mengen vergleichen).

**Werkzeuge (alle in `~/.cache/bench-scripts/triton-sm7x/`):** `build_triton.sh` (Aufruf `timeout 3600 bash …`);
`run_round.sh <tag>` = `run_checks_volta.sh` (check_skinny/dots/batched auf GPU 1/3/4 + `run_dot_tests_par.sh` auf allen
5 Karten, 4:38 min statt 8 min) + `run_exp_decode.sh` (exp_w8a16_decode.py auf 3 V100 + wo_b allein); Start als Einheit:
`env XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus systemd-run --user --unit=… bash -c
'exec bash run_round.sh TAG >> run_round_TAG.out 2>&1'`; Wächter `timeout 14400 bash ~/.cache/bench-scripts/watch_job.sh
unit:… 15 <logs>` (setzt den User-Bus jetzt selbst). `compare_ptx_sets.py <cacheA> <cacheB>` (Regressionsbeleg),
`run_budget.sh <tag>` (FA/Indexer/GLM; GLM-Bench ohne Graph ist im Parallellauf unzuverlässig, Einzel-A/B mit
`run_glm_once.py`), `inspect_exp_kernel.py`/`inspect_cta_kernel.py` (TTGIR/PTX eines Tiles), `profile_exp.py` (Scratchpad,
ncu-Ziel). Familienmodul-Entwurf `w8a16_block_fp8.py` (gepackte Dekodierung, Split-K in einem Start, multiple_of) —
`bench_w8a16_fp8.py` soll ihn importieren (erst ändern, wenn kein Lauf den Bench importiert; exp importiert ihn als
„per-element“). Messverfahren: Gewichtsrotation ≥ 48 MB + vorgefüllte Warteschlange (`feedback_microbench_in_cuda_graphs.md`).

**Nächste Schritte:** (1) 72 Fehlfälle der A-Direktlade klären, Runde wiederholen, dann committen (Nachweis PTX-Mengen +
Budget). (2) Skalen-Load ohne Barriere. (3) Bench auf Familienmodul, Decode+Prefill auf V100 und RTX. (4) Prefill V100:
Volta-GEMM-Lücke 74 vs 86 TFLOPS (ncu Triton vs cuBLAS). (5) Route im Fork `triton-sm7x-routes` (Triton wo schneller,
sonst QPN8/Dequant+cuBLAS), dann MXFP8/NVFP4 und weitere Familien. (6) Erkannt, nicht dringend: e4m3→fp16 vor sm_89 als
Compiler-Weg (exakt inkl. NaN) für allgemeine Kerne.

## Nachtrag 22.10. 08:45 — Triton-A/B abgebrochen, neues Vorgehen

- Das Triton-A/B (27B + DSv4) ist abgebrochen: Beide Modelle haben **keinen Triton-Kern mit `tl.dot`** im Pfad, beide Arme
  waren bis auf Rauschen gleich (Zahlen und Belege im STAND.md-Nachtrag 22.10. 08:40). Nichts läuft, Config gleich Backup,
  Karten leer.
- **Peuqui 22.10. ~08:40: erst sämtliche Optimierungen umsetzen, dann gemeinsam testen.** Jede neue Route als eigener
  Schalter (KernelConfig-Feld bzw. Policy), je Schritt nur Kerntests + Mikrobench im CUDA-Graph, am Ende je Modell ein
  A/B/A/B (Vorlage `run_ab_triton_2026-10-22.sh` im Modus `prodpv`, Start über `systemd-run --user`).
- Peer-Status: „sm7x-Triton: Routen auf Triton-Tensorkern-Kerne umstellen“.
- **Stand 12:55:** Familie 1 (Indexer-Prefill) fertig, Fork `triton-sm7x-routes` bis 7458c6257 gepusht. Triton-Dot-Tests
  repariert (Baum bis d902b6c272, beide Karten 287/0, Leistung unverändert, Bundle erneuert; Details STAND-Nachtrag
  22.10. 09:00). Aufräumen erledigt (157 GB frei). Nächstes: Familie 2 (quantisierte Linear-GEMM W8A16/W4A16: Block-FP8,
  MXFP8, NVFP4). Offen bei Peuqui: stt-suite-Token ins eigene Volume statt HF-Cache-Mount (Dienst-Neustart nötig).

## Ergebnisse 21./22.10. (alle gemessen, Dateien in STAND.md)

**Eigenes Triton 3.6 mit Tensorkernen** (Baum `/home/mp/Projekte/vllm-research/triton-sm7x`, Zweig `sm7x-tensor-cores`,
lokal; Bundle-Sicherung `/mnt/speicher/backup/triton-sm7x-tensor-cores-2026-10-22.bundle`, nur bis bb909d9c5c):
- ba18facf27 Turing MMAv2 m16n8k8 frei, 3519c6abea Register-Pipelining sm_75, a47131f3b5 Reduktion/`delinearize` bei
  verstreuten Lane-Bits (Volta-Layouts), bb909d9c5c Volta MMAv1 über Linear Layouts (+ Prefetch, Pipelining sm_70),
  433472b167 schleifeninvariante Dot-Operanden ≤ 64 Register vor sm_80 außerhalb der Schleife.
- Eingefrorene Kopie für Läufe: `/home/mp/.cache/triton-sm7x-v2` (Stand 433472b167, per `PYTHONPATH`). Ältere Kopie
  `/home/mp/.cache/triton-sm7x-turing` (nur Turing) kann weg (vorher fragen).
- Bau: `scratchpad/triton/build_triton.sh` (Bau-venv `~/.venv/triton-build`, LLVM `~/.cache/triton-build-home`); Vollbau
  4:20 h, inkrementell Minuten. **Nicht im Baum bauen, solange ein Lauf das Baum-Triton lädt.**
- RTX: Triton-FA (27B-Form, 32k) 14,42 ms = 55,4 TFLOPS gegen FA2 von Hand 24,76 ms; GEMM fp16 = cuBLAS bis M 512, groß
  82–85 %; DSv4-Indexer 1,04–1,21× Handkern (ohne 64-Register-Grenze gleichauf). V100: FA 24,6 ms = 32,4 TFLOPS gegen
  `flash_attn_v100` 75,5 ms; GEMM ~70 % cuBLAS (Ursache offen); Indexer 0,72× der Produktions-cuBLAS-Route.
- Korrektheit: `test_core.py -k dot` auf RTX und V100 ohne einen Fehlschlag, den unverändertes Triton nicht auch hat.

**GLM-5.3** (Baum `1Cat-vLLM-glm53`, Zweig `glm53-turing-2026-10-21`, gepusht bis 0b8b98c4f):
- 422184292 Sparse-MLA-Prefill über Tensorkern-Blöcke statt FMA-Triton-Kern; 0b8b98c4f CSF-Expansion: Grid über
  min(Slots, Experten) statt Token × top_k, ein Programm je Kachelzeile (Raster bytegleich, Test erweitert).
- Profil-Skript, 14 038 Token, Einzelläufe unter nsys (Zeit s / Token/s): 21.10. 68,72 / 204 → Blockroute 10,75 / 1306 →
  + CSF **8,20 / 1712**. Decode langer Kontext (128 Token, inkl. erstes Token): 303,2 → 89,0 ms/Token (3,3 → 11,2 Token/s);
  kurz unverändert ~62 ms/Token (16,1 Token/s), ohne nsys 17,3–17,6 Token/s. Noch kein A/B/A/B → nach dem Triton-A/B
  GLM-A/B/A/B gestriger gegen heutigen Baum, ohne nsys, mit Langkontext-Probe (Peuqui will ms und Token/s).
- GLM mit sm7x-Triton: Qualität 8/8 von Hand richtig, RTX-Stufen −0,6 s Rechenzeit; Eintrag
  `glm53_test_entry_sm7x_2026-10-22.yaml`. Run-12-Antworten gesichert als `glm53-text-quality-run12-2026-10-21.json`.
- Jetzt binden die RTX-Stufen (13/11 Schichten, 6,5–6,7 s gegen V100 3,7–3,9 s); Umverteilung braucht VRAM.

## Entscheidungen Peuqui (22.10.)

- Triton-Basis bleibt 3.6 (Torch-Pin), **kein Port auf Triton main**. Erst alles ausreizen, bis die eigenen Patches auf
  Stand sind — dann sortieren und Triton upstream UND 1Cat anbieten (RFC zuerst, Texte vorher zeigen). Der Port-Zweig
  `sm7x-upstream` im Worktree `triton-sm7x-upstream` liegt geparkt, ungebaut.
- Leistungsangaben immer in ms und Token/s, nicht als Faktor.

## Offene Arbeit (Reihenfolge)

1. Routen auf Triton-Tensorkern-Kerne umstellen, wo sie den heutigen Weg schlagen (Mikrobench je Gerät): DSv4-Indexer
   Prefill/Decode und Sparse-MLA (Triton-Kerne liegen schon im Baum, heute cuBLAS/BMM), 27B-Prefill-Attention (FA2 40 % des
   150k-Prefills; Triton-FA RTX 14,42 gegen FA2 24,76 ms, braucht einen prefill-tauglichen Kern mit Seitentabelle), GLM
   (KDA-Kette, Variante C). Danach gemeinsame Testreihe → Entscheidungsvorlage sm7x-Triton in die Produktion.
2. GLM Variante C: MXFP8-Gewichte in 8 Bit (KDA q/k/v/o, dichtes MLP, lm_head; spart ~5 GiB), Triton-Kern
   `scratchpad/triton/mxfp8_w8a16.py` (GEMM korrekt, Prefill M 512 noch 1,6× langsamer als cuBLAS-fp16; neue GEMV-Variante
   ohne tl.dot für M ≤ 8 geschrieben, noch ungetestet). Prefill braucht Tensorkern-Triton. Danach Schichtverteilung
   umbauen (V100 mehr Schichten).
3. GLM-A/B/A/B (s. o.) für belastbare ms/Token/s.
4. DSv4: Triton-Indexer auf V100 (−25 % gegen cuBLAS-Route) — nur mit sm7x-Triton in Produktion.
5. Volta-GEMM-Abstand (ncu; 8-Warp-Kacheln 256×128 in `bench_gemm.py` schon eingetragen).
6. Fork-main: lokaler Commit 7d83b7032 (fusionierte Turing-Indexer-Scores, −54 bis −59 % im Mikrotest) ist **nicht gebaut
   und nicht gepusht**; `run_ab_turing_2026-10-21.sh` (Turing-Kern-A/B 27B + DSv4 gegen Prod) wartet darauf.
7. 27B-Allreduce (46 % Prefill) als eigener Strang.

## Werkzeuge und Fallen von heute

- Scratchpad: `/tmp/claude-1000/-home-mp-Projekte-vllm-research/508c7d38-9687-45d1-9c5d-79a5efeb3358/scratchpad`
  (`triton/`: bench_gemm/fa/fa_v100/idx/idx_v100/glm_sparse, check_dots_volta, check_loop_volta, check_fa_v100,
  run_hoist.sh, sass_gemm_volta.py; `glm/`: nsys_top.py, csf_expand_bench.py, glm53_profile_copy.sh).
- GLM-Profil: `glm53_profile_copy.sh <tag> 14000` (ENTRY_FILE für den sm7x-Eintrag), GLM-Qualität:
  `glm53_boot_copy_sm7x.sh text` mit ENTRY_FILE — schreibt `quality_2026-09-27/glm53-text-quality.json` und
  `gpu_glm53_boot_text_2026-10-21.csv` unter festen Namen (vorher sichern).
- vLLM-AOT-Cache hat keinen Code-Hash: Varianten mit anderem Triton immer mit eigenem `VLLM_CACHE_ROOT`; Inductor-Schlüssel
  enthalten `triton_key()` und sind sicher.
- Mikrotests nur außerhalb gezählter Läufe (Kaltläufe zählen nicht), Kerne nur im CUDA-Graph messen.
