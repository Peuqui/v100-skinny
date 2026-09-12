# Paket E — Turing-Pfad für 1Cats kompilierte Skinny-Ops (Stand 12.09. 09:15)

## Befunde aus dem aktuellen main (ae75fb9b), alle am Code belegt

1. **1Cat trägt unsere Kernel kompiliert**: `csrc/sm70_turbomind/ops/`
   (`nvfp4_qpn2`, `nvfp4_qpn4`, `fp8_qpn8`, `awq_qpn_m1`, `mxfp4_qpn_m1`, je
   „derived from dnv2003/v100-skinny (MIT)", eigene `LICENSE.v100-skinny`).
   Gebaut nur für 7.0 (`CMakeLists.txt:366`), Wheel ist reines sm_70.
2. **Laufzeit-Gates**: Gewichtsvorbereitung gattert auf dem Gerät des Tensors
   (`is_exact_sm70_cuda`, `== (7, 0)`), die Quant-Methoden-Wahl auf Gerät 0
   (`is_exact_sm70_cuda_platform`). 35 Aufrufstellen in 12 Dateien; die
   Linear-GEMM-Teilmenge ist kartiert (Agent-Bericht 12.09., siehe unten).
3. **Harter Blocker im C++**: `lmdeploy/.../gemm/arch.h`: `Sm70: Arch<700, 750>`.
   Die TurboMind-Registry (`tm_registry_sm70.cu`) registriert Kernel nur bei
   `is_arch_compatible(700, darch)`, und `darch = major·100 + minor·10 = 750`
   auf Turing → **kein Kernel**. `nvfp4_gemm_sm70_out` (der Prefill-Teil von
   `nvfp4_qpn2_dispatch_sm70_out` ab `kQpn2DispatchMaxRows`) würde auf einer
   RTX scheitern. Python-Gates allein reichen also nicht.
4. **Unser Fork fährt auf der RTX nie TurboMind**: Routenzähler 12.09.
   (27B MTP, RTX-Paar): `qpn2` M1/2/4/8, `qpn` M15/16, `dense` M20/M2048.
   `dense` = `qpn_dequant.py` (Triton, QPN-Prepack → transientes fp16) +
   `torch.matmul` (cuBLAS). Begründung in der Datei: Marlin FP4 auf Turing
   ~27 TFLOPS, cuBLAS fp16 mehr; ein residentes Layout statt drei (~10 GiB
   je Layout beim 27B).
5. Auf reinem main nimmt ein Turing-Worker für NVFP4 heute Marlin sm75 für
   ALLES (`marlin_utils_fp4.is_fp4_marlin_supported`: `has_device_capability(75)`).

## Routenkampagne 12.09. (Schritt 1, `handover/2026-09-12/routes/`)

| Konfiguration | Skinny-Pfad geladen? | Routen (Dispatch-Entscheidungen je Rang) |
|---|---|---|
| 27B MTP k=3, RTX-Paar (73,4 tok/s) | ja | qpn2 M1=14, M2=2, **M4=1190**, M8=512; qpn M15=768, M16=129; dense M20=129, M2048=256 |
| 27B DFlash2, RTX-Paar (75,3 tok/s) | ja | qpn2 M1=6, M4=3, M7=2, **M8=1118**; qpn M15=768; dense M28/32/36, M2048, M5120 |
| 27B MTP k=3, V100-Paar (64,9 tok/s) | **nein** — 1Cats TurboMind-Pfad (`VLLM_SM70_NVFP4_TURBOMIND=1`) | keine Datei |
| Flash-Next TP2×PP2 (23–30 tok/s @13k) | **nein** — Experten über `MARLIN` NvFp4-MoE-Backend, keine Skinny-Meldung | keine Datei |
| DeepSeek PP5 | **nein** — `VLLM_SM70_QUANT_BACKEND=marlin`, keine Skinny-Meldung | keine Datei |

**Schluss:** Unser Skinny-Pfad ist in Produktion genau der Turing-Pfad für
dichte NVFP4-Modelle (modelopt). Decode = qpn2 (M ≤ 8) und qpn (M 9–16),
Prefill/Verify ab M ≈ 20 = Dequant + cuBLAS. Das Skinny-MoE-Backend ist in
keiner Produktionskonfiguration aktiv → **E-3 ohne Beleg, entfällt** (außer
eine eigene Messung Marlin-MoE gegen sm70_skinny auf Turing zeigt einen
Gewinn — separat, nach E-1/E-2). Der Zähler zählt Capture-/Eager-
Entscheidungen, nicht Replays; die M-Verteilung ist daher formen-, nicht
tokengewichtet.

## Entscheidung für E-1 (Vorschlag)

**E-1 = Turing-Pfad in 1Cats NVFP4-Op: QPN2 fürs Decode, QPN-Dequant + cuBLAS
fürs Prefill, ein Layout, Gates pro Worker-Gerät auf Pre-Ampere.**

- Vorbereitung (`prepare_nvfp4_linear` in `sm70_turbomind.py`): auf Turing
  nur den QPN2-Prepack (deren `nvfp4_qpn2_prepare_sm70`) anlegen, kein
  TurboMind-`tm_weight`.
- Apply: `op_kind="nvfp4_qpn2_dense"` (Arbeitsname): M ≤ `kQpn2DispatchMaxRows`
  → deren `nvfp4_qpn2_gemm_sm70_out`/`_gated_`; darüber → Dequant-Kernel
  (Port von `qpn_dequant.py`, muss auf DEREN qpn2-Prepack-Layout passen — zu
  prüfen: `nvfp4_qpn2_prepack_codes_kernel` gegen unser `_qpn_prepack`) +
  `torch.nn.functional.linear`.
- Gates: neuer Helfer `is_pre_ampere_cuda(tensor)` neben `is_exact_sm70_cuda`;
  Volta behält seinen Pfad unverändert (Bitgleichheit der V100-Abnahme),
  Turing bekommt den neuen. Quant-Methoden-Wahl liest das eigene Gerät
  (`torch.accelerator.current_device_index()`), nicht Gerät 0.
- `get_min_capability` für `modelopt_mixed` (Zeile ~2405): 70 auch auf
  Turing, wenn der Pfad aktiv ist (heute „Minimum capability: 89").
- NICHT in E-1: AWQ/MXFP4/GPTQ/uint4 (keine Turing-Messung), MoE (E-3),
  Attention/Indexer, CMake-Archs (Wheel bleibt sm_70; die RTX fährt den
  sm_70-Cubin per Binärkompatibilität — belegt: Punkt 15 bitgleich, DFlash2
  SHA gleich auf beiden Paaren).

**Alternative, verworfen, außer Peuqui will die Zahl:** `Sm70: Arch<700, 800>`
und TurboMind s884 auf Turing messen. m8n8k4 läuft auf Turing mit
reduziertem Durchsatz; unser Fork hat diesen Weg nie genutzt. Kostet einen
inkrementellen Bau + eine Prefill-Messung, falls gewünscht.

## Belege, die der PR braucht (AGENTS.md)

- Bitgleichheit Turing: Text-SHA 27B DFlash2 auf RTX gegen V100 (`0106…`),
  Kernel-A/B Dequant gegen Referenz-Dequant (fp32) über die 27B-Formen.
- Tempo RTX: main (Marlin sm75) gegen main+E-1, Decode (DFlash2/MTP) und
  Prefill (TTFT bei 2k/13k), je 5 Läufe.
- V100 unverändert: SHA und tok/s gegen Referenz (76,27).
- CPU-Tests: Gate-Helfer (Volta/Turing/Ampere, eigenes Gerät vs Gerät 0),
  Dequant-Layout-Roundtrip auf synthetischen Gewichten (GPU-Test, skip ohne
  CUDA).
- Duplikatssuche, komplette bestehende Testdateien, pre-commit + mypy.

## E-2 (danach): Block-Pack `skinny_pack_x8` in deren qpn2-Kernel, Schwelle
sm75 ≥ M5, sm70 ≥ M8 (STAND Punkt 8/13). E-3: MoE, wenn die Kampagne
(Flash-Next, DeepSeek) die MoE-Route zeigt.

## Werkzeuge
- Bau: `handover/2026-09-12/build_pr_turing.sh` → `.venv-pr-turing`
  (reines main + lokale #601-Nachhilfe), Worktree `1Cat-vLLM-pr-turing-ops`.
- Routen: `handover/2026-09-12/routes/route_campaign.sh`.
- Gate-Karte: Agent-Bericht 12.09. (in `paket-e-gates.md` abgelegt).


## Nachträge 12.09. mittags (Umsetzung E-1)

- **Der 27B ist `modelopt_mixed`**: GDN-Projektionen (`in_proj_qkv`,
  `in_proj_z`, `out_proj`) FP8 per-Tensor (Gewicht e4m3 `[N, K]`, ein
  `weight_scale`, statische `input_scale`), MLP NVFP4 (Gruppe 16). Ein reines
  NVFP4-Modell mit dichten Linears liegt nicht auf der Platte (Flash-Next ist
  NVFP4, aber die Linears sind MoE über Marlin).
- **Reines main lädt den 27B auf Turing nicht**: Mixed-Config Mindest-
  Capability 89; mit gesenktem Gate (Diagnose) stirbt der generische
  FP8-Rückfall in `marlin_utils_fp8.prepare_fp8_layer_for_marlin` an
  `orig_dtype` (vorbestehend, Turing). Es gibt keine main-Basis für den
  Checkpoint auf der RTX; Referenzen sind V100 auf main (TurboMind) und
  unser Fork (RTX 73,4 MTP / 75,3 DFlash2).
- **E-1 deckt beide Linear-Typen**: NVFP4 → qpn2-Decode (ihre Kernel) +
  Dequant-Prefill (neuer Triton-Op `nvfp4_qpn2_dequant.py`, Layout identisch
  zu ihrem Prepack); FP8 → ihre `fp8_qpn8_dispatch_sm70_out` (M ≤ 8 qpn8,
  9–32 gechunkt nur für Qwen3.8-TP4-Formen, sonst Dequant + `at::mm`), per-
  Tensor-Skala als Kanalskalen `[N, 1]` float32 an `fp8_qpn8_prepare_sm70`.
  Gates: `is_turing_cuda(tensor)`, `is_turing_cuda_platform()` (eigenes
  Gerät), `should_prepare_turing_qpn2`; Mixed-Config `get_min_capability`
  über `is_pre_ampere_cuda_platform()`.
- Tests: `tests/quantization/test_sm70_turbomind_turing_gates.py` (13, CPU),
  `tests/kernels/quantization/test_nvfp4_qpn2_dequant.py` (6, GPU,
  bitgleich gegen Checkpoint-Dequant und fp16-Matmul; V100 bestanden, RTX
  ausstehend).
- Worktree `1Cat-vLLM-pr-turing-ops` auf ae75fb9b; `setup.py` trägt die
  lokale #601-Nachhilfe (NICHT Teil des PRs). E2E-Kette
  `handover/2026-09-12/e1_chain2.sh`.
