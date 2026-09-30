# Skinny-Reihe an 1Cat — Plan (30.09.2026, Peuqui: „mach den PR, egal ob 1Cat auf #718 antwortet“)

## Befund: was die Produktion wirklich nutzt (Log der Starts 30.09.)
- DSv4: `QPN8Fp8BlockScaledMMLinearKernel` (qpn8_blk.py → gemm_qpn8_blk, _mt2,
  _wmma, qpn8_blk_dequant) für FP8-Block-Linear; MoE `SM70_SKINNY`
  (Mxfp4SkinnySm70Experts, moe_qpn, 256 Experten, Skala je 32).
- Flash-Next: MoE `SM70_SKINNY` (Nvfp4SkinnySm70Experts, 512 Experten, Skala je 16);
  dicht V100 = 1Cat TurboMind, RTX = QPN2 (#604).
- 27B: KEIN Skinny-Code (dicht über 1Cat QPN2/QPN8 auf SM75 via #604).
- Ungenutzt in Produktion: dichte Skinny-Linear-Routen in marlin.py (~900 Zeilen),
  gemm_qpn8/_mt2 aus modelopt.py, Forschungskerne (simt, wmma, dp4a, mma8,
  wmma_cfg/_splitk, moe_simt, pack_x8-Op) → NICHT einreichen.
- 1Cat hat eigene native QP-N-Kernel (nvfp4_qpn2/qpn4, fp8_qpn8, mxfp4_qpn_m1,
  nvfp4_grouped_decode) → keine Doppelangebote.

## Reihe
- **S1 – Skinny-MoE-Backend NVFP4 + MXFP4 (SM70/SM75)**: Kernel-Abschnitte
  Hilfsfunktionen + QPN (gemm_qpn) + MoE-QPN (moe_qpn) nach
  `csrc/sm70_skinny/` (MIT-Kopf, dnv2003 + unsere Ergänzungen dnv2003/v100-skinny#8),
  Registrierung in torch_bindings.cpp/ops.h (statt PYBIND/JIT), Prepack
  (`_qpn_prepack`), Nvfp4/Mxfp4SkinnySm70Experts, Oracle-Einträge nvfp4/mxfp4,
  `sm70_skinny` in config/kernel.py, `eager_break_during_capture(ignore_full_mode)`,
  envs. Tests: test_skinny_mxfp4_moe + NVFP4-Pendant gegen Referenz.
  Nutzen belegen: Flash-Next Prefill/Decode gegen 1Cats Standard-MoE, DSv4.
- **S1b (klein, eigenständig)**: Emulations-MoE dequantisiert nur gewählte
  Experten in Häppchen (nvfp4_emulation_moe.py, VLLM_SM70_NVFP4_EMU_CHUNK) —
  sonst OOM bei DSv4 (256 Experten ≈ 13 GB/Schicht).
- **S2 – FP8-Block-Linear über QPN8 (DSv4 dicht)**: FP8-QPN8-BLOCKED-Abschnitte,
  4 Ops, QPN8Fp8BlockScaledMMLinearKernel. Nutzen gegen 1Cats Pfad messen.
- **S3 – DeepSeek-V4 auf Turing**: Torch-Ersatz DeepGEMM-Logits
  (sparse_attn_indexer, attention.py), Triton-„ROCM“-Pfad für pre-Hopper,
  `has_cutlass` erst ab sm_80 (import_utils), cache_utils/fused_indexer_q usw.

## Pflichten je PR
Basis 1Cat main per rebase; `git diff --stat main HEAD` nur Themen-Dateien;
Revert-Check ([[feedback_pr_revert_check_after_rebase]]); Tests V100 + RTX;
pre-commit + mypy-3.10; A/B gegen Produktion (Greedy bitgleich); Texte an
Peuqui vor dem Senden; #718 verlinken, dnv2003 nennen.

## Ausgeschlossen (entschieden)
- PP-Token-Prüfung (cd7bbf3c): bleibt im Fork, nicht an 1Cat (Peuqui 30.09.).
- QSA-XQA-E4M3-Arbeitsspeicher: wörtlich aus fremdem offenem #664.
- AOT-Kerneltabelle: von 1Cat selbst gelöst (#675); torch-Patch in #621
  vermutlich überflüssig → Nacharbeit #621 (Beleg: Warmstart ohne torch-Patch).

## Neu 30.09. abends
- S1b gestrichen: Emulations-Häppchen synchronisiert per torch.unique/int() mit dem
  Host (nicht aufzeichnungsfähig), nützt nur NVFP4-Checkpoints mit swiglu_limit
  (NVIDIA-DSv4, gelöscht, nicht prüfbar); S1 deckt den Fall über
  SM70_SKINNY in NVFP4_BACKENDS_WITH_CLAMP ab.
- Kandidat T1 (klein, 1Cat-Code): mxfp4_sm70_moe.process_weights_after_loading
  hält beim Umbau je Schicht Original + Liste der vorbereiteten Experten + Stapel
  (3 Kopien, DSv4 ≈ 3,2 GB je Kopie), während alle Schichten der Stufe schon
  geladen sind → DSv4 PP5 10,8,8,8,9 OOM auf allen V100-Stufen („Tried to
  allocate 2.00 GiB“, 30.09. 18:20). Fix: Stapel vorab anlegen und je Experte
  hineinkopieren, w13 vor w2 abschließen und Original freigeben.
- 1Cat-Standard für DSv4-MXFP4 im Fork nachbilden: `--moe-backend marlin` +
  QUANT_BACKEND=auto (V100 → TurboMind über das Fork-Gate, RTX → Marlin wie
  1Cats Auto-Liste). Ohne Flag nimmt der Fork auf Turing SM70_SKINNY (steht in
  der Fork-Auto-Liste, im PR nicht).
- S2-Beleg: +58 % war Qwen3-0.6B-FP8 eager, nicht DSv4 → Kernel-Matrix
  (fp8_blk_backend_bench.py) gegen 1Cats aktuellen TurboMind-FP8 neu messen;
  1Cat hat eigene QPN8-Tabellen nur für DSv4 TP4 (_SM70_FP8_QPN8_PP2_TP4_*).
