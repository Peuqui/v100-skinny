# Unabhängige Prüfung Triton-RFC-Entwurf (27.10. ~12:10): NOT READY — Entscheidung Peuqui nötig

BLOCKER
1. Maintainer haben genau das abgelehnt: #9970 „Restore MMA support for Turing GPUs“ (nlzy, April 2026, ungemergt geschlossen):
   Jokeren „we support >= 80“, Bedingung „dedicated person maintaining the Turing path“ oder „Turing GPU CI“; Raoux „I don't think
   either of those are possible … this is not something we want to support“. #10918 (Juli 2026, Lezcano): „sm75 is not currently
   supported … not implementing or accepting fixes“. #9349 (int8-FMA-Absturz, unser 03b2bcb3f6) als „expected“ geschlossen.
   README main: „Compute Capability 8.0+“. #11969 (29.09.2026) Verifier lehnt Volta ab. Raoux #5066: „unlikely to accept complex
   performance related patches“. Optionen: (a) Wartung + selbst gehosteter sm_75/sm_70-CI-Runner anbieten (Peuqui-Entscheidung),
   (b) „Show and tell“-Discussion (Fork ankündigen, kein Upstream-Antrag), (c) nur Plugin-Haken erfragen (TRITON_PLUGIN_PATHS, #10774),
   (d) lassen.
2. Zahlen falsch/veraltet: V100-FMA-Attention 2,5 TFLOPS (319,35 ms, fa_stock_v100.log), nicht 1,4 (= RTX); test_core -k dot am Kopf
   0825af230a 283 passed / 0 failed, 2329 skipped (287 nur bei faa9e75b75); „fixes failures … MMAv1 fragment slots“ war ein eigener
   Fehler der Reihe — auf 3.6.0 waren es int8-FMA (40), fp16-Akku FMA (26), 4 fp8 über Smem (jetzt übersprungen). GEMM/FA-Zahlen aus
   frühen Ständen; Vorschlag Ersatz: RTX 8,9/51,5/75,4/69,4 (cuBLAS 7,3/34,0/92,9/82,0, FMA 3,0/6,6/6,8/7,5; gemm_onebarrier4_rtx.log),
   V100 10,5/48,0/71,9/76,4 (cuBLAS 8,1/44,8/78,8/83,8, FMA 3,9/6,8/8,0/8,6; gemm_onebarrier4_v100.log, Stand b633185304);
   FA 2048 Queries gegen 32k Keys: RTX 14,45 ms = 55,3 TFLOPS (FA2-sm75 24,4–30,8 ms), V100 16,79 ms = 47,6 (flash_attn_v100 75,47 ms
   = 10,6, FMA 319 ms = 2,5); Logs fa_region_*, fa_stock_*. RTX-cuBLAS M=512 71,8 ist Ausreißer (sonst 92–93).
3. Aussagen zu main: Reduktion mit verstreuten Lane-Bits auf main schon gelöst (ReduceOpToLLVM reduceLaneIdMask, #12013) → Scope 1
   streichen; „blockTileSize“ falsch; CTA->CGA und Volta-Verifier stimmen.
SHOULD-FIX: „Triton 3.x“ → „seit 3.3 (#5066)“; „#5099“ zitieren statt Commit 4480f86045; „still present“ statt „maintained“
(#12404 änderte batchOffset); Produktionsaussage: Triton-MoE nur V100 (RTX Handkern); Link fork-main-v4 statt main; lit/unittest nie
gelaufen (accelerate-matmul.mlir dot_fall_back_fma_before_ampere bricht vermutlich), sm_80+ ungetestet → lit am Kopf laufen lassen
oder Umfang ehrlich nennen; sechs Commit-Nachrichten verweisen auf SHAs nur im privaten deutschen Zweig (433472b167→cbba5daa82,
35d55f38f3→c540f3fc84, dcf200731d→e7acb065e8, d105df138b→274bdb8631, 2ddd426b2a→994efb52ec, e74d755ee8→7e4b714ea4) → Reword +
Force-Push des Fork-Zweigs (kein PR-Kopf); unveröffentlichte Werkzeuge/Jargon in Nachrichten; Stock-RTX „70 failures“ ohne volles Log
→ nachmessen.
Zweig-Hygiene in Ordnung (Autor noreply, englisch, keine Pfade/Secrets, Lizenz unberührt).
