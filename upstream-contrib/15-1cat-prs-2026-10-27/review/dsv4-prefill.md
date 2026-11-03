# Unabhängige Prüfung DSv4-Prefill-PR (27.10. ~13:05): NOT READY

In Ordnung: Kerne; alle Modellzahlen im Text gegen die Logs belegt (ab_dsv4idx/ab_dsv4sm, Texte per sha1).
BLOCKER
1. tests/models/test_deepseek_v4_sm70_routes.py:491,504-510 (36e84abe8) scheitert überall: setzt layer.sparse_tensor_cores, das auf diesem Zweig
   nicht existiert (Rest von d77726c11 aus dem Routen-Zweig) → Zeilen entfernen, Nachricht bereinigen („sparse_tensor_cores … both routes“,
   „sm7x Triton“), in 58bf56833 einfalten.
2. Kopfzahl-Tests in 1cb3d8eeb (test_prefill_model_head_counts_match_the_reference, test_turing_prefill_takes_the_fused_scorer) scheitern auf beiden
   Karten (Referenz ohne fp16-Rundung; Korrektur kam erst in 7458c6257) → _reference_prefill_logits/_assert_close_to_max ohne triton-Parametrierung
   hierher portieren.
3. „Commits 3 and 5 were measured … together with Triton routes“ falsch für Commit 3: mit triton_sm7x_tensor_cores ersetzt der Triton-Scorer den
   fusionierten Turing-Scorer (indexer.py:778), alle Läufe seit 23.10. mit Schalter → Commit 3 lief nie in einem Modell, wird mit diesem PR Standard
   auf jeder SM75-Stufe (ersetzt dort auch den gemessenen Epilog). → Zweig booten (Triton aus: Routen-Treffer, Greedy, 8 Antworten, Langkontext,
   möglichst A/B gegen Commits 1-2) oder ehrlich sagen; info_once-Routenlog für den Prefill-Scorer ergänzen.
SHOULD-FIX
4. KV-Gather (f17e11c49) gilt für alle GPUs (nvidia/flashmla.py ohne CuteDSL/FP8-Emulation, amd/rocm.py) → Text/Pfadtabelle; NUM_WORKERS 2048 fest
   (aus multiProcessorCount ableiten?); Kommentar „reach the memory rate“ unbelegt (≈ 280/345 GB/s).
5. indexer_prefill_cublas ändert auf Turing still die Bedeutung (fusionierter Pfad vor dem Schalter, indexer.py:673 vs 682) → fusioniert unter den
   Schalter oder Bedeutung dokumentieren (sm70_sparse.py:103-104, Vertrag).
6. Duplikat-Absatz: #1260 berührt sm70/indexer.py + Testdatei (FP32-Scores in _prefill_logits_cublas, Kopfmaskierung im Epilog, del scores je
   Kachel, Kommentarblock bei _EPILOGUE_BLOCK_K) → Text- und Semantikkonflikt mit Commits 1 und 3; #1222 patchgleich zu 3 der 9 Commits.
7. Gleicher Rebinding-Fehler im Zwilling sparse_attn_indexer_kpool.py (Prefill-Schleife ~469-578, Profilreservierung :313) → gleiches del oder begründen.
8. Commit-Nachrichten: „nsys/ds-stages-1020“, „DSv4 (PP5) production entry“; „routes tests 30 passed“ vom Routen-Zweig; „sm7x Triton“, 104-Zeichen-
   Zeile; Signed-off-by 3/6; 5bb23f18c „nvidia-smi peak 19246 vs 19466“ (mini_live3) widerspricht mini_lg64 → streichen/erklären.
9. Arme: 5669da106/e1cb50fd9 tragen zusätzlich 6c0ef8efb/0314585ac (QPN8-Split-Regel, trifft keine DSv4-Form) → „reference 4da0bcfa4, changed
   5669da106 / e1cb50fd9“; Mini-Läufe mit sm7x-Triton + Schalter offenlegen.
NIT: Umbruch der _decode_cublas_blocker-Strings in 1cb3d8eeb streichen; Purpose quantifizieren (Epilog 14,1/17,7 s von ≈ 52/60 s Nicht-MoE-Zeit
= 27-30 %), „820 of 1434 us“ nur RTX; „one boot per run, after an uncounted cold boot“; Vertrag: --max-num-batched-tokens 512, --max-num-seqs 1,
--max-model-len 307200, --num-gpu-blocks-override 3000; „(160 + 113)“ erklären, „within 0,3 %“; „Decode step 1,4-3,4 ms shorter“; .cu-Kopfkommentar
Umbruch; q.stride(0)-Geradheit prüfen; Titelstil „[Perf][SM70][DeepSeek-V4] …“.
