# Unabhängige Prüfung GLM-5.3-PR (27.10. ~12:20): READY AFTER FIXES

In Ordnung: acht Basis-Commits patchgleich zu Loader/MLA-Zweigen; Revert-Prüfung; Gerätegates; CsfScaleConsumer-Ablehnung (ValueError);
Fünf-Bit-Verweigerungspfad; MXFP8-Exaktheit 95..134; csf32_expand-Wrapper (16 Argumente = Schema); CUDA-Kerne ohne Fehlerbefund.
BLOCKER
1. Text „PYTHONPATH … none (stock Triton 3.6)“ falsch: alle GLM-Reihen mit sm7x-Triton vorn (triton-sm7x-v2 bzw. triton-sm7x-glm-prod-2026-10-25).
2. MLA-Block-FP8 kein A/B: ab_glm_dev1_2026-10-24.log Einzelarm D512, ein Boot (23,58 tok/s, 4,34 s, 139 810); Basis 19,96 aus anderer Reihe,
   Dev-Baum enthielt auch 9c7c2a21b/ac75fad2e → als Einzelboot aller drei Commits gegen Arm B der MXFP8-Reihe kennzeichnen; Commit 9a4eafd7b ebenso.
3. Duplikat-Absatz: #1125 (rivetphilbot) ersetzt dieselbe _sm70_glm5_sparse_paged_fp8_kernel-Prefill-Route durch FP32-Softmax-Kern; #1092
   argumentiert, fp16-QK-Scores im Decode verlieren Präzision/überlaufen — wir übertragen genau diese fp16-Route auf den Prefill; #1260 berührt
   11 unserer Dateien (GLM5NextSM70SparseBackend.supports_compute_capability, cuda.py-Priorität, KPool needs_software_fp8). Dateiüberschneidung
   auch #1159, #1165, #1166, #1102, #1271. → Abgrenzung erklären oder abstimmen.
SHOULD-FIX
4. Qpn8Mxfp8LinearKernel steht für JEDES MXFP8-Modell auf SM70/75 vorn (vorher SM75 Marlin, SM70 Emulation); can_implement sieht keine Form →
   Ladefehler statt nächster Kern bei N%32≠0, K/16 nicht durch 8/12/16, E8M0 außerhalb 95..134. Fix: Form in Mxfp8LinearLayerConfig, can_implement
   lehnt ab, was prepare ablehnt (oder Kern auf ModelOpt-Mixed beschränken). group_routing trifft alle Skinny-MoE-Modelle → im Text nennen.
5. Einbettungs-Offload nicht strikt opt-in: _is_parameter_selected True ohne --cpu-offload-params, embed_tokens wird vor make_layers gewickelt →
   bestehende --cpu-offload-gb-Nutzer verlieren 1,18 GiB Budget an die Einbettung. Fix: nur wickeln, wenn cpu_offload_params embed_tokens nennt.
6. clang-format (ColumnLimit 80): skinny_moe_qpn_sm70.cu 221/502/539/817/869/876 + Makro ohne Leerzeichen vor „\“; fp8_qpn8_sm70.cu 333;
   fp8_qpn8_mma_sm75.cu 5/100.
7. Ohne Test: CsfScaleConsumer-Ablehnung; CSF-Teilrouting in Glm5NextModel.load_weights (7303dc246); KDA-Split + _kda_projection_mapping;
   Pfadtabelle nennt test_modelopt.py ohne CSF-Test.
8. _kda_projection_mapping (model.py:1160) global statt je Schicht → KeyError bei teilweise quantisierten KDA-Schichten; all-or-none prüfen.
9. Text: DFlash2-vs-ohne kein A/B (D Einzelboot, Partition 13,7,7,7,11 und Capture [1] gegen 14,7,7,7,10 und [8]); „prefill +6,6 %“ = langsamer
   (4,26 → 4,54 s); MXFP8 Prefill 2,5 % langsamer klar sagen, Bereich acb0fe649/23515aaae/f78b43972; „after 14k“ falsch (128 Token nach 8 687er
   Prompt; Log-Etikett veraltet); Kopfzeile: Decode kurz 192 Token nach 30er Prompt; Quellzeile je Reihe Referenz/Änderung; „shared raster freed“
   nur auf Stufen, deren Schichten alle packen; „Unverified scope: none“ (TP>1 nie); Einbettungs-Rauchtest nicht eingecheckt; „built and
   unit-tested“ noch nicht wahr; Test Plan: Modell-Benchmark-Vorgehen; Gruppe 2 ohne Modellzahl (nsys 68,72 → 10,75 s für 14 038 Token);
   516k-Zahl vor Gruppen 4/5 (Fünf-Bit-Bau 251,63 s ≈ 2 051 tok/s, Einzellauf); „about 19“; „accepts 95..134“; KDA o_proj in Gruppe 3;
   FP8_PB_WO-Änderung in modelopt.py (alle Architekturen) nennen.
10. Doku sm70_glm53_pp5_dflash2_mixed.md: lokale Pfade/Skriptnamen, „all numbers from files named at the end“ falsch, „same code and flags
    except draft“ falsch, Zahlen vor Gruppen 4/5, „last stage V100 equal in speed“ widerspricht dflash2i (37,15 gegen 32,33 tok/s).
NIT: _software_e4m3() dupliziert needs_software_fp8() (cache_utils.py:26); s5_scales.py-Docstring „at most 23“/„up to 56“; S5-Test mit
raster.new_empty(0); csf32_expand_sm70 TORCH_CHECK Größe + Experten-ID; static_assert kBlockK==32; _release_rasters Platzhalter; batched_pv-
Docstring; TensorFold-Verweis mit URL; Commit-Nachrichten: „way B“/„TensorFold“, „at most 23“ vs 0c3296413; Signed-off-by 8/22.
