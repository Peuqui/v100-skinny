# Unabhängige Prüfung wo_a/Freeze-PR (27.10. ~12:30): READY AFTER FIXES

In Ordnung: Revert-Prüfung; Fork-Gleichheit (Diff = f21d1c564..1addd22db); CONFIGURED_OPERATORS 108 = 108 Schemas; gruppierter Pfad
(Views, Offsets, Gerätegates, keine Verklemmung); #903-Aussage stimmt (dieselben Fakes, nicht Warmup/Freeze).
BLOCKER
1. docs/design/architecture/runtime_reference.md veraltet (sm70_runtime.py-Zeilenanker +7) → `python -m tools.generate_architecture_reference
   --write` und mitcommitten (füllt REFERENCE-Platzhalter).
2. Kernel-Config der Arme: alter Arm KC_OLD {"triton_sm7x_tensor_cores":true} (Tuning AUS, Fork-Standard dank #1187), neuer Arm Tuning AN +
   Freeze; main hat Tuning standardmäßig AN → A/B = (alt, Tuning aus) gegen (gruppiert + Tuning an + Freeze); laut STAND 24.10 08:35 ≈ halber
   Decode-Gewinn aus dem gruppierten Pfad (Prosa 57 vs 59 ms, Einzelboots v2a/v2b), Rest aus dem Tuning. Gegen main-Standard nicht gemessen
   → beide Konfigurationen + Standardunterschied nennen.
SHOULD-FIX
3. TM_GEMM_TRACE-Aussage falsch (Text + Commit 8ddbfbcf0): 28 Trace-Zeilen im Warmup mit policy=default measured=0; nach dem Fix policy=measure
   measured=1; „frozen plan held 0 shapes (14 after the fix)“; Pfadtabelle „export non-empty“ unbelegt.
4. Docstring sm70_runtime.py:66-72 veraltet („export and import“, „up to 2.2 s“, „nearest shape“ → nächstgrößerer Plan, sonst Heuristik).
5. Duplikat-Absatz: #1187 (gemm_runtime.cpp, awq_sm70_warmup.py, test_sm70_warmup.py; Test direkt nach dem von uns geänderten → Textkonflikt;
   entscheidet, ob der gruppierte Start standardmäßig getunt wird); #1167 (gemm_runtime.cpp, partials_size im selben Struct); #822/#890/#1092/#1125
   teilen auch vllm/_sm70_ops.py.
6. Verhalten mit main-Standard (Tuning an, Freeze aus) offenlegen: gruppierter Pfad je Form getunt → Greedy-Text ändert sich an Einzelstellen;
   Gruppengrößen außerhalb Warmup (nur 1,2,4,6,8,16; 3,5,7,9–15 fehlen) werden beim ersten Auftreten gemessen — prefill_probe v8-kc16-many:
   35 711 Token (11 je Gruppe) 12,8 s statt 7,0 s. Vorschlag: awq_sm70_warmup.py:711 alle Gruppengrößen 1..16 aufwärmen (≈ 10 Messungen mehr
   je V100-Stufe), sonst Einschränkung nennen.
7. Pfadtabelle: Freeze-Zeile „executed: tests“ nur Python-Mock → GPU-Test (Vorlage test_sm70_gemm_workspace.py) oder „measured“; Freeze wirkt
   auf alle getunten TurboMind-Familien (FP16, AWQ, FP8, MXFP4, NVFP4) → andere als unverifiziert; gruppiert: statt „other grouped BMM users“
   DeepSeek-V4 TP2/TP4 (nicht gelaufen); Warmup-Fix ändert jedes SM70-TurboMind-Warmup und macht gemm_lut_path/fp8_coordinated_tuning
   wirksam; Fehler nur bei nativen Feldern in kernel_config.
8. „tok/s +4,0 .. +6,2 %“ aus ab_tfacc (ein Boot je Arm) → kennzeichnen, Sampling-Parameter nennen.
9. Tests: „assert that it is passed“ nur in einem Test (test_sm70_warmup.py:152); kein Test, dass Freeze nur mit Schalter gerufen wird;
   „13 failed on main“ auf main laufen lassen und IDs nennen.
10. Historie: 00f1989ac/e9f0916be beschreiben Mechanismen, die 8ddbfbcf0 entfernt → drei Commits zu einem Freeze-Commit zusammenfassen;
    e6bc9c384 „91 passed (1 failed)“ neu laufen lassen, „(1 failed before)“; „DeepSeek-V4 Mini8“, „the second DeepSeek-V4 series“ intern.
11. „bit-identical“ für den gruppierten Pfad → „identical greedy text, token count and acceptance“ (Logprob-Gleichheit nur am 8-Schicht-Ausschnitt).
NIT: Frozen-Prüfung unter vorhandenem tune_mutex; kReuse-Lücke fp8_dense_tune_max_m > awq_warmup_max_m dokumentieren; Text: „per layer (TP1,
8 local groups)“, „each warmup decode size up to 16“, „within the tuning limits“, fp8.py:41 „51 µs mit Tuning (125,4 ohne)“; 35,7k „7,0 s in all
arms“; Prosa fm-a 55/56 ms; Speicher −36/−16/−36 MiB; Rücksprung-Schalter sm70_fp8.native.fp8_grouped_bmm_decode=false nennen; PYTHONPATH-Zeile
(vLLM-Baum verschieden); übrige Optionen gleich; „plus the first four commits“; Freeze still ignoriert ohne awq_warmup/ohne quantisierte
Schichten/nicht SM70 → loggen oder dokumentieren.
