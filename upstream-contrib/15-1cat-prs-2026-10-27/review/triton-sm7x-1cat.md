# Unabhängige Prüfung 1Cat-Triton-Routen-PR (27.10. ~12:15): READY AFTER FIXES

Geprüft und in Ordnung: Schalter an + Stock-Triton → RuntimeError beim Bau einer Schicht, die den Schalter nutzt (andere Modelle laufen
ohne Wirkung); Gerätewahl je Schicht/Tensor; mHC-Kompensation; Commit 3 unabhängig vom Schalter; Revert-Prüfung; Triton-Aussagen.
BLOCKER
1. DCO: Signed-off-by fehlt (hier 4/10; übergreifend fast überall) → Schlussdurchgang `git rebase --signoff` je Zweig, dann Hashes im Text.
SHOULD-FIX
1. Basis-PR (DSv4) liefert GPU-rote Tests, die Korrektur steckt in 26062dddc (dieser PR): _reference_prefill_logits/_assert_close_to_max
   + die zwei Teständerungen (test_prefill_model_head_counts_match_the_reference, test_turing_prefill_takes_the_fused_scorer) in den DSv4-PR.
2. Turing-Indexer: indexer.py:778 `if policy.value("indexer_relu") and not triton_tensor_cores` → mit Schalter nimmt Turing nie den
   nativen Scorer, ist aber summiert langsamer (+1,0 % bei 64 Köpfen, +4,7 % bei 32). Fix: Turing behält den nativen sm75-Scorer
   (oder Wahl nach Schlüsselzahl); sonst Text ehrlich.
3. Volta-MoE-Zahlen im Text aus der ersten Kernfassung (4e83ea6a9); Endstand 435b8a563: 1020 Token Zipf 0/0,8/1,2 9,085/9,900/10,332
   gegen Handkern 11,600/11,638/12,268; 2040: 14,673/15,092/15,335 gegen 21,175/21,267/21,390. Schwelle 16 Zeilen/Experte auf dem
   Endkern nie neu gemessen. Text „32- or 64-row“ → 16/32/64.
4. mHC: „equal or lower“ → „max error over 20 inputs lower on both cards (V100 lower on 16 of 20 inputs); split keeps 22 of 24 mantissa bits“.
5. Commit 3 (immer an) ändert Standardausgaben (0,02–0,04 % um eine fp16-Stufe) und trifft alle Skinny-QPN-Modelle (auch GLM) → im
   Text nennen; Pfadtabelle „Unverified scope: other skinny-QPN models not booted on this base“.
6. A/B-Offenlegung unvollständig: nativer sm_75-Bau der SM70-Familie (Turing-PR 2f70712b0/5028a3fc4) fehlt; alter Arm Pool 3000
   (neu 3500); sm7x-Triton macht ALLE vorhandenen tl.dot-Kerne zu MMA (unabhängig vom Schalter) → „Schalter aus = keine Änderung“ gilt
   nur mit Stock-Triton.
7. Turing-Durchlaufgröße nach Architektur, begründet mit Kartengröße (sparse_prefill_bmm.py:53-65 `_tc_tiles`): T4 16 GiB / V100 16 GiB
   → gegen hardware-agnostische Regel; nach Gerätespeicher oder KernelConfig wählen, mindestens Kartengrößen-Aussage streichen;
   Workspace-Aussage 110 → 30 MiB gilt nur Volta (Turing 1024er-Durchlauf 240 MiB).
8. Commit 6 (Volta-Triton-MoE) hängt funktional an Commit 3 (Chunkgrenze 682 → 1365; ohne 3 greift die Triton-Route bei DSv4 nie) → nennen.
NIT: Testdatei-Name test_triton_sm7x_tensor_cores veraltet; interne Namen/Daten im Code (indexer.py:694 sweep_idx_table_…, 22.10.2026;
sparse_prefill_bmm.py:56 23.10.2026) entfernen; ef5816176 in d354f36e3 einfalten; fremde Lambda-Umformatierung in
test_deepseek_v4_sm70_routes.py:155 (a65db609a) entfernen; kernel.py:744 Gerät selbst auflösen statt device_index durchreichen, Methode
zwischen Datenfeldern; Fehlertext ohne Bezugsquelle; _selected_pv_kernel prüft output.stride(-1)==1 nicht; _mtp_weighted_reduce
Kontiguität annehmen (assert); tests/kernels/utils.py:35 Probe beim Import; alter Arm 35,7k 12,5–12,6 (a) / 12,4–12,6 (b);
Speicher GPU 0 +380 MiB auch nennen; Decode tok/s ergänzen (178k 43,1/41,4 → 43,2/48,4; 287k 33,6/34,2 → 36,6/37,1); mma.sync-Prüfung
war am früheren Bau 7b8fb5387 (Cache nachgeprüft: hält); Duplikat-Absatz: #1030/#1031/#1243 (skinny_sm70_moe.py), #1166/#1159 (mHC),
#1125/#1092 (GLM sparse MLA) als Nicht-Duplikate nennen.
