Status: GESENDET 2026-10-02 18:1x nach Go von Peuqui — Push 7a4c665cf auf pr-fp8-block-qpn8, Kommentar mit Stapel-Ergebnis (Text: scratchpad comment750.txt, unten nachgetragen).
Anlass: #750 führt Fp8LinearMethod.use_qpn8 ein; tests/model_executor/test_sm70_fp8_qpn8_pp2_tp4.py
baut die Methode per __new__ und setzt nur use_marlin → 3 Tests rot mit AttributeError, auf dem
PR-Branch selbst (59329bec) und im Fork seit 01.10. Beim Einreichen nur die neue Testdatei
gelaufen. Fix-Commit lokal im Worktree 1Cat-vLLM-pr-qpn8: 7a4c665cf (nur Test, 3 Zeilen),
merge-tree gegen main 3e6d0a3c sauber. Push (kein Force): git push fork pr-fp8-block-qpn8
Gegenprobe des Stapels: scratchpad/stack750_tests.sh (Ausgabe tasks/boh3xzuuv.output).

==================== Kommentar in 1Cat #750 ====================

Pushed 7a4c665c, a test-only fix I missed when opening this PR:
tests/model_executor/test_sm70_fp8_qpn8_pp2_tp4.py builds Fp8LinearMethod through __new__ and
sets the attributes process_weights_after_loading reads. With this PR that includes use_qpn8, so
three of its tests failed with an AttributeError. They now set use_qpn8 = False, as __init__
does. 13 passed on a Tesla V100 and on a Quadro RTX 8000.

[STAPEL-ERGEBNIS NACHTRAGEN: bestehende SM70-/FP8-/QPN8-/DeepSeek-V4-Testmodule auf dem Stapel
gegen main 3e6d0a3c, V100]

==================== GESENDETE FASSUNG ====================

Pushed 7a4c665c, a test-only fix I missed when opening this PR: tests/model_executor/test_sm70_fp8_qpn8_pp2_tp4.py builds Fp8LinearMethod through __new__ and sets the attributes process_weights_after_loading reads. With this PR that includes use_qpn8, so three of its tests failed with an AttributeError. They now set use_qpn8 = False, as __init__ does. 13 passed on a Tesla V100 and on a Quadro RTX 8000.

I also ran the other SM70, FP8, QPN8 and DeepSeek-V4 test modules (49 files, one pytest process each, Tesla V100) on this branch against current main (3e6d0a3c). Apart from this PR's own tests, the only differences come from this branch's older base: test_compressed_tensors_channel_fp8_qpn8_prepares_and_dispatches, which #773 fixed on main, and the repack test #773 added. Merged into current main together with our other open PRs, both of those modules pass.
