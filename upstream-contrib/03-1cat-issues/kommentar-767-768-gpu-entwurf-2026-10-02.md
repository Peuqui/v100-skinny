Entwürfe 2026-10-02 abends, NICHT gesendet (Freigabe Peuqui abwarten).
Anlass: yangzhuxinyzx hat #743 als 1Cat #767 und #710 als 1Cat #768 integriert und
schreibt jeweils, dass GPU-Tests bzw. kalter/warmer Modellstart nicht nachgefahren wurden.
Vor dem Senden: Testlauf und main-Vergleich gegen das DANN aktuelle main wiederholen (main
bewegt sich heute im Minutentakt; Stand der Zahlen: main 90235b3c).
Belege: /tmp/…/scratchpad/pr767_768_tests.sh (Ausgabe tasks/bhk8hogrn.output),
git range-diff gegen 400baa11 (#743) und c3d29f3a (#710), llama-swap-Journal ab 27.09.

==================== Kommentar in 1Cat #767 ====================

Ran this branch (c0c8f943) on our rig, one pytest process per file, with native
extensions from our fork's build (main d3046986 plus our open PRs; main's csrc is
unchanged since d3046986).

tests/quantization/test_sm70_mxfp4_moe.py: 33 passed on a Tesla V100, 28 passed and 5
skipped on a Quadro RTX 8000. The five tests the merge gate skips are the V100-only ones
("requires NVIDIA V100/SM70"); they pass on the V100, including the allocation bound of
the repack test.

git range-diff against #743 shows the repack commit unchanged apart from your sign-off.

==================== Kommentar in 1Cat #768 ====================

Ran this branch (512b213f) on our rig, one pytest process per file, with native
extensions from our fork's build (main d3046986 plus our open PRs; main's csrc is
unchanged since d3046986).

The six modules from the test plan of #710 (test_sm70_fp8_workspace_aot_reload,
test_sm70_fp8_prefill_exact_dense, test_sm70_fp8_qpn8_pp2_tp4,
test_sm70_ct_fp8_cache_workspace, test_sm70_turbomind_adapter,
test_qwen3_5_quantization): 71 passed on a Tesla V100 and 71 passed on a Quadro RTX 8000.
On main 90235b3c the same modules give one failure,
test_compressed_tensors_channel_fp8_qpn8_prepares_and_dispatches, the CPU-backend case
described in #710.

On cold/hot model execution: git range-diff shows the workspace commit unchanged from
#710 apart from your sign-off, and our fork has carried it since 2026-09-27. Since then
Qwen3.8-27B-NVFP4 (TP2 with MTP, compile cache on) started warm from its AOT artifacts
on 2x Quadro RTX 8000 and served requests 19 times, without this error. Earlier that
day, before the change, four warm starts of it failed with "The specified pointer
resides on host memory and is not registered with any CUDA device." On Turing its FP8
layers reach these sites through the QPN8 route of #604, so those runs are our fork with
#604 on top, not this branch alone.
