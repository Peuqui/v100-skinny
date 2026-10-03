Entwurf 2026-10-03 nachmittags. Darf ich eigenständig posten (wasserdicht, AGENTS.md, Peuqui 03.10.:
#742-Ergebnisse ungefragt posten). Vor dem Senden: Zustände frisch abfragen (main-SHA, #837/#742 gemergt,
offene PRs), Platzhalter <...> nur aus Messdateien füllen, Absätze zu Zeilen zusammenfügen.
Belege: quality_2026-09-27/pr837-kerneltests-2026-10-03.log, ab_837_2026-10-03.log (Runde 1),
ab_837b_2026-10-03.log (Runde 2), Journale quality_2026-09-27/pr837-*.journal.txt, Greedy/Qualität
pr837-*.greedy.json / pr837-*-ds.json (von Hand gelesen).

==================== Kommentar in 1Cat #837 ====================

Native results for this integration on our rig (3x Tesla V100-PCIE-32GB + 2x Quadro RTX 8000, CUDA 12.8, Torch 2.10.0+cu128). The extensions were built from source at the PR head 035be3644 through the repository's CMake target with TORCH_CUDA_ARCH_LIST=7.0, the same sm_70-only build we deploy; the RTX 8000 runs those sm_70 cubins. The skinny MoE files are identical between 035be3644 and the merged head 61db7c02b, and current main (<SHA>) differs from them only in the compile-hash bookkeeping of #847.

SM75 operator cases: tests/kernels/moe/test_skinny_sm70_moe.py and test_unquantized_backend_selection.py with CUDA_VISIBLE_DEVICES set to one RTX 8000 and one V100 (CUDA_DEVICE_ORDER=PCI_BUS_ID): 91 passed, 1 skipped (the ROCm selection case). All 13 sm75 cases pass on the RTX 8000, and the 13 sm70 cases pass on the V100.

Complete models, on the PR head plus two Python-only fixes that main needs on this rig (PRs <PLE-PR> and <SPARSE-PR>): with moe_backend left at auto, the oracle selected the skinny experts on its own. Stage 0 of each pipeline is a Quadro RTX 8000 and logged "Using 'SM70_SKINNY' NvFp4 MoE backend" for Qwen3.8-Flash-Next-NVFP4 and "Using 'SM70_SKINNY' Mxfp4 MoE backend" for DeepSeek-V4-Flash (MXFP4, 256 experts, one scale per 32 codes). The other stages log nothing because info_once is local-rank only; <Beleg für die übrigen Stufen aus Runde 2>.

Paired comparison on the same build, Qwen3.8-Flash-Next-NVFP4 at PP4 (RTX, RTX, V100, V100; MTP with 4 draft tokens; 29k-token prompts; the first boot of each arm and the first long request after each boot excluded): default (skinny) against --kernel-config '{"sm70_skinny_moe":false}':
<Tabelle: Prefill 29k s, Decode-Schritt kurz/lang ms, tok/s; je Arm Spannweite über die Boots>

Quality: greedy outputs and our eight reference questions, read by hand. <Ergebnis>. Against our fork, which runs the original #742 path with tuned split-K settings, three greedy prompts gave one identical output and two equivalent rewordings at both PP4 and TP2xPP2; the arithmetic answers agree.

DeepSeek-V4-Flash at PP5 (MXFP4 experts, DSpark drafter): <Start, Greedy, Prefill 35.7k, Decode-Schritt gegen unseren Fork>.

AI assistance was used for running and analysing these measurements. Commands: <ab_837.sh/ab_837b.sh beschreiben: llama-swap-Einträge, Bench-Skripte>.
