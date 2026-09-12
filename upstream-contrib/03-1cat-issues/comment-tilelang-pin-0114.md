# Entwurf: Pin-Vorschlag tilelang 0.1.14 an 1Cat (Issue/Diskussion, NICHT gepostet)

Stand 12.09. nachts. Freigabe Peuqui ausstehend. Ort: neues Issue oder
Kommentar in einem passenden bestehenden Thread (zu klären: gibt es bei 1Cat
einen Thread zu tilelang/apache-tvm-ffi-Pins? `gh issue list --search
tilelang` vor dem Posten).

## Text (Englisch, ohne Markdown-Schnörkel)

Proposal: pin tilelang 0.1.14 (and its apache-tvm-ffi) instead of 0.1.10

requirements/cuda.txt and rocm.txt pin tilelang==0.1.10 with
apache-tvm-ffi==0.1.10. With 0.1.10 the CUDA target is detected from device
0 (tilelang/utils/target.py), so on a node that mixes card generations a
worker on a different card compiles its MHC kernels for the wrong
architecture; we carry a local patch for that. tilelang 0.1.14 resolves the
target from torch.cuda.current_device() (tilelang/cuda/target.py,
_detect_torch_cuda_arch), which makes that patch unnecessary.

What I ran with 0.1.14 in an otherwise unchanged environment (2x Quadro RTX
8000 + 3x Tesla V100, torch 2.10.0+cu128, CUDA 12.8 toolkit for the JIT):

- tests/kernels/test_mhc_kernels.py: V100 43 passed / 8 skipped, RTX 8000
  43 passed / 8 skipped.
- tests/kernels/test_mhc_sm70_fp16.py: V100 24 passed / 1 skipped, RTX 8000
  10 passed / 15 skipped.
- DeepSeek-V4-Flash NVFP4, TP1 x PP5 across all five cards, DSpark k=5:
  boots, the eight-prompt coherence probe run twice in the same server
  process gives byte-identical answers (8/8), code prompt 26.5 tok/s
  (reference with 0.1.10: 26.7).
- Qwen3.8-27B NVFP4 (GDN model) with the DFlash2 draft head, TP2 on the RTX
  pair: identical greedy text (SHA-256 of 400 tokens equal to the 0.1.10
  run), acceptance length 3.325, 77.1 tok/s.

One environment note for others trying this: tilelang's JIT calls nvcc from
CUDA_HOME; with a system nvcc 12.0 and gcc 13 the compile fails with
"unsupported GNU version", so CUDA_HOME has to point at a 12.8 toolkit.

If you prefer to stay on 0.1.10, the alternative is a one-line change in
tilelang's target detection, which is what our patch does; I can open that
upstream at tilelang instead.

AI assistance (Claude) was used to run the test matrix; I reviewed the
results.
