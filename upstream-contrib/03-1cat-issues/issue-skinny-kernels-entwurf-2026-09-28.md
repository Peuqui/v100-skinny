Title: Proposal: bring the v100-skinny NVFP4/MXFP4 kernels into csrc (Turing MXFP4 MoE for DeepSeek-V4, faster NVFP4 MoE prefill)

Before opening a large PR I would like to ask whether you want this, and in
which shape.

What it is. v100-skinny (https://github.com/dnv2003/v100-skinny, MIT, by
dnv2003, with our extensions, submitted there as dnv2003/v100-skinny#8) is a
set of hand-written CUDA kernels for weight-only FP4/FP8 on Volta and Turing:
skinny NVFP4 GEMMs (WMMA/mma.m8n8k4, M <= 64), grouped NVFP4 and MXFP4 MoE
kernels with device-side routing (moe_qpn, moe_simt), QPN8 kernels for block
FP8, and block-packed activations. About 3,000 lines in one .cu file, about 20
ops, built for sm_70 and sm_75.

What it does on our rig (3x V100 + 2x Quadro RTX 8000), all greedy outputs
identical across the compared backends:

- DeepSeek-V4-Flash (DeepSeek's MXFP4 checkpoint) with pipeline parallelism
  over all five cards: the RTX 8000 stages need an MXFP4 MoE path on Turing,
  and main has none; the skinny MoE is what makes those stages run.
- Qwen3.8-Flash-Next NVFP4, TP2 x PP2, cold 18k prefill: TurboMind MoE
  26.6-27.2 s, skinny MoE 19.3-19.4 s, Marlin 17.7-17.8 s; decode step
  53-56 ms, 52-55 ms and 56-59 ms respectively (code prompts 77-78 tok/s with
  skinny, 58-70 with Marlin).

How it runs today in our fork: the vLLM side (an NVFP4 linear kernel, an
`sm70_skinny` MoE backend, QPN8 for block FP8) compiles the .cu at run time
with torch.utils.cpp_extension.load from a path set by an environment
variable. That is fine for a fork, not for 1Cat-vLLM.

Proposal:

1. Kernels into csrc as their own extension (for example `_skinny_C`, sm_70
   and sm_75), with an MIT notice naming dnv2003 in the file; ops registered
   under torch.ops like the other SM70 extensions, with unit tests against a
   torch reference on both card generations.
2. Then the vLLM integration in separate PRs: the MoE backend (the part that
   unlocks Turing for DeepSeek-V4), the NVFP4 linear route, QPN8 for block FP8.

Questions:

- Do you want these kernels in 1Cat-vLLM at all, given that TurboMind and
  Marlin already cover Volta?
- Is a separate extension in csrc the right place, or would you prefer them
  under csrc/sm70_turbomind/ or elsewhere?
- One PR for the kernels plus a first consumer, or kernels alone first?

AI assistance was used to prepare this proposal and the measurements.
