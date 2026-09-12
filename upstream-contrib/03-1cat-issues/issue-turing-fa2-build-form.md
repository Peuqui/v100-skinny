# Entwurf 1Cat-Issue: Bauform für die sm75-FlashAttention-2 (NICHT veröffentlicht, Freigabe Peuqui)

Titel: [RFC][SM75] Bundling a Turing (sm75) FlashAttention-2 build: which form do you prefer?

---

Turing GPUs (RTX 8000, RTX 6000, T4, RTX 20xx) currently get no FlashAttention backend in 1Cat-vLLM. CMakeLists.txt fetches vllm-project/flash-attention for 8.0+ and zhinianqin/flash-attention-v100 for 7.0, so an sm75-only build or a mixed 7.0;7.5 build ends up without FA2 kernels for Turing. At runtime the platform then offers FLASHINFER first, which fails in BatchPrefillWithPagedKVCache with "invalid argument" on sm75, and TRITON_ATTN as the working fallback.

We have an FA2 fork that builds and runs on sm75: https://github.com/Peuqui/flash-attention, branch sm75-enablement-pr, on top of the vllm-project pin 28e862d. It touches 7 files (232 insertions, 31 deletions): FA2_ARCHS extended to 7.5 with bf16 excluded for that arch, the forward path enabled for sm75 in flash_api.cpp, flash_fwd_kernel.h, flash_fwd_launch_template.h and static_switch.h, a kMmaPerCopy fix in utils.h that is latent upstream for head dims above 64 on sm75, and split-KV for paged multi-token queries so speculative decoding verification runs through the same kernels. It is fp16-only and forward-only, which matches what an inference server needs.

Measured on a pair of RTX 8000 (TP2) with Qwen3.8-27B-NVFP4, MTP k=3, greedy, fp16 KV cache, on current main plus PRs 604 and 611 (the Turing route for the linears): the 400-token probe goes from 70.91 tok/s (TRITON_ATTN) to 74.19 tok/s (FLASH_ATTN via the sm75 build) with identical output text. With a 13k-token prefix, time to first token goes from 36.26 s to 17.75 s and the decode rate from 15.03 to 56.48 tok/s, again with identical output text.

Two FA2 libraries have to coexist in one wheel, because a mixed Volta/Turing rig needs the sm70 build and the sm75 build side by side. Both FA2 projects define the same target name _vllm_fa2_C and vllm_flash_attn.cmake warns that the FA project overwrites global CMake functions, so a second FetchContent of the same project collides. The sm75 library therefore has to come in as a separate build. We see three forms and would like to know which one you want before we open the PR:

1. ExternalProject_Add in a new cmake/external_projects/vllm_flash_attn_sm75.cmake, included only when 7.5 is in CUDA_ARCHS, pinned to a tag on Peuqui/flash-attention. The fork gets a small CMake option that renames the extension to _vllm_fa2_C_sm75 (default unchanged). setup.py adds the extension when the arch list contains 7.5, so it is built and installed like every other component. This mirrors how the sm70 build already pins zhinianqin's fork. Cost for you: a build-time dependency on a repository outside your organization.

2. Same as 1, but the pinned repository is a fork under the 1CatAI organization that you take from us. Your build then does not depend on an external account, and you control when the pin moves.

3. Vendoring the sm75 FA2 source tree into this repository, the way flash-attention-v100 is vendored today. This is the only form that needs no external repository at all; the tree is large (the full FA2 source plus the CUTLASS submodule), so it is the most to carry in this repository.

We will build whichever form you pick. The Python side is the same in all three cases: a per-device loader in vllm_flash_attn/flash_attn_interface.py that picks the library for the capability of the worker's own GPU, FLASH_ATTN ordered first on (7,5) in platforms/cuda.py, and the FlashAttentionBackend capability floor lowered to (7,5) with an fp16 gate. That part is ready and lint-clean against main.

Related: PR 604 and PR 611 (Turing route for the quantized linears), issue 441 (pre-Ampere tuning findings).

This work was done with AI assistance (Claude); every change is reviewed, built and measured by me on the hardware named above.
