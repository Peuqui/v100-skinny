The MoE part is now up as #PRNUMMER: --moe-backend sm70_skinny for NVFP4 and MXFP4 MoE on SM70 and SM75. To your placement question I went with what fits best today: the two kernels sit in csrc/sm70_turbomind/ops/skinny_moe_qpn_sm70.cu next to the QPN2 code from the same project and are registered in _C like the other SM70 ops, not as a separate extension. The backend is opt-in; automatic selection does not change.

One correction to the proposal above: main does have an MXFP4 MoE path on Turing. Marlin serves the RTX 8000 stages of DeepSeek-V4-Flash, and the PR compares against exactly that (TurboMind on the V100 stages, Marlin on the RTX stages). What does not fit on our five cards is the default path together with the DSpark drafter; the numbers are in the PR.

@valentijnvenus this is the first of the pieces you asked about; QPN8 for block FP8 follows as its own PR. @dnv2003 the kernels come from your v100-skinny with our MoE additions from dnv2003/v100-skinny#8, and the file carries the MIT notice with your name.

Status: gepostet 2026-09-30 abends (#PRNUMMER = #742)
