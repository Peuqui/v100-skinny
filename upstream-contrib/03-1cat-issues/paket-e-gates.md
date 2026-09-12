# Gate-Karte Volta-Helfer in 1Cat main ae75fb9b (Agent-Bericht 12.09.)

Helfer in `sm70_turbomind.py`: `is_exact_sm70_cuda(tensor, enabled)` (Tensor-
Gerät, `== (7, 0)`), `is_exact_sm70_cuda_platform()` (Gerät 0 / Plattform),
`should_prepare_turbomind(_or_marlin)` (Tensor-Gerät + `use_sm70_turbomind`),
`should_use_{mxfp4,nvfp4}_moe_turbomind` (Plattform). Apply-Seite ist
zustandsgetrieben (`STATE_ATTR`), ohne Capability-Prüfung.

## (a) Linear-GEMM-Teilmenge (Kandidaten für E-1)
- NVFP4 modelopt: `modelopt.py:1102` (prepare), `:461`, `:1197-1199`,
  `:2405-2407` (min-capability / init, Gerät 0)
- NVFP4 compressed-tensors: `compressed_tensors_w4a4_nvfp4.py:313` (QPN4-Wahl
  darin), `compressed_tensors_w4a16_nvfp4.py:96`
- FP8 / QPN8: `compressed_tensors_w8a16_fp8.py:145` (Gerät 0, pro Schema-
  Objekt), `fp8.py:538-548, 730-743`, `sm70_online_qpn8.py:138` (Gerät 0)
- AWQ: `awq_marlin.py:104` (umgeht `use_turbomind`), `:300-310`;
  `awq.py:443-450` (roher Tensor-Check), `:179-180`, `:634` (Gerät 0)
- MXFP4 linear: `compressed_tensors_w4a4_mxfp4.py:103` (+`:44,50-51`)
- uint4 TurboMind: `compressed_tensors_wNa16.py:230`, `auto_gptq.py:454`
- Apply-Stellen ohne Änderungsbedarf: `modelopt.py:1386,1537`,
  `w4a4_nvfp4.py:469,480`, `w4a16_nvfp4.py:123`, `w4a4_mxfp4.py:130`,
  `wNa16.py:297`, `auto_gptq.py:511`

## (b) Nicht anfassen
MoE: `mxfp4.py:816`, `mxfp4_sm70_moe.py:769`, `nvfp4_sm70_moe.py:1472`,
`modelopt.py:1174,2602,2620`, `fp8.py:612-628,1905`, `awq.py:216,254`;
MTP-Config `qwen4_exp/nvidia/mtp.py:202,211`; Warmup; gesamter
Attention/Indexer/Runtime-Block (Indexer, deepseek_v4, glm5next, qwen4_exp
qsa/hc/ple, gpu_model_runner, cudagraph, spec_decode, layernorm, ...).
`marlin_utils_fp4.py:31` ist Marlin, sm75 dort schon über
`has_device_capability(75)`.

## Env-Helfer
`envs.py:973 use_sm70_turbomind(default)` liest `VLLM_SM70_QUANT_BACKEND`
(`auto|marlin|turbomind`) über `get_sm70_quant_backend()` (`:964`); `auto` →
Feature-Default (`VLLM_SM70_{NVFP4,FP8,AWQ,MXFP4,GPTQ,COMPRESSED_TENSORS}_TURBOMIND`).
`envs.py:982 force_sm70_marlin()` → true bei `marlin`.

## C++-Laufzeit-Gates (12.09., eigener Fund)
`arch.h`: `Sm70: Arch<700, 750>` — TurboMind-Registry lässt Turing (750)
leer. `TORCH_CHECK(major == 7 && minor == 0)` in `awq_qpn_m1_sm70.cu:345`,
`glm53_cublaslt_sm70.cpp:95`, `exact_row_reduce.cu:143`, `h3_w8a16.cu:126`
(nicht NVFP4/QPN2/QPN8). `gemm.cu:184,237` `desc.arch == 700` nur für
Sonderpfade.
