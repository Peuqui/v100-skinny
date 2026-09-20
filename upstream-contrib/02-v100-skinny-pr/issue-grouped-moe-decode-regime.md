# Entwurf: Issue an dnv2003/v100-skinny — gebündeltes NVFP4-MoE im Decode, Kachel läuft zu ~80 % leer

Status: ÜBERHOLT — die Frage steht im Text von PR #8 (https://github.com/dnv2003/v100-skinny/pull/8), dieses Issue wird NICHT gepostet. War: ENTWURF vom 20.09.2026. Erst nach Peuquis Freigabe posten
(`gh issue create -R dnv2003/v100-skinny --title … --body-file …`). Kein Fettdruck im
Text. Zahlen-Quellen: `scripts/nvfp4_skinny_moe_qpn_test.py` (V100 + RTX 8000, echte
DeepSeek-V4-Flash-Gewichte Schicht 5), `benchmarks/nsys-dsv4-decode-step-anatomy-2026-09-20.txt`,
`benchmarks/torchprof-dsv4-decode-pp0-2026-09-20.txt`, Scratchpad-Mikrobenchmark vom 19.09.
(Schleife gegen moe_qpn bis 4096 Tokens; Zahlen stehen im Fork-Commit 0b8604bb).

Wichtig für den Ton: `moe_qpn` und `moe_simt` sind UNSERE Kernel (Fork-Ergänzungen in
`kernels/skinny_kernels.cu`, Branch `work` auf Peuqui/v100-skinny), gebaut auf seinem
`MMA_8N8K4`-Primitiv und seinem QPN-Prepack. In seinem Upstream gibt es kein MoE. Das
Issue ist also (a) eine Frage an den Kenner des Instruktionssatzes und (b) ein Angebot,
die MoE-Kernel upstream beizusteuern.

## Titel

Grouped NVFP4 MoE on the QPN tensor-core path: decode runs the m8n8k4 tile about 80 % empty — is there a better shape for 1-2 rows per expert?

## Text

Hi @dnv2003,

this is a question about the tensor-core path, plus an offer.

We serve NVFP4 MoE models (DeepSeek-V4-Flash, 256 experts, top-6, experts 4096x4096 and
4096x2048) on a mixed box, two Quadro RTX 8000 and three Tesla V100, through a fork of
this repository. Your dense QPN kernels carry the linear layers. For the experts we
added a grouped kernel on top of your building blocks: `skinny_nvfp4_moe_qpn` in
kernels/skinny_kernels.cu on the `work` branch of Peuqui/v100-skinny. It keeps your
`_qpn_prepack` fragment order per expert and your `MMA_8N8K4` inner loop, takes compact
device-side routing (slots sorted by expert, one grid row per group, padding groups
exit before touching weights) and is CUDA-graph safe. An expert with more than 8 slots
takes one pass per 8 rows.

Where it stands, on real layer-5 experts, w13 plus w2 plus activation, per layer:

- Decode and speculative verify, 6 tokens (36 slots, about 29 experts):
  0.83 ms on V100, 0.97 ms on RTX 8000. A SIMT variant of the same routing
  (`skinny_nvfp4_moe_simt`) measures 1.12 and 1.19 ms, so the tensor-core path wins
  by 1.3x even at one or two rows per expert.
- Prefill chunks: 128 tokens 5.5 ms (V100) and 6.5 ms (RTX 8000), 512 tokens 12 and
  16 ms, against 32 to 38 ms for a per-expert loop over `gemm_qpn`.
- Same output as the per-expert loop to fp16 rounding (max abs diff 2.4e-4 at values
  around 0.3); scripts/nvfp4_skinny_moe_qpn_test.py checks 1 to 512 tokens including
  the multi-pass, on both cards.

With that kernel a decode step of the whole model (43 layers plus a 3-layer drafter,
pipeline over five cards, 5 draft tokens) is 81 ms, and an nsys timeline shows about
69 ms of it as serial kernel time. Roughly half of that is this MoE kernel: 0.75 ms of
the 1.4 to 1.5 ms per layer. So it is now the single largest item.

The part I would like your view on: at 6 verifier tokens an expert gets one or two of
the eight rows of the tile, so the m8n8k4 instruction runs mostly on zero activations.
The memory-bandwidth floor for the expert bytes that have to be read (29 experts x
12 MB per layer) is about 0.39 ms on V100 and 0.52 ms on RTX 8000, so the kernel is 1.5
to 2x above it. Questions:

1. Is there a tensor-core shape that fits 1 to 2 activation rows better than m8n8k4,
   or a way to let several experts' rows share one instruction although their B
   operands differ? As far as I can see the quadpair shares one A tile across four
   N-slices, which is the opposite of what this regime needs.
2. In your M sweep SIMT won for M <= 3 on dense GEMMs. In the grouped case it loses
   (numbers above). Do you see why, or a SIMT layout for the prepacked fragments that
   should do better?
3. Would you take the grouped MoE kernels upstream? They are self-contained in
   skinny_kernels.cu (two kernels, two pybind entries, one test script). I can send
   them as a focused PR, unlike my earlier #7.

Harness, numbers and the kernel are on the `work` branch; I am happy to run anything
you want measured on either card generation.

AI assistance (Claude) was used for the kernels, the measurements and this text; I have
run everything above on my own hardware.
