## Purpose

Stacked PR on #1286 (GLM-5.3-Flash-FP4-Local on SM70 and SM75). The branch is the head of #1286 (f7df3f270) plus three commits; please review only those (git log f7df3f270..ceceb328e). The first two change GLM-5.3 paths and have run in our deployment since 2026-10-27: PP=5 over three Tesla V100 and two Quadro RTX 8000 with DFlash2 and 524,288 tokens of context (docs/design/sm70_glm53_pp5_dflash2_mixed.md from #1286). The third makes the second safe for a padded vocabulary.

KDA prefill autotune in the profile run (a4dfcec45). The Triton autotuners of the chunked KDA prefill kernels (chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter and _intra_sub_intra) key on chunk constants and dtypes, not on the sequence length, and GLM-5.3's linear attention returned early in the profile run. Whatever request first reached a KDA prefill therefore fixed the choice, and vLLM keeps it on disk (TRITON_CACHE_AUTOTUNING=1 by default). The choice matters only at prefill length. At 16 tokens all 24 configurations of the KKT sub-chunk kernel tie: 0.0021 to 0.0035 ms on a Quadro RTX 8000 and 0.0030 to 0.0032 ms on a Tesla V100 (CUDA graph, GLM-5.3 shape 64 heads x 128). At a 512-token chunk they span 0.2340 to 1.4057 ms (RTX 8000) and 0.1485 to 0.9854 ms (V100); the outputs of all configurations are identical. In an nsys profile of an 8,766-token prefill, a cache that held the slow choice (BK 64, one warp, three or four stages) spent 268 instead of 48 ms of kernel time in this kernel on the first stage, an RTX 8000. The profile run now calls the real prefill path once per layer with inputs of its own token count (the batch budget, 512 in our entry) and the dtypes of a real prefill, which are part of the autotune keys. A Triton cache that already holds a choice keeps it; a fresh cache gets one tuned at the prefill length. In our A/B/A/B (Test Result) every fresh cache of the changed arm held the fastest 512-token configuration (BK 32, one warp, two stages), with tuning timings of 512-token magnitude, and prefill matched the reference arm. This removes the dependence on the first request; it does not make a cache that already holds the good choice faster.

MXFP8 LM head through QPN8 (a9da0755f). The checkpoint stores lm_head as MXFP8 (fp8 e4m3 [154880, 4096], E8M0 scales [154880, 128]). ModelOptMixedPrecisionConfig skipped ParallelLMHead for MXFP8 and the loader dequantized the pair to fp16, so the last pipeline stage ran two dense fp16 GEMMs per round, the target head and the draft's shared head (2.13 ms each in an nsys profile of our deployment). The head is now built with ModelOptMxFp8LinearMethod like every other MXFP8 linear of the model, and the loader keeps its weight pair packed (_mxfp8_held_modules also finds the top-level module: lm_head in the checkpoint, language_model.lm_head in the multimodal wrapper). The MXFP8 kernel is chosen per layer shape in create_weights (#1286): the head's partition (154880, 4096) passes QPN8's shape check (N % 32 == 0, K % 16 == 0, split-K 16 over 256 code groups), so Qpn8Mxfp8LinearKernel serves it on Volta and Turing. On a Quadro RTX 8000 with the real checkpoint tensors, against the dense fp16 product of the dequantized weights: logits differ by at most one fp16 step (0.0078 at values near 13), bit-identical at M 9, argmax equal in 256 of 256 random rows; one call in a CUDA graph takes 1.11 to 1.17 ms instead of 2.08 to 2.17 ms (M 1 to 16), and the weight shrinks from 1.18 GiB to about 0.63 GiB.

The DFlash2 candidate top-k then runs over a quantized head, which compute_candidates in qwen3_dflash2.py only allows with speculative_config.sm70_dflash2.quant_lm_head (dense logits through quant_method.apply, then top-k). This PR uses that existing field with its existing meaning; it is on by default only for the qualified Qwen3.8 verifier contract, so a GLM-5.3 DFlash2 configuration has to set it, otherwise the profile run stops with "DFlash2 requires an unquantized target LM head for candidate TopK; got ModelOptMxFp8LinearMethod". The PP5 note of #1286 now says so.

Padded vocabulary (ceceb328e). A vocabulary that is not a multiple of the padding size (64), or a TP split that pads, leaves rows past the checkpoint in the LM head, and the vocab weight loader filled them with zero. For an E8M0 scale, zero is the code of 2^-127; QPN8 accepts exponents 95..134 only and stopped such a head at load. This applies to every MXFP8 head on SM70/SM75: through ModelOptMxFp8Config, the mixed-precision config above, or the compressed-tensors MXFP8 scheme. The loader now fills padding rows with the value a parameter names (padding_value, zero when it names none), and the MXFP8 scale parameters of the ModelOpt and compressed-tensors methods name 127, the scale 1. The padding codes stay zero, so the padded logits stay exactly zero for every MXFP8 kernel: QPN8 gets the group scale 1.0, emulation and FlashInfer multiply zero codes by 1, and Marlin's conversion of the scale through fp16 keeps code 127 (code 0 also stayed zero there). GLM-5.3 itself needs no padding (154,880 = 64 x 2,420).

Why one PR: the first two commits were measured together (A/B/A/B below) and serve together in our deployment, and the third completes the second. The head change alone first measured slower in prefill (8,687 tokens in 4.51 / 4.44 s against 4.34 / 4.26 s, 2026-10-25). An nsys profile on 2026-10-27 of a cache from that changed arm against one from the reference arm put the difference in the KDA kernel's autotune choice (268 against 48 ms above), while the head's own kernel time fell from 39.6 to 20.4 ms over the prefill. The commits do not depend on each other and can be split if you prefer.

Not a duplicate. On 2026-11-02 I searched the open PRs of this repository (gh pr list --state open --search) for KDA, autotune, "kda prefill", "lm_head MXFP8", "MXFP8 LM head", "lm head", quant_lm_head, GLM-5.3, glm53, Glm5Next, dequantize_mxfp8_weight_pairs and ParallelLMHead, and the merged and closed ones for KDA, the LM head terms, quant_lm_head and glm53. None changes the KDA profile run or the MXFP8 LM head. #1159 and #1165 (rivetphilbot) change vllm/models/glm5next/nvidia/kda.py in its decode and projection paths, not in the profile run or the prefill autotune; #1261 (LoRA lm_head for DFlash2) and #1218 (reduced draft vocabulary) concern other heads and paths; #1260 (GLM-5.3 GGUF) does not touch the KDA profile run or the ModelOpt LM head.

AI assistance was used for this change. I reviewed every line and ran the tests and measurements below.

## Test Plan

One pytest process per file on a Tesla V100 (SM70) and on a Quadro RTX 8000 (SM75), each as device 0, with stock Triton 3.6.0 and with the sm7x Triton build of #1282 first on PYTHONPATH:
python -m pytest tests/models/glm5next/test_kda_prefill_warmup.py
python -m pytest tests/models/glm5next/test_mxfp8_held_modules.py
python -m pytest tests/quantization/test_modelopt.py
python -m pytest tests/quantization/test_mxfp8_padded_lm_head.py
python -m pytest tests/kernels/quantization/test_sm70_qpn8_mxfp8_linear.py
python -m pytest tests/quantization/test_mxfp8_weight_pairs.py
python -m pytest tests/models/glm5next/test_pp.py
python -m pytest tests/models/glm5next/test_sm70_kda.py
python -m pytest tests/kernels/test_kda.py
python -m pytest tests/v1/spec_decode/test_dflash2.py
pre-commit run --from-ref 4d57473db --to-ref HEAD
pre-commit run check-layering --all-files
pre-commit run mypy-3.10 --all-files --hook-stage manual

New tests: test_kda_prefill_warmup.py checks that the profile run calls the prefill path once, at its own token count and with the dtypes of a real prefill (fp32 beta and state, int32 offsets). test_mxfp8_held_modules.py finds an MXFP8 LM head under its checkpoint name; test_modelopt.py serves a listed MXFP8 LM head with ModelOptMxFp8LinearMethod and leaves an unlisted one unquantized. test_mxfp8_padded_lm_head.py loads a head of 100 checkpoint rows, padded to 128, through the vocab weight loader: zero codes and scale code 127 in the padding, QPN8's exponent check passes (and fails for zero codes), on CPU up to the native packing; on SM70/SM75 the QPN8 logits are zero in the padding and match the dequantized weight elsewhere; the compressed-tensors MXFP8 scale names the unit scale too.

Full boot of this branch through our server entry (PP5, DFlash2, fresh compile and Triton caches), with a short request before the first long prefill.

## Path coverage

| Path / provider | Format | Key conditions (shape, TP/EP, graph, capability) | Expected selection / fallback | Evidence (source / static / executed / measured) | Unverified scope |
| --- | --- | --- | --- | --- | --- |
| GLM-5.3 KDA chunked prefill, profile run | fp16 activations, fp32 state | any card a KDA layer runs on; inputs of the profile run's token count | Triton autotune at that length, once per layer instance; a cache that already holds a choice keeps it | executed: unit test; measured: fork A/B/A/B and production, V100 and RTX 8000 | other KDA models (only GLM-5.3's layer changes); TP > 1 |
| ModelOpt MIXED_PRECISION LM head named MXFP8 in quantized_layers | fp8 e4m3 + E8M0 per 32 | SM70 or SM75, QPN8 operators built, sm70_fp8 enabled, partition passes fp8_qpn8_shape_reason | Qpn8Mxfp8LinearKernel | source; measured: fork, GLM-5.3 (154880, 4096) on an RTX 8000 last stage | TP > 1; a V100 last stage |
| same | same | QPN8 declines the shape, or sm70_fp8 disabled / force_marlin, or operators missing | next MXFP8 kernel of the registry: Marlin on SM75, emulation on SM70 (dequantizes the whole head on every call) | source only | all of it; slower than the former head dequantized at load |
| same | same | SM80 and newer | FlashInfer CUTLASS MXFP8 (SM100) or Marlin instead of the former fp16 copy | source only | all of it (no such hardware here) |
| same, and ModelOptMxFp8Config or compressed-tensors MXFP8 heads | same | vocabulary padded beyond the checkpoint rows (not a multiple of 64, or a TP split that pads) | padding rows load zero codes and scale code 127; QPN8 accepts them, padded logits are zero | executed: test_mxfp8_padded_lm_head.py (TP 1, 100 rows padded to 128) | a padded head in a model run (GLM-5.3 needs no padding: 154,880 = 64 x 2,420); TP > 1 |
| same | same | padded rows with Marlin, emulation, FlashInfer | zero codes times scale 1, padded logits zero as with the former zero scale | source; Marlin's fp16 round trip of codes 0 and 127 checked on CPU | executed runs on these kernels |
| DFlash2 candidate top-k over the target head | MXFP8 head | speculative_config.sm70_dflash2.quant_lm_head true | dense logits via quant_method.apply, then top-k | measured: fork production; source | none for GLM-5.3 PP5 |
| same | same | quant_lm_head unset (GLM-5.3 is not a qualified contract) | ValueError in the profile run, no silent route | executed: fork boot without the field | none |
| Greedy argmax (use_local_argmax_reduction) | MXFP8 head | SM70 LM-head top-1 state exists only for the embedding method | maybe_get_sm70_lm_head_top1 returns None; dense logits via QPN8 | source; executed in the boot | none |
| GLM-5.3 MTP (shared_head.head borrows the target head) | MXFP8 head | method mtp | logits via quant_method.apply | source only | not booted here; our entry uses DFlash2 |

## Acceleration and benchmark contract (required for performance changes)

- Default enabled or opt-in: both default enabled, no switch. The LM head is served quantized for every ModelOpt MIXED_PRECISION checkpoint that lists lm_head as MXFP8; DFlash2 on such a checkpoint needs speculative_config.sm70_dflash2.quant_lm_head (existing field).
- Required CLI options and environment switches: the PP5 entry of docs/design/sm70_glm53_pp5_dflash2_mixed.md, with "sm70_dflash2": {"verify_fastpath": true, "fused_gdn_metadata": true, "quant_lm_head": true} in --speculative-config; the fork A/B ran with --gpu-memory-utilization 0.972 against 0.98 in the reference arm (see Test Result).
- KV cache dtype used for the benchmark: fp8_e4m3 (target), auto (draft).
- Wheel SHA or source commit: fork measurements on github.com/Peuqui/1Cat-vLLM: both commits, changed arm 0181fdc6c (= 3f6bdcc1f + 0181fdc6c on our line, tag verified-2026-10-27-prod-v4b), reference arm tag verified-2026-10-27-prod-v4; the head alone, changed arm eee514533 (branch glm53-lmhead-test), reference arm tag verified-2026-10-25-glm53-sched. This branch: a4dfcec45 is 3f6bdcc1f unchanged; a9da0755f is 0181fdc6c without three hunks that #1286 already carries (typed CSF consumer, regex import, a comment) plus the documentation line; ceceb328e is new here and not in the fork measurements (it changes only padding rows, which GLM-5.3 does not have).
- PYTHONPATH, source overlays, or external native libraries used: in all fork arms the sm7x Triton build of #1282 first on PYTHONPATH (without kernel_config.triton_sm7x_tensor_cores) and the fork source tree on PYTHONPATH. Boot of this branch: the sm7x Triton build first on PYTHONPATH, then this branch's tree; its native libraries are described with the boot under Test Result.
- User-entry route-hit, speed, and output-quality evidence: OpenAI API server entry, below.
- Existing kernel/backend registry and capability-rejection reasons: no new kernel; the LM head goes through the MXFP8 registry and its per-shape rejection reasons from #1286.
- Per-engine configuration and deprecated-variable compatibility: no new field; speculative_config.sm70_dflash2.quant_lm_head keeps its meaning (candidate top-k from dense logits over a quantized target head). The padding value is a weight attribute (padding_value), like output_dim, which the vocab weight loader already reads, not a setting.
- New/changed environment metadata and generated reference check: none.
- Route snapshot command and intentional changed rows: none.
- SM70 registered variables / unregistered reads / model-parameter restrictions, before -> after: no change.

## Test Result

@@TESTS@@

Static checks on this branch: pre-commit run --from-ref 4d57473db --to-ref HEAD passes every hook except check-layering, which reports only vllm/model_executor/layers/fused_moe/runner/moe_runner.py (platform coupling 37 -> 39), a file this stack does not touch; check-layering --all-files reports the same, exactly as main 4d57473db does. pre-commit run mypy-3.10 --all-files --hook-stage manual: no issues apart from three in vllm/v1/attention/backends/flash_v100/runtime.py that come from a local flash_attn_v100 checkout and that main 4d57473db shows the same way.

@@BOOT@@

Fork measurements (our line, not this branch), GLM-5.3 PP5 with DFlash2 (k = 7), one request, greedy. Units: s and ms lower is better, tok/s higher is better.

A/B/A/B of both commits, 2026-10-27, arms T P T P T (T#1 compiled with fresh caches and is not counted): T = both commits, fresh caches, --gpu-memory-utilization 0.972, quant_lm_head; P = the production of the day with its own caches, 0.98. Prefill of 8,687 tokens (median of three) T 4.22 / 4.25 s against P 4.24 / 4.30 s. GSM8K-30 29/30 in every arm, acceptance length 5.4262 (T) and 5.4252 (P), 61.36 / 61.80 tok/s against 60.54 / 60.20 tok/s. Prompts of 169,693 tokens: first token 77.91 / 77.82 s against 78.36 / 78.29 s, 89.6 / 87.9 ms per round against 91.3 / 92.4 ms; 283,230 tokens: 132.26 / 132.15 s against 133.09 / 133.10 s, 89.4 / 90.1 ms against 91.9 / 91.0 ms. The fresh Triton cache of every T arm held BK 32, one warp, two stages for the KKT sub-chunk kernel, tuned at 0.2353 and 0.1536 ms per call: the magnitude of the 512-token sweep above (best 0.2340 ms on the RTX 8000, 0.1485 ms on the V100), not of the 16-token one. KV pool 676,645 tokens (T, 0.972) against 645,277 (P, 0.98). Peak memory.used per card in PCI order, which is pipeline stages 0, 1, 4, 3, 2 (stages 0 and 4 are the RTX 8000): T 47,859 / 30,522 / 47,289 / 30,502 / 30,488 and 48,139 / 30,446 / 47,517 / 30,502 / 30,520 MiB; P 47,579 / 30,522 / 47,667 / 30,502 / 30,520 and 48,139 / 30,406 / 47,877 / 30,502 / 30,520 MiB. The first request after a boot with fresh caches took 6.20 s (T#1), with warm caches 4.21 and 4.25 s (T#3, T#5); that one-time cost of fresh caches is not the KDA autotune, and this PR does not change it.

The head alone, A/B/A/B 2026-10-25, arms T T P T P, both arms at 0.98: short decode 37.77 / 37.78 tok/s (26.5 ms per token) against 36.87 / 36.88 tok/s (27.1 ms per token), +2.4 %; GSM8K-60 61.85 / 61.97 against 60.92 / 60.63 tok/s, 59/60 and acceptance length 5.5109 in all five arms; KV pool 748,342 against 645,277 tokens; seven of eight fixed answers byte-identical, the eighth reworded in the reasoning and equivalent when read by hand. Prefill of 8,687 tokens 4.51 / 4.44 s against 4.34 / 4.26 s: the slow KDA autotune choice that the nsys profile found in this arm's cache, not the head (see Purpose).

Production control after the switch (2026-10-27, both commits): the eight fixed answers byte-identical to the first T arm of the A/B, GSM8K-30 29/30, acceptance length 5.4262, 516,196 tokens reached (first token 249.34 s, 94.3 ms per round). Model loading on the last stage 41.79 GiB, against 42.34 GiB in a boot of #1286's head with the dequantized head (2026-11-02).

🤖 Generated with Claude Code (https://claude.com/claude-code)

https://claude.ai/code/session_0172gtUkBtN8FSrthHLCZe8W

=== FUNDSTELLEN (intern, vor dem Senden entfernen) ===

Alle Pfade unter ~/.cache/bench-scripts/, falls nicht anders angegeben.

Purpose, KDA:
- Autotune-Schlüssel ohne Länge: vllm/model_executor/layers/fla/ops/kda.py:583-590 (key=["BC"]); Schlüssel im Cache [16, dtypes …] bzw. [128, 64, dtypes …] (Produktionscache triton-cache-glm53-prod-v5b-2026-10-31, gelesen 02.11. 12:2x).
- TRITON_CACHE_AUTOTUNING=1: vllm/env_override.py:113-115.
- Profillauf kehrte früh zurück: 3f6bdcc1f^ vllm/models/glm5next/nvidia/kda.py (_forward, attn_metadata_raw is None → return).
- 24 Konfigurationen, T = 16 RTX 0,0021–0,0035, V100 0,0030–0,0032; T = 512 RTX 0,2340–1,4057, V100 0,1485–0,9854; Ausgaben gleich („max |diff| to first config: 0“): kda_kkt_sweep_2026-10-27/rtx.log Z. 1 und 51, v100.log Z. 1 und 51 (Dateien 27.10. 09:41–09:44). Form 64 Köpfe × 128: Commit 3f6bdcc1f, STAND.md Nachtrag 27.10. 10:10.
- Gewählte Konfiguration in den T-Armen (BK 32, 1 Warp, 2 Stufen; 0,2353 / 0,1536 ms) und gleicher Prefill: ab_glm_kdafix_2026-10-27b.log Zeilen „[T1|T3|T5 autotune]“ und ZUSAMMENFASSUNG. Die Zuordnung 0,2353 → RTX, 0,1536 → V100 steht nicht im Log, nur die Größenordnung (Vergleich mit dem Sweep); der Text nennt deshalb keine Karte.
- Kopf-Serie langsamer im Prefill (4,51 / 4,44 gegen 4,34 / 4,26 s): ab_lmhead_2026-10-25e.log, ZUSAMMENFASSUNG T2/T4 gegen P3/P5. Cache-Linie: Eintrag glm53_lmhead_entry_2026-10-25.yaml (TRITON_CACHE_DIR triton-cache-glm53-lmhead-2026-10-25) → Kopie *-glm53-prod-lmhead-2026-10-26 (make_glm_prod_lmhead_2026-10-26.sh Z. 13–16) → nsys „lmhead“ vom 27.10. (Gedächtnis: „lief im lm_head-Cache mit BK64/w1/s3“).
- nsys 8 766 Token, 268 statt 48 ms Stufe 0, Variante BK64/w1/s3–4: STAND.md Nachtrag 27.10. 10:10 (nsys/glm53-{prevprod,lmhead}-prefill-1027/, kernel_diff.py).
- 512 im Eintrag: --max-num-batched-tokens 512 (llama-swap-Eintrag GLM-5.3-Flash-NVFP4-DFlash2-vllm; glm53_kdafix_entry_2026-10-27.yaml).

Purpose, LM-Kopf:
- Formen [154880, 4096] F8_E4M3, [154880, 128] U8: Safetensors-Header model-00003-of-00040.safetensors (gelesen 02.11.); quantized_layers["lm_head"] = MXFP8, block [1, 32]: /home/mp/models/GLM-5.3-Flash-NVFP4-MTP/hf_quant_config.json; vocab 154880, hidden 4096: config.json text_config.
- 2,13 ms je Kopf-GEMM, zwei je Runde: STAND.md Nachtrag 25.10. 19:20 (nsys/glm53-prod-sched-1025/).
- Formprüfung: vllm/model_executor/layers/quantization/sm70_turbomind.py fp8_qpn8_shape_reason, _QPN8_SPLIT_K (16, 12, 8), QPN2_GROUP_SIZE 16; 4096/16 = 256, 256 % 16 = 0.
- Rauchtest (0,0078, M 9 bitgleich, 256/256, 1,11–1,17 gegen 2,08–2,17 ms, 1,18 → 0,63 GiB): Commit 0181fdc6c, STAND.md Nachtrag 26.10. 06:10, Gedächtnis project_glm_lm_head_mxfp8_qpn8_2026-10-25.md. ACHTUNG: Die Ausgabe von lm_head_mxfp8_smoke_2026-10-25.py wurde nicht als Datei abgelegt.
- 39,6 → 20,4 ms Kopf-Kernzeit, Prefill „4 % langsamer“: STAND.md Nachträge 27.10. 10:10 und 26.10. 15:40 (ab_lmhead_longctx_2026-10-26a.log: 170k 82,01/81,97 gegen 78,43/78,74 s, 283k 139,12/138,83 gegen 133,09/133,12 s).
- Fehlermeldung ohne quant_lm_head im Profillauf: validate_prod_v4b_glm_2026-10-27-fehlstart/journal.txt (profile_run → _dummy_run → propose, ValueError …ModelOptMxFp8LinearMethod.); STAND.md Nachtrag 27.10. 15:10.
- quant_lm_head nur für die qualifizierte Qwen3.8-Vereinbarung automatisch: vllm/config/sm70_dflash2.py SM70_DFLASH2_VERIFIER_DEFAULTS und resolve(); vllm/model_executor/models/config.py sm70_dflash2_verifier_qualified; Prüfung in vllm/model_executor/models/qwen3_dflash2.py compute_candidates (um Z. 526–560).

Füllzeilen (ceceb328e):
- Füllen mit 0 vorher: vllm/model_executor/layers/vocab_parallel_embedding.py weight_loader (vor dem Commit Z. 596 fill_(0)); Bereich 95..134: sm70_turbomind.py MXFP8_QPN8_EXPONENT_RANGE = (127 - 24 - 8, 127 + 15 - 8), Prüfung in prepare_mxfp8_qpn8_linear.
- Reine MXFP8-Config bedient den Kopf ebenfalls: modelopt.py ModelOptQuantConfigBase.get_quant_method (isinstance(layer, (LinearBase, ParallelLMHead)) → LinearMethodCls), ModelOptMxFp8Config.LinearMethodCls = ModelOptMxFp8LinearMethod; Compressed-Tensors: compressed_tensors.py get_quant_method für ParallelLMHead mit Schema.
- Online-MXFP8 betrifft keinen Kopf: online/base.py nur LinearBase.
- Marlin: marlin_utils_fp8.py prepare_mxfp8_layer_for_marlin (view e8m0fnu → param_dtype) und mxfp8_marlin_process_scales (→ float8_e8m0fnu); CPU-Probe 02.11. mit /home/mp/vllm/venv (torch 2.10.0): Codes 0, 127 → fp16 0.0, 1.0 → zurück 0, 127. Emulation: dequant_mxfp8_to_bf16 (exp2(code − 127)); FlashInfer: Skala unverändert umgeordnet.
- 154 880 = 64 × 2 420: config.json vocab_size, DEFAULT_VOCAB_PADDING_SIZE = 64.

Duplikatsuche: Skript und Ausgabe scratchpad dup_search.sh / dup_search.log (02.11. 12:23 CET); Dateien von #1159/#1165/#1260 per gh pr view/gh pr diff geprüft.

Test Result, A/B 27.10.: ab_glm_kdafix_2026-10-27b.log (Arme, Prefill-Mediane Zeilen ZUSAMMENFASSUNG, GSM8K-Blöcke, longctx-Zeilen, [Tn autotune], Speicherspitzen, Pool); Eintrag T glm53_kdafix_entry_2026-10-27.yaml (0.972, quant_lm_head). Zuordnung PCI → Stufe: CUDA_VISIBLE_DEVICES=0,1,4,3,2 im GLM-Eintrag (/home/mp/.config/llama-swap/config.yaml) und im PP5-Dokument.
Prefill-Werte sind total_s des Clients (Prefill + 1 Token), wie in den Logzeilen „Prefill n: … in x s“; ttft_s liegt je ≈ 0,02–0,04 s darunter (pr_boot_2026-11-02_glm/client.json).
Test Result, Kopf allein 25.10.: ab_lmhead_2026-10-25e.log (Decode kurz, gsm8k decode_tok_s, Prefill, Pool, Textvergleich); 59/60, Annahme 5,5109, kuanda von Hand: STAND.md Nachtrag 26.10. 06:10.
Produktionskontrolle: validate_prod_v4b_2026-10-27.log (Textvergleich 8/8, GSM8K, ctxmax 249,34 s / 94,3 ms); Ladespeicher PP4 41,79 GiB: validate_prod_v4b_glm_2026-10-27/journal; 42,34 GiB: pr_boot_2026-11-02_glm/journal.txt.
