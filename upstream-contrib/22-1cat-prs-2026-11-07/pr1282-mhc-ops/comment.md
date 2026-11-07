I added a commit to this PR, b9fbe637c (perf(mhc): let the mHC layer ops take the sm7x tensor-core route).

GLM-5 next reaches mHC through MHCPreOp and MHCFusedPostPreOp, which never passed tensor_cores to mhc_pre_tilelang and mhc_fused_post_pre_tilelang, so with triton_sm7x_tensor_cores set its prefill projection still took the default kernels. DeepSeek-V4 calls the kernel functions directly and takes the tensor-core projection with the commit "perf(mhc): prefill mHC projection on tensor cores for the sm7x route" of this PR. The ops now read the route once for their card from KernelConfig.use_triton_sm7x_tensor_cores, like the other layers of the route. GLM-5.3-Flash has the mHC shape of DeepSeek-V4 (hidden size 4096, hc_mult 4, 24 mixes), so the projection timings in that commit's message apply. Decode batches up to 16 tokens keep their kernels, as before.

The new test builds both ops with the setting off and on and checks their outputs bit for bit against the kernel functions with the same route.

Model A/B on our fork (github.com/Peuqui/1Cat-vLLM, 0c5cbfe9b = this commit on top of our build): GLM-5.3-Flash NVFP4 with DFlash2, PP=5 on 3 Tesla V100 + 2 Quadro RTX 8000, --max-model-len 524288. Changed: the same build and entry with --kernel-config '{"triton_sm7x_tensor_cores":true}'; reference: the build before this commit without the setting. The setting also moves the prefill index scorer of the V100 stages to the Triton scorer of this PR, so this measures the route of this PR on GLM-5.3-Flash as a whole, with the mHC projection included. Runs T T P T P, counted T#2 + T#4 against P#3 + P#5; lower is better unless noted:
Prefill of an 8,687-token prompt (median of three): 3.87 / 3.85 s (2,242 / 2,254 tok/s) changed, 4.18 / 4.21 s (2,080 / 2,061 tok/s) reference, -7.8 %.
First token at 169,693 tokens: 72.40 / 72.31 s changed, 77.54 / 77.56 s reference.
First token at 283,230 tokens: 122.91 / 122.34 s changed, 131.22 / 131.27 s reference.
First token at 516,196 tokens: 228.67 s (T#4) changed, 244.66 / 244.77 s reference; T#2 is left out because the system disk of the host reset during that request. Decode round there 94.7 ms changed, 95.6 / 95.6 ms reference.
GSM8K-30: 29/30 in all counted runs, acceptance length 5.533 changed, 5.423 reference.
Memory peak per card (MiB): 48,311 / 30,522 / 47,015 / 30,502 / 30,520 changed, 48,311 / 30,646 / 47,015 / 30,502 / 30,848 reference. KV pool 582,542 tokens in all runs.
The eight fixed questions were read by hand and pass in both builds; seven of the eight texts are worded differently (both kernels of the route round differently from the default kernels).

Tests on this branch (b9fbe637c):
[FILL GPU]
pre-commit run --from-ref [FILL MAIN] --to-ref HEAD: [FILL PRECOMMIT]
