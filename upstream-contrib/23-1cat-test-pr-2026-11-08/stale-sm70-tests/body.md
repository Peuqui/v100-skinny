## Purpose

Seven test files fail on main c98a8d664 on a Tesla V100 with the stock Triton 3.6.0 wheel, 71 cases in all. In each case the code under test does what main intends; the test still describes main before a later change. This PR changes only these tests, no runtime code.

1. tests/kernels/core/test_dflash2_sampler_graph.py (4 failed) and tests/v1/sample/test_dflash2_logits_reuse.py: #1398 calls _supports_sparse_sampling_contract(..., allow_penalties=True), and the fakes were lambda *a: True and lambda *args: True ("got an unexpected keyword argument 'allow_penalties'"). #1398 already changed the same fake in test_topk_topp_tied_cutoffs.py to lambda *a, **k; the two others get the same form. #1398 also reads rejection_sampler.sampler.penalties_state, so the sampler fake in test_dflash2_logits_reuse.py gets the penalties state that #1398 added to the fakes it touched (use_penalty all False).

2. tests/v1/sample/test_dflash2_request_fallback.py (58 failed) and tests/v1/sample/test_dflash2_logits_reuse.py (2 failed): both patch sparse_rejection.envs, which the module no longer imports ("has no attribute 'envs'"). 85a45b396 moved the sparse-rejection switch to the per-engine DFlash2 policy (sm70_dflash2_enabled), 1ca9ea9f8 the alignment dump to the speculator trace (speculator._trace.value("alignment")) and dropped the envs import. The tests now patch sparse_rejection.sm70_dflash2_enabled, as test_dflash2_sampler_graph.py does, and give their speculator fake _trace = SimpleNamespace(value=lambda name: False), the fake that test_dflash2_sampler_graph.py uses.

3. tests/kernels/core/test_sm70_fp8_projection_output.py (1 failed): the test builds the scheme with object.__new__ and skips process_weights_after_loading, where c9d825d5c binds native_ops ("'CompressedTensorsW8A16Fp8' object has no attribute 'native_ops'"). The test now binds them the same way, NativeBindings(capture_linear_native_config("fp8").values).

4. tests/kernels/quantization/test_sm70_gdn_ba_verify.py (1 failed): Qwen3_5GatedDeltaNet.forward_cuda reads self.gdn_policy.projection (495fb442a) and calls self._allocate_core_attn_out (deadbde95), and the SimpleNamespace layer of test_verify_fusion_reaches_qwen35_override has neither. It now gets a projection policy with mixed_qkv_contiguous=False, so that the fused projection's QKV reaches the recurrent stub unchanged (the stub checks its identity), and an output buffer of the right shape. The test still checks what it was written for: the verifier b/a fusion reaches the Qwen3.5 override, the recurrent core and the output projection.

5. tests/kernels/moe/test_sm70_mtp_moe_fp16.py (4 failed): #1369 admits the exact SM70 FP16 draft MoE for 1 to _SM70_MTP_MOE_MAX_M = 80 rows (up to 16 MTP4 verification requests), and test_shape_admission still expected only m in (1, 5). It now expects m <= _SM70_MTP_MOE_MAX_M and also runs m = 80 and m = 81, so the limit stays covered.

6. tests/v1/sample/test_topk_topp_tied_cutoffs.py (1 failed): test_compact_classifies_rows_without_rejecting_the_batch asserted that exactly two rows take the dense reference route (sum(dense_rows) == 2, got 4 = 2 + 2). The filtered result matches the reference (torch.equal passes). With torch 2.10.0 the third dense row is row 7 (k=127, top_p=0.95): its cumulative probability is 4.5e-05 away from the top-p boundary, inside the 1.22e-04 FP32 margin, so _apply_top_k_top_p_compact sends it to the dense route as designed, in the two-row chunks it uses for that route. The test now checks the rows by their (k, top_p) pair: rows 5 and 6 must take the dense route, rows 0 to 3 (greedy-style and top-k-only) never, and the sampled rows 4 and 7 may.

Not a duplicate. On 2026-11-08 I searched the PRs of this repository, open and closed, for the seven test file names, for sparse_rejection.envs, native_ops, "stale tests" and "fix tests"; none fixes these tests. I also compared the changed files of all 109 open PRs with the seven files. Two open PRs touch some of them: #953 appends a new test to test_dflash2_logits_reuse.py, and #828 (last updated 2026-10-06) brings five of the files (test_dflash2_sampler_graph.py, test_sm70_fp8_projection_output.py, test_sm70_gdn_ba_verify.py, test_dflash2_request_fallback.py, test_topk_topp_tied_cutoffs.py) in an earlier form that does not contain these fixes. #1214 was an earlier repair of the same kind in other tests.

Scope: I ran 30 SM70 GPU test files on our fork (github.com/Peuqui/1Cat-vLLM) after merging main c98a8d664 into it. The six that failed there fail the same way on main itself, and so does test_dflash2_logits_reuse.py, which I found while fixing them. I have not run the whole test suite on main.

AI assistance (Claude) was used for this change. I reviewed every changed line and ran the tests below.

## Test Plan

On a Tesla V100-PCIE-32GB (SM70) as the only visible device (CUDA_DEVICE_ORDER=PCI_BUS_ID, CUDA_VISIBLE_DEVICES=<the V100>), torch 2.10.0+cu128, stock Triton 3.6.0, one pytest process per file, on main c98a8d664 and on this branch:
python -m pytest tests/kernels/core/test_dflash2_sampler_graph.py tests/kernels/core/test_sm70_fp8_projection_output.py tests/kernels/moe/test_sm70_mtp_moe_fp16.py tests/kernels/quantization/test_sm70_gdn_ba_verify.py tests/v1/sample/test_dflash2_request_fallback.py tests/v1/sample/test_topk_topp_tied_cutoffs.py
python -m pytest tests/v1/sample/test_dflash2_logits_reuse.py, on this branch on the V100, on main and on this branch also with CUDA_VISIBLE_DEVICES="" (the file needs no GPU)
pre-commit run --from-ref origin/main --to-ref HEAD
pre-commit run mypy-3.10 --hook-stage manual --from-ref origin/main --to-ref HEAD

## Path coverage

N/A, test-only change: no runtime path is touched.

## Acceleration and benchmark contract (required for performance changes)

N/A, not a performance change.

## Test Result

On the V100, main c98a8d664 (failed / passed): test_dflash2_sampler_graph.py 4 / 0, test_sm70_fp8_projection_output.py 1 / 0, test_sm70_mtp_moe_fp16.py 4 / 12, test_sm70_gdn_ba_verify.py 1 / 5, test_dflash2_request_fallback.py 58 / 6, test_topk_topp_tied_cutoffs.py 1 / 70. This branch: 4, 1, 20 (four new cases for m = 80 and 81), 6, 64 and 71 passed, nothing failed; test_dflash2_logits_reuse.py 15 passed.
test_dflash2_logits_reuse.py without a visible GPU: 2 failed, 13 passed on main; 15 passed on this branch.
pre-commit: every hook passed or had no files to check. mypy-3.10 (manual stage): passed.

One more observation, not part of this PR: with a Triton build that uses tensor cores on SM70, the six graph cases of test_sm70_mtp_moe_fp16.py fail, because the Triton reference kernel then rounds differently from the exact native kernel. With the stock Triton wheel they pass.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_0172gtUn2th65t88H9KvYkwK
