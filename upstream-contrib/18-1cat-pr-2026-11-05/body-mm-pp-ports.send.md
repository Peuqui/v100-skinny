## Purpose

Model Runner V2 serves multimodal models under pipeline parallelism with three gaps that upstream vLLM has closed. This PR ports the three upstream changes, one commit each. We hit all three while bringing up the image part of GLM-5.3-Flash-FP4-Local at PP5 under V2 on our fork.

1. vllm-project/vllm#60238 (168a86ab8f), "Skip encoder towers on later PP stages under V2". V2 runs the multimodal encoder on the first pipeline stage only, but every stage built its towers and loaded their weights. SupportsMultiModal._mark_tower_model now builds them as StageMissingLayer on the later stages under V2, as it already does when --limit-mm-per-prompt disables all of a tower's modalities. Qwen4Exp (nvidia and amd) checks its deepstack layer split against the vision config instead of use_deepstack, which is now false on those stages; this is the same change as upstream. Upstream adjusts tests/models/transformers/test_backend.py, which this repository does not have; here the fake config in tests/model_executor/test_uva_tower_offload.py gains use_v2_model_runner instead.

2. The frame search part of vllm-project/vllm#48729 (bf578e1abd), "Fix video dummy profiling and memory usage". Glm4vProcessingInfo._get_max_video_frames counted frames until the next one exceeded the token budget, against a fixed 47,040,000-pixel area. With a long context the budget is never reached: at --max-model-len 524288, GLM-5.3-Flash counted up to about 60,000 frames, where smart_resize raises ("max_pixels=47040000 is too small"). The search now tries 1 to _MAX_FRAMES_PER_VIDEO frames against the video processor's own pixel budget and keeps the count with the most tokens within the budget. get_num_video_tokens of this class was only used there and goes. Upstream's test for the search is ported; the dummy video memory part of that PR is not.

3. vllm-project/vllm#47985 (ea0e9c8f2e), "Add encoder cache profiling implementation". V2 sized the KV cache without running the encoder, so with images enabled the first stage had no memory left for encoder activations and cached image embeddings. profile_run now runs the encoder on the first stage with the dummy items of the largest feature size, as the V1 runner does, and clears the encoder outputs afterwards. This builds the dummy multimodal inputs at startup, which is where commit 2 was needed: without it, GLM-5.3-Flash at 524288 tokens failed in the profiling even with --limit-mm-per-prompt '{"image":12,"video":0}'.

Effect on other models: under V2 with images enabled, the KV cache of the first stage is now smaller by what the encoder needs at its peak, as under V1. Under V1, and with all modalities disabled, nothing changes.

Not a duplicate: I searched the open and closed PRs for 60238, 48729, 47985, StageMissingLayer, profile_encoder_cache, "encoder cache profiling", "MRV2 encoder", "multimodal profile_run", "encoder tower pipeline", "dummy video" and _get_max_video_frames. None of them ports these changes. Ten open PRs change vllm/v1/worker/gpu/model_runner.py (#1364, #1347, #1282, #1280, #1279, #1261, #1221, #1217, #1167, #903). I compared their hunks with these: none edits profile_run; #1167 and #903 edit the import block next to the new import, with unchanged lines in between. #1167 and #1020 change vllm/models/qwen4_exp/nvidia/model.py in other hunks than the deepstack check.

AI assistance was used for this change. I reviewed every line and ran the tests and measurements below.

## Test Plan

One pytest process per file:
python -m pytest tests/model_executor/test_uva_tower_offload.py tests/models/multimodal/processing/test_glm4_1v.py tests/multimodal/test_encoder_microbatch.py tests/models/qwen4_exp/test_compact_initialization.py tests/models/qwen4_exp/test_qsa_rope_cache_length.py tests/models/multimodal/test_nano_nemotron_vl.py tests/v1/core/test_mamba_align_chunk_split.py (CPU)
python -m pytest tests/v1/cudagraph/test_encoder_cudagraph.py tests/v1/worker/test_gpu_worker_profile_failure.py tests/v1/worker/test_gpu_worker_memory_profile.py (one V100)
pre-commit run --from-ref origin/main --to-ref HEAD; pre-commit run mypy-3.10 --hook-stage manual --from-ref origin/main --to-ref HEAD
The ported test test_get_max_video_frames_matches_glm_resize fails on main (5 of 5 cases) and passes with commit 2.

On our fork the three commits run, together with the GLM-5.3 image part (offered separately as a follow-up to #1286), in our GLM-5.3-Flash-FP4-Local production entry since 2026-11-05: PP5 over 3x V100 + 2x RTX 8000, V2 runner, --max-model-len 524288, --limit-mm-per-prompt '{"image":12,"video":0}'.

## Path coverage

| Path / provider | Format | Key conditions (shape, TP/EP, graph, capability) | Expected selection / fallback | Evidence (source / static / executed / measured) | Unverified scope |
| --- | --- | --- | --- | --- | --- |
| SupportsMultiModal._mark_tower_model | any | V2 runner, pipeline stage after the first | tower built as StageMissingLayer | executed: test_uva_tower_offload.py; fork: GLM-5.3 PP5 boots, stages 1 to 4 build the tower as StageMissingLayer (the loader change of #1362 logs it) | Qwen4Exp vision under PP with V2: not booted |
| same | any | V1 runner, or first stage, or PP 1 | unchanged | source | none |
| Glm4vProcessingInfo._get_max_video_frames | GLM-4.1V, GLM-4.5V, GLM-OCR and the GLM-5.3 processor that subclasses it | any max_model_len | 1 to _MAX_FRAMES_PER_VIDEO frames tried, most tokens within the budget | executed: test_glm4_1v.py; fork: GLM-5.3 at 524288 boots | none |
| GPUModelRunner.profile_run (V2) | any multimodal model | first stage, images or video enabled, skip_mm_profiling false | encoder runs on the dummy items, encoder outputs cleared afterwards | fork: GLM-5.3 PP5 logs "Encoder cache will be initialized with a budget of 7921 tokens, and profiled with 1 image items of the maximum feature size", image requests below | other V2 multimodal models: not booted |

## Acceleration and benchmark contract (required for performance changes)

Not a performance change. Memory behaviour: see "Effect on other models" above.

## Test Result

Tests on main b2b6b3bb9 with this branch:
- tests/model_executor/test_uva_tower_offload.py: 4 passed
- tests/models/multimodal/processing/test_glm4_1v.py: 13 passed
- tests/multimodal/test_encoder_microbatch.py: 8 passed
- tests/models/qwen4_exp/test_compact_initialization.py: 16 passed
- tests/models/qwen4_exp/test_qsa_rope_cache_length.py: 3 passed
- tests/models/multimodal/test_nano_nemotron_vl.py: 3 passed
- tests/v1/core/test_mamba_align_chunk_split.py: 37 passed
- tests/v1/cudagraph/test_encoder_cudagraph.py: 48 passed (one V100)
- tests/v1/worker/test_gpu_worker_profile_failure.py: 4 passed (one V100)
- tests/v1/worker/test_gpu_worker_memory_profile.py: 7 failed, 2 passed (one V100); the same 7 tests fail on main b2b6b3bb9 without this branch (a fake config without speculative_config and a charged-memory bound), #1167 edits this file.
pre-commit run --from-ref origin/main --to-ref HEAD: every hook passes except check-layering, which reports vllm/model_executor/layers/fused_moe/runner/moe_runner.py (platform coupling 37 -> 39); this PR does not touch that file, which is identical to main b2b6b3bb9. pre-commit run mypy-3.10 --hook-stage manual: passed.

Fork runs of GLM-5.3-Flash-FP4-Local at PP5 with the image part (logs on our side):
- Without commit 2 the boot failed in the encoder profiling of stage 0 with "max_pixels=47040000 is too small. At least 47041568 pixels are required for one aligned patch", with video disabled.
- With all three: stages 1 to 4 build no vision tower (the loader change of #1362 logs it among the modules the stage does not hold), stage 0 profiles the encoder, the KV cache of stage 0 goes from 2.35 to 1.85 GiB (vision tower weights plus the profiled encoder peak, which the image part follow-up bounds by running the tower in 4096-row chunks), the encoder profiling takes 3.6 s of engine init (init engine +3.5 s). A/B against the production entry before (the same tree without these three commits and the image part, image limit 0), two counted boots each: prefill of 8687 tokens 4.27 / 4.27 s against 4.29 / 4.17 s, short decode 41.81 / 41.81 tok/s against 41.82 / 41.81 tok/s, GSM8K-30 29 / 29 against 29 / 29, eight quality answers identical token for token. Control run of the installed entry: seven image requests (one or two images, 3026 to 7295 prompt tokens, reasoning effort max, at most 4000 tokens), first token after 2.9 to 6.4 s; six answers describe or transcribe their image correctly (read by hand), the printed medication plan spent all 4000 tokens on reasoning.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_0172gtUn2th65t88H9KvYkwK
