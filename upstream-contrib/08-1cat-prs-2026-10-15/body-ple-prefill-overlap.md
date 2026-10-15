## Purpose

With CPU-offloaded PLE in a pipeline, stage 0 waits before every prefill chunk until the PLE worker has gathered that chunk's rows. Two things serialize the gather behind the previous forward:

1. `PleOffloadConnector._launch` queues the D2H copy of the token ids on the model stream, so the request for chunk N+1 reaches the worker only after forward N has finished (`ple_offload.wait_d2h`).
2. With the mapped result transport the worker gathers straight into the host buffer the GPU reads, and it may only overwrite it after `release_outputs()`, i.e. after the whole forward, although the consumer copies the result to the device right at `ple_offload_wait`.

This PR adds `kernel_config.ple_prefill_early_request` (default off; outside the compile hash):

- Prompt-only batches (no draft tokens, no padding, every request still inside its prefill) send their PLE request at scheduling time with host inputs built from a host copy of each request's prefill tokens (`Qwen4ExpModelState.ple_cpu_inputs`, same rules as `prepare_prefill_inputs` and `_prepare_ngram_context`). `all_token_ids` is not read on the host because its prompt rows are written by a GPU kernel. All other batches keep the existing D2H path.
- The worker publishes each finished request's sequence number through a shared counter; the connector's request thread stages the next host inputs only after the previous request is done, so the shared input buffers and the worker's one-request-per-DP-rank batching stay safe.
- In forwards without CUDA graph replay, `ple_offload_wait` releases a mapped result in stream order right after its consumer H2D copy; `release_outputs()` skips exactly the results released that way, so graph replays and the CUDA transport keep the existing release.

Not a duplicate: gh pr list --state all --search "ple_offload", "PLE prefill", "wait_d2h", "early request", "release_outputs", "PLE overlap", "prefill chunk PLE" finds no PR that changes when the PLE request is sent or when mapped results are released; the open #1205, #1209 and #1199 do not touch the PLE connector, worker request path or result release. The fast lane (#1203) serves CPU-owned GGUF rows from a native thread and is not used for FP8 disk-tier PLE.

This builds on #1214, which repairs the test fakes this PR also extends.

AI assistance was used for this change. I reviewed every line and ran the tests and measurements below.

## Test Plan

```bash
pytest tests/v1/worker/test_ple_prefill_early_request.py tests/v1/worker/test_ple_offload_worker.py tests/v1/worker/test_early_model_inputs.py tests/models/qwen4_exp/test_gguf_pinned_policy.py tests/compile/test_sm70_decode_graph.py tests/kernels/ple/test_host_result.py tests/models/qwen4_exp/test_ple.py
pre-commit run --from-ref origin/main --to-ref HEAD
```

Serving on our fork (main 6f3557c97 plus our open PRs and fork commits): Qwen3.8-Flash-Next-180B NVFP4 + MTP, PP4 over Quadro RTX 8000 / Quadro RTX 8000 / Tesla V100 / Tesla V100, FP8 PLE with a disk tier, 29k-token cold prompts, same code with the field off and on, alternating boots. nsys profile of one 9.7k-token prefill with the field on and off.

## Path coverage

| Path / provider | Format | Key conditions (shape, TP/EP, graph, capability) | Expected selection / fallback | Evidence (source / static / executed / measured) | Unverified scope |
| --- | --- | --- | --- | --- | --- |
| PLE request, prompt-only batch | int32 inputs | field on, MRV2, no draft tokens, no padding | host inputs, sent at scheduling | executed + measured: NVTX `wait_previous_request` / `write_cpu_inputs` replace `wait_d2h` | MRV1 (keeps its own CPU path) |
| PLE request, any other batch | int32 inputs | decode, spec verify, padded, field off | existing D2H path | executed: decode/verify in the same runs, greedy unchanged | none |
| Mapped result release | FP16 / FP8 PLE rows | field on, CUDA graph mode NONE | released after the consumer H2D copy | executed + test on V100 and RTX 8000 | CUDA transport (unchanged by design) |
| Mapped result release | same | CUDA graph replay | release after the forward, unchanged | test (`armed=False`) | none |

## Acceleration and benchmark contract (required for performance changes)

- Default enabled or opt-in: opt-in, `kernel_config.ple_prefill_early_request=true`.
- Required CLI options and environment switches: `--kernel-config '{"ple_prefill_early_request":true}'` on an engine with CPU-offloaded PLE.
- KV cache dtype used for the benchmark: FP16 (model default for this entry).
- Wheel SHA or source commit: our fork, branch ple-prefill-overlap-v2 at baaa5a687 (main 6f3557c97 + our commits).
- PYTHONPATH, source overlays, or external native libraries used: PYTHONPATH on the source tree, native build of the same tree.
- User-entry route-hit, speed, and output-quality evidence: see Test Result.
- Existing kernel/backend registry and capability-rejection reasons: none added.
- Per-engine configuration and deprecated-variable compatibility: KernelConfig field only, no environment variable.
- New/changed environment metadata and generated reference check: none; architecture reference regenerated.
- Route snapshot command and intentional changed rows: none.
- SM70 registered variables / unregistered reads / model-parameter restrictions, before -> after: unchanged.

## Test Result

- New tests (tests/v1/worker/test_ple_prefill_early_request.py, 13 cases: host inputs equal the device rules, admission, request ordering, worker counter after the last layer, release skipping, wait-op release armed/unarmed on GPU): pass on a Tesla V100 and a Quadro RTX 8000. On this branch (main + #1214 + this change), the seven files above: 251 passed with a V100 as device 0, 248 passed and 3 skipped with an RTX 8000 as device 0. pre-commit passes.
- Greedy output with the field on is identical to the field off (3 of 3 prompts).
- nsys, 9.7k-token prefill: without the field, stage 0 idles 107.8 ms before the PLE consumer kernel of every chunk and `wait_d2h` takes 276 ms per request; with the field, only the first chunk waits (273 ms, nothing to overlap), and the pipeline cadence drops from 639-685 ms to 614-643 ms per chunk. Stage 1 then computes without gaps, so the end of the prefill does not move until stage 1 gets less work.
- 29k-token cold prompts, seconds, prompts 2/3/4 of each boot (prompt 1 is the first long request after boot and is not counted): partition 11,11,14,12 off 14.1/13.1/13.2 (three boots), on 13.5/13.1/13.2 (two boots); partition 11,10,15,12 off 14.0/12.5/12.6, on 12.9/12.5/12.5-12.6 (two boots). The field shortens the first cold prompt of a boot by 0.6-1.1 s (its PLE rows come from disk); prompts 3 and 4 do not change with the field. The steady state is set by the partition: 11,10,15,12 is about 4 % shorter than 11,11,14,12 with or without the field. Decode step at 29k is unchanged (61-64 ms).

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01Ly7LM3m9fa6DRJPYQjnt95
