# PR-Stau bei 1Cat: Aufräumen und Überblick (Entwurf 22.09.2026)

Stand geprüft 22.09. früh gegen origin/main 8d5d8233: 28 offene PRs von uns,
kein Review seit 12.09. Lokal per `git merge-tree`: 26 konfliktfrei, keiner
still in main enthalten; #601 überholt durch 8a10215d; #618 kollidierte mit
4bbaf64f und ist rebased (dcd6be03, 39 Tests grün, pre-commit + mypy-3.10).

GESENDET 22.09.2026 (Peuquis Go): #601 mit Kommentar geschlossen; #618 per force-with-lease auf
dcd6be03 gepusht + Kommentar (issuecomment-5771274617), jetzt MERGEABLE; Überblick als
https://github.com/1CatAI/1Cat-vLLM/issues/674.

---

## 1. Kommentar zu #601, danach schließen

Closing this one: 8a10215d ("[Kernel][SM70] Share FP16 fast paths with E4M3 KV") declares the same five pybind11 extensions with `py_limited_api=False` and moves the default into `kwa.setdefault`, which is exactly what this PR did. Thanks for picking it up.

AI assistance was used to compare the two changes.

---

## 2. Kommentar zu #618 nach dem Force-Push

Rebased onto main (8d5d8233). The only conflict was in `vllm/v1/worker/gpu/cudagraph_utils.py`, where `get_sm70_cudagraph_memory_reserve` from 4bbaf64f landed next to `_worker_device_is_pre_ampere`; both are kept unchanged. The PR's three test files together with `tests/v1/worker/test_gpu_cudagraph_memory_reserve.py`: 39 passed; pre-commit and mypy-3.10 clean.

---

## 3. Überblick als Issue

Titel: Overview of my open PRs: what each fixes, dependencies, and an offer to bundle

Hi, I have accumulated a fair number of open PRs here and wanted to make them easier to go through. All of them come from running 1Cat-vLLM on a mixed rig (3x V100 + 2x Quadro RTX 8000, pipeline parallel over all five), mostly with DeepSeek-V4-Flash + DSpark and Qwen3.8. As of today every one of them merges cleanly into main except #618, which I just rebased, and #601, which I closed because 8a10215d already contains it.

Fixes for crashes or wrong results

- #670 Keep dummy-run positions inside max_model_len (models with a learned position table fail in profile_run)
- #658 Keep mHC finite under float16 for attention-sink rows
- #662 Ship the speculative round state to non-last PP ranks
- #574 Trim the optimistic spec-decode tokens on every pipeline rank
- #636 Keep the MTP drafter stage-local under pipeline parallelism (Qwen3.5)
- #603 Align the SWA decode threshold with the sparse MLA builder
- #613 Honor a checkpoint's KV-cache quantization directive only on Ampere and newer
- #665 Order SimpleCPUOffloadConnector stores behind the compute stream
- #592 DFlash: fuse context K/V through quant_method so a quantized draft head loads

Mixed Volta/Turing nodes (per-worker device instead of device 0)

- #600 Resolve an unspecified device_id to the worker's own device
- #576 Read the quantization SM70 gate from the worker's own device
- #618 Gate the SM70 graph tunings on the worker's own pre-Ampere device
- #599 Gate DFlash2's BF16 emulation and FlashInfer top-k on the worker's own device
- #604 Run ModelOpt NVFP4 and FP8 linears on Turing through the SM70 QPN kernels
- #623 Build and load a Turing FlashAttention-2 library next to the Volta one

Prefix cache (#667 builds on #657)

- #657 Backport vllm-project/vllm#44082: cache the EAGLE/MTP lookahead block in the SWA prefix-cache mask
- #667 Reuse a sliding window's dead blocks before other requests' cached ones

Performance

- #669 Skip checkpoint tensors a model does not load before reading them (DSpark boot 10:25 to 6:03 min)
- #611 Block-pack the activations of the NVFP4 QPN2 kernels
- #659 Apply min_p in the rejection sampler

Qwen4Exp / PLE (#646 builds on #622, #640 and #639)

- #622 Resolve the PLE table pointer inside the gather op instead of baking it into the graph
- #640 Detect the FP8 PLE table from ModelOpt mixed-precision configs
- #639 Allow checkpoint FP8 MTP experts under pipeline parallelism
- #646 PLE overflow cascade: device, pinned host, spare GPU, disk
- #654 PLE offload: send registration inputs by file descriptor

Infrastructure

- #619 Pass --distributed-timeout-seconds to the NCCL subgroups
- #621 Drop the forced compile-cache opt-out for the Flash-V100 graph

If fewer, larger PRs would be easier for you, I am happy to fold these into a handful of themed ones (for example one for the mixed-rig device gates and one for the PP/spec-decode fixes), or to close whatever does not fit your plans. Just let me know what works best.

AI assistance was used to prepare this overview. Each PR was checked against current main with git merge-tree.
