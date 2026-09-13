# PR-Entwurf 1Cat: SM70-Cudagraph-Gates pre-Ampere auf dem eigenen Gerät (13.09., VERÖFFENTLICHT als PR #618, Commit 594e04a9; Overlay in work-main als eigener Commit)

Worktree `1Cat-vLLM-pr-tailgraphs`, Branch `sm70-cudagraphs-pre-ampere` auf origin/main dfef3342.
Befund aus der Merge-Abnahme 12./13.09.: Tail-Cudagraphs (+1,2 % V100) fehlen auf Turing.

Titel: [Bugfix][SM70/SM75] Gate the SM70 graph tunings on the worker's own pre-Ampere device

---

Three gates in vllm/v1/worker/gpu/cudagraph_utils.py ask current_platform.is_device_capability((7, 0)): the DFlash2 verifier tail graphs (VLLM_SM70_DFLASH2_TAIL_CUDAGRAPHS, on by default since #609), the split MTP draft graphs (VLLM_SM70_MTP_SPLIT_DRAFT_CUDAGRAPHS) and the one-shot log of the 0.0.3 compile graph. That check is wrong on two counts.

It is exact Volta. The SM70 baseline is a pre-Ampere tuning, not a Volta tuning (#572 made VllmConfig decide that way with _any_participating_device_is_pre_ampere): Turing runs the same kernels, the same fp16 contract and the same compile graph, and the 0.0.3 compile graph is already enabled for it. A Turing worker therefore boots with the compile graph but without the graph tunings that go with it. Measured on 2x RTX 8000 (TP2), Qwen3.8-27B-NVFP4, fp16 KV, greedy, 5x400 tokens: with MTP k=3 and VLLM_SM70_MTP_SPLIT_DRAFT_CUDAGRAPHS=1 the split managers now engage on Turing ("Using split SM70 MTP CUDA graph manager shapes" for query lengths 1 and 4) and decode goes from 72.03 to 74.59 tok/s, same output text and acceptance length 2.920 in both runs. With DFlash2 n=7 the verifier tail graphs now capture on Turing as well ("Capturing SM70 DFlash2 target tail query lengths (8, 7, 6, 5, 4, 3, 2, 1)"); measured on our fork tree, because DFlash2 on Turing also needs #592 and #599, they are neutral there: 76.50 and 76.41 tok/s with the tail graphs against 76.55 without, same output text and acceptance length 3.325. On 2x V100 the same tail graphs are worth 75.09 against 74.21 tok/s, and the patched tree still gives 76.47 tok/s there with the production drafter. So the measurable win on Turing is the split MTP graphs; the tail graphs are about the gate being consistent, not about speed.

It is device 0. The cudagraph manager lives in the worker, but the query is answered for index 0 of the visibility list. On a mixed rig that is another worker's card: a Volta worker sitting behind a Turing or Ampere card at index 0 loses the tail graphs, and the reverse enables them where the check was meant to refuse. #576, #599 and #600 (open) address the same pattern elsewhere.

The change adds _worker_device_is_pre_ampere(), which asks current_platform.get_device_capability for torch.accelerator.current_device_index() and answers True for (7, 0) and (7, 5), and routes the three gates through it. The long-context E4M3 attention gate in the same file stays exact Volta on purpose: its scalar tail kernels exist for sm_70 only. The env switches keep their defaults.

Tests: tests/v1/cudagraph/test_sm70_graph_gates_pre_ampere.py (new: Volta and Turing pass, Ampere does not, the worker's own index is the one asked, off without CUDA), tests/v1/cudagraph/test_sm70_mtp_split_cudagraphs.py and tests/v1/worker/test_sm70_long_attention_graphs.py extended with Turing cases; 36 passed. pre-commit and mypy-3.10 clean.

Duplicate check: no open PR or issue touches these gates; #600 (open) changes the platform default device index and would make the device-0 half of this redundant, the pre-Ampere half stays.

This work was done with AI assistance (Claude); the change is reviewed, tested and measured by me on the hardware named above.
