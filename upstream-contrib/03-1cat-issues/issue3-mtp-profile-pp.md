# [Bug]: VLLM_SM70_MTP_PROFILE report never prints under pipeline parallelism

`_sm70_mtp_profile_report` (gpu_model_runner.py) returns early on
`not is_global_first_rank()`. Under PP, the MTP timing events
(`target_forward`, `draft_total`, ...) are recorded on the **last** PP
stage — which is never the global first rank. Result: with
`VLLM_SM70_MTP_PROFILE=1` and PP>1, the profile is collected every
interval and silently discarded; the operator sees nothing.

Suggested fix: report on the last PP rank (where the spec-decode events
live), or on any rank that actually accumulated events. One-line change;
we ran it as a local patch to obtain draft-vs-target GPU timings
(draft_total=83.9ms vs target_forward=24.3ms — the measurement that
located a checkpoint problem for us).

## Hardware environment (unusual, but that is the point)

- Host: GEM10 mini-PC (AMD Ryzen APU), 32 GB RAM (~30 GiB usable after iGPU carve-out), boot NVMe on USB 3.2
- GPUs: 5 external cards, 192 GB VRAM total — 2x Quadro RTX 8000 48 GB
  (sm75, GDDR6 672 GB/s) + 3x Tesla V100-PCIE-32GB (sm70, HBM2 900 GB/s)
- Attachment: 4 cards via M.2-to-OCuLink adapters, 1 (one of the V100s)
  via a USB4 tunnel (AG02 dock) — and that tunneled V100 is an active
  member of the last PP stage, i.e. the MTP drafter itself runs on it. All five run PCIe Gen3 x4 under load (~3.9 GB/s per
  card); the USB4-tunneled card measures only ~5% slower than OCuLink.
- Peer-to-peer DMA between the Ryzen root ports does NOT work: NCCL P2P
  hangs in init (both ranks 100% util at ~40 W, spin-wait). We run
  `NCCL_P2P_DISABLE=1` everywhere; with TP allreduces in the ~10 KB
  range the host-RAM detour is not the bottleneck.
- Single-stream decode is VRAM-bandwidth-bound on this box, so the
  narrow PCIe links matter less than one would expect; TP=2 per card
  generation + PP across generations is the operating point that wins.
