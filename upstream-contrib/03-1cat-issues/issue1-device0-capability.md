# [Bug]: Capability gates probe device 0 only — breaks every heterogeneous multi-GPU deployment

## Affected (three independent instances of the same pattern)

1. **SM70 baseline env defaults** (`vllm/config/vllm.py`): the block that
   auto-sets the eight/nine `VLLM_SM70_*` defaults is gated on
   `current_platform.is_device_capability((7, 0))`, which only inspects
   device 0 of the visible list. In a mixed Turing+Volta rig with an RTX
   card first (`CUDA_VISIBLE_DEVICES=0,2,1,4`, PP stage 0 = sm75), the
   whole block is skipped and the V100 PP stage runs without its tuning.
   Observed: 0 instead of 9 "Auto-setting VLLM_SM70_*" log lines; MTP
   acceptance degraded from ~21% to 2-9%, output word salad.
2. **Quantization gate** (`VllmConfig._get_quantization_config`): calls
   `current_platform.get_device_capability()` without a device_id, i.e.
   device 0 speaks for the whole system. With a V100 first, a
   modelopt_fp4 checkpoint refuses to boot ("Minimum capability: 75.
   Current capability: 70") even though the same rig computes NVFP4
   layers on those V100s in production when an RTX happens to be first.
3. **`ModelOptNvFp4Config.get_min_capability()`** returned a hard 75 while
   the sibling configs (e.g. `ModelOptFp8Config`) already return
   `_SM70_MIN_CAP` when the SM70 modelopt path is enabled — NVFP4 was
   simply missed during the Volta enablement.

## Why it matters

Heterogeneous rigs (mixed Turing/Volta/Ampere) are common in exactly the
homelab segment this fork targets. The failure modes are nasty because
they are *silent or misleading*: (1) degrades quality without an error,
(2) blocks boots that would work, (3) blocks a card class that provably
works.

## Fix we run

- Gate (1) on "any visible device has capability (7,0)" — the defaults
  are already SM70-gated at their point of use, so Turing stages ignore
  them; homogeneous setups are unchanged.
- (3): return `_SM70_MIN_CAP if _SM70_MODELOPT else 75`, matching the
  sibling configs.
- (2) is the same one-line pattern as (1) if you want full symmetry.

Patches available; happy to open a PR. Verified on 2x RTX 8000 + 2x V100,
TP=2/PP=2, both card orders, eleven boot gates green, throughput
unchanged on homogeneous setups.

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
