# [Bug]: E5 metadata cache crashes on CSA/QSA models (qwen4_exp) — shape [] vs broadcast shape [1]

## Symptom

With `VLLM_SM70_E5_CACHE=1` (the default) and Qwen3.8-Flash-Next
(qwen4_exp, QSA sparse attention), the first decode after prefill dies on
every PP0 rank:

```
RuntimeError: output with shape [] doesn't match the broadcast shape [1]
  gpu_model_runner.py, in _e5_apply_ints
```

The crash site is `t[0].copy_(torch.tensor(row, ...))` on
`spec_state_indices_tensor`: the cache assumes a 2-D block-table layout,
but the QSA ring holds one block per request (1-D), so `t[0]` is a
scalar and the length-1 row assignment fails.

Follow-on damage under PP: worker exceptions are not propagated cleanly,
the ranks desynchronize, the next round trips the spec-decode stash guard
("missing stashed scheduler_output"), and the engine dies with
`RPC call to sample_tokens timed out` — which looks like a deadlock and
sent us chasing CUDA-graph ghosts for two sessions.

## Workaround

`VLLM_SM70_E5_CACHE=0` — but note `_E5_CACHE` is a module-level constant
read at import time; the variable must be in the environment before the
Python process starts. Setting it "too late" is silently ineffective,
which is how the workaround was once wrongly ruled out.

## Suggested fix

Either teach `_e5_apply_ints` the 1-D single-block layout, or
auto-disable the E5 cache when the model registers a CSA/QSA-style spec
(the incompatibility is structural). A hard error pointing at the env
var would already save users days.

With the cache off, qwen4_exp + MTP k=4 runs stable and coherent on our
rig (51.9 tok/s vs 32.2 without speculation after also fixing our
checkpoint's draft head — separate topic).

## Hardware environment (unusual, but that is the point)

- Host: GEM10 mini-PC (AMD Ryzen APU), 32 GB RAM (~30 GiB usable after iGPU carve-out), boot NVMe on USB 3.2
- GPUs: 5 external cards, 192 GB VRAM total — 2x Quadro RTX 8000 48 GB
  (sm75, GDDR6 672 GB/s) + 3x Tesla V100-PCIE-32GB (sm70, HBM2 900 GB/s)
- Attachment: 1 card on the GEM10's native OCuLink port, 3 via
  M.2-to-OCuLink adapters, 1 (one of the V100s) via a USB4 tunnel
  (AG02 dock) — and that tunneled V100 is an active
  member of the last PP stage, i.e. the MTP drafter itself runs on it. All five run PCIe Gen3 x4 under load (~3.9 GB/s per
  card); the USB4-tunneled card measures only ~5% slower than OCuLink.
- Peer-to-peer DMA between the Ryzen root ports does NOT work: NCCL P2P
  hangs in init (both ranks 100% util at ~40 W, spin-wait). We run
  `NCCL_P2P_DISABLE=1` everywhere; with TP allreduces in the ~10 KB
  range the host-RAM detour is not the bottleneck.
- Single-stream decode is VRAM-bandwidth-bound on this box, so the
  narrow PCIe links matter less than one would expect; TP=2 per card
  generation + PP across generations is the operating point that wins.
