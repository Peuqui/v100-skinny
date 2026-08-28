# Titel: PP x TP with MTP on mixed Turing+Volta: 2x2 grid, hybrid-model spec transport, device-0 capability fixes, PLE VRAM->host cascade

Ziel: dnv2003/v100-skinny · Basis: main · Branch: pp-mtp-merge (15 Commits)
Voraussetzung vor dem Posten: Fork unter Peuquis Account, `git push fork pp-mtp-merge`.

---

You asked for reproductions and findings from real deployments — here is
a big one. This branch makes a single vLLM span a heterogeneous
5-GPU box (2x Quadro RTX 8000 sm75 + 3x Tesla V100 sm70): TP=2 within
each card generation, PP across the generation boundary, MTP on top.

## Headline numbers (Qwen3.8-27B-NVFP4, bench.py, n=3)

| Topology | plain | MTP k=7 |
|---|---:|---:|
| PP=2 (2x V100) | 32.3 | 79.8 ± 1.2 |
| 2x2 grid (TP2 RTX stage + TP2 V100 stage) | 32.2 | **85.0 ± 1.1** |

Also validated on Qwen3.8-Flash-Next (qwen4_exp, 125B+PLE): boots on the
same grid, 8/8 coherence, up to 51.9 tok/s with MTP k=4 vs 32.2 without.

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

## What's in the branch

1. **PP spec-decode transport** (async scheduling): the last stage
   broadcasts the full sampled matrix [num_reqs, k+1] plus the draft
   token ids [num_reqs, k]; non-last ranks derive next-token ids,
   accepted counts and the hybrid (GDN/mamba) state rollback locally.
   Without this, draft token VALUES only exist on the last rank and PP0
   crashes in round 1 (masked as an engine deadlock by the executor).
2. **Output-length trim on all ranks** (elif -> if in _update_states):
   non-last ranks extend output_token_ids optimistically every spec
   round and were never trimmed; the overshoot eventually flips the
   discard mask on one rank and the PP broadcast guard desynchronizes
   until NCCL wedges. This one is an UPSTREAM vLLM bug too (verified
   present in 0.27.1) — we intend to report it there separately.
3. **sm75 GDN package**: upstream-0.27.1 copies of the GDN attention
   pair + upstream FLA ops, registered under suffixed names, dispatched
   per-stage by local device capability. The fork's Triton prefill path
   miscomputes on both generations; per-stage backend selection is the
   only correct route on mixed rigs.
4. **Device-0 capability fixes** (three instances of the same bug
   class): SM70 baseline env defaults, quantization config gate, and
   ModelOptNvFp4Config.get_min_capability() — all decided by device 0
   of the visibility list, all wrong on mixed rigs. Details and failure
   modes in the commit messages; the "silently degraded MTP verifier"
   variant cost us a full session.
5. **PLE VRAM->host cascade** for qwen4_exp: the 51 GiB hash-n-gram
   table is pinned to one layer; the cascade keeps as much as fits in
   VRAM and serves the rest from pinned host memory (UVA), auto-sized
   from the KV budget. Makes TP=1 bootable and full 262k context
   reachable on 4 cards.
6. **Serve script** with observed-execution boot gates (XQA path,
   census, spec depth), SPEC_ATTN override (drafter lives on the last
   stage whose arch may differ from the global backend), and the
   platform workarounds this box needs (NCCL_P2P_DISABLE etc. — Ryzen
   root-port P2P is broken, documented in the script).

## Known limitations / honest notes

- Capture sizes >8 degrade or hang qwen4_exp spec decoding on this
  stack; we run [1,2,4,5,8]. Not yet root-caused.
- Without the XQA path the SM70 MTP verifier computes silently wrong at
  q>1 (degrading but coherent-looking output) — worth a hard fail.
  The XQA kernel also only covers q_per_kv in {4,6,8}; qwen4_exp's 12
  falls through to its own QSA backend, so this is a latent trap for
  other models.
- E5 metadata cache is structurally incompatible with the QSA ring
  (crash + masked PP death); we run VLLM_SM70_E5_CACHE=0 for qwen4_exp.
- The PR's IndexShare hooks (set_skip_topk/compact_topk_indices) are dead
  code. We measured the drafter's QSA indexer with CUDA events at three
  context lengths (19 / 29,579 / 91,600 prompt tokens): per-call cost is
  flat at 0.2-0.9 ms across all of them — the top-k over compressed keys
  is launch-overhead-dominated even at 22.9k entries. Wiring IndexShare
  up is worth only 1-4% below ~100k context; the feature targets far
  longer regimes.

Full measurement history, dead ends and repro commands are in the two
handover documents in the repo root of our working tree; happy to trim
them into the PR description or a docs/ file if you prefer.
