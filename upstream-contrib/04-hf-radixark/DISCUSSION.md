# Titel-Vorschlag: MTP block is unquantized (4.86 GiB BF16) — makes speculative decoding a net LOSS on pre-Hopper GPUs (fix: 1.5 GiB transplant, 14 -> 52 tok/s)

Ziel: Discussion-Tab von huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4

---

First off: thanks for publishing this checkpoint — the main model runs
great on our rig. This is not a defect report about the quantization
itself, but a heads-up about a side effect that probably hits everyone
who is *not* on Hopper/Blackwell, plus a cheap fix.

## The issue

The quantization config excludes the MTP draft head (`ignore: mtp.*`),
so the 31 MTP tensors ship as **4.86 GiB of BF16** while the main model
is NVFP4. Single-stream decode is bandwidth-bound: the NVFP4 main model
reads ~6.5 GiB/token, so **one draft step reads almost as many bytes as
the entire 125B forward pass** — and MTP runs k draft steps per
iteration.

On a B200 (~8 TB/s HBM3e) those 4.86 GiB cost ~0.6 ms and nobody
notices. On our mixed Quadro RTX 8000 (672 GB/s) + Tesla V100 (900 GB/s)
box, measured with CUDA events over 120 iterations:

```
target_forward (full 125B model): 24.3 ms
draft_total (4 draft steps):      83.9 ms   <- 3.4x the target model
```

Net effect: MTP made generation **slower** than no speculation at all
(14.0 vs 32.2 tok/s on free-form text), even though the draft head
itself is fine (acceptance length 3.9, right in line with the 3.3
reported for B200). No acceptance rate can amortize a draft step that
costs as much as the model it is speculating for.

## The fix (no re-download of the model needed)

Some community re-quants ship an NVFP4 MTP block (~1.49 GiB) as a
self-contained shard, e.g. `nvfp4_experts_mtp.safetensors` in
provsalt/Qwen3.8-Flash-Next-NVFP4-PLE-NVFP4. Transplanting just that
file into this checkpoint works because the per-expert layout matches
the main model's expert layout exactly:

1. new model dir with symlinks to the existing snapshot files,
2. rewrite `model.safetensors.index.json`: drop the 31 `mtp.*` entries,
   map the 6173 quantized `mtp.*` tensors to the new shard,
3. in `config.json` remove `mtp.*` / `model.mtp.*` from
   `quantization_config.ignore`, but ADD the four tensors that stay
   BF16 inside the new block (`mtp.fc_embedding`, `mtp.fc_hidden`,
   `mtp.pre_fc_norm_embedding`, `mtp.pre_fc_norm_hidden`) — the first
   two are real Linears and the loader will otherwise look for scales
   that do not exist.

Results on our rig (TP=2, PP=2, k=4, temperature 0, 200-token decode,
identical operating point):

```
no speculation:              32.2 tok/s
k=4, BF16 MTP (as shipped):  14.0 tok/s (hard prompt) / 19.4 (easy)
k=4, NVFP4 MTP (transplant): 51.9 tok/s (hard prompt) / 68.2 (easy)
```

Acceptance 70-73%, output quality unchanged (speculative decoding is
verified by the target model, so a quantized drafter can only cost
acceptance, never correctness — and acceptance actually went up).

## Ask

Would you consider publishing an NVFP4 (or FP8) MTP block alongside the
checkpoint, or including it in a future revision? It is ~1.5 GiB and
turns MTP from a net loss into a 1.6x gain for everyone below Hopper.
Happy to share exact scripts/measurements.
