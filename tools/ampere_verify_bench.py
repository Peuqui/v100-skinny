"""Upstream-Kernel auf Ampere: Kosten des Multi-Token-Verify bei langem
Kontext. Misst paged varlen Attention fuer q=1 (Decode) gegen q>1
(Spekulations-Verify) bei identischem KV-Cache.

Laeuft gegen das UNVERAENDERTE vllm_flash_attn aus dem installierten vLLM.
"""
import time
import torch
from vllm.vllm_flash_attn import flash_attn_varlen_func

torch.manual_seed(7)
dev = "cuda:0"
N = 31488          # KV-Laenge (~31k Kontext)
BLOCK = 16
D = 128

print(f"GPU: {torch.cuda.get_device_name(0)} "
      f"(sm{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]})")
import vllm
print(f"vLLM: {vllm.__version__}, torch {torch.__version__}")
print(f"KV: {N} Token, Blockgroesse {BLOCK}, hdim {D}\n")

for H, HK in [(4, 1), (8, 2), (32, 8)]:
    nb = N // BLOCK
    k = torch.randn(nb + 1, BLOCK, HK, D, device=dev).half()
    v = torch.randn(nb + 1, BLOCK, HK, D, device=dev).half()
    bt = torch.arange(1, nb + 1, dtype=torch.int32, device=dev).unsqueeze(0)
    seqused = torch.tensor([N], dtype=torch.int32, device=dev)
    scale = D ** -0.5
    print(f"--- {H} Q-Heads / {HK} KV-Heads ---")
    base = None
    for q_len in (1, 2, 3, 4, 8):
        q = torch.randn(q_len, H, D, device=dev).half()
        cuq = torch.tensor([0, q_len], dtype=torch.int32, device=dev)

        def run():
            return flash_attn_varlen_func(
                q, k, v, max_seqlen_q=q_len, cu_seqlens_q=cuq,
                max_seqlen_k=N, seqused_k=seqused, softmax_scale=scale,
                causal=True, block_table=bt, fa_version=2)

        for _ in range(5):
            run()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        iters = 50
        for _ in range(iters):
            run()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / iters * 1000
        if q_len == 1:
            base = ms
            print(f"  q=1: {ms:.3f} ms  (Referenz)")
        else:
            print(f"  q={q_len}: {ms:.3f} ms  = {ms / base:.1f}x gegen q=1")
    del k, v
    torch.cuda.empty_cache()
    print()
