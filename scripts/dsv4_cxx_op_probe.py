"""Second pass: the two DSv4 C++ ops whose first probe hit my own arg errors."""
import torch
from vllm import _custom_ops as ops

cap = torch.cuda.get_device_capability()
print(f"device={torch.cuda.get_device_name()} sm_{cap[0]}{cap[1]}")
D = "cuda"
HEAD, QB, TOPK = 128, 128, 512


def run(name, fn):
    try:
        fn()
        torch.cuda.synchronize()
        print(f"  {name}: OK")
    except Exception as exc:
        print(f"  {name}: FAIL -- {' '.join(str(exc).split())[:220]}")


def indexer_cache(scale_fmt):
    def _f():
        num_tokens, block_size, num_blocks = 4, 64, 8
        k = torch.randn(num_tokens, HEAD, device=D, dtype=torch.bfloat16)
        # (num_blocks, block_size, head_size): 128 fp8 bytes + 4 scale bytes
        cache = torch.zeros(num_blocks, block_size, HEAD + 4, device=D, dtype=torch.uint8)
        slots = torch.arange(num_tokens, device=D, dtype=torch.int64)
        ops.indexer_k_quant_and_cache(k, cache, slots, QB, scale_fmt)
    return _f


def persistent_topk():
    rows, cols, max_seq = 4, 2048, 2048
    logits = torch.randn(rows, cols, device=D, dtype=torch.float32)
    seq_lens = torch.full((rows,), cols, device=D, dtype=torch.int32)
    out = torch.zeros(rows, TOPK, device=D, dtype=torch.int32)
    ws = torch.zeros(1024 * 1024, device=D, dtype=torch.uint8)
    torch.ops._C.persistent_topk(logits, seq_lens, out, ws, TOPK, max_seq)


for fmt in ("ue8m0", "fp8_e4m3", "auto"):
    run(f"indexer_k_quant_and_cache[{fmt}]", indexer_cache(fmt))
run("persistent_topk", persistent_topk)
