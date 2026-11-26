"""DeepSeek-V4 SM70 indexer prefill epilogue on the fp32 score tile (25.11.2026).

The commit "vectorize the sm70 indexer prefill epilogue loads" chose BLOCK_K 512 over
2 warps when the tile was fp16 (21.10.2026: 105 -> 562 GB/s on RTX 8000, 184 -> 782 GB/s
on V100). Since 02.11. the tile is fp32. This measures both launch shapes on the fp32 tile
inside a CUDA graph, real DeepSeek-V4-Flash indexer shapes (64 index heads), and checks
that both write the same bytes.

Run inside the tree under test: python bench_... (cwd = tree), one GPU visible.
"""

import torch

from vllm.models.deepseek_v4.sm70.indexer import (
    _EPILOGUE_BLOCK_H,
    _relu_weight_headsum_kernel,
)

HEADS = 64
REPEATS = 20
VARIANTS = [(256, 4), (512, 2)]


def launch(scores, weights, scales, out, num_q, width, block_k, warps):
    _relu_weight_headsum_kernel[(num_q, triton_cdiv(width, block_k))](
        scores,
        weights,
        scales,
        out,
        scores.stride(0) * HEADS,
        scores.stride(0),
        weights.stride(0),
        out.stride(0),
        width,
        num_heads=HEADS,
        BLOCK_K=block_k,
        BLOCK_H=_EPILOGUE_BLOCK_H,
        num_warps=warps,
    )


def triton_cdiv(a, b):
    return (a + b - 1) // b


def main():
    device = torch.device("cuda")
    name = torch.cuda.get_device_name(device)
    torch.manual_seed(0)
    for num_q, width in [(1024, 1024), (1024, 2048), (512, 4096)]:
        scores = torch.randn(num_q * HEADS, width, device=device, dtype=torch.float32)
        weights = torch.randn(num_q, HEADS, device=device, dtype=torch.float32)
        scales = torch.rand(width, device=device, dtype=torch.float32)
        tile_bytes = scores.numel() * 4
        outputs = {}
        for block_k, warps in VARIANTS:
            out = torch.empty(num_q, width, device=device, dtype=torch.float32)
            args = (scores, weights, scales, out, num_q, width, block_k, warps)
            launch(*args)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(REPEATS):
                    launch(*args)
            graph.replay()
            torch.cuda.synchronize()
            times = []
            for _ in range(10):
                start = torch.cuda.Event(enable_timing=True)
                stop = torch.cuda.Event(enable_timing=True)
                start.record()
                graph.replay()
                stop.record()
                stop.synchronize()
                times.append(start.elapsed_time(stop) * 1000 / REPEATS)
            us = sorted(times)[len(times) // 2]
            outputs[(block_k, warps)] = out
            print(
                f"{name} tokens={num_q} keys={width} BLOCK_K={block_k} warps={warps}: "
                f"{us:.1f} us per call, {tile_bytes / us / 1e3:.0f} GB/s over the fp32 tile"
            )
        reference = outputs[VARIANTS[0]]
        for variant, out in outputs.items():
            print(f"  {variant} byte-identical to {VARIANTS[0]}: {torch.equal(out, reference)}")


if __name__ == "__main__":
    main()
