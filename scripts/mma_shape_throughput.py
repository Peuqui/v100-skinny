#!/usr/bin/env python3
"""Raw tensor-core throughput of Volta's mma.m8n8k4 against Turing's
mma.m16n8k8 (both fp16 in, fp32 accumulate, 2048 FLOP per warp instruction),
with eight independent accumulator chains per warp. Answers whether the
Volta MMA shape the skinny kernels use costs throughput on sm75.
Run: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<i> CUDA_HOME=/home/mp/vllm/cuda python scripts/mma_shape_throughput.py
"""
import os
import torch
from torch.utils.cpp_extension import load_inline

major, minor = torch.cuda.get_device_capability(0)
os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
CHAINS = 8
cuda_src = r"""
#include <cstdint>
template <int SHAPE>
__global__ void mma_loop(float *out, int iters) {
  uint32_t a[4] = {0x3c003c00u, 0x3c003c00u, 0x3c003c00u, 0x3c003c00u};
  uint32_t b[2] = {0x3c003c00u, 0x3c003c00u};
  float acc[CHAINS][8] = {};
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int c = 0; c < CHAINS; ++c) {
      if (SHAPE == 884) {
        asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};"
          : "+f"(acc[c][0]), "+f"(acc[c][1]), "+f"(acc[c][2]), "+f"(acc[c][3]),
            "+f"(acc[c][4]), "+f"(acc[c][5]), "+f"(acc[c][6]), "+f"(acc[c][7])
          : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(b[1]));
      } else {
#if __CUDA_ARCH__ >= 750
        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
          : "+f"(acc[c][0]), "+f"(acc[c][1]), "+f"(acc[c][2]), "+f"(acc[c][3])
          : "r"(a[0]), "r"(a[1]), "r"(b[0]));
#endif
      }
    }
  }
  float s = 0.f;
  for (int c = 0; c < CHAINS; ++c)
    for (int i = 0; i < 8; ++i) s += acc[c][i];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
void run(torch::Tensor out, int64_t iters, int64_t shape, int64_t blocks, int64_t threads) {
  if (shape == 884) mma_loop<884><<<blocks, threads>>>(out.data_ptr<float>(), iters);
  else mma_loop<1688><<<blocks, threads>>>(out.data_ptr<float>(), iters);
}
""".replace("CHAINS", str(CHAINS))
cpp_src = "void run(torch::Tensor out, int64_t iters, int64_t shape, int64_t blocks, int64_t threads);"
ext = load_inline(f"mma_shape_sm{major}{minor}", cpp_src, cuda_sources=cuda_src,
                  functions=["run"], extra_cuda_cflags=["-O3"], verbose=False)
sms = torch.cuda.get_device_properties(0).multi_processor_count
blocks, threads, iters = sms * 8, 256, 4096
out = torch.empty(blocks * threads, device="cuda")
print(f"device {torch.cuda.get_device_name()} sm{major}{minor}, {sms} SMs")
shapes = [884] + ([1688] if (major, minor) >= (7, 5) else [])
for shape in shapes:
    ext.run(out, 16, shape, blocks, threads)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    ext.run(out, iters, shape, blocks, threads)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end)
    warps = blocks * threads // 32
    flops = warps * iters * CHAINS * 2048
    print(f"mma {'m8n8k4 ' if shape == 884 else 'm16n8k8'}: {flops / (ms / 1e3) / 1e12:6.1f} TFLOP/s")
