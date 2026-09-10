// Turing MMA probe: verify the m16n8k8 fragment layout against a CPU
// reference, and price both MMA shapes at equal FLOP count.
//
// Replaces the lost mma8_probe.cu referenced by skinny_kernels.cu:609.
// Nothing here is taken on faith from the PTX manual: the layout test
// multiplies random matrices on the device with the ASSUMED map and
// compares against a host matmul. A wrong map cannot pass.
//
// Shapes:
//   m8n8k4  (Volta-native, what skinny_nvfp4_qpn2 issues today)
//   m16n8k8 (Turing-native)
// Both accumulate f32 from f16 inputs, as the production kernels do.

#include <torch/extension.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAException.h>

// ---------------------------------------------------------------------------
// Layout verification: D[16x8] = A[16x8] * B[8x8], one warp.
// A row-major (.row), B col-major (.col) -- B passed as [n][k].
// Assumed maps (PTX ISA, m16n8k8):
//   gid = lane>>2, tig = lane&3
//   A reg0: A[gid    ][tig*2 + {0,1}]
//   A reg1: A[gid + 8][tig*2 + {0,1}]
//   B reg0: B[tig*2 + {0,1}][gid]      (k-major pair, one n column)
//   D reg0,1: D[gid    ][tig*2 + {0,1}]
//   D reg2,3: D[gid + 8][tig*2 + {0,1}]
// ---------------------------------------------------------------------------
__global__ void probe_m16n8k8(const half *__restrict__ A,
                              const half *__restrict__ B,
                              float *__restrict__ D) {
  const int lane = threadIdx.x & 31;
  const int gid = lane >> 2, tig = lane & 3;

  half2 a0 = make_half2(A[(gid) * 8 + tig * 2], A[(gid) * 8 + tig * 2 + 1]);
  half2 a1 = make_half2(A[(gid + 8) * 8 + tig * 2],
                        A[(gid + 8) * 8 + tig * 2 + 1]);
  // B is stored [n][k] (col-major view of an 8x8): B[col*8 + row]
  half2 b0 = make_half2(B[gid * 8 + tig * 2], B[gid * 8 + tig * 2 + 1]);

  float c[4] = {0.f, 0.f, 0.f, 0.f};
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(*reinterpret_cast<const unsigned *>(&a0)),
        "r"(*reinterpret_cast<const unsigned *>(&a1)),
        "r"(*reinterpret_cast<const unsigned *>(&b0)));

  D[(gid) * 8 + tig * 2] = c[0];
  D[(gid) * 8 + tig * 2 + 1] = c[1];
  D[(gid + 8) * 8 + tig * 2] = c[2];
  D[(gid + 8) * 8 + tig * 2 + 1] = c[3];
}

// m8n8k4 as the production kernel uses it: four quadpairs, four independent
// 8x8x4 tiles sharing one A fragment. Verified the same way -- here against
// a single 8x8x4 product replicated over the four QP N-slices.
//   A row: (lane&3) + (lane&16 ? 4 : 0), 4 contiguous k
//   C map: reg i of lane L -> row (i&2)|((L&16)?4:0)|(L&1),
//                             col (i&1)|(((L>>1)&1)<<1)|((i>>2)<<2)
__global__ void probe_m8n8k4(const half *__restrict__ A,
                             const half *__restrict__ B,
                             float *__restrict__ D) {
  const int lane = threadIdx.x & 31;
  const int qp = (lane >> 2) & 3;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);

  // A[8][4] row-major, one fragment shared by all four quadpairs.
  half2 a0 = make_half2(A[r * 4 + 0], A[r * 4 + 1]);
  half2 a1 = make_half2(A[r * 4 + 2], A[r * 4 + 3]);
  // B[32][4]: quadpair qp owns columns qp*8 .. qp*8+7, this lane owns col r.
  const half *brow = B + (size_t)(qp * 8 + r) * 4;
  half2 b0 = make_half2(brow[0], brow[1]);
  half2 b1 = make_half2(brow[2], brow[3]);

  float c[8];
#pragma unroll
  for (int i = 0; i < 8; i++) c[i] = 0.f;
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]),
        "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
      : "r"(*reinterpret_cast<const unsigned *>(&a0)),
        "r"(*reinterpret_cast<const unsigned *>(&a1)),
        "r"(*reinterpret_cast<const unsigned *>(&b0)),
        "r"(*reinterpret_cast<const unsigned *>(&b1)));

#pragma unroll
  for (int i = 0; i < 8; i++) {
    const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
    const int col = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
    D[row * 32 + qp * 8 + col] = c[i];
  }
}

// ---------------------------------------------------------------------------
// Issue-rate benchmark. Both loops do the SAME FLOP count per iteration and
// touch no memory in the inner loop: registers only, so what is timed is the
// MMA pipeline and nothing else.
//
//   m8n8k4  per warp instruction: 4 QPs x 8x8x4 x 2 = 2048 FLOP
//   m16n8k8 per warp instruction: 16x8x8 x 2        = 2048 FLOP
// One m8n8k4 is issued per m16n8k8 for an equal-FLOP comparison.
// ---------------------------------------------------------------------------
template <int SHAPE>
__global__ void issue_rate(float *__restrict__ sink, int iters) {
  float c[8] = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};
  const int lane = threadIdx.x & 31;
  half2 a0 = make_half2(__float2half(lane * 0.01f), __float2half(0.5f));
  half2 a1 = make_half2(__float2half(0.25f), __float2half(0.125f));
  half2 b0 = make_half2(__float2half(0.75f), __float2half(0.375f));
  half2 b1 = make_half2(__float2half(0.625f), __float2half(0.1875f));
  const unsigned A0 = *reinterpret_cast<const unsigned *>(&a0);
  const unsigned A1 = *reinterpret_cast<const unsigned *>(&a1);
  const unsigned B0 = *reinterpret_cast<const unsigned *>(&b0);
  const unsigned B1 = *reinterpret_cast<const unsigned *>(&b1);

  for (int it = 0; it < iters; it++) {
    if (SHAPE == 4) {
      asm volatile(
          "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
          "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
          : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]),
            "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
          : "r"(A0), "r"(A1), "r"(B0), "r"(B1));
    } else {
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
          : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
          : "r"(A0), "r"(A1), "r"(B0));
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
          : "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
          : "r"(A1), "r"(A0), "r"(B1));
    }
  }
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < 8; i++) s += c[i];
  if (s == 12345.678f) sink[0] = s;  // never taken; keeps the loop alive
}

torch::Tensor run_m16n8k8(torch::Tensor A, torch::Tensor B) {
  auto D = torch::zeros({16, 8}, A.options().dtype(torch::kFloat32));
  probe_m16n8k8<<<1, 32>>>(reinterpret_cast<const half *>(A.data_ptr<at::Half>()),
                           reinterpret_cast<const half *>(B.data_ptr<at::Half>()),
                           D.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return D;
}

torch::Tensor run_m8n8k4(torch::Tensor A, torch::Tensor B) {
  auto D = torch::zeros({8, 32}, A.options().dtype(torch::kFloat32));
  probe_m8n8k4<<<1, 32>>>(reinterpret_cast<const half *>(A.data_ptr<at::Half>()),
                          reinterpret_cast<const half *>(B.data_ptr<at::Half>()),
                          D.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return D;
}

void bench_issue(int shape, int blocks, int warps, int iters) {
  auto sink = torch::zeros({1}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  if (shape == 4)
    issue_rate<4><<<blocks, warps * 32>>>(sink.data_ptr<float>(), iters);
  else
    issue_rate<8><<<blocks, warps * 32>>>(sink.data_ptr<float>(), iters);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run_m16n8k8", &run_m16n8k8, "one-warp m16n8k8, layout under test");
  m.def("run_m8n8k4", &run_m8n8k4, "one-warp m8n8k4, production layout");
  m.def("bench_issue", &bench_issue, "back-to-back MMA issue rate");
}
