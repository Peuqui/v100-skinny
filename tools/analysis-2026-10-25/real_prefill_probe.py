"""Prefill-sized M on real GLM MXFP8 projections: ms per call in CUDA graphs (less is better) for
(A) apply_prepared (Turing tensor-core GEMM on the 8-bit codes; Volta: dequantize + cuBLAS),
(B) dequantize + cuBLAS explicitly (the route of the Phase-2 profile),
(C) fp16 cuBLAS on the resident dequantized weight (the route before MXFP8 stayed packed)."""
import json, statistics
import torch
from safetensors import safe_open
from torch import nn

from vllm import _sm70_ops
from vllm.model_executor.kernels.linear.qpn import pre_ampere  # noqa: F401
from vllm.model_executor.layers.quantization import sm70_turbomind as tm
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import dequant_mxfp8_to_bf16

ROOT = "/home/mp/models/GLM-5.3-Flash-NVFP4-MTP"
index = json.load(open(f"{ROOT}/model.safetensors.index.json"))["weight_map"]
MODULES = ["layers.4.self_attn.o_proj", "layers.1.self_attn.q_proj", "layers.0.mlp.gate_proj", "layers.0.mlp.down_proj"]


def graph_ms(fn, calls=5, reps=7):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            fn()
    graph.replay(); torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); graph.replay(); b.record(); b.synchronize()
        times.append(a.elapsed_time(b) / calls)
    return statistics.median(times)


print(torch.cuda.get_device_name(0), flush=True)
for name in MODULES:
    t = {}
    for suffix in ("weight", "weight_scale"):
        key = f"model.language_model.{name}.{suffix}"
        with safe_open(f"{ROOT}/{index[key]}", "pt", device="cuda") as sf:
            t[suffix] = sf.get_tensor(key)
    weight, scale = t["weight"], t["weight_scale"]
    n, k = weight.shape
    fp16_weight = dequant_mxfp8_to_bf16(weight, scale).to(torch.float16)
    layer = nn.Module()
    tm.prepare_mxfp8_qpn8_linear(layer, weight, scale)
    state = getattr(layer, tm.FP8_QPN8_STATE_ATTR)
    dense = torch.empty(k, n, device="cuda", dtype=torch.float16)
    for rows in (16, 64, 256, 512):
        x = (torch.randn(rows, k, device="cuda") * 0.5).to(torch.float16)
        out = tm.apply_prepared_fp8_qpn8_linear(layer, x, None)
        ref = torch.nn.functional.linear(x, fp16_weight)
        exact = x.double() @ fp16_weight.double().t()
        err = ((out.double() - exact).abs().max() / exact.abs().max()).item()
        a = graph_ms(lambda: tm.apply_prepared_fp8_qpn8_linear(layer, x, None))
        def dequant_route():
            _sm70_ops.fp8_qpn8_dequantize_sm70_out(dense, state.codes, state.group_scales)
            return torch.mm(x, dense)
        b = graph_ms(dequant_route)
        c = graph_ms(lambda: torch.nn.functional.linear(x, fp16_weight))
        print(f"{name:26s} N {n:5d} K {k:5d} M {rows:3d} | rel err {err:.1e} | ms: A apply {a:.3f}  B dequant+cuBLAS {b:.3f}  C resident fp16 {c:.3f} | A/C {a / c:.2f}  B/C {b / c:.2f}  (mma={state.sm75_mma})", flush=True)
    del layer, fp16_weight, weight, scale, dense
    torch.cuda.empty_cache()
