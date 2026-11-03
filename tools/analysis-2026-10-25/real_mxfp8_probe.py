"""Real GLM-5.3 MXFP8 tensors through the new QPN8 row-group mode: error against the fp16 weights the loader makes today,
and decode time (CUDA graph, 10 calls per replay, ms, less is better) against the fp16 cuBLAS GEMV it replaces."""
import json, statistics, sys
import torch
from safetensors import safe_open
from torch import nn

from vllm.model_executor.kernels.linear.qpn import pre_ampere  # noqa: F401
from vllm.model_executor.layers.quantization import sm70_turbomind as tm
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import dequant_mxfp8_to_bf16

ROOT = "/home/mp/models/GLM-5.3-Flash-NVFP4-MTP"
index = json.load(open(f"{ROOT}/model.safetensors.index.json"))["weight_map"]
MODULES = [
    "model.language_model.layers.4.self_attn.o_proj",
    "model.language_model.layers.0.mlp.gate_proj",
    "model.language_model.layers.0.mlp.down_proj",
    "model.language_model.layers.5.self_attn.o_proj",
]


def graph_ms(fn, calls=10, reps=9):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); graph.replay(); b.record(); b.synchronize()
        times.append(a.elapsed_time(b) / calls)
    return statistics.median(times)


def load(name):
    out = {}
    for suffix in ("weight", "weight_scale"):
        key = f"{name}.{suffix}"
        with safe_open(f"{ROOT}/{index[key]}", "pt", device="cuda") as sf:
            out[suffix] = sf.get_tensor(key)
    return out


print(torch.cuda.get_device_name(0), flush=True)
torch.manual_seed(0)
for name in MODULES:
    t = load(name)
    weight, scale = t["weight"], t["weight_scale"]
    n, k = weight.shape
    fp16_weight = dequant_mxfp8_to_bf16(weight, scale).to(torch.float16)
    layer = nn.Module()
    tm.prepare_mxfp8_qpn8_linear(layer, weight, scale)
    short = name.split("layers.")[1]
    for rows in (1, 4, 8, 33, 512):
        x = (torch.randn(rows, k, device="cuda") * 0.5).to(torch.float16)
        out = tm.apply_prepared_fp8_qpn8_linear(layer, x, None)
        ref = torch.nn.functional.linear(x, fp16_weight)
        exact = x.double() @ fp16_weight.double().t()
        err_q = ((out.double() - exact).abs().max() / exact.abs().max()).item()
        err_c = ((ref.double() - exact).abs().max() / exact.abs().max()).item()
        line = f"{short:32s} N {n:5d} K {k:5d} M {rows:3d} | rel err vs exact: QPN8 {err_q:.1e}  fp16 cuBLAS {err_c:.1e}"
        if rows <= 8:
            q_ms = graph_ms(lambda: tm.apply_prepared_fp8_qpn8_linear(layer, x, None))
            c_ms = graph_ms(lambda: torch.nn.functional.linear(x, fp16_weight))
            line += f" | ms QPN8 {q_ms:.3f}  cuBLAS {c_ms:.3f}  ratio {q_ms / c_ms:.2f}  ({(n * k * (1 + 1/32)) / q_ms / 1e6:.0f} GB/s)"
        print(line, flush=True)
    del layer, fp16_weight, weight, scale
    torch.cuda.empty_cache()
