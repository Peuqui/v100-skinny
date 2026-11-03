"""Rauchtest: Einbettungstabelle (GLM-5.3: 154 880 x 4 096 bf16 = 1,27 GiB) im gepinnten Hostspeicher per UVA.

Geprüft werden Korrektheit (bitgleich zur GPU-Tabelle), der Ladeweg (GPU-Tensor in die UVA-Sicht kopieren),
die Zeit für 8 Token (Decode/Verifikation) und 512 Token (Prefill-Chunk) und das Capture in einen CUDA-Graphen.
Aufruf mit PYTHONPATH=<Baum> und CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<Karte>.
"""
import sys

import torch
import torch.nn.functional as F

import vllm
import vllm._C  # noqa: F401  (registriert torch.ops._C.get_cuda_view_from_cpu_tensor)
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

VOCAB, HIDDEN = 154_880, 4_096
device = torch.device("cuda")
print("vllm aus", vllm.__file__, "| Karte", torch.cuda.get_device_name(0))

generator = torch.Generator(device="cpu").manual_seed(0)
reference = torch.randn(VOCAB, HIDDEN, generator=generator, dtype=torch.float32).to(torch.bfloat16).to(device)

host = torch.empty(VOCAB, HIDDEN, dtype=torch.bfloat16, device="cpu").pin_memory()
uva = get_accelerator_view_from_cpu_tensor(host)
uva.copy_(reference)  # derselbe Weg wie weight_loader: GPU-Tensor in die UVA-Sicht
torch.cuda.synchronize()
assert torch.equal(host.to(device), reference), "Ladeweg: Hostspeicher weicht von der Referenz ab"


def timed(function, repeats=200):
    function()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        function()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeats * 1000.0


for tokens in (8, 512):
    ids = torch.randint(0, VOCAB, (tokens,), device=device)
    assert torch.equal(F.embedding(ids, uva), F.embedding(ids, reference)), f"{tokens} Token: Ausgabe weicht ab"
    print(f"{tokens:4d} Token: GPU-Tabelle {timed(lambda: F.embedding(ids, reference)):8.1f} us | "
          f"UVA-Tabelle {timed(lambda: F.embedding(ids, uva)):8.1f} us | bitgleich")

ids = torch.randint(0, VOCAB, (8,), device=device)
out = torch.empty(8, HIDDEN, dtype=torch.bfloat16, device=device)
stream = torch.cuda.Stream()
with torch.cuda.stream(stream):
    out.copy_(F.embedding(ids, uva))
stream.synchronize()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    out.copy_(F.embedding(ids, uva))
ids.copy_(torch.randint(0, VOCAB, (8,), device=device))
graph.replay()
torch.cuda.synchronize()
assert torch.equal(out, F.embedding(ids, reference)), "CUDA-Graph: Ausgabe weicht ab"
print(f"   8 Token im CUDA-Graph (neue Ids nach Capture): bitgleich, Replay {timed(graph.replay):8.1f} us")
print("GPU-Speicher belegt (Tabelle nur im Host):", f"{torch.cuda.memory_allocated() / 2**30:.2f} GiB (Referenz {reference.numel() * 2 / 2**30:.2f} GiB)")
