"""Kosten eines zufaelligen PLE-Zeilen-Lookups: VRAM gegen UVA-Host-RAM.

Bildet die reale PLE-Geometrie ab: Zeilen von 160 Byte (ple_embed_dim/ngram_heads
= 2560/16), FP8-Speicherung, gleichverteilte Hash-Indizes ueber die ganze Tabelle.
Pro Token fragt das Modell 16 Zeilen ab.
"""
import os
import sys
import time

import torch
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

ROWS = int(os.environ.get("ROWS", 25_000_000))   # ~4,0 GB bei 160 Byte/Zeile
COLS = 160
HEADS_PER_TOKEN = 16
DTYPE = torch.float8_e4m3fn
DEV = torch.device("cuda:0")


def timed_gather(table, ids, iters):
    torch.cuda.synchronize()
    for _ in range(3):                       # warmup
        torch.nn.functional.embedding(ids, table)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        torch.nn.functional.embedding(ids, table)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters


def main():
    print(f"device: {torch.cuda.get_device_name(DEV)}")
    print(f"table : {ROWS:_} x {COLS} {DTYPE} = {ROWS * COLS / 1024**3:.2f} GiB")

    cpu = torch.empty((ROWS, COLS), dtype=DTYPE, device="cpu")
    cpu.view(torch.int8).random_(-127, 127)
    print("pinning host tensor ...", flush=True)
    cpu = cpu.pin_memory()
    host_view = get_accelerator_view_from_cpu_tensor(cpu)
    print(f"uva view device={host_view.device} shape={tuple(host_view.shape)}")

    gpu = torch.empty((ROWS, COLS), dtype=DTYPE, device=DEV)
    gpu.view(torch.int8).copy_(cpu.view(torch.int8))
    print(f"vram copy allocated: {torch.cuda.memory_allocated(DEV) / 1024**3:.2f} GiB")

    gen = torch.Generator(device=DEV).manual_seed(1234)
    print(f"\n{'tokens':>8} {'lookups':>9} {'bytes':>10} "
          f"{'VRAM ms':>9} {'HOST ms':>9} {'faktor':>7} {'host GB/s':>10}")
    for tokens in (1, 8, 64, 512, 2048, 4096):
        n = tokens * HEADS_PER_TOKEN
        ids = torch.randint(0, ROWS, (n,), device=DEV, generator=gen)
        iters = 200 if tokens <= 512 else 30
        t_gpu = timed_gather(gpu, ids, iters)
        t_host = timed_gather(host_view, ids, iters)
        nbytes = n * COLS
        print(f"{tokens:>8} {n:>9} {nbytes/1024**2:>9.2f}M "
              f"{t_gpu*1e3:>9.3f} {t_host*1e3:>9.3f} "
              f"{t_host/t_gpu:>7.1f} {nbytes/t_host/1e9:>10.2f}")


if __name__ == "__main__":
    sys.exit(main())
