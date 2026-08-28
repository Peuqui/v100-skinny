"""Unit-Test der geteilten PLE-Platzierung ohne Modell.

Prueft, dass Laden und Lookup bitgleich zum ungeteilten Pfad bleiben -- fuer
mehrere Host-Anteile, mehrere TP-Raenge und mit der echten Shard-Geometrie.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = torch.device("cuda:0")
DTYPE = torch.float8_e4m3fn
COLS = 160
ROWS = 65536          # TP-lokale Zeilen
SHARDS = 8
FAILURES = []


def check(name, ok):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        FAILURES.append(name)


def build_checkpoint(global_rows):
    """Ein Checkpoint als Shard-Liste, wie load_weights ihn sieht."""
    full = torch.empty((global_rows, COLS), dtype=DTYPE, device="cpu")
    full.view(torch.int8).random_(-127, 127)
    shard_size = (global_rows + SHARDS - 1) // SHARDS
    shards = []
    for i in range(SHARDS):
        start = i * shard_size
        rows = max(0, min(shard_size, global_rows - start))
        shards.append((start, full[start:start + rows].clone()))
    return full, shards


def setup_distributed():
    """ModelWeightParameter reads the TP rank; a single-rank group suffices.

    The TP ranges under test are passed explicitly, so they stay independent of
    this group.
    """
    from vllm.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29591")
    from vllm.config import VllmConfig, set_current_vllm_config
    init_distributed_environment(world_size=1, rank=0, local_rank=0,
                                 distributed_init_method="env://", backend="gloo")
    ctx = set_current_vllm_config(VllmConfig())
    ctx.__enter__()
    initialize_model_parallel(tensor_model_parallel_size=1)
    return ctx


def main():
    from vllm.models.qwen4_exp.amd.ple_layer import Qwen4ExpPLEFp8EmbeddingMethod
    from vllm.models.qwen4_exp.common.ple import plan_ple_placement

    _ctx = setup_distributed()

    print("== plan_ple_placement ==")
    p = plan_ple_placement(total_rows=1000, row_bytes=100, host_budget_bytes=0)
    check("budget 0 keeps everything on device", (p.vram_rows, p.host_rows) == (1000, 0))
    p = plan_ple_placement(total_rows=1000, row_bytes=100, host_budget_bytes=25_000)
    check("budget 25k bytes -> 250 host rows", (p.vram_rows, p.host_rows) == (750, 250))
    p = plan_ple_placement(total_rows=1000, row_bytes=100, host_budget_bytes=10**9)
    check("oversized budget clamps to table", (p.vram_rows, p.host_rows) == (0, 1000))
    try:
        plan_ple_placement(total_rows=-1, row_bytes=100, host_budget_bytes=0)
        check("negative row count rejected", False)
    except ValueError:
        check("negative row count rejected", True)

    tp_size = 2
    global_rows = ROWS * tp_size
    full, shards = build_checkpoint(global_rows)

    for host_gib in (0.0, 0.002, 0.005, 0.02):
        budget = int(host_gib * 1024**3)
        print(f"\n== host budget {host_gib} GiB ({budget} bytes) ==")
        for tp_rank in range(tp_size):
            tp_start = tp_rank * ROWS
            tp_end = tp_start + ROWS

            os.environ["VLLM_QWEN4EXP_PLE_HOST_GIB"] = str(host_gib)
            method = Qwen4ExpPLEFp8EmbeddingMethod(budget)
            layer = nn.Module()
            with torch.device(DEV):
                method.create_weights(layer, COLS, [ROWS], COLS, ROWS,
                                      torch.float16, weight_loader=None)
            check(f"rank{tp_rank}: placeholder has no rows",
                  layer.weight.shape == (0, COLS))

            for checkpoint_start, shard in shards:
                method.load_shard(layer, shard, checkpoint_start=checkpoint_start,
                                  tp_start=tp_start, tp_end=tp_end)
            method.process_weights_after_loading(layer)

            pl = method.placement
            expect_host = min(ROWS, budget // COLS)
            check(f"rank{tp_rank}: placement {pl.vram_rows}/{pl.host_rows}",
                  (pl.vram_rows, pl.host_rows) == (ROWS - expect_host, expect_host))
            check(f"rank{tp_rank}: host tensor pinned or empty",
                  layer.ple_host_storage.numel() == 0
                  or layer.ple_host_storage.is_pinned())

            # Die geladenen Bytes muessen dem TP-Ausschnitt entsprechen.
            reference = full[tp_start:tp_end].to(DEV)
            loaded = torch.cat([
                layer.weight.data.view(torch.int8).cpu(),
                layer.ple_host_storage.view(torch.int8).cpu(),
            ], dim=0)
            check(f"rank{tp_rank}: loaded bytes match checkpoint",
                  torch.equal(loaded, full[tp_start:tp_end].view(torch.int8)))

            gen = torch.Generator(device=DEV).manual_seed(11)
            ids = torch.randint(0, ROWS, (4096,), device=DEV, generator=gen)
            got = method.embedding(layer, ids)
            want = F.embedding(ids, reference)
            check(f"rank{tp_rank}: lookup bit-identical",
                  got.dtype == DTYPE
                  and torch.equal(got.view(torch.int8), want.view(torch.int8)))

            del layer, reference
            torch.cuda.empty_cache()

    print(f"\n{'ALL PASS' if not FAILURES else 'FAILURES: ' + ', '.join(FAILURES)}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
