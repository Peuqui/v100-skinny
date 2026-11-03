"""CPU check (Triton interpreter) of the rewritten CSF expansion kernel against the old one, byte for byte."""
import importlib.util
import sys

import torch

old_path, tree = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("csf_old", old_path)
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
spec = importlib.util.spec_from_file_location("csf_new", f"{tree}/vllm/model_executor/layers/quantization/utils/csf32_expand.py")
new = importlib.util.module_from_spec(spec); spec.loader.exec_module(new)

torch.manual_seed(0)
for rows, groups, projections, experts in ((128, 8, 1, 3), (256, 32, 2, 4), (128, 64, 2, 2)):
    matrices = experts * projections
    bases = torch.randint(100, 120, (matrices, rows), dtype=torch.uint8)
    codes = torch.randint(0, 256, (matrices, rows * groups // 2), dtype=torch.uint8)
    entries = torch.randperm(32, dtype=torch.int32)
    gids = torch.randperm(experts, dtype=torch.int32)
    num_slots = experts
    goff = torch.arange(num_slots + 1, dtype=torch.int32)  # one slot per group
    per = projections * rows * groups
    outs = []
    for module in (old, new):
        out = torch.zeros(experts, per, dtype=torch.uint8)
        row_tiles = projections * (rows // 128)
        tile_pairs = groups // 4
        runs = module._runs_per_row(tile_pairs, num_slots * row_tiles)
        kwargs = dict(ROWS=rows, GROUPS=groups, PROJECTIONS=projections, PAIRS_PER_PROGRAM=tile_pairs // runs)
        if module is new:
            kwargs["UNROLL"] = module._unroll(tile_pairs // runs)
        module._csf32_expand_kernel[(num_slots, row_tiles, runs)](
            out, out.stride(0), gids, goff, num_slots, bases, codes, entries, **kwargs)
        outs.append(out)
    same = torch.equal(outs[0], outs[1])
    written = int((outs[0] != 0).sum())
    print(f"rows {rows} groups {groups} proj {projections} experts {experts}: gleich={same}, beschriebene Bytes {written} von {outs[0].numel()}", flush=True)
    assert same
