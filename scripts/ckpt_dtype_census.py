"""Safetensors header census: per-module dtype/shape/bytes for a checkpoint.

Reads only the JSON header of each shard (no tensor data), so it is fast and
does not need a GPU. Used to compare the two served arms module-for-module.

Usage: python scripts/ckpt_dtype_census.py <ckpt_dir> [<ckpt_dir> ...]
"""
import json
import os
import re
import struct
import sys
from collections import defaultdict

DTYPE_BITS = {
    "F64": 64, "F32": 32, "F16": 16, "BF16": 16,
    "I64": 64, "I32": 32, "I16": 16, "I8": 8, "U8": 8, "BOOL": 8,
    "F8_E4M3": 8, "F8_E5M2": 8,
}


def read_header(path):
    with open(path, "rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        return json.loads(fh.read(n).decode("utf-8"))


def shard_paths(ckpt):
    idx = os.path.join(ckpt, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx) as fh:
            files = sorted(set(json.load(fh)["weight_map"].values()))
        return [os.path.join(ckpt, f) for f in files]
    return sorted(
        os.path.join(ckpt, f) for f in os.listdir(ckpt)
        if f.endswith(".safetensors")
    )


def nbytes(meta):
    off = meta.get("data_offsets")
    if off:
        return off[1] - off[0]
    n = 1
    for d in meta["shape"]:
        n *= d
    return n * DTYPE_BITS.get(meta["dtype"], 32) // 8


def canon(name):
    """Collapse layer indices so modules group across layers."""
    return re.sub(r"\.(\d+)\.", ".N.", name)


def census(ckpt):
    tensors = {}
    for p in shard_paths(ckpt):
        for k, v in read_header(p).items():
            if k == "__metadata__":
                continue
            tensors[k] = v
    return tensors


def report(ckpt):
    tensors = census(ckpt)
    print("=" * 78)
    print(f"CKPT {ckpt}   ({len(tensors)} tensors)")
    print("=" * 78)

    by_dtype = defaultdict(lambda: [0, 0])
    for k, v in tensors.items():
        e = by_dtype[v["dtype"]]
        e[0] += 1
        e[1] += nbytes(v)
    print("-- global dtype histogram --")
    for dt, (cnt, byt) in sorted(by_dtype.items(), key=lambda x: -x[1][1]):
        print(f"   {dt:<10} {cnt:>6} tensors  {byt / 2**30:8.3f} GiB")

    print("-- lm_head / embed --")
    for k in sorted(tensors):
        if "lm_head" in k or "embed_tokens" in k:
            v = tensors[k]
            print(f"   {k:<62} {v['dtype']:<8} {str(v['shape']):<20} "
                  f"{nbytes(v) / 2**20:9.2f} MiB")

    print("-- MTP / drafter modules (grouped, index collapsed) --")
    groups = defaultdict(lambda: [0, 0, set(), None])
    for k, v in tensors.items():
        low = k.lower()
        if not any(t in low for t in ("mtp", "draft", "nextn", "eh_proj",
                                      "shared_head", "speculat")):
            continue
        g = groups[canon(k)]
        g[0] += 1
        g[1] += nbytes(v)
        g[2].add(v["dtype"])
        g[3] = v["shape"]
    if not groups:
        print("   (no tensor name matched mtp/draft/nextn/eh_proj/shared_head)")
    for k in sorted(groups):
        cnt, byt, dts, shp = groups[k]
        print(f"   {k:<62} {','.join(sorted(dts)):<8} n={cnt:<4} "
              f"{str(shp):<18} {byt / 2**20:9.2f} MiB")

    print("-- decoder body: dtype per module family --")
    fam = defaultdict(lambda: [0, 0, set()])
    for k, v in tensors.items():
        low = k.lower()
        if any(t in low for t in ("mtp", "draft", "nextn", "lm_head",
                                  "embed_tokens")):
            continue
        f = fam[canon(k)]
        f[0] += 1
        f[1] += nbytes(v)
        f[2].add(v["dtype"])
    for k in sorted(fam):
        cnt, byt, dts = fam[k]
        if byt < 1 << 20:
            continue
        print(f"   {k:<62} {','.join(sorted(dts)):<8} n={cnt:<4} "
              f"{byt / 2**20:9.2f} MiB")
    print()


if __name__ == "__main__":
    for d in sys.argv[1:]:
        report(d)
