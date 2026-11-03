"""Parameter counts from safetensors headers only (no weights read): routed experts vs rest, quantized packing undone."""
import glob, json, re, struct, sys
from collections import defaultdict

model = sys.argv[1]
cfg = json.load(open(f"{model}/config.json"))
cfg = cfg.get("text_config", cfg)
n_exp, top_k = cfg["n_routed_experts"], cfg["num_experts_per_tok"]
cat = defaultdict(lambda: defaultdict(float))   # group -> kind -> params
seen = defaultdict(set)
for path in sorted(glob.glob(f"{model}/*.safetensors")):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    for name, t in header.items():
        if name == "__metadata__":
            continue
        numel = 1
        for d in t["shape"]:
            numel *= d
        low = name.lower()
        if "scale" in low or "global" in low or low.endswith(".bias_dummy"):
            cat["(Skalen, keine Parameter)"][t["dtype"]] += numel
            continue
        group = "mtp/Entwurf" if re.search(r"(^|\.)mtp(\.|$)|nextn|draft", low) else "Hauptmodell"
        if re.search(r"\.experts\.\d+\.", name) or re.search(r"\.experts\.(w|gate|up|down)", low):
            kind = "routed Experten"
        elif "shared_expert" in low:
            kind = "shared Experte"
        elif "embed" in low or "lm_head" in low:
            kind = "Einbettung/lm_head"
        else:
            kind = "Rest (Attention, dichte MLP, Router, Norm, ...)"
        pack = 2 if (t["dtype"] == "U8" and kind in ("routed Experten", "shared Experte") and name.endswith("weight")) else 1
        cat[group][kind + f" [{t['dtype']}{'x2 gepackt' if pack == 2 else ''}]"] += numel * pack
for g, d in cat.items():
    print("==", g)
    for k, v in sorted(d.items()):
        print(f"   {k:70s} {v/1e9:9.3f} Mrd")
