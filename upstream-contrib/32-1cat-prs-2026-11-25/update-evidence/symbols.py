import re, subprocess, sys, glob, json, collections
repo = "/home/mp/Projekte/vllm-research/1Cat-vLLM"
ranges = {
 "1278": ("73da3445d", "pr-turing-kernels-r1125"),
 "1282": ("pr-turing-kernels-r1125", "pr-triton-sm7x-r1125"),
 "1347": ("pr-triton-sm7x-r1125~2", "pr-dsv4-longctx-decode-r1125"),
 "1286": ("pr-triton-sm7x-r1125~2", "pr-glm53-r1125"),
 "1348": ("pr-glm53-r1125~3", "pr-glm53-prefill-head-r1125"),
 "1221": ("73da3445d", "pr-mrv2-dspark-r1125"),
}
# load other PR diffs (added/removed lines only)
others = {}
for p in glob.glob(f"{sys.argv[1]}/diffs/*.diff"):
    n = p.split('/')[-1][:-5]
    lines = [l[1:] for l in open(p, errors="replace") if (l.startswith('+') or l.startswith('-')) and not l.startswith(('+++', '---'))]
    others[n] = "\n".join(lines)
res = {}
for pr, (a, b) in ranges.items():
    d = subprocess.run(["git", "-C", repo, "diff", "-U0", a, b], capture_output=True, text=True, errors="replace").stdout
    names = set()
    for l in d.splitlines():
        if not l.startswith('+') or l.startswith('+++'): continue
        for m in re.finditer(r'\b(?:def|class)\s+([A-Za-z_]\w{7,})', l): names.add(m.group(1))
        for m in re.finditer(r'\b(?:void|bool|int|auto|__global__\s+void|template)\s+\**([a-z_][a-z0-9_]{9,})\s*\(', l): names.add(m.group(1))
        m = re.match(r'\+_?([A-Z][A-Z0-9_]{8,}) *[:=]', l)
        if m: names.add(m.group(1))
    hits = {}
    for nm in sorted(names):
        hs = [n for n, t in others.items() if nm in t]
        if hs: hits[nm] = hs
    res[pr] = (len(names), hits)
    print(f"== #{pr}: {len(names)} added symbols, {len(hits)} found in other PR diffs")
    for nm, hs in hits.items(): print(f"   {nm}: {', '.join('#'+h for h in sorted(hs, key=int))}")
