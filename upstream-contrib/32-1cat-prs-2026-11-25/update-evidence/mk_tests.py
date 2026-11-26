"""Build the per-file test result text and sources for one PR tree (runs5)."""
import re, sys, collections, json
from pathlib import Path
R = Path.home()/".cache/bench-scripts/prtest_2026-11-25/runs5"
DEVS = [("cpu", "CPU"), ("v100", "Tesla V100"), ("rtx", "Quadro RTX 8000")]
def summary(tree, dev):
    p = R/tree/dev/"summary.log"; out = {}
    if not p.exists(): return out
    for i, l in enumerate(p.read_text(errors="replace").splitlines(), 1):
        m = re.match(r'\S+ (ok|FAIL|HANG)\s+rc=(\S+)\s+(\d+)s (\S+)\s+(.*)', l)
        if m:
            res = re.sub(r'\s+in [\d.]+s( \(.*\))?$', '', m.group(5))
            res = re.sub(r', \d+ warnings?', '', res)
            out[m.group(4)] = (m.group(1), res, i)
    return out
def logfile(tree, dev, f): return R/tree/dev/(f.replace("/", "_")+".log")
def failing(tree, dev, f):
    lg = logfile(tree, dev, f)
    if not lg.exists(): return None, {}
    t = lg.read_text(errors="replace")
    ids = re.findall(r'^(?:FAILED|ERROR) (\S+)', t, re.M)
    secs = re.split(r'^_+ (.+?) _+$', t, flags=re.M); cs = {}
    for k in range(1, len(secs), 2):
        es = [x[1:].strip() for x in secs[k+1].splitlines() if x.startswith('E ') and x.strip() != 'E']
        pick = next((e for e in es if re.match(r'^[\w.]*(Error|Exception|Failed)\b', e)), es[0] if es else '?')
        cs[secs[k]] = re.sub(r'0x[0-9a-f]+', '0x..', pick)[:140]
    return ids, cs
def norm(i):  # test_dsv4_sparse_prefill_bmm: ours has one extra leading tensor_cores parameter (False on stock Triton)
    i = re.sub(r'^(tests/kernels/attention/test_dsv4_sparse_prefill_bmm.py::test_sparse_attn_prefill_bmm_matches_reference)\[False-', r'\1[', i)
    return re.sub(r'^(tests/kernels/attention/test_dsv4_sparse_prefill_bmm.py::test_prefill_retains_fp32_scores_before_softmax)\[False\]$', r'\1', i)
def main(tree, label):
    S = {d: summary(tree, d) for d, _ in DEVS}
    M = {d: summary("prmain2", d) for d, _ in DEVS}
    files = sorted(set().union(*[set(s) for s in S.values()]))
    lines, src, fails_out = [], [], []
    for f in files:
        parts = []
        for d, nm in DEVS:
            s = S[d].get(f)
            parts.append(f"{nm} {s[1] if s else 'not run'}")
            if s: src.append(f"{f} {nm}: runs5/{tree}/{d}/summary.log:{s[2]}")
        lines.append(f"- {f}: " + "; ".join(parts))
        for d, nm in DEVS:
            s = S[d].get(f)
            if not s or ' failed' not in s[1] and 'error' not in s[1]: continue
            ids, cs = failing(tree, d, f)
            mids = set(); mfound = False
            for mt in ("prmain2", ):
                mi, _ = failing(mt, d, f)
                if mi is not None: mids |= set(mi); mfound = True
            if d == "cpu":
                lg = R/"prmain2-indep"/"cpu"/(f.replace("/", "_")+".log")
                if not mfound and lg.exists():
                    mids |= set(re.findall(r'^(?:FAILED|ERROR) (\S+)', lg.read_text(errors="replace"), re.M)); mfound = True
            mnorm = {norm(i) for i in mids}
            same = mfound and all((i in mids) or (norm(i) in mids and d == "cpu") for i in ids)
            grp = collections.Counter(cs.values())
            ms = M[d].get(f)
            fails_out.append((f, nm, len(ids), same, mfound, dict(grp), ms[1] if ms else None, ms[2] if ms else None))
            src.append(f"{f} {nm} main: runs5/prmain2/{d}/summary.log:{ms[2] if ms else '-'}; causes: runs5/{tree}/{d}/{f.replace('/', '_')}.log")
    print(f"#TREE {tree} {label}")
    print("\n".join(lines))
    print("#FAILS")
    for f, nm, n, same, mfound, grp, ms, msl in fails_out:
        print(f"* {f} [{nm}] {n} failing ids; same ids on main: {same} (main run found: {mfound}); main result: {ms}")
        for k, v in sorted(grp.items(), key=lambda x: -x[1]): print(f"      {v} x {k}")
    print("#SOURCES")
    print("\n".join(src))
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "")
