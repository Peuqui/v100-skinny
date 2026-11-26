"""Text blocks for the test results of one tree (runs5)."""
import re, collections
from pathlib import Path
import mk_tests as mt

DEV_LABEL = {"cpu": "CPU", "v100": "Tesla V100", "rtx": "Quadro RTX 8000"}
CAUSE_FIX = {
    "triton.compiler.errors.CompilationError: at 1:0:": "Triton CompilationError (fp8e4nv is not supported on sm_75)",
}
def short_cause(c):
    c = CAUSE_FIX.get(c, c)
    c = re.sub(r"TypeError: test_compressed_tensors_channel_fp8_qpn8_prepares_and_dispatches\.<locals>\.<lambda>\(\) got an unexpected keyword argument 'cap_to_b.*", "TypeError: a test stand-in lambda got an unexpected keyword argument 'cap_to_bounded_ceiling'", c)
    c = c.replace("AssertionError: assert (16, 2, False) is None", "AssertionError: assert (16, 2, False) is None (lm_head shape gate)")
    c = re.sub(r"RuntimeError: Test subprocess '(\w+)' failed \(exit code 1\):", r"RuntimeError: test subprocess '\1' failed", c)
    return c

def data(tree):
    S = {d: mt.summary(tree, d) for d, _ in mt.DEVS}
    files = sorted(set().union(*[set(s) for s in S.values()]))
    out = []
    for f in files:
        row = {}
        for d, nm in mt.DEVS:
            s = S[d].get(f)
            if s: row[d] = dict(res=s[1], line=s[2])
        out.append((f, row))
    return S, out

def result_lines(tree):
    S, rows = data(tree)
    lines = []
    for f, row in rows:
        parts = [f"{mt.DEVS[i][1]} {row[d]['res']}" if d in row else f"{mt.DEVS[i][1]} not run" for i, (d, _) in enumerate(mt.DEVS)]
        lines.append(f"- {f}: " + "; ".join(parts))
    return lines

def failure_groups(tree):
    """list of (file, device label, n failing, same_on_main, cause-counter)"""
    S, rows = data(tree)
    res = []
    for f, row in rows:
        for d, nm in mt.DEVS:
            if d not in row: continue
            r = row[d]['res']
            if ' failed' not in r and 'error' not in r: continue
            ids, cs = mt.failing(tree, d, f)
            mids = set(); mfound = False
            mi, _ = mt.failing("prmain2", d, f)
            if mi is not None: mids |= set(mi); mfound = True
            if d == "cpu":
                lg = mt.R/"prmain2-indep"/"cpu"/(f.replace("/", "_")+".log")
                if not mfound and lg.exists():
                    mids |= set(re.findall(r'^(?:FAILED|ERROR) (\S+)', lg.read_text(errors="replace"), re.M)); mfound = True
            same = mfound and all((i in mids) or (mt.norm(i) in mids and d == "cpu") for i in ids)
            grp = collections.Counter(short_cause(c) for c in cs.values())
            if not grp: grp = collections.Counter({"(see log)": len(ids)})
            res.append((f, nm, d, len(ids), same, grp, row[d]['res']))
    return res

def fails_text(tree):
    """Lines describing every failing file/device group with causes; all verified same on main by failure_groups."""
    g = failure_groups(tree)
    by_file = collections.OrderedDict()
    for f, nm, d, n, same, grp, res in g:
        by_file.setdefault(f, []).append((nm, n, same, grp, res))
    lines = []
    for f, items in by_file.items():
        assert all(s for _, _, s, _, _ in items), f
        # group devices with same cause counter
        groups = collections.OrderedDict()
        for nm, n, same, grp, res in items:
            key = tuple(sorted(grp.items()))
            groups.setdefault(key, []).append((nm, n))
        parts = []
        for key, devs in groups.items():
            cause = "; ".join((f"{c} x {k}" if len(key) > 1 else k) for k, c in key)
            devtxt = ", ".join(f"{nm} {n}" for nm, n in devs)
            parts.append(f"{devtxt} failing: {cause}")
        lines.append(f"- {f}: " + " | ".join(parts))
    return lines
