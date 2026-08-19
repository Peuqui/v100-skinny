"""AIME fixtures: throughput AND correctness, per reasoning effort.

The ~30% that `medium` gains over the template's default `xhigh` is only
free if the answers hold up, so this scores the boxed answer against
independently derived ground truth (benchmarks/aime_ground_truth.py) rather
than against the model's own output.

Env: PROBE_MODEL, AIME_EFFORTS (csv), AIME_SEEDS (csv), AIME_FIXTURES (csv),
     AIME_MAX_TOKENS, ARM_TAG
"""
import json
import os
import re
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
MODEL = os.environ["PROBE_MODEL"]
TAG = os.environ.get("ARM_TAG", "aime")
OUTDIR = os.environ.get("AC_OUTDIR", os.path.join(os.getcwd(), "ac_suite"))
FIXDIR = os.environ.get("AIME_FIXDIR",
                        os.path.join(os.path.dirname(__file__),
                                     "ninfer_fixtures"))
MAXTOK = int(os.environ.get("AIME_MAX_TOKENS", "24000"))
EFFORTS = os.environ.get("AIME_EFFORTS", "xhigh,medium").split(",")
SEEDS = [int(s) for s in os.environ.get("AIME_SEEDS", "1001").split(",")]
# ninfer's profile pins presence_penalty 1.0. On a 20k+ token reasoning trace
# that is a termination bias, not a repetition guard: presence_penalty
# penalises tokens that HAVE appeared, and EOS has not appeared, so every
# useful continuation carries -1.0 while EOS carries nothing. Its relative
# logit rises monotonically with trace length. Sweepable to test that.
PENALTIES = [float(x) for x in
             os.environ.get("AIME_PENALTIES", "1.0").split(",")]
FIXTURES = os.environ.get("AIME_FIXTURES", "01,15,30").split(",")

TRUTH = {"01": 277, "15": 83, "30": 393}
BOXED = re.compile(r"\\boxed\{([^}]*)\}")


def met():
    with urllib.request.urlopen(BASE + "/metrics", timeout=10) as r:
        t = r.read().decode()
    a = s = 0.0
    for line in t.splitlines():
        if line.startswith("vllm:spec_decode_num_accepted_tokens_total"):
            a += float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:spec_decode_num_drafts_total"):
            s += float(line.rsplit(" ", 1)[1])
    return a, s


def run(messages, effort, seed, penalty):
    body = {"model": MODEL, "messages": messages,
            "max_completion_tokens": MAXTOK,
            "chat_template_kwargs": {"enable_thinking": True,
                                     "reasoning_effort": effort},
            "temperature": 0.6, "top_p": 0.95, "top_k": 20,
            "presence_penalty": penalty, "seed": seed,
            "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    tf = tl = None
    gen = 0
    finish = None
    usage = {}
    content, reasoning = [], []
    with urllib.request.urlopen(req, timeout=7200) as r:
        for raw in r:
            if not raw.startswith(b"data:"):
                continue
            p = raw[5:].strip()
            t = time.perf_counter()
            if p == b"[DONE]":
                break
            d = json.loads(p)
            if d.get("usage"):
                usage = d["usage"]
                gen = usage["completion_tokens"]
                continue
            ch = d.get("choices")
            if not ch:
                continue
            if ch[0].get("finish_reason"):
                finish = ch[0]["finish_reason"]
            c, rs = ch[0]["delta"].get("content"), ch[0]["delta"].get("reasoning")
            if c or rs:
                if tf is None:
                    tf = t
                tl = t
                if c:
                    content.append(c)
                if rs:
                    reasoning.append(rs)
    return (gen, tf, tl, "".join(content), "".join(reasoning), finish, usage)


def answer_of(content, reasoning):
    """Last boxed value in the answer, falling back to the reasoning."""
    for text in (content, reasoning):
        hits = BOXED.findall(text or "")
        if hits:
            raw = hits[-1].strip().replace(",", "").replace("$", "")
            m = re.search(r"-?\d+", raw)
            if m:
                return int(m.group(0)), raw
    return None, None


os.makedirs(OUTDIR, exist_ok=True)
rows = []
print(f"{'fix':>5}{'effort':>8}{'seed':>6}{'tok/s':>8}{'tau':>6}{'ms/rnd':>8}"
      f"{'gen':>7}{'answer':>8}{'truth':>7}{'':>3}")
for fx in FIXTURES:
    with open(f"{FIXDIR}/long_decode_aime26_{fx}.json") as fh:
        messages = json.load(fh)
    for effort in EFFORTS:
      for penalty in PENALTIES:
        for seed in SEEDS:
            a0, s0 = met()
            (gen, tf, tl, content, reasoning, finish,
             usage) = run(messages, effort, seed, penalty)
            a1, s1 = met()
            steps = s1 - s0
            tau = (a1 - a0) / steps if steps else 0.0
            span = (tl - tf) if (tf and tl) else 0.0
            toks = (gen - (tau + 1)) / span if span > 0 else 0.0
            dms = span / (steps - 1) * 1000 if steps > 1 else 0.0
            ans, raw = answer_of(content, reasoning)
            truth = TRUTH[fx]
            ok = (ans == truth)
            capped = gen >= MAXTOK - 2
            mark = "OK" if ok else ("CAP" if capped and ans is None else
                                    "--" if ans is None else "X")
            rows.append({"fixture": fx, "effort": effort,
                         "penalty": penalty, "seed": seed,
                         "tok_s": round(toks, 1), "tau": round(tau, 2),
                         "ms_round": round(dms, 2), "gen": gen,
                         "answer": ans, "raw_boxed": raw, "truth": truth,
                         "correct": ok, "capped": capped,
                         "finish_reason": finish, "usage": usage,
                         "n_content": len(content or ""),
                         "n_reasoning": len(reasoning or ""),
                         "content": content, "reasoning": reasoning})
            print(f"{fx:>5}{effort:>8}{seed:>6}{toks:>8.1f}{tau:>6.2f}"
                  f"{dms:>8.2f}{gen:>7}{str(ans):>8}{truth:>7}{mark:>3}"
                  f"  pen={penalty} finish={finish} "
                  f"closed_think={'</think>' in (content or '') or bool(reasoning)} "
                  f"chars={len(content or '')}", flush=True)

print("\n--- summary by effort ---")
for effort in EFFORTS:
    r = [x for x in rows if x["effort"] == effort]
    if not r:
        continue
    n_ok = sum(1 for x in r if x["correct"])
    mt = sum(x["tok_s"] for x in r) / len(r)
    print(f"  {effort:>7}: correct {n_ok}/{len(r)}   mean {mt:.1f} tok/s   "
          f"mean gen {sum(x['gen'] for x in r) / len(r):.0f} tok")

with open(f"{OUTDIR}/{TAG}.json", "w") as fh:
    json.dump(rows, fh, indent=1)
print(f"wrote {OUTDIR}/{TAG}.json")
