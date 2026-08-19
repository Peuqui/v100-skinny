"""Reproduce ninfer's single-request long-reasoning benchmark, their way.

Their published method (ninfer/docs/performance.md), implemented verbatim:

  decode_tok_s          = (completion_tokens - 1) / decode_seconds
  spec_acceptance       = accepted_tokens / drafted_tokens
  spec_tokens_per_round = 1 + accepted_tokens / speculative_rounds
  sampling              = temp 0.6, top-p 0.95, top-k 20, presence penalty 1.0
  output limit          = 65,536 tokens
  thinking              = enabled
  reasoning_effort      = NOT SET (their docs never configure it)
  samples               = five fixed seeds; mean +/- sample standard deviation

Differences from our own harness, which is why this file exists rather than a
flag on ac_suite: we subtract a whole round of tokens (gen - (tau+1)) where
they subtract one, and we report tau where they report acceptance = tau/k.
On short fixtures that is <1%; it is not nothing on the numbers we publish.

`reasoning_effort` is deliberately left unset here. Their docs do not set it
and do not say which chat template their converter embeds; our template
defaults to xhigh. Set NINFER_EFFORT to override for the medium-equivalent
comparison.
"""
import json
import os
import re
import statistics
import time
import urllib.request

# Engine-agnostic: ninfer-serve is OpenAI-compatible (/v1/chat/completions
# with streaming + usage, /v1/models), so the identical harness drives both
# engines and the comparison no longer depends on reimplementing their metric.
BASE = os.environ.get("PROBE_BASE", "http://127.0.0.1:8000")
# vLLM exposes spec-decode counters at /metrics; ninfer does not. Acceptance
# and tokens/round are therefore vLLM-only -- decode tok/s, completion length
# and correctness are the portable, cross-engine metrics.
HAVE_COUNTERS = os.environ.get("PROBE_COUNTERS", "1") == "1"
MODEL = os.environ["PROBE_MODEL"]
TAG = os.environ.get("ARM_TAG", "ninfer_repro")
OUTDIR = os.environ.get("AC_OUTDIR", os.path.join(os.getcwd(), "ac_suite"))
FIXDIR = os.environ.get("AIME_FIXDIR",
                        os.path.join(os.path.dirname(__file__),
                                     "ninfer_fixtures"))
FIXTURES = os.environ.get("NINFER_FIXTURES", "01,15,30").split(",")
SEEDS = [int(s) for s in os.environ.get("NINFER_SEEDS",
                                        "1001,2002,3003,4004,5005").split(",")]
MAXTOK = int(os.environ.get("NINFER_MAXTOK", "65536"))
EFFORT = os.environ.get("NINFER_EFFORT", "")          # "" = leave unset
# ninfer rejects chat_template_kwargs outright and controls thinking with the
# server-side --no-thinking flag, so the request body differs by engine.
DIALECT = os.environ.get("PROBE_DIALECT", "vllm")
NINFER_LOG = os.environ.get("PROBE_NINFER_LOG")


def ninfer_last_done():
    """Speculative counters from ninfer's --request-log-jsonl."""
    if not NINFER_LOG:
        return None
    try:
        recs = [json.loads(l) for l in open(NINFER_LOG)]
    except Exception:
        return None
    done = [r for r in recs if r.get("event") == "request_done"]
    if not done:
        return None
    sp = done[-1].get("speculative", {})
    return (sp.get("accepted_tokens", 0), sp.get("drafted_tokens", 0),
            sp.get("rounds", 0))
PENALTY = float(os.environ.get("NINFER_PENALTY", "1.0"))
TRUTH = {"01": 277, "15": 83, "30": 393}
BOXED = re.compile(r"\\boxed\{([^}]*)\}")


def counters():
    """accepted, drafted, rounds -- the three server counters they cite.

    Returns zeros when the engine exposes no such endpoint; the caller then
    reports acceptance and tokens/round as not-available rather than wrong."""
    if not HAVE_COUNTERS:
        return 0.0, 0.0, 0.0
    with urllib.request.urlopen(BASE + "/metrics", timeout=10) as r:
        t = r.read().decode()
    a = d = s = 0.0
    for line in t.splitlines():
        if line.startswith("vllm:spec_decode_num_accepted_tokens_total"):
            a += float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:spec_decode_num_draft_tokens_total"):
            d += float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:spec_decode_num_drafts_total"):
            s += float(line.rsplit(" ", 1)[1])
    return a, d, s


def run(messages, seed):
    # Wall-clock starts BEFORE the request is sent: ttft then covers
    # prefill, which decode-only rates exclude and which is not ours.
    t0 = time.perf_counter()
    body = {"model": MODEL, "messages": messages,
            "max_completion_tokens": MAXTOK,
            "temperature": 0.6, "top_p": 0.95, "top_k": 20,
            "presence_penalty": PENALTY, "frequency_penalty": 0.0,
            "seed": seed,
            "stream": True, "stream_options": {"include_usage": True}}
    # ninfer rejects chat_template_kwargs and takes thinking from the
    # server-side --no-thinking flag instead.
    if DIALECT == "vllm":
        ctk = {"enable_thinking": True}
        if EFFORT:
            ctk["reasoning_effort"] = EFFORT
        body["chat_template_kwargs"] = ctk
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    tf = tl = None
    gen = 0
    finish = None
    text = []
    with urllib.request.urlopen(req, timeout=14400) as r:
        for raw in r:
            if not raw.startswith(b"data:"):
                continue
            p = raw[5:].strip()
            t = time.perf_counter()
            if p == b"[DONE]":
                break
            d = json.loads(p)
            if d.get("usage"):
                gen = d["usage"]["completion_tokens"]
                continue
            ch = d.get("choices")
            if not ch:
                continue
            if ch[0].get("finish_reason"):
                finish = ch[0]["finish_reason"]
            # vLLM streams thinking as "reasoning"; ninfer uses
            # "reasoning_content". Missing the latter would put t_first after
            # the whole reasoning phase and inflate their tok/s.
            dl = ch[0]["delta"]
            piece = (dl.get("content") or dl.get("reasoning")
                     or dl.get("reasoning_content"))
            if piece:
                if tf is None:
                    tf = t
                tl = t
                text.append(piece)
    return gen, tf, tl, finish, "".join(text), t0


def answer_of(text):
    hits = BOXED.findall(text or "")
    if not hits:
        return None
    m = re.search(r"-?\d+", hits[-1].replace(",", "").replace("$", ""))
    return int(m.group(0)) if m else None


os.makedirs(OUTDIR, exist_ok=True)
print(f"ninfer-method repro | effort={EFFORT or 'UNSET (template default)'} "
      f"penalty={PENALTY} maxtok={MAXTOK} seeds={SEEDS}")
print(f"{'fixture':>22}{'seed':>7}{'decode tok/s':>14}{'accept%':>9}"
      f"{'tok/round':>11}{'gen':>8}{'answer':>8}{'finish':>8}")
rows = []
for fx in FIXTURES:
    with open(f"{FIXDIR}/long_decode_aime26_{fx}.json") as fh:
        messages = json.load(fh)
    for seed in SEEDS:
        a0, d0, s0 = counters()
        gen, tf, tl, finish, text, t0 = run(messages, seed)
        a1, d1, s1 = counters()
        acc, drafted, rounds = a1 - a0, d1 - d0, s1 - s0
        nl = ninfer_last_done()
        if nl:
            acc, drafted, rounds = nl
        dec_s = (tl - tf) if (tf and tl) else 0.0
        # ttft covers prefill; wall is what a user actually waits. Decode-only
        # rates exclude prefill, which is NOT ours (~4x theirs), so a
        # cross-engine seconds-to-answer must quote wall, not dec_s.
        ttft_ms = (tf - t0) * 1000.0 if tf else 0.0
        wall_s = (tl - t0) if tl else 0.0
        # their definitions, verbatim
        tok_s = (gen - 1) / dec_s if dec_s > 0 else 0.0
        acceptance = acc / drafted if drafted else float("nan")
        tpr = 1 + acc / rounds if rounds else float("nan")
        ans = answer_of(text)
        rows.append({"fixture": fx, "seed": seed, "decode_tok_s": tok_s,
                     "acceptance": acceptance, "tokens_per_round": tpr,
                     "completion_tokens": gen, "answer": ans,
                     "truth": TRUTH[fx], "correct": ans == TRUTH[fx],
                     "finish_reason": finish, "decode_seconds": dec_s,
                     "ttft_ms": ttft_ms, "wall_s": wall_s})
        print(f"{'long_decode_aime26_' + fx:>22}{seed:>7}{tok_s:>14.1f}"
              f"{acceptance * 100:>8.1f}%{tpr:>11.2f}{gen:>8}"
              f"{str(ans):>8}{str(finish):>8}", flush=True)

print(f"\n{'fixture':>22}{'completion tokens':>22}{'decode tok/s':>18}"
      f"{'acceptance':>16}{'tok/round':>14}")
summary = {}
for fx in FIXTURES:
    r = [x for x in rows if x["fixture"] == fx]
    if not r:
        continue
    def ms(key, scale=1.0):
        v = [x[key] * scale for x in r]
        sd = statistics.stdev(v) if len(v) > 1 else 0.0
        return statistics.mean(v), sd
    ct, ctsd = ms("completion_tokens")
    ts, tssd = ms("decode_tok_s")
    ac, acsd = ms("acceptance", 100.0)
    tp, tpsd = ms("tokens_per_round")
    summary[fx] = {"completion": [ct, ctsd], "tok_s": [ts, tssd],
                   "acceptance": [ac, acsd], "tokens_per_round": [tp, tpsd],
                   "correct": sum(1 for x in r if x["correct"]), "n": len(r)}
    print(f"{'long_decode_aime26_' + fx:>22}"
          f"{ct:>13,.1f} ± {ctsd:<6,.1f}{ts:>11.1f} ± {tssd:<4.1f}"
          f"{ac:>10.1f}% ± {acsd:<3.1f}{tp:>9.2f} ± {tpsd:<4.2f}"
          f"   correct {summary[fx]['correct']}/{summary[fx]['n']}")

with open(f"{OUTDIR}/{TAG}.json", "w") as fh:
    json.dump({"rows": rows, "summary": summary, "effort": EFFORT,
               "penalty": PENALTY, "maxtok": MAXTOK}, fh, indent=1)
print(f"wrote {OUTDIR}/{TAG}.json")
