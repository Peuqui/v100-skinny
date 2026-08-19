"""The v1.1 launch table: one row per domain, at k=7 and k=15.

Changes from the v1.0 table:

  * `extract` (a verbatim-copy prompt, "repeat this passage exactly") is
    REPLACED by `doc2json` and `log2json` -- real extraction workloads.
    The synthetic cell measured a ceiling nobody serves; doc2json is invoice
    parsing and log2json is log ingestion, both of which people actually run,
    and both land within 6% of the synthetic number anyway.
  * `mergesort` added: a short, highly conventional code-generation task, to
    sit between the free-composition cells (prose) and the templated ones.
  * every row carries tau and ms/round, because tok/s alone hides which of
    the two moved.

Env: PROBE_MODEL, ARM_TAG, AC_OUTDIR.
"""
import json
import os
import time
import urllib.request

# Engine-agnostic: point at vLLM or ninfer-serve (both OpenAI-compatible).
BASE = os.environ.get("PROBE_BASE", "http://127.0.0.1:8000")
# tau/acceptance come from vLLM /metrics; ninfer has no equivalent.
HAVE_COUNTERS = os.environ.get("PROBE_COUNTERS", "1") == "1"
# Request dialect. vLLM takes chat_template_kwargs per request; ninfer
# rejects it ("chat_template_option_not_supported") and controls thinking
# server-side via --no-thinking. So a mixed-mode suite needs two ninfer boots,
# and PROBE_THINKING states which mode the server is currently in so we only
# run the cells that match it.
DIALECT = os.environ.get("PROBE_DIALECT", "vllm")
ONLY_THINKING = os.environ.get("PROBE_THINKING")  # "on" | "off" | None = all
# Subset of CELLS to run, comma-separated, in CELLS order. Used by sweeps
# that boot the server once per arm and only need a couple of context
# lengths (e.g. the MML ladder), where running all seven cells per arm
# would triple the wall time without adding a data point.
ONLY_CELLS = [c for c in os.environ.get("PROBE_CELLS", "").split(",") if c]
# vLLM disables top-k with -1; ninfer uses 0.
TOPK_OFF = 0 if DIALECT == "ninfer" else -1
# Qwen3.8's template defaults reasoning_effort to "xhigh", which injects a
# "think carefully..." system message. ninfer rejects chat_template_kwargs and
# injects no such instruction, so "medium" (which emits no system message at
# all) is the arm that matches their prompt. Set explicitly, never defaulted.
EFFORT = os.environ.get("PROBE_EFFORT")
# ninfer has no /metrics, but --request-log-jsonl emits a `request_done`
# record per request carrying speculative.{accepted_tokens,drafted_tokens,
# rounds,accepted_per_position} and the EFFECTIVE sampling it applied. That
# gives the same tau/acceptance we read from vLLM's counters, so both engines
# can be reported on identical fields instead of ours-only.
NINFER_LOG = os.environ.get("PROBE_NINFER_LOG")


def ninfer_last_done():
    """(accepted, drafted, rounds, sampling, finish) from the newest record."""
    if not NINFER_LOG:
        return None
    try:
        recs = [json.loads(l) for l in open(NINFER_LOG)]
    except Exception:
        return None
    done = [r for r in recs if r.get("event") == "request_done"]
    if not done:
        return None
    r = done[-1]
    sp = r.get("speculative", {})
    return (sp.get("accepted_tokens", 0), sp.get("drafted_tokens", 0),
            sp.get("rounds", 0), r.get("request", {}).get("sampling", {}),
            r.get("result", {}).get("finish_reason"))
MODEL = os.environ["PROBE_MODEL"]
TAG = os.environ.get("ARM_TAG", "v11")
OUTDIR = os.environ.get("AC_OUTDIR", os.path.join(os.getcwd(), "ac_suite"))

INVOICE = """INVOICE #INV-2026-04417
Issued: 2026-03-14   Due: 2026-04-13
Bill To: Halvorsen Marine Supply, 44 Dock Road, Aberdeen AB11 5DQ
Vendor: Northlight Instruments Ltd, VAT GB418826314

LINE ITEMS
1  Fresnel lens assembly, 3rd order      qty 2   unit 1875.00   total 3750.00
2  Brass gimbal mount BM-40              qty 6   unit  212.50   total 1275.00
3  Marine cable, 12mm, per metre         qty 240 unit    4.35   total 1044.00
4  Sealed bearing kit SB-77              qty 12  unit   88.20   total 1058.40
5  Calibration service, on-site          qty 1   unit  640.00   total  640.00

Subtotal 7767.40   VAT 20% 1553.48   Shipping 145.00   TOTAL DUE 9465.88
Payment terms: net 30. Late fee 2% monthly. PO reference PO-88213."""

LOGS = "\n".join(
    f"2026-03-{14 + i % 3:02d} 0{i % 8}:{(i * 7) % 60:02d}:11 "
    f"{'ERROR' if i % 4 == 0 else 'WARN' if i % 3 == 0 else 'INFO'} "
    f"svc-{'auth' if i % 2 else 'billing'} "
    f"req={4400 + i * 13} lat={12 + (i * 17) % 380}ms "
    f"status={500 if i % 4 == 0 else 200} path=/v1/"
    f"{'token' if i % 2 else 'charge'}" for i in range(28))

# (name, prompt, thinking, max_tokens)
CELLS = [
    ("prose", "Write a long, detailed short story about a lighthouse keeper "
     "who discovers something unusual washed ashore. Keep going.",
     False, 1024),
    ("math", "A train leaves station A at 60 km/h. Two hours later a second "
     "train leaves the same station at 90 km/h on a parallel track. How far "
     "from the station does the second train catch the first? Work through "
     "this carefully.", True, 2048),
    ("mergesort", "Write a complete Python implementation of merge sort: a "
     "recursive top-down version, an iterative bottom-up version, and a "
     "merge helper. Include docstrings and a note on stability and space "
     "complexity. Output only the code.", False, 1024),
    ("code", "Write a complete Python implementation of a red-black tree "
     "with insert, delete, and search, with docstrings.", False, 1024),
    # csv RETIRED (2026-08-19). The prompt asked for 50 rows of "realistic
    # hardware-store products": templated scaffolding wrapped around 50
    # invented product names, i.e. high-entropy content the drafter cannot
    # anticipate. It never measured the easily-speculated structured domain it
    # was meant to -- `json` covers templated generation and doc2json/log2json
    # cover extraction. It is also the cell where the all-FP4 arm scored tau
    # 6.03 by collapsing into 4 repeated product names, i.e. where higher
    # acceptance meant a worse answer. Do not re-add without a prompt whose
    # content is determined by the input.
    ("json", "Generate a JSON array of 40 user records with fields id "
     "(sequential integer), name, email, signup_date (ISO 8601), and plan "
     "(one of free/pro/enterprise). Output only the JSON.", False, 1024),
    ("doc2json", "Extract this invoice into JSON with fields invoice_number, "
     "issued, due, bill_to, vendor, vat_number, line_items (each with "
     "description, qty, unit_price, line_total), subtotal, vat, shipping, "
     "total_due, po_reference. Output only the JSON.\n\n" + INVOICE,
     False, 1024),
    ("log2json", "Convert these log lines into a JSON array of records with "
     "fields timestamp, level, service, request_id, latency_ms, status, "
     "path. Output only the JSON.\n\n" + LOGS, False, 1536),
]


def met():
    if not HAVE_COUNTERS:
        return 0.0, 0.0
    with urllib.request.urlopen(BASE + "/metrics", timeout=10) as r:
        t = r.read().decode()
    a = s = 0.0
    for line in t.splitlines():
        if line.startswith("vllm:spec_decode_num_accepted_tokens_total"):
            a += float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:spec_decode_num_drafts_total"):
            s += float(line.rsplit(" ", 1)[1])
    return a, s


def stream(prompt, thinking, mt):
    # Wall-clock starts BEFORE the request is sent: ttft then covers
    # prefill, which decode-only rates exclude and which is not ours.
    t0 = time.perf_counter()
    # EVERY sampling field is sent explicitly. Neither engine may be left to
    # its own defaults: ninfer fills unset fields from the artifact's model
    # card (measured: presence_penalty 1.5, top_p 0.8, top_k 20) while vLLM
    # defaults presence to 0 and does plain argmax at temperature 0. Sending
    # only `temperature` produces a silently unmatched comparison.
    payload = {
        "model": MODEL, "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": mt,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": TOPK_OFF,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "seed": 1001,
        "stream": True, "stream_options": {"include_usage": True},
    }
    if DIALECT == "vllm":
        ctk = {"enable_thinking": thinking}
        if thinking and EFFORT:
            ctk["reasoning_effort"] = EFFORT
        payload["chat_template_kwargs"] = ctk
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    tf = tl = None
    gen = 0
    parts = []
    with urllib.request.urlopen(req, timeout=1800) as r:
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
            dl = ch[0]["delta"] if ch else {}
            if dl.get("content") or dl.get("reasoning") or dl.get("reasoning_content"):
                if tf is None:
                    tf = t
                tl = t
                if dl.get("content"):
                    parts.append(dl["content"])
    return gen, tf, tl, "".join(parts), t0


os.makedirs(OUTDIR, exist_ok=True)
out = {"tag": TAG, "cells": {}}
# Two decode-rate conventions are reported side by side, ALWAYS labelled.
# `ours` subtracts a whole speculative round (gen - (tau+1)); ninfer's
# subtracts a single token (gen - 1). On short cells the gap is ~1%, but it
# is the difference between two published tables, and the JSON's primary
# `tok_s` field carries OURS. Printing one unlabelled column invites a
# silent comparison of a live console reading against a results file
# written in the other convention.
print(f"{'cell':>10}{'t/s ours':>10}{'t/s ninf':>10}{'tau':>7}{'tok/rnd':>8}"
      f"{'accept':>7}{'ms/rnd':>8}{'gen':>7}")
print(f"{'':>10}{'(gen-(t+1))':>10}{'(gen-1)':>10}"
      f"   <- results/ and the launch table use 'ours'")
for name, prompt, thinking, mt in CELLS:
    if ONLY_CELLS and name not in ONLY_CELLS:
        continue
    if ONLY_THINKING == "on" and not thinking:
        continue
    if ONLY_THINKING == "off" and thinking:
        continue
    a0, s0 = met()
    gen, tf, tl, text, t0 = stream(prompt, thinking, mt)
    a1, s1 = met()
    steps = s1 - s0
    tau = (a1 - a0) / steps if steps else 0.0
    accept = float("nan")
    nl = ninfer_last_done()
    if nl:
        acc_n, drafted_n, rounds_n, samp_n, fin_n = nl
        tau = acc_n / rounds_n if rounds_n else 0.0
        accept = acc_n / drafted_n if drafted_n else float("nan")
        steps = rounds_n
    span = (tl - tf) if (tf and tl) else 0.0
    # Decode span excludes prefill. Bank ttft and wall so a cross-engine
    # seconds-to-answer can be quoted without reconstructing it later.
    ttft_ms = (tf - t0) * 1000.0 if tf else 0.0
    wall_s = (tl - t0) if tl else 0.0
    toks = (gen - (tau + 1)) / span if span > 0 else 0.0
    toks_ninfer = (gen - 1) / span if span > 0 else 0.0
    dms = span / (steps - 1) * 1000 if steps > 1 else 0.0
    pred = (tau + 1) * 1000.0 / dms if dms > 0 else 0.0
    err = abs(toks - pred) / toks * 100 if toks else 0.0
    out["cells"][name] = {"tok_s": round(toks, 1),
                          "tok_s_ninfer_metric": round(toks_ninfer, 1),
                          "tau": round(tau, 2),
                          "acceptance": (None if accept != accept
                                         else round(accept, 4)),
                          "ms_round": round(dms, 2), "gen": gen,
                          "ttft_ms": round(ttft_ms, 1),
                          "wall_s": round(wall_s, 3),
                          "decode_seconds": round(span, 3),
                          "check_err_pct": round(err, 1), "text": text}
    acc_s = "  n/a " if accept != accept else f"{accept * 100:5.1f}%"
    print(f"{name:>10}{toks:>10.1f}{toks_ninfer:>10.1f}{tau:>7.2f}"
          f"{1 + tau:>8.2f}{acc_s:>7}{dms:>8.2f}{gen:>7}", flush=True)

with open(f"{OUTDIR}/{TAG}.json", "w") as fh:
    json.dump(out, fh, indent=1)
print(f"wrote {OUTDIR}/{TAG}.json")
