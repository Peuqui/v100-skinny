#!/usr/bin/env python3
"""Aggregierter Durchsatz bei 1/2/4 gleichzeitigen Anfragen.
Prueft, wieviel Leerlauf die PP-Stufen bei einer einzelnen Sequenz haben."""
import json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL, LABEL, N = sys.argv[1], sys.argv[2], 160
PROMPTS = ["Explain how a pipelined CPU executes instructions.",
           "Describe the memory hierarchy of a modern computer.",
           "Explain what a compiler optimization pass does.",
           "Describe how virtual memory paging works."]

def one(i):
    body = {"model": "qwen3.8-flash-next", "prompt": PROMPTS[i % len(PROMPTS)],
            "max_tokens": N, "temperature": 0, "ignore_eos": True}
    req = urllib.request.Request(f"{URL}/v1/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["usage"]["completion_tokens"]

one(0)  # aufwaermen
for c in (1, 2, 4):
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=c) as ex:
        toks = sum(ex.map(one, range(c)))
    dt = time.time() - t0
    print(f"[{LABEL}] parallel={c}: {toks} tok in {dt:.1f}s -> gesamt {toks/dt:.1f} tok/s "
          f"(pro Strom {toks/dt/c:.1f})")
