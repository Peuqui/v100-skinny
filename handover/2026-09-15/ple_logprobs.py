#!/usr/bin/env python3
"""Top-2-Logprobs je Token fuer den Prompt, dessen Hash abwich.

Aufruf: ple_logprobs.py URL MODEL OUT.json
Eigene Anfrage mit logprobs=2 (greedy, 260 Token). Vergleich nur zwischen
Laeufen dieser Sonde, nicht gegen ple_probe.py.
"""
import json
import sys
import urllib.request

sys.path.insert(0, __file__.rsplit("/", 1)[0])
URL, MODEL, OUT = sys.argv[1:4]
PROMPT = "Write a Python function that merges overlapping intervals, then explain it."
body = {"model": MODEL, "prompt": PROMPT, "max_tokens": 260, "temperature": 0,
        "ignore_eos": True, "logprobs": 2}
req = urllib.request.Request(f"{URL}/v1/completions", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=900) as r:
    choice = json.load(r)["choices"][0]
lp = choice["logprobs"]
json.dump({"text": choice["text"], "tokens": lp["tokens"],
           "token_logprobs": lp["token_logprobs"], "top_logprobs": lp["top_logprobs"]},
          open(OUT, "w"), indent=1, ensure_ascii=False)
print(f"{len(lp['tokens'])} Token gespeichert in {OUT}")
