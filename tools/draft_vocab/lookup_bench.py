"""Spec-decode bench incl. a copy-heavy edit (plan Oct 2026, item 4), derived from dsv4_bench.py.

Bench at AIfred's production sampling. Usage: dsv4_bench.py <label> [flags]

Flags: --sentences=N (filler length, default 905), --lang (a long German filler prompt, ~18k tokens on DeepSeek-V4's
tokenizer; the run prints the real count, do not trust the label), --once (one run per
prompt instead of two), --from=N (seed of the filler text, so a fresh N gives a
prompt the prefix cache has never seen). The model comes from $MODEL.
Reports ttft, decode tok/s, ms per engine step and per-position draft
acceptance from the vLLM /metrics deltas.

Lives outside /tmp on purpose: a reboot wipes /tmp and took the earlier copy.
"""
import json
import os
import random
import re
import sys
import time
import urllib.request

LABEL = sys.argv[1]
FLAGS = sys.argv[2:]
ONCE = "--once" in FLAGS
LANG = "--lang" in FLAGS
SEED = next((int(f.split("=")[1]) for f in FLAGS if f.startswith("--from=")), 0)
SENTENCES = next((int(f.split("=")[1]) for f in FLAGS if f.startswith("--sentences=")), 905)
MODEL = os.environ.get("MODEL", "DeepSeek-V4-Flash-284B-A13B-MXFP4-FP8-DSpark-vllm")
SWAP = "http://127.0.0.1:11435"

WORDS = (
    "Der Die Das Fluss Berg Wald Stadt Haus Garten Wind Regen Sonne leise "
    "schnell alt neu wandert trägt findet baut über unter neben".split()
)


def filler(seed, sentences=905):
    # 905 sentences is ~18k tokens for DeepSeek-V4's tokenizer (measured
    # 17,785-18,064 over five seeds). Other tokenizers land elsewhere, so read
    # the prompt token count the run prints instead of trusting this number.
    rnd = random.Random(seed)
    text = " ".join(
        " ".join(rnd.choice(WORDS) for _ in range(14)) + "." for _ in range(sentences)
    )
    return text + "\n\nFasse den Text in drei Sätzen zusammen."


EDIT_FILE = "/home/mp/Projekte/AI-Connect/client/tools.py"
PROMPTS = {
    "prosa": "Erkläre in etwa 300 Wörtern auf Deutsch, wie ein Regenbogen entsteht.",
    "code": (
        "Schreibe eine Python-Klasse LRUCache mit get und put in O(1), "
        "mit Typannotationen und kurzem Docstring. Nur Code."
    ),
    "edit": (
        "Benenne in der folgenden Python-Datei die Funktion `_format_time` in "
        "`_format_clock` um, einschließlich aller Aufrufe, und gib die vollständige "
        "Datei ohne Erklärung aus.\n\n```python\n" + open(EDIT_FILE).read() + "```"
    ),
}
MAX_TOKENS = {"prosa": 500, "code": 500, "edit": 2500}


def port():
    running = json.load(urllib.request.urlopen(f"{SWAP}/running", timeout=10))["running"]
    return re.search(r"--port (\d+)", running[0]["cmd"]).group(1)


def metrics(p):
    text = urllib.request.urlopen(f"http://127.0.0.1:{p}/metrics", timeout=10).read().decode()
    out = {}
    for line in text.splitlines():
        m = re.match(r'vllm:spec_decode_num_(drafts|draft_tokens|accepted_tokens)_total\{[^}]*\} ([0-9.e+]+)', line)
        if m:
            out[m.group(1)] = float(m.group(2))
        m = re.match(r'vllm:spec_decode_num_accepted_tokens_per_pos_total\{[^}]*position="(\d+)"[^}]*\} ([0-9.e+]+)', line)
        if m:
            out[f"pos{m.group(1)}"] = float(m.group(2))
    return out


def run(name, prompt, p):
    body = {
        "model": MODEL, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS[name], "temperature": 1.0, "top_k": 40, "top_p": 1.0,
        "stream": True, "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    before = metrics(p)
    prompt_tokens = 0
    req = urllib.request.Request(
        f"{SWAP}/v1/chat/completions", json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    t0 = time.time()
    first = None
    n = 0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                n = d["usage"]["completion_tokens"]
                prompt_tokens = d["usage"]["prompt_tokens"]
            if d["choices"] and d["choices"][0]["delta"].get("content") and first is None:
                first = time.time()
    end = time.time()
    after = metrics(p)
    dl = {k: after.get(k, 0) - before.get(k, 0) for k in after}
    drafts = dl.get("drafts", 0) or 1
    pos = [round(100 * dl.get(f"pos{i}", 0) / drafts) for i in range(8) if f"pos{i}" in dl]
    step_ms = 1000 * (end - first) / drafts if first else 0
    print(
        f"{LABEL:>14} {name:>11}: {prompt_tokens:6d} prompt tok, "
        f"ttft {first - t0:5.1f}s, {n:4d} tok, "
        f"{n / (end - first):5.1f} tok/s decode, Schritt {step_ms:5.0f} ms, "
        f"Annahme/Runde {1 + dl.get('accepted_tokens', 0) / drafts:4.2f}, "
        f"je Position % {pos}",
        flush=True,
    )


p = port()
for name, prompt in PROMPTS.items():
    for _ in range(1 if ONCE else 2):
        run(name, prompt, p)
