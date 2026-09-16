#!/usr/bin/env python3
"""PLE-Kaskade: Prefill-Zeit gegen einen laufenden Server.

Aufruf: ple_prefill.py URL MODEL LABEL [OUT.json]
Jeder Lauf bekommt einen eigenen Zufallstext, damit der Prefix-Cache des
Eintrags (--enable-prefix-caching) nicht mitmisst; die Saat haengt nur an
Laenge und Laufnummer, also sind die Prompts zwischen zwei Servern gleich.
max_tokens=1, gemessen wird die Wanduhr der Anfrage.
"""
import json
import random
import sys
import time
import urllib.request

URL, MODEL, LABEL = sys.argv[1], sys.argv[2], sys.argv[3]
OUT = sys.argv[4] if len(sys.argv) > 4 else None
# Echter Fliesstext statt Zufallswoertern: ein MoE-Modell routet einen Text aus
# wenigen Woertern an wenige Experten und misst dann viel zu schnell (16.09.2026:
# 1.100 statt der im Alltag ueblichen 550-580 tok/s).
CORPUS_GLOBS = (
    "/home/mp/Projekte/vllm-research/v100-skinny/**/*.md",
    "/home/mp/Projekte/AIfred-Intelligence/docs/**/*.md",
    "/home/mp/Projekte/AIfred-Intelligence/*.md",
)
LENGTHS = [9000, 28000]  # Woerter; die Token-Zahl steht im Ergebnis
REPEATS = 2


def corpus_words() -> list[str]:
    import glob
    text = []
    paths = sorted({f for pattern in CORPUS_GLOBS for f in glob.glob(pattern, recursive=True)})
    for path in paths:
        text.append(open(path, encoding="utf-8", errors="ignore").read())
    words = " ".join(text).split()
    if len(words) < 60000:
        raise SystemExit(f"Korpus zu klein: {len(words)} Woerter")
    return words


WORDS = corpus_words()


def prompt_of(words: int, seed: int) -> str:
    # Je Lauf ein anderer Abschnitt, damit der Prefix-Cache nicht mitmisst.
    start = (seed * 4099) % (len(WORDS) - words - 1)
    return " ".join(WORDS[start : start + words])


def prefill(prompt: str) -> tuple[int, float]:
    body = {"model": MODEL, "prompt": prompt, "max_tokens": 1, "temperature": 0}
    req = urllib.request.Request(
        f"{URL}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.time()
    with urllib.request.urlopen(req, timeout=1800) as response:
        payload = json.load(response)
    return payload["usage"]["prompt_tokens"], time.time() - started


prefill(prompt_of(200, 999))  # Aufwaermen, verworfen
runs = []
for words in LENGTHS:
    for repeat in range(REPEATS):
        tokens, seconds = prefill(prompt_of(words, words * 1000 + repeat))
        runs.append(
            {
                "words": words,
                "repeat": repeat,
                "prompt_tokens": tokens,
                "seconds": round(seconds, 3),
                "tok_s": round(tokens / seconds, 1),
            }
        )
        print(f"[{LABEL}] {tokens} Prompt-Token in {seconds:.2f}s = {tokens / seconds:.0f} tok/s")
if OUT:
    json.dump(
        {"label": LABEL, "model": MODEL, "url": URL, "runs": runs,
         "time": time.strftime("%Y-%m-%d %H:%M:%S")},
        open(OUT, "w"), indent=1,
    )
