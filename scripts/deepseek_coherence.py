#!/usr/bin/env python3
"""Coherence probe for the DeepSeek-V4-Flash Volta bring-up.

Short greedy generations against an OpenAI-compatible endpoint. The point is
not throughput or benchmark scores but a verdict on whether the ported stack
(QPN8-blk attention, torch Lightning-Indexer, fp16 mHC, reference O-path,
chunked NVFP4 expert emulation) produces the same text as the reference
engine. Run it against vLLM and against llama.cpp with the same prompts and
diff the JSONL.
"""
import argparse
import json
import re
import sys
import time
import urllib.request

# Short, closed-form answers: a broken numeric path shows up as a wrong
# number, a broken attention path as drift or repetition.
PROMPTS = [
    ("capital", "What is the capital of France? Answer in one word."),
    ("arith", "Compute 37 * 43. Reply with just the number."),
    ("count", "How many letters are in the word 'strawberry'? Reply with just the number."),
    ("seq", "Continue the sequence with the next three numbers: 2, 4, 8, 16,"),
    ("recall", "Name the planet closest to the Sun. Answer in one word."),
    ("prose", "In one sentence, explain what a pipeline parallel split does."),
    ("code", "Write a Python one-liner that reverses the string s."),
    ("longctx", "Repeat the following list back in the same order: apple, brick, "
                "candle, dune, ember, forge, granite, harbor."),
]


def chat(url, model, prompt, max_tokens, thinking, timeout):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0, "top_p": 1.0, "seed": 1001,
        "max_completion_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    req = urllib.request.Request(f"{url}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    elapsed = time.time() - t0
    msg = d["choices"][0]["message"]
    usage = d.get("usage", {})
    return {
        "text": msg.get("content") or "",
        "reasoning": msg.get("reasoning_content") or "",
        "completion_tokens": usage.get("completion_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "seconds": round(elapsed, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--only", default=None, help="comma-separated prompt keys")
    args = ap.parse_args()

    keys = args.only.split(",") if args.only else None
    with open(args.out, "w") as fh:
        for key, prompt in PROMPTS:
            if keys and key not in keys:
                continue
            try:
                res = chat(args.url, args.model, prompt, args.max_tokens,
                           args.thinking, args.timeout)
            except Exception as exc:  # noqa: BLE001 - probe reports, never hides
                print(f"{key}: REQUEST FAILED: {exc}", file=sys.stderr)
                return 1
            rec = {"label": args.label, "key": key, "prompt": prompt, **res}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            flat = re.sub(r"\s+", " ", res["text"]).strip()
            rate = (res["completion_tokens"] / res["seconds"]
                    if res["completion_tokens"] and res["seconds"] else 0.0)
            print(f"[{key:8s}] {res['completion_tokens']} tok in "
                  f"{res['seconds']}s ({rate:.2f} tok/s): {flat[:160]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
