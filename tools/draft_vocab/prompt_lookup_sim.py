#!/usr/bin/env python3
"""Offline estimate: how much would prompt-lookup drafting help on real replies?

For each reply, simulates greedy prompt-lookup speculative decoding against the
text that precedes it (earlier turns + the user prompt) plus the reply so far:
at every step the last n tokens (n = 4, 3, 2, longest match first) are searched
in that history, the K tokens after the most recent match are proposed, and the
leading tokens that equal the real continuation are accepted; the target model
then adds one token of its own. Reports reply tokens per verification step
(1.0 = no gain) and how many reply tokens fall into accepted runs of >= 4,
which is where a lookup draft would beat an MTP head that usually stops after
2-3 tokens.

Usage: prompt_lookup_sim.py
"""
import glob
import json

from tokenizers import Tokenizer

TOKENIZER = glob.glob("/home/mp/.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4/snapshots/*/tokenizer.json")[0]
SESSIONS = "/home/mp/Projekte/AIfred-Intelligence/data/sessions/*.json"
QUALITY_PROMPTS = "/home/mp/.cache/bench-scripts/quality_prompts.json"
QUALITY_ANSWERS = "/home/mp/.cache/bench-scripts/quality_2026-09-27/produnion-fntp2-ds.json"
K = 8
NGRAMS = (4, 3, 2)

tok = Tokenizer.from_file(TOKENIZER)


def encode(text: str) -> list[int]:
    return tok.encode(text, add_special_tokens=False).ids


def simulate(context: list[int], reply: list[int]) -> tuple[int, int]:
    """Return (verification steps, reply tokens inside accepted runs of >= 4)."""
    history = list(context)
    i = steps = long_run_tokens = 0
    while i < len(reply):
        accepted = 0
        for n in NGRAMS:
            if len(history) <= n:
                continue
            key = history[-n:]
            # most recent earlier occurrence of the key
            for start in range(len(history) - n - 1, -1, -1):
                if history[start:start + n] == key:
                    proposal = history[start + n:start + n + K]
                    while (accepted < len(proposal) and i + accepted < len(reply)
                           and proposal[accepted] == reply[i + accepted]):
                        accepted += 1
                    break
            if accepted:
                break
        if accepted >= 4:
            long_run_tokens += accepted
        advance = min(accepted + 1, len(reply) - i)
        history.extend(reply[i:i + advance])
        i += advance
        steps += 1
    return steps, long_run_tokens


def report(name: str, pairs: list[tuple[list[int], list[int]]]) -> None:
    tokens = sum(len(r) for _, r in pairs)
    steps = longs = 0
    for context, reply in pairs:
        s, lr = simulate(context, reply)
        steps += s
        longs += lr
    print(f"{name:28s} replies {len(pairs):3d}  reply tokens {tokens:6d}  "
          f"tokens/step {tokens / steps:.2f}  in runs >=4: {100 * longs / tokens:.1f} %")


def sessions() -> list[tuple[list[int], list[int]]]:
    pairs = []
    for path in glob.glob(SESSIONS):
        history = json.load(open(path))["data"].get("llm_history", [])
        context: list[int] = []
        for msg in history:
            ids = encode(msg.get("content") or "")
            if msg["role"] == "assistant" and context:
                pairs.append((list(context), ids))
            context.extend(ids)
    return pairs


def quality() -> dict[str, tuple[list[int], list[int]]]:
    prompts = {p["id"]: p["prompt"] for p in json.load(open(QUALITY_PROMPTS))}
    out = {}
    for a in json.load(open(QUALITY_ANSWERS)):
        content = a["text"].split("\n---\n", 1)[-1]
        out[a["id"]] = (encode(prompts[a["id"]]), encode(content))
    return out


q = quality()
report("AIfred-Sitzungen (Prosa)", sessions())
report("Qualitätsfragen ohne Code", [v for k, v in q.items() if k != "code_bug"])
report("code_bug (Code korrigieren)", [q["code_bug"]])
