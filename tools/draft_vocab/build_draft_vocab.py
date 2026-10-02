#!/usr/bin/env python3
"""Build the MTP draft token list for Flash-Next (plan Oct 2026, item 2).

Ranking: tokens by frequency in the model-generated corpus (reasoning +
answer), then the remaining ids in BPE order (low ids first), so code and rare
tokens still get a slot. Every special/added token (think tags, tool-call
markers, EOS) is always included. Coverage is measured on held-out text the
list never saw: AIfred's real sessions (assistant turns) and the quality
answers.

Usage: build_draft_vocab.py <corpus.jsonl> <out_dir>
Writes draft_vocab_<N>.json for N in SIZES and prints the coverage table.
"""
import collections
import glob
import json
import os
import sys

from tokenizers import Tokenizer

TOKENIZER = glob.glob("/home/mp/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/*/tokenizer.json")[0]
SIZES = (32768, 49152, 65536, 98304)
corpus_path, out_dir = sys.argv[1:]

tok = Tokenizer.from_file(TOKENIZER)
vocab_size = tok.get_vocab_size()
added = sorted(a["id"] for a in json.load(open(TOKENIZER))["added_tokens"])


def ids(text: str) -> list[int]:
    return tok.encode(text, add_special_tokens=False).ids


counts: collections.Counter = collections.Counter()
corpus_tokens = 0
for line in open(corpus_path):
    row = json.loads(line)
    t = ids(row["reasoning"]) + ids(row["content"])
    counts.update(t)
    corpus_tokens += len(t)

# Second tier: German running text (AIfred's German docs and prompts) — real word
# frequencies for German, which the small corpus alone misses.
german: collections.Counter = collections.Counter()
for path in glob.glob("/home/mp/Projekte/AIfred-Intelligence/docs/de/**/*.md", recursive=True) + \
        glob.glob("/home/mp/Projekte/AIfred-Intelligence/prompts/de/**/*.txt", recursive=True):
    german.update(ids(open(path, encoding="utf-8").read()))
# Code tier: Python sources (AIfred + vLLM model/layer code) — identifiers, keywords and
# indentation runs that German prose never produces. AI-Connect is left out on purpose: the
# edit bench rewrites one of its files.
code: collections.Counter = collections.Counter()
for path in glob.glob("/home/mp/Projekte/AIfred-Intelligence/aifred/**/*.py", recursive=True) + \
        glob.glob("/home/mp/Projekte/vllm-research/1Cat-vLLM-work/vllm/model_executor/layers/*.py") + \
        glob.glob("/home/mp/Projekte/vllm-research/1Cat-vLLM-work/vllm/v1/core/sched/*.py"):
    code.update(ids(open(path, encoding="utf-8", errors="ignore").read()))
# Third tier: every inflected word of the German dictionary, as it appears mid-sentence
# (leading space), lower- and capitalised.
dictionary: collections.Counter = collections.Counter()
for word in open("/usr/share/dict/ngerman", encoding="utf-8", errors="ignore"):
    word = word.strip()
    if word:
        dictionary.update(ids(" " + word))
        dictionary.update(ids(" " + word[:1].upper() + word[1:]))

order: list[int] = []
seen: set[int] = set()
for tier in (counts, german, code, dictionary):
    for i, _ in tier.most_common():
        if i not in seen:
            seen.add(i)
            order.append(i)
order += [i for i in range(vocab_size) if i not in seen]

holdout: dict[str, list[int]] = {"AIfred-Sitzungen": [], "Qualitätsantworten": []}
for p in glob.glob("/home/mp/Projekte/AIfred-Intelligence/data/sessions/*.json"):
    for m in json.load(open(p))["data"].get("llm_history", []):
        if m["role"] == "assistant":
            holdout["AIfred-Sitzungen"] += ids(m.get("content") or "")
holdout["Bearbeitungsdatei (AI-Connect)"] = ids(open("/home/mp/Projekte/AI-Connect/client/tools.py").read())
for p in glob.glob("/home/mp/.cache/bench-scripts/quality_2026-09-27/produnion-*-ds.json"):
    for a in json.load(open(p)):
        holdout["Qualitätsantworten"] += ids(a["text"])

print(f"Korpus: {corpus_tokens} Token, {len(counts)} verschiedene; deutscher Fließtext "
      f"{sum(german.values())} Token; Code {sum(code.values())} Token; Wörterbuch {len(dictionary)} verschiedene; Sondertokens: {len(added)}")
os.makedirs(out_dir, exist_ok=True)
for size in SIZES:
    chosen = set(added)
    for i in order:
        if len(chosen) >= size:
            break
        chosen.add(i)
    out = os.path.join(out_dir, f"draft_vocab_{size}.json")
    json.dump(sorted(chosen), open(out, "w"))
    cov = "  ".join(f"{name} {100 * sum(1 for i in h if i in chosen) / len(h):5.1f} %"
                    for name, h in holdout.items())
    print(f"{size:6d} Token: {cov}  -> {out}")
