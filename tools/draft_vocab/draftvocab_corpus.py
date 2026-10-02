#!/usr/bin/env python3
"""Generate a corpus of Flash-Next output for building the MTP draft token list.

Plan Oct 2026, item 2 (reduced draft vocabulary, vLLM #59740). The list must
come from text like the traffic we serve — mostly German, some English and
code — and from what the model itself generates (reasoning + answer, both are
drafted by MTP). AIfred's real sessions are NOT used here; they are the
held-out set that measures the list's coverage afterwards.

Usage: MODEL=<served name> draftvocab_corpus.py <out.jsonl>
"""
import concurrent.futures as cf
import itertools
import json
import os
import random
import sys
import urllib.request

SWAP = "http://127.0.0.1:11435"
MODEL = os.environ["MODEL"]
OUT = sys.argv[1]
CONCURRENCY = 4
MAX_TOKENS = 700

DE_TOPICS = [
    "die Photosynthese", "den Klimawandel", "die Funktionsweise eines Kühlschranks",
    "die Geschichte der Hanse", "Schwarze Löcher", "das deutsche Rentensystem",
    "Impfungen und das Immunsystem", "die Reformation", "Quantencomputer",
    "die Ursachen des Ersten Weltkriegs", "Wärmepumpen", "den Blutkreislauf",
    "die Grundrechte im Grundgesetz", "Inflation", "die Entstehung der Alpen",
    "Bienen und Bestäubung", "Künstliche Intelligenz im Alltag", "Diabetes Typ 2",
    "die Mondlandung", "Kernfusion", "die Psalmen im Alten Testament",
    "Gleichnisse Jesu", "Datenschutz bei Smartphones", "Solaranlagen auf dem Dach",
    "Stromspeicher für Zuhause", "das Bauhaus", "Goethes Faust", "Vulkane",
    "den Kaffeeanbau", "Narkose und Anästhesie", "Antibiotikaresistenzen",
    "Elektroautos im Winter", "die Geschichte des Internets", "Zeitzonen",
    "den Goldenen Schnitt", "Sauerteigbrot", "Gartenkompost", "Schlafhygiene",
    "Erste Hilfe bei Herzstillstand", "Mietrecht bei Nebenkosten",
]
DE_TASKS = [
    "Erkläre {t} so, dass es ein interessierter Laie versteht.",
    "Fasse die wichtigsten Fakten über {t} in Stichpunkten zusammen.",
    "Welche häufigen Missverständnisse gibt es über {t}? Stelle sie richtig.",
    "Schreibe einen kurzen Zeitungsartikel über {t}.",
    "Nenne Vor- und Nachteile im Zusammenhang mit {t} und wäge ab.",
    "Erkläre {t} einem zehnjährigen Kind.",
]
DE_FREE = [
    "Schreibe ein kurzes Morgengebet für einen arbeitsreichen Tag.",
    "Schreibe ein Abendgebet mit einem Bezug auf Psalm 23.",
    "Formuliere eine höfliche E-Mail an meinen Vermieter wegen einer defekten Heizung.",
    "Plane ein Abendessen für sechs Personen, eine davon isst vegetarisch.",
    "Wie bereite ich mich auf ein Vorstellungsgespräch als Krankenpfleger vor?",
    "Vergleiche Bahn und Auto für eine Reise von Hamburg nach München.",
    "Was sollte ich beim Kauf eines gebrauchten Fahrrads beachten?",
    "Schreibe eine kurze Geschichte über einen Leuchtturmwärter.",
    "Wie viele Tage liegen zwischen dem 3. März und dem 19. August? Rechne nachvollziehbar.",
    "Ein Zug fährt 312 km in 2 Stunden 40 Minuten. Wie schnell fährt er im Schnitt?",
    "Erkläre den Unterschied zwischen Wetter und Klima mit Beispielen.",
    "Gib mir Tipps, wie ich mir Vokabeln besser merken kann.",
    "Wie funktioniert eine Steuererklärung für Rentner in Deutschland grob?",
    "Diskutiere: Sollte es ein Tempolimit auf Autobahnen geben?",
]
EN_TASKS = [
    "Explain how {t} works in plain English.",
    "Summarize the key facts about {t} as a bullet list.",
    "Write a short blog post about {t}.",
]
EN_TOPICS = [
    "transformer neural networks", "the French Revolution", "compound interest",
    "plate tectonics", "the human microbiome", "public key cryptography",
    "the water cycle", "speculative decoding in language models",
]
CODE_TASKS = [
    "Schreibe eine Python-Funktion, die prüft, ob ein String ein Palindrom ist, mit Tests.",
    "Write a Python class for an LRU cache with type hints and docstrings.",
    "Erkläre diesen Fehler und behebe ihn:\n```python\ndef mean(xs):\n    return sum(xs) / len(xs) + 1\n```",
    "Write a bash script that backs up a directory to a dated tar.gz and keeps the last 7.",
    "Schreibe eine SQL-Abfrage, die pro Kunde die Summe der Bestellungen 2025 liefert.",
    "Implement binary search in Python and explain its complexity.",
    "Schreibe ein kleines FastAPI-Beispiel mit einem GET- und einem POST-Endpunkt.",
    "Refactor this into smaller functions:\n```python\ndef report(rows):\n    out=[]\n    for r in rows:\n        if r['age']>18 and r['active']:\n            out.append(r['name'].upper()+': '+str(r['score']*1.1))\n    return '\\n'.join(out)\n```",
    "Write a TypeScript function that debounces another function.",
    "Erkläre den Unterschied zwischen Liste und Tupel in Python mit Codebeispielen.",
]


def prompts() -> list[str]:
    rng = random.Random(42)
    de = [task.format(t=t) for t, task in itertools.product(DE_TOPICS, DE_TASKS)]
    rng.shuffle(de)
    en = [task.format(t=t) for t, task in itertools.product(EN_TOPICS, EN_TASKS)]
    return de[:110] + DE_FREE + en + CODE_TASKS


def generate(prompt: str) -> dict:
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7, "top_p": 0.95, "max_tokens": MAX_TOKENS}
    req = urllib.request.Request(f"{SWAP}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        msg = json.load(r)["choices"][0]["message"]
    return {"prompt": prompt,
            "reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "",
            "content": msg.get("content") or ""}


items = prompts()
print(f"{len(items)} prompts", flush=True)
with open(OUT, "w") as f, cf.ThreadPoolExecutor(CONCURRENCY) as pool:
    for n, result in enumerate(pool.map(generate, items), 1):
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
        if n % 20 == 0:
            print(f"{n}/{len(items)}", flush=True)
print("done", flush=True)
