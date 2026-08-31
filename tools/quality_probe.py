#!/usr/bin/env python3
"""Qualitaetssonde: dieselben drei Prompts wie in den AIfred-Sitzungen,
direkt gegen einen laufenden vLLM-Server.

Ziel ist die Vergleichbarkeit mit den Sitzungen vom 31.08. (NVFP4 mit und
ohne Kachel-Patch, Q6 ueber llama.cpp). Deshalb identischer Wortlaut samt
Tippfehler "Kuanda" und dieselbe Reihenfolge — der dritte Turn traegt den
groessten Kontext und zeigte dort den staerksten Zerfall.

Aufruf: quality_probe.py <port> <ausgabedatei>
"""
import json
import re
import sys
import urllib.request

PORT, OUT = sys.argv[1], sys.argv[2]
NAME = "grid-test"

PROMPTS = [
    "Erkläre Quantenphysik in 30 Sätzen.",
    "Erkläre den Regenbogeneffekt in 30 Sätzen.",
    "Erkläre den Kuanda-Effekt in 30 Sätzen.",
]
# Knapper Butler-Systemprompt: die Persona treibt den Stil, auf den die
# frueheren Sitzungen gemessen wurden (englische Einwuerfe sind gewollt).
# Naeher an AIfreds Persona-Prompt: Die Nummerierungs-Anweisung ist
# wichtig, weil die Sitzungen 4.500-5.900 Zeichen lange Antworten
# erzeugten — und der Zerfall trat im LETZTEN VIERTEL langer Antworten
# auf. Eine kurze Antwort wuerde ihn gar nicht ausloesen.
SYSTEM = ("Du bist AIfred, ein kultivierter britischer Butler. Antworte auf "
          "DEUTSCH, hoeflich-formell mit trockenem Humor. Du darfst einzelne "
          "englische Woerter einstreuen: indeed, rather, quite, splendid. "
          "Wenn eine Anzahl Saetze verlangt wird, nummeriere sie und liefere "
          "GENAU diese Anzahl, jeden Satz inhaltlich ausgefuehrt. "
          "Korrigiere Rechtschreibfehler des Nutzers stillschweigend und "
          "beantworte, was er gemeint hat.")


def ask(messages, max_tokens=2000):
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps({
            "model": NAME, "messages": messages, "max_tokens": max_tokens,
            "temperature": 1.0, "top_p": 0.95,
        }).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1200) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


CJK = re.compile(r'[一-鿿぀-ヿ가-힯]')
messages = [{"role": "system", "content": SYSTEM}]
answers = []
for i, p in enumerate(PROMPTS, 1):
    messages.append({"role": "user", "content": p})
    text = ask(messages)
    messages.append({"role": "assistant", "content": text})
    answers.append(text)
    cjk = CJK.findall(text)
    nums = re.findall(r'^\s*(\d+)\.', text, re.M)
    print(f"Antwort {i}: {len(text)} Zeichen, {len(nums)} nummerierte Saetze, "
          f"{len(cjk)} CJK {cjk[:8]}", flush=True)

with open(OUT, "w") as f:
    json.dump(answers, f, ensure_ascii=False, indent=1)
print(f"gespeichert: {OUT}")
