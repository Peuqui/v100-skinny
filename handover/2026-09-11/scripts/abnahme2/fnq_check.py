"""Vorfilter fuer Flash-Next-Antworten: Coandă-Erkennung, CJK, Wiederholungen,
Denkblock-Anteil. Ersetzt NICHT das Lesen (STAND.md: Zaehler sagen nichts ueber
Qualitaet), sortiert nur vor. Aufruf: fnq_check.py <fnq_dir>... [--show q3]"""
import collections
import re
import sys
from pathlib import Path

show = None
dirs = []
for a in sys.argv[1:]:
    if a.startswith("--show="):
        show = a.split("=", 1)[1]
    else:
        dirs.append(Path(a))

CJK = re.compile(r"[一-鿿぀-ヿ가-힯]")

for d in dirs:
    print(f"== {d.name}")
    for q in ("q1", "q2", "q3"):
        p = d / f"text_{q}.txt"
        if not p.exists():
            print(f"  {q}: fehlt")
            continue
        t = p.read_text()
        think = ""
        m = re.search(r"<think>(.*?)</think>", t, re.S)
        if m:
            think = m.group(1)
            answer = t[m.end():]
        else:
            answer = t
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.strip()) > 20]
        dups = sum(n - 1 for n in collections.Counter(sents).values() if n > 1)
        numbered = len(re.findall(r"^\s*\d+\.\s", answer, re.M))
        flags = []
        if q == "q3":
            # Regel Peuqui 11.09.: bestanden = Coandă erklaert ODER Begriff
            # zurueckgewiesen und Coandă als Kandidat genannt; Durchfall nur
            # bei Erfindung/Zerfasern (das entscheidet das Lesen).
            rejects = bool(re.search(r"nicht bekannt|kenne ich nicht|existiert nicht|kein bekannter|gibt (es )?keinen|nicht belegt", answer))
            if "Coand" not in t:
                flags.append("COANDĂ FEHLT - lesen!")
            elif rejects:
                flags.append("bestanden (zurueckgewiesen, Coandă genannt)")
            else:
                flags.append("bestanden (Coandă erklaert)")
        cjk = len(CJK.findall(t))
        if cjk:
            flags.append(f"CJK={cjk}")
        if dups:
            flags.append(f"Satzwiederholungen={dups}")
        if not m:
            flags.append("kein </think>")
        elif not answer.strip():
            flags.append("Antwort leer (nur Denkblock)")
        print(f"  {q}: Zeichen {len(t):5d} | Denkblock {len(think):5d} | Antwort {len(answer):5d} | "
              f"Saetze {len(sents):3d} | nummeriert {numbered:3d} | {', '.join(flags) or 'ok'}")
        if show == q:
            print("  --- Antwort (Anfang):", answer.strip()[:700].replace("\n", " "))
            print("  --- Antwort (Ende):", answer.strip()[-400:].replace("\n", " "))
