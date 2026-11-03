"""Count the leading command (plus subcommand for a few tools) of every Bash tool call in the most recent session transcripts."""
import json
import re
import shlex
import sys
from collections import Counter
from pathlib import Path

LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 50
files = sorted(Path.home().glob(".claude/projects/*/*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)[:LIMIT]
counts: Counter[str] = Counter()
examples: dict[str, str] = {}
SUB = {"git", "systemctl", "journalctl", "docker", "gh", "nvidia-smi", "curl"}
ENV = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def leading(part: str) -> str | None:
    part = part.strip()
    if not part or part.startswith(("#", "then", "do", "done", "fi", "}", "{", "(")):
        return None
    try:
        tokens = shlex.split(part, posix=True)
    except ValueError:
        tokens = part.split()
    while tokens and (ENV.match(tokens[0]) or tokens[0] in ("timeout", "nice", "env", "time")):
        if tokens[0] == "timeout" and len(tokens) > 1:
            tokens = tokens[2:]
        elif tokens[0] in ("nice", "env", "time"):
            tokens = tokens[1:]
        else:
            tokens = tokens[1:]
    if not tokens:
        return None
    head = tokens[0]
    if head == "git" and len(tokens) > 2 and tokens[1] == "-C":
        tokens = ["git"] + tokens[3:]
    if head in SUB and len(tokens) > 1:
        sub = next((t for t in tokens[1:] if not t.startswith("-")), "")
        if head == "systemctl":
            sub = " ".join(t for t in tokens[1:3])
        return f"{head} {sub}".strip()
    return head


for path in files:
    for line in path.open(errors="ignore"):
        if '"Bash"' not in line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        content = obj.get("message", {}).get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if item.get("type") == "tool_use" and item.get("name") == "Bash":
                command = item.get("input", {}).get("command", "")
                for part in re.split(r"&&|\|\||;|\n|\|", command):
                    key = leading(part)
                    if key:
                        counts[key] += 1
                        examples.setdefault(key, part.strip()[:110])
print(f"{len(files)} Transkripte, {sum(counts.values())} Teilbefehle")
for key, n in counts.most_common(80):
    print(f"{n:5d}  {key:34s} {examples[key]}")
