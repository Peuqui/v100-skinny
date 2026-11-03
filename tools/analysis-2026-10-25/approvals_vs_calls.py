"""Match the user's approval clicks (journal timestamps of POST .../approval, local time) to the tool call that was pending at that moment.
A call is pending at time T when tool_use.timestamp < T <= tool_result.timestamp. Usage: approvals_vs_calls.py <transcript.jsonl> <since HH:MM> <utc_offset_hours>"""
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

transcript = Path(sys.argv[1])
since = sys.argv[2]
offset = int(sys.argv[3])
tz = timezone(timedelta(hours=offset))

journal = subprocess.run(
    ["journalctl", "--user", "--since", f"{since}:00", "--no-pager"],
    capture_output=True,
    text=True,
    env={**os.environ, "XDG_RUNTIME_DIR": "/run/user/1000"},
).stdout
approvals = []
for line in journal.splitlines():
    if "/approval HTTP" in line and "vllm-research" in line:
        m = re.match(r"(\w+ +\d+) (\d+:\d+:\d+)", line)
        if m:
            day = datetime.now(tz).date()
            hh, mm, ss = map(int, m.group(2).split(":"))
            approvals.append(datetime(day.year, day.month, day.day, hh, mm, ss, tzinfo=tz))

uses: dict[str, tuple[datetime, str, str]] = {}
results: dict[str, datetime] = {}
for line in transcript.open(errors="ignore"):
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        continue
    stamp = obj.get("timestamp")
    if not stamp:
        continue
    when = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    content = obj.get("message", {}).get("content")
    if not isinstance(content, list):
        continue
    for item in content:
        if item.get("type") == "tool_use":
            command = item.get("input", {}).get("command") or item.get("input", {}).get("file_path") or json.dumps(item.get("input"))[:160]
            uses[item["id"]] = (when, item["name"], str(command).replace("\n", " ")[:170])
        elif item.get("type") == "tool_result":
            results[item.get("tool_use_id", "")] = when

print(f"{len(approvals)} Genehmigungen seit {since}")
for approval in approvals:
    pending = [
        (used, name, command)
        for tool_id, (used, name, command) in uses.items()
        if used < approval <= results.get(tool_id, used) + timedelta(seconds=2)
    ]
    for used, name, command in sorted(pending)[:2]:
        wait = (approval - used).total_seconds()
        print(f"{approval.strftime('%H:%M:%S')}  wartete {wait:6.1f} s  {name}: {command}")
    if not pending:
        print(f"{approval.strftime('%H:%M:%S')}  (kein wartender Aufruf gefunden)")
