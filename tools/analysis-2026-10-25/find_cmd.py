import json, sys
for line in open(sys.argv[1]):
    try:
        entry = json.loads(line)
    except ValueError:
        continue
    content = entry.get("message", {}).get("content")
    if not isinstance(content, list):
        continue
    for block in content:
        if block.get("type") == "tool_use":
            cmd = block.get("input", {}).get("command", "")
            if "a082b8d19" in cmd and ("woa" in cmd):
                print(entry.get("timestamp"), cmd[:700])
                print("-----")
