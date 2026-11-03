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
        text = json.dumps(block.get("input", {})) if block.get("type") == "tool_use" else ""
        if "fp8_tune_small_shapes" in text and "kernel" in text and "start_ab" in text:
            print(entry.get("timestamp"), text[:900])
            print("-----")
