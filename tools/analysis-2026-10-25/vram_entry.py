import json, sys
for path in sys.argv[1:]:
    data = json.load(open(path))
    print("==", path, "Top-Level-Schlüssel:", list(data)[:6], "Einträge:", len(data.get("models", data)))
    models = data.get("models", data)
    for key, value in models.items():
        if "Qwen3VL-30B" in key:
            print(key)
            print(json.dumps(value, indent=1, ensure_ascii=False)[:1800])
