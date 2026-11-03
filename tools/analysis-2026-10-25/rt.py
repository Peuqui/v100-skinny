import json
path = "/home/mp/Projekte/AIfred-Intelligence/data/model_vram_cache.json"
raw = open(path, encoding="utf-8").read()
data = json.loads(raw)
for indent in (2, 4):
    for ascii_flag in (True, False):
        out = json.dumps(data, indent=indent, ensure_ascii=ascii_flag)
        print(indent, ascii_flag, out == raw, out + "\n" == raw)
print(repr(raw[:160]))
