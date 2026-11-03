"""Print the environment and the non-command keys of one llama-swap model entry. Usage: show_entry.py <model name>"""
import sys

import yaml

config = yaml.safe_load(open("/home/mp/.config/llama-swap/config.yaml"))
entry = config["models"][sys.argv[1]]
for key, value in entry.items():
    if key == "cmd":
        continue
    print(key, "=", value if key != "env" else "")
for item in entry.get("env", []):
    print("   ", item)
groups = [name for name, group in config["groups"].items() if sys.argv[1] in group["members"]]
print("groups:", groups)
