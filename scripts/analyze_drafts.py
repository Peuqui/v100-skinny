#!/usr/bin/env python3
"""Zeigt pro MTP-Runde: was der Drafter vorschlug vs. was akzeptiert wurde."""
import glob, os, sys, torch
from transformers import AutoTokenizer

DUMP = sys.argv[1]
SNAP = sys.argv[2]
tok = AutoTokenizer.from_pretrained(SNAP, trust_remote_code=True)

def dec(ids):
    out = []
    for i in ids:
        i = int(i)
        if i < 0:
            out.append("·")          # -1 = Padding/verworfen
        else:
            out.append(repr(tok.decode([i]))[1:-1])
    return out

files = sorted(glob.glob(os.path.join(DUMP, "*.pt")))
print(f"{len(files)} Dumps\n")
shown = 0
for f in files:
    d = torch.load(f, weights_only=False)
    drafts = d.get("draft_token_ids")
    sampled = d.get("sampled_token_ids")
    if drafts is None or sampled is None:
        continue
    if isinstance(drafts, dict) or isinstance(sampled, dict):
        continue                     # gekuerzte Grosstensoren ueberspringen
    dr = drafts.reshape(-1).tolist() if hasattr(drafts, "reshape") else list(drafts)
    sp = sampled.reshape(-1).tolist() if hasattr(sampled, "reshape") else list(sampled)
    if not dr:
        continue
    valid = d.get("valid_sampled_count")
    v = int(valid.reshape(-1)[0]) if hasattr(valid, "reshape") and valid.numel() else -1
    print(f"--- {os.path.basename(f).split('_rank')[0]} … akzeptiert={v} von {len(dr)+1}")
    print(f"    Drafter schlug vor : {dec(dr)}")
    print(f"    Verifier lieferte  : {dec(sp)}")
    shown += 1
    if shown >= 14:
        break
