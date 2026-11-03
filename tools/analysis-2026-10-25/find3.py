import re, sys
text = open(sys.argv[1], encoding="utf-8").read()
for match in re.finditer(r"start_ab_woa_2026-10-23\.sh[^\n]{0,400}", text):
    print(match.group(0)[:450])
    print("-----")
