"""28.10.2026: new body of our overview issue #674 after 1Cat's merge wave and the rebase of the open PRs.

Reads live/body-674.md (fetched right before) and merged-titles.tsv (titles of the 17 merged PRs from the API), puts an update
paragraph on top, moves the merged PRs from "Still open" to "Merged", adds the PRs opened on 2026-10-27 that are still open, and
marks #739 as closed. Every number in the new entries comes from the PR's own body. Writes body-674.v2.md.
"""

import re
from pathlib import Path

HERE = Path(__file__).parent

UPDATE = (
    "Update 2026-10-28: thank you for merging #715, #1183, #1213, #1214, #1219, #1222, #1226, #1227, #1229, #1234, #1237, "
    "#1240, #1263, #1264, #1272, #1274 and #1275 overnight, and for closing #739. Since the last update I opened #1272, "
    "#1273, #1274, #1275, #1276, #1278, #1279, #1280 and #1282; the ones still open are listed below. After the merge I "
    "rebased every open PR that conflicted onto current main: the stacks of #1279, #1280 and #1282 no longer carry #1222 "
    "and #1227, #1276 no longer carries #1240, and #1217 no longer carries #1214. Each rebased PR has a short update on top "
    "of its body with the tests rerun on the new base. 14 PRs are open, and all of them merge cleanly into current main "
    "(abec8ba85). On main itself, check-layering still reports "
    "vllm/model_executor/layers/fused_moe/runner/moe_runner.py (platform coupling 37 -> 39)."
)

NEW_OPEN = {
    1273: (
        "- #1273 One grouped GEMM for DeepSeek-V4 wo_a on the V100, and an opt-in TurboMind tuning freeze after a warmup "
        "that now measures in the engine's own policy. DeepSeek-V4-Flash PP5: decode step after a 35.7k-token prompt "
        "66.9/67.2 -> 63.0/63.2 ms; before the warmup fix 6 of 24 prompt lengths took 1.4-3.7 s longer on their first "
        "request."
    ),
    1276: (
        "- #1276 Lookahead block hashes for EAGLE/MTP prefix hits (backport of vllm-project/vllm#61589); DSpark keeps the "
        "last-block drop. Qwen3.8-27B TP2 with MTP: first token of a repeated 29k-token prompt 1.7 s -> 0.7-0.9 s."
    ),
    1278: (
        "- #1278 Turing tensor-core paths for block-FP8 QPN8 linears and the grouped skinny MoE, and the MoE grid bounded "
        "by the expert count. Skinny MoE on a Quadro RTX 8000, DeepSeek-V4 shapes at 512 tokens: w13 6555 -> 4303 us, "
        "w2 3271 -> 2238 us."
    ),
    1279: (
        "- #1279 Turing paged prefill attention for head dim 256, and TP prefill all-reduces hidden behind compute on GPUs "
        "without P2P (copy-engine mode of the host all-reduce of #1216). Qwen3.8-27B TP2 on two Quadro RTX 8000: 29k-token "
        "prompts 33.0-33.1 -> 19.1-19.3 s, 184,134 tokens 269.05-269.21 -> 174.34-174.38 s. Stacked on #1278 and #1216."
    ),
    1280: (
        "- #1280 DeepSeek-V4 long prefill: vectorized indexer epilogue, fused sparse MLA softmax, fused Turing indexer "
        "scorer, indexer logits freed per sub-chunk, wider compressed KV gather. DeepSeek-V4-Flash PP5 with the first two: "
        "287k-token prefill 222.26 / 222.19 -> 149.73 / 149.72 s. Stacked on #1279."
    ),
    1282: (
        "- #1282 Opt-in tensor-core Triton routes for DeepSeek-V4 prefill on Volta and Turing, for a Triton 3.6 that lowers "
        "tl.dot to mma on sm_70 and sm_75 (github.com/Peuqui/triton, shown in "
        "https://github.com/triton-lang/triton/discussions/12517). DeepSeek-V4-Flash PP5, together with parts of the stacked "
        "PRs and 1024-token chunks: 287,039-token prefill 144.74 / 144.64 -> 73.57 / 73.55 s. Stacked on #1280."
    ),
}

ENTRY_FIXES = [
    (" Builds on #1214.", ""),
    (
        "Bug report #739: the first test was fixed on main by 91701504c, the second is fixed by #1237.",
        "Bug report #739 is closed: the first test was fixed on main by 91701504c, the second by #1237.",
    ),
]


def entry_number(line: str) -> int:
    match = re.match(r"- #(\d+) ", line)
    assert match, line
    return int(match.group(1))


body = (HERE / "live" / "body-674.md").read_text(encoding="utf-8")
merged_titles = {
    int(number): re.sub(r"^(\[[^\]]+\])+ *", "", title)
    for number, title in (
        line.split("\t", 1)
        for line in (HERE / "merged-titles.tsv")
        .read_text(encoding="utf-8")
        .splitlines()
    )
}
head, rest = body.split("\nStill open\n\n", 1)
open_block, rest = rest.split("\n\nBug report #739", 1)
rest = "Bug report #739" + rest
middle, merged_block = rest.split("\nMerged\n\n", 1)

open_lines = open_block.split("\n")
assert all(line.startswith("- #") for line in open_lines)
# #1272, #1274 and #1275 were opened after the last update of the overview and merged before this one.
NEVER_LISTED = {1272, 1274, 1275}
moved = {
    entry_number(line) for line in open_lines if entry_number(line) in merged_titles
}
assert moved == set(merged_titles) - NEVER_LISTED, sorted(
    set(merged_titles) - NEVER_LISTED - moved
)
still_open = [
    line for line in open_lines if entry_number(line) not in merged_titles
] + list(NEW_OPEN.values())
still_open.sort(key=entry_number)
assert len(still_open) == 14, len(still_open)

new_merged = [
    f"- #{number} {merged_titles[number]}"
    for number in sorted(merged_titles, reverse=True)
]
new_body = (
    UPDATE
    + "\n\n"
    + head
    + "\nStill open\n\n"
    + "\n".join(still_open)
    + "\n\n"
    + middle
    + "\nMerged\n\n"
    + "\n".join(new_merged)
    + "\n"
    + merged_block
)
for old, new in ENTRY_FIXES:
    assert new_body.count(old) == 1, (old, new_body.count(old))
    new_body = new_body.replace(old, new)
(HERE / "body-674.v2.md").write_text(new_body, encoding="utf-8")
print(
    f"offen {len(still_open)}, neu gemergt {len(new_merged)}, Zeichen {len(body)} -> {len(new_body)}"
)
