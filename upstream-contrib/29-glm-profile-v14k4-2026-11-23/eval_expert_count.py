"""Evaluate the expert-count JSON files of the diag-expert-count tree.

Usage: eval_expert_count.py [dir]   (default: expert_count_2026-11-23 in bench-scripts)
Per MoE layer, for 5-row steps (verify rounds with 1 + 4 draft tokens): number of steps, mean/median/p90 of distinct experts, and
per PP stage the mean over its layers plus the implied bytes read per layer and step (13.44 MB per expert).
The histogram row index i < 8 means i + 1 rows; row index 8 means more than 8 rows (prefill) and is ignored here.
"""

import json
import re
import statistics
import sys
from pathlib import Path

ROWS = 5
BYTES_PER_EXPERT = 13.44e6
DIR = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else "/home/mp/.cache/bench-scripts/expert_count_2026-11-23"
)


def quantile(counts, fraction):
    total = sum(counts)
    target = fraction * total
    running = 0
    for distinct, count in enumerate(counts):
        running += count
        if running >= target:
            return distinct
    return len(counts) - 1


def mean(counts):
    return sum(i * c for i, c in enumerate(counts)) / sum(counts)


print(
    f"{'stage':>5} {'layer':<28} {'steps':>7} {'mean':>6} {'median':>7} {'p90':>5} {'MB/step':>8}"
)
for path in sorted(DIR.glob("rank*.json")):
    stage = re.search(r"rank(\d+)", path.name).group(1)
    layers = json.loads(path.read_text())["layers"]
    stage_means = []
    for name, histogram in layers.items():
        counts = histogram[ROWS - 1]
        if sum(counts) == 0:
            continue
        layer_mean = mean(counts)
        stage_means.append(layer_mean)
        print(
            f"{stage:>5} {name:<28} {sum(counts):>7} {layer_mean:>6.1f} {quantile(counts, 0.5):>7} "
            f"{quantile(counts, 0.9):>5} {layer_mean * BYTES_PER_EXPERT / 1e6:>8.0f}"
        )
    if stage_means:
        overall = statistics.fmean(stage_means)
        print(
            f"stage {stage}: {len(stage_means)} layers, mean distinct {overall:.1f} of 288 "
            f"(uniform routing, 40 slots: 37.8), {overall * BYTES_PER_EXPERT / 1e6:.0f} MB per layer and step"
        )
