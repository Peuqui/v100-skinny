"""28.10.2026 07:20: #1276 and #1218 rebased once more, onto main abec8ba85 (c59b3c4cb + #1284, which regenerated the
SpeculativeConfig anchors both PRs move and removed the speculative.py coupling count). Turns their sent bodies
(body-<n>.v2.md, live since 07:12) into body-<n>.v3.md; every replaced text must occur exactly once.
Numbers: prtests_2026-10-28/{lookahead,draftmap}-abec-*.
"""

from pathlib import Path

HERE = Path(__file__).parent

PRECOMMIT_C59 = (
    "pre-commit run --from-ref origin/main --to-ref HEAD on this head: every hook passes except check-layering, which now "
    "reports vllm/config/speculative.py (model coupling 142 -> 144) next to "
    "vllm/model_executor/layers/fused_moe/runner/moe_runner.py (platform coupling 37 -> 39); main c59b3c4cb reports the same "
    "two counts, so this PR adds none. pre-commit run mypy-3.10 --hook-stage manual: passed."
)
PRECOMMIT_ABEC = (
    "pre-commit run --from-ref origin/main --to-ref HEAD on this head: every hook passes except check-layering, which "
    "reports vllm/model_executor/layers/fused_moe/runner/moe_runner.py (platform coupling 37 -> 39); main abec8ba85 reports "
    "the same, so this PR adds none. pre-commit run mypy-3.10 --hook-stage manual: passed."
)

REPLACEMENTS: dict[int, list[tuple[str, str]]] = {
    1276: [
        (
            "The branch is rebased onto main c59b3c4cb and now holds only the three commits of this PR;",
            "The branch is rebased onto main abec8ba85 (c59b3c4cb plus #1284) and now holds only the three commits of this PR;",
        ),
        (PRECOMMIT_C59, PRECOMMIT_ABEC),
        (
            "The commits here are rebased onto main c59b3c4cb; not booted with a full model on this base.",
            "The commits here are rebased onto main abec8ba85; not booted with a full model on this base.",
        ),
        (
            "Tests on main c59b3c4cb with this branch (one process per file, Tesla V100 visible):",
            "Tests on main abec8ba85 with this branch (one process per file, Tesla V100 visible):",
        ),
    ],
    1218: [
        (
            "Update 2026-10-28: rebased onto main c59b3c4cb.",
            "Update 2026-10-28: rebased onto main abec8ba85 (c59b3c4cb plus #1284).",
        ),
        (PRECOMMIT_C59, PRECOMMIT_ABEC),
    ],
}

for number, replacements in REPLACEMENTS.items():
    body = (HERE / f"body-{number}.v2.md").read_text(encoding="utf-8")
    for old, new in replacements:
        assert body.count(old) == 1, (number, old[:60], body.count(old))
        body = body.replace(old, new)
    (HERE / f"body-{number}.v3.md").write_text(body, encoding="utf-8")
    print(f"#{number}: {len(replacements)} Ersetzungen")
