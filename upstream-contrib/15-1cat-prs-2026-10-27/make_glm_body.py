"""28.10.2026: GLM-5.3 PR text for the branch after the rebase onto the new #1282 head (main c59b3c4cb; #1274/#1275 merged).

Reads body-glm53.v2.md (filled 28.10. 00:01 against older generations of the branch), rewrites the stack sentences, maps every
commit hash of the branch to its current commit by subject (the fork measurement arms named under Test Result stay), and
fills TESTS_PLACEHOLDER, PRECOMMIT_PLACEHOLDER and BOOT_RESULT_PLACEHOLDER from the given files when they exist.
Writes body-glm53.v3.md. Run from anywhere: paths are absolute.
Usage: make_glm_body.py [tests.txt] [precommit.txt] [boot.txt]
"""

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
TREE = "/home/mp/Projekte/vllm-research/1Cat-vLLM-pr-glm53"
# Fork builds named as A/B arms in Test Result: not commits of this branch, so they keep their hashes.
FORK_ARMS = {"0b8b98c4f", "c213a336e", "31f961300", "5fef3ad60"}

STACK_OLD = (
    "Stacked PR. The branch sits on #1282 (Triton routes, itself on #1280, #1279, #1278, #1222, #1216 and #1227). Above it "
    "come copies of #1274 (2 commits) and #1275 (4 commits), identical to the open PRs, which the boot needs, then three "
    "commits that #1274 and #1275 do not carry"
)
STACK_NEW = (
    "Stacked PR. The branch sits on #1282 (Triton routes, itself on #1280, #1279, #1278 and #1216; #1222 and #1227 below "
    "them are merged). #1274 and #1275, which the boot needs, are merged as well. Above #1282 come three commits that #1274 "
    "and #1275 do not carry"
)

# Facts the full boot of 2026-10-28 added: the copied Turing FA2 library, and #1274/#1275 merged since the fork comparison.
EXTRA_REPLACEMENTS = [
    (
        (
            "Not a duplicate. I compared the changed files with those of all 84 open PRs (complete file lists through the "
            "API), ran a three-way merge with every PR that shares a file (new conflicts counted against #1282),"
        ),
        (
            "Not a duplicate. I compared the changed files with those of all 84 open PRs (complete file lists through the "
            "API), ran a three-way merge with every PR that shares a file (new conflicts counted against #1282; repeated on "
            "2026-10-28 against all 57 PRs then open, with the same findings except one file fewer for #1260),"
        ),
    ),
    ("gets new conflicts in 10 files:", "gets new conflicts in 9 files:"),
    (
        "GLM attention, KDA, the kpool compressor and two GLM tests.",
        "GLM attention, KDA and two GLM tests.",
    ),
    (
        "then this branch's tree (built in place).",
        (
            "then this branch's tree (built in place). The Turing FlashAttention-2 library "
            "(vllm/vllm_flash_attn/_vllm_fa2_C_sm75.abi3.so, which main builds from the pinned fork in "
            "cmake/external_projects/vllm_flash_attn_sm75.cmake and this PR does not change) was copied in from our "
            "production tree, because the in-place build with TORCH_CUDA_ARCH_LIST=7.0 does not build it; the DFlash2 "
            "draft model on the last stage, a Quadro RTX 8000, runs FLASH_ATTN."
        ),
    ),
    (
        "\n\nhttps://claude.ai/code/session_0172gtUiymppDzziDqKfy78Q",
        "",
    ),
    (
        "the copies of #1274 and #1275 as sent).",
        "the copies of #1274 and #1275 as sent, both merged since).",
    ),
]


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", TREE, *args], check=True, capture_output=True, text=True
    ).stdout


def subject(commit: str) -> str:
    return git("log", "-1", "--format=%s", commit).strip()


def main() -> None:
    body = (HERE / "body-glm53.v2.md").read_text(encoding="utf-8")
    assert body.count(STACK_OLD) == 1
    body = body.replace(STACK_OLD, STACK_NEW)

    current = {
        line.split("\t", 1)[1]: line.split("\t", 1)[0][:9]
        for line in git("log", "--format=%H\t%s", "pr-triton-sm7x..HEAD").splitlines()
    }
    assert len(current) == 27, len(current)
    for old_hash in sorted(set(re.findall(r"\b[0-9a-f]{9}\b", body)) - FORK_ARMS):
        new_hash = current[subject(old_hash)]
        body = re.sub(rf"\b{old_hash}\b", new_hash, body)

    for old, new in EXTRA_REPLACEMENTS:
        assert body.count(old) == 1, old[:60]
        body = body.replace(old, new)

    placeholders = [
        "TESTS_PLACEHOLDER",
        "PRECOMMIT_PLACEHOLDER",
        "BOOT_RESULT_PLACEHOLDER",
    ]
    for placeholder, path in zip(placeholders, sys.argv[1:]):
        text = Path(path).read_text(encoding="utf-8").strip()
        assert body.count(placeholder) == 1, placeholder
        body = body.replace(placeholder, text)
    (HERE / "body-glm53.v3.md").write_text(body, encoding="utf-8")
    left = [p for p in placeholders if p in body]
    print(
        f"Hashes: {sorted(set(re.findall(r'\b[0-9a-f]{9}\b', body)))}\noffene Platzhalter: {left}"
    )


if __name__ == "__main__":
    main()
