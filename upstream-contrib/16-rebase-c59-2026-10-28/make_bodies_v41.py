"""28.10.2026 ~08:30: #1280 and #1282 after the DeepSeek-V4.1 caller fix (the CI mypy run on the 07:10 push found that
vllm/models/deepseek_v41/attention_ops.py, from #1277, still called attend_gathered_keys with the logits buffer that #1280's
fused-softmax commit removed). Also corrects two prompt lengths in #1282 that our client mislabelled (its long prompts have
8,921 tokens with the DeepSeek-V4 tokenizer; it printed "14k"). Turns the sent bodies (body-<n>.v2.md) into body-<n>.v3.md;
every replaced text must occur exactly once. Evidence: scratchpad check_v41_attend.py (V100 and RTX 8000), prtests_2026-10-28/
v41-*, ab_tfix_dsv4_a/*-client.json.
"""

from pathlib import Path

HERE = Path(__file__).parent

V41_NOTE = (
    "the fused-softmax commit now also updates the two DeepSeek-V4.1 callers of attend_gathered_keys in "
    "vllm/models/deepseek_v41/attention_ops.py (from #1277), which still passed the removed logits buffer and relied on "
    "the function to zero unused keys; the CI mypy run on the earlier push caught it. Against main's function on random "
    "data with NaN in the unused slots, 99.69-100 % of their outputs are bit-identical and the rest differ by one fp16 "
    "step, on a Tesla V100 and a Quadro RTX 8000, and the DeepSeek-V4.1 test files give the same results as on main."
)

REPLACEMENTS: dict[int, list[tuple[str, str]]] = {
    1280: [
        (
            "The six are unchanged apart from the regenerated reference (git range-diff). Test Plan files on this head,",
            "The six are unchanged apart from the regenerated reference (git range-diff), with one addition: "
            + V41_NOTE
            + " Test Plan files on the head before that addition (6dc35fdda),",
        ),
        (
            "(git log 3f483154e..6dc35fdda, where 3f483154e is the head of #1279)",
            "(git log 3f483154e..32e4372db, where 3f483154e is the head of #1279)",
        ),
    ],
    1282: [
        (
            (
                "and the nine are unchanged apart from the regenerated reference (git range-diff). Test Plan files on this "
                "head,"
            ),
            (
                "and the nine are unchanged apart from the regenerated reference (git range-diff). #1280's fused-softmax "
                "commit, which this branch carries, now also updates the DeepSeek-V4.1 callers of attend_gathered_keys (see "
                "#1280). Two prompt lengths in Test Result are corrected: the prompts of the decision A/B have 8,921 tokens "
                "(this text called them 8.7k and 14k). Test Plan files on the head before the DeepSeek-V4.1 update "
                "(dbf0f3c3e),"
            ),
        ),
        (
            "(git log 6dc35fdda..dbf0f3c3e, where 6dc35fdda is the head of #1280)",
            "(git log 32e4372db..bff037870, where 32e4372db is the head of #1280)",
        ),
        (
            "8.7k tokens 2.30 / 2.29 -> 2.29 / 2.29 s.",
            "8,921 tokens 2.30 / 2.29 -> 2.29 / 2.29 s.",
        ),
        (
            "after a 14k prompt 53.45 / 53.55 -> 55.13 / 55.04 tok/s",
            "after an 8,921-token prompt 53.45 / 53.55 -> 55.13 / 55.04 tok/s",
        ),
    ],
}

for number, replacements in REPLACEMENTS.items():
    body = (HERE / f"body-{number}.v2.md").read_text(encoding="utf-8")
    for old, new in replacements:
        assert body.count(old) == 1, (number, old[:70], body.count(old))
        body = body.replace(old, new)
    (HERE / f"body-{number}.v3.md").write_text(body, encoding="utf-8")
    print(f"#{number}: {len(replacements)} Ersetzungen")
