"""Build the 2026-10-25 body of the overview issue #674 from the live body: one new update paragraph on top, three bullets
(#1240, #1263, #1264) after #1237 in 'Still open', and the closing sentence about the checks corrected.
Usage: make_issue674_update.py <live body file> <output file>"""
import sys

before = open(sys.argv[1], encoding="utf-8").read().rstrip("\n") + "\n"

update = (
    "Update 2026-10-25: three more PRs. #1240 aligns the Mamba replay boundaries when no KV cache group drops the EAGLE last block. "
    "#1263 stops the scheduler from copying and serializing the whole token history of a request on every decode step under pipeline "
    "parallelism, a list that only the V1 runner reads (DeepSeek-V4-Flash PP5 at 287,039 tokens: 85.0-86.5 -> 72.9-73.0 ms per step). "
    "#1264 lets the fused GDN speculative metadata launch run on Turing as well (Qwen3.8-27B TP2 on two Quadro RTX 8000: short decode "
    "73.77-73.93 -> 77.55-77.59 tok/s). 22 PRs are open. Against current main (6a143197e) 18 of them merge cleanly; #1216, #1217, #1218 "
    "and #1235 conflict in the generated docs/design/architecture/runtime_reference.md (which tools/generate_architecture_reference.py "
    "--write regenerates), and #1216 also in vllm/config/kernel.py. The pre-commit CI job is red on main itself at the moment: "
    "ruff-check and mypy report an undefined name tp_slice_packed in vllm/model_executor/layers/quantization/gguf_turbomind_moe.py:609, "
    "mypy reports attr-defined in vllm/v1/kv_cache_interface.py:156-158, and check-layering reports platform coupling 37 -> 39 in "
    "vllm/model_executor/layers/fused_moe/runner/moe_runner.py; it reports the same findings on #1263."
)

bullets = (
    "- #1240 Align the Mamba replay boundaries when no KV cache group drops the EAGLE last block. Without it, a prompt of a model without "
    "EAGLE/MTP that takes the sparse replay boundaries ends its prefill with an extra step (816-token Mamba blocks, 50-token prompt: "
    "chunk ends [49, 50] -> [50]).\n"
    "- #1263 Do not send a request's token history to the V2 runner every decode step. DeepSeek-V4-Flash PP5 at 287,039 tokens: "
    "85.0-86.5 ms per step, with it 72.9-73.0 ms; at 178,480 tokens 79.1-79.6 -> 72.5-73.0 ms; short contexts unchanged.\n"
    "- #1264 Run the fused GDN speculative metadata launch on Turing too. Qwen3.8-27B TP2 on two Quadro RTX 8000, short decode "
    "73.77-73.93 -> 77.55-77.59 tok/s; GLM-5.3-Flash PP5 with DFlash2, short decode 35.98-36.04 -> 36.87-36.88 tok/s.\n"
)

marker = "- #1237 Reduce the PLE offload config to DP1 before it is built."
position = before.index(marker)
line_end = before.index("\n", position) + 1
body = update + "\n\n" + before[:line_end] + bullets + before[line_end:]

old_closing = (
    "The merge state of each open PR was checked against current main with git merge-tree, and the two tests of #739 were run again on "
    "current main."
)
new_closing = (
    "The merge state of each open PR was checked against current main (6a143197e, 2026-10-25) with git merge-tree; the two tests of #739 "
    "were last run again on main 6f3557c97 on 2026-10-19."
)
assert body.count(old_closing) == 1
body = body.replace(old_closing, new_closing)

open(sys.argv[2], "w", encoding="utf-8").write(body)
print(f"geschrieben: {sys.argv[2]}, {len(body)} Zeichen, {body.count(chr(10))} Zeilen")
