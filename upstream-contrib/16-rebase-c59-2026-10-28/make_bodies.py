"""28.10.2026: new bodies for the PRs force-pushed after the rebase onto main c59b3c4cb.

Reads the live bodies (live/body-<n>.md, fetched before the edit), puts an update paragraph on top and replaces the sentences the
merge wave made wrong; every replaced text must occur exactly once. Numbers: test-results-2026-10-28.txt (prtests_2026-10-28).
Writes body-<n>.v2.md next to this script.
"""

from pathlib import Path

HERE = Path(__file__).parent

PRECOMMIT = (
    "pre-commit run --from-ref origin/main --to-ref HEAD on this head: every hook passes except check-layering, which now "
    "reports vllm/config/speculative.py (model coupling 142 -> 144) next to "
    "vllm/model_executor/layers/fused_moe/runner/moe_runner.py (platform coupling 37 -> 39); main c59b3c4cb reports the same "
    "two counts, so this PR adds none. pre-commit run mypy-3.10 --hook-stage manual: passed."
)

UPDATES: dict[int, tuple[str, list[tuple[str, str]]]] = {
    1278: (
        "Update 2026-10-28: rebased onto main c59b3c4cb. The seven commits apply unchanged (git range-diff shows no difference), "
        "and #1279, #1280 and #1282 stack on these same commits again. Test Plan files on this head, one process per file, on a "
        "Quadro RTX 8000 / Tesla V100: test_skinny_sm70_moe.py 103 / 89 passed, test_sm70_qpn8_block_fp8.py 113 passed / 95 "
        "passed and 1 failed, test_block_fp8_qpn8_capabilities.py 24, test_sm70_moe_mainflow.py 55, "
        "test_turing_qpn_kernel_selection.py 34 and test_sm70_path_explanations.py 29 passed on both cards. The failures "
        "(test_sm70_fp4_mainflow.py 168 and test_sm70_native_policy.py 1 on both cards, "
        "test_volta_medium_prefill_keeps_original_turbomind_output[64] on the V100) are the same test IDs on main c59b3c4cb, "
        "as described under Known failures below. " + PRECOMMIT,
        [],
    ),
    1216: (
        "Update 2026-10-28: rebased onto main c59b3c4cb. The commit is unchanged apart from the regenerated "
        "docs/design/architecture/runtime_reference.md (git range-diff). Built in place on this head: "
        "tests/distributed/test_host_reduce.py 1 passed on two Quadro RTX 8000 with NCCL_P2P_DISABLE=1. "
        + PRECOMMIT,
        [],
    ),
    1279: (
        "Update 2026-10-28: rebased onto main c59b3c4cb, which contains #1227; its two commits are no longer part of this "
        "branch, which now holds 14 commits (7 of #1278, 1 of #1216, 6 new). The six are unchanged apart from the regenerated "
        "reference (git range-diff). Test Plan files on this head, Quadro RTX 8000 / Tesla V100: test_prefill_overlap.py 14 / "
        "14, test_paged_prefill_attention_sm75.py 21 / 5 passed (16 skipped), test_nvfp4_qpn2_dequant.py 11 / 11 and "
        "test_flash_v100_dense_prefill_availability.py 2 / 2 passed; test_gpu_model_runner_v2.py 5 passed with the two "
        "failures described below on both cards, which fail the same way on main c59b3c4cb; test_flash_attn_packed_gqa_verify.py "
        "skips as a module as before; tests/distributed/test_host_reduce.py 3 passed on the RTX 8000 pair with "
        "NCCL_P2P_DISABLE=1. " + PRECOMMIT,
        [
            (
                (
                    "Stacked PR. The branch sits on main 899e9ed33 plus three open PRs: #1278 (Turing tensor-core paths, 7 commits), "
                    "#1216 (two-rank host all-reduce, 1 commit, which the copy-engine mode extends) and #1227 (packed GQA verify for "
                    "FA2, 2 commits, same attention backend file). Only the last six commits are new here, so please review those "
                    "(git log 1732a24ea..0d83b1b95, where 1732a24ea is the last commit of #1227). The copies of #1278 and #1227 are "
                    "identical to the open PRs. #1216 as opened conflicts with main 899e9ed33; its copy here is rebased and differs "
                    "from it only in the anchors of the generated docs/design/architecture/runtime_reference.md."
                ),
                (
                    "Stacked PR. The branch sits on main c59b3c4cb plus two open PRs: #1278 (Turing tensor-core paths, 7 commits) "
                    "and #1216 (two-rank host all-reduce, 1 commit, which the copy-engine mode extends); #1227 (packed GQA verify for "
                    "FA2, same attention backend file), on which it was built, is merged. Only the last six commits are new here, so "
                    "please review those (git log e4127aa5a..3f483154e, where e4127aa5a is the copy of #1216). The copy of #1278 is "
                    "identical to the open PR; the copy of #1216 differs from it only in the anchors of the generated "
                    "docs/design/architecture/runtime_reference.md."
                ),
            ),
            (
                "Apart from #1260 and the stacked #1216 and #1227, the merge shows",
                "Apart from #1260 and the stacked #1216 and #1227 (merged since), the merge shows",
            ),
            (
                "This branch carries the same changes rebased onto main 899e9ed33 and #1278,",
                "This branch carries the same changes rebased onto main c59b3c4cb and #1278,",
            ),
        ],
    ),
    1280: (
        "Update 2026-10-28: rebased onto main c59b3c4cb, which contains #1222 and #1227; their commits are no longer part of "
        "this branch, which now holds 20 commits (the 14 of #1279 and 6 new). The six are unchanged apart from the regenerated "
        "reference (git range-diff). Test Plan files on this head, Quadro RTX 8000 / Tesla V100, all passing: "
        "test_deepseek_v4_sm70_indexer.py 31 / 30 (1 skipped), test_deepseek_v4_sm70_routes.py 30 / 30, "
        "test_cudagraph_dispatch.py 51 / 51, test_dsv4_sm70_sparse_bmm.py 12 skipped / 12 passed, "
        "test_dsv4_sparse_prefill_bmm.py 17 / 17, test_deepseek_v4_sm70_qnorm_rope_kv_insert.py 4 / 4, "
        "test_compressor_kv_cache.py 32 / 32 (6 skipped each), test_sparse_indexer_decode_seq_lens.py 11 / 11. "
        + PRECOMMIT,
        [
            (
                (
                    "Stacked PR. The branch sits on #1279 (TP prefill, itself on #1278, #1216 and #1227) and on #1222 (cuBLAS decode "
                    "indexer above 131,072 tokens, 3 commits, same indexer file). The copies of #1222 and of the #1279 stack are "
                    "identical to the open PRs. Only the last six commits are new here, so please review those (git log "
                    "64eb4dbd6..437ad0997, where 64eb4dbd6 is the last commit of #1222)."
                ),
                (
                    "Stacked PR. The branch sits on #1279 (TP prefill, itself on #1278 and #1216). #1222 (cuBLAS decode indexer above "
                    "131,072 tokens, same indexer file), on which it was built, is merged. The copy of the #1279 stack is identical to "
                    "the open PR. Only the last six commits are new here, so please review those (git log 3f483154e..6dc35fdda, where "
                    "3f483154e is the head of #1279)."
                ),
            ),
            (
                "- #1222 is stacked here unchanged.",
                "- #1222 was stacked here unchanged and is merged now.",
            ),
        ],
    ),
    1282: (
        "Update 2026-10-28: rebased onto main c59b3c4cb, which contains #1222 and #1227 from the stack below; this branch now "
        "holds 29 commits (the 20 of #1280 and 9 new), and the nine are unchanged apart from the regenerated reference "
        "(git range-diff). Test Plan files on this head, each with the sm7x Triton and with stock Triton 3.6, on a Quadro RTX "
        "8000 and a Tesla V100: all 11 files pass in all four runs (sm7x Triton: 286 passed / 142 skipped on the RTX 8000, "
        "380 / 48 on the V100; stock Triton: 253 / 175 and 341 / 87). " + PRECOMMIT,
        [
            (
                (
                    "Stacked PR. The branch sits on #1280 (DeepSeek-V4 prefill, itself on #1222 and #1279, which stacks on #1278, "
                    "#1216 and #1227). The copies of those PRs are identical to the open ones. Only the last nine commits are new "
                    "here, so please review those (git log 437ad0997..0356875b6, where 437ad0997 is the head of #1280)."
                ),
                (
                    "Stacked PR. The branch sits on #1280 (DeepSeek-V4 prefill, itself on #1279, which stacks on #1278 and #1216; "
                    "#1222 and #1227 from that stack are merged). The copies of those PRs are identical to the open ones, except that "
                    "the copy of #1216 differs from it in the anchors of the generated docs/design/architecture/runtime_reference.md. "
                    "Only the last nine commits are new here, so please review those (git log 6dc35fdda..dbf0f3c3e, where 6dc35fdda "
                    "is the head of #1280)."
                ),
            ),
        ],
    ),
    1273: (
        "Update 2026-10-28: rebased onto main c59b3c4cb. The three commits are unchanged apart from the regenerated "
        "docs/design/architecture/runtime_reference.md (git range-diff). Built in place on this head; the six Test Plan files "
        "pass on a Tesla V100: test_sm70_warmup.py 29, test_sm70_native_policy.py 91, test_sm70_native_cache.py 3, "
        "test_sm70_fp8_prefill_exact_dense.py 24, test_runtime_default_ownership.py 31 and test_unused_kernel_policy_hash.py "
        "6 passed. On main c59b3c4cb test_sm70_warmup.py still fails 14 of 25 (Tesla V100) and test_sm70_native_policy.py "
        "fails test_packaged_native_policy_schemas_parse_before_build (Quadro RTX 8000). "
        + PRECOMMIT,
        [],
    ),
    1276: (
        "Update 2026-10-28: #1240 and #1265 are merged. The branch is rebased onto main c59b3c4cb and now holds only the three "
        "commits of this PR; they are unchanged apart from the regenerated reference and the context of #1265's "
        "disable_eagle_block_drop line in use_eagle_kv_cache() (git range-diff), and both SpeculativeConfig fields are kept. "
        "The unit tests listed under Test Result ran again on this head (counts there updated). "
        + PRECOMMIT,
        [
            (
                (
                    "Stacked on #1240: its commit (Align Mamba replay boundaries without an EAGLE block drop) is the first commit here "
                    "(the same patch as #1240's head, rebased onto main 6a143197e), because without the drop the replay boundaries "
                    "must be aligned. Please review the three commits after it; if #1240 is merged first, this branch rebases onto it "
                    "without change."
                ),
                (
                    "This PR needs the aligned Mamba replay boundaries of #1240 (merged), because without the drop the replay "
                    "boundaries must be aligned."
                ),
            ),
            (
                "Relation to #1265 (open, backport of vllm#53388)",
                "Relation to #1265 (merged, backport of vllm#53388)",
            ),
            (
                "With both merged, the flag therefore keeps",
                "With this PR merged as well, the flag therefore keeps",
            ),
            (
                "Both PRs add a field to SpeculativeConfig, so whichever is merged second needs a small rebase.",
                "Both PRs add a field to SpeculativeConfig; the rebase onto main kept both.",
            ),
            (
                "Relation to #1244 (open, offloading connector)",
                "Relation to #1244 (merged, offloading connector)",
            ),
            (
                "The commits here are rebased onto main 6a143197e and #1240; not booted with a full model on this base.",
                "The commits here are rebased onto main c59b3c4cb; not booted with a full model on this base.",
            ),
            (
                "Tests on main 6a143197e with this branch (one process per file, Tesla V100 visible):",
                "Tests on main c59b3c4cb with this branch (one process per file, Tesla V100 visible):",
            ),
            (
                "- tests/v1/core/test_async_scheduler.py: 12 passed with two visible GPUs",
                "- tests/v1/core/test_async_scheduler.py: 14 passed with two visible GPUs",
            ),
        ],
    ),
    1235: (
        "Update 2026-10-28: rebased onto main c59b3c4cb. The two commits are unchanged apart from the regenerated "
        "docs/design/architecture/runtime_reference.md (git range-diff). _C rebuilt from this tree; the Test Plan files pass "
        "on this head without a failure: 277 passed on a Tesla V100, 270 passed and 7 skipped on a Quadro RTX 8000 (the test "
        "fakes that #1214 repaired are on main now). " + PRECOMMIT,
        [],
    ),
    1217: (
        "Update 2026-10-28: #1214 is merged, so this branch is now the single commit of this PR on main c59b3c4cb; it is "
        "unchanged apart from the regenerated docs/design/architecture/runtime_reference.md (git range-diff). The seven Test "
        "Plan files pass on this head: 256 passed and 1 skipped with a Tesla V100, 253 passed and 4 skipped with a Quadro RTX "
        "8000. " + PRECOMMIT,
        [
            (
                "This builds on #1214, which repairs the test fakes this PR also extends.",
                "It extends the test fakes that #1214 (merged) repaired.",
            ),
        ],
    ),
    1218: (
        "Update 2026-10-28: rebased onto main c59b3c4cb. The commit is unchanged apart from the regenerated "
        "docs/design/architecture/runtime_reference.md (git range-diff). Test Plan files on this head, on a Tesla V100 and on "
        "a Quadro RTX 8000: test_gpu_draft_vocab.py 12 passed; the other four files 108 passed, 63 skipped, 1 failed, the same "
        "test_default_head_preparation_precedes_graph_mode_guard failure as on main c59b3c4cb. "
        + PRECOMMIT,
        [],
    ),
}

for number, (update, replacements) in UPDATES.items():
    body = (HERE / "live" / f"body-{number}.md").read_text(encoding="utf-8")
    assert body.startswith("## Purpose"), number
    for old, new in replacements:
        assert body.count(old) == 1, (number, old[:60], body.count(old))
        body = body.replace(old, new)
    (HERE / f"body-{number}.v2.md").write_text(update + "\n\n" + body, encoding="utf-8")
    print(f"#{number}: {len(replacements)} Ersetzungen, {len(update)} Zeichen Update")
