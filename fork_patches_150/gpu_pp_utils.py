# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline Parallelism utils for V2 Model Runner."""

import torch

from vllm.distributed.parallel_state import get_pp_group


def pp_broadcast(
    sampled_token_ids: torch.Tensor,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    max_sample_len: int = 1,
) -> None:
    pp = get_pp_group()
    assert pp.is_last_rank

    assert sampled_token_ids.dtype == torch.int64
    # Fork fix (v100-skinny): the row count travels on both broadcasts and
    # pp_receive derives it from its own num_reqs. Tie the two payloads
    # together here so a divergence raises instead of deadlocking.
    assert sampled_token_ids.shape[0] == num_sampled.shape[0]
    assert num_rejected.shape[0] == num_sampled.shape[0]
    # Fork fix (v100-skinny): pad to the wire shape pp_receive allocates.
    # The sampler only emits the columns a round actually produced -- one
    # after a prefill step, and fewer than max_sample_len in any round with
    # no scheduled drafts -- while pp_receive unconditionally allocates
    # [num_reqs, max_sample_len]. NCCL requires equal counts on every rank,
    # so an unpadded send does not raise, it deadlocks: the last stage
    # pushes its short buffer and moves on while the earlier stages spin in
    # a broadcast kernel that never completes. With spec decode this is
    # deterministic -- the first sample after warmup prefill has one column
    # and max_sample_len is num_speculative_steps + 1, so no PP boot with
    # k > 0 ever gets past it. -1 is the rejection sampler's placeholder;
    # the post_update kernel reads only the first num_sampled columns of
    # each row, so the padding is never observed.
    num_cols = sampled_token_ids.shape[1]
    assert num_cols <= max_sample_len
    if num_cols < max_sample_len:
        padded = sampled_token_ids.new_full(
            (sampled_token_ids.shape[0], max_sample_len), -1
        )
        padded[:, :num_cols] = sampled_token_ids
        sampled_token_ids = padded
    torch.distributed.broadcast(
        sampled_token_ids.contiguous(), src=pp.last_rank, group=pp.device_group
    )

    combined = torch.stack((num_sampled, num_rejected), dim=0)
    torch.distributed.broadcast(combined, src=pp.last_rank, group=pp.device_group)


def pp_receive(
    num_reqs: int, max_sample_len: int = 1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pp = get_pp_group()
    assert not pp.is_last_rank

    sampled_tokens = torch.empty(
        num_reqs, max_sample_len, dtype=torch.int64, device=pp.device
    )
    torch.distributed.broadcast(sampled_tokens, src=pp.last_rank, group=pp.device_group)

    combined = torch.empty(2, num_reqs, dtype=torch.int32, device=pp.device)
    torch.distributed.broadcast(combined, src=pp.last_rank, group=pp.device_group)
    num_sampled, num_rejected = combined.unbind(dim=0)
    return sampled_tokens, num_sampled, num_rejected


def pp_broadcast_drafts(draft_tokens: torch.Tensor, num_reqs: int) -> None:
    """Fork addition (v100-skinny): ship this step's draft token ids from the
    last PP stage.

    Only the last stage owns a speculator, so only it produces draft tokens --
    but ``req_states.draft_tokens`` is read by ``combine_sampled_and_draft_tokens``
    on EVERY rank, and under PP it is the FIRST stage that embeds the tokens.
    Without this transfer the earlier stages keep the zero-initialised buffer and
    embed token id 0 in every draft slot: the model boots and then emits fluent
    garbage. The draft COUNT needs no wire -- it reaches all ranks through
    scheduler_output -- only the values do.
    """
    pp = get_pp_group()
    assert pp.is_last_rank

    assert draft_tokens.dtype == torch.int64
    assert draft_tokens.shape[0] == num_reqs
    torch.distributed.broadcast(
        draft_tokens.contiguous(), src=pp.last_rank, group=pp.device_group
    )


def pp_receive_drafts(num_reqs: int, num_speculative_steps: int) -> torch.Tensor:
    pp = get_pp_group()
    assert not pp.is_last_rank

    drafts = torch.empty(
        num_reqs, num_speculative_steps, dtype=torch.int64, device=pp.device
    )
    torch.distributed.broadcast(drafts, src=pp.last_rank, group=pp.device_group)
    return drafts
