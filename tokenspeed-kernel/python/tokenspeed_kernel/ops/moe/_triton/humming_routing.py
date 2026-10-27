# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Compact, stable local-expert routing for Humming's indexed GEMM."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton


@triton.jit(do_not_specialize=["route_count"])
def _count_local_routes(ids, counts, route_count, expert_start, BLOCK: tl.constexpr):
    expert = tl.program_id(0)
    count = 0
    for start in range(0, route_count, BLOCK):
        routes = start + tl.arange(0, BLOCK)
        selected = tl.load(ids + routes, routes < route_count, other=-1)
        count += tl.sum(
            ((routes < route_count) & (selected == expert_start + expert)).to(tl.int32)
        )
    tl.store(counts + expert, count)


@triton.jit
def _prefix_local_routes(
    counts,
    offsets,
    padded_count,
    num_experts: tl.constexpr,
    BLOCK_M: tl.constexpr,
    EXPERTS: tl.constexpr,
):
    experts = tl.arange(0, EXPERTS)
    count = tl.load(counts + experts, experts < num_experts, other=0)
    padded = tl.cdiv(count, BLOCK_M) * BLOCK_M
    ends = tl.cumsum(padded)
    tl.store(offsets + experts, ends - padded, experts < num_experts)
    tl.store(padded_count, tl.sum(padded))


@triton.jit(do_not_specialize=["route_count"])
def _write_local_routes(
    ids,
    counts,
    offsets,
    sorted_ids,
    expert_ids,
    route_count,
    expert_start,
    BLOCK_M: tl.constexpr,
    BLOCK: tl.constexpr,
    PADDING: tl.constexpr,
):
    expert = tl.program_id(0)
    count = tl.load(counts + expert)
    begin = tl.load(offsets + expert)
    copied = 0
    for start in range(0, route_count, BLOCK):
        routes = start + tl.arange(0, BLOCK)
        selected = tl.load(ids + routes, routes < route_count, other=-1)
        match = (routes < route_count) & (selected == expert_start + expert)
        position = tl.cumsum(match.to(tl.int32)) - 1
        tl.store(sorted_ids + begin + copied + position, routes, match)
        copied += tl.sum(match.to(tl.int32))
    padded = tl.cdiv(count, BLOCK_M) * BLOCK_M
    padding = tl.arange(0, PADDING)
    tl.store(
        sorted_ids + begin + count + padding, route_count, padding < padded - count
    )
    for start in range(0, padded // BLOCK_M, BLOCK):
        blocks = start + tl.arange(0, BLOCK)
        tl.store(
            expert_ids + begin // BLOCK_M + blocks,
            expert,
            blocks < padded // BLOCK_M,
        )


def align_humming_routes(
    topk_ids: torch.Tensor,
    *,
    num_local_experts: int,
    expert_start: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group flattened route IDs by local expert and pad to ``block_size``.

    ``topk_ids`` contains global expert IDs, shaped ``[tokens, top_k]``.
    Routes outside ``expert_start + [0, num_local_experts)`` are omitted,
    including negative sentinel IDs. The result consists of int32 sorted
    route IDs, local expert IDs per block, and a device scalar padded count.
    Padding contains ``topk_ids.numel()``. Only prefixes bounded by the
    returned count are initialized; Humming must not read the unused capacity.

    Humming cannot skip invalid expert IDs. Filtering before scheduling also
    avoids reserving one invalid-expert block for every remote route. The
    implementation scans each expert's routes in order, uses O(routes +
    experts * block_size) storage, and never synchronizes to the host.
    """
    if topk_ids.ndim != 2 or topk_ids.dtype not in {torch.int32, torch.int64}:
        raise TypeError("Humming routing requires integer [tokens, top_k] IDs")
    if not topk_ids.is_cuda or not topk_ids.is_contiguous():
        raise ValueError("Humming routing IDs must be contiguous CUDA tensors")
    if num_local_experts <= 0 or expert_start < 0:
        raise ValueError("Humming routing requires a nonempty contiguous expert range")
    if block_size <= 0 or block_size % 8:
        raise ValueError(
            "Humming routing block_size must be a positive multiple of eight"
        )
    routes = topk_ids.numel()
    capacity = min(routes * block_size, routes + num_local_experts * (block_size - 1))
    if capacity >= 2**31:
        raise ValueError("Humming routing exceeds its int32 indexed capacity")
    kwargs = {"dtype": torch.int32, "device": topk_ids.device}
    sorted_ids = torch.empty(capacity, **kwargs)
    expert_ids = torch.empty(triton.cdiv(capacity, block_size), **kwargs)
    padded_count = torch.zeros(1, **kwargs)
    if routes == 0:
        return sorted_ids, expert_ids, padded_count
    counts = torch.empty(num_local_experts, **kwargs)
    offsets = torch.empty_like(counts)
    _count_local_routes[(num_local_experts,)](
        topk_ids, counts, routes, expert_start, BLOCK=1024
    )
    _prefix_local_routes[(1,)](
        counts,
        offsets,
        padded_count,
        num_experts=num_local_experts,
        BLOCK_M=block_size,
        EXPERTS=triton.next_power_of_2(num_local_experts),
    )
    _write_local_routes[(num_local_experts,)](
        topk_ids,
        counts,
        offsets,
        sorted_ids,
        expert_ids,
        routes,
        expert_start,
        BLOCK_M=block_size,
        BLOCK=1024,
        PADDING=triton.next_power_of_2(block_size),
    )
    return sorted_ids, expert_ids, padded_count
