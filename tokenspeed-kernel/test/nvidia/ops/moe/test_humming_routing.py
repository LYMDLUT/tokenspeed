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

"""Exact local-expert routing contracts, independent of optional Humming."""

import pytest
import torch
from tokenspeed_kernel.ops.moe._triton import humming_routing as routing
from utils import assert_no_triton_compile

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="requires NVIDIA CUDA and TokenSpeed Triton",
)


def _reference(
    ids: torch.Tensor, local_experts: int, expert_start: int, block_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    flattened = ids.reshape(-1).tolist()
    sorted_ids = []
    expert_ids = []
    for local in range(local_experts):
        selected = [
            route
            for route, expert in enumerate(flattened)
            if expert == expert_start + local
        ]
        blocks = (len(selected) + block_size - 1) // block_size
        sorted_ids.extend(selected)
        sorted_ids.extend([len(flattened)] * (blocks * block_size - len(selected)))
        expert_ids.extend([local] * blocks)
    return torch.tensor(sorted_ids, dtype=torch.int32), torch.tensor(
        expert_ids, dtype=torch.int32
    )


def _check(
    actual: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ids: torch.Tensor,
    *,
    local_experts: int,
    expert_start: int,
    block_size: int,
) -> None:
    expected_routes, expected_experts = _reference(
        ids, local_experts, expert_start, block_size
    )
    sorted_ids, expert_ids, padded_count = actual
    count = padded_count.item()
    assert count == expected_routes.numel()
    assert count % block_size == 0
    assert sorted_ids.dtype == expert_ids.dtype == padded_count.dtype == torch.int32
    torch.testing.assert_close(
        sorted_ids[:count].cpu(), expected_routes, rtol=0, atol=0
    )
    torch.testing.assert_close(
        expert_ids[: count // block_size].cpu(), expected_experts, rtol=0, atol=0
    )


def _case(name: str, dtype: torch.dtype, expert_start: int) -> torch.Tensor:
    if name == "empty":
        return torch.empty((0, 3), dtype=dtype)
    if name == "all_nonlocal":
        return torch.tensor([[-1, expert_start - 1, expert_start + 3]] * 7, dtype=dtype)
    if name == "duplicates":
        return torch.tensor(
            [[expert_start + 2, expert_start, expert_start + 2]] * 19,
            dtype=dtype,
        )
    if name == "mixed_ep":
        return torch.tensor(
            [
                [expert_start + 2, -1, expert_start],
                [expert_start + 1, expert_start + 3, expert_start + 1],
                [expert_start - 1, expert_start + 2, expert_start],
            ],
            dtype=dtype,
        )
    assert name == "multiple_scan_blocks"
    return (torch.arange(513 * 3, dtype=dtype) % 7 + expert_start - 2).reshape(-1, 3)


@pytest.mark.parametrize("block_size", [8, 24, 48, 64, 80])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "case", ["empty", "all_nonlocal", "duplicates", "mixed_ep", "multiple_scan_blocks"]
)
def test_routing_matches_stable_cpu_reference(
    block_size: int, dtype: torch.dtype, case: str
) -> None:
    ids = _case(case, dtype, expert_start=17)
    actual = routing.align_humming_routes(
        ids.cuda(), num_local_experts=3, expert_start=17, block_size=block_size
    )
    _check(actual, ids, local_experts=3, expert_start=17, block_size=block_size)


def test_int64_expert_ids_are_not_truncated_to_local_int32_ids() -> None:
    ids = torch.tensor(
        [[4, 2**32 + 4, -(2**32) + 4], [2**63 - 1, -(2**63), 5]],
        dtype=torch.int64,
    )
    actual = routing.align_humming_routes(
        ids.cuda(), num_local_experts=3, expert_start=4, block_size=24
    )
    _check(actual, ids, local_experts=3, expert_start=4, block_size=24)


def test_graph_replay_refreshes_routes_and_zero_local_counts() -> None:
    cpu_ids = (torch.arange(19 * 3) % 9 + 6).reshape(19, 3).to(torch.int64)
    ids = cpu_ids.cuda()

    def run():
        return routing.align_humming_routes(
            ids, num_local_experts=5, expert_start=8, block_size=24
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    replacements = (
        cpu_ids,
        cpu_ids.flip(0).contiguous(),
        torch.full_like(cpu_ids, -1),
        torch.full_like(cpu_ids, 10),
    )
    for replacement in replacements:
        ids.copy_(replacement)
        graph.replay()
        _check(captured, replacement, local_experts=5, expert_start=8, block_size=24)


def test_dynamic_token_counts_do_not_compile_new_route_kernels() -> None:
    def run(tokens: int) -> None:
        ids = (torch.arange(tokens * 3) % 9 + 30).reshape(tokens, 3).to(torch.int32)
        actual = routing.align_humming_routes(
            ids.cuda(), num_local_experts=5, expert_start=32, block_size=24
        )
        _check(actual, ids, local_experts=5, expert_start=32, block_size=24)

    # route_count opts out of value and alignment specialization; one warmup
    # must cover both sides of the 1024-route scan-loop boundary.
    run(17)
    with assert_no_triton_compile(
        routing._count_local_routes,
        routing._prefix_local_routes,
        routing._write_local_routes,
    ):
        for tokens in (0, 1, 2, 16, 18, 31, 32, 33, 257, 513, 1025):
            run(tokens)
