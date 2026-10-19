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

"""Stage rounding, local expert ownership, and replay for the BF16 reference."""

from __future__ import annotations

from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch
from tokenspeed_kernel.ops.moe import moe_plan
from tokenspeed_kernel.ops.moe.triton.bf16_staged import (
    _activation_kernel,
    _combine_kernel,
    _gemm1_kernel,
    _gemm2_kernel,
    _routing_kernel,
    _run_stages,
    triton_bf16_staged_moe_apply,
)
from utils import assert_no_triton_compile, kernel_supported

pytestmark = pytest.mark.skipif(
    not kernel_supported("triton_bf16_staged_moe_apply"),
    reason="requires a GPU supported by the staged BF16 kernel",
)


class _Stages(NamedTuple):
    gate_up: torch.Tensor
    activated: torch.Tensor
    weighted: torch.Tensor
    output: torch.Tensor


class _Weights(torch.nn.Module):
    def __init__(
        self,
        w13: torch.Tensor,
        w2: torch.Tensor,
        top_k: int,
        *,
        ep_size: int,
        ep_rank: int,
        clamp_limit: float | None,
    ) -> None:
        super().__init__()
        self.w13_weight = torch.nn.Parameter(w13, requires_grad=False)
        self.w2_weight = torch.nn.Parameter(w2, requires_grad=False)
        self.top_k = top_k
        self.num_local_experts = w13.shape[0]
        self.num_experts = self.num_local_experts * ep_size
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.tp_size = 1
        self.activation = "silu" if clamp_limit is None else "swiglu"
        self.swiglu_arg = (
            None
            if clamp_limit is None
            else SimpleNamespace(alpha=None, limit=clamp_limit)
        )
        self.swiglu_beta = 0.0
        self.w13_input_layout = "concatenated"


def _plan(weights: _Weights) -> dict:
    plan = moe_plan(
        "unquant",
        input_dtype=torch.bfloat16,
        activation=weights.activation,
        routing_mode="precomputed_topk",
        a2a_backend=None,
        ep_size=weights.ep_size,
        hidden=weights.w13_weight.shape[2],
        ispp=weights.w2_weight.shape[2],
        swiglu_form="standard" if weights.activation == "swiglu" else None,
        activation_clamped=weights.swiglu_arg is not None,
        expert_id_repeats=True,
        internal_activation_dtype="input",
        requires_deferred_finalize=False,
        requires_deterministic_finalize=True,
        fast_math=False,
        combine_order="rank",
        solution="triton_staged",
    )
    assert plan["apply_kernel_name"] == "triton_bf16_staged_moe_apply"
    return plan


def _apply(
    plan: dict,
    x: torch.Tensor,
    weights: _Weights,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    return triton_bf16_staged_moe_apply(
        plan,
        x,
        weights,
        None,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        num_tokens_global=None,
        max_num_tokens_per_gpu=None,
        do_finalize=True,
        enable_pdl=False,
    )


def _oracle(
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    expert_offset: int,
    clamp_limit: float | None,
) -> _Stages:
    """FP64 CPU math with explicit BF16 stores and an FP32 slot left fold.

    Random cases use tolerances: FP64 exp and accumulation are independent
    checks, not a claim of bitwise agreement with the GPU's FP32 operations.
    The sparse dyadic case keeps the dot products exactly representable.
    """
    x64, w1364, w264 = (value.detach().cpu().double() for value in (x, w13, w2))
    ids, route_weights = topk_ids.cpu(), topk_weights.cpu()
    tokens, hidden = x.shape
    top_k = ids.shape[1]
    intermediate = w2.shape[2]
    gate_up = torch.zeros(tokens * top_k, 2 * intermediate, dtype=torch.bfloat16)
    activated = torch.zeros(tokens * top_k, intermediate, dtype=torch.bfloat16)
    weighted = torch.zeros(tokens * top_k, hidden, dtype=torch.bfloat16)
    output = torch.zeros(tokens, hidden, dtype=torch.float32)
    for token in range(tokens):
        for slot in range(top_k):
            expert = int(ids[token, slot]) - expert_offset
            row = token * top_k + slot
            if 0 <= expert < w13.shape[0]:
                gate_up[row] = (x64[token] @ w1364[expert].T).to(torch.bfloat16)
                gate, up = gate_up[row].double().chunk(2)
                if clamp_limit is not None:
                    gate = gate.clamp(max=clamp_limit).to(torch.bfloat16).double()
                    up = up.clamp(-clamp_limit, clamp_limit).to(torch.bfloat16).double()
                silu = (gate / (1.0 + torch.exp(-gate))).to(torch.bfloat16)
                activated[row] = (silu.double() * up).to(torch.bfloat16)
                gemm2 = activated[row].double() @ w264[expert].T
                weighted[row] = (gemm2 * route_weights[token, slot].double()).to(
                    torch.bfloat16
                )
            # Do not use torch.sum: its reduction tree need not match slots.
            output[token] = output[token] + weighted[row].float()
    return _Stages(gate_up, activated, weighted, output.to(torch.bfloat16))


def _assert_bits(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.dtype == expected.dtype == torch.bfloat16
    assert actual.shape == expected.shape
    actual_cpu = actual.detach().cpu()
    assert torch.isfinite(actual_cpu).all()
    assert torch.equal(actual_cpu.view(torch.int16), expected.cpu().view(torch.int16))


def _assert_close_stages(actual: _Stages, expected: _Stages) -> None:
    for name, got, want in zip(_Stages._fields, actual, expected, strict=True):
        assert torch.isfinite(got).all(), name
        torch.testing.assert_close(
            got.cpu(), want, rtol=8e-3, atol=5e-5, msg=f"stage: {name}"
        )


def _random_inputs(
    tokens: int, experts: int, hidden: int, intermediate: int, top_k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(23)

    def bf16(*shape: int) -> torch.Tensor:
        return (torch.randn(shape, device="cuda", generator=generator) * 0.25).to(
            torch.bfloat16
        )

    x = bf16(tokens, hidden)
    w13 = bf16(experts, 2 * intermediate, hidden)
    w2 = bf16(experts, hidden, intermediate)
    ids = (
        (
            torch.arange(tokens, device="cuda")[:, None]
            + torch.arange(top_k, device="cuda")[None, :]
        )
        .remainder(experts)
        .to(torch.int32)
    )
    weights = torch.full((tokens, top_k), 1.0 / top_k, device="cuda")
    return x, w13, w2, weights, ids


@pytest.mark.parametrize("clamp_limit", [None, 0.1])
def test_sparse_stages_preserve_each_rounding_boundary(
    clamp_limit: float | None,
) -> None:
    # Non-aligned H/I also exercise nonzero values in both dimension tails.
    hidden, intermediate, experts = 19, 17, 4
    x = torch.zeros(2, hidden, dtype=torch.bfloat16, device="cuda")
    x[:, 0], x[:, 1], x[:, -1] = 1.0, 2**-8, 1.0
    w13 = torch.zeros(
        experts, 2 * intermediate, hidden, dtype=torch.bfloat16, device="cuda"
    )
    w13[:, 0, :2] = 1.0  # G1: 1 + 2^-8 rounds to 1, ties to even.
    w13[:, 1, 0] = 1.0
    w13[:, 2:4, 0] = 8.0
    w13[:, 4, 0], w13[:, 5, 0] = -8.0, 8.0
    w13[:, intermediate - 1, 0] = 0.75
    w13[:, intermediate - 1, -1] = 0.25
    for column, up in enumerate((1.0, 1.0 + 2**-7, 2**-3, 2**-11, 8.0, -8.0)):
        w13[:, intermediate + column, 0] = up
    w13[:, -1, 0] = 1.0
    w2 = torch.zeros(experts, hidden, intermediate, dtype=torch.bfloat16, device="cuda")
    w2[:, 0, 0], w2[:, 1, 1], w2[:, 2, 2:4], w2[:, -1, -1] = 1.0, 1.0, 1.0, 1.0
    w2[:, 3, 2] = torch.tensor([2**32, 4, -(2**32), 4], device="cuda")
    ids = torch.arange(experts, device="cuda", dtype=torch.int64).repeat(2, 1)
    weights = torch.tensor([[0.75, 0.25, 0.0, 0.0], [0.25] * 4], device="cuda")
    actual = _run_stages(
        x, w13, w2, weights, ids, expert_offset=0, clamp_limit=clamp_limit
    )
    expected = _oracle(
        x, w13, w2, weights, ids, expert_offset=0, clamp_limit=clamp_limit
    )
    for got, want in zip(actual, expected, strict=True):
        _assert_bits(got, want)
    if clamp_limit is None:
        assert expected.gate_up[0, 0].item() == 1.0
        assert expected.activated[0, 0].item() == 0.73046875
        # Rounding only after silu(gate) * up would instead give 0.73828125.
        assert expected.activated[0, 1].item() == 0.734375
        # Rounding G2 before its route weight would instead give 0.75.
        assert expected.weighted[0, 2].item() == 0.75390625
        # [2^30, 1, -2^30, 1] left-folds to 1; a paired tree gives 0.
        assert expected.output[1, 3].item() == 1.0


@pytest.mark.parametrize("clamp_limit", [None, 1.5])
def test_random_stages_match_fp64_oracle_and_repeat_exactly(
    clamp_limit: float | None,
) -> None:
    inputs = _random_inputs(7, 3, 35, 21, 3)
    expected = _oracle(*inputs, expert_offset=0, clamp_limit=clamp_limit)
    first = _run_stages(*inputs, expert_offset=0, clamp_limit=clamp_limit)
    _assert_close_stages(first, expected)
    for _ in range(3):
        repeated = _run_stages(*inputs, expert_offset=0, clamp_limit=clamp_limit)
        for got, want in zip(repeated, first, strict=True):
            _assert_bits(got, want)


def test_ep_offsets_repeated_routes_and_int64_invalid_ids() -> None:
    x, w13, w2, weights, _ = _random_inputs(3, 6, 35, 21, 4)
    ids = torch.tensor(
        [[0, 2, 2, 2**32 + 2], [-1, 3, 0, 6], [2**63 - 1, 1, -2, 2**32]],
        device="cuda",
        dtype=torch.int64,
    )
    for rank in range(3):
        local13, local2 = w13[rank * 2 : rank * 2 + 2], w2[rank * 2 : rank * 2 + 2]
        actual = _run_stages(
            x, local13, local2, weights, ids, expert_offset=rank * 2, clamp_limit=1.5
        )
        expected = _oracle(
            x, local13, local2, weights, ids, expert_offset=rank * 2, clamp_limit=1.5
        )
        _assert_close_stages(actual, expected)
        nonlocal_rows = ((ids < rank * 2) | (ids >= rank * 2 + 2)).flatten().cpu()
        for stage in actual[:3]:
            assert torch.count_nonzero(stage.cpu()[nonlocal_rows]).item() == 0
        module = _Weights(local13, local2, 4, ep_size=3, ep_rank=rank, clamp_limit=1.5)
        _assert_bits(_apply(_plan(module), x, module, weights, ids), actual.output)
        if rank == 2:  # This rank owns experts, but receives no valid routes.
            _assert_bits(actual.output, torch.zeros_like(expected.output))


def test_public_apply_graph_replay_reads_updated_inputs_and_routes() -> None:
    x, w13, w2, weights, ids = _random_inputs(5, 4, 35, 21, 3)
    module = _Weights(w13, w2, 3, ep_size=1, ep_rank=0, clamp_limit=1.5)
    plan = _plan(module)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _apply(plan, x, module, weights, ids)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _apply(plan, x, module, weights, ids)
    graph.replay()
    _assert_bits(captured, _apply(plan, x, module, weights, ids))
    before = captured.clone()
    x.mul_(-2.0)
    ids.copy_((ids + 1).remainder(4))
    ids[0, 0] = -1
    weights.copy_(torch.tensor([0.5, 0.25, 0.25], device="cuda").expand_as(weights))
    graph.replay()
    _assert_bits(captured, _apply(plan, x, module, weights, ids))
    assert not torch.equal(captured, before)
    expected = _oracle(x, w13, w2, weights, ids, expert_offset=0, clamp_limit=1.5)
    torch.testing.assert_close(captured.cpu(), expected.output, rtol=8e-3, atol=5e-5)
    first_replay = captured.clone()
    graph.replay()
    _assert_bits(captured, first_replay)


def test_batch_size_changes_do_not_compile_or_change_token_outputs() -> None:
    x, w13, w2, weights, ids = _random_inputs(257, 3, 35, 21, 3)
    module = _Weights(w13, w2, 3, ep_size=1, ep_rank=0, clamp_limit=None)
    plan = _plan(module)

    def run(tokens: int) -> torch.Tensor:
        return _apply(plan, x[:tokens], module, weights[:tokens], ids[:tokens])

    # Only one batch size is warmed: runtime counts must not specialize on
    # the usual one/divisible-by-16/other scalar classes or route-count tails.
    run(1)
    with assert_no_triton_compile(
        _routing_kernel,
        _gemm1_kernel,
        _activation_kernel,
        _gemm2_kernel,
        _combine_kernel,
    ):
        full = run(257)
        for tokens in (0, 2, 7, 16, 31, 64, 129, 255):
            _assert_bits(run(tokens), full[:tokens])
