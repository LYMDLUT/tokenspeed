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

"""BF16 expert MLPs with explicit rounding at each materialized stage.

Import this module and select ``solution="triton_staged"`` explicitly. Gate/up
GEMM and clamped gate/up results round to BF16. FP32 SiLU rounds to BF16 before
the up product rounds to BF16. GEMM2 accumulates in FP32, applies the route weight,
then rounds to BF16. A fixed slot-order FP32 left fold rounds once to BF16.
The fold is deterministic, but need not match another framework's reduction
tree bit for bit.

Contiguous expert parallel shards return rank-local partials: the caller owns
cross-rank communication and reduction. Invalid and nonlocal expert IDs
contribute zero. There is no all-to-all, tensor parallelism, or implicit routing.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
from tokenspeed_kernel._triton import libdevice, tl, triton
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


class StagedOutputs(NamedTuple):
    gate_up: torch.Tensor
    activated: torch.Tensor
    weighted: torch.Tensor
    output: torch.Tensor


@triton.jit(do_not_specialize=["num_routes"])
def _routing_kernel(
    ids_ptr,
    routes_ptr,
    counts_ptr,
    num_routes,
    EXPERT_OFFSET: tl.constexpr,
    BLOCK: tl.constexpr,
):
    expert = tl.program_id(0)
    count = 0
    for start in range(0, num_routes, BLOCK):
        route = start + tl.arange(0, BLOCK)
        # Compare before any narrowing: a large int64 ID must not wrap to local.
        selected = tl.load(ids_ptr + route, route < num_routes, other=-1)
        matches = (route < num_routes) & (selected == expert + EXPERT_OFFSET)
        position = count + tl.cumsum(matches.to(tl.int32), axis=0) - 1
        tl.store(
            routes_ptr + expert.to(tl.int64) * num_routes + position,
            route,
            mask=matches,
        )
        count += tl.sum(matches.to(tl.int32), axis=0)
    tl.store(counts_ptr + expert, count)


@triton.jit(do_not_specialize=["num_tokens"])
def _gemm1_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    routes_ptr,
    counts_ptr,
    num_tokens,
    H: tl.constexpr,
    INTERMEDIATE: tl.constexpr,
    E: tl.constexpr,
    TOP_K: tl.constexpr,
    PROGRAMS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    num_routes = num_tokens * TOP_K
    tile = tl.program_id(0)
    problem_start = 0
    for expert in range(E):
        count = tl.load(counts_ptr + expert)
        columns = tl.cdiv(2 * INTERMEDIATE, BN)
        tiles = tl.cdiv(count, BM) * columns
        while (tile >= problem_start) & (tile < problem_start + tiles):
            local_tile = tile - problem_start
            row = local_tile // columns * BM + tl.arange(0, BM)
            col = local_tile % columns * BN + tl.arange(0, BN)
            route = tl.load(
                routes_ptr + expert.to(tl.int64) * num_routes + row,
                row < count,
                other=0,
            )
            token = (route // TOP_K).to(tl.int64)
            acc = tl.zeros((BM, BN), tl.float32)
            for start in range(0, H, BK):
                k = start + tl.arange(0, BK)
                x = tl.load(
                    x_ptr + token[:, None] * H + k[None, :],
                    (row[:, None] < count) & (k[None, :] < H),
                    other=0,
                )
                weight = tl.load(
                    w_ptr
                    + expert.to(tl.int64) * (2 * INTERMEDIATE * H)
                    + col[None, :].to(tl.int64) * H
                    + k[:, None],
                    (k[:, None] < H) & (col[None, :] < 2 * INTERMEDIATE),
                    other=0,
                )
                acc = tl.dot(x, weight, acc)
            tl.store(
                out_ptr
                + route[:, None].to(tl.int64) * (2 * INTERMEDIATE)
                + col[None, :],
                acc.to(tl.bfloat16),
                (row[:, None] < count) & (col[None, :] < 2 * INTERMEDIATE),
            )
            tile += PROGRAMS
        problem_start += tiles


@triton.jit(do_not_specialize=["num_routes"])
def _activation_kernel(
    gate_up_ptr,
    out_ptr,
    num_routes,
    INTERMEDIATE: tl.constexpr,
    HAS_LIMIT: tl.constexpr,
    LIMIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    row, col = offset // INTERMEDIATE, offset % INTERMEDIATE
    mask = row < num_routes
    gate = tl.load(gate_up_ptr + row * (2 * INTERMEDIATE) + col, mask, other=0).to(
        tl.float32
    )
    up = tl.load(
        gate_up_ptr + row * (2 * INTERMEDIATE) + INTERMEDIATE + col, mask, other=0
    ).to(tl.float32)
    if HAS_LIMIT:
        gate = tl.minimum(gate, LIMIT)
        up = tl.minimum(tl.maximum(up, -LIMIT), LIMIT)
    gate = gate.to(tl.bfloat16).to(tl.float32)
    up = up.to(tl.bfloat16).to(tl.float32)
    activated = tl.div_rn(gate, 1.0 + libdevice.exp(-gate))
    activated = activated.to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + offset, (activated * up).to(tl.bfloat16), mask)


@triton.jit(do_not_specialize=["num_tokens"])
def _gemm2_kernel(
    activated_ptr,
    w_ptr,
    weights_ptr,
    out_ptr,
    routes_ptr,
    counts_ptr,
    num_tokens,
    H: tl.constexpr,
    INTERMEDIATE: tl.constexpr,
    E: tl.constexpr,
    TOP_K: tl.constexpr,
    PROGRAMS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    num_routes = num_tokens * TOP_K
    tile = tl.program_id(0)
    problem_start = 0
    for expert in range(E):
        count = tl.load(counts_ptr + expert)
        columns = tl.cdiv(H, BN)
        tiles = tl.cdiv(count, BM) * columns
        while (tile >= problem_start) & (tile < problem_start + tiles):
            local_tile = tile - problem_start
            row = local_tile // columns * BM + tl.arange(0, BM)
            col = local_tile % columns * BN + tl.arange(0, BN)
            route = tl.load(
                routes_ptr + expert.to(tl.int64) * num_routes + row,
                row < count,
                other=0,
            )
            acc = tl.zeros((BM, BN), tl.float32)
            for start in range(0, INTERMEDIATE, BK):
                k = start + tl.arange(0, BK)
                x = tl.load(
                    activated_ptr
                    + route[:, None].to(tl.int64) * INTERMEDIATE
                    + k[None, :],
                    (row[:, None] < count) & (k[None, :] < INTERMEDIATE),
                    other=0,
                )
                weight = tl.load(
                    w_ptr
                    + expert.to(tl.int64) * (H * INTERMEDIATE)
                    + col[None, :].to(tl.int64) * INTERMEDIATE
                    + k[:, None],
                    (k[:, None] < INTERMEDIATE) & (col[None, :] < H),
                    other=0,
                )
                acc = tl.dot(x, weight, acc)
            scale = tl.load(weights_ptr + route, row < count, other=0)
            weighted = (acc * scale[:, None]).to(tl.bfloat16)
            tl.store(
                out_ptr + route[:, None].to(tl.int64) * H + col[None, :],
                weighted,
                (row[:, None] < count) & (col[None, :] < H),
            )
            tile += PROGRAMS
        problem_start += tiles


@triton.jit(do_not_specialize=["num_tokens"])
def _combine_kernel(
    weighted_ptr,
    out_ptr,
    num_tokens,
    H: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    token, col = offset // H, offset % H
    mask = token < num_tokens
    acc = tl.full((BLOCK,), 0, tl.float32)
    for slot in range(TOP_K):
        value = tl.load(weighted_ptr + (token * TOP_K + slot) * H + col, mask, other=0)
        acc = acc + value.to(tl.float32)
    tl.store(out_ptr + offset, acc.to(tl.bfloat16), mask)


def _run_stages(
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    expert_offset: int,
    clamp_limit: float | None,
) -> StagedOutputs:
    """Run the validated stages; expose materialized boundaries for tests."""
    tokens, hidden = x.shape
    experts, twice_intermediate, _ = w13.shape
    intermediate = twice_intermediate // 2
    top_k = topk_ids.shape[1]
    routes = tokens * top_k
    # Zero canonical rows that belong to another rank or an invalid expert.
    gate_up = torch.zeros((routes, 2 * intermediate), device=x.device, dtype=x.dtype)
    activated = torch.empty((routes, intermediate), device=x.device, dtype=x.dtype)
    weighted = torch.zeros((routes, hidden), device=x.device, dtype=x.dtype)
    output = torch.empty_like(x)
    stages = StagedOutputs(gate_up, activated, weighted, output)
    if tokens == 0:
        return stages
    route_ids = torch.empty((experts, routes), device=x.device, dtype=torch.int32)
    counts = torch.empty((experts,), device=x.device, dtype=torch.int32)
    # Fixed for a device, never chosen from the batch's route count.
    programs = torch.cuda.get_device_properties(x.device).multi_processor_count
    _routing_kernel[(experts,)](
        topk_ids,
        route_ids,
        counts,
        routes,
        EXPERT_OFFSET=expert_offset,
        BLOCK=128,
        num_warps=4,
    )
    _gemm1_kernel[(programs,)](
        x,
        w13,
        gate_up,
        route_ids,
        counts,
        tokens,
        H=hidden,
        INTERMEDIATE=intermediate,
        E=experts,
        TOP_K=top_k,
        PROGRAMS=programs,
        BM=16,
        BN=64,
        BK=32,
        num_warps=4,
        num_stages=2,
        enable_fp_fusion=False,
    )
    _activation_kernel[(triton.cdiv(routes * intermediate, 256),)](
        gate_up,
        activated,
        routes,
        INTERMEDIATE=intermediate,
        HAS_LIMIT=clamp_limit is not None,
        LIMIT=0.0 if clamp_limit is None else clamp_limit,
        BLOCK=256,
        num_warps=4,
        enable_fp_fusion=False,
    )
    _gemm2_kernel[(programs,)](
        activated,
        w2,
        topk_weights,
        weighted,
        route_ids,
        counts,
        tokens,
        H=hidden,
        INTERMEDIATE=intermediate,
        E=experts,
        TOP_K=top_k,
        PROGRAMS=programs,
        BM=16,
        BN=64,
        BK=32,
        num_warps=4,
        num_stages=2,
        enable_fp_fusion=False,
    )
    _combine_kernel[(triton.cdiv(tokens * hidden, 256),)](
        weighted,
        output,
        tokens,
        H=hidden,
        TOP_K=top_k,
        BLOCK=256,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return stages


def _validate(
    plan: dict,
    x: torch.Tensor,
    w: torch.nn.Module,
    topk_weights: torch.Tensor | None,
    topk_ids: torch.Tensor | None,
    *,
    do_finalize: bool,
) -> tuple[int, float | None]:
    if not do_finalize:
        raise ValueError("staged MoE does not support deferred finalization")
    if plan["weight_dtype"] != "unquant":
        raise ValueError("staged MoE requires unquantized weights")
    if plan["a2a_backend"] not in {None, "none"} or plan["supports_all_to_all_ep"]:
        raise ValueError("staged MoE does not support all-to-all")
    if plan["combine_order"] != "rank":
        raise ValueError("staged MoE requires host rank-order combination")
    if plan["internal_activation_dtype"] not in {None, "input"}:
        raise ValueError("staged MoE requires BF16 intermediate storage")
    if (
        plan["activation"] not in {"silu", "swiglu"}
        or plan["activation"] != w.activation
    ):
        raise ValueError("staged MoE requires standard SiLU/SwiGLU")
    if w.tp_size != 1:
        raise ValueError("staged MoE requires tensor parallel size 1")
    if w.ep_size < 1 or not 0 <= w.ep_rank < w.ep_size:
        raise ValueError("staged MoE requires a valid expert parallel rank")
    if w.swiglu_beta not in {None, 0.0}:
        raise ValueError("staged MoE requires SwiGLU beta 0")
    limit = None
    if w.swiglu_arg is not None:
        if w.swiglu_arg.alpha not in {None, 1.0}:
            raise ValueError("staged MoE requires SwiGLU alpha 1")
        limit = w.swiglu_arg.limit
        if limit is not None and (not math.isfinite(limit) or limit < 0):
            raise ValueError("SwiGLU clamp limit must be finite and nonnegative")
    if w.w13_input_layout != "concatenated":
        raise ValueError("staged MoE requires concatenated [gate, up] weights")
    if any(
        w._parameters.get(name) is not None
        or w._buffers.get(name) is not None
        or vars(w).get(name) is not None
        for name in ("w13_weight_bias", "w2_weight_bias")
    ):
        raise ValueError("staged MoE does not support expert bias")
    if topk_weights is None or topk_ids is None:
        raise ValueError("staged MoE requires precomputed routing")
    w13, w2 = w.w13_weight, w.w2_weight
    if x.ndim != 2 or w13.ndim != 3 or w2.ndim != 3:
        raise ValueError("staged MoE requires rank-2 input and rank-3 weights")
    if any(t.dtype != torch.bfloat16 for t in (x, w13, w2)):
        raise TypeError("staged MoE input and expert weights must be BF16")
    if topk_weights.dtype != torch.float32 or topk_ids.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise TypeError("routing requires FP32 weights and int32/int64 IDs")
    tensors = (x, w13, w2, topk_weights, topk_ids)
    if any(
        not t.is_cuda or t.device != x.device or not t.is_contiguous() for t in tensors
    ):
        raise ValueError("staged MoE requires contiguous tensors on one GPU")
    experts, twice_intermediate, hidden = w13.shape
    intermediate = twice_intermediate // 2
    if experts < 1 or hidden < 1 or intermediate < 1 or twice_intermediate % 2:
        raise ValueError("staged MoE requires positive compatible weight dimensions")
    if x.shape[1] != hidden or w2.shape != (experts, hidden, intermediate):
        raise ValueError("staged MoE input and expert weight shapes disagree")
    if experts != w.num_local_experts or w.num_experts != experts * w.ep_size:
        raise ValueError("staged MoE requires equal contiguous expert shards")
    if topk_ids.ndim != 2 or topk_ids.shape != (x.shape[0], w.top_k) or w.top_k < 1:
        raise ValueError("top-k IDs must have shape [tokens, top_k > 0]")
    if topk_weights.shape != topk_ids.shape:
        raise ValueError("top-k weights and IDs must have identical shapes")
    if topk_ids.numel() >= 2**31:
        raise ValueError("staged MoE route count must fit int32")
    return w.ep_rank * experts, limit


@register_kernel(
    "moe",
    "apply",
    name="triton_bf16_staged_moe_apply",
    solution="triton_staged",
    signatures=format_signatures("x", "dense", {torch.bfloat16}),
    traits={
        "weight_dtype": frozenset({"unquant"}),
        "activation": frozenset({"silu", "swiglu"}),
        "activation_clamped": frozenset({False, True}),
        "swiglu_form": frozenset({"standard"}),
        "routing_mode": frozenset({"precomputed_topk"}),
        "supports_deferred_finalize": frozenset({False}),
        "deterministic_finalize": frozenset({True}),
        "supports_ep": frozenset({True}),
        "supports_all_to_all_ep": frozenset({False}),
        "expert_id_repeats": frozenset({False, True}),
        "combine_order": frozenset({"rank"}),
        "internal_activation_dtype": frozenset({"input"}),
        "supports_bias": frozenset({False}),
    },
    priority=Priority.REFERENCE,
)
def triton_bf16_staged_moe_apply(
    plan: dict,
    x: torch.Tensor,
    w: torch.nn.Module,
    router_logits: torch.Tensor | None,
    *,
    topk_weights: torch.Tensor | None,
    topk_ids: torch.Tensor | None,
    num_tokens_global: int | None,
    max_num_tokens_per_gpu: int | None,
    do_finalize: bool,
    enable_pdl: bool,
) -> torch.Tensor:
    """Apply staged BF16 MoE and return this rank's BF16 ``[tokens, hidden]``.

    Args:
        plan: Unquantized, rank-combine MoE plan for ``triton_staged`` with
            explicit SiLU/SwiGLU activation and no all-to-all backend.
        x: Contiguous BF16 hidden states ``[tokens, hidden]`` on GPU.
        w: Expert module with canonical BF16 ``w13_weight[local_E, 2I, H]``
            and ``w2_weight[local_E, H, INTERMEDIATE]``; declares activation, swiglu_arg,
            swiglu_beta, w13_input_layout, top_k, num_experts,
            num_local_experts, ep_size, ep_rank and tp_size (must be 1).
        router_logits: Unused because both routing tensors are required.
        topk_weights: Contiguous FP32 weights ``[tokens, top_k]`` on x's GPU.
        topk_ids: Matching int32/int64 global expert IDs; duplicates are allowed.
            IDs outside this rank's contiguous shard contribute zero.
        num_tokens_global: Optional nonnegative scheduling hint. Every rank
            must already receive the same routed input rows; no communication
            is performed here and the hint does not change those rows.
        max_num_tokens_per_gpu: Optional nonnegative capacity hint; unused by
            this dynamically allocated implementation.
        do_finalize: Must be true; finalizes the rank-local routed partial.
        enable_pdl: Launch hint; accepted but this portable leaf does not use PDL.

    Returns:
        BF16 rank-local partial. The caller performs any EP rank reduction.
    """
    for hint in (num_tokens_global, max_num_tokens_per_gpu):
        if hint is not None and hint < 0:
            raise ValueError("token-count hints must be nonnegative")
    expert_offset, limit = _validate(
        plan, x, w, topk_weights, topk_ids, do_finalize=do_finalize
    )
    return _run_stages(
        x,
        w.w13_weight,
        w.w2_weight,
        topk_weights,
        topk_ids,
        expert_offset=expert_offset,
        clamp_limit=limit,
    ).output
