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

"""Explicit group32 W8A8 experts using optional Humming Hopper GEMMs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tokenspeed_kernel.ops.moe._triton.bf16_epilogue import (
    clamped_swiglu_bf16,
    finalize_bf16_routes,
)
from tokenspeed_kernel.ops.moe._triton.humming_routing import align_humming_routes
from tokenspeed_kernel.ops.quantization import quantize_fp8
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures
from tokenspeed_kernel.thirdparty.humming import (
    HummingIndexedPlan,
    humming_indexed_gemm,
    prepare_humming_indexed,
    prepare_humming_mxfp8_weights,
)


@dataclass(frozen=True)
class _HummingMxfp8State:
    gate_up: HummingIndexedPlan
    down: HummingIndexedPlan
    expert_start: int
    limit: float | None


def humming_mxfp8_moe_weights(plan: dict, w: torch.nn.Module) -> None:
    """Repack E4M3/E8M0 experts and prepare every Humming launch interval.

    The loader provides canonical ``[E,2I,H]`` / ``[E,H,I]`` weights and
    ``[E,N,K/32]`` uint8 scale bytes. Repacking replaces these Parameters,
    releasing their original storage; no BF16 expert copy is retained.
    """
    if isinstance(w._moe_backend_state, _HummingMxfp8State):
        return
    if w._moe_backend_state is not None:
        raise ValueError("Humming cannot consume another backend's prepared weights")
    if plan["internal_activation_dtype"] != "fp8":
        raise ValueError("Humming MXFP8 requires explicit FP8 activation selection")
    if plan["combine_order"] != "rank":
        raise ValueError("Humming MXFP8 supports rank-local finalization only")
    if plan.get("a2a_backend") not in (None, "none"):
        raise ValueError("Humming MXFP8 does not implement all-to-all dispatch")
    if w.activation not in {"silu", "swiglu"}:
        raise ValueError("Humming MXFP8 supports standard SiLU/SwiGLU experts")
    if w.w13_input_layout != "concatenated":
        raise ValueError("Humming MXFP8 requires concatenated gate/up weights")
    alpha = None if w.swiglu_arg is None else w.swiglu_arg.alpha
    if alpha not in (None, 1.0) or w.swiglu_beta not in (None, 0.0):
        raise ValueError("Humming MXFP8 requires SwiGLU alpha=1 and beta=0")
    limit = None if w.swiglu_arg is None else w.swiglu_arg.limit
    if limit is not None and (not 0 < limit < float("inf")):
        raise ValueError("Humming MXFP8 requires a finite positive SwiGLU clamp")
    if w.w13_weight.ndim != 3 or w.w2_weight.ndim != 3:
        raise ValueError("Humming MXFP8 weights must have an expert axis")
    experts, two_intermediate, hidden = w.w13_weight.shape
    if two_intermediate % 2 or w.w2_weight.shape != (
        experts,
        hidden,
        two_intermediate // 2,
    ):
        raise ValueError("Humming gate/up and down weights disagree on their shapes")
    if experts != w.num_local_experts or not 0 <= w.ep_rank < w.ep_size:
        raise ValueError(
            "Humming weights must cover this rank's contiguous expert range"
        )
    gate_up = prepare_humming_indexed(
        prepare_humming_mxfp8_weights(w.w13_weight, w.w13_weight_scale_inv)
    )
    down = prepare_humming_indexed(
        prepare_humming_mxfp8_weights(w.w2_weight, w.w2_weight_scale_inv)
    )
    w.w13_weight = torch.nn.Parameter(gate_up.state.weight, requires_grad=False)
    w.w13_weight_scale_inv = torch.nn.Parameter(
        gate_up.state.scales, requires_grad=False
    )
    w.w2_weight = torch.nn.Parameter(down.state.weight, requires_grad=False)
    w.w2_weight_scale_inv = torch.nn.Parameter(down.state.scales, requires_grad=False)
    w._moe_backend_state = _HummingMxfp8State(
        gate_up=gate_up,
        down=down,
        expert_start=w.ep_rank * experts,
        limit=limit,
    )


def _quantize_group32(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return quantize_fp8(
        x,
        granularity="token_group",
        group_size=32,
        scale_encoding="float32",
        enable_pdl=False,
        required_features=frozenset({"power_of_two_scales"}),
        solution="triton",
    )


@register_kernel(
    "moe",
    "apply",
    name="humming_mxfp8_precomputed_moe_apply",
    solution="humming",
    weight_preprocessor=humming_mxfp8_moe_weights,
    capability=CapabilityRequirement(
        vendors=frozenset({"nvidia"}),
        min_arch_version=ArchVersion(9, 0),
        max_arch_version=ArchVersion(9, 0),
    ),
    signatures=format_signatures("x", "dense", {torch.bfloat16}),
    traits={
        "weight_dtype": frozenset({"fp8"}),
        "fp8_scale_block_shape": frozenset({(1, 32)}),
        "internal_activation_dtype": frozenset({"fp8"}),
        "activation": frozenset({"silu", "swiglu"}),
        "swiglu_form": frozenset({"standard"}),
        "activation_clamped": frozenset({False, True}),
        "expert_id_repeats": frozenset({False, True}),
        "routing_mode": frozenset({"precomputed_topk"}),
        "supports_deferred_finalize": frozenset({False}),
        "deterministic_finalize": frozenset({True}),
        "combine_order": frozenset({"rank"}),
        "supports_ep": frozenset({False, True}),
        "supports_all_to_all_ep": frozenset({False}),
        "ispp_alignment": frozenset({64}),
        "hidden_alignment": frozenset({64}),
        "supports_bias": frozenset({False}),
    },
    priority=Priority.PORTABLE,
)
def humming_mxfp8_precomputed_moe_apply(
    plan: dict,
    x: torch.Tensor,
    w: torch.nn.Module,
    router_logits: torch.Tensor | None,
    topk_weights: torch.Tensor | None = None,
    topk_ids: torch.Tensor | None = None,
    num_tokens_global: int | None = None,
    max_num_tokens_per_gpu: int | None = None,
    do_finalize: bool = True,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Apply explicitly quantized W8A8 experts and return a BF16 rank partial.

    Each GEMM accumulates FP32 without Stream-K and materializes BF16.
    Standard clamped SwiGLU also materializes BF16 before the second group32
    quantizer. Values and route weights round to BF16, their product rounds
    to BF16, and top-k slots sum in fixed FP32 order. Precomputed routing
    makes router logits and global-capacity hints unnecessary.

    The caller owns TP/EP reduction. DeepEP, deferred finalization, PDL, and
    batch-invariant numerics are unsupported and cannot select this path.
    """
    del router_logits, num_tokens_global, max_num_tokens_per_gpu
    if enable_pdl:
        raise ValueError("Humming MXFP8 requires PDL disabled")
    if not do_finalize or plan["combine_order"] != "rank":
        raise ValueError("Humming MXFP8 requires rank-local finalization")
    state = w._moe_backend_state
    if not isinstance(state, _HummingMxfp8State):
        raise ValueError("Humming MXFP8 weights have not been prepared")
    hidden = state.gate_up.state.size_k
    intermediate = state.down.state.size_k
    experts = state.gate_up.state.num_experts
    if (
        x.ndim != 2
        or x.shape[1] != hidden
        or x.dtype != torch.bfloat16
        or x.device != state.gate_up.state.device
    ):
        raise ValueError(
            "Humming MXFP8 input must be BF16 [tokens, hidden] on its device"
        )
    if (
        topk_ids is None
        or topk_weights is None
        or topk_ids.ndim != 2
        or topk_ids.shape != (x.shape[0], w.top_k)
        or topk_weights.shape != topk_ids.shape
        or topk_ids.dtype not in {torch.int32, torch.int64}
        or topk_weights.dtype != torch.float32
        or topk_ids.device != x.device
        or topk_weights.device != x.device
    ):
        raise ValueError(
            "Humming requires integer IDs and FP32 weights [tokens, top_k]"
        )
    if x.shape[0] == 0:
        return torch.empty_like(x)
    topk_ids = topk_ids.contiguous()
    routes = topk_ids.numel()
    first_block = state.gate_up.block_m(routes)
    first_routes = align_humming_routes(
        topk_ids,
        num_local_experts=experts,
        expert_start=state.expert_start,
        block_size=first_block,
    )
    codes, scales = _quantize_group32(x.contiguous())
    gate_up = torch.empty((routes, 2 * intermediate), device=x.device, dtype=x.dtype)
    humming_indexed_gemm(
        state.gate_up,
        codes,
        scales,
        gate_up,
        *first_routes,
        top_k=w.top_k,
        valid_shape_m=routes,
    )
    activated = clamped_swiglu_bf16(
        gate_up,
        topk_ids,
        expert_start=state.expert_start,
        num_local_experts=experts,
        limit=state.limit,
    )
    codes, scales = _quantize_group32(activated)
    second_block = state.down.block_m(routes)
    second_routes = first_routes
    if second_block != first_block:
        second_routes = align_humming_routes(
            topk_ids,
            num_local_experts=experts,
            expert_start=state.expert_start,
            block_size=second_block,
        )
    down = torch.empty((routes, hidden), device=x.device, dtype=x.dtype)
    humming_indexed_gemm(
        state.down,
        codes,
        scales,
        down,
        *second_routes,
        top_k=1,
        valid_shape_m=routes,
    )
    return finalize_bf16_routes(
        down,
        topk_weights,
        topk_ids,
        expert_start=state.expert_start,
        num_local_experts=experts,
        weighted_routes=False,
    )


def _humming_mxfp8_warmup(plan: dict, w: torch.nn.Module) -> None:
    state = w._moe_backend_state
    if not isinstance(state, _HummingMxfp8State):
        raise ValueError("Humming warmup requires prepared weights")
    maximum = plan["warmup_max_num_tokens"]
    counts = {1, 2, 16, 17, 18}
    for kernel_plan in (state.gate_up, state.down):
        counts.update(lower // w.top_k + 1 for lower, _, _ in kernel_plan.intervals)
    for tokens in sorted(count for count in counts if 0 < count <= maximum):
        x = torch.zeros(
            (tokens, state.gate_up.state.size_k),
            device=state.gate_up.state.device,
            dtype=torch.bfloat16,
        )
        ids = (
            torch.arange(tokens * w.top_k, device=x.device, dtype=torch.int32)
            % state.gate_up.state.num_experts
            + state.expert_start
        ).reshape(tokens, w.top_k)
        weights = torch.full(
            ids.shape, 1.0 / w.top_k, device=x.device, dtype=torch.float32
        )
        humming_mxfp8_precomputed_moe_apply(
            plan,
            x,
            w,
            None,
            topk_weights=weights,
            topk_ids=ids,
            do_finalize=True,
            enable_pdl=False,
        )


humming_mxfp8_precomputed_moe_apply._tokenspeed_warmup = _humming_mxfp8_warmup
