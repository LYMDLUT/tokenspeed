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

"""Explicit public vLLM v0.29 single-group CUDA sigmoid TopK candidate."""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures
from tokenspeed_kernel.thirdparty.cuda.public_v029_topk import single_group_sigmoid_topk


@register_kernel(
    "moe",
    "sigmoid_bias_topk",
    name="cuda_public_v029_sigmoid_bias_topk",
    solution="cuda",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=format_signatures("router_logits", "dense", {torch.float32}),
    priority=Priority.PORTABLE,
)
def cuda_public_v029_sigmoid_bias_topk(
    *,
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    routed_scaling_factor: float,
    normalize_topk_weights: bool,
    weights_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Route ungrouped FP32 logits using a pinned public v0.29 implementation.

    Args:
        router_logits: Contiguous CUDA FP32 ``[N, E]`` logits with ``E <= 1024``.
        correction_bias: Contiguous colocated FP32 ``[E]`` selection bias.
        topk: In ``[1, min(E, 22)]`` for E <= 512, or ``[1, 16]`` otherwise.
        routed_scaling_factor: Scale within the finite FP32 range. The public
            implementation rounds it to FP32 before optional normalization.
        normalize_topk_weights: Whether to normalize selected unbiased scores.
        weights_dtype: Explicitly requested floating output dtype.

    Returns:
        ``(weights, ids)`` with shape ``[N, topk]``. Weights honor the requested
        dtype and ids are INT32. This family is single-group only and retains
        the public implementation's tier-dependent warp/block arithmetic.
    """
    if (
        not isinstance(weights_dtype, torch.dtype)
        or not weights_dtype.is_floating_point
    ):
        raise TypeError("weights_dtype must be a floating torch.dtype")
    weights, ids = single_group_sigmoid_topk(
        router_logits,
        correction_bias,
        topk=topk,
        renormalize=normalize_topk_weights,
        routed_scaling_factor=routed_scaling_factor,
    )
    return weights.to(weights_dtype), ids
