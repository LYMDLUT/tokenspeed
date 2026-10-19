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

"""Explicitly selectable CUDA sigmoid TopK from a pinned public source."""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures
from tokenspeed_kernel.thirdparty.cuda.public_grouped_topk import grouped_sigmoid_topk


@register_kernel(
    "moe",
    "sigmoid_bias_topk",
    name="cuda_public_sigmoid_bias_topk",
    solution="cuda",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=format_signatures("router_logits", "dense", {torch.float32}),
    priority=Priority.PORTABLE,
)
def cuda_public_sigmoid_bias_topk(
    *,
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    routed_scaling_factor: float,
    normalize_topk_weights: bool,
    weights_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Route ungrouped FP32 logits through the public grouped CUDA kernel.

    Args:
        router_logits: Contiguous CUDA FP32 ``[N, E]`` logits.
        correction_bias: Contiguous colocated FP32 ``[E]`` selection bias.
        topk: Number of selected experts, in ``[1, min(E, 32)]``.
        routed_scaling_factor: Finite weight scale, applied after normalization.
        normalize_topk_weights: Whether to normalize selected unbiased scores.
        weights_dtype: Explicitly requested floating output dtype.

    Returns:
        ``(weights, ids)`` with shape ``[N, topk]``. Weights honor the requested
        dtype and ids are INT32. This registered family is explicitly ungrouped;
        the lower-level thirdparty API exposes arbitrary supported group counts.
    """
    if (
        not isinstance(weights_dtype, torch.dtype)
        or not weights_dtype.is_floating_point
    ):
        raise TypeError("weights_dtype must be a floating torch.dtype")
    weights, ids = grouped_sigmoid_topk(
        router_logits,
        correction_bias,
        num_expert_group=1,
        topk_group=1,
        topk=topk,
        renormalize=normalize_topk_weights,
        routed_scaling_factor=routed_scaling_factor,
    )
    return weights.to(weights_dtype), ids
