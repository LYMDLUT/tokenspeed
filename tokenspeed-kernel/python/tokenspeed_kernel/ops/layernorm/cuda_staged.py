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

"""Explicit RMSNorm candidate; deliberately absent from kernel registration."""

import math

import torch
from tokenspeed_kernel.thirdparty.vllm_staged_norm_activation import (
    check_bf16_rows,
    exact_alias,
    load_module,
    overlaps,
)


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply the public-source FP32 RMS reduction and one BF16 output cast.

    Args:
        x: CUDA BF16 rank 2..4 tensor with dense, nonoverlapping, 16-byte
            aligned rows; the final dimension must be divisible by eight.
        weight: Contiguous vector on x's device, one value per channel.
            BF16 follows the source dtype contract; FP32 is our extension.
        eps: Finite positive epsilon, passed to CUDA as FP32.
        out: Optional BF16 output with x's shape/device and the same row
            constraints. An exact alias of all x elements is allowed; partial
            overlap and any overlap with weight are rejected. None allocates.

    Returns:
        The output tensor. This entry has no residual or implicit dispatch.
        It is a candidate, not a claim of reference-binary equivalence.
    """
    check_bf16_rows(x, "x", vector_width=8)
    if (
        weight.device != x.device
        or weight.dtype not in (torch.bfloat16, torch.float32)
        or weight.ndim != 1
        or weight.numel() != x.shape[-1]
        or not weight.is_contiguous()
    ):
        raise ValueError(
            "weight must be a contiguous BF16/FP32 channel vector on x's device"
        )
    eps = float(eps)
    if not math.isfinite(eps) or not 0 < eps <= torch.finfo(torch.float32).max:
        raise ValueError("eps must be finite, positive, and representable in FP32")
    if out is None:
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    check_bf16_rows(out, "out", vector_width=8)
    if out.shape != x.shape or out.device != x.device:
        raise ValueError("out must match x's shape and device")
    if overlaps(x, out) and not exact_alias(x, out):
        raise ValueError("out may only overlap x as an exact full alias")
    if overlaps(weight, out):
        raise ValueError("out must not overlap weight")
    if x.numel():
        load_module().rmsnorm(x, weight, out, eps)
    return out
