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

"""Explicit BF16 SiLU candidate for unclamped dense/shared feed-forwards."""

import torch
from tokenspeed_kernel.thirdparty.vllm_staged_norm_activation import (
    check_bf16_rows,
    load_module,
    overlaps,
)


def silu_and_mul(x: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
    """Compute BF16(BF16(SiLU(gate)) * up) from packed gate/up rows.

    Args:
        x: CUDA BF16 rank 2..4 tensor with dense nonoverlapping rows and an
            even final dimension. The first half is gate; the second is up.
        out: Optional BF16 output of shape x.shape[:-1] + (x.shape[-1] // 2,)
            on x's device. Its rows must be dense and nonoverlapping. No
            input/output overlap is allowed. None allocates an output.

    Returns:
        The output tensor. The source formula uses FP32 expf/division, then
        BF16 activation rounding, FP32 multiplication, and BF16 output.
        No clamp, alpha/beta variation, quantization, or implicit dispatch.
    """
    check_bf16_rows(x, "x", vector_width=1)
    if x.shape[-1] % 2:
        raise ValueError("x must contain equally sized gate and up halves")
    shape = (*x.shape[:-1], x.shape[-1] // 2)
    if out is None:
        out = torch.empty(shape, dtype=x.dtype, device=x.device)
    check_bf16_rows(out, "out", vector_width=1)
    if out.shape != shape or out.device != x.device:
        raise ValueError("out must match the activation shape and x's device")
    if overlaps(x, out):
        raise ValueError("out must not overlap x")
    if x.numel():
        load_module().silu_and_mul(x, out)
    return out
