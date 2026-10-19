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

"""Single-group sigmoid TopK from a pinned public vLLM v0.29.0 source."""

from __future__ import annotations

import hashlib
import inspect
import math
from functools import cache
from numbers import Real
from pathlib import Path

import torch

_UPSTREAM_COMMIT = "98dff2a81d747d1dba01a47f939f48c3526d4206"
_CXX_FLAGS = ("-std=c++20",)
_CUDA_FLAGS = (
    "-std=c++20",
    "--ftz=false",
    "--prec-div=true",
    "--prec-sqrt=true",
    "--fmad=true",
)
_SIGMOID_TIERS = (
    (128, 8),
    (256, 8),
    (384, 8),
    (512, 8),
    (512, 22),
    (768, 16),
    (896, 16),
    (1024, 16),
)
_MAX_TOKENS = 2**31 - 1 - 32768
_FLOAT32_MAX = torch.finfo(torch.float32).max


@cache
def _load_public_v029_topk_module():
    """Build once outside capture with the declared public-source policy."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Warm public v0.29 TopK before CUDA graph capture")
    from flashinfer.jit.core import gen_jit_spec

    if "use_fast_math" not in inspect.signature(gen_jit_spec).parameters:
        raise RuntimeError(
            "public v0.29 TopK requires FlashInfer gen_jit_spec with "
            "use_fast_math support"
        )
    root = Path(__file__).resolve().parent / "csrc"
    source = root / "public_v029_topk.cu"
    digest = hashlib.sha256(_UPSTREAM_COMMIT.encode())
    for path in (
        source,
        root / "public_v029_moeTopKFuncs.cuh",
        root / "tvm_ffi_utils.h",
    ):
        digest.update(path.read_bytes())
    digest.update("\n".join((*_CXX_FLAGS, *_CUDA_FLAGS)).encode())
    name = f"tokenspeed_public_v029_topk_{digest.hexdigest()[:16]}"
    return gen_jit_spec(
        name,
        [source],
        extra_include_paths=[root],
        extra_cflags=list(_CXX_FLAGS),
        extra_cuda_cflags=list(_CUDA_FLAGS),
        use_fast_math=False,
    ).build_and_load()


def single_group_sigmoid_topk(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    *,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the public v0.29 single-group sigmoid routing implementation.

    Args:
        router_logits: Contiguous CUDA FP32 ``[N, E]`` logits, with
            ``N <= INT32_MAX-32768`` and ``1 <= E <= 1024``.
        correction_bias: Contiguous colocated FP32 ``[E]`` selection bias.
            Finite logits and bias are the input value contract; this wrapper
            does not scan GPU values in the hot path.
        topk: Selected expert count. For ``E <= 512``, require
            ``1 <= topk <= min(E, 22)``. For larger E, require ``topk <= 16``.
        renormalize: Whether to normalize the selected unbiased scores.
        routed_scaling_factor: Real scalar within the finite FP32 range. The
            public launcher rounds this scalar to FP32 before weight arithmetic.

    Returns:
        Contiguous ``(weights, ids)`` tensors with shape ``[N, topk]``, FP32
        weights and INT32 ids, ordered by descending biased selection score.
        Score ties prefer smaller expert ids.

    The complete upstream tier and warp/block dispatch is retained. The warp
    path reconstructs unbiased scores by subtracting bias from selected FP32
    biased scores; the block path retains unbiased scores separately. Both
    paths form a FP32 scale divided by the optional normalization sum before
    multiplying each score. This public implementation can lose low bits in
    the warp path's bias subtraction, and is not an ideal-arithmetic reference.

    The first nonempty call must be warmed before CUDA graph capture. An empty
    batch validates metadata and returns without loading the CUDA module.
    """
    if not isinstance(router_logits, torch.Tensor) or not isinstance(
        correction_bias, torch.Tensor
    ):
        raise TypeError("router_logits and correction_bias must be tensors")
    if router_logits.ndim != 2:
        raise ValueError("router_logits must have shape [N, E]")
    tokens, experts = router_logits.shape
    if correction_bias.shape != (experts,):
        raise ValueError("correction_bias must have shape [E]")
    if router_logits.dtype != torch.float32 or correction_bias.dtype != torch.float32:
        raise TypeError("router_logits and correction_bias must be FP32")
    if not router_logits.is_cuda or not correction_bias.is_cuda:
        raise ValueError("router_logits and correction_bias must be CUDA tensors")
    if router_logits.device != correction_bias.device:
        raise ValueError("router_logits and correction_bias must share a device")
    if not router_logits.is_contiguous() or not correction_bias.is_contiguous():
        raise ValueError("router_logits and correction_bias must be contiguous")
    if not isinstance(topk, int) or isinstance(topk, bool):
        raise TypeError("topk must be an integer")
    if not 1 <= experts <= 1024 or not 1 <= topk <= experts:
        raise ValueError("Require 1 <= E <= 1024 and 1 <= topk <= E")
    if not any(
        experts <= tier_e and topk <= tier_k for tier_e, tier_k in _SIGMOID_TIERS
    ):
        raise ValueError("Shape is outside the public sigmoid single-group tiers")
    if tokens > _MAX_TOKENS:
        raise ValueError("N exceeds the public warp path's safe INT32 loop bound")
    if not isinstance(renormalize, bool):
        raise TypeError("renormalize must be bool")
    if not isinstance(routed_scaling_factor, Real) or isinstance(
        routed_scaling_factor, bool
    ):
        raise TypeError("routed_scaling_factor must be a real scalar")
    if (
        not math.isfinite(routed_scaling_factor)
        or abs(routed_scaling_factor) > _FLOAT32_MAX
    ):
        raise ValueError("routed_scaling_factor must be within the finite FP32 range")

    weights = torch.empty(
        (tokens, topk), device=router_logits.device, dtype=torch.float32
    )
    ids = torch.empty((tokens, topk), device=router_logits.device, dtype=torch.int32)
    if tokens == 0:
        return weights, ids
    with torch.cuda.device(router_logits.device):
        module = _load_public_v029_topk_module()
    module.public_v029_single_group_sigmoid_topk(
        router_logits,
        correction_bias,
        weights,
        ids,
        topk,
        renormalize,
        float(routed_scaling_factor),
    )
    return weights, ids
