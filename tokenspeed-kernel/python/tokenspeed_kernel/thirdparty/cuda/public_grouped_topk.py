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

"""Pinned public CUDA grouped sigmoid TopK, loaded only on explicit use."""

from __future__ import annotations

import hashlib
import inspect
import math
from functools import cache
from numbers import Real
from pathlib import Path

import torch

_UPSTREAM_COMMIT = "4fd9d6a85c00ac0186aa9abbeff73fc2ac6c721e"
_CUDA_FLAGS = ("--ftz=false", "--prec-div=true", "--prec-sqrt=true", "--fmad=false")
_INT32_MAX = 2**31 - 1


@cache
def _load_public_grouped_topk_module():
    """Build once outside graph capture with an explicit arithmetic policy."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Warm public grouped TopK before CUDA graph capture")
    from flashinfer.jit.core import gen_jit_spec

    if "use_fast_math" not in inspect.signature(gen_jit_spec).parameters:
        raise RuntimeError(
            "public grouped TopK requires FlashInfer gen_jit_spec with "
            "use_fast_math support"
        )
    root = Path(__file__).resolve().parent / "csrc"
    source = root / "public_grouped_topk.cu"
    digest = hashlib.sha256()
    for path in (source, root / "tvm_ffi_utils.h"):
        digest.update(path.read_bytes())
    digest.update("\n".join(_CUDA_FLAGS).encode())
    name = f"tokenspeed_public_topk_{_UPSTREAM_COMMIT[:12]}_{digest.hexdigest()[:16]}"
    return gen_jit_spec(
        name,
        [source],
        extra_include_paths=[root],
        extra_cuda_cflags=list(_CUDA_FLAGS),
        use_fast_math=False,
    ).build_and_load()


def grouped_sigmoid_topk(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    *,
    num_expert_group: int,
    topk_group: int,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select groups and experts with the pinned public vLLM CUDA algorithm.

    Args:
        router_logits: Contiguous CUDA FP32 tensor with shape ``[N, E]``.
        correction_bias: Contiguous CUDA FP32 tensor ``[E]`` on the same device.
            Logits, biases, and the resulting group scores must be finite.
            This value contract is not scanned on the GPU in the hot path.
        num_expert_group: Number of equal expert groups in ``[1, 32]``. Each
            group must have at least two experts for its top-two score.
        topk_group: Number of groups selected by their top-two biased scores.
        topk: Number of experts selected from those groups, at most 32.
        renormalize: Normalize the selected unbiased sigmoid scores when true.
        routed_scaling_factor: Finite scale applied with the upstream double
            scalar arithmetic, including when ``renormalize`` is false.

    Returns:
        Contiguous ``(weights, ids)`` tensors with shape ``[N, topk]``, FP32
        weights and INT32 ids. Biased-score ties prefer smaller expert ids.
        The upstream sigmoid uses ``0.5*tanh(0.5*x)+0.5`` and its normalization
        sum starts at ``1e-20``. These are public implementation semantics,
        not a claim of bitwise equality with another installed binary.

    The first nonempty call builds the optional CUDA module and must happen
    before CUDA graph capture. Empty calls validate metadata but do not build.
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
    for name, value in (
        ("num_expert_group", num_expert_group),
        ("topk_group", topk_group),
        ("topk", topk),
    ):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer")
    if not 1 <= num_expert_group <= 32:
        raise ValueError("num_expert_group must be in [1, 32]")
    if experts < 2 or experts > _INT32_MAX - 31:
        raise ValueError("E must fit the public kernel's padded INT32 indexing")
    if experts % num_expert_group or experts // num_expert_group < 2:
        raise ValueError("E must divide into groups of at least two experts")
    if tokens > _INT32_MAX // num_expert_group:
        raise ValueError("N * num_expert_group must fit INT32 indexing")
    if not 1 <= topk_group <= num_expert_group:
        raise ValueError("topk_group must be in [1, num_expert_group]")
    max_topk = min(32, topk_group * (experts // num_expert_group))
    if not 1 <= topk <= max_topk:
        raise ValueError(f"topk must be in [1, {max_topk}]")
    if not isinstance(renormalize, bool):
        raise TypeError("renormalize must be bool")
    if not isinstance(routed_scaling_factor, Real) or isinstance(
        routed_scaling_factor, bool
    ):
        raise TypeError("routed_scaling_factor must be a real scalar")
    if not math.isfinite(routed_scaling_factor):
        raise ValueError("routed_scaling_factor must be finite")

    weights = torch.empty(
        (tokens, topk), device=router_logits.device, dtype=torch.float32
    )
    ids = torch.empty((tokens, topk), device=router_logits.device, dtype=torch.int32)
    if tokens == 0:
        return weights, ids
    group_scores = torch.empty(
        (tokens, num_expert_group), device=router_logits.device, dtype=torch.float32
    )
    with torch.cuda.device(router_logits.device):
        module = _load_public_grouped_topk_module()
    module.public_grouped_sigmoid_topk(
        router_logits,
        correction_bias,
        group_scores,
        weights,
        ids,
        num_expert_group,
        topk_group,
        topk,
        renormalize,
        float(routed_scaling_factor),
    )
    return weights, ids
