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

"""Optional, explicitly invoked CUDA candidates; importing does not compile."""

import hashlib
from functools import cache
from inspect import signature
from pathlib import Path

import torch

_CUDA_FLAGS = ("--ftz=false", "--prec-div=true", "--prec-sqrt=true", "--fmad=true")


@cache
def load_module():
    """Build the independent TVM-FFI module before CUDA graph capture.

    This is the sole compilation entry. It does not load a vLLM library or
    change any registered kernel. Compiler and dependency failures propagate.
    """
    from flashinfer.jit.core import gen_jit_spec

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Call load_module() before CUDA graph capture")
    if "use_fast_math" not in signature(gen_jit_spec).parameters:
        raise RuntimeError("FlashInfer gen_jit_spec must support use_fast_math")
    root = Path(__file__).resolve().parent
    source = root / "kernels.cu"
    include_dir = root.parent / "cuda" / "csrc"
    digest = hashlib.sha256()
    for path in (source, include_dir / "tvm_ffi_utils.h"):
        digest.update(path.read_bytes())
    digest.update("\n".join(_CUDA_FLAGS).encode())
    digest.update(b"\nuse_fast_math=False")
    name = f"tokenspeed_vllm_staged_norm_activation_v1_{digest.hexdigest()[:16]}"
    return gen_jit_spec(
        name,
        [source],
        extra_include_paths=[include_dir],
        use_fast_math=False,
        extra_cuda_cflags=list(_CUDA_FLAGS),
    ).build_and_load()


def check_bf16_rows(x: torch.Tensor, name: str, *, vector_width: int) -> None:
    """Check metadata only, including that rows cannot overlap."""
    if x.device.type != "cuda" or x.dtype != torch.bfloat16:
        raise ValueError(f"{name} must be a CUDA BF16 tensor")
    if not 2 <= x.ndim <= 4 or x.shape[-1] <= 0 or x.stride(-1) != 1:
        raise ValueError(f"{name} must have rank 2..4 and dense nonempty rows")
    if x.shape[-1] % vector_width:
        raise ValueError(f"{name} row width must be divisible by {vector_width}")
    span = 1
    for stride, size in sorted(zip(x.stride(), x.shape)):
        if size <= 1:
            continue
        if stride < span:
            raise ValueError(f"{name} must not have overlapping elements or rows")
        span += (size - 1) * stride
    if x.numel() // x.shape[-1] > 2**31 - 1 or x.shape[-1] > 2**31 - 1:
        raise ValueError(f"{name} exceeds the supported CUDA launch dimensions")
    if x.numel() and x.data_ptr() % (vector_width * x.element_size()):
        raise ValueError(f"{name} must have aligned rows")
    if any(
        size > 1 and stride % vector_width
        for size, stride in zip(x.shape[:-1], x.stride()[:-1])
    ):
        raise ValueError(f"{name} must have aligned row strides")


def overlaps(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Conservatively compare address spans, including gaps between rows."""
    if not a.numel() or not b.numel():
        return False
    a_end = a.data_ptr() + a.element_size() * (
        1 + sum((n - 1) * s for n, s in zip(a.shape, a.stride()))
    )
    b_end = b.data_ptr() + b.element_size() * (
        1 + sum((n - 1) * s for n, s in zip(b.shape, b.stride()))
    )
    return a.data_ptr() < b_end and b.data_ptr() < a_end


def exact_alias(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Whether the two tensors describe exactly the same addressed elements."""
    return (
        a.data_ptr() == b.data_ptr()
        and a.shape == b.shape
        and a.stride() == b.stride()
        and a.dtype == b.dtype
        and a.device == b.device
    )
