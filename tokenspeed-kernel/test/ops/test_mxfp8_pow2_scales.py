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

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.quantization import quantize_fp8
from tokenspeed_kernel.ops.quantization.triton import (
    _fp8_group32_ue8m0_quantize,
    triton_quantize_fp8_group32_pow2_float32,
    triton_quantize_fp8_with_scale,
)
from utils import assert_no_triton_compile

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA-compatible GPU"
)


def _pow2(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return quantize_fp8(
        x,
        granularity="token_group",
        group_size=32,
        scale_encoding="float32",
        enable_pdl=False,
        solution="triton",
        required_features=frozenset({"power_of_two_scales"}),
    )


def _assert_ue8m0_equivalent(x: torch.Tensor, actual) -> None:
    expected_codes, exponent_bytes = quantize_fp8(
        x,
        granularity="token_group",
        group_size=32,
        scale_encoding="ue8m0",
        enable_pdl=False,
        solution="triton",
    )
    codes, scales = actual
    assert codes.dtype == torch.float8_e4m3fn
    assert scales.dtype == torch.float32
    assert scales.shape == (x.shape[0], x.shape[1] // 32)
    assert torch.equal(codes.view(torch.uint8), expected_codes.view(torch.uint8))
    expected_scales = torch.exp2(exponent_bytes.float() - 127.0)
    assert torch.equal(scales.view(torch.int32), expected_scales.view(torch.int32))
    assert torch.isfinite(codes.float()).all()
    assert torch.isfinite(scales).all()


@pytest.mark.parametrize("rows", [0, 1, 17, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_group32_fp32_scale_storage_preserves_ue8m0_contract(rows, dtype):
    torch.manual_seed(83)
    # Padding the row and slicing columns exercises a noncontiguous row stride.
    x = torch.randn(rows, 160, device="cuda", dtype=dtype)[:, :96]
    _assert_ue8m0_equivalent(x, _pow2(x))


def test_group32_fp32_scale_boundaries_and_finite_bf16_max():
    threshold = torch.tensor(448.0 * 2**-10, dtype=torch.bfloat16)
    values = torch.stack(
        [
            torch.tensor(0.0, dtype=torch.bfloat16),
            torch.tensor(-0.0, dtype=torch.bfloat16),
            torch.tensor(1e-8, dtype=torch.bfloat16),
            torch.tensor(-1e-8, dtype=torch.bfloat16),
            torch.nextafter(threshold, torch.tensor(0.0, dtype=torch.bfloat16)),
            threshold,
            torch.nextafter(
                threshold, torch.tensor(float("inf"), dtype=torch.bfloat16)
            ),
            torch.tensor(torch.finfo(torch.bfloat16).max, dtype=torch.bfloat16),
            torch.tensor(-torch.finfo(torch.bfloat16).max, dtype=torch.bfloat16),
        ]
    )
    x = values[:, None].expand(-1, 64).contiguous().to("cuda")
    actual = _pow2(x)
    _assert_ue8m0_equivalent(x, actual)
    scales = actual[1]
    assert scales[4, 0].item() == 2**-10
    assert scales[5, 0].item() == 2**-10
    assert scales[6, 0].item() == 2**-9
    assert (scales[:4] == 2**-22).all()


def test_group32_pow2_does_not_replace_default_float32_scale_recipe():
    x = torch.ones(3, 96, device="cuda", dtype=torch.bfloat16)
    ordinary = quantize_fp8(
        x,
        granularity="token_group",
        group_size=32,
        scale_encoding="float32",
        enable_pdl=False,
        solution="triton",
    )
    expected = triton_quantize_fp8_with_scale(x, "token_group", 32, "float32", False)
    for got, reference in zip(ordinary, expected, strict=True):
        assert torch.equal(got.view(torch.uint8), reference.view(torch.uint8))
    requested = _pow2(x)
    assert (ordinary[0].float() == 448.0).all()
    assert (requested[0].float() == 256.0).all()
    assert (requested[1] == 2**-8).all()
    assert not torch.equal(ordinary[1], requested[1])


def test_group32_pow2_cuda_graph_replays_changed_input():
    x = torch.randn(17, 160, device="cuda", dtype=torch.bfloat16)[:, :96]
    _pow2(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = _pow2(x)
    for magnitude in (1.0, 1e-8, 0.0):
        x.normal_().mul_(magnitude)
        graph.replay()
        _assert_ue8m0_equivalent(x, actual)


def test_group32_pow2_reuses_compiled_kernel_across_row_counts():
    x = torch.randn(130, 128, device="cuda", dtype=torch.bfloat16)
    # Both row strides are model-static specializations; M itself must not be.
    _pow2(x[:17])
    _pow2(x[::2][:17])
    with assert_no_triton_compile(_fp8_group32_ue8m0_quantize):
        for rows in (1, 16, 33, 64):
            _pow2(x[:rows])
            _pow2(x[::2][:rows])


def test_group32_pow2_rejects_unsupported_pdl():
    x = torch.ones(1, 32, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="PDL"):
        triton_quantize_fp8_group32_pow2_float32(x, "token_group", 32, "float32", True)
