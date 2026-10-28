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

"""CPU contracts that do not require installing the optional Humming backend."""

import pytest
import torch
from tokenspeed_kernel.thirdparty.humming import (
    HummingIndexedPlan,
    HummingMXFP8WeightState,
    _validate_mxfp8_weights,
    humming_indexed_gemm,
)


def _plan() -> HummingIndexedPlan:
    state = HummingMXFP8WeightState(
        weight=torch.empty(2, 64, 16, dtype=torch.int32),
        scales=torch.full((2, 64, 2), 127, dtype=torch.uint8),
        layer_config="{}",
        num_experts=2,
        size_n=64,
        size_k=64,
        device=torch.device("cpu"),
    )
    return HummingIndexedPlan(
        state=state,
        compute_config="{}",
        tuning_config="[]",
        intervals=((0, 8, 16), (8, 64, 32), (64, 1024, 64)),
        kernels=torch.empty(0, dtype=torch.int64),
    )


@pytest.mark.parametrize("code", [0, 255])
def test_unsupported_e8m0_values_fail_before_repacking(code: int) -> None:
    weight = torch.zeros(2, 64, 64, dtype=torch.float8_e4m3fn)
    scales = torch.full((2, 64, 2), 127, dtype=torch.uint8)
    scales[1, 63, 1] = code
    with pytest.raises(ValueError, match="codes in \\[1, 254\\]"):
        _validate_mxfp8_weights(weight, scales)


def test_supported_e8m0_extremes_preserve_raw_bytes() -> None:
    weight = torch.zeros(2, 64, 64, dtype=torch.float8_e4m3fn)
    scales = torch.empty(2, 64, 2, dtype=torch.uint8)
    scales[..., 0] = 1
    scales[..., 1] = 254
    original = scales.clone()
    _validate_mxfp8_weights(weight, scales)
    torch.testing.assert_close(scales, original)


@pytest.mark.parametrize("code", [127, 255])
def test_nonfinite_weight_codes_are_rejected(code: int) -> None:
    weight = torch.zeros(1, 64, 64, dtype=torch.float8_e4m3fn)
    weight.view(torch.uint8)[0, 0, 0] = code
    scales = torch.full((1, 64, 2), 127, dtype=torch.uint8)
    with pytest.raises(ValueError, match="weights must be finite"):
        _validate_mxfp8_weights(weight, scales)


@pytest.mark.parametrize("size_n,size_k", [(96, 64), (64, 96)])
def test_matrix_alignment_is_checked_before_loading_humming(
    size_n: int, size_k: int
) -> None:
    weight = torch.empty(2, size_n, size_k, dtype=torch.float8_e4m3fn)
    scales = torch.full((2, size_n, size_k // 32), 127, dtype=torch.uint8)
    with pytest.raises(ValueError, match="divisible by 64"):
        _validate_mxfp8_weights(weight, scales)


def test_group128_scales_cannot_be_reinterpreted_as_group32() -> None:
    weight = torch.empty(2, 64, 128, dtype=torch.float8_e4m3fn)
    scales = torch.full((2, 64, 1), 127, dtype=torch.uint8)
    with pytest.raises(ValueError, match="K / 32"):
        _validate_mxfp8_weights(weight, scales)


@pytest.mark.parametrize(
    "routes,alignment", [(1, 16), (8, 16), (9, 32), (64, 32), (65, 64), (1024, 64)]
)
def test_route_alignment_matches_interval_boundaries(
    routes: int, alignment: int
) -> None:
    assert _plan().block_m(routes) == alignment


@pytest.mark.parametrize("routes", [-1, 0, 1025])
def test_uncovered_route_capacity_is_rejected(routes: int) -> None:
    with pytest.raises(ValueError, match="No Humming tuning interval"):
        _plan().block_m(routes)


def test_unquantized_inputs_are_rejected_without_automatic_quantization() -> None:
    with pytest.raises(ValueError, match="inputs must have dtype"):
        humming_indexed_gemm(
            _plan(),
            torch.zeros(1, 64, dtype=torch.bfloat16),
            torch.ones(1, 2),
            torch.empty(2, 64, dtype=torch.bfloat16),
            torch.zeros(16, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
            torch.tensor([16], dtype=torch.int32),
            top_k=2,
            valid_shape_m=2,
        )


@pytest.mark.parametrize("invalid", ["scales", "output", "route_dtype"])
def test_invalid_gemm_buffers_are_rejected_before_launch(invalid: str) -> None:
    scales = torch.ones(1, 2)
    output = torch.empty(2, 64, dtype=torch.bfloat16)
    routes = torch.zeros(16, dtype=torch.int32)
    if invalid == "scales":
        scales = scales.to(torch.bfloat16)
        error = "input_scales must have dtype"
    elif invalid == "output":
        output = output[:1]
        error = "outputs must have dtype"
    else:
        routes = routes.to(torch.int64)
        error = "sorted_ids must be contiguous 1D int32"
    with pytest.raises(ValueError, match=error):
        humming_indexed_gemm(
            _plan(),
            torch.empty(1, 64, dtype=torch.float8_e4m3fn),
            scales,
            output,
            routes,
            torch.zeros(1, dtype=torch.int32),
            torch.tensor([16], dtype=torch.int32),
            top_k=2,
            valid_shape_m=2,
        )


def test_empty_input_returns_caller_output_without_optional_dependency() -> None:
    output = torch.empty(0, 64, dtype=torch.bfloat16)
    actual = humming_indexed_gemm(
        _plan(),
        torch.empty(0, 64, dtype=torch.float8_e4m3fn),
        torch.empty(0, 2),
        output,
        torch.empty(0, dtype=torch.int32),
        torch.empty(0, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        top_k=2,
        valid_shape_m=0,
    )
    assert actual is output
