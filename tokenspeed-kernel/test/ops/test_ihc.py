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

"""iHC boundaries on FP32 residual channels: one BF16 rounding per output."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel import ihc_head, ihc_post, ihc_pre

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="a CUDA/ROCm device is required"
)

HC_MULT = 4
HIDDEN_SIZE = 1536
MAGNITUDE = 2.0
RMS_EPS = 1e-5
HC_EPS = 1e-6
NORM_EPS = 1e-5
SOLUTIONS = ("torch", "triton")


def _randn(*shape: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=generator).cuda()


def _params(gate_count: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fn = _randn(gate_count, HC_MULT * HIDDEN_SIZE, seed=1) * 0.02
    hc_scale = torch.tensor([0.7, 1.3][: gate_count // HC_MULT], device="cuda")
    hc_base = _randn(gate_count, seed=2)
    return fn, hc_scale, hc_base


def _fp64_gates(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    index: int,
) -> torch.Tensor:
    flat = residual.double().flatten(1)
    rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + RMS_EPS)
    mixes = (flat @ fn.double().T) * rsqrt
    cols = slice(index * HC_MULT, (index + 1) * HC_MULT)
    return torch.sigmoid(mixes[:, cols] * float(hc_scale[index]) + hc_base[cols])


@pytest.mark.parametrize("solution", SOLUTIONS)
@pytest.mark.parametrize("rows", [3, 300])
def test_ihc_pre_fp32_residual_matches_fp64(solution: str, rows: int) -> None:
    residual = _randn(rows, HC_MULT, HIDDEN_SIZE, seed=rows) * 3
    fn, hc_scale, hc_base = _params(2 * HC_MULT)
    norm_weight = _randn(HIDDEN_SIZE, seed=3).bfloat16()
    pre = _fp64_gates(residual, fn, hc_scale, hc_base, 0) + HC_EPS
    post = MAGNITUDE * _fp64_gates(residual, fn, hc_scale, hc_base, 1) + HC_EPS
    reduced = (pre.unsqueeze(-1) * residual.double()).sum(1)
    rstd = torch.rsqrt(reduced.square().mean(-1, keepdim=True) + NORM_EPS)
    expected = reduced * rstd * norm_weight.double()

    layer_input, gates = ihc_pre(
        residual,
        fn,
        hc_scale,
        hc_base,
        MAGNITUDE,
        RMS_EPS,
        HC_EPS,
        norm_weight=norm_weight,
        norm_eps=NORM_EPS,
        solution=solution,
    )

    assert layer_input.dtype == torch.bfloat16
    torch.testing.assert_close(gates.double(), post, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(layer_input, expected.bfloat16(), rtol=2**-7, atol=1e-2)


@pytest.mark.parametrize("solution", SOLUTIONS)
def test_ihc_post_keeps_fp32_channels(solution: str) -> None:
    residual = _randn(300, HC_MULT, HIDDEN_SIZE, seed=4) * 3
    block_output = _randn(300, HIDDEN_SIZE, seed=5).bfloat16()
    gates = _randn(300, HC_MULT, seed=6).abs()

    actual = ihc_post(block_output, residual, gates, solution=solution)

    assert actual.dtype == torch.float32
    expected = (
        gates.double().unsqueeze(-1) * block_output.double().unsqueeze(-2)
        + residual.double()
    )
    torch.testing.assert_close(actual.double(), expected, rtol=1e-6, atol=1e-5)


@pytest.mark.parametrize("solution", SOLUTIONS)
def test_ihc_head_fp32_residual_matches_fp64(solution: str) -> None:
    residual = _randn(37, HC_MULT, HIDDEN_SIZE, seed=7) * 3
    fn, hc_scale, hc_base = _params(HC_MULT)
    gates = _fp64_gates(residual, fn, hc_scale, hc_base, 0) + HC_EPS
    expected = (gates.unsqueeze(-1) * residual.double()).sum(1)

    actual = ihc_head(
        residual, fn, hc_scale, hc_base, RMS_EPS, HC_EPS, solution=solution
    )

    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected.bfloat16(), rtol=2**-7, atol=1e-2)
