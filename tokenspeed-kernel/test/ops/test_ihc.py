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

"""iHC numerical boundaries, public operand checks and graph replay."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.residual import ihc_head, ihc_post, ihc_pre
from tokenspeed_kernel.ops.residual.torch import (
    torch_ihc_head,
    torch_ihc_post,
    torch_ihc_pre,
)
from utils import assert_no_triton_compile

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="a CUDA/ROCm device is required"
)

MAGNITUDE = 2.0
RMS_EPS = 1e-5
HC_EPS = 1e-6
NORM_EPS = 1e-5


def _inputs(
    rows: int,
    hc_mult: int,
    hidden: int,
    dtype: torch.dtype,
    device: str,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(173)
    residual = torch.randn(rows, hc_mult, hidden, generator=generator) * 3
    fn = torch.randn(2 * hc_mult, hc_mult * hidden, generator=generator) * 0.02
    scale = torch.tensor([0.7, 1.3])
    base = torch.randn(2 * hc_mult, generator=generator)
    norm = torch.randn(hidden, generator=generator)
    return (
        residual.to(device=device, dtype=dtype),
        *(value.to(device) for value in (fn, scale, base, norm)),
    )


def _reference_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    norm: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    channels, hidden = residual.shape[-2:]
    flat = residual.double().reshape(-1, channels * hidden)
    projected = flat @ fn.double().T
    projected *= (flat.square().mean(-1, keepdim=True) + RMS_EPS).rsqrt()
    pre = torch.sigmoid(projected[:, :channels] * scale[0] + base[:channels])
    pre += HC_EPS
    post = (
        MAGNITUDE * torch.sigmoid(projected[:, channels:] * scale[1] + base[channels:])
        + HC_EPS
    )
    reduced = (pre.unsqueeze(-1) * residual.double().reshape(-1, channels, hidden)).sum(
        1
    )
    # The following norm reads the residual-dtype materialization, including
    # the BF16 rounding before its variance calculation.
    reduced = reduced.to(residual.dtype).double()
    if norm is not None:
        reduced *= (reduced.square().mean(-1, keepdim=True) + NORM_EPS).rsqrt()
        reduced *= norm.double()
    return (
        reduced.bfloat16().reshape(*residual.shape[:-2], hidden),
        post.reshape(*residual.shape[:-2], channels),
    )


def _reference_head(
    residual: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
) -> torch.Tensor:
    channels, hidden = residual.shape[-2:]
    flat = residual.double().reshape(-1, channels * hidden)
    logits = flat @ fn.double().T
    logits *= (flat.square().mean(-1, keepdim=True) + RMS_EPS).rsqrt()
    gates = torch.sigmoid(logits * scale[0] + base.double()) + HC_EPS
    merged = (
        gates.unsqueeze(-1) * residual.double().reshape(-1, channels, hidden)
    ).sum(1)
    return merged.bfloat16().reshape(*residual.shape[:-2], hidden)


def _assert_pre_close(
    actual: tuple[torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor],
) -> None:
    values, gates = actual
    assert values.dtype == torch.bfloat16
    assert gates.dtype == torch.float32
    torch.testing.assert_close(gates.double(), expected[1], rtol=1e-5, atol=1e-6)
    # FP32 summation can move a value across one BF16 rounding midpoint.
    torch.testing.assert_close(values, expected[0], rtol=2**-7, atol=2**-10)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("with_norm", [False, True])
def test_torch_ihc_pre_matches_fp64_on_cpu(dtype: torch.dtype, with_norm: bool) -> None:
    residual, fn, scale, base, norm = _inputs(7, 3, 64, dtype, "cpu")
    norm = norm if with_norm else None
    actual = torch_ihc_pre(
        residual,
        fn,
        scale,
        base,
        MAGNITUDE,
        RMS_EPS,
        HC_EPS,
        norm,
        NORM_EPS if with_norm else None,
    )
    _assert_pre_close(actual, _reference_pre(residual, fn, scale, base, norm))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_torch_ihc_post_and_head_match_fp64_on_cpu(dtype: torch.dtype) -> None:
    residual, fn, scale, base, _ = _inputs(7, 3, 64, dtype, "cpu")
    output = torch.linspace(-3, 3, 7 * 64).reshape(7, 64).bfloat16()
    gates = torch.linspace(0.1, 1.9, 7 * 3).reshape(7, 3)
    actual = torch_ihc_post(output, residual, gates)
    expected = (
        residual.double() + gates.double().unsqueeze(-1) * output.double().unsqueeze(1)
    ).to(dtype)
    torch.testing.assert_close(
        actual,
        expected,
        rtol=2**-7 if dtype == torch.bfloat16 else 1e-6,
        atol=2**-10 if dtype == torch.bfloat16 else 1e-6,
    )
    head = torch_ihc_head(residual, fn[:3], scale[:1], base[:3], RMS_EPS, HC_EPS)
    torch.testing.assert_close(
        head,
        _reference_head(residual, fn[:3], scale[:1], base[:3]),
        rtol=2**-7,
        atol=2**-10,
    )


def test_ihc_pre_rounds_bf16_reduction_before_norm() -> None:
    residual = torch.tensor([[[1.0, 1.0], [1.0078125, 1.03125]]], dtype=torch.bfloat16)
    norm = torch.ones(2)
    result, gates = torch_ihc_pre(
        residual,
        torch.zeros(4, 4),
        torch.ones(2),
        torch.zeros(4),
        2.0,
        RMS_EPS,
        0.0,
        norm,
        NORM_EPS,
    )
    rounded = torch.tensor([[1.0, 1.015625]])
    expected = (
        rounded * (rounded.square().mean(-1, keepdim=True) + NORM_EPS).rsqrt()
    ).bfloat16()
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    torch.testing.assert_close(gates, torch.ones(1, 2), rtol=0, atol=0)


@requires_gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("solution", ["torch", "triton"])
@pytest.mark.parametrize(
    "rows,channels,hidden", [(3, 4, 1536), (300, 4, 64), (37, 3, 512)]
)
def test_ihc_gpu_boundaries_match_fp64(
    dtype: torch.dtype,
    solution: str,
    rows: int,
    channels: int,
    hidden: int,
) -> None:
    residual, fn, scale, base, norm = _inputs(rows, channels, hidden, dtype, "cuda")
    actual = ihc_pre(
        residual,
        fn,
        scale,
        base,
        MAGNITUDE,
        RMS_EPS,
        HC_EPS,
        norm_weight=norm,
        norm_eps=NORM_EPS,
        solution=solution,
    )
    _assert_pre_close(actual, _reference_pre(residual, fn, scale, base, norm))
    updated = ihc_post(actual[0], residual, actual[1], solution=solution)
    expected = (
        residual.double()
        + actual[1].double().unsqueeze(-1) * actual[0].double().unsqueeze(1)
    ).to(dtype)
    torch.testing.assert_close(
        updated,
        expected,
        rtol=2**-7 if dtype == torch.bfloat16 else 1e-6,
        atol=2**-10 if dtype == torch.bfloat16 else 1e-5,
    )
    head = ihc_head(
        updated,
        fn[:channels],
        scale[:1],
        base[:channels],
        RMS_EPS,
        HC_EPS,
        solution=solution,
    )
    torch.testing.assert_close(
        head,
        _reference_head(updated, fn[:channels], scale[:1], base[:channels]),
        rtol=2**-7,
        atol=2**-10,
    )


@requires_gpu
@pytest.mark.parametrize("rows", [1, 1024])
def test_triton_ihc_pre_at_hy4_projection_width(rows: int) -> None:
    residual, fn, scale, base, norm = _inputs(rows, 4, 6144, torch.bfloat16, "cuda")
    actual = ihc_pre(
        residual,
        fn,
        scale,
        base,
        MAGNITUDE,
        RMS_EPS,
        HC_EPS,
        norm_weight=norm,
        norm_eps=NORM_EPS,
        solution="triton",
    )
    _assert_pre_close(actual, _reference_pre(residual, fn, scale, base, norm))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_torch_ihc_empty_rows(dtype: torch.dtype) -> None:
    residual, fn, scale, base, norm = _inputs(0, 4, 64, dtype, "cpu")
    values, gates = torch_ihc_pre(
        residual, fn, scale, base, MAGNITUDE, RMS_EPS, HC_EPS, norm, NORM_EPS
    )
    updated = torch_ihc_post(values, residual, gates)
    merged = torch_ihc_head(updated, fn[:4], scale[:1], base[:4], RMS_EPS, HC_EPS)
    assert values.shape == (0, 64) and values.dtype == torch.bfloat16
    assert gates.shape == (0, 4) and gates.dtype == torch.float32
    assert updated.shape == residual.shape and updated.dtype == dtype
    assert merged.shape == (0, 64) and merged.dtype == torch.bfloat16


@requires_gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_triton_ihc_empty_rows(dtype: torch.dtype) -> None:
    residual, fn, scale, base, norm = _inputs(0, 4, 64, dtype, "cuda")
    values, gates = ihc_pre(
        residual,
        fn,
        scale,
        base,
        MAGNITUDE,
        RMS_EPS,
        HC_EPS,
        norm_weight=norm,
        norm_eps=NORM_EPS,
        solution="triton",
    )
    updated = ihc_post(values, residual, gates, solution="triton")
    merged = ihc_head(
        updated, fn[:4], scale[:1], base[:4], RMS_EPS, HC_EPS, solution="triton"
    )
    assert values.shape == (0, 64) and values.dtype == torch.bfloat16
    assert gates.shape == (0, 4) and gates.dtype == torch.float32
    assert updated.shape == residual.shape and updated.dtype == dtype
    assert merged.shape == (0, 64) and merged.dtype == torch.bfloat16


@requires_gpu
def test_triton_ihc_leading_dimensions_and_strided_residual() -> None:
    source, fn, scale, base, norm = _inputs(12, 3, 64, torch.bfloat16, "cuda")
    residual = source.reshape(2, 6, 3, 64)[:, ::2]
    actual = ihc_pre(
        residual,
        fn,
        scale,
        base,
        MAGNITUDE,
        RMS_EPS,
        HC_EPS,
        norm_weight=norm,
        norm_eps=NORM_EPS,
        solution="triton",
    )
    _assert_pre_close(actual, _reference_pre(residual, fn, scale, base, norm))
    assert actual[0].shape == (2, 3, 64)
    assert actual[1].shape == (2, 3, 3)


def _public_pre(
    operands: dict[str, torch.Tensor], *, norm_eps: float | None = NORM_EPS
):
    return ihc_pre(
        operands["residual"],
        operands["fn"],
        operands["scale"],
        operands["base"],
        MAGNITUDE,
        RMS_EPS,
        HC_EPS,
        norm_weight=operands["norm"],
        norm_eps=norm_eps,
        solution="torch",
    )


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("residual", torch.zeros(3, 1, 32)),
        ("residual", torch.zeros(3, 4, 8, dtype=torch.float16)),
        ("fn", torch.zeros(4, 32)),
        ("fn", torch.zeros(32, 8).T),
        ("fn", torch.zeros(8, 32, dtype=torch.bfloat16)),
        ("scale", torch.zeros(1)),
        ("scale", torch.zeros(2, dtype=torch.bfloat16)),
        ("base", torch.zeros(8, device="meta")),
        ("base", torch.zeros(8, dtype=torch.bfloat16)),
        ("norm", torch.ones(4)),
    ],
)
def test_ihc_pre_rejects_invalid_operands(
    field: str, replacement: torch.Tensor
) -> None:
    residual, fn, scale, base, norm = _inputs(3, 4, 8, torch.bfloat16, "cpu")
    operands = dict(residual=residual, fn=fn, scale=scale, base=base, norm=norm)
    operands[field] = replacement
    with pytest.raises((ValueError, TypeError)):
        _public_pre(operands)


def test_ihc_pre_rejects_unpaired_norm_arguments() -> None:
    residual, fn, scale, base, norm = _inputs(3, 4, 8, torch.bfloat16, "cpu")
    with pytest.raises(ValueError, match="norm_weight and norm_eps"):
        _public_pre(
            dict(residual=residual, fn=fn, scale=scale, base=base, norm=norm),
            norm_eps=None,
        )


@pytest.mark.parametrize(
    "output,gates",
    [
        (torch.zeros(1, 8, dtype=torch.bfloat16), torch.zeros(3, 4)),
        (torch.zeros(3, 8), torch.zeros(3, 4)),
        (
            torch.zeros(3, 8, dtype=torch.bfloat16),
            torch.zeros(3, 4, dtype=torch.bfloat16),
        ),
        (torch.zeros(3, 8, dtype=torch.bfloat16), torch.zeros(3, 4, device="meta")),
    ],
)
def test_ihc_post_rejects_invalid_operands(
    output: torch.Tensor, gates: torch.Tensor
) -> None:
    with pytest.raises((ValueError, TypeError)):
        ihc_post(
            output, torch.zeros(3, 4, 8, dtype=torch.bfloat16), gates, solution="torch"
        )


def test_ihc_head_rejects_pre_projection() -> None:
    residual, fn, scale, base, _ = _inputs(3, 4, 8, torch.bfloat16, "cpu")
    with pytest.raises(ValueError):
        ihc_head(residual, fn, scale, base, RMS_EPS, HC_EPS, solution="torch")


@requires_gpu
def test_triton_ihc_graph_replay_uses_changed_values() -> None:
    residual, fn, scale, base, norm = _inputs(33, 4, 64, torch.bfloat16, "cuda")

    def forward():
        values, gates = ihc_pre(
            residual,
            fn,
            scale,
            base,
            MAGNITUDE,
            RMS_EPS,
            HC_EPS,
            norm_weight=norm,
            norm_eps=NORM_EPS,
            solution="triton",
        )
        updated = ihc_post(values, residual, gates, solution="triton")
        merged = ihc_head(
            updated, fn[:4], scale[:1], base[:4], RMS_EPS, HC_EPS, solution="triton"
        )
        return values, gates, updated, merged

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            forward()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = forward()
    for factor in (0.5, -1.75):
        residual.mul_(factor)
        base.add_(0.125)
        expected = forward()
        graph.replay()
        torch.cuda.synchronize()
        for actual, reference in zip(captured, expected, strict=True):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        _assert_pre_close(captured[:2], _reference_pre(residual, fn, scale, base, norm))


@requires_gpu
def test_triton_ihc_token_counts_reuse_specializations() -> None:
    from tokenspeed_kernel.ops.residual.triton import (
        _ihc_gate_reduce_kernel,
        _ihc_post_kernel,
        _ihc_project_kernel,
    )

    def forward(rows: int) -> None:
        residual, fn, scale, base, norm = _inputs(rows, 4, 64, torch.bfloat16, "cuda")
        values, gates = ihc_pre(
            residual,
            fn,
            scale,
            base,
            MAGNITUDE,
            RMS_EPS,
            HC_EPS,
            norm_weight=norm,
            norm_eps=NORM_EPS,
            solution="triton",
        )
        ihc_post(values, residual, gates, solution="triton")
        ihc_head(
            residual, fn[:4], scale[:1], base[:4], RMS_EPS, HC_EPS, solution="triton"
        )

    # This token ladder shares the projection tile and split bound. Warm both
    # runtime-scalar alignment classes, then vary the actual number of rows.
    forward(33)
    forward(48)
    with assert_no_triton_compile(
        _ihc_project_kernel, _ihc_gate_reduce_kernel, _ihc_post_kernel
    ):
        for rows in (35, 47, 63):
            forward(rows)
