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

"""GPU contract tests; these are not calibration against a reference binary."""

import pytest
import torch
from tokenspeed_kernel.ops.activation.cuda_staged import silu_and_mul
from tokenspeed_kernel.ops.layernorm.cuda_staged import rmsnorm

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="The staged CUDA candidates require NVIDIA CUDA",
)


@pytest.mark.parametrize("rows,hidden", [(1, 128), (30, 512), (30, 2048), (257, 6144)])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
def test_rmsnorm_formula_and_repeat(rows, hidden, weight_dtype):
    torch.manual_seed(71)
    x = torch.randn(rows, hidden, dtype=torch.bfloat16, device="cuda")
    weight = (torch.rand(hidden, device="cuda") + 0.5).to(weight_dtype)
    before = x.clone()
    actual = rmsnorm(x, weight, 1e-6)
    values = x.float()
    expected = (
        values
        * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)
        * weight.float()
    ).bfloat16()
    # Torch and CUB have different FP32 reduction trees. This checks the
    # mathematical/cast contract within one BF16 ulp, not binary equivalence.
    torch.testing.assert_close(actual.float(), expected.float(), rtol=2**-7, atol=0)
    assert torch.equal(x, before)
    assert torch.equal(actual, rmsnorm(x, weight, 1e-6))


@pytest.mark.parametrize("shape", [(7, 512), (2, 3, 512), (2, 3, 4, 512)])
def test_rmsnorm_strided_output_and_full_alias(shape):
    storage = torch.randn(
        *shape[:-1], shape[-1] * 2, device="cuda", dtype=torch.bfloat16
    )
    x = storage[..., : shape[-1]]
    weight = torch.randn(shape[-1], device="cuda", dtype=torch.float32)
    expected = rmsnorm(x, weight, 1e-6)
    output_storage = torch.empty_like(storage)
    out = output_storage[..., : shape[-1]]
    assert rmsnorm(x, weight, 1e-6, out=out) is out
    assert torch.equal(out, expected)
    untouched = storage[..., shape[-1] :].clone()
    assert rmsnorm(x, weight, 1e-6, out=x) is x
    assert torch.equal(x, expected)
    assert torch.equal(storage[..., shape[-1] :], untouched)


def test_rmsnorm_rejects_partial_alias_and_overlapping_rows():
    base = torch.empty(4, 264, device="cuda", dtype=torch.bfloat16)
    x, out = base[:, :256], base[:, 8:264]
    weight = torch.ones(256, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="exact full alias"):
        rmsnorm(x, weight, 1e-6, out=out)
    overlapping = base.as_strided((4, 256), (8, 1))
    with pytest.raises(ValueError, match="overlapping"):
        rmsnorm(overlapping, weight, 1e-6)
    with pytest.raises(ValueError, match="channel vector"):
        rmsnorm(x, weight.half(), 1e-6)


def test_qkv_views_preserve_query_and_rope_storage():
    qkv = torch.randn(30, 2624, device="cuda", dtype=torch.bfloat16)
    q, kv, rope = qkv.split((2048, 512, 64), dim=-1)
    before = qkv.clone()
    q_weight = torch.randn(2048, device="cuda", dtype=torch.bfloat16)
    kv_weight = torch.randn(512, device="cuda", dtype=torch.bfloat16)
    expected_q = rmsnorm(q.contiguous(), q_weight, 1e-6)
    expected_kv = rmsnorm(kv.contiguous(), kv_weight, 1e-6)
    q_out = rmsnorm(q, q_weight, 1e-6)
    assert q_out.data_ptr() != q.data_ptr()
    assert rmsnorm(kv, kv_weight, 1e-6, out=kv) is kv
    assert torch.equal(q_out, expected_q)
    assert torch.equal(kv, expected_kv)
    assert torch.equal(q, before[:, :2048])
    assert torch.equal(rope, before[:, 2560:])


@pytest.mark.parametrize(
    "shape", [(1, 8192), (30, 4096), (30, 36864), (2, 3, 66), (2, 3, 4, 512)]
)
def test_silu_retains_intermediate_bf16_rounding(shape):
    torch.manual_seed(19)
    x = (torch.randn(shape, device="cuda") * 4).bfloat16()
    hidden = shape[-1] // 2
    gate, up = x[..., :hidden].float(), x[..., hidden:].float()
    activation = gate / (1.0 + torch.exp(-gate))
    expected = (activation.bfloat16().float() * (up + 0.0)).bfloat16()
    single_cast = (activation * up).bfloat16()
    assert not torch.equal(expected, single_cast)
    actual = silu_and_mul(x)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=2**-7, atol=0)
    assert (actual == expected).float().mean().item() > 0.999
    assert torch.equal(actual, silu_and_mul(x))


def test_silu_strided_output_and_alias_rejection():
    base = torch.randn(7, 544, device="cuda", dtype=torch.bfloat16)
    x = base[:, :512]
    expected = silu_and_mul(x.contiguous())
    output_base = torch.empty(7, 288, device="cuda", dtype=torch.bfloat16)
    out = output_base[:, :256]
    assert silu_and_mul(x, out=out) is out
    assert torch.equal(out, expected)
    with pytest.raises(ValueError, match="must not overlap"):
        silu_and_mul(x, out=x[:, :256])


def test_empty_rows_do_not_require_a_launch():
    x = torch.empty(0, 512, device="cuda", dtype=torch.bfloat16)
    weight = torch.ones(512, device="cuda", dtype=torch.bfloat16)
    assert rmsnorm(x, weight, 1e-6).shape == (0, 512)
    assert silu_and_mul(x).shape == (0, 256)


def test_current_stream_and_warmed_cuda_graph():
    x = torch.empty(7, 512, device="cuda", dtype=torch.bfloat16)
    weight = torch.ones(512, device="cuda", dtype=torch.bfloat16)
    norm_out = torch.empty_like(x)
    silu_out = torch.empty(7, 256, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        x.fill_(1)
        rmsnorm(x, weight, 1e-6, out=norm_out)
        silu_and_mul(x, out=silu_out)
    torch.cuda.current_stream().wait_stream(stream)
    assert torch.equal(norm_out, torch.ones_like(norm_out))
    assert torch.equal(silu_out, torch.full_like(silu_out, 0.73046875))

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        rmsnorm(x, weight, 1e-6, out=norm_out)
        silu_and_mul(x, out=silu_out)
    x.copy_(torch.randn_like(x))
    expected_norm = rmsnorm(x, weight, 1e-6)
    expected_silu = silu_and_mul(x)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(norm_out, expected_norm)
    assert torch.equal(silu_out, expected_silu)
