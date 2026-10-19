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

"""CPU checks of MXFP8 MoE dispatch boundaries and the GPU tests' oracle.

The dispatch test replaces device kernels with recording boundaries and executes
the production Python dispatcher. It does not validate GPU arithmetic. The
oracle test independently checks when routed outputs round to BF16.
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from types import SimpleNamespace

# CI Registration (parsed via AST, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
KERNEL = ROOT / "tokenspeed-kernel/python/tokenspeed_kernel"
GPU_TESTS = ROOT / "tokenspeed-kernel/test/nvidia/ops/moe/test_marlin_mxfp8_apply.py"


def _functions(path, names, namespace):
    tree = ast.parse(path.read_text(), filename=str(path))
    body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            body.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in names
            for target in node.targets
        ):
            body.append(node)
    assert len(body) == len(names)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def test_reference_matches_native_bf16_routing_epilogue():
    oracle = _functions(
        GPU_TESTS, {"_dequant", "_reference"}, {"torch": torch, "BLOCK": 32}
    )
    hidden, intermediate = 128, 64
    w13 = torch.zeros(1, 2 * intermediate, hidden)
    w13[:, :, 0] = 1
    w2 = torch.zeros(1, hidden, intermediate)
    w2[:, :, 0] = 1
    w2[:, :, 1] = 1 / 512
    raw = {
        "w13_weight": w13.to(torch.float8_e4m3fn),
        "w13_weight_scale_inv": torch.full(
            (1, 2 * intermediate, hidden // 32), 127, dtype=torch.uint8
        ),
        "w2_weight": w2.to(torch.float8_e4m3fn),
        "w2_weight_scale_inv": torch.full(
            (1, hidden, intermediate // 32), 127, dtype=torch.uint8
        ),
    }
    x = torch.zeros(1, hidden, dtype=torch.bfloat16)
    x[0, 0] = 1
    ids = torch.zeros(1, 1, dtype=torch.int32)
    weights = torch.tensor([[0.7]], dtype=torch.float32)
    result = oracle["_reference"](x, raw, ids, weights, limit=None)
    # GEMM1 is exactly [1, 1, ...]. BF16 SiLU(1) = 0.73046875, and
    # GEMM2 is 0.73046875 * (1 + 1/512) before applying the route weight.
    accumulated = torch.tensor(0.73046875 * (1 + 1 / 512), dtype=torch.float32)
    expected_route = (
        (accumulated.bfloat16().float() * weights[0, 0].bfloat16().float())
        .bfloat16()
        .float()
    )
    prematurely_rounded = accumulated.bfloat16().float() * weights[0, 0]
    assert expected_route.item() == 0.51171875
    assert expected_route != prematurely_rounded
    torch.testing.assert_close(result, expected_route.expand_as(result), rtol=0, atol=0)


class _BranchReached(Exception):
    pass


@pytest.mark.parametrize(
    "tokens,experts,ep_size,dense,expected",
    [
        (1, 256, 32, True, "gemv"),
        (2, 256, 32, True, "gemv"),
        (3, 256, 32, True, "marlin"),
        (1, 256, 1, True, "marlin"),
        (6143, 256, 32, True, "marlin"),
        (6144, 256, 32, True, "cutlass"),
        (6144, 256, 32, False, "marlin"),
    ],
)
def test_production_dispatch_uses_global_route_density(
    monkeypatch, tokens, experts, ep_size, dense, expected
):
    reached = []

    def branch(name):
        def record(*args, **kwargs):
            reached.append(name)
            raise _BranchReached(name)

        return record

    namespace = {
        "torch": torch,
        "_MarlinMxfp8State": SimpleNamespace,
        "_dense_bf16_apply": branch("cutlass"),
        "_gemv_apply": branch("gemv"),
        "moe_align_block_size": branch("marlin"),
    }
    code = _functions(
        KERNEL / "ops/moe/marlin/mxfp8.py",
        {
            "_DENSE_MIN_ROUTES_PER_EXPERT",
            "_GEMV_MAX_ROUTES",
            "_gemv_is_faster",
            "_block_size_m",
            "marlin_mxfp8_precomputed_moe_apply",
        },
        namespace,
    )
    x = torch.empty(tokens, 128, dtype=torch.bfloat16)
    ids = torch.arange(8, dtype=torch.int32).expand(tokens, -1).contiguous()
    weights = torch.ones(tokens, 8, dtype=torch.float32)
    state = SimpleNamespace(
        hidden_size=128, ispp=128, dense_prefill=dense, expert_start=0
    )
    w = SimpleNamespace(
        w13_weight=torch.empty(experts // ep_size, 1, 1, dtype=torch.int32),
        _moe_backend_state=state,
        activation="swiglu",
        ep_size=ep_size,
    )

    def no_value_read(*args, **kwargs):
        raise AssertionError("Dispatch must use host shapes, not tensor values")

    monkeypatch.setattr(torch.Tensor, "item", no_value_read)
    with pytest.raises(_BranchReached, match=expected):
        code["marlin_mxfp8_precomputed_moe_apply"](
            {"activation": "swiglu", "deterministic_finalize": True},
            x,
            w,
            None,
            topk_weights=weights,
            topk_ids=ids,
        )
    assert reached == [expected]


def test_gemv_finalize_matches_native_bf16_routing_epilogue():
    """Execute the production GEMV epilogue with CPU tensors, not a GPU JIT.

    Native Marlin writes BF16 GEMM2 fragments and stages BF16 route weights
    before its BF16 product. An FP32 product with only a final cast differs.
    """
    source = KERNEL / "ops/moe/marlin/mxfp8.py"
    tree = ast.parse(source.read_text(), filename=str(source))
    kernel = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_gemv_topk_sum_kernel"
    )
    slots = next(node for node in kernel.body if isinstance(node, ast.For))
    local = next(node for node in slots.body if isinstance(node, ast.If))
    split_loop = next(
        i for i, node in enumerate(local.body) if isinstance(node, ast.For)
    )
    # The reduced FP32 GEMM2 value is the input to this extracted arithmetic
    # region. Pointer loads and stores are outside the contract under test.
    epilogue = local.body[split_loop + 1 :]
    assert any(isinstance(node, ast.AugAssign) for node in epilogue)
    helpers = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_marlin_weighted_route":
            node.decorator_list = []
            helpers.append(node)
    y = torch.tensor(
        [1.00390625, 0.7318954467773438, 3.1415927, -1.00390625, 17.125, 0.1234567]
    )[:, None]
    weights = torch.tensor([0.7, 0.31, 0.1234567, 0.9999, 0.1, 2.827])[None, :]
    expected = (y.bfloat16().float() * weights.bfloat16().float()).bfloat16().float()
    old_combined = (y * weights).bfloat16().float()
    assert int((expected != old_combined).sum()) == 10
    namespace = {
        "y": y,
        "acc": torch.zeros_like(expected),
        "weights_ptr": 0,
        "route": 0,
        "tl": SimpleNamespace(
            bfloat16=torch.bfloat16,
            float32=torch.float32,
            load=lambda pointer: weights,
        ),
    }
    exec(
        compile(
            ast.Module(body=helpers + epilogue, type_ignores=[]), str(source), "exec"
        ),
        namespace,
    )
    torch.testing.assert_close(namespace["acc"], expected, rtol=0, atol=0)
