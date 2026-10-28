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

"""Hopper W8A8 GEMM, MoE, and graph contracts with an independent FP32 oracle.

The GEMM test supplies fixed quantized operands, isolating kernel error. The
complete MoE tests additionally include the specified activation quantization
and BF16 rounding boundaries. Missing optional Humming skips its GPU tests;
installed-but-broken kernels must fail rather than falling back to another op.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import pytest
import torch
from tokenspeed_kernel.benchmark.mxfp8_replay import (
    ReplayArtifact,
    dequantize_weight,
    reference_output,
)
from tokenspeed_kernel.ops.moe._triton.bf16_epilogue import finalize_bf16_routes
from tokenspeed_kernel.ops.moe._triton.humming_routing import align_humming_routes
from tokenspeed_kernel.ops.moe.humming import (
    humming_mxfp8_moe_weights,
    humming_mxfp8_precomputed_moe_apply,
)
from tokenspeed_kernel.thirdparty.humming import (
    humming_indexed_gemm,
    prepare_humming_mxfp8_weights,
)

EXPERTS, HIDDEN, INTERMEDIATE, TOP_K = 4, 128, 64, 3
EXPERT_START = EXPERTS
PLAN = {
    "internal_activation_dtype": "fp8",
    "combine_order": "rank",
    "fast_math": False,
    "a2a_backend": None,
}


def _require_sm90() -> None:
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("requires NVIDIA CUDA")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("requires SM90, including H20")


@dataclass(frozen=True)
class _SwiGLU:
    alpha: float
    limit: float


class _Weights(torch.nn.Module):
    """Synthetic canonical expert parameters with the real loader contract."""

    def __init__(self, canonical: dict[str, torch.Tensor]):
        super().__init__()
        self.w13_weight = torch.nn.Parameter(
            canonical["w13_weight"].cuda(), requires_grad=False
        )
        self.w13_weight_scale_inv = torch.nn.Parameter(
            canonical["w13_weight_scale_inv"].cuda(), requires_grad=False
        )
        self.w2_weight = torch.nn.Parameter(
            canonical["w2_weight"].cuda(), requires_grad=False
        )
        self.w2_weight_scale_inv = torch.nn.Parameter(
            canonical["w2_weight_scale_inv"].cuda(), requires_grad=False
        )
        self.activation = "swiglu"
        self.swiglu_arg = _SwiGLU(alpha=1.0, limit=10.0)
        self.swiglu_beta = 0.0
        self.w13_input_layout = "concatenated"
        self.w13_weight_bias = None
        self.w2_weight_bias = None
        self.ep_size = 2
        self.ep_rank = 1
        self.num_experts = EXPERTS * self.ep_size
        self.num_local_experts = EXPERTS
        self.hidden_size = HIDDEN
        self.intermediate_size_per_partition = INTERMEDIATE
        self.top_k = TOP_K
        self._moe_backend_state: object | None = None


@dataclass
class _Prepared:
    layer: _Weights
    canonical: dict[str, torch.Tensor]


@pytest.fixture(scope="module")
def prepared() -> _Prepared:
    _require_sm90()
    if importlib.util.find_spec("humming") is None:
        pytest.skip("install optional requirements/cuda-humming.txt for Humming")
    generator = torch.Generator().manual_seed(92)
    canonical = {}
    for name, size_n, size_k in (
        ("w13", 2 * INTERMEDIATE, HIDDEN),
        ("w2", HIDDEN, INTERMEDIATE),
    ):
        canonical[f"{name}_weight"] = torch.randint(
            -4, 5, (EXPERTS, size_n, size_k), generator=generator
        ).to(torch.float8_e4m3fn)
        canonical[f"{name}_weight_scale_inv"] = torch.randint(
            121,
            125,
            (EXPERTS, size_n, size_k // 32),
            generator=generator,
            dtype=torch.uint8,
        )
    layer = _Weights(canonical)
    humming_mxfp8_moe_weights(PLAN, layer)
    return _Prepared(layer, canonical)


def _oracle(
    prepared: _Prepared,
    x: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    artifact = ReplayArtifact(
        manifest={"kind": "moe", "ep_rank": 1, "activation_limit": 10.0},
        tensors={
            **prepared.canonical,
            "x": x.cpu(),
            "topk_ids": ids.cpu(),
            "topk_weights": weights.cpu(),
        },
        sha256="",
    )
    # CPU FP32 matmul prevents TF32 configuration from weakening the oracle.
    return reference_output(
        artifact, activation="fp8", device=torch.device("cpu"), rows_per_chunk=64
    )


def _apply(
    prepared: _Prepared,
    x: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    return humming_mxfp8_precomputed_moe_apply(
        PLAN,
        x,
        prepared.layer,
        None,
        topk_weights=weights,
        topk_ids=ids,
        do_finalize=True,
        enable_pdl=False,
    )


@pytest.mark.parametrize("tokens", [1, 7, 33])
@pytest.mark.parametrize("stage", ["gate_up", "down"])
def test_fixed_quantized_gemm_matches_fp32_without_activation_error(
    prepared: _Prepared, tokens: int, stage: str
) -> None:
    state = prepared.layer._moe_backend_state
    if stage == "gate_up":
        plan, prefix, top_k = state.gate_up, "w13", TOP_K
    else:
        plan, prefix, top_k = state.down, "w2", 1
    size_k, size_n = plan.state.size_k, plan.state.size_n
    generator = torch.Generator().manual_seed(105 + tokens)
    codes = torch.randint(-4, 5, (tokens, size_k), generator=generator).to(
        torch.float8_e4m3fn
    )
    scales = torch.exp2(
        torch.randint(-4, 1, (tokens, size_k // 32), generator=generator).float()
    )
    ids = (torch.arange(tokens * top_k) % EXPERTS).reshape(tokens, top_k)
    if top_k > 1:
        ids[:, 1] = ids[:, 0]
    global_ids = (ids + EXPERT_START).to(torch.int32)
    route_count = global_ids.numel()
    routing = align_humming_routes(
        global_ids.cuda(),
        num_local_experts=EXPERTS,
        expert_start=EXPERT_START,
        block_size=plan.block_m(route_count),
    )
    output = torch.full(
        (route_count, size_n), float("nan"), device="cuda", dtype=torch.bfloat16
    )
    actual = humming_indexed_gemm(
        plan,
        codes.cuda(),
        scales.cuda(),
        output,
        *routing,
        top_k=top_k,
        valid_shape_m=route_count,
    )
    dequantized_a = (codes.float().reshape(tokens, -1, 32) * scales[..., None]).reshape(
        tokens, size_k
    )
    dequantized_b = dequantize_weight(
        prepared.canonical[f"{prefix}_weight"],
        prepared.canonical[f"{prefix}_weight_scale_inv"],
    )
    expected = torch.stack(
        [
            dequantized_a[route // top_k] @ dequantized_b[expert].T
            for route, expert in enumerate(ids.reshape(-1).tolist())
        ]
    ).to(torch.bfloat16)
    # Small integer FP8 codes and bounded power-of-two scales make these
    # reductions exactly representable in FP32, including their BF16 rounding.
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    assert actual.data_ptr() == output.data_ptr()


@pytest.mark.parametrize("stage", ["gate_up", "down"])
@pytest.mark.parametrize("tokens", [19, 65])
def test_fixed_fp8_exponents_preserve_small_terms(prepared, stage, tokens):
    state = prepared.layer._moe_backend_state
    plan = state.gate_up if stage == "gate_up" else state.down
    prefix = "w13" if stage == "gate_up" else "w2"
    top_k = TOP_K if stage == "gate_up" else 1
    generator = torch.Generator().manual_seed(34)
    values = (
        torch.randn(tokens, plan.state.size_k, generator=generator) * 1.25
    ).bfloat16()
    values[0, :32] = 0
    values[1, :32] = 1e-5
    values[2, 0] = 64
    groups = values.float().unflatten(-1, (-1, 32))
    scales = torch.exp2(
        torch.ceil(torch.log2(groups.abs().amax(-1).clamp_min(1e-4) / 448))
    )
    codes = (groups / scales[..., None]).to(torch.float8_e4m3fn).flatten(-2)
    operands = (
        codes.double().unflatten(-1, (-1, 32)) * scales.double()[..., None]
    ).flatten(-2)
    weights = dequantize_weight(
        prepared.canonical[prefix + "_weight"],
        prepared.canonical[prefix + "_weight_scale_inv"],
    ).double()
    ids = (torch.arange(tokens * top_k) % EXPERTS).reshape(tokens, top_k)
    global_ids = (ids + EXPERT_START).to(torch.int32).cuda()
    routes = align_humming_routes(
        global_ids,
        num_local_experts=EXPERTS,
        expert_start=EXPERT_START,
        block_size=plan.block_m(global_ids.numel()),
    )
    output = torch.empty(
        (global_ids.numel(), plan.state.size_n), device="cuda", dtype=torch.bfloat16
    )
    actual = humming_indexed_gemm(
        plan,
        codes.cuda(),
        scales.cuda(),
        output,
        *routes,
        top_k=top_k,
        valid_shape_m=global_ids.numel(),
    )
    expected = torch.stack(
        [
            operands[r // top_k] @ weights[e].T
            for r, e in enumerate(ids.flatten().tolist())
        ]
    ).bfloat16()
    # Dyadic E4M3 operands with bounded power-of-two weight scales make the
    # FP64 reference exact here. Integer-only operands missed the WGMMA loss.
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("code", [0, 255])
def test_unsupported_scale_codes_fail_before_upstream_preparation(code: int) -> None:
    _require_sm90()
    weight = torch.zeros((1, 64, 64), device="cuda").to(torch.float8_e4m3fn)
    scales = torch.full((1, 64, 2), 127, device="cuda", dtype=torch.uint8)
    scales[0, -1, -1] = code
    with pytest.raises(ValueError, match="codes in \\[1, 254\\]"):
        prepare_humming_mxfp8_weights(weight, scales)


def _inputs(case: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = 0 if case == "empty" else 19
    generator = torch.Generator().manual_seed(34)
    x = (torch.randn(tokens, HIDDEN, generator=generator) * 1.25).to(torch.bfloat16)
    ids = (torch.arange(tokens * TOP_K) % EXPERTS).reshape(tokens, TOP_K)
    ids = (ids + EXPERT_START).to(torch.int64)
    weights = torch.rand(tokens, TOP_K, generator=generator)
    weights /= weights.sum(dim=-1, keepdim=True)
    if tokens:
        x[0, :32] = 0
        x[1, :32] = 1e-5
        x[2, 0] = 64
    if case == "duplicates":
        ids[:, 1] = ids[:, 0]
    elif case == "mixed_ep":
        ids[:, 0] = EXPERT_START - 1
        ids[::3, 1] = -1
        ids[1::4, 2] = EXPERT_START + EXPERTS
    elif case == "all_nonlocal":
        ids[:] = torch.tensor([EXPERT_START - 1, EXPERT_START + EXPERTS, -1])
    else:
        assert case in {"ordinary", "empty"}
    return x, ids, weights


@pytest.mark.parametrize(
    "case", ["ordinary", "duplicates", "mixed_ep", "all_nonlocal", "empty"]
)
def test_complete_moe_matches_same_quantization_rule_and_repeats_exactly(
    prepared: _Prepared, case: str
) -> None:
    x, ids, weights = _inputs(case)
    expected = _oracle(prepared, x, ids, weights)
    gpu_x, gpu_ids, gpu_weights = x.cuda(), ids.cuda(), weights.cuda()
    first = _apply(prepared, gpu_x, gpu_ids, gpu_weights)
    second = _apply(prepared, gpu_x, gpu_ids, gpu_weights)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    if case in {"all_nonlocal", "empty"}:
        torch.testing.assert_close(first.cpu(), expected, rtol=0, atol=0)
    else:
        # The FP32 SiLU approximation may move a BF16 midpoint and then one
        # FP8 code. Keep both elementwise and global error bounds tight.
        torch.testing.assert_close(first.cpu(), expected, rtol=8e-3, atol=8e-3)
        error = (first.cpu().float() - expected.float()).norm()
        assert error / expected.float().norm().clamp_min(1e-30) < 2e-3


@pytest.mark.parametrize("weighted_routes", [False, True])
def test_topk_rounding_and_nonlocal_nan_masking(weighted_routes: bool) -> None:
    _require_sm90()
    ids = torch.tensor([[4, 4, 5, 3], [6, 7, -1, 4]], dtype=torch.int64)
    values = torch.arange(ids.numel() * 64).reshape(*ids.shape, 64).float()
    values = (values % 17 - 8) * 0.0078125 + 0.501953125
    weights = torch.tensor(
        [
            [0.5009765625, 0.3341, 0.1653, float("nan")],
            [0.25048828125, 0.6671, float("nan"), 0.0827],
        ]
    )
    local = (ids >= EXPERT_START) & (ids < EXPERT_START + EXPERTS)
    values[~local] = float("nan")
    rounded = (
        (
            values.to(torch.bfloat16).float()
            * weights.to(torch.bfloat16).float()[..., None]
        )
        .to(torch.bfloat16)
        .float()
    )
    rounded[~local] = 0
    expected = torch.zeros(2, 64)
    for slot in range(ids.shape[1]):
        expected += rounded[:, slot]
    expected = expected.to(torch.bfloat16)
    supplied = (rounded if weighted_routes else values).clone()
    supplied[~local] = float("nan")
    actual = finalize_bf16_routes(
        supplied.reshape(-1, 64).cuda(),
        weights.cuda(),
        ids.cuda(),
        expert_start=EXPERT_START,
        num_local_experts=EXPERTS,
        weighted_routes=weighted_routes,
    )
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    unrounded = values * weights[..., None]
    unrounded[~local] = 0
    assert not torch.equal(unrounded.sum(dim=1).to(torch.bfloat16), expected)


def test_complete_moe_graph_replay_refreshes_inputs_and_remote_routes(
    prepared: _Prepared,
) -> None:
    x, ids, weights = _inputs("ordinary")
    gpu_x, gpu_ids, gpu_weights = x.cuda(), ids.cuda(), weights.cuda()

    def run():
        return _apply(prepared, gpu_x, gpu_ids, gpu_weights)

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    for case in ("ordinary", "all_nonlocal", "duplicates"):
        x, ids, weights = _inputs(case)
        x = (x.float() * 0.875).to(torch.bfloat16)
        gpu_x.copy_(x)
        gpu_ids.copy_(ids)
        gpu_weights.copy_(weights)
        graph.replay()
        eager = run()
        torch.testing.assert_close(captured, eager, rtol=0, atol=0)
        expected = _oracle(prepared, x, ids, weights)
        torch.testing.assert_close(captured.cpu(), expected, rtol=8e-3, atol=8e-3)
