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

"""Routing semantics for the explicitly imported public CUDA candidate.

The FP64 control checks the mathematical contract, not bitwise equivalence
with another implementation's sigmoid or reduction instructions.
"""

import builtins

import pytest
import tokenspeed_kernel.ops.moe.cuda_public_topk as public_adapter
import tokenspeed_kernel.selection as selection
import tokenspeed_kernel.thirdparty.cuda.public_grouped_topk as public_grouped
import torch
from tokenspeed_kernel.ops.moe import moe_topk
from tokenspeed_kernel.ops.moe.sigmoid_topk import triton_minimax_sigmoid_bias_topk
from tokenspeed_kernel.platform import Platform
from tokenspeed_kernel.registry import KernelRegistry, Priority
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)
if not Platform.get().is_nvidia:
    pytest.skip("NVIDIA-only CUDA implementation", allow_module_level=True)


def _inputs(tokens: int, experts: int, groups: int):
    """Bound logits and separate selection scores well beyond FP32 error."""
    row = torch.arange(tokens, dtype=torch.float32).unsqueeze(1)
    expert = torch.arange(experts, dtype=torch.float32).unsqueeze(0)
    logits = ((row * 7 + expert * 3) % 33 - 16) / 8
    per_group = experts // groups
    expert_id = torch.arange(experts)
    local_rank = (expert_id % per_group * 5) % per_group
    group_rank = (expert_id // per_group * 3) % groups
    bias = (2 * local_rank + 4 * per_group * group_rank).float()
    return logits.cuda(), bias.cuda()


def _fp64_control(
    logits: torch.Tensor,
    bias: torch.Tensor,
    *,
    groups: int,
    selected_groups: int,
    topk: int,
    normalize: bool,
    scale: float,
):
    """Use exact input values, stable ranks, and unbiased sigmoid weights."""
    scores = logits.cpu().double().sigmoid()
    choice = scores + bias.cpu().double()
    tokens, experts = scores.shape
    grouped = choice.reshape(tokens, groups, experts // groups)
    group_scores = grouped.sort(dim=-1, descending=True).values[..., :2].sum(-1)
    group_ids = group_scores.argsort(dim=-1, descending=True, stable=True)
    selected = torch.zeros_like(group_scores, dtype=torch.bool)
    selected.scatter_(1, group_ids[:, :selected_groups], True)
    allowed = selected.repeat_interleave(experts // groups, dim=1)
    ranked = choice.masked_fill(~allowed, -torch.inf)
    ids = ranked.argsort(dim=-1, descending=True, stable=True)[:, :topk]
    weights = scores.gather(1, ids)
    if normalize:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights * scale, ids.to(torch.int32)


def _assert_control(weights, ids, expected_weights, expected_ids, device):
    assert weights.dtype == torch.float32
    assert ids.dtype == torch.int32
    assert weights.device == ids.device == device
    assert torch.equal(ids.cpu(), expected_ids)
    # With logits in [-2, 2], sigmoid stays away from saturation. This permits
    # FP32 tanh, a <=32-term reduction, division, and final scaling/rounding,
    # while remaining sensitive to using biased scores as routing weights.
    torch.testing.assert_close(
        weights.cpu().double(), expected_weights, rtol=1e-6, atol=2e-7
    )


@pytest.mark.parametrize(
    "tokens,experts,groups,selected_groups,topk,normalize,scale",
    [
        (1, 8, 1, 1, 1, False, 1.25),
        (30, 8, 1, 1, 8, True, 2.827),
        (17, 64, 1, 1, 16, True, 1.25),
        (65, 128, 1, 1, 32, False, 2.827),
        (1, 256, 1, 1, 8, True, 2.827),
        (17, 64, 4, 2, 16, False, 2.827),
        (17, 70, 5, 2, 16, False, 1.25),
        (30, 128, 8, 3, 8, True, 1.25),
        (65, 256, 4, 2, 32, True, 2.827),
    ],
)
def test_grouped_sigmoid_matches_fp64_control(
    tokens, experts, groups, selected_groups, topk, normalize, scale
):
    logits, bias = _inputs(tokens, experts, groups)
    weights, ids = public_grouped.grouped_sigmoid_topk(
        logits,
        bias,
        num_expert_group=groups,
        topk_group=selected_groups,
        topk=topk,
        renormalize=normalize,
        routed_scaling_factor=scale,
    )
    expected_weights, expected_ids = _fp64_control(
        logits,
        bias,
        groups=groups,
        selected_groups=selected_groups,
        topk=topk,
        normalize=normalize,
        scale=scale,
    )
    _assert_control(weights, ids, expected_weights, expected_ids, logits.device)
    if normalize:
        torch.testing.assert_close(
            weights.sum(-1),
            torch.full((tokens,), scale, device=logits.device),
            rtol=1e-6,
            atol=2e-7,
        )


@pytest.mark.parametrize("normalize", [False, True])
def test_scaling_uses_double_scalar_before_final_fp32_rounding(normalize):
    logits, bias = _inputs(30, 64, 1)

    def route(scale):
        return public_grouped.grouped_sigmoid_topk(
            logits,
            bias,
            num_expert_group=1,
            topk_group=1,
            topk=32,
            renormalize=normalize,
            routed_scaling_factor=scale,
        )

    base, base_ids = route(1.0)
    actual, actual_ids = route(2.827)
    assert base.dtype == actual.dtype == torch.float32
    # Scale one preserves the FP32 sigmoid (or FP32 quotient) immediately
    # before multiplication by the double scalar in the public algorithm.
    expected = (base.double() * 2.827).float()
    fp32_scale = torch.tensor(2.827, dtype=torch.float32, device=base.device)
    prematurely_rounded = base * fp32_scale
    assert not torch.equal(
        expected, prematurely_rounded
    ), "fixture must distinguish double scaling from an early FP32 scale cast"
    assert torch.equal(actual_ids, base_ids)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("normalize", [False, True])
def test_group_selection_uses_top_two_and_bias_does_not_change_weights(normalize):
    logits = torch.tensor(
        [[-1, 1, -0.5, 0.5, -1.5, 1.5, -2, 2]],
        dtype=torch.float32,
        device="cuda",
    )
    bias = torch.tensor([9, 0, 5, 5, 6, 3.5, 8, -1], dtype=torch.float32, device="cuda")
    # Group sums are 10, 11, 10.5, and 8: select groups 1 and 2. The largest
    # individual scores (experts 0 and 6) belong to excluded groups.
    weights, ids = public_grouped.grouped_sigmoid_topk(
        logits,
        bias,
        num_expert_group=4,
        topk_group=2,
        topk=2,
        renormalize=normalize,
        routed_scaling_factor=2.827,
    )
    expected_ids = torch.tensor([[4, 3]], dtype=torch.int32)
    expected_weights = logits.cpu().double().sigmoid()[:, [4, 3]]
    if normalize:
        expected_weights /= expected_weights.sum(-1, keepdim=True)
    _assert_control(weights, ids, expected_weights * 2.827, expected_ids, logits.device)


@pytest.mark.parametrize(
    "experts,groups,selected_groups,topk", [(8, 1, 1, 8), (64, 4, 2, 32)]
)
def test_equal_scores_choose_lowest_group_and_expert_ids(
    experts, groups, selected_groups, topk
):
    logits = torch.zeros((2, experts), dtype=torch.float32, device="cuda")
    bias = torch.zeros(experts, dtype=torch.float32, device="cuda")
    weights, ids = public_grouped.grouped_sigmoid_topk(
        logits,
        bias,
        num_expert_group=groups,
        topk_group=selected_groups,
        topk=topk,
        renormalize=True,
        routed_scaling_factor=1.0,
    )
    expected_ids = torch.arange(topk, dtype=torch.int32).expand(2, -1)
    assert torch.equal(ids.cpu(), expected_ids)
    assert torch.equal(weights, torch.full_like(weights, 1 / topk))


def test_saturated_negative_logits_preserve_zero_weights():
    logits = torch.full((1, 8), -100.0, dtype=torch.float32, device="cuda")
    bias = torch.arange(8, dtype=torch.float32, device="cuda") * 2
    weights, ids = public_grouped.grouped_sigmoid_topk(
        logits,
        bias,
        num_expert_group=1,
        topk_group=1,
        topk=8,
        renormalize=True,
        routed_scaling_factor=2.827,
    )
    # The CUDA tanh sigmoid saturates to zero; the public algorithm's small
    # positive denominator leaves these weights at zero, not a uniform row.
    assert torch.equal(ids.cpu(), torch.arange(7, -1, -1, dtype=torch.int32)[None])
    assert torch.equal(weights, torch.zeros_like(weights))


def _forbidden_loader():
    pytest.fail("input validation or an empty batch reached the CUDA loader")


def test_empty_batch_never_loads_cuda_module(monkeypatch):
    monkeypatch.setattr(
        public_grouped, "_load_public_grouped_topk_module", _forbidden_loader
    )
    logits, bias = _inputs(0, 64, 4)
    weights, ids = public_grouped.grouped_sigmoid_topk(
        logits,
        bias,
        num_expert_group=4,
        topk_group=2,
        topk=16,
        renormalize=True,
        routed_scaling_factor=2.827,
    )
    converted, converted_ids = public_adapter.cuda_public_sigmoid_bias_topk(
        router_logits=logits,
        correction_bias=bias,
        topk=16,
        routed_scaling_factor=2.827,
        normalize_topk_weights=False,
        weights_dtype=torch.bfloat16,
    )
    assert weights.shape == ids.shape == (0, 16)
    assert converted.shape == converted_ids.shape == (0, 16)
    assert weights.dtype == torch.float32 and converted.dtype == torch.bfloat16
    assert ids.dtype == converted_ids.dtype == torch.int32
    assert weights.device == ids.device == logits.device
    assert converted.device == converted_ids.device == logits.device


def test_cold_capture_rejected_before_importing_compiler(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "flashinfer" or name.startswith("flashinfer."):
            pytest.fail("cold capture reached the optional compiler import")
        return real_import(name, *args, **kwargs)

    with monkeypatch.context() as guard:
        guard.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
        guard.setattr(builtins, "__import__", guarded_import)
        with pytest.raises(RuntimeError, match="before CUDA graph capture"):
            # Bypass only the cache, preserving other tests' warmed modules.
            public_grouped._load_public_grouped_topk_module.__wrapped__()


@pytest.fixture
def routing_selection(monkeypatch):
    """Exercise default ranking without ambient user overrides or autotuning."""
    monkeypatch.delenv(
        "TOKENSPEED_KERNEL_OVERRIDE_MOE_SIGMOID_BIAS_TOPK", raising=False
    )
    monkeypatch.delitem(
        selection._global_overrides, ("moe", "sigmoid_bias_topk"), raising=False
    )
    monkeypatch.setattr(selection, "_policy", selection.SelectionPolicy())
    KernelRegistry.get().clear_cache()
    yield
    KernelRegistry.get().clear_cache()


@pytest.mark.parametrize(
    "normalize,weights_dtype",
    [
        (False, torch.float32),
        (True, torch.float16),
        (True, torch.bfloat16),
        (False, torch.float64),
    ],
)
def test_explicit_override_and_output_cast_match_direct_call(
    normalize, weights_dtype, routing_selection
):
    logits, bias = _inputs(17, 64, 1)
    base_weights, base_ids = public_grouped.grouped_sigmoid_topk(
        logits,
        bias,
        num_expert_group=1,
        topk_group=1,
        topk=16,
        renormalize=normalize,
        routed_scaling_factor=2.827,
    )
    direct_weights, direct_ids = public_adapter.cuda_public_sigmoid_bias_topk(
        router_logits=logits,
        correction_bias=bias,
        topk=16,
        routed_scaling_factor=2.827,
        normalize_topk_weights=normalize,
        weights_dtype=weights_dtype,
    )
    weights, ids = moe_topk(
        logits,
        16,
        score_function="sigmoid",
        selection_method="topk",
        renormalize=normalize,
        routed_scaling_factor=2.827,
        correction_bias=bias,
        topk_weights_dtype=weights_dtype,
        override="cuda_public_sigmoid_bias_topk",
    )
    assert weights.dtype == direct_weights.dtype == weights_dtype
    assert ids.dtype == direct_ids.dtype == torch.int32
    assert torch.equal(weights, direct_weights)
    assert torch.equal(weights, base_weights.to(weights_dtype))
    assert torch.equal(ids, direct_ids) and torch.equal(ids, base_ids)


def test_import_keeps_existing_default_selection(routing_selection):
    spec = KernelRegistry.get().get_by_name("cuda_public_sigmoid_bias_topk")
    assert spec is not None and spec.solution == "cuda"
    assert spec.priority == Priority.PORTABLE
    selected = selection.select_kernel(
        "moe",
        "sigmoid_bias_topk",
        format_signature(router_logits=dense_tensor_format(torch.float32)),
        traits={"tokens": 30, "experts": 8, "topk": 8},
    )
    assert selected.name == "triton_minimax_sigmoid_bias_topk"
    logits, bias = _inputs(30, 8, 1)
    expected_weights, expected_ids = triton_minimax_sigmoid_bias_topk(
        router_logits=logits,
        correction_bias=bias,
        topk=8,
        routed_scaling_factor=2.827,
        normalize_topk_weights=True,
    )
    weights, ids = moe_topk(
        logits,
        8,
        score_function="sigmoid",
        selection_method="topk",
        renormalize=True,
        routed_scaling_factor=2.827,
        correction_bias=bias,
    )
    assert torch.equal(ids, expected_ids)
    assert torch.equal(weights, expected_weights)


@pytest.mark.parametrize(
    "case",
    [
        "logits_rank",
        "bias_rank",
        "bias_size",
        "logits_dtype",
        "bias_dtype",
        "logits_stride",
        "bias_stride",
        "cpu_inputs",
        "cpu_bias",
    ],
)
def test_invalid_tensor_contract_rejected_before_loading(case, monkeypatch):
    monkeypatch.setattr(
        public_grouped, "_load_public_grouped_topk_module", _forbidden_loader
    )
    logits, bias = _inputs(2, 8, 1)
    if case == "logits_rank":
        logits = logits.flatten()
    elif case == "bias_rank":
        bias = bias.unsqueeze(0)
    elif case == "bias_size":
        bias = bias[:-1]
    elif case == "logits_dtype":
        logits = logits.half()
    elif case == "bias_dtype":
        bias = bias.bfloat16()
    elif case == "logits_stride":
        logits = logits.t().contiguous().t()
    elif case == "bias_stride":
        bias = bias.repeat_interleave(2)[::2]
    elif case == "cpu_inputs":
        logits, bias = logits.cpu(), bias.cpu()
    elif case == "cpu_bias":
        bias = bias.cpu()
    with pytest.raises((TypeError, ValueError)):
        public_grouped.grouped_sigmoid_topk(
            logits,
            bias,
            num_expert_group=1,
            topk_group=1,
            topk=8,
            renormalize=True,
            routed_scaling_factor=2.827,
        )


def test_different_cuda_devices_rejected_before_loading(monkeypatch):
    device_count = torch.cuda.device_count()
    if device_count < 2:
        pytest.skip("two CUDA devices required to exercise the device mismatch")
    monkeypatch.setattr(
        public_grouped, "_load_public_grouped_topk_module", _forbidden_loader
    )
    logits, bias = _inputs(2, 8, 1)
    other_device = torch.device("cuda", (logits.device.index + 1) % device_count)
    bias = bias.to(other_device)
    with pytest.raises(ValueError, match="share a device"):
        public_grouped.grouped_sigmoid_topk(
            logits,
            bias,
            num_expert_group=1,
            topk_group=1,
            topk=8,
            renormalize=True,
            routed_scaling_factor=2.827,
        )


@pytest.mark.parametrize(
    "experts,groups,selected_groups,topk,scale",
    [
        (8, 0, 1, 1, 1.0),
        (128, 64, 1, 1, 1.0),
        (8, 3, 1, 1, 1.0),
        (8, 8, 1, 1, 1.0),
        (8, 2, 0, 1, 1.0),
        (8, 2, 3, 1, 1.0),
        (8, 1, 1, 0, 1.0),
        (64, 1, 1, 33, 1.0),
        (8, 4, 1, 3, 1.0),
        (8, 1, 1, 8, float("nan")),
        (8, 1, 1, 8, float("inf")),
    ],
)
def test_invalid_routing_parameters_rejected_before_loading(
    experts, groups, selected_groups, topk, scale, monkeypatch
):
    monkeypatch.setattr(
        public_grouped, "_load_public_grouped_topk_module", _forbidden_loader
    )
    logits, bias = _inputs(2, experts, 1)
    with pytest.raises((TypeError, ValueError)):
        public_grouped.grouped_sigmoid_topk(
            logits,
            bias,
            num_expert_group=groups,
            topk_group=selected_groups,
            topk=topk,
            renormalize=True,
            routed_scaling_factor=scale,
        )


def test_cuda_graph_replay_matches_warmed_eager_call():
    logits, bias = _inputs(17, 64, 4)

    def route():
        return public_grouped.grouped_sigmoid_topk(
            logits,
            bias,
            num_expert_group=4,
            topk_group=2,
            topk=16,
            renormalize=True,
            routed_scaling_factor=2.827,
        )

    # Lazy compilation and CUDA initialization must happen before capture.
    expected_weights, expected_ids = route()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        route()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        weights, ids = route()
    for replay in range(2):
        if replay:
            # Preserve captured addresses while changing both the selected ids
            # and their unbiased weights; replay must consume current values.
            logits.add_(0.25)
            bias.neg_()
            next_weights, next_ids = route()
            assert not torch.equal(next_ids, expected_ids)
            assert not torch.equal(next_weights, expected_weights)
            expected_weights, expected_ids = next_weights, next_ids
        weights.fill_(float("nan"))
        ids.fill_(-1)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(weights, expected_weights)
        assert torch.equal(ids, expected_ids)
