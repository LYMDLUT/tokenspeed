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

"""Public v0.29 single-group sigmoid routing semantics.

The mathematical control permits FP32 rounding; separate exact fixtures check
the warp numerator reconstruction and FP32 scale-before-multiply contract.
"""

import builtins

import pytest
import tokenspeed_kernel.ops.moe.cuda_public_v029_topk as public_adapter
import tokenspeed_kernel.selection as selection
import tokenspeed_kernel.thirdparty.cuda.public_v029_topk as public_topk
import torch
from tokenspeed_kernel.ops.moe import moe_topk
from tokenspeed_kernel.platform import Platform
from tokenspeed_kernel.registry import KernelRegistry, Priority
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)
if not Platform.get().is_nvidia:
    pytest.skip("NVIDIA-only CUDA implementation", allow_module_level=True)


def _inputs(tokens: int, experts: int):
    # Neighboring logits stay far enough apart to give unambiguous ranks.
    # A small bias limits cancellation when the warp path subtracts it again.
    # Scatter the high scores across workers/chunks and make the last expert
    # the maximum; a kernel that only reads its first chunk must fail.
    rank = ((torch.arange(experts) + 1) * 37) % max(experts, 1)
    logits = 1.5 - 3 * rank.float() / max(experts - 1, 1)
    row_offset = (torch.arange(tokens, dtype=torch.float32) % 7 - 3) / 100
    logits = logits.unsqueeze(0) + row_offset.unsqueeze(1)
    bias = torch.full((experts,), 1 / 64, dtype=torch.float32)
    return logits.cuda(), bias.cuda()


def _mathematical_control(logits, bias, *, topk, normalize, scale):
    scores = logits.cpu().double().sigmoid()
    choice = scores + bias.cpu().double()
    ids = choice.argsort(dim=-1, descending=True, stable=True)[:, :topk]
    weights = scores.gather(1, ids)
    if normalize:
        weights /= weights.sum(dim=-1, keepdim=True)
    # The launch interface rounds the scalar before any weight arithmetic.
    scale32 = torch.tensor(scale, dtype=torch.float32).item()
    return weights * scale32, ids.to(torch.int32)


def _assert_math(weights, ids, expected_weights, expected_ids, device):
    assert weights.dtype == torch.float32 and ids.dtype == torch.int32
    assert weights.device == ids.device == device
    assert torch.equal(ids.cpu(), expected_ids)
    # Bounded logits and small bias allow tanh rounding, the warp's bias
    # subtraction, a <=22-term FP32 sum, division, and final multiplication.
    # This checks the mathematical contract, not another binary's instruction
    # sequence; the exact arithmetic boundary tests below are separate.
    torch.testing.assert_close(
        weights.cpu().double(), expected_weights, rtol=1e-6, atol=2e-7
    )


@pytest.mark.parametrize(
    "tokens,experts,topk,normalize,scale",
    [
        (1, 1, 1, False, 1.25),
        (30, 8, 8, True, 2.827),
        (17, 129, 8, True, 1.25),
        (1, 257, 8, False, 2.827),
        (17, 385, 8, True, 2.827),
        (1, 512, 22, True, 2.827),
        (17, 513, 16, False, 1.25),
        (1, 769, 16, True, 2.827),
        (17, 897, 16, False, 2.827),
        (1, 1024, 16, True, 1.25),
        (1025, 256, 8, True, 1.25),
    ],
)
def test_tier_boundaries_match_mathematical_control(
    tokens, experts, topk, normalize, scale
):
    logits, bias = _inputs(tokens, experts)
    weights, ids = public_topk.single_group_sigmoid_topk(
        logits,
        bias,
        topk=topk,
        renormalize=normalize,
        routed_scaling_factor=scale,
    )
    expected_weights, expected_ids = _mathematical_control(
        logits, bias, topk=topk, normalize=normalize, scale=scale
    )
    assert torch.all(expected_ids[:, 0] == experts - 1)
    _assert_math(weights, ids, expected_weights, expected_ids, logits.device)


@pytest.mark.parametrize("normalize", [False, True])
def test_bias_changes_selection_without_becoming_a_routing_weight(normalize):
    logits = torch.tensor(
        [[-0.2, 0.2, -0.1, 0.1, -0.3, 0.3, -0.4, 0.4]],
        dtype=torch.float32,
        device="cuda",
    )
    bias = torch.tensor(
        [0.5, -0.5, 0.25, -0.25, 0.125, -0.125, 0.0625, -0.0625],
        dtype=torch.float32,
        device="cuda",
    )
    weights, ids = public_topk.single_group_sigmoid_topk(
        logits,
        bias,
        topk=3,
        renormalize=normalize,
        routed_scaling_factor=2.827,
    )
    expected_weights, expected_ids = _mathematical_control(
        logits, bias, topk=3, normalize=normalize, scale=2.827
    )
    assert torch.equal(expected_ids, torch.tensor([[0, 2, 4]], dtype=torch.int32))
    _assert_math(weights, ids, expected_weights, expected_ids, logits.device)


@pytest.mark.parametrize("tokens,experts,topk", [(30, 8, 8), (1, 256, 8), (1, 512, 22)])
def test_equal_scores_choose_lowest_expert_ids(tokens, experts, topk):
    logits = torch.zeros((tokens, experts), device="cuda", dtype=torch.float32)
    bias = torch.zeros(experts, device="cuda", dtype=torch.float32)
    weights, ids = public_topk.single_group_sigmoid_topk(
        logits,
        bias,
        topk=topk,
        renormalize=True,
        routed_scaling_factor=1.0,
    )
    expected_ids = torch.arange(topk, dtype=torch.int32).expand(tokens, -1)
    assert torch.equal(ids.cpu(), expected_ids)
    assert torch.equal(weights, torch.full_like(weights, 1 / topk))


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize(
    "tokens,experts,keeps_numerator",
    [
        (30, 8, False),
        (1, 256, True),
        (1025, 256, False),
        (1025, 769, True),
    ],
)
def test_warp_reconstruction_and_block_preservation(
    normalize, tokens, experts, keeps_numerator
):
    logits = torch.zeros((tokens, experts), device="cuda", dtype=torch.float32)
    bias = torch.full((experts,), 2**24, device="cuda", dtype=torch.float32)
    weights, ids = public_topk.single_group_sigmoid_topk(
        logits,
        bias,
        topk=8,
        renormalize=normalize,
        routed_scaling_factor=1.0,
    )
    # FP32(0.5 + 2**24) == 2**24. The warp path recovers a zero numerator
    # by subtracting bias; the block path retains the original sigmoid 0.5.
    # The 896/16 tier keeps its block path even above 1024 tokens.
    expected_weight = (1 / 8 if normalize else 0.5) if keeps_numerator else 0.0
    assert torch.equal(weights, torch.full_like(weights, expected_weight))
    assert torch.equal(ids.cpu(), torch.arange(8, dtype=torch.int32).expand(tokens, -1))


@pytest.mark.parametrize("experts", [8, 256])
def test_unnormalized_scaling_uses_fp32_scalar(experts):
    logits, bias = _inputs(30, experts)

    def route(scale):
        return public_topk.single_group_sigmoid_topk(
            logits,
            bias,
            topk=8,
            renormalize=False,
            routed_scaling_factor=scale,
        )

    base, base_ids = route(1.0)
    weights, ids = route(2.827)
    scale32 = torch.tensor(2.827, dtype=torch.float32, device=base.device)
    expected = base * scale32
    double_scaled = (base.double() * 2.827).float()
    assert not torch.equal(
        expected, double_scaled
    ), "fixture must distinguish FP32 scaling from double scaling"
    assert torch.equal(ids, base_ids)
    assert torch.equal(weights, expected)


@pytest.mark.parametrize("experts", [8, 256])
def test_normalization_divides_fp32_scale_before_multiplying(experts):
    logits = torch.zeros((1, experts), device="cuda", dtype=torch.float32)
    bias = torch.zeros(experts, device="cuda", dtype=torch.float32)
    weights, ids = public_topk.single_group_sigmoid_topk(
        logits,
        bias,
        topk=3,
        renormalize=True,
        routed_scaling_factor=2.827,
    )
    numerator = torch.tensor(0.5, dtype=torch.float32, device=logits.device)
    denominator = torch.tensor(1.5, dtype=torch.float32, device=logits.device)
    scale32 = torch.tensor(2.827, dtype=torch.float32, device=logits.device)
    expected = numerator * (scale32 / denominator)
    quotient_first = (numerator / denominator) * scale32
    double_scaled = (numerator.double() * (2.827 / 1.5)).float()
    assert not torch.equal(expected, quotient_first)
    assert not torch.equal(expected, double_scaled)
    assert torch.equal(weights, expected.expand_as(weights))
    assert torch.equal(ids.cpu(), torch.tensor([[0, 1, 2]], dtype=torch.int32))


def _forbidden_loader():
    pytest.fail("empty input or invalid metadata reached the CUDA loader")


def test_empty_input_does_not_load_cuda(monkeypatch):
    monkeypatch.setattr(public_topk, "_load_public_v029_topk_module", _forbidden_loader)
    logits, bias = _inputs(0, 512)
    weights, ids = public_topk.single_group_sigmoid_topk(
        logits,
        bias,
        topk=22,
        renormalize=True,
        routed_scaling_factor=2.827,
    )
    converted, converted_ids = public_adapter.cuda_public_v029_sigmoid_bias_topk(
        router_logits=logits,
        correction_bias=bias,
        topk=22,
        routed_scaling_factor=2.827,
        normalize_topk_weights=False,
        weights_dtype=torch.bfloat16,
    )
    assert weights.shape == ids.shape == (0, 22)
    assert converted.shape == converted_ids.shape == (0, 22)
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
        with pytest.raises(RuntimeError, match="capture"):
            public_topk._load_public_v029_topk_module.__wrapped__()


@pytest.fixture
def routing_selection(monkeypatch):
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
    [(False, torch.float32), (True, torch.float16), (True, torch.bfloat16)],
)
def test_named_override_and_output_cast_match_direct_call(
    normalize, weights_dtype, routing_selection
):
    logits, bias = _inputs(30, 8)
    base, base_ids = public_topk.single_group_sigmoid_topk(
        logits,
        bias,
        topk=8,
        renormalize=normalize,
        routed_scaling_factor=2.827,
    )
    direct, direct_ids = public_adapter.cuda_public_v029_sigmoid_bias_topk(
        router_logits=logits,
        correction_bias=bias,
        topk=8,
        routed_scaling_factor=2.827,
        normalize_topk_weights=normalize,
        weights_dtype=weights_dtype,
    )
    weights, ids = moe_topk(
        logits,
        8,
        score_function="sigmoid",
        selection_method="topk",
        renormalize=normalize,
        routed_scaling_factor=2.827,
        correction_bias=bias,
        topk_weights_dtype=weights_dtype,
        override="cuda_public_v029_sigmoid_bias_topk",
    )
    assert weights.dtype == direct.dtype == weights_dtype
    assert ids.dtype == direct_ids.dtype == torch.int32
    assert torch.equal(weights, direct) and torch.equal(weights, base.to(weights_dtype))
    assert torch.equal(ids, direct_ids) and torch.equal(ids, base_ids)


def test_import_keeps_existing_default_selection(routing_selection):
    spec = KernelRegistry.get().get_by_name("cuda_public_v029_sigmoid_bias_topk")
    assert spec is not None and spec.solution == "cuda"
    assert spec.priority == Priority.PORTABLE
    selected = selection.select_kernel(
        "moe",
        "sigmoid_bias_topk",
        format_signature(router_logits=dense_tensor_format(torch.float32)),
        traits={"tokens": 30, "experts": 8, "topk": 8},
    )
    assert selected.name == "triton_minimax_sigmoid_bias_topk"


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
def test_invalid_tensor_metadata_rejected_before_loading(case, monkeypatch):
    monkeypatch.setattr(public_topk, "_load_public_v029_topk_module", _forbidden_loader)
    logits, bias = _inputs(2, 8)
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
        public_topk.single_group_sigmoid_topk(
            logits,
            bias,
            topk=8,
            renormalize=True,
            routed_scaling_factor=2.827,
        )


def test_different_cuda_devices_rejected_before_loading(monkeypatch):
    count = torch.cuda.device_count()
    if count < 2:
        pytest.skip("two CUDA devices required to exercise device mismatch")
    monkeypatch.setattr(public_topk, "_load_public_v029_topk_module", _forbidden_loader)
    logits, bias = _inputs(2, 8)
    bias = bias.to(torch.device("cuda", (logits.device.index + 1) % count))
    with pytest.raises(ValueError, match="device"):
        public_topk.single_group_sigmoid_topk(
            logits,
            bias,
            topk=8,
            renormalize=True,
            routed_scaling_factor=2.827,
        )


@pytest.mark.parametrize(
    "experts,topk,scale",
    [
        (0, 1, 1.0),
        (8, 0, 1.0),
        (8, 9, 1.0),
        (1025, 8, 1.0),
        (512, 23, 1.0),
        (513, 17, 1.0),
        (513, 22, 1.0),
        (8, 8, float("nan")),
        (8, 8, float("inf")),
        (8, 8, 1e39),
        (8, 8, -1e39),
    ],
)
def test_unsupported_tier_or_scale_rejected_before_loading(
    experts, topk, scale, monkeypatch
):
    monkeypatch.setattr(public_topk, "_load_public_v029_topk_module", _forbidden_loader)
    logits, bias = _inputs(2, experts)
    with pytest.raises((TypeError, ValueError)):
        public_topk.single_group_sigmoid_topk(
            logits,
            bias,
            topk=topk,
            renormalize=True,
            routed_scaling_factor=scale,
        )


@pytest.mark.parametrize("tokens,experts", [(30, 8), (1, 256)])
def test_warmed_cuda_graph_replays_current_inputs(tokens, experts):
    logits, bias = _inputs(tokens, experts)

    def route():
        return public_topk.single_group_sigmoid_topk(
            logits,
            bias,
            topk=8,
            renormalize=True,
            routed_scaling_factor=2.827,
        )

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
            logits.mul_(-0.75).add_(0.125)
            bias.add_(1 / 32)
            next_weights, next_ids = route()
            assert not torch.equal(next_weights, expected_weights)
            assert not torch.equal(next_ids, expected_ids)
            expected_weights, expected_ids = next_weights, next_ids
        weights.fill_(float("nan"))
        ids.fill_(-1)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(weights, expected_weights)
        assert torch.equal(ids, expected_ids)
