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

import json
import math

import pytest
import torch
from tokenspeed_kernel.benchmark.mxfp8_replay import (
    ReplayArtifact,
    bitwise_equal,
    error_metrics,
    export_artifact,
    load_artifact,
    percentile,
    precision_verdict,
    quantize_dequantize_group32,
    reference_output,
    validate_artifact,
)


def linear_capture():
    manifest = {
        "schema_version": 1,
        "kind": "linear",
        "checkpoint": "test/model",
        "checkpoint_revision": "0123456789",
        "layer": "test.linear",
        "capture_revision": "abcdef",
        "tp_size": 1,
        "tp_rank": 0,
        "ep_size": 1,
        "ep_rank": 0,
    }
    x = torch.zeros(2, 64, dtype=torch.bfloat16)
    x[:, 0] = torch.tensor([1.0, 2.0])
    weight = torch.zeros(4, 64).to(torch.float8_e4m3fn)
    weight[:, 0] = 2.0
    scale = torch.full((4, 2), 127, dtype=torch.uint8)
    return manifest, {"x": x, "weight": weight, "weight_scale": scale}


def moe_capture():
    manifest, linear = linear_capture()
    manifest.update(
        kind="moe", num_experts=2, ep_size=2, ep_rank=1, activation_limit=10.0
    )
    gate_up = torch.zeros(1, 128, 64)
    gate_up[:, :64, 0] = 1.0
    gate_up[:, 64:, 0] = 2.0
    down = torch.zeros(1, 64, 64)
    down[:, :, 0] = 1.0
    return manifest, {
        "x": linear["x"],
        "w13_weight": gate_up.to(torch.float8_e4m3fn),
        "w13_weight_scale_inv": torch.full((1, 128, 2), 127, dtype=torch.uint8),
        "w2_weight": down.to(torch.float8_e4m3fn),
        "w2_weight_scale_inv": torch.full((1, 64, 2), 127, dtype=torch.uint8),
        "topk_ids": torch.tensor([[0, 1], [1, -1]], dtype=torch.int32),
        "topk_weights": torch.tensor(
            [[0.75, 0.25], [0.375, 0.625]], dtype=torch.float32
        ),
    }


def test_export_load_preserves_raw_e4m3_bytes_and_detects_corruption(tmp_path):
    manifest, tensors = linear_capture()
    destination = tmp_path / "capture"
    export_artifact(destination, manifest, tensors)
    result = load_artifact(destination)
    assert result.manifest["checkpoint_revision"] == manifest["checkpoint_revision"]
    assert torch.equal(
        result.tensors["weight"].view(torch.uint8), tensors["weight"].view(torch.uint8)
    )
    assert torch.equal(result.tensors["weight_scale"], tensors["weight_scale"])
    with pytest.raises(FileExistsError):
        export_artifact(destination, manifest, tensors)
    with (destination / "tensors.pt").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="SHA256"):
        load_artifact(destination)


@pytest.mark.parametrize("encoding", [0, 255])
def test_artifact_rejects_scale_encodings_unsupported_by_baseline(encoding):
    manifest, tensors = linear_capture()
    tensors["weight_scale"][0, 0] = encoding
    with pytest.raises(ValueError, match=r"\[1, 254\]"):
        validate_artifact(manifest, tensors)


def test_artifact_rejects_ambiguous_expert_shards_and_duplicate_routes():
    manifest, tensors = moe_capture()
    validate_artifact(manifest, tensors)
    with pytest.raises(ValueError, match="contiguous EP shards"):
        validate_artifact({**manifest, "num_experts": 3}, tensors)
    tensors["topk_ids"][0] = 1
    with pytest.raises(ValueError, match="repeat"):
        validate_artifact(manifest, tensors)


def test_group32_quantizer_does_not_share_scale_across_groups():
    values = torch.zeros(2, 64, dtype=torch.bfloat16)
    values[:, 0] = 1.0
    values[:, 32] = 256.0
    values[:, 1] = 2**-12
    restored = quantize_dequantize_group32(values)
    assert torch.equal(restored, values.float())
    assert torch.equal(
        quantize_dequantize_group32(torch.zeros_like(values)),
        torch.zeros_like(values).float(),
    )


@pytest.mark.parametrize("activation", ["bf16", "fp8"])
def test_linear_reference_matches_hand_computed_result(activation):
    manifest, tensors = linear_capture()
    artifact = ReplayArtifact(manifest, tensors, "unused")
    output = reference_output(
        artifact, activation=activation, device=torch.device("cpu"), rows_per_chunk=1
    )
    assert torch.equal(
        output, torch.tensor([[2.0] * 4, [4.0] * 4], dtype=torch.bfloat16)
    )


def test_moe_reference_filters_remote_experts_and_rounds_route_products():
    manifest, tensors = moe_capture()
    artifact = ReplayArtifact(manifest, tensors, "unused")
    output = reference_output(
        artifact, activation="bf16", device=torch.device("cpu"), rows_per_chunk=1
    )
    x = torch.tensor([1.0, 2.0])
    activation = (x * torch.sigmoid(x) * (2 * x)).to(torch.bfloat16).float()
    expected = (activation * torch.tensor([0.25, 0.375]).to(torch.bfloat16).float()).to(
        torch.bfloat16
    )
    assert torch.equal(output, expected[:, None].expand(2, 64))


def test_moe_reference_keeps_gate_lower_tail_and_clamps_both_up_sides():
    manifest, tensors = moe_capture()
    w13 = torch.zeros(1, 128, 64)
    w13[0, 0, 0], w13[0, 1, 0] = 12.0, -12.0
    w13[0, 64, 0], w13[0, 65, 0] = 16.0, -16.0
    w2 = torch.zeros(1, 64, 64)
    w2[0, 0, 0], w2[0, 1, 1] = 1.0, 1.0
    tensors["w13_weight"] = w13.to(torch.float8_e4m3fn)
    tensors["w2_weight"] = w2.to(torch.float8_e4m3fn)
    tensors["topk_ids"] = torch.tensor([[1, -1], [1, -1]], dtype=torch.int32)
    tensors["topk_weights"] = torch.tensor([[0.1001, 0.0], [0.3003, 0.0]])
    output = reference_output(
        ReplayArtifact(manifest, tensors, "unused"),
        activation="bf16",
        device=torch.device("cpu"),
        rows_per_chunk=2,
    )
    unweighted = torch.tensor(
        [
            [100.0 / (1 + math.exp(-10.0)), 120.0 / (1 + math.exp(12.0))],
            [100.0 / (1 + math.exp(-10.0)), 240.0 / (1 + math.exp(24.0))],
        ]
    ).to(torch.bfloat16)
    weights = tensors["topk_weights"][:, :1].to(torch.bfloat16)
    expected = (unweighted.float() * weights.float()).to(torch.bfloat16)
    assert bitwise_equal(output[:, :2], expected)
    assert torch.count_nonzero(output[:, 2:]) == 0


def test_error_metrics_report_zero_reference_without_invalid_json():
    metrics = error_metrics(torch.ones(4), torch.zeros(4))
    assert metrics["relative_l2"] is None
    assert metrics["error_l2"] == 2.0
    json.dumps(metrics, allow_nan=False)
    assert error_metrics(torch.zeros(4), torch.zeros(4))["relative_l2"] == 0.0
    with pytest.raises(ValueError, match="Non-finite"):
        error_metrics(torch.tensor([float("nan")]), torch.zeros(1))


def test_error_metrics_keep_finite_large_outputs_json_serializable():
    values = torch.full((4,), 1e20)
    metrics = error_metrics(values, values)
    assert metrics["relative_l2"] == 0.0
    assert metrics["reference_l2"] == pytest.approx(2e20)
    json.dumps(metrics, allow_nan=False)


def test_bitwise_comparison_distinguishes_signed_zero():
    assert not bitwise_equal(torch.tensor([0.0]), torch.tensor([-0.0]))
    assert bitwise_equal(torch.tensor([1.0]), torch.tensor([1.0]))


def test_bad_graph_output_cannot_pass_using_correct_eager_output():
    eager = torch.ones(2, 32, dtype=torch.bfloat16)
    bad_measured = torch.zeros_like(eager)
    verdict = precision_verdict(bad_measured, eager, eager, eager, atol=0, rtol=0)
    assert not verdict["passed"]
    assert not verdict["measured_vs_eager_bitwise_equal"]
    assert verdict["eager_repeat_bitwise_equal"]


def test_percentiles_include_all_samples_and_interpolate():
    assert percentile([4.0, 1.0, 3.0, 2.0], 0.5) == 2.5
    assert percentile([4.0, 1.0, 3.0, 2.0], 0.95) == pytest.approx(3.85)
    with pytest.raises(ValueError):
        percentile([], 0.5)
