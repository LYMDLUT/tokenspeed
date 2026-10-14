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

"""CPU regressions for layer-parity verdicts and real collection boundaries."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from test.hy_v4_parity.hy_v4_dump import compare_dumps
from test.hy_v4_parity.hy_v4_parity import _compare_outputs
from test.hy_v4_parity.hy_v4_parity_hook import first_divergence, worst_status

import pytest
import torch


def _record():
    return {
        "shape": [2, 2],
        "dtype": "float32",
        "sample": [[1.0, 2.0], [3.0, 4.0]],
        "sample_tokens": [0, 1],
        "sample_channels": [0, 1],
        "norm": 5.0,
        "abs_max": 4.0,
        "abs_mean": 2.5,
    }


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("field", ["sample", "norm", "abs_max", "abs_mean"])
def test_nonfinite_layer_evidence_fails(field, value):
    reference = {"tap": _record()}
    candidate = copy.deepcopy(reference)
    if field == "sample":
        candidate["tap"][field][0][0] = value
    else:
        candidate["tap"][field] = value
    assert (
        worst_status(first_divergence(reference, candidate, atol=0.01, rtol=0.01))[0]
        != "ok"
    )


def test_empty_layer_dumps_are_not_a_pass(tmp_path):
    left, right = tmp_path / "left.json", tmp_path / "right.json"
    left.write_text("{}")
    right.write_text("{}")
    assert not compare_dumps(left, right)[0]
    assert worst_status([])[0] != "ok"


def test_zero_tolerance_supports_exact_comparison():
    reference = {"tap": _record()}
    assert (
        worst_status(first_divergence(reference, reference, atol=0, rtol=0))[0] == "ok"
    )
    candidate = copy.deepcopy(reference)
    candidate["tap"]["norm"] += 0.01
    assert (
        worst_status(first_divergence(reference, candidate, atol=0, rtol=0))[0] != "ok"
    )


@pytest.mark.parametrize("change", ["missing", "empty", "nan", "short", "empty_ids"])
def test_output_evidence_is_required_per_generated_token(change):
    reference = {"output_ids": [3, 4], "logprobs": [-0.1, -0.2]}
    candidate = copy.deepcopy(reference)
    if change == "missing":
        del reference["logprobs"]
        del candidate["logprobs"]
    elif change == "empty":
        reference["logprobs"] = candidate["logprobs"] = []
    elif change == "nan":
        candidate["logprobs"][0] = float("nan")
    elif change == "short":
        reference["logprobs"] = candidate["logprobs"] = [-0.1]
    else:
        reference = candidate = {"output_ids": [], "logprobs": []}
    assert not _compare_outputs(reference, candidate, atol=0.01)[0]


def test_identical_finite_output_evidence_passes():
    summary = {"output_ids": [3, 4], "logprobs": [-0.1, -0.2]}
    assert _compare_outputs(summary, summary, atol=0)[0]


@pytest.mark.parametrize("tolerance", [-1.0, float("nan"), float("inf")])
def test_invalid_comparison_tolerance_is_rejected(tolerance):
    with pytest.raises(ValueError):
        first_divergence(
            {"tap": _record()}, {"tap": _record()}, atol=0.01, rtol=tolerance
        )
    with pytest.raises(ValueError):
        _compare_outputs(
            {"output_ids": [1], "logprobs": [0.0]},
            {"output_ids": [1], "logprobs": [0.0]},
            atol=tolerance,
        )


# Match the functional residual API: the boundary itself is a parameter holder.
from types import SimpleNamespace


def _pre(residual, fn):
    return residual.sum(dim=1), torch.ones_like(residual[..., 0]) * fn[0]


def _fused(x_prev, residual_prev, post_prev, fn):
    updated = residual_prev + x_prev[:, None, :] * post_prev[..., None]
    return updated, updated.sum(dim=1), torch.ones_like(updated[..., 0]) * fn[0]


residual_ops = SimpleNamespace(ihc_pre=_pre, ihc_fused_hc=_fused)


class _ParameterBoundary(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hc_fn = torch.nn.Parameter(torch.ones(1))


class _Attention(torch.nn.Module):
    def forward(self, positions, hidden_states):
        return hidden_states + 1


class _Mlp(torch.nn.Module):
    def forward(self, hidden_states):
        return hidden_states * 2


class _FunctionalLayer(torch.nn.Module):
    def __init__(self, fused):
        super().__init__()
        self.hc_attn_layer = _ParameterBoundary()
        self.hc_mlp_layer = _ParameterBoundary()
        self.self_attn = _Attention()
        self.mlp = _Mlp()
        self.fused = fused

    def forward(self, hidden_states):
        residual = hidden_states[:, None, :].expand(-1, 2, -1).clone()
        x, gates = residual_ops.ihc_pre(residual, self.hc_attn_layer.hc_fn)
        x = self.self_attn(positions=torch.arange(x.shape[0]), hidden_states=x)
        if self.fused:
            residual, x, gates = residual_ops.ihc_fused_hc(
                x, residual, gates, self.hc_mlp_layer.hc_fn
            )
        else:
            residual = residual + x[:, None, :] * gates[..., None]
            x, gates = residual_ops.ihc_pre(residual, self.hc_mlp_layer.hc_fn)
        return self.mlp(x)


@pytest.mark.parametrize("fused", [False, True])
def test_parameter_boundaries_and_keyword_inputs_are_collected(fused, monkeypatch):
    from test.hy_v4_parity.hy_v4_parity_hook import LayerDumper

    monkeypatch.setenv("HY_V4_DUMP_ONLY_TOKENS", "4")
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([_FunctionalLayer(fused)])
    inputs = torch.ones(4, 3)
    expected = model.layers[0](inputs)
    original_pre, original_fused = residual_ops.ihc_pre, residual_ops.ihc_fused_hc
    dumper = LayerDumper()
    try:
        dumper.attach(model, num_layers=1)
        actual = model.layers[0](inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert len(dumper.records) == 8
        assert dumper.records["p0_layer0_hc_attn_in"]["shape"] == [4, 2, 3]
        assert dumper.records["p0_layer0_attn_in"]["sample"][0] == [2.0] * 3
        assert dumper.records["p0_layer0_hc_mlp_in"]["sample"][0] == [4.0] * 3
        dumper.validate_complete()
    finally:
        dumper.detach()
    assert (
        residual_ops.ihc_pre is original_pre
        and residual_ops.ihc_fused_hc is original_fused
    )
    before = copy.deepcopy(dumper.records)
    model.layers[0](inputs * 7)
    assert dumper.records == before


def test_expected_but_unexecuted_boundary_fails():
    from test.hy_v4_parity.hy_v4_parity_hook import LayerDumper

    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([_FunctionalLayer(False)])
    dumper = LayerDumper()
    try:
        dumper.attach(model, num_layers=1)
        # Simulate an unexpected execution path which bypasses all iHC calls.
        model.layers[0].self_attn(
            positions=torch.zeros(2), hidden_states=torch.ones(2, 3)
        )
        with pytest.raises(RuntimeError, match="missing taps"):
            dumper.validate_complete()
    finally:
        dumper.detach()


def test_enabled_ihc_cannot_silently_omit_a_boundary():
    from test.hy_v4_parity.hy_v4_parity_hook import LayerDumper

    model = torch.nn.Module()
    model.config = SimpleNamespace(enable_ihc=True)
    layer = _FunctionalLayer(False)
    layer.hc_mlp_layer = None
    model.layers = torch.nn.ModuleList([layer])
    with pytest.raises(ValueError, match="hc_mlp_layer"):
        LayerDumper().attach(model, num_layers=1)
