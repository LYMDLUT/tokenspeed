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
