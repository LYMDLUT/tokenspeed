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

"""CPU checks of the GPU harness's refusal and numerical verdict contracts."""

from test.hy_v4_parity import gpu_kernel_check
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def cpu_harness(monkeypatch):
    """Keep the harness's allocation calls on CPU; no kernel is executed."""
    monkeypatch.setattr(gpu_kernel_check, "_FAILURES", [])

    def cpu_allocation(function):
        def allocate(*args, **kwargs):
            assert kwargs["device"] == "cuda"
            return function(*args, **{**kwargs, "device": "cpu"})

        return allocate

    for name in ("randn", "rand"):
        monkeypatch.setattr(torch, name, cpu_allocation(getattr(torch, name)))
    return gpu_kernel_check


def test_oversized_refusal_targets_the_forced_triton_kernel(cpu_harness):
    def ihc_pre(*args, **kwargs):
        if kwargs.get("override") == "triton_ihc_pre":
            raise ValueError("iHC supports up to 8 channels, got 16")
        # A successful public fallback must not satisfy the refusal check.
        return None

    cpu_harness.check_ihc_rejects_oversized(
        SimpleNamespace(ihc_pre=ihc_pre), dtype=torch.bfloat16
    )
    assert not cpu_harness._FAILURES


@pytest.mark.parametrize(
    ("error_type", "message"),
    [
        (RuntimeError, "unrelated allocation failure on device 8"),
        (RuntimeError, "iHC supports up to 8 channels, got 16"),
        (ValueError, "iHC supports up to 8 channels, got 32"),
        (ValueError, "unrelated error 8"),
        (None, None),
    ],
    ids=[
        "unrelated-eight",
        "wrong-type",
        "wrong-channels",
        "wrong-message",
        "no-error",
    ],
)
def test_oversized_refusal_rejects_other_results(cpu_harness, error_type, message):
    def ihc_pre(*args, **kwargs):
        assert kwargs["override"] == "triton_ihc_pre"
        if error_type is not None:
            raise error_type(message)

    cpu_harness.check_ihc_rejects_oversized(
        SimpleNamespace(ihc_pre=ihc_pre), dtype=torch.bfloat16
    )
    assert len(cpu_harness._FAILURES) == 1


@pytest.mark.parametrize("corruption", [None, "output", "gates", "nonfinite"])
def test_public_fallback_checks_both_results(cpu_harness, corruption):
    def ihc_pre(x, fn, scale, base, magnitude, rms_eps, hc_eps, **kwargs):
        assert "override" not in kwargs
        assert "solution" not in kwargs
        assert x.shape[1] == 16
        out, gates = cpu_harness._reference_ihc_pre(
            x, fn, scale, base, 16, magnitude, hc_eps, rms_eps
        )
        if corruption == "output":
            out = out + 1
        elif corruption == "gates":
            gates = gates + 0.1
        elif corruption == "nonfinite":
            out.fill_(float("nan"))
            gates.fill_(float("inf"))
        return out, gates

    cpu_harness.check_ihc_oversized_fallback(
        SimpleNamespace(ihc_pre=ihc_pre), dtype=torch.bfloat16
    )
    expected_failures = (
        0 if corruption is None else 2 if corruption == "nonfinite" else 1
    )
    assert len(cpu_harness._FAILURES) == expected_failures
