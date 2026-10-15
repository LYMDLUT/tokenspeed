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

"""CPU checks of CLI arguments delivered to the draft graph engine."""

import sys
from test.hy_v4_parity import draft_graph_check
from types import ModuleType

import pytest


@pytest.mark.parametrize("seed", [0, 1337])
def test_cli_seed_reaches_engine(tmp_path, monkeypatch, seed):
    captured = {}

    class StubEngine:
        def __init__(self, **kwargs):
            captured["engine"] = kwargs

        def generate(self, *, input_ids, sampling_params):
            captured["input_ids"] = input_ids
            return [
                {"output_ids": [7], "meta_info": {"spec_verify_ct": 1}}
                for _ in input_ids
            ]

        def shutdown(self):
            captured["shutdown"] = True

    # Shadow every package so importing Engine cannot initialize the runtime.
    names = (
        "tokenspeed",
        "tokenspeed.runtime",
        "tokenspeed.runtime.entrypoints",
        "tokenspeed.runtime.entrypoints.engine",
    )
    modules = {name: ModuleType(name) for name in names}
    for name, module in modules.items():
        if name != names[-1]:
            module.__path__ = []
        if "." in name:
            parent, _, child = name.rpartition(".")
            setattr(modules[parent], child, module)
        monkeypatch.setitem(sys.modules, name, module)
    modules[names[-1]].Engine = StubEngine

    def forbidden_launch(*args, **kwargs):
        pytest.fail("CLI wiring must not export weights or launch a subprocess")

    monkeypatch.setattr(
        draft_graph_check, "build_truncated_checkpoint", forbidden_launch
    )
    monkeypatch.setattr(draft_graph_check, "_spawn", forbidden_launch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "draft_graph_check",
            "--run",
            "--model",
            "unused-model",
            "--tokenizer",
            "unused-tokenizer",
            "--out",
            str(tmp_path / "outputs.json"),
            "--prompts",
            "2",
            "--max-new-tokens",
            "1",
            "--seed",
            str(seed),
        ],
    )

    assert draft_graph_check.main() == 0
    assert captured["engine"].get("seed") == seed
    assert captured["input_ids"] == draft_graph_check._prompts(2, seed)
    assert captured["shutdown"] is True
