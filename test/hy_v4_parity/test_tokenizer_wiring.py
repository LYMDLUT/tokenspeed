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

"""CPU checks of checkpoint/tokenizer wiring; no model or tokenizer is loaded."""

from __future__ import annotations

import json
import sys
import types
from test.hy_v4_parity import hy_v4_dump, hy_v4_parity, run_engine_dump
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("choice", ["source", "explicit", "legacy_env"])
def test_parent_uses_source_for_export_and_separate_tokenizer(
    tmp_path, monkeypatch, choice
):
    source = str(tmp_path / "original checkpoint")
    tokenizer = str(tmp_path / "original tokenizer")
    truncated = tmp_path / "truncated"
    captured = {}
    argv = [
        "hy_v4_parity",
        "--source",
        source,
        "--workdir",
        str(tmp_path / "work"),
        "--skip-reference",
    ]
    monkeypatch.delenv("HY_V4_TS_ENGINE_KWARGS", raising=False)
    if choice == "explicit":
        argv += ["--tokenizer", tokenizer]
    elif choice == "legacy_env":
        monkeypatch.setenv(
            "HY_V4_TS_ENGINE_KWARGS", json.dumps({"tokenizer": tokenizer})
        )

    def build(output, truncation, *, repo_id, **kwargs):
        captured["source"] = repo_id
        return truncated

    def run(**kwargs):
        captured["engine"] = kwargs
        return {"output_ids": [3], "logprobs": [-0.1]}, tmp_path / "dump.json"

    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(hy_v4_parity, "build_truncated_checkpoint", build)
    monkeypatch.setattr(hy_v4_parity, "_run_engine", run)
    assert hy_v4_parity.main() == 0
    assert captured["source"] == source
    assert captured["engine"]["model_path"] == truncated
    expected = source if choice == "source" else tokenizer
    assert captured["engine"]["extra"] == ["--tokenizer", expected]


@pytest.mark.parametrize("choice", ["explicit", "legacy_env", "both_agree"])
def test_tokenspeed_keeps_truncated_model_and_supplies_scheduler_tokenizer(
    tmp_path, monkeypatch, choice
):
    tokenizer = str(tmp_path / "original")
    model = str(tmp_path / "truncated")
    calls = {}
    monkeypatch.setenv(
        "HY_V4_TS_ENGINE_KWARGS",
        json.dumps(
            {"seed": 0, **({"tokenizer": tokenizer} if choice != "explicit" else {})}
        ),
    )

    class TokenEngine:
        def __init__(self, **kwargs):
            calls["kwargs"] = kwargs

        def generate(self, **kwargs):
            assert kwargs["input_ids"] == run_engine_dump.PROMPT_IDS
            return {
                "output_ids": [3],
                "meta_info": {"output_token_logprobs": [[-0.1, 3]]},
            }

        def shutdown(self):
            calls["shutdown"] = True

    for name in ("tokenspeed", "tokenspeed.runtime", "tokenspeed.runtime.entrypoints"):
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    module = types.ModuleType("tokenspeed.runtime.entrypoints.engine")
    module.Engine = TokenEngine
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(
        hy_v4_dump, "instrument_hy_v4_classes", lambda **kwargs: ["fixture"]
    )
    args = SimpleNamespace(
        model_path=model,
        tokenizer=None if choice == "legacy_env" else tokenizer,
        load_format="auto",
        attention_backend="dsa",
        quantization=None,
        max_model_len=2048,
        num_layers=3,
        gpu_memory_utilization=0.4,
    )
    result = run_engine_dump.run_tokenspeed(args, tmp_path / "dump.json")
    assert result == {"output_ids": [3], "logprobs": [-0.1]}
    assert calls["kwargs"]["model"] == model
    assert calls["kwargs"]["tokenizer"] == tokenizer
    assert calls["kwargs"]["skip_tokenizer_init"] is True
    assert calls["kwargs"]["seed"] == 0
    assert calls["shutdown"] is True


@pytest.mark.parametrize(
    "tokenizer,extra,message",
    [
        (None, {}, "--tokenizer"),
        ("original", {"tokenizer": "other"}, "disagree"),
        (None, {"tokenizer": ""}, "nonempty"),
        (None, {"tokenizer": 123}, "nonempty"),
    ],
)
def test_missing_invalid_or_conflicting_tokenizer_is_actionable(
    monkeypatch, tokenizer, extra, message
):
    monkeypatch.setenv("HY_V4_TS_ENGINE_KWARGS", json.dumps(extra))
    with pytest.raises(ValueError, match=message):
        run_engine_dump._tokenspeed_engine_kwargs(tokenizer, default_tokenizer=None)


def test_worker_cli_forwards_explicit_tokenizer(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "")
    captured = {}
    tokenizer = str(tmp_path / "original tokenizer")
    model = str(tmp_path / "truncated")
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_engine_dump",
            "--engine",
            "tokenspeed",
            "--model-path",
            model,
            "--tokenizer",
            tokenizer,
            "--dump-path",
            str(tmp_path / "dump.json"),
            "--summary-path",
            str(tmp_path / "summary.json"),
        ],
    )
    monkeypatch.setattr(
        run_engine_dump,
        "worker_environment",
        lambda **kwargs: {"PYTHONPATH": str(tmp_path)},
    )
    monkeypatch.setattr(
        hy_v4_dump, "finish_collected_dump", lambda *args, **kwargs: None
    )

    def run(args, dump_path):
        captured.update(model=args.model_path, tokenizer=args.tokenizer)
        return {"output_ids": [3], "logprobs": [-0.1]}

    monkeypatch.setattr(run_engine_dump, "run_tokenspeed", run)
    assert run_engine_dump.main() == 0
    assert captured == {"model": model, "tokenizer": tokenizer}


def test_reference_cli_rejects_a_tokenizer_instead_of_ignoring_it(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_engine_dump",
            "--engine",
            "reference",
            "--model-path",
            str(tmp_path),
            "--tokenizer",
            "original",
            "--dump-path",
            str(tmp_path / "dump.json"),
            "--summary-path",
            str(tmp_path / "summary.json"),
        ],
    )
    with pytest.raises(SystemExit) as error:
        run_engine_dump.main()
    assert error.value.code == 2
    assert "only supported for --engine tokenspeed" in capsys.readouterr().err


def test_parent_rejects_tokenizer_inside_extra_args(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("HY_V4_TS_ENGINE_KWARGS", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hy_v4_parity",
            "--source",
            "original",
            "--tokenizer",
            "wanted",
            "--tokenspeed-args=--tokenizer=other",
            "--workdir",
            str(tmp_path / "work"),
            "--skip-reference",
        ],
    )
    calls = []

    def build(*args, **kwargs):
        calls.append("build")
        return tmp_path / "truncated"

    def run(**kwargs):
        calls.append("run")
        return {}, tmp_path / "dump.json"

    monkeypatch.setattr(hy_v4_parity, "build_truncated_checkpoint", build)
    monkeypatch.setattr(hy_v4_parity, "_run_engine", run)
    with pytest.raises(SystemExit) as error:
        hy_v4_parity.main()
    assert error.value.code == 2
    assert "--tokenspeed-args" in capsys.readouterr().err
    assert calls == []
