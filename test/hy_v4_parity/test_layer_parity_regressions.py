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


def test_old_dump_cannot_satisfy_a_run_that_produced_nothing(tmp_path, monkeypatch):
    import subprocess
    import sys
    from test.hy_v4_parity import hy_v4_parity as parity

    stale = tmp_path / "tokenspeed.dump.json"
    stale.write_text(json.dumps({"old_tap": _record()}))
    (tmp_path / "tokenspeed.summary.json").write_text(
        json.dumps({"output_ids": [3], "logprobs": [-0.1]})
    )
    monkeypatch.setattr(
        parity.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0),
    )
    with pytest.raises(SystemExit, match="no layer dump"):
        parity._run_engine(
            python=sys.executable,
            engine="tokenspeed",
            model_path=tmp_path,
            workdir=tmp_path,
            num_layers=1,
            extra=[],
        )
    assert json.loads(stale.read_text()) == {"old_tap": _record()}


def test_spawned_worker_persists_exact_prompt_before_hard_exit(tmp_path, monkeypatch):
    import os
    import subprocess
    import sys
    from test.hy_v4_parity import hy_v4_parity as parity

    fixture = tmp_path / "parity_spawn_fixture.py"
    fixture.write_text("""import json
import multiprocessing as mp
import os
from pathlib import Path
from types import SimpleNamespace
import torch
from test.hy_v4_parity.test_layer_parity_regressions import _FunctionalLayer
from test.hy_v4_parity.run_engine_dump import PROMPT_IDS

class FixtureModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="hy_v4", num_hidden_layers=1, enable_ihc=True)
        self.layers = torch.nn.ModuleList([_FunctionalLayer(True)])
    def forward(self, input_ids):
        return self.layers[0](input_ids.float()[:, None].expand(-1, 3))

def worker():
    model = FixtureModel()
    # Same length as the prompt, but different ids: this must not be collected.
    model.forward(torch.zeros(len(PROMPT_IDS), dtype=torch.int64))
    model.forward(torch.tensor(PROMPT_IDS))
    # A later decode must not overwrite the selected prefill.
    model.forward(torch.tensor([7]))
    os._exit(0)

def launch(dump_path, summary_path, run_id):
    process = mp.get_context("spawn").Process(target=worker)
    process.start()
    process.join(60)
    if process.is_alive():
        process.terminate()
        process.join()
        raise RuntimeError("test worker timed out")
    if process.exitcode != 0:
        raise RuntimeError(f"test worker failed: {process.exitcode}")
    from test.hy_v4_parity.hy_v4_dump import finish_collected_dump
    finish_collected_dump(Path(dump_path), run_id=run_id, prompt_ids=PROMPT_IDS, num_layers=1)
    Path(summary_path).write_text(json.dumps({"output_ids":[3,4], "logprobs":[-.1,-.2], "run_id":run_id}))
""")
    monkeypatch.setenv("HY_V4_DUMP_TARGETS", "parity_spawn_fixture:FixtureModel")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    real_run = subprocess.run
    environments = []

    def run_fixture(command, **kwargs):
        environments.append(kwargs["env"])
        dump = command[command.index("--dump-path") + 1]
        summary = command[command.index("--summary-path") + 1]
        run_id = command[command.index("--run-id") + 1]
        program = f"from parity_spawn_fixture import launch; launch({dump!r}, {summary!r}, {run_id!r})"
        return real_run([sys.executable, "-c", program], timeout=90, **kwargs)

    monkeypatch.setattr(parity.subprocess, "run", run_fixture)
    summary, dump_path = parity._run_engine(
        python=sys.executable,
        engine="tokenspeed",
        model_path=tmp_path,
        workdir=tmp_path,
        num_layers=1,
        extra=[],
    )
    records = json.loads(dump_path.read_text())
    metadata = json.loads(
        dump_path.with_suffix(dump_path.suffix + ".meta.json").read_text()
    )
    assert len(records) == 8
    assert metadata["pid"] != os.getpid()
    assert metadata["complete"] is True
    assert metadata["run_id"] == summary["run_id"]
    assert records["p0_layer0_hc_attn_in"]["sample"][0] == [15496.0] * 3
    assert str(tmp_path) in environments[0]["PYTHONPATH"].split(os.pathsep)


@pytest.mark.parametrize("corruption", ["run", "prompt", "missing", "duplicate"])
def test_worker_artifact_mismatch_is_rejected(tmp_path, corruption):
    from test.hy_v4_parity.hy_v4_dump import finish_collected_dump

    target = tmp_path / "dump.json"
    worker = tmp_path / "dump.worker-42.json"
    key = "p0_layer0_attn_in"
    records = {key: _record()}
    metadata = {
        "run_id": "current",
        "prompt_ids": [3],
        "expected_keys": [key],
        "complete": True,
    }
    if corruption == "run":
        metadata["run_id"] = "previous"
    elif corruption == "prompt":
        metadata["prompt_ids"] = [4]
    elif corruption == "missing":
        records = {}
    else:
        (tmp_path / "dump.worker-43.json").write_text("{}")
    worker.write_text(json.dumps(records))
    worker.with_suffix(worker.suffix + ".meta.json").write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError):
        finish_collected_dump(target, run_id="current", prompt_ids=[3], num_layers=1)
    assert not target.exists()


def test_hc_output_is_measured_after_a_separate_norm():
    from test.hy_v4_parity.hy_v4_parity_hook import LayerDumper

    class SeparateNormLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.hc_attn_layer = _Mlp()
            self.hc_mlp_layer = _Mlp()
            self.self_attn = _Attention()
            self.mlp = _Mlp()

        def forward(self, hidden_states):
            x = self.hc_attn_layer(hidden_states)
            x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5)
            x = self.self_attn(positions=torch.arange(x.shape[0]), hidden_states=x)
            x = self.hc_mlp_layer(x)
            x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5)
            return self.mlp(x)

    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([SeparateNormLayer()])
    x = torch.tensor([[1.0, 2.0, 4.0], [3.0, 1.0, 2.0]])
    expected = model.layers[0](x)
    dumper = LayerDumper()
    try:
        dumper.attach(model, num_layers=1)
        torch.testing.assert_close(model.layers[0](x), expected, rtol=0, atol=0)
        dumper.validate_complete()
        for stage in ("attn", "mlp"):
            assert (
                dumper.records[f"p0_layer0_hc_{stage}_out"]
                == dumper.records[f"p0_layer0_{stage}_in"]
            )
    finally:
        dumper.detach()


class _PreModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.last_output = None

    def forward(self, x):
        self.calls += 1
        self.last_output = x.sum(dim=1)
        return self.last_output, torch.ones_like(x[..., 0])


class _MethodBoundary(torch.nn.Module):
    """Match vLLM's prepare_input/pre/post wrapper, which has no forward."""

    def __init__(self, enabled):
        super().__init__()
        self.enable_ihc = enabled
        self.pre_calls = 0
        if enabled:
            self.hc_pre = _PreModule()

    def prepare_input(self, hidden_states):
        if self.enable_ihc and hidden_states.ndim == 2:
            return hidden_states[:, None, :].repeat(1, 2, 1)
        return hidden_states

    def pre(self, hidden_states):
        self.pre_calls += 1
        if not self.enable_ihc:
            return hidden_states, None, hidden_states
        reduced, gates = self.hc_pre(hidden_states)
        return reduced, gates, hidden_states

    def post(self, output, residual, gates):
        if not self.enable_ihc:
            return output + residual
        return residual + output[:, None, :] * gates[..., None]


@pytest.mark.parametrize("enabled", [False, True])
def test_method_ihc_wrapper_uses_pre_module_and_normalized_consumer(enabled):
    from test.hy_v4_parity.hy_v4_parity_hook import LayerDumper

    class MethodLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.hc_attn_layer = _MethodBoundary(enabled)
            self.hc_mlp_layer = _MethodBoundary(enabled)
            self.self_attn = _Attention()
            self.mlp = _Mlp()
            self.consumer_inputs = {}

        def forward(self, hidden_states):
            for stage, boundary, consumer in (
                ("attn", self.hc_attn_layer, self.self_attn),
                ("mlp", self.hc_mlp_layer, self.mlp),
            ):
                prepared = boundary.prepare_input(hidden_states)
                x, gates, residual = boundary.pre(prepared)
                x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5)
                self.consumer_inputs[stage] = x.detach().clone()
                if stage == "attn":
                    x = consumer(positions=torch.arange(x.shape[0]), hidden_states=x)
                else:
                    x = consumer(hidden_states=x)
                hidden_states = boundary.post(x, residual, gates)
            return hidden_states

    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([MethodLayer()])
    layer = model.layers[0]
    inputs = torch.tensor([[1.0, 2.0, 4.0], [3.0, 1.0, 2.0]])
    expected = layer(inputs)
    before_calls = [
        boundary.pre_calls for boundary in (layer.hc_attn_layer, layer.hc_mlp_layer)
    ]
    dumper = LayerDumper()
    try:
        dumper.attach(model, num_layers=1)
        torch.testing.assert_close(layer(inputs), expected, rtol=0, atol=0)
        dumper.validate_complete()
        assert len(dumper.records) == (8 if enabled else 4)
        for i, stage in enumerate(("attn", "mlp")):
            boundary = getattr(layer, f"hc_{stage}_layer")
            assert boundary.pre_calls == before_calls[i] + 1
            if enabled:
                assert boundary.hc_pre.calls == before_calls[i] + 1
                assert dumper.records[f"p0_layer0_hc_{stage}_in"]["shape"] == [2, 2, 3]
                assert (
                    dumper.records[f"p0_layer0_hc_{stage}_out"]
                    == dumper.records[f"p0_layer0_{stage}_in"]
                )
                torch.testing.assert_close(
                    torch.tensor(dumper.records[f"p0_layer0_hc_{stage}_out"]["sample"]),
                    layer.consumer_inputs[stage],
                    rtol=0,
                    atol=0,
                )
                assert not torch.equal(
                    boundary.hc_pre.last_output, layer.consumer_inputs[stage]
                )
    finally:
        dumper.detach()
    before = copy.deepcopy(dumper.records)
    layer(inputs * 7)
    assert dumper.records == before


def test_unknown_method_boundary_is_rejected():
    from test.hy_v4_parity.hy_v4_parity_hook import LayerDumper

    class UnknownMethodLayer(torch.nn.Module):
        __module__ = "torch.nn"

    model = torch.nn.Module()
    layer = UnknownMethodLayer()
    # No residual_ops or known hc_pre module means this layout is unsupported.
    layer.hc_attn_layer = torch.nn.Module()
    model.layers = torch.nn.ModuleList([layer])
    with pytest.raises(ValueError, match="unsupported functional boundary"):
        LayerDumper().attach(model, num_layers=1)


def test_reference_uses_token_ids_without_loading_a_tokenizer(tmp_path, monkeypatch):
    import sys
    import types
    from test.hy_v4_parity import hy_v4_dump, run_engine_dump

    class TokenOnlyLLM:
        def __init__(self, **kwargs):
            assert kwargs["skip_tokenizer_init"] is True
            assert not kwargs.get("tokenizer")

        def generate(self, prompts, sampling):
            assert prompts == [{"prompt_token_ids": run_engine_dump.PROMPT_IDS}]
            assert sampling.detokenize is False
            result = SimpleNamespace(
                token_ids=[3, 4],
                logprobs=[
                    {3: SimpleNamespace(logprob=-0.1)},
                    {4: SimpleNamespace(logprob=-0.2)},
                ],
            )
            return [SimpleNamespace(outputs=[result])]

    module = types.ModuleType("vllm")
    module.LLM = TokenOnlyLLM
    module.SamplingParams = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, "vllm", module)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(
        hy_v4_dump, "instrument_hy_v4_classes", lambda **kwargs: ["fixture"]
    )
    args = SimpleNamespace(
        reference_repo=str(tmp_path),
        model_path=str(tmp_path),
        num_layers=1,
        reference_attention_backend="FLASHMLA_SPARSE",
        max_model_len=128,
        gpu_memory_utilization=0.4,
    )
    result = run_engine_dump.run_reference(args, tmp_path / "dump.json")
    assert result == {"output_ids": [3, 4], "logprobs": [-0.1, -0.2]}


def test_numeric_tolerance_does_not_require_identical_storage_dtypes():
    reference = {"tap": _record()}
    candidate = copy.deepcopy(reference)
    candidate["tap"]["dtype"] = "bfloat16"
    assert (
        worst_status(first_divergence(reference, candidate, atol=0.01, rtol=0.01))[0]
        == "ok"
    )


def test_export_index_counts_tensor_bytes_for_mixed_dtypes(tmp_path):
    from test.hy_v4_parity.hy_v4_fixtures import Truncation, build_truncated_checkpoint

    from safetensors.torch import load_file, save_file

    source = tmp_path / "source"
    source.mkdir()
    tensors = {
        "model.embed_tokens.weight": torch.ones(2, 2, dtype=torch.bfloat16),
        "model.norm.weight": torch.ones(2, dtype=torch.float32),
        "lm_head.weight": torch.ones(2, 2).to(torch.float8_e4m3fn),
    }
    (source / "config.json").write_text(
        json.dumps({"num_hidden_layers": 1, "n_routed_experts": 8})
    )
    save_file(tensors, str(source / "weights.safetensors"))
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "weights.safetensors" for name in tensors}})
    )
    target = build_truncated_checkpoint(
        tmp_path / "output", Truncation(1, None), repo_id=str(source)
    )
    index = json.loads((target / "model.safetensors.index.json").read_text())
    assert index["metadata"]["total_size"] == 20
    loaded = load_file(str(target / "model.safetensors"))
    for name, value in tensors.items():
        assert loaded[name].dtype == value.dtype
        assert torch.equal(loaded[name].view(torch.uint8), value.view(torch.uint8))
