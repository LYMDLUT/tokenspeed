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

"""CPU regression tests for the GLM fused-only indexer checkpoint loader.

Execute the production loader class with real tensors and a small module tree.
The base-model/GPU import boundaries are replaced; the GLM loading, parameter
lookup and pair/shard bookkeeping are unchanged. This is not a GPU kernel test.
"""

from __future__ import annotations

import ast
import logging
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

# CI Registration (parsed via AST, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

import pytest
import torch
from torch import nn

MODELS = Path(__file__).resolve().parents[2] / "python/tokenspeed/runtime/models"
PREFIX = "model.layers.0.self_attn.indexer"
FUSED = f"{PREFIX}.wk_weights_proj.weight"


def _copy(param, value):
    with torch.no_grad():
        param.copy_(value)


class _Projection(nn.Module):
    def __init__(self, rows, *, fused):
        super().__init__()
        self.weight = nn.Parameter(
            torch.full((rows, 32), float("nan"), dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.fused = fused
        self.weight.weight_loader = self.load

    def load(self, param, value, shard_id=None):
        if shard_id is None:
            _copy(param, value)
        else:
            assert self.fused and shard_id in (0, 1)
            _copy(param[:2] if shard_id == 0 else param[2:], value)


class _Indexer(nn.Module):
    def __init__(self, *, direct):
        super().__init__()
        self.wk_weights_proj = _Projection(3, fused=True)
        self.wk = _Projection(2, fused=False) if direct else None
        self.weights_proj = _Projection(1, fused=False) if direct else None
        self._wk_weights_proj_loaded = False

    def set_wk_weights_proj_loaded(self, loaded):
        self._wk_weights_proj_loaded = loaded


class _Base(nn.Module):
    def __init__(self, config, mapping, *, model, quant_config, prefix):
        super().__init__()
        self.config = config
        self.mapping = mapping
        self.model = model
        self.quant_config = quant_config
        self.prefix = prefix
        self.active_update = False

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        loaded = set()
        for name, value in weights:
            _copy(params[name], value)
            loaded.add(name)
        return loaded

    def begin_weight_update(self):
        self.active_update = True

    def abort_weight_update(self):
        self.active_update = False

    def end_weight_update(self):
        self.active_update = False


def _block_dequant(weight, scale, block_size):
    # The test substitutes only the device-specific dequantization boundary.
    rows, cols = block_size
    expanded = scale.repeat_interleave(rows, 0).repeat_interleave(cols, 1)
    return weight.float() * expanded


def _loader_class():
    source = MODELS / "glm5.py"
    tree = ast.parse(source.read_text(), filename=str(source))
    node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "GlmMoeDsaForCausalLM"
    )
    namespace = {
        "nn": nn,
        "torch": torch,
        "DeepseekV3ForCausalLM": _Base,
        "GlmMoeDsaModel": nn.Module,
        "GlmDsaIndexer": _Indexer,
        "block_dequant": _block_dequant,
        "default_weight_loader": _copy,
        "pad_fused_qkv_a_proj_weight_for_fp8_blockscale": lambda attn: None,
        "get_layer_id": lambda name: (
            int(re.search(r"\.layers\.(\d+)\.", name).group(1))
            if re.search(r"\.layers\.(\d+)\.", name)
            else None
        ),
    }
    # Use the actual parent parameter-name resolver, including missing names.
    parent_source = MODELS / "deepseek_v3.py"
    parent_tree = ast.parse(parent_source.read_text(), filename=str(parent_source))
    parent = next(
        node
        for node in parent_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DeepseekV3ForCausalLM"
    )
    resolver = next(
        node
        for node in parent.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_param"
    )
    resolver_ns = {
        "logger": logging.getLogger(__name__),
        "_OPTIONAL_MISSING_WEIGHT_SUFFIXES": (),
    }
    exec(
        compile(
            ast.Module(body=[resolver], type_ignores=[]), str(parent_source), "exec"
        ),
        resolver_ns,
    )
    _Base.get_param = resolver_ns["get_param"]
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace["GlmMoeDsaForCausalLM"]


def _model(*, direct):
    indexer = _Indexer(direct=direct)
    attn = nn.Module()
    attn.indexer = indexer
    layer = nn.Module()
    layer.self_attn = attn
    inner = nn.Module()
    inner.layers = nn.ModuleList([layer])
    inner.other = nn.Parameter(torch.zeros(1), requires_grad=False)
    owner = _loader_class()(
        SimpleNamespace(num_hidden_layers=1),
        SimpleNamespace(),
        model=inner,
        quant_config=SimpleNamespace(weight_block_size=[1, 32]),
    )
    return owner, indexer


def _weights():
    wk = (torch.arange(64).reshape(2, 32) % 7).to(torch.bfloat16)
    head = torch.ones(1, 32, dtype=torch.bfloat16)
    return wk, head


@pytest.mark.parametrize("head_first", [False, True])
def test_fused_only_loads_separate_bf16_checkpoint_shards(head_first):
    owner, indexer = _model(direct=False)
    wk, head = _weights()
    stream = [(f"{PREFIX}.wk.weight", wk), (f"{PREFIX}.weights_proj.weight", head)]
    if head_first:
        stream.reverse()
    loaded = owner.load_weights(stream)
    assert loaded == {FUSED}
    assert indexer._wk_weights_proj_loaded
    torch.testing.assert_close(
        indexer.wk_weights_proj.weight, torch.cat([wk, head]), rtol=0, atol=0
    )


@pytest.mark.parametrize("scale_first", [False, True])
def test_fp8_pair_and_fused_head_can_arrive_in_separate_calls(scale_first):
    owner, indexer = _model(direct=False)
    wk, head = _weights()
    raw = wk.to(torch.float8_e4m3fn)
    scale = torch.tensor([[0.5], [2.0]], dtype=torch.float32)
    pair = [(f"{PREFIX}.wk.weight", raw), (f"{PREFIX}.wk.weight_scale_inv", scale)]
    if scale_first:
        pair.reverse()
    owner.begin_weight_update()
    assert owner.load_weights([pair[0]]) == set()
    assert owner._pending_fp8_wk and not indexer._wk_weights_proj_loaded
    assert owner.load_weights([(f"{PREFIX}.weights_proj.weight", head)]) == set()
    assert owner.load_weights([pair[1]]) == {FUSED}
    assert not owner._pending_fp8_wk and indexer._wk_weights_proj_loaded
    expected = torch.cat([(raw.float() * scale).to(torch.bfloat16), head])
    torch.testing.assert_close(indexer.wk_weights_proj.weight, expected, rtol=0, atol=0)
    owner.end_weight_update()


def test_incomplete_fused_shard_is_not_reported_ready():
    owner, indexer = _model(direct=False)
    wk, _ = _weights()
    assert owner.load_weights([(f"{PREFIX}.wk.weight", wk)]) == set()
    assert not indexer._wk_weights_proj_loaded
    torch.testing.assert_close(indexer.wk_weights_proj.weight[:2], wk, rtol=0, atol=0)
    assert torch.isnan(indexer.wk_weights_proj.weight[2:]).all()


def test_direct_and_fused_projections_both_receive_legacy_checkpoint():
    owner, indexer = _model(direct=True)
    wk, head = _weights()
    loaded = owner.load_weights(
        [(f"{PREFIX}.wk.weight", wk), (f"{PREFIX}.weights_proj.weight", head)]
    )
    assert loaded == {FUSED, f"{PREFIX}.wk.weight", f"{PREFIX}.weights_proj.weight"}
    torch.testing.assert_close(indexer.wk.weight, wk, rtol=0, atol=0)
    torch.testing.assert_close(indexer.weights_proj.weight, head, rtol=0, atol=0)
    torch.testing.assert_close(
        indexer.wk_weights_proj.weight, torch.cat([wk, head]), rtol=0, atol=0
    )


def test_absent_shared_indexer_does_not_buffer_checkpoint_tensors():
    owner, indexer = _model(direct=False)
    del owner.model.layers[0].self_attn.indexer
    wk, _ = _weights()
    assert (
        owner.load_weights([(f"{PREFIX}.wk.weight", wk.to(torch.float8_e4m3fn))])
        == set()
    )
    assert not owner._pending_fp8_wk


def test_non_indexer_parameters_still_reach_parent_loader():
    owner, _ = _model(direct=False)
    assert owner.load_weights([("model.other", torch.ones(1))]) == {"model.other"}
    torch.testing.assert_close(owner.model.other, torch.ones(1), rtol=0, atol=0)


def test_premerged_checkpoint_marks_fused_projection_ready():
    owner, indexer = _model(direct=False)
    wk, head = _weights()
    assert owner.load_weights([(FUSED, torch.cat([wk, head]))]) == {FUSED}
    assert indexer._wk_weights_proj_loaded


def test_unpaired_fp8_update_is_rejected_and_pairing_state_cleared():
    owner, _ = _model(direct=False)
    wk, _ = _weights()
    owner.begin_weight_update()
    owner.load_weights([(f"{PREFIX}.wk.weight", wk.to(torch.float8_e4m3fn))])
    with pytest.raises(RuntimeError, match="without its partner"):
        owner.end_weight_update()
    assert not owner._pending_fp8_wk and not owner.active_update
