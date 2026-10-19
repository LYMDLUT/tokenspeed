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

"""CPU contracts for HY V4 residual boundaries and output gating.

Execute production AST nodes without importing the GPU runtime. Only attention's
base constructor, linear backend and unrelated MoE loader are substituted. This
checks model wiring and portable arithmetic, not CUDA or Marlin numerics.
"""

from __future__ import annotations

import ast
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

# CI discovers this CPU-capable test without importing the GPU runtime.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "python/tokenspeed/runtime/models/hy_v4.py"
RESIDUAL = ROOT / "tokenspeed-kernel/python/tokenspeed_kernel/ops/residual/torch.py"


def _extract(path, names, namespace, *, methods):
    tree = ast.parse(path.read_text(), filename=str(path))
    body = [
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
    ]
    found = set()
    for node in tree.body:
        name = node.name if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else None
        if isinstance(node, ast.Assign):
            name = next((t.id for t in node.targets if isinstance(t, ast.Name)), None)
        if name not in names:
            continue
        found.add(name)
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            node.decorator_list = []
        if isinstance(node, ast.ClassDef) and name in methods:
            node.body = [
                n
                for n in node.body
                if isinstance(n, ast.FunctionDef) and n.name in methods[name]
            ]
        body.append(node)
    assert found == set(names)
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


class _BaseAttention(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.v_head_dim = kwargs["v_head_dim"]
        self.num_local_heads = kwargs["num_heads"] // kwargs["mapping"].attn.tp_size


class _Linear(nn.Module):
    def __init__(self, inputs, outputs, **kwargs):
        super().__init__()
        self.options = kwargs
        self.weight = nn.Parameter(
            torch.zeros(outputs // kwargs["tp_size"], inputs), requires_grad=False
        )

    def forward(self, value):
        return nn.functional.linear(value, self.weight), None


@pytest.fixture
def production():
    ns = {"torch": torch, "nn": nn, "ClassVar": ClassVar}
    _extract(
        ROOT / "python/tokenspeed/runtime/layers/layernorm.py",
        {"RMSNorm"},
        ns,
        methods={"RMSNorm": {"__init__"}},
    )
    ns.update(
        {
            "GlmMoeDsaAttention": _BaseAttention,
            "DeepseekV3AttentionMLA": _BaseAttention,
            "ColumnParallelLinear": _Linear,
            "FusedRMSNorm": lambda q, kv: SimpleNamespace(q=q, kv=kv),
            "hyv4_staged_bf16": lambda: False,
            "_glm_dsa_skip_indexer_topk": lambda config, layer: True,
            "add_prefix": lambda name, prefix: f"{prefix}.{name}",
            "logger": logging.getLogger(__name__),
            "ExpertCheckpointSchema": SimpleNamespace,
            "build_moe_checkpoint_loader": lambda **kwargs: None,
            "ModelOptMxfp8Config": type("ModelOptMxfp8Config", (), {}),
        }
    )
    _extract(
        ROOT / "python/tokenspeed/runtime/model_loader/weight_utils.py",
        {"default_weight_loader"},
        ns,
        methods={},
    )
    _extract(
        ROOT / "python/tokenspeed/runtime/layers/quantization/modelopt_mxfp8.py",
        {"preprocess_modelopt_mxfp8_weights"},
        ns,
        methods={},
    )
    _extract(
        MODEL,
        {
            "_HC_CHECKPOINT_RENAMES",
            "_STACKED_DENSE_PROJECTIONS",
            "normalize_hyv4_checkpoint_name",
            "load_hyv4_weights",
            "HYV4HCBoundary",
            "hyv4_rmsnorm",
            "HYV4Attention",
            "HYV4DecoderLayer",
        },
        ns,
        methods={
            "HYV4Attention": {"__init__", "_apply_output_gate"},
            "HYV4DecoderLayer": {"_ihc_pre"},
        },
    )
    _extract(
        RESIDUAL,
        {
            "_ihc_check_common",
            "_ihc_gates",
            "torch_ihc_pre",
            "torch_ihc_post",
            "torch_ihc_head",
            "torch_ihc_fused_hc",
        },
        ns,
        methods={},
    )
    ns["_IHC_RESIDUAL_DTYPES"] = (torch.bfloat16, torch.float32)
    ns["residual_ops"] = SimpleNamespace(
        ihc_pre=ns["torch_ihc_pre"], ihc_fused_hc=ns["torch_ihc_fused_hc"]
    )
    return SimpleNamespace(**ns)


def _config(*, gating_type):
    return SimpleNamespace(
        hidden_size=32,
        hc_mult=4,
        rms_norm_eps=0.25,
        hc_eps=0.1,
        hc_magnitude=1.7,
        n_routed_experts=0,
        q_lora_rank=8,
        kv_lora_rank=8,
        num_attention_heads=4,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        rope_theta=10000,
        max_position_embeddings=128,
        index_topk=4,
        gated_mla=True,
        gating_type=gating_type,
        learnable_sink=False,
    )


def _mapping():
    return SimpleNamespace(
        attn=SimpleNamespace(tp_rank=1, tp_size=2, tp_group=(0, 1)),
        moe=SimpleNamespace(ep_rank=0, ep_size=1),
    )


@pytest.mark.parametrize("is_head", [False, True])
@pytest.mark.parametrize("source_dtype", [torch.float32, torch.bfloat16])
def test_boundary_construction_and_checkpoint_precision(
    production, is_head, source_dtype
):
    old = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        boundary = production.HYV4HCBoundary(
            _config(gating_type="elementwise"), is_head=is_head
        )
    finally:
        torch.set_default_dtype(old)
    owner = nn.Module()
    owner.model = nn.Module()
    owner.model.hc_head = boundary if is_head else nn.Module()
    if not is_head:
        owner.model.layers = nn.ModuleList([nn.Module()])
        owner.model.layers[0].hc_attn_layer = boundary
    owner.config = _config(gating_type="elementwise")
    owner.mapping = _mapping()
    owner.quant_config = None
    weights = []
    expected = {}
    for name, param in boundary.named_parameters():
        assert param.dtype == torch.float32
        value = (
            torch.arange(param.numel(), dtype=torch.float32).reshape(param.shape) / 128
            + 1
            + 2**-12
        ).to(source_dtype)
        checkpoint = (
            f"model.hc_head.hc_head_{name[3:]}"
            if is_head
            else f"model.layers.0.hc_attn_layer.hc_pre.{name}"
        )
        weights.append((checkpoint, value))
        expected[name] = value.float()
    production.load_hyv4_weights(
        owner, weights, module_prefix="model", checkpoint_prefix="model"
    )
    for name, param in boundary.named_parameters():
        torch.testing.assert_close(param, expected[name], rtol=0, atol=0)
    if source_dtype == torch.float32:
        assert not torch.equal(boundary.hc_fn, boundary.hc_fn.bfloat16().float())


@pytest.mark.parametrize("gating_type", ["headwise", "elementwise"])
def test_attention_constructor_and_gate_rounding(production, gating_type):
    config = _config(gating_type=gating_type)
    old = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        attn = production.HYV4Attention(
            config, _mapping(), 0, None, "model.attn", is_nextn=False
        )
    finally:
        torch.set_default_dtype(old)
    assert attn.q_a_layernorm.variance_epsilon == config.rms_norm_eps
    assert attn.kv_a_layernorm.variance_epsilon == config.rms_norm_eps
    assert attn.q_a_layernorm.weight.dtype == torch.bfloat16
    width = 1 if gating_type == "headwise" else config.v_head_dim
    assert attn.linear_gate.weight.shape == (2 * width, config.hidden_size)
    assert attn.linear_gate.options["tp_rank"] == 1
    assert attn.linear_gate.options["quant_config"] is None
    # One nonzero input isolates gate dtype/product stages from GEMM reduction.
    weight = torch.zeros_like(attn.linear_gate.weight)
    weight[:, 0] = torch.linspace(-1.5, 1.5, weight.shape[0]).bfloat16()
    owner = nn.Module()
    owner.model = attn
    owner.config = config
    owner.mapping = _mapping()
    owner.quant_config = None
    checkpoint = [
        (
            f"model.{name}",
            weight if name == "linear_gate.weight" else torch.ones_like(param),
        )
        for name, param in attn.named_parameters()
    ]
    production.load_hyv4_weights(
        owner, checkpoint, module_prefix="model", checkpoint_prefix="model"
    )
    hidden = torch.zeros(3, config.hidden_size, dtype=torch.bfloat16)
    hidden[:, 0] = 1
    output = torch.linspace(-3, 3, 24).reshape(3, 8).bfloat16()
    logits = weight[:, 0].expand(3, -1)
    gates = logits.sigmoid()
    expected = (
        (output.reshape(3, 2, 4) * gates.unsqueeze(-1)).reshape(3, 8)
        if width == 1
        else output * gates
    )
    actual = attn._apply_output_gate(output, hidden)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.dtype == torch.bfloat16
    unrounded = logits.float().sigmoid()
    changed = (
        (output.float().reshape(3, 2, 4) * unrounded.unsqueeze(-1))
        .reshape(3, 8)
        .bfloat16()
        if width == 1
        else (output.float() * unrounded).bfloat16()
    )
    assert not torch.equal(expected, changed)
    empty = output[:0]
    assert attn._apply_output_gate(empty, hidden[:0]) is empty
    attn.linear_gate = None
    assert attn._apply_output_gate(output, hidden) is output


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_ihc_formula_and_fused_model_wiring(production, dtype):
    torch.manual_seed(7)
    config = _config(gating_type="elementwise")
    boundary = production.HYV4HCBoundary(config, is_head=False)
    with torch.no_grad():
        boundary.hc_fn.copy_(torch.randn_like(boundary.hc_fn) * 0.15)
        boundary.hc_scale.copy_(torch.tensor([0.7, 1.3]))
        boundary.hc_base.copy_(torch.linspace(-1, 1, 8))
    residual = torch.randn(2, 3, 4, 32).to(dtype)
    block = torch.randn(2, 3, 32).bfloat16()
    previous_gates = torch.rand(2, 3, 4)
    pending = SimpleNamespace(block_output=block, gates=previous_gates)
    norm = SimpleNamespace(
        weight=torch.linspace(0.5, 1.5, 32).bfloat16(), variance_epsilon=0.125
    )
    layer = SimpleNamespace(config=config, staged_bf16=False)
    updated = (
        residual.float() + previous_gates.unsqueeze(-1) * block.float().unsqueeze(-2)
    ).to(dtype)
    flat = updated.flatten(-2).float()
    mix = (
        nn.functional.linear(flat, boundary.hc_fn)
        / (flat.square().mean(-1, keepdim=True) + config.rms_norm_eps).sqrt()
    )
    pre = (
        mix[..., :4] * boundary.hc_scale[0] + boundary.hc_base[:4]
    ).sigmoid() + config.hc_eps
    post = (
        config.hc_magnitude
        * (mix[..., 4:] * boundary.hc_scale[1] + boundary.hc_base[4:]).sigmoid()
        + config.hc_eps
    )
    reduced = (pre.unsqueeze(-1) * updated.float()).sum(-2).to(dtype).float()
    expected = (
        reduced
        * (reduced.square().mean(-1, keepdim=True) + norm.variance_epsilon).rsqrt()
        * norm.weight.float()
    ).bfloat16()
    actual_residual, actual, actual_post = production.HYV4DecoderLayer._ihc_pre(
        layer, residual, pending, boundary, norm
    )
    torch.testing.assert_close(actual_residual, updated, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_post, post, rtol=1e-6, atol=1e-7)
    assert actual_post.dtype == torch.float32
    first_residual, first, first_post = production.HYV4DecoderLayer._ihc_pre(
        layer, updated, None, boundary, norm
    )
    assert first_residual is updated
    torch.testing.assert_close(first, actual, rtol=0, atol=0)
    torch.testing.assert_close(first_post, actual_post, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_head_epsilon_and_output_dtype(production, dtype):
    config = _config(gating_type="elementwise")
    boundary = production.HYV4HCBoundary(config, is_head=True)
    with torch.no_grad():
        boundary.hc_fn.zero_()
        boundary.hc_scale.fill_(1)
        boundary.hc_base.zero_()
    residual = (
        torch.arange(256, dtype=torch.float32).reshape(2, 4, 32).div(128).to(dtype)
    )
    expected = (residual.float() * (0.5 + config.hc_eps)).sum(-2).bfloat16()
    actual = production.torch_ihc_head(
        residual,
        boundary.hc_fn,
        boundary.hc_scale,
        boundary.hc_base,
        config.rms_norm_eps,
        config.hc_eps,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
