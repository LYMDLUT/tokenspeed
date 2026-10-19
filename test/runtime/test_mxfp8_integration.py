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

"""MXFP8 preparation routing and construction-time indexer fusion contracts."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest import mock

# CI Registration (parsed via AST, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=30, suite="runtime-1gpu")

import pytest
import torch

from tokenspeed.runtime.configs.hy_v4_config import HYV4Config
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.layers.dense import UnquantizedLinearMethod
from tokenspeed.runtime.layers.linear import ColumnParallelLinear
from tokenspeed.runtime.layers.quantization.modelopt_mxfp8 import ModelOptMxfp8Config
from tokenspeed.runtime.models import glm5, hy_v4


def _config():
    return HYV4Config(
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        q_lora_rank=32,
        kv_lora_rank=32,
        qk_nope_head_dim=32,
        qk_rope_head_dim=8,
        v_head_dim=32,
        index_n_heads=2,
        index_head_dim=32,
        index_topk=2,
        max_position_embeddings=128,
        mlp_layer_types=["dense"],
        indexer_types=["full"],
    )


def _quant_config():
    return ModelOptMxfp8Config(
        exclude_modules=["*.linear_gate", "*.weights_proj"],
        kv_cache_quant_algo=None,
    )


@pytest.fixture
def model_defaults():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with mock.patch.dict(
            hy_v4.global_server_args_dict,
            {
                "tp_batch_invariant": "none",
                "numerics": "auto",
                "dense_gemm_backend": "auto",
                "attention_backend": "dsa",
                "hyv4_numerics": "auto",
            },
        ):
            yield
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize("fuse", [False, True])
@pytest.mark.parametrize("indexer_type", [glm5.GlmDsaIndexer, hy_v4.HYV4DsaIndexer])
def test_indexer_allocates_only_the_requested_projection(
    model_defaults, fuse, indexer_type
):
    with mock.patch.object(
        glm5, "MergedColumnParallelLinear", wraps=glm5.MergedColumnParallelLinear
    ) as merged:
        indexer = indexer_type(
            config=_config(),
            hidden_size=64,
            q_lora_rank=32,
            qk_rope_head_dim=8,
            rope_theta=10000.0,
            rope_scaling=None,
            max_position_embeddings=128,
            quant_config=None,
            prefix="model.layers.0.self_attn.indexer",
            fuse_wk_weights_proj=fuse,
        )
    assert merged.call_count == int(fuse)
    assert (indexer.wk_weights_proj is not None) == fuse
    assert (indexer.wk is None) == fuse
    assert (indexer.weights_proj is None) == fuse
    assert not indexer._wk_weights_proj_loaded
    if fuse:
        indexer.set_wk_weights_proj_loaded(True)
        assert indexer._wk_weights_proj_loaded
    else:
        with pytest.raises(ValueError, match="absent"):
            indexer.set_wk_weights_proj_loaded(True)
        assert "wk_weights_proj.weight" not in dict(indexer.named_parameters())
    indexer.set_wk_weights_proj_loaded(False)
    assert not indexer._wk_weights_proj_loaded


@pytest.mark.parametrize("quantized", [False, True])
def test_hyv4_selects_fusion_during_indexer_construction(model_defaults, quantized):
    quant = _quant_config() if quantized else None
    with torch.device("cpu"), mock.patch.object(
        glm5, "MergedColumnParallelLinear", wraps=glm5.MergedColumnParallelLinear
    ) as merged:
        attention = hy_v4.HYV4Attention(
            _config(),
            Mapping(rank=0, world_size=1),
            layer_id=0,
            quant_config=quant,
            prefix="model.layers.0.self_attn",
            is_nextn=False,
        )
    assert merged.call_count == 1
    assert attention.indexer.wk_weights_proj is not None
    assert attention.indexer.wk is None
    assert attention.indexer.weights_proj is None
    assert isinstance(attention.linear_gate.quant_method, UnquantizedLinearMethod)


def test_quantization_config_forwards_explicit_override():
    layer = SimpleNamespace(
        override_kernel_name="marlin_mxfp8_linear", interleave_linear_and_gate=False
    )
    method = _quant_config().get_quant_method(layer, "model.layers.0.self_attn.o_proj")
    assert method.override == "marlin_mxfp8_linear"
    layer.interleave_linear_and_gate = True
    with pytest.raises(ValueError, match="interleaving"):
        _quant_config().get_quant_method(layer, "model.layers.0.self_attn.o_proj")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_runtime_prepares_through_registered_public_entry():
    from tokenspeed_kernel.registry import KernelRegistry

    import tokenspeed.runtime.layers.dense.mxfp8 as runtime

    with torch.device("cuda"):
        layer = ColumnParallelLinear(
            128,
            64,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=_quant_config(),
            prefix="model.layers.0.self_attn.o_proj",
            tp_rank=0,
            tp_size=1,
            tp_group=(0,),
            gather_output=False,
            use_presharded_weights=True,
            override_kernel_name="marlin_mxfp8_linear",
        )
    raw = torch.ones(64, 128, device="cuda").to(torch.float8_e4m3fn)
    scale = torch.full((64, 4), 127, device="cuda", dtype=torch.uint8)
    layer.weight.weight_loader(layer.weight, raw)
    layer.weight_scale.weight_loader(layer.weight_scale, scale)
    with mock.patch.object(
        runtime, "prepare_mxfp8_linear", wraps=runtime.prepare_mxfp8_linear
    ) as prepare:
        layer.quant_method.process_weights_after_loading(layer)
        layer.quant_method.process_weights_after_loading(layer)
    prepare.assert_called_once()
    assert prepare.call_args.kwargs["override"] == "marlin_mxfp8_linear"
    assert (
        KernelRegistry.get().get_by_name(layer.quant_method.kernel.kernel_name)
        is not None
    )
    x = torch.ones(2, 128, dtype=torch.bfloat16, device="cuda")
    out = layer(x)[0]
    torch.testing.assert_close(out, torch.full_like(out, 128), rtol=0, atol=0)
    assert layer.weight.dtype == torch.int32
    assert layer.weight_scale.dtype == torch.uint8
    with pytest.raises(RuntimeError, match="fresh model"):
        layer.weight.weight_loader(layer.weight, raw)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("order", ["scale_first", "weight_first"])
def test_hyv4_mxfp8_indexer_fused_wk_matches_reference_load(model_defaults, order):
    """The official MXFP8 wk and BF16 weight-head share one BF16 GEMM."""
    from torch import nn

    from tokenspeed.runtime.layers.quantization.modelopt_mxfp8 import (
        dequantize_mxfp8_weight,
    )

    config = _config()
    quant = _quant_config()
    mapping = Mapping(rank=0, world_size=1)
    with torch.device("cuda"):
        attn = hy_v4.HYV4Attention(
            config,
            mapping,
            layer_id=0,
            quant_config=quant,
            prefix="model.layers.0.self_attn",
            is_nextn=False,
        )
    assert attn.indexer.wk_weights_proj is not None
    assert attn.indexer.wk is None

    class Owner(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([nn.Module()])
            self.model.layers[0].self_attn = attn
            self.config = config
            self.mapping = mapping
            self.quant_config = quant

    owner = Owner()
    prefix = "model.layers.0.self_attn.indexer"
    wk = (torch.arange(32 * 64, device="cuda").reshape(32, 64) % 35 - 17).to(
        torch.float8_e4m3fn
    )
    scale = (torch.arange(32 * 2, device="cuda").reshape(32, 2) % 7 + 122).to(
        torch.uint8
    )
    weight_head = (
        torch.arange(2 * 64, device="cuda").reshape(2, 64).to(torch.bfloat16) / 512
    )
    fp8_pair = [
        (prefix + ".wk.weight", wk),
        (prefix + ".wk.weight_scale", scale),
    ]
    if order == "scale_first":
        fp8_pair.reverse()
    stream = [
        (prefix + ".weights_proj.weight", weight_head),
        *fp8_pair,
    ]
    fused_name = prefix + ".wk_weights_proj.weight"
    remaining = tuple(
        name for name, _ in owner.named_parameters() if name != fused_name
    )
    hy_v4.load_hyv4_weights(
        owner,
        stream,
        module_prefix="model",
        checkpoint_prefix="model",
        shared_params=remaining,
    )
    expected_weight = torch.cat(
        [dequantize_mxfp8_weight(wk, scale), weight_head], dim=0
    )
    actual_weight = attn.indexer.wk_weights_proj.weight
    torch.testing.assert_close(actual_weight, expected_weight, rtol=0, atol=0)
    assert attn.indexer._wk_weights_proj_loaded
    x = (torch.arange(4 * 64, device="cuda").reshape(4, 64) % 23).to(
        torch.bfloat16
    ) / 23
    observed = attn.indexer.wk_weights_proj(x)[0]
    expected = torch.nn.functional.linear(x, expected_weight)
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)
