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

"""Explicit ModelOpt MXFP8 W8A8 requests fail closed at launch."""

import json

import pytest
import torch
from tokenspeed_kernel.platform import pdl_enabled

from tokenspeed.runtime.utils.env import (
    global_server_args_dict,
    global_server_args_dict_update,
)
from tokenspeed.runtime.utils.server_args import ServerArgs


def _config(activation_dtype: str):
    from tokenspeed.runtime.layers.quantization.modelopt_mxfp8 import (
        ModelOptMxfp8Config,
    )

    return ModelOptMxfp8Config(
        exclude_modules=["lm_head"],
        kv_cache_quant_algo=None,
        activation_dtype=activation_dtype,
    )


def _loaded_linear(activation_dtype: str, device: str):
    from tokenspeed.runtime.layers.linear import ReplicatedLinear

    with torch.device(device):
        layer = ReplicatedLinear(
            128,
            64,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=_config(activation_dtype),
            prefix="model.layers.0.self_attn.q_a_proj",
        )
    generator = torch.Generator().manual_seed(41)
    weight = (torch.randn(64, 128, generator=generator) * 0.1).to(torch.float8_e4m3fn)
    scales = torch.randint(124, 128, (64, 4), generator=generator, dtype=torch.uint8)
    layer.weight.weight_loader(layer.weight, weight)
    layer.weight_scale.weight_loader(layer.weight_scale, scales)
    return layer, weight, scales


@pytest.fixture
def local_model(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"architectures": []}))
    return str(tmp_path)


def test_modelopt_mxfp8_keeps_bf16_default(local_model):
    args = ServerArgs(model=local_model, world_size=1, nprocs_per_node=1)
    assert args.modelopt_mxfp8_activation_dtype == "bf16"
    assert args.moe_backend == "auto"


def test_modelopt_mxfp8_explicit_fp8_reaches_workers(local_model):
    args = ServerArgs(
        model=local_model,
        world_size=1,
        nprocs_per_node=1,
        modelopt_mxfp8_activation_dtype="fp8",
        moe_backend="humming",
        disable_pdl=True,
    )
    original = global_server_args_dict.copy()
    original_pdl = pdl_enabled()
    try:
        global_server_args_dict_update(args)
        assert global_server_args_dict["modelopt_mxfp8_activation_dtype"] == "fp8"
        assert global_server_args_dict["moe_backend"] == "humming"
        assert global_server_args_dict["disable_pdl"] is True
    finally:
        global_server_args_dict.clear()
        global_server_args_dict.update(original)
        pdl_enabled(original_pdl)


@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"moe_backend": "auto"}, "--moe-backend humming"),
        ({"moe_backend": "marlin"}, "--moe-backend humming"),
        ({"disable_pdl": False}, "--disable-pdl"),
        ({"draft_moe_backend": "marlin"}, "draft model"),
        ({"all2all_backend": "deepep"}, "all-to-all"),
        ({"moe_combine_order": "slot"}, "--moe-combine-order rank"),
        ({"enable_eplb": True}, "contiguous expert"),
        ({"ep_num_redundant_experts": 1}, "contiguous expert"),
        ({"init_expert_location": "custom.json"}, "contiguous expert"),
    ],
)
def test_modelopt_mxfp8_rejects_incompatible_fp8_launch(local_model, overrides, reason):
    options = {
        "model": local_model,
        "world_size": 1,
        "nprocs_per_node": 1,
        "modelopt_mxfp8_activation_dtype": "fp8",
        "moe_backend": "humming",
        "disable_pdl": True,
        **overrides,
    }
    with pytest.raises(ValueError, match=reason):
        ServerArgs(**options)


@pytest.mark.parametrize("activation_dtype", ["bf16", "fp8"])
def test_modelopt_activation_choice_preserves_checkpoint_loading(activation_dtype):
    layer, weight, scales = _loaded_linear(activation_dtype, "cpu")
    assert layer.quant_method.activation_dtype == activation_dtype
    assert torch.equal(layer.weight.view(torch.uint8), weight.view(torch.uint8))
    assert torch.equal(layer.weight_scale, scales)


def test_w8a8_preparation_keeps_bytes_and_refuses_live_refits():
    layer, weight, scales = _loaded_linear("fp8", "cpu")
    layer.quant_method.process_weights_after_loading(layer)
    assert torch.equal(layer.weight.view(torch.uint8), weight.view(torch.uint8))
    assert torch.equal(layer.weight_scale, scales)
    with pytest.raises(RuntimeError, match="fresh model load"):
        layer.weight.weight_loader(layer.weight, weight)
    with pytest.raises(RuntimeError, match="fresh model load"):
        layer.weight_scale.weight_loader(layer.weight_scale, scales)


@pytest.mark.parametrize("code", [0, 255])
def test_w8a8_dense_rejects_unsupported_scale_codes(code):
    layer, _, _ = _loaded_linear("fp8", "cpu")
    layer.weight_scale.data[0, 0] = code
    with pytest.raises(ValueError, match="codes in \\[1, 254\\]"):
        layer.quant_method.process_weights_after_loading(layer)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_w8a8_dense_matches_its_quantized_operands_and_absorption():
    from tokenspeed_kernel.ops.quantization import quantize_fp8

    layer, raw_weight, raw_scales = _loaded_linear("fp8", "cuda")
    layer.quant_method.process_weights_after_loading(layer)
    x = torch.randn(19, 128, device="cuda", dtype=torch.bfloat16)
    actual = layer.quant_method.apply(layer, x)
    codes, scales = quantize_fp8(
        x,
        granularity="token_group",
        group_size=32,
        scale_encoding="ue8m0",
        enable_pdl=False,
        solution="triton",
    )
    dequantized_x = codes.float() * torch.exp2(scales.float() - 127).repeat_interleave(
        32, dim=-1
    )
    dequantized_weight = raw_weight.float() * torch.exp2(
        raw_scales.float() - 127
    ).repeat_interleave(32, dim=-1)
    reference = dequantized_x.cpu() @ dequantized_weight.T
    torch.testing.assert_close(actual.cpu().float(), reference, rtol=5e-3, atol=1e-3)
    torch.testing.assert_close(
        layer.quant_method.weight_for_absorption(layer).cpu(),
        dequantized_weight.bfloat16(),
        rtol=0,
        atol=0,
    )
