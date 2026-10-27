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

"""ModelOpt MXFP8 linear loading with explicit W8A16 or group32 W8A8 execution."""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.gemm import Mxfp8LinearPlan, mm, prepare_mxfp8_linear
from torch.nn.parameter import Parameter

from tokenspeed.runtime.layers.quantization.base_config import LinearMethodBase
from tokenspeed.runtime.utils import set_weight_attrs


class ModelOptMxfp8LinearMethod(LinearMethodBase):
    """Keep E4M3/E8M0 shards and explicitly select activation arithmetic."""

    def __init__(
        self,
        *,
        activation_dtype: str,
        use_atomic_add: bool,
        use_fp32_reduce: bool,
        override: str | None,
    ) -> None:
        if activation_dtype not in {"bf16", "fp8"}:
            raise ValueError("ModelOpt MXFP8 activations must be bf16 or fp8")
        if activation_dtype == "fp8" and (use_atomic_add or not use_fp32_reduce):
            raise ValueError(
                "Group32 W8A8 requires full FP32 reduction without atomics"
            )
        self.activation_dtype = activation_dtype
        self.use_atomic_add = use_atomic_add
        self.use_fp32_reduce = use_fp32_reduce
        self.override = override
        self.kernel: Mxfp8LinearPlan | None = None
        self._weights_prepared = False

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        if params_dtype != torch.bfloat16:
            raise ValueError("ModelOpt MXFP8 dense requires BF16 layer inputs")
        if input_size_per_partition <= 0 or input_size_per_partition % 32:
            raise ValueError(
                "ModelOpt MXFP8 TP input partitions must be divisible by 32"
            )
        if not output_partition_sizes or min(output_partition_sizes) <= 0:
            raise ValueError("ModelOpt MXFP8 output partitions must be positive")
        n, k = sum(output_partition_sizes), input_size_per_partition
        loader = extra_weight_attrs.get("weight_loader")

        def load_weight(param, loaded_weight, *args, **kwargs):
            if self._weights_prepared:
                raise RuntimeError("MXFP8 weight refits require a fresh model load")
            if loaded_weight.dtype != torch.float8_e4m3fn:
                raise TypeError(
                    "Expected E4M3 MXFP8 weight; BF16 refits cannot be silently quantized"
                )
            if loader is None:
                if args or kwargs:
                    raise TypeError("No sharded loader was supplied")
                if param.shape != loaded_weight.shape:
                    raise ValueError("MXFP8 weight shape mismatch")
                param.data.copy_(loaded_weight)
            else:
                loader(param, loaded_weight, *args, **kwargs)

        def load_scale(param, loaded_weight, *args, **kwargs):
            if self._weights_prepared:
                raise RuntimeError("MXFP8 scale refits require a fresh model load")
            if loaded_weight.dtype not in (torch.uint8, torch.float8_e8m0fnu):
                raise TypeError("MXFP8 scales must be raw uint8/E8M0 bytes")
            raw = loaded_weight.view(torch.uint8)
            if loader is None:
                if args or kwargs:
                    raise TypeError("No sharded loader was supplied")
                if param.shape != raw.shape:
                    raise ValueError("MXFP8 scale shape mismatch")
                param.data.copy_(raw)
            else:
                loader(param, raw, *args, **kwargs)

        for name, shape, dtype, wrapped_loader in (
            ("weight", (n, k), torch.float8_e4m3fn, load_weight),
            ("weight_scale", (n, k // 32), torch.uint8, load_scale),
        ):
            param = Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
            set_weight_attrs(
                param,
                {
                    **extra_weight_attrs,
                    "input_dim": 1,
                    "output_dim": 0,
                    "weight_loader": wrapped_loader,
                },
            )
            layer.register_parameter(name, param)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self._weights_prepared:
            return
        if self.activation_dtype == "fp8":
            # The same loader serves both backends. Keep its canonical scale
            # name and bytes; public mm performs group32 UE8M0 activation
            # quantization and selects the existing W8A8 GEMM.
            minimum, maximum = torch.aminmax(layer.weight_scale)
            if minimum.item() == 0 or maximum.item() == 255:
                raise ValueError("MXFP8 W8A8 requires E8M0 scale codes in [1, 254]")
            if bool(((layer.weight.view(torch.uint8) & 0x7F) == 0x7F).any()):
                raise ValueError("MXFP8 W8A8 weights must be finite")
            self._weights_prepared = True
            return
        self.kernel = prepare_mxfp8_linear(
            layer.weight.data,
            layer.weight_scale.data,
            input_dtype=torch.bfloat16,
            use_atomic_add=self.use_atomic_add,
            use_fp32_reduce=self.use_fp32_reduce,
            solution="marlin",
            override=self.override,
        )
        # Replace raw parameters rather than retaining an extra full weight copy.
        # Keep their loader contracts so an unsupported refit fails explicitly.
        for name, prepared in (
            ("weight", self.kernel.weight),
            ("weight_scale", self.kernel.weight_scale),
        ):
            old = layer.get_parameter(name)
            param = Parameter(prepared, requires_grad=False)
            set_weight_attrs(param, dict(old.__dict__))
            layer.register_parameter(name, param)
        self._weights_prepared = True

    def weight_for_absorption(self, layer: torch.nn.Module) -> torch.Tensor:
        """Return the BF16 tensor needed by absorbed MLA, not by dense GEMM."""
        if self.kernel is not None:
            return self.kernel.dequantize_for_absorption()
        from tokenspeed.runtime.layers.quantization.modelopt_mxfp8 import (
            dequantize_mxfp8_weight,
        )

        return dequantize_mxfp8_weight(layer.weight, layer.weight_scale)

    def apply(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        if not self._weights_prepared:
            raise RuntimeError("MXFP8 weights must be prepared before forward")
        if self.activation_dtype == "fp8":
            if x.dtype != torch.bfloat16 or x.shape[-1] != layer.weight.shape[1]:
                raise ValueError("ModelOpt MXFP8 W8A8 requires BF16 [..., K] inputs")
            output = mm(
                x.reshape(-1, x.shape[-1]),
                layer.weight,
                B_scales=layer.weight_scale,
                bias=bias,
                out_dtype=torch.bfloat16,
                quant="mxfp8",
                block_size=[1, 32],
                solution="triton",
                override=self.override,
            )
            return output.reshape(*x.shape[:-1], layer.weight.shape[0])
        if self.kernel is None:
            raise RuntimeError("Marlin MXFP8 plan is missing after weight preparation")
        return self.kernel.apply(x, bias)
