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

"""Optional Humming W8A8 bridge for Hopper group32 expert GEMMs.

Install ``requirements/cuda-humming.txt`` explicitly. Weight preparation only
repacks E4M3 weights and E8M0 scales. Activations arrive already quantized, with
power-of-two scales represented as FP32; Humming's default FP32-scale quantizer
uses a different numerical rule and is intentionally not used here.

The prepared plan compiles a fixed Hopper MMA tile before serving. Stream-K is
disabled so each output has one FP32 reduction followed by one BF16 store.
On the validated H20 compiler, this MMA instruction unpacks E4M3 to FP16 before
HMMA. It is an FP8-operand experiment, not an accepted native-FP8 compute path.
Routing, the activation, and weighted reduction belong to the calling op.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from functools import cache
from types import ModuleType

import torch

HUMMING_COMMIT = "64ab616cc26e6d5e4000a8fc3022814d2abd90bb"


@dataclass(frozen=True)
class _HummingImports:
    dtypes: ModuleType
    schema: ModuleType
    transform: ModuleType
    kernel: ModuleType
    launcher: ModuleType


@cache
def _require_humming() -> _HummingImports:
    from tokenspeed_kernel._triton import redirect_triton_to_tokenspeed_triton

    try:
        # Bind upstream's Triton imports to the runtime used by the rest of
        # TokenSpeed. Load every module here so forwards only use cached APIs.
        with redirect_triton_to_tokenspeed_triton():
            return _HummingImports(
                dtypes=importlib.import_module("humming.dtypes"),
                schema=importlib.import_module("humming.schema"),
                transform=importlib.import_module("humming.transform"),
                kernel=importlib.import_module("humming.kernel.humming"),
                launcher=importlib.import_module("humming.ops.launcher"),
            )
    except ImportError as error:
        raise RuntimeError(
            "Humming group32 W8A8 requires the optional dependency from "
            "tokenspeed-kernel/python/requirements/cuda-humming.txt "
            f"(upstream commit {HUMMING_COMMIT})."
        ) from error


@dataclass(frozen=True)
class HummingMXFP8WeightState:
    """Prepared expert weights and fixed, unpadded matrix dimensions."""

    weight: torch.Tensor
    scales: torch.Tensor
    layer_config: str
    num_experts: int
    size_n: int
    size_k: int
    device: torch.device


@dataclass(frozen=True)
class HummingIndexedPlan:
    """Compiled interval table and the alignment needed by each interval.

    ``intervals`` contains ``(exclusive_lower, inclusive_upper, block_m)``.
    ``kernels`` is Humming's CPU launch table, retained for eager and graph use.
    No locks are needed because the plan never splits an output across CTAs.
    """

    state: HummingMXFP8WeightState
    compute_config: str
    tuning_config: str
    intervals: tuple[tuple[int, int, int], ...]
    kernels: torch.Tensor

    @property
    def locks_numel(self) -> int:
        return 0

    def block_m(self, num_routes: int) -> int:
        """Return alignment for a positive, host-known route capacity.

        The caller passes the same count as ``valid_shape_m`` when launching.
        Route counts stored in device memory must not be downloaded to select a
        plan; their host-known capacity is sufficient for this tuning choice.
        """
        for lower, upper, block_m in self.intervals:
            if lower < num_routes <= upper:
                return block_m
        raise ValueError(f"No Humming tuning interval covers {num_routes} routes")


def _validate_mxfp8_weights(weight: torch.Tensor, scales: torch.Tensor) -> None:
    if weight.ndim != 3 or weight.dtype != torch.float8_e4m3fn:
        raise ValueError("Humming weights must be E4M3 [experts, N, K]")
    num_experts, size_n, size_k = weight.shape
    if num_experts == 0 or size_n == 0 or size_k == 0:
        raise ValueError("Humming expert and matrix dimensions must be positive")
    if size_n % 64 or size_k % 64:
        raise ValueError("Humming group32 requires N and K divisible by 64")
    if scales.dtype != torch.uint8 or scales.shape != (
        num_experts,
        size_n,
        size_k // 32,
    ):
        raise ValueError("Humming scales must be uint8 [experts, N, K / 32]")
    if scales.device != weight.device:
        raise ValueError("Humming weights and scales must be on the same device")
    if bool(((weight.view(torch.uint8) & 0x7F) == 0x7F).any()):
        raise ValueError("Humming E4M3 weights must be finite")
    # Load-time validation: Hopper's upstream software E8M0 decoder does not
    # implement code 0 (2^-127), and code 255 denotes an invalid NaN scale.
    minimum, maximum = torch.aminmax(scales)
    if minimum.item() == 0 or maximum.item() == 255:
        raise ValueError(
            "Humming on Hopper requires E8M0 weight scale codes in [1, 254]; "
            "code 0 and NaN code 255 are unsupported"
        )


def prepare_humming_mxfp8_weights(
    weight: torch.Tensor, scales: torch.Tensor
) -> HummingMXFP8WeightState:
    """Repack official group32 expert weights without requantizing them.

    Args:
        weight: E4M3 ``[experts, N, K]`` CUDA tensor. N and K divide by 64.
        scales: Raw E8M0 bytes ``[experts, N, K / 32]`` with codes 1..254.

    Returns:
        A state owning Humming's repacked weights and scales. Call once while
        loading weights, before preparing GEMM plans or capturing CUDA graphs.
    """
    _validate_mxfp8_weights(weight, scales)
    if weight.device.type != "cuda" or torch.cuda.get_device_capability(
        weight.device
    ) != (9, 0):
        raise ValueError("Humming group32 W8A8 is currently supported on SM90")
    api = _require_humming()
    dtypes = api.dtypes

    num_experts, size_n, size_k = weight.shape
    config, tensors = api.transform.prepare_and_transform_humming_tensors(
        tensors={
            "weight": weight.contiguous().view(torch.int32),
            "weight_scale": scales.contiguous().view(torch.float8_e8m0fnu),
        },
        shape_n=size_n,
        shape_k=size_k,
        weight_schema=api.schema.HummingWeightSchema(
            b_dtype=dtypes.float8e4m3,
            bs_dtype=dtypes.float8e8m0,
            weight_scale_group_size=32,
        ),
        input_schema=api.schema.HummingInputSchema(
            a_dtype=dtypes.float8e4m3,
            input_scale_group_size=32,
            input_scale_dtype=dtypes.float32,
            input_quant_mode="dynamic_group",
        ),
        num_experts=num_experts,
        torch_dtype=torch.bfloat16,
        device=weight.device,
    )
    return HummingMXFP8WeightState(
        weight=tensors["weight"],
        scales=tensors["weight_scale"],
        layer_config=config.to_str(),
        num_experts=num_experts,
        size_n=size_n,
        size_k=size_k,
        device=weight.device,
    )


def prepare_humming_indexed(state: HummingMXFP8WeightState) -> HummingIndexedPlan:
    """Compile the fixed MMA tile for the explicit W8A8 indexed path.

    The returned plan owns immutable configuration strings and a cached launch
    table. Preparing it may compile kernels; launching it never tunes or JITs.
    """
    api = _require_humming()

    compute = {
        "gemm_type": "indexed",
        "use_f16_accum": False,
        "use_batch_invariant": False,
        "use_m_major_input_scale": False,
    }
    # The upstream Hopper WGMMA heuristic loses small terms with varied FP8
    # exponents, including within one group32 dot product. Use the verified
    # MMA tile and FP32 group promotion for every route count. The H20 compiler
    # lowers this MMA to FP16 HMMA; it does not meet native-FP8 acceptance. K is
    # not split across CTAs; BF16 rounding follows the complete reduction.
    upper = 1 << 30
    tuning_config = json.dumps(
        [
            (
                0,
                upper,
                {
                    "mma_type": "mma",
                    "block_shape": [32, 64, 64],
                    "warp_shape": [16, 32, 64],
                    "num_stages": 3,
                    "use_stream_k": False,
                },
            )
        ]
    )
    intervals = ((0, upper, 32),)
    compute_config = json.dumps(compute)
    kernels = api.kernel.HummingKernel.prepare_kernels(
        state.layer_config, compute_config, tuning_config, state.device
    )
    return HummingIndexedPlan(
        state=state,
        compute_config=compute_config,
        tuning_config=tuning_config,
        intervals=intervals,
        kernels=kernels,
    )


def humming_indexed_gemm(
    plan: HummingIndexedPlan,
    inputs: torch.Tensor,
    input_scales: torch.Tensor,
    outputs: torch.Tensor,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_padded: torch.Tensor,
    *,
    top_k: int,
    valid_shape_m: int,
) -> torch.Tensor:
    """Execute one prepared expert GEMM into the original flat route order.

    Args:
        plan: Startup-prepared weights and kernel launch table.
        inputs: E4M3 ``[M, K]`` values, already quantized by the caller.
        input_scales: FP32 ``[M, K / 32]`` power-of-two dequantization scales.
        outputs: BF16 ``[M * top_k, N]`` destination.
        sorted_ids: Int32 flat route IDs, grouped by expert and padded to the
            plan's block M. Padding route IDs must be at least ``M * top_k``.
        expert_ids: Int32 local expert ID for every padded M block. All IDs
            used by the kernel must be in ``[0, num_experts)``: Humming does
            not skip negative IDs. The caller must exclude nonlocal routes.
        num_tokens_padded: Int32 device scalar counting only local padded routes.
        top_k: Input-row divisor: routing top-k for GEMM1, one for GEMM2.
        valid_shape_m: Host-known route capacity used for interval selection;
            use the same value when choosing ``plan.block_m`` for alignment.

    Returns:
        ``outputs``. Rows belonging to excluded experts remain unwritten and
        must be masked by the calling activation and deterministic finalizer.
    """
    state = plan.state
    if top_k <= 0 or inputs.ndim != 2:
        raise ValueError("Humming requires positive top_k and two-dimensional input")
    num_rows = inputs.shape[0]
    expected = (
        (inputs, torch.float8_e4m3fn, (num_rows, state.size_k), "inputs"),
        (
            input_scales,
            torch.float32,
            (num_rows, state.size_k // 32),
            "input_scales",
        ),
        (outputs, torch.bfloat16, (num_rows * top_k, state.size_n), "outputs"),
    )
    for tensor, dtype, shape, name in expected:
        if tensor.dtype != dtype or tensor.shape != shape:
            raise ValueError(
                f"Humming {name} must have dtype {dtype} and shape {shape}"
            )
        if tensor.device != state.device or not tensor.is_contiguous():
            raise ValueError(f"Humming {name} must be contiguous on {state.device}")
    for tensor, name in (
        (sorted_ids, "sorted_ids"),
        (expert_ids, "expert_ids"),
        (num_tokens_padded, "num_tokens_padded"),
    ):
        if (
            tensor.dtype != torch.int32
            or tensor.ndim != 1
            or tensor.device != state.device
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"Humming {name} must be contiguous 1D int32 on {state.device}"
            )
    if num_tokens_padded.numel() != 1:
        raise ValueError("Humming num_tokens_padded must contain one device scalar")
    if num_rows == 0:
        return outputs
    if not 0 < valid_shape_m <= num_rows * top_k:
        raise ValueError("Humming valid_shape_m must be a positive route capacity")
    block_m = plan.block_m(valid_shape_m)
    if expert_ids.numel() * block_m < sorted_ids.numel():
        raise ValueError("Humming expert_ids capacity does not cover sorted_ids")
    api = _require_humming()
    return api.launcher.launch_kernel(
        configs=plan.kernels,
        inputs=inputs,
        input_scale=input_scales,
        weight=state.weight,
        weight_scale=state.scales,
        outputs=outputs,
        sorted_ids=sorted_ids,
        expert_ids=expert_ids,
        num_tokens_padded=num_tokens_padded,
        top_k=top_k,
        valid_shape_m=valid_shape_m,
        locks=None,
    )
