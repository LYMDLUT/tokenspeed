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

"""Marlin W8A16 MXFP8 MoE apply for SM90 (Hopper).

Weight-only FP8 experts: MXFP8 weights (E4M3 with one E8M0 scale per 32 input
channels) are dequantized inside the Marlin grouped GEMM and multiplied with
bf16 activations. Hopper has no MXFP8 tensor-core path, and the E8M0 x E4M3
product is exact in bf16, so this matches a bf16 GEMM over the dequantized
weights up to accumulation order.

Both GEMMs reduce split-K partial tiles in fp32 in a fixed, lock-ordered
sequence and the top-k finalize is a plain sum outside the kernel. With
``plan["deterministic_finalize"]`` the route alignment is a stable sort too,
which makes the whole output reproducible run to run.

Marlin runs on Ampere-style mma.sync, which leaves Hopper tensor cores idle
once a batch is compute bound. Above ``_DENSE_MIN_ROUTES_PER_EXPERT`` routes
per expert (prefill), the local experts are instead dequantized from the
Marlin layout into one bf16 scratch per device and fed to the CUTLASS grouped
GEMM; the dequant is exact, so only the accumulation order changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.moe.marlin.mxfp4 import _swiglu_limit
from tokenspeed_kernel.ops.tuning import get_autotune_max_num_tokens
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures
from tokenspeed_kernel.thirdparty.cuda.marlin import gptq_marlin_repack
from tokenspeed_kernel.thirdparty.cuda.marlin_moe import (
    FLOAT8_E4M3FN_ID,
    marlin_make_workspace,
    marlin_permute_scales,
    moe_align_block_size,
    moe_wna16_marlin_gemm,
    mxfp4_marlin_process_scales,
)

if current_platform().is_nvidia:
    from flashinfer import ActivationType, cutlass_fused_moe

MXFP8_BLOCK = 32
# GEMM1 writes 2 * ispp columns and GEMM2 reduces over ispp; Marlin tiles N in
# multiples of 128 and K in multiples of 64.
_ISPP_ALIGNMENT = 64
_HIDDEN_ALIGNMENT = 128
# Marlin's 8-bit repack stores each 16 (k) x 64 (n) tile as 1024 contiguous bytes.
_TILE_K = 16
_TILE_N = 64
# H20 crossover of Marlin vs dequant + CUTLASS bf16 at HY V4 expert shapes.
_DENSE_MIN_ROUTES_PER_EXPERT = 192
# Dequant program width along K: 8 Marlin k-tiles, 4 scale groups.
_DEQUANT_K_TILES = 8

# One bf16 copy of a layer's local experts per (device, geometry). MoE layers
# run one at a time on the compute stream, so every layer reuses it.
_DENSE_SCRATCH: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def _dense_scratch(
    device: torch.device, num_experts: int, hidden: int, ispp: int
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (device, num_experts, hidden, ispp)
    if key not in _DENSE_SCRATCH:
        _DENSE_SCRATCH[key] = (
            torch.empty(
                (num_experts, 2 * ispp, hidden), dtype=torch.bfloat16, device=device
            ),
            torch.empty(
                (num_experts, hidden, ispp), dtype=torch.bfloat16, device=device
            ),
        )
    return _DENSE_SCRATCH[key]


@dataclass(frozen=True)
class _MarlinMxfp8State:
    expert_start: int
    hidden_size: int
    ispp: int
    cutlass_swiglu_limit: torch.Tensor | None
    dense_prefill: bool


@triton.jit
def _marlin_mxfp8_to_bf16_kernel(
    q_ptr,
    s_ptr,
    out_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    ROW_SHIFT: tl.constexpr,
    K_TILES: tl.constexpr,
):
    n_tile = tl.program_id(0)
    k_block = tl.program_id(1)
    expert = tl.program_id(2).to(tl.int64)
    GROUPS: tl.constexpr = K_TILES * 16 // 32

    # Byte b of tile word ((tc_col * 4 + t4) * 8 + warp * 2 + which) holds
    # n = warp * 16 + which * 8 + tc_col, k = (b & 1) * 8 + t4 * 2 + (b >> 1).
    k_tiles = k_block * K_TILES + tl.arange(0, K_TILES)
    q_offs = (
        expert * (N * K)
        + (k_tiles.to(tl.int64)[:, None] * (N // 64) + n_tile) * 1024
        + tl.arange(0, 1024)[None, :]
    )
    w = tl.load(q_ptr + q_offs).to(tl.float8e4nv, bitcast=True).to(tl.float32)
    w = tl.reshape(w, (K_TILES, 8, 4, 4, 2, 2, 2))
    w = tl.permute(w, (3, 4, 1, 0, 6, 2, 5))
    w = tl.reshape(w, (64, GROUPS, 32))

    # Scale byte p of a 64-wide block holds n = (p >> 2 & 1) * 32 + (p & 1) * 16
    # + (p >> 1 & 1) * 8 + (p >> 3); E8M0 e -> 2^(e - 127), 0 -> 0 as in Marlin.
    groups = k_block * GROUPS + tl.arange(0, GROUPS)
    s_offs = (
        expert * (K // 32 * N)
        + groups.to(tl.int64)[:, None] * N
        + n_tile * 64
        + tl.arange(0, 64)[None, :]
    )
    s = tl.load(s_ptr + s_offs).to(tl.int32)
    s = tl.reshape(s, (GROUPS, 8, 2, 2, 2))
    s = tl.permute(s, (2, 4, 3, 1, 0))
    s = tl.reshape(s, (64, GROUPS))
    scale = (s << 23).to(tl.float32, bitcast=True)

    out = tl.reshape(w * scale[:, :, None], (64, K_TILES * 16)).to(tl.bfloat16)
    rows = (n_tile * 64 + ROW_SHIFT) % N + tl.arange(0, 64)
    cols = k_block * K_TILES * 16 + tl.arange(0, K_TILES * 16)
    tl.store(
        out_ptr + expert * (N * K) + rows.to(tl.int64)[:, None] * K + cols[None, :],
        out,
    )


def dequantize_marlin_mxfp8(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    out: torch.Tensor,
    *,
    swap_halves: bool,
) -> torch.Tensor:
    """Dequantize Marlin-repacked MXFP8 experts back to bf16 ``[E, N, K]``.

    Args:
        qweight: :func:`marlin_mxfp8_moe_weights` int32 weights ``[E, K/16, 4N]``.
        scales: Its float8_e8m0fnu scales ``[E, K/32, N]``.
        out: bf16 ``[E, N, K]`` destination.
        swap_halves: Write the second half of N first, turning a
            ``[gate; up]`` w13 into CUTLASS's ``[up; gate]`` order.
    """
    num_experts, size_n, size_k = out.shape
    if out.dtype != torch.bfloat16 or not out.is_contiguous():
        raise ValueError("out must be a contiguous bf16 [E, N, K] tensor")
    if tuple(qweight.shape) != (num_experts, size_k // _TILE_K, size_n * 4):
        raise ValueError(
            f"Marlin weights {tuple(qweight.shape)} do not match out {tuple(out.shape)}"
        )
    if tuple(scales.shape) != (num_experts, size_k // MXFP8_BLOCK, size_n):
        raise ValueError(
            f"Marlin scales {tuple(scales.shape)} do not match out {tuple(out.shape)}"
        )
    k_width = _DEQUANT_K_TILES * _TILE_K
    if size_n % _TILE_N != 0 or size_k % k_width != 0:
        raise ValueError(
            f"dequant needs N%{_TILE_N}==0 and K%{k_width}==0, got N={size_n}, K={size_k}"
        )
    _marlin_mxfp8_to_bf16_kernel[(size_n // _TILE_N, size_k // k_width, num_experts)](
        qweight.view(torch.uint8),
        scales.view(torch.uint8),
        out,
        N=size_n,
        K=size_k,
        ROW_SHIFT=size_n // 2 if swap_halves else 0,
        K_TILES=_DEQUANT_K_TILES,
        num_warps=8,
    )
    return out


@triton.jit
def _local_silu_and_mul_kernel(
    x_ptr,
    out_ptr,
    ids_ptr,
    expert_start,
    num_local_experts,
    num_rows,
    ISPP: tl.constexpr,
    BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
    limit,
    HAS_LIMIT: tl.constexpr,
):
    cols = tl.arange(0, BLOCK)
    cmask = cols < ISPP
    for i in range(ROWS):
        row = tl.program_id(0) * ROWS + i
        if row < num_rows:
            local = tl.load(ids_ptr + row) - expert_start
            if (local >= 0) & (local < num_local_experts):
                base = x_ptr + row.to(tl.int64) * (2 * ISPP)
                gate = tl.load(base + cols, mask=cmask).to(tl.float32)
                up = tl.load(base + ISPP + cols, mask=cmask).to(tl.float32)
                if HAS_LIMIT:
                    gate = tl.minimum(gate, limit)
                    up = tl.clamp(up, -limit, limit)
                out = gate * tl.sigmoid(gate) * up
                tl.store(out_ptr + row.to(tl.int64) * ISPP + cols, out, mask=cmask)


@triton.jit
def _local_topk_sum_kernel(
    y_ptr,
    ids_ptr,
    expert_start,
    num_local_experts,
    out_ptr,
    HIDDEN: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    cmask = cols < HIDDEN
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for k in tl.static_range(TOP_K):
        local = tl.load(ids_ptr + token * TOP_K + k) - expert_start
        if (local >= 0) & (local < num_local_experts):
            acc += tl.load(y_ptr + (token * TOP_K + k) * HIDDEN + cols, mask=cmask).to(
                tl.float32
            )
    tl.store(out_ptr + token * HIDDEN + cols, acc, mask=cmask)


def _block_size_m(routes_per_expert: float) -> int:
    """Pick the token block granularity Marlin schedules over.

    Grow the block until each expert holds under ~0.9 blocks of routed tokens,
    keeping decode batches on tight blocks and prefill batches on wide ones.
    """
    block = 8
    for candidate in (8, 16, 32, 48, 64):
        block = candidate
        if routes_per_expert / candidate < 0.9:
            break
    return block


def marlin_mxfp8_moe_weights(plan: dict, w: torch.nn.Module) -> None:
    """Repack loader-format MXFP8 experts into the Marlin layout, once.

    Input (from the block-FP8 weight loader with ``(1, 32)`` blocks):
    ``w13_weight`` ``[E, 2*ispp, hidden]`` / ``w2_weight`` ``[E, hidden, ispp]``
    float8_e4m3fn and ``w13_weight_scale_inv`` / ``w2_weight_scale_inv`` raw
    E8M0 bytes as uint8 ``[E, N, K//32]``. Output: Marlin-repacked int32
    weights and permuted float8_e8m0fnu scales written back under the same
    names, and the expert geometry in ``w._moe_backend_state``, which must be
    None until then.
    """
    if isinstance(w._moe_backend_state, _MarlinMxfp8State):
        return

    activation = plan["activation"] or w.activation
    if activation not in {"silu", "swiglu"}:
        raise ValueError(f"Marlin MXFP8 MoE does not support activation {activation!r}")
    _swiglu_limit(w)

    w13 = w.w13_weight.data
    w2 = w.w2_weight.data
    w13_scale = w.w13_weight_scale_inv.data
    w2_scale = w.w2_weight_scale_inv.data
    if w13.dtype != torch.float8_e4m3fn or w2.dtype != torch.float8_e4m3fn:
        raise TypeError(
            f"Marlin MXFP8 expects float8_e4m3fn experts, got {w13.dtype}/{w2.dtype}"
        )
    if w13_scale.dtype != torch.uint8 or w2_scale.dtype != torch.uint8:
        raise TypeError(
            "Marlin MXFP8 expects raw E8M0 scales as uint8, got "
            f"{w13_scale.dtype}/{w2_scale.dtype}"
        )
    num_experts, two_ispp, hidden = w13.shape
    ispp = w2.shape[2]
    if two_ispp != 2 * ispp or w2.shape[1] != hidden:
        raise ValueError(
            f"w13 {tuple(w13.shape)} and w2 {tuple(w2.shape)} disagree on the "
            "expert geometry"
        )
    if hidden % _HIDDEN_ALIGNMENT != 0 or ispp % _ISPP_ALIGNMENT != 0:
        raise ValueError(
            f"Marlin MXFP8 needs hidden%{_HIDDEN_ALIGNMENT}==0 and "
            f"ispp%{_ISPP_ALIGNMENT}==0, got hidden={hidden}, ispp={ispp}"
        )
    expected_scales = (
        (num_experts, two_ispp, hidden // MXFP8_BLOCK),
        (num_experts, hidden, ispp // MXFP8_BLOCK),
    )
    if (tuple(w13_scale.shape), tuple(w2_scale.shape)) != expected_scales:
        raise ValueError(
            f"MXFP8 scales {tuple(w13_scale.shape)}/{tuple(w2_scale.shape)} do "
            f"not match the 1x{MXFP8_BLOCK} blocks of the weights {expected_scales}"
        )

    device = w13.device
    perm = torch.empty(0, dtype=torch.int, device=device)

    def _repack(weight: torch.Tensor, size_n: int, size_k: int) -> torch.Tensor:
        # gptq_marlin_repack wants int32 [size_k/4, size_n]: four E4M3 bytes
        # per int32 along K, i.e. the transposed int32 view of [size_n, size_k].
        out = torch.empty(
            (num_experts, size_k // 16, size_n * 4), dtype=torch.int32, device=device
        )
        for e in range(num_experts):
            qw = weight[e].view(torch.int32).T.contiguous()
            out[e] = gptq_marlin_repack(
                qw, perm, size_k=size_k, size_n=size_n, num_bits=8
            )
        return out

    def _permute(scale: torch.Tensor, size_n: int, size_k: int) -> torch.Tensor:
        out = torch.empty(
            (num_experts, size_k // MXFP8_BLOCK, size_n),
            dtype=torch.uint8,
            device=device,
        )
        for e in range(num_experts):
            permuted = marlin_permute_scales(
                scale[e].T.contiguous(),
                size_k=size_k,
                size_n=size_n,
                group_size=MXFP8_BLOCK,
            )
            out[e] = mxfp4_marlin_process_scales(permuted).view(torch.uint8)
        return out.view(torch.float8_e8m0fnu)

    w13_marlin = _repack(w13, size_n=two_ispp, size_k=hidden)
    w13_scale_marlin = _permute(w13_scale, size_n=two_ispp, size_k=hidden)
    w.w13_weight = torch.nn.Parameter(w13_marlin, requires_grad=False)
    w.w13_weight_scale_inv = torch.nn.Parameter(w13_scale_marlin, requires_grad=False)
    del w13, w13_scale
    w2_marlin = _repack(w2, size_n=hidden, size_k=ispp)
    w2_scale_marlin = _permute(w2_scale, size_n=hidden, size_k=ispp)
    w.w2_weight = torch.nn.Parameter(w2_marlin, requires_grad=False)
    w.w2_weight_scale_inv = torch.nn.Parameter(w2_scale_marlin, requires_grad=False)

    limit = _swiglu_limit(w)
    # hidden is already a multiple of the dequant K width (_HIDDEN_ALIGNMENT).
    dense_prefill = ispp % (_DEQUANT_K_TILES * _TILE_K) == 0
    if dense_prefill:
        # Allocated with the weights so KV sizing sees it and no graph pool owns it.
        _dense_scratch(device, num_experts, hidden, ispp)
    w._moe_backend_state = _MarlinMxfp8State(
        # EP ranks own contiguous expert ranges. Routes to any other expert
        # land in the align kernel's invalid lane, which the GEMMs skip.
        expert_start=w.ep_rank * num_experts,
        hidden_size=hidden,
        ispp=ispp,
        cutlass_swiglu_limit=(
            None
            if limit is None
            else torch.full((num_experts,), limit, dtype=torch.float32, device=device)
        ),
        dense_prefill=dense_prefill,
    )


@register_kernel(
    "moe",
    "apply",
    name="marlin_mxfp8_precomputed_moe_apply",
    solution="marlin",
    weight_preprocessor=marlin_mxfp8_moe_weights,
    capability=CapabilityRequirement(
        vendors=frozenset({"nvidia"}),
        min_arch_version=ArchVersion(9, 0),
    ),
    signatures=format_signatures("x", "dense", {torch.bfloat16}),
    traits={
        "weight_dtype": frozenset({"fp8"}),
        "fp8_scale_block_shape": frozenset({(1, MXFP8_BLOCK)}),
        "activation": frozenset({"silu", "swiglu"}),
        # silu_and_mul has no alpha/beta (see _swiglu_limit): standard form only.
        "swiglu_form": frozenset({"standard"}),
        # Large batches run FlashInfer CUTLASS, whose permutation needs
        # distinct expert ids per token.
        "expert_id_repeats": frozenset({False}),
        "routing_mode": frozenset({"precomputed_topk"}),
        "supports_deferred_finalize": frozenset({False}),
        "deterministic_finalize": frozenset({False, True}),
        "supports_ep": frozenset({True}),
        "supports_all_to_all_ep": frozenset({False}),
        "ispp_alignment": frozenset({_ISPP_ALIGNMENT}),
        "hidden_alignment": frozenset({_HIDDEN_ALIGNMENT}),
        "internal_activation_dtype": frozenset({"input"}),
        "supports_bias": frozenset({False}),
    },
    priority=Priority.PORTABLE,
)
def marlin_mxfp8_precomputed_moe_apply(
    plan: dict,
    x: torch.Tensor,
    w: torch.nn.Module,
    router_logits: torch.Tensor,
    topk_weights: torch.Tensor | None = None,
    topk_ids: torch.Tensor | None = None,
    num_tokens_global: int | None = None,
    max_num_tokens_per_gpu: int | None = None,
    do_finalize: bool = True,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Apply a Marlin W8A16 MXFP8 MoE with precomputed routing.

    Args:
        plan: MoE plan; ``activation`` (``silu``/``swiglu``) computes
            ``silu(gate) * up`` clamped to ``w.swiglu_arg.limit`` when the
            checkpoint sets one (``gate`` from above, ``up`` on both sides).
            ``deterministic_finalize`` selects the stable route alignment on
            the Marlin path and the unfused top-k reduction on the dense one.
        x: bf16 hidden states ``[tokens, hidden]``.
        w: MoE layer processed by :func:`marlin_mxfp8_moe_weights`: the
            Marlin-repacked ``w13_weight``/``w2_weight`` (int32), permuted
            ``w13_weight_scale_inv``/``w2_weight_scale_inv`` (float8_e8m0fnu)
            and ``_moe_backend_state``, plus the layer's ``ep_size``/``ep_rank``
            and ``tp_size``/``tp_rank``.
        router_logits: Unused; routing is precomputed.
        topk_weights: Route weights ``[tokens, top_k]``.
        topk_ids: Global expert ids ``[tokens, top_k]``; a token's valid ids
            are distinct. Ids outside this rank's expert range, ``-1``
            included, contribute zero.
        num_tokens_global: Unused; distributed EP dispatch is not owned here.
        max_num_tokens_per_gpu: Unused capacity hint.
        do_finalize: Must be true (no deferred finalize).
        enable_pdl: Launch hint for the dense CUTLASS path.

    Returns:
        This rank's expert contributions ``[tokens, hidden]`` in the dtype of
        ``x``, summed over the top-k routes in a fixed order.
    """
    del router_logits, num_tokens_global, max_num_tokens_per_gpu
    if not do_finalize:
        raise ValueError("Marlin MXFP8 MoE cannot defer finalization")
    if topk_weights is None or topk_ids is None:
        raise ValueError("Marlin MXFP8 MoE requires precomputed top-k")
    if x.dtype != torch.bfloat16:
        raise TypeError(f"Marlin MXFP8 MoE requires bf16 activations, got {x.dtype}")

    activation = plan["activation"] or w.activation
    if activation not in {"silu", "swiglu"}:
        raise ValueError(f"Marlin MXFP8 MoE does not support activation {activation!r}")
    state = w._moe_backend_state
    if not isinstance(state, _MarlinMxfp8State):
        raise TypeError("invalid Marlin MXFP8 MoE state")
    hidden = state.hidden_size
    ispp = state.ispp
    num_local_experts = int(w.w13_weight.shape[0])

    topk_ids = topk_ids.to(torch.int32)
    num_tokens, top_k = topk_ids.shape
    # Size blocks by the routes each expert receives across all ranks: in EP
    # only 1/ep_size of the routes land on this rank's experts.
    routes_per_expert = num_tokens * top_k / (num_local_experts * w.ep_size)
    if state.dense_prefill and routes_per_expert >= _DENSE_MIN_ROUTES_PER_EXPERT:
        return _dense_bf16_apply(plan, x, w, state, topk_weights, topk_ids, enable_pdl)

    # Ids stay global: the align kernel and the two Triton kernels below test
    # them against this rank's range instead of a separately remapped copy.
    topk_ids = topk_ids.contiguous()
    expert_start = state.expert_start
    topk_weights = topk_weights.to(torch.float32)

    block_m = _block_size_m(routes_per_expert)
    sorted_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids,
        block_m,
        num_local_experts,
        expert_start=expert_start,
        deterministic=plan["deterministic_finalize"],
    )
    workspace = marlin_make_workspace(x.device)

    # GEMMs only touch local routes (1/ep_size of the rows), so the outputs are
    # left uninitialized and the activation and the finalize skip the rest
    # instead of streaming all T * top_k rows. The other routes, -1 included,
    # fill blocks of expert id -1 with or without EP; only the EP
    # specialization skips them instead of reading expert -1's weights.
    num_routes = num_tokens * top_k
    gemm1_n = 2 * ispp
    intermediate1 = moe_wna16_marlin_gemm(
        x,
        torch.empty(num_routes, gemm1_n, dtype=x.dtype, device=x.device),
        w.w13_weight,
        w.w13_weight_scale_inv,
        workspace,
        sorted_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        b_q_type_id=FLOAT8_E4M3FN_ID,
        moe_block_size=block_m,
        top_k=top_k,
        mul_topk_weights=False,
        is_ep=True,
        size_m=num_tokens,
        size_n=gemm1_n,
        size_k=hidden,
    )
    limit = _swiglu_limit(w)
    intermediate2 = torch.empty(num_routes, ispp, dtype=x.dtype, device=x.device)
    rows_per_program = 4
    _local_silu_and_mul_kernel[(triton.cdiv(num_routes, rows_per_program),)](
        intermediate1,
        intermediate2,
        topk_ids,
        expert_start,
        num_local_experts,
        num_routes,
        ISPP=ispp,
        BLOCK=triton.next_power_of_2(ispp),
        ROWS=rows_per_program,
        limit=0.0 if limit is None else limit,
        HAS_LIMIT=limit is not None,
        num_warps=8,
    )

    # GEMM2 folds the route weights in, so finalize is a plain sum over top_k.
    intermediate3 = moe_wna16_marlin_gemm(
        intermediate2,
        torch.empty(num_routes, hidden, dtype=x.dtype, device=x.device),
        w.w2_weight,
        w.w2_weight_scale_inv,
        workspace,
        sorted_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        b_q_type_id=FLOAT8_E4M3FN_ID,
        moe_block_size=block_m,
        top_k=1,
        mul_topk_weights=True,
        is_ep=True,
        size_m=num_routes,
        size_n=hidden,
        size_k=ispp,
    )
    out = torch.empty(num_tokens, hidden, dtype=x.dtype, device=x.device)
    block = 2048
    _local_topk_sum_kernel[(num_tokens, triton.cdiv(hidden, block))](
        intermediate3,
        topk_ids,
        expert_start,
        num_local_experts,
        out,
        HIDDEN=hidden,
        TOP_K=top_k,
        BLOCK=block,
        num_warps=8,
    )
    return out


def _dense_bf16_apply(
    plan: dict,
    x: torch.Tensor,
    w: torch.nn.Module,
    state: _MarlinMxfp8State,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    enable_pdl: bool,
) -> torch.Tensor:
    num_local_experts = int(w.w13_weight.shape[0])
    w13, w2 = _dense_scratch(x.device, num_local_experts, state.hidden_size, state.ispp)
    dequantize_marlin_mxfp8(w.w13_weight, w.w13_weight_scale_inv, w13, swap_halves=True)
    dequantize_marlin_mxfp8(w.w2_weight, w.w2_weight_scale_inv, w2, swap_halves=False)
    limit = state.cutlass_swiglu_limit
    return cutlass_fused_moe(
        input=x,
        token_selected_experts=topk_ids,
        token_final_scales=topk_weights.to(torch.float32),
        fc1_expert_weights=w13,
        fc2_expert_weights=w2,
        output_dtype=x.dtype,
        quant_scales=None,
        swiglu_alpha=None,
        swiglu_beta=None,
        swiglu_limit=limit,
        ep_size=w.ep_size,
        ep_rank=w.ep_rank,
        tp_size=w.tp_size,
        tp_rank=w.tp_rank,
        tune_max_num_tokens=get_autotune_max_num_tokens(),
        activation_type=(
            ActivationType.Swiglu if limit is None else ActivationType.SwigluBias
        ),
        enable_pdl=enable_pdl,
        # The fused GEMM2 epilogue reduces the top-k outputs with atomics.
        use_fused_finalize=not plan["deterministic_finalize"],
    )[0]
