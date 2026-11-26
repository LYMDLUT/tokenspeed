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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


"""Hy4-flash fused MoE routing, SwiGLU split, and FP8 quantization Triton kernels."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.attention.kpool.hy4_compress import (
    _torch_to_triton_dtype,
)
from tokenspeed_kernel.ops.moe._triton.common import _float32_to_ordered_key
from tokenspeed_kernel.platform import pdl_enabled


@triton.jit(do_not_specialize=["num_tokens"])
def _hy4_sigmoid_bias_topk_kernel(
    logits_ptr,
    bias_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    num_tokens,
    stride_lt,
    ROUTED_SCALING_FACTOR: tl.constexpr,
    NORM_TOPK_PROB: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    PADDED_EXPERTS: tl.constexpr,
    TOP_K: tl.constexpr,
    ENABLE_PDL: tl.constexpr = False,
    WAIT_AFTER: tl.constexpr = False,
):
    """Single-kernel sigmoid + correction-bias packed-key bitonic top-k router."""
    row = tl.program_id(0).to(tl.int64)
    if row >= num_tokens:
        return
    if ENABLE_PDL and not WAIT_AFTER:
        tl.extra.cuda.gdc_wait()

    expert = tl.arange(0, PADDED_EXPERTS)
    valid = expert < NUM_EXPERTS
    row_logits_ptr = logits_ptr + row * stride_lt
    all_logits = tl.load(
        row_logits_ptr + expert,
        mask=valid,
        other=-float("inf"),
    ).to(tl.float32)
    scores = tl.sigmoid(all_logits)
    bias = tl.load(bias_ptr + expert, mask=valid, other=0.0).to(tl.float32)
    choice = scores + bias
    choice = tl.where(valid & (choice == choice), choice, -float("inf"))

    # Pack ordered FP32 choice score and inverse expert id for lower-id tie break.
    packed = (_float32_to_ordered_key(choice).to(tl.uint64) << 32) | (
        PADDED_EXPERTS - expert
    ).to(tl.uint64)
    selected = tl.topk(packed, TOP_K, dim=0)
    selected_ids = (PADDED_EXPERTS - (selected & 0xFFFFFFFF).to(tl.int32)).to(tl.int32)

    selected_logits = tl.load(row_logits_ptr + selected_ids).to(tl.float32)
    selected_weights = tl.sigmoid(selected_logits)
    if NORM_TOPK_PROB:
        selected_weights = selected_weights / (tl.sum(selected_weights, axis=0) + 1e-20)
    selected_weights = selected_weights * ROUTED_SCALING_FACTOR

    k_idx = tl.arange(0, TOP_K)
    tl.store(topk_ids_ptr + row * TOP_K + k_idx, selected_ids)
    tl.store(topk_weights_ptr + row * TOP_K + k_idx, selected_weights)
    if ENABLE_PDL and WAIT_AFTER:
        tl.extra.cuda.gdc_wait()
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def hy4_sigmoid_bias_topk(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    *,
    top_k: int,
    routed_scaling_factor: float = 1.0,
    norm_topk_prob: bool = True,
    wait_after_compute: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused sigmoid + correction-bias top-k routing for ungrouped Hy4-flash MoE.

    Args:
        router_logits: ``[T, E]`` FP32 router logits.
        correction_bias: ``[E]`` FP32 expert score correction bias.
        top_k: Number of selected experts per token.
        routed_scaling_factor: Post-normalization weight scale factor.
        norm_topk_prob: Whether to L1-normalize selected sigmoid weights.
        wait_after_compute: When True and PDL is active, execute top-k before
            ``gdc_wait()`` so this small kernel overlaps with an independent
            predecessor GEMM that already waited on ``router_logits``'s producer,
            while still enforcing completion before launching dependents.

    Returns:
        ``(topk_weights [T, top_k] fp32, topk_ids [T, top_k] int32)``.
    """
    num_tokens, num_experts = router_logits.shape
    if num_tokens == 0:
        return (
            torch.empty((0, top_k), dtype=torch.float32, device=router_logits.device),
            torch.empty((0, top_k), dtype=torch.int32, device=router_logits.device),
        )
    is_pow2_k = top_k > 0 and (top_k & (top_k - 1)) == 0
    if (
        not router_logits.is_cuda
        or not is_pow2_k
        or num_experts > 1024
        or router_logits.stride(-1) != 1
        or correction_bias.stride(-1) != 1
    ):
        scores = router_logits.float().sigmoid()
        scores_for_choice = scores + correction_bias.float()
        topk_ids = torch.topk(scores_for_choice, k=top_k, dim=-1, sorted=False).indices
        topk_weights = scores.gather(1, topk_ids)
        if norm_topk_prob:
            topk_weights = topk_weights / (
                topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            )
        topk_weights = topk_weights * routed_scaling_factor
        return topk_weights.to(torch.float32), topk_ids.to(torch.int32)

    logits_f32 = (
        router_logits
        if router_logits.dtype == torch.float32
        else router_logits.float().contiguous()
    )
    bias_f32 = (
        correction_bias
        if correction_bias.dtype == torch.float32
        else correction_bias.float().contiguous()
    )
    topk_weights = torch.empty(
        (num_tokens, top_k), dtype=torch.float32, device=router_logits.device
    )
    topk_ids = torch.empty(
        (num_tokens, top_k), dtype=torch.int32, device=router_logits.device
    )
    padded_experts = triton.next_power_of_2(num_experts)
    enable_pdl = pdl_enabled()
    _hy4_sigmoid_bias_topk_kernel[(num_tokens,)](
        logits_f32,
        bias_f32,
        topk_weights,
        topk_ids,
        num_tokens,
        logits_f32.stride(0),
        ROUTED_SCALING_FACTOR=float(routed_scaling_factor),
        NORM_TOPK_PROB=bool(norm_topk_prob),
        NUM_EXPERTS=num_experts,
        PADDED_EXPERTS=padded_experts,
        TOP_K=top_k,
        ENABLE_PDL=enable_pdl,
        WAIT_AFTER=bool(wait_after_compute and enable_pdl),
        num_warps=4 if padded_experts <= 512 else 8,
        **({"launch_pdl": True} if enable_pdl else {}),
    )
    return topk_weights, topk_ids


@triton.jit(
    do_not_specialize=[
        "num_tokens",
        "num_groups",
        "stride_st",
        "stride_sg",
    ]
)
def _hy4_sigmoid_gate_quantize_fp8_kernel(
    attn_ptr,
    gate_ptr,
    q_out_ptr,
    scale_out_ptr,
    num_tokens,
    num_groups,
    stride_at,
    stride_gt,
    stride_qt,
    stride_st,
    stride_sg,
    ORIG_DTYPE: tl.constexpr,
    ENABLE_PDL: tl.constexpr = False,
):
    """Fused ``attn_out * sigmoid(gate)`` + 1x128 E4M3 quantization."""
    tok = tl.program_id(0).to(tl.int64)
    grp = tl.program_id(1).to(tl.int64)
    if tok >= num_tokens or grp >= num_groups:
        return
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()

    cols = grp * 128 + tl.arange(0, 128)
    a = tl.load(attn_ptr + tok * stride_at + cols).to(tl.float32)
    g = tl.load(gate_ptr + tok * stride_gt + cols).to(tl.float32)
    # Match unfused BF16/FP16 sigmoid + multiplication rounding prior to FP8 quant.
    sig_g = tl.sigmoid(g).to(ORIG_DTYPE).to(tl.float32)
    y = (a * sig_g).to(ORIG_DTYPE).to(tl.float32)

    absmax = tl.maximum(tl.max(tl.abs(y), axis=0), 1e-10)
    scale = absmax / 448.0
    q = tl.minimum(tl.maximum(y / scale, -448.0), 448.0).to(tl.float8e4nv)

    tl.store(q_out_ptr + tok * stride_qt + cols, q)
    tl.store(scale_out_ptr + tok * stride_st + grp * stride_sg, scale)
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def hy4_sigmoid_gate_quantize_fp8(
    attn_out: torch.Tensor,
    gate: torch.Tensor,
    *,
    mn_major_tma: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse ``attn_out * sigmoid(gate)`` with 1x128 FP8 E4M3 block quantization.

    Args:
        attn_out: ``[T, D]`` BF16/FP16 attention output with ``D % 128 == 0``.
        gate: ``[T, D]`` gate logits matching ``attn_out`` shape and dtype.
        mn_major_tma: When True, store FP32 scales in Hopper TMA-aligned
            MN-major layout ``stride=(1, ceil(T / 4) * 4)`` so DeepGEMM consumes
            them with zero relayout copy.

    Returns:
        ``(q_fp8 [T, D] float8_e4m3fn, scales [T, D // 128] float32)``.
    """
    num_tokens, dim = attn_out.shape
    if dim % 128 != 0:
        raise ValueError(f"Expected dim % 128 == 0, got {dim}")
    num_groups = dim // 128
    q_out = torch.empty(
        (num_tokens, dim), dtype=torch.float8_e4m3fn, device=attn_out.device
    )
    if mn_major_tma:
        aligned_t = max(((num_tokens + 3) // 4) * 4, 4)
        scale_storage = torch.empty(
            (num_groups, aligned_t), dtype=torch.float32, device=attn_out.device
        )
        scales = scale_storage.t()[:num_tokens]
    else:
        scales = torch.empty(
            (num_tokens, num_groups), dtype=torch.float32, device=attn_out.device
        )
    if num_tokens == 0:
        return q_out, scales

    a_c = attn_out if attn_out.stride(-1) == 1 else attn_out.contiguous()
    g_c = gate if gate.stride(-1) == 1 else gate.contiguous()
    if not a_c.is_cuda:
        y = (a_c * torch.sigmoid(g_c)).float().view(num_tokens, num_groups, 128)
        s = (y.abs().amax(dim=-1).clamp_min(1e-10) / 448.0).to(torch.float32)
        q = (y / s.unsqueeze(-1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        q_out.copy_(q.view(num_tokens, dim))
        scales.copy_(s)
        return q_out, scales

    enable_pdl = pdl_enabled()
    _hy4_sigmoid_gate_quantize_fp8_kernel[(num_tokens, num_groups)](
        a_c,
        g_c,
        q_out,
        scales,
        num_tokens,
        num_groups,
        a_c.stride(0),
        g_c.stride(0),
        q_out.stride(0),
        scales.stride(0),
        scales.stride(1),
        ORIG_DTYPE=_torch_to_triton_dtype(a_c.dtype),
        ENABLE_PDL=enable_pdl,
        num_warps=4,
        **({"launch_pdl": True} if enable_pdl else {}),
    )
    return q_out, scales


@triton.jit(do_not_specialize=["num_tokens"])
def _hy4_split_routed_and_shared_swiglu_kernel(
    packed_ptr,
    routed_out_ptr,
    shared_out_ptr,
    num_tokens,
    stride_pt,
    stride_st,
    ROUTED_DIM: tl.constexpr,
    SHARED_INTER: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_S: tl.constexpr,
    ENABLE_PDL: tl.constexpr = False,
):
    """Fused split of packed ``[routed_down, shared_gate, shared_up]`` + SiLU*up."""
    tok = tl.program_id(0).to(tl.int64)
    branch = tl.program_id(1)
    if tok >= num_tokens:
        return
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()

    row_ptr = packed_ptr + tok * stride_pt
    if branch == 0:
        offs = tl.arange(0, BLOCK_R)
        for start in range(0, ROUTED_DIM, BLOCK_R):
            cols = start + offs
            mask = cols < ROUTED_DIM
            val = tl.load(row_ptr + cols, mask=mask, other=0.0)
            tl.store(routed_out_ptr + tok * ROUTED_DIM + cols, val, mask=mask)
    else:
        g_base = row_ptr + ROUTED_DIM
        u_base = g_base + SHARED_INTER
        out_base = shared_out_ptr + tok * stride_st
        offs = tl.arange(0, BLOCK_S)
        for start in range(0, SHARED_INTER, BLOCK_S):
            cols = start + offs
            mask = cols < SHARED_INTER
            g = tl.load(g_base + cols, mask=mask, other=0.0).to(tl.float32)
            u = tl.load(u_base + cols, mask=mask, other=0.0).to(tl.float32)
            act = (g * tl.sigmoid(g) * u).to(shared_out_ptr.dtype.element_ty)
            tl.store(out_base + cols, act, mask=mask)
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def hy4_split_routed_and_shared_swiglu(
    packed: torch.Tensor,
    routed_dim: int,
    *,
    shared_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split ``packed [T, routed_dim + 2 * shared_inter]`` into ``(routed_in, silu(gate)*up)``.

    When ``shared_out`` is supplied (possibly row-strided with unit inner stride, such as the
    right half of a preallocated ``[T, routed_dim + shared_inter]`` buffer), the shared SwiGLU
    result is written directly into ``shared_out`` to avoid a downstream ``torch.cat``.
    """
    num_tokens, total_dim = packed.shape
    shared_inter = (total_dim - routed_dim) // 2
    routed_out = torch.empty(
        (num_tokens, routed_dim), dtype=packed.dtype, device=packed.device
    )
    if shared_out is None:
        shared_out = torch.empty(
            (num_tokens, shared_inter), dtype=packed.dtype, device=packed.device
        )
    else:
        if (
            shared_out.shape != (num_tokens, shared_inter)
            or shared_out.dtype != packed.dtype
            or shared_out.device != packed.device
            or shared_out.stride(-1) != 1
        ):
            raise ValueError(
                "shared_out must match (num_tokens, shared_inter), dtype, device, and unit inner stride"
            )
    if num_tokens == 0:
        return routed_out, shared_out
    if not packed.is_cuda or packed.stride(-1) != 1:
        routed_out.copy_(packed[:, :routed_dim])
        gate = packed[:, routed_dim : routed_dim + shared_inter]
        up = packed[:, routed_dim + shared_inter :]
        shared_out.copy_(F.silu(gate) * up)
        return routed_out, shared_out

    block_r = min(triton.next_power_of_2(routed_dim), 1024)
    block_s = min(triton.next_power_of_2(shared_inter), 1024)
    enable_pdl = pdl_enabled()
    _hy4_split_routed_and_shared_swiglu_kernel[(num_tokens, 2)](
        packed,
        routed_out,
        shared_out,
        num_tokens,
        packed.stride(0),
        shared_out.stride(0),
        ROUTED_DIM=routed_dim,
        SHARED_INTER=shared_inter,
        BLOCK_R=block_r,
        BLOCK_S=block_s,
        ENABLE_PDL=enable_pdl,
        num_warps=4,
        **({"launch_pdl": True} if enable_pdl else {}),
    )
    return routed_out, shared_out


@triton.jit(
    do_not_specialize=[
        "num_tokens",
        "stride_st",
        "stride_sg",
    ]
)
def _hy4_add_rmsnorm_quantize_fp8_kernel(
    x_ptr,
    res_ptr,
    pre_w_ptr,
    w_ptr,
    res_out_ptr,
    norm_out_ptr,
    q_out_ptr,
    scale_out_ptr,
    num_tokens,
    stride_xt,
    stride_rt,
    stride_rot,
    stride_nt,
    stride_qt,
    stride_st,
    stride_sg,
    pre_eps: tl.constexpr,
    eps: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    BLOCK_G: tl.constexpr,
    ORIG_DTYPE: tl.constexpr,
    HAS_PRE_NORM: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    WRITE_BF16: tl.constexpr,
):
    """Fused (optional pre-RMSNorm + residual add) + RMSNorm + 1x128 FP8 E4M3 quantization."""
    tok = tl.program_id(0).to(tl.int64)
    if tok >= num_tokens:
        return

    offs_g = tl.arange(0, BLOCK_G)
    offs_c = tl.arange(0, 128)
    mask_g = offs_g < NUM_GROUPS
    cols = offs_g[:, None] * 128 + offs_c[None, :]
    mask = mask_g[:, None]

    x_val = tl.load(x_ptr + tok * stride_xt + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_PRE_NORM:
        pre_inv_rms = tl.rsqrt(
            tl.sum(tl.sum(x_val * x_val, axis=1), axis=0) / HIDDEN_SIZE + pre_eps
        )
        pre_w = tl.load(pre_w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x_val = ((x_val * pre_inv_rms) * pre_w).to(ORIG_DTYPE).to(tl.float32)
    if HAS_RESIDUAL:
        r_val = tl.load(res_ptr + tok * stride_rt + cols, mask=mask, other=0.0).to(
            tl.float32
        )
        h_orig = (x_val + r_val).to(ORIG_DTYPE)
        tl.store(res_out_ptr + tok * stride_rot + cols, h_orig, mask=mask)
        h_f = h_orig.to(tl.float32)
    else:
        h_f = x_val

    inv_rms = tl.rsqrt(tl.sum(tl.sum(h_f * h_f, axis=1), axis=0) / HIDDEN_SIZE + eps)
    w_val = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    normed_orig = ((h_f * inv_rms) * w_val).to(ORIG_DTYPE)
    if WRITE_BF16:
        tl.store(norm_out_ptr + tok * stride_nt + cols, normed_orig, mask=mask)

    normed_f = normed_orig.to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(normed_f), axis=1), 1e-10)
    scale = absmax / 448.0
    q = tl.minimum(tl.maximum(normed_f / scale[:, None], -448.0), 448.0).to(
        tl.float8e4nv
    )
    tl.store(q_out_ptr + tok * stride_qt + cols, q, mask=mask)
    tl.store(
        scale_out_ptr + tok * stride_st + offs_g * stride_sg,
        scale,
        mask=mask_g,
    )


def hy4_add_rmsnorm_quantize_fp8(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    residual: torch.Tensor | None = None,
    eps: float = 1e-5,
    pre_norm_weight: torch.Tensor | None = None,
    pre_eps: float = 1e-5,
    mn_major_tma: bool = True,
    write_bf16: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """Fused ``(residual + (pre_norm(x)))`` -> RMSNorm -> 1x128 FP8 E4M3 quantization.

    Returns ``(new_residual, normed_bf16, q_fp8, q_scales)`` where
    ``new_residual`` is None when ``residual`` is None, and ``normed_bf16`` is
    None when ``write_bf16`` is False.
    """
    orig_shape = x.shape
    dim = orig_shape[-1]
    if dim % 128 != 0:
        raise ValueError(f"Expected dim % 128 == 0, got {dim}")
    flat_x = x.reshape(-1, dim)
    if flat_x.stride(-1) != 1:
        flat_x = flat_x.contiguous()
    flat_res = None
    if residual is not None:
        flat_res = residual.reshape(-1, dim)
        if flat_res.stride(-1) != 1:
            flat_res = flat_res.contiguous()

    num_tokens = flat_x.shape[0]
    num_groups = dim // 128
    q_out = torch.empty(
        (num_tokens, dim), dtype=torch.float8_e4m3fn, device=flat_x.device
    )
    if mn_major_tma:
        aligned_t = max(((num_tokens + 3) // 4) * 4, 4)
        scale_storage = torch.empty(
            (num_groups, aligned_t), dtype=torch.float32, device=flat_x.device
        )
        scales = scale_storage.t()[:num_tokens]
    else:
        scales = torch.empty(
            (num_tokens, num_groups), dtype=torch.float32, device=flat_x.device
        )
    res_out = torch.empty_like(flat_x) if flat_res is not None else None
    norm_out = torch.empty_like(flat_x) if write_bf16 else None
    if num_tokens == 0:
        return (
            None if res_out is None else res_out.view(orig_shape),
            None if norm_out is None else norm_out.view(orig_shape),
            q_out,
            scales,
        )

    if not flat_x.is_cuda or num_groups > 64:
        x_in = flat_x
        if pre_norm_weight is not None:
            xf = x_in.float()
            x_in = (
                xf
                * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + pre_eps)
                * pre_norm_weight.float()
            ).to(flat_x.dtype)
        h = x_in if flat_res is None else (x_in + flat_res)
        if res_out is not None:
            res_out.copy_(h)
        hf = h.float()
        normed = (
            hf
            * torch.rsqrt(hf.pow(2).mean(dim=-1, keepdim=True) + eps)
            * weight.float()
        ).to(flat_x.dtype)
        if norm_out is not None:
            norm_out.copy_(normed)
        nf = normed.float().view(num_tokens, num_groups, 128)
        s = (nf.abs().amax(dim=-1).clamp_min(1e-10) / 448.0).to(torch.float32)
        q = (nf / s.unsqueeze(-1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        q_out.copy_(q.view(num_tokens, dim))
        scales.copy_(s)
        return (
            None if res_out is None else res_out.view(orig_shape),
            None if norm_out is None else norm_out.view(orig_shape),
            q_out,
            scales,
        )

    block_g = triton.next_power_of_2(num_groups)
    _hy4_add_rmsnorm_quantize_fp8_kernel[(num_tokens,)](
        flat_x,
        flat_res if flat_res is not None else flat_x,
        pre_norm_weight if pre_norm_weight is not None else weight,
        weight,
        res_out if res_out is not None else flat_x,
        norm_out if norm_out is not None else flat_x,
        q_out,
        scales,
        num_tokens,
        flat_x.stride(0),
        flat_res.stride(0) if flat_res is not None else flat_x.stride(0),
        res_out.stride(0) if res_out is not None else flat_x.stride(0),
        norm_out.stride(0) if norm_out is not None else flat_x.stride(0),
        q_out.stride(0),
        scales.stride(0),
        scales.stride(1),
        pre_eps=float(pre_eps),
        eps=float(eps),
        HIDDEN_SIZE=dim,
        NUM_GROUPS=num_groups,
        BLOCK_G=block_g,
        ORIG_DTYPE=_torch_to_triton_dtype(flat_x.dtype),
        HAS_PRE_NORM=pre_norm_weight is not None,
        HAS_RESIDUAL=flat_res is not None,
        WRITE_BF16=write_bf16,
        num_warps=8 if block_g >= 16 else 4,
    )
    return (
        None if res_out is None else res_out.view(orig_shape),
        None if norm_out is None else norm_out.view(orig_shape),
        q_out,
        scales,
    )


@triton.jit(
    do_not_specialize=[
        "num_tokens",
        "stride_st",
        "stride_sg",
    ]
)
def _hy4_silu_and_mul_quantize_fp8_kernel(
    gate_up_ptr,
    q_out_ptr,
    scale_out_ptr,
    num_tokens,
    num_groups,
    stride_gt,
    stride_qt,
    stride_st,
    stride_sg,
    INTER_SIZE: tl.constexpr,
    ORIG_DTYPE: tl.constexpr,
):
    """Fused ``silu(gate) * up`` + 1x128 FP8 E4M3 quantization."""
    tok = tl.program_id(0).to(tl.int64)
    grp = tl.program_id(1).to(tl.int64)
    if tok >= num_tokens or grp >= num_groups:
        return

    cols = grp * 128 + tl.arange(0, 128)
    row_base = gate_up_ptr + tok * stride_gt
    g = tl.load(row_base + cols).to(tl.float32)
    u = tl.load(row_base + INTER_SIZE + cols).to(tl.float32)
    y = ((g * tl.sigmoid(g)) * u).to(ORIG_DTYPE).to(tl.float32)

    absmax = tl.maximum(tl.max(tl.abs(y), axis=0), 1e-10)
    scale = absmax / 448.0
    q = tl.minimum(tl.maximum(y / scale, -448.0), 448.0).to(tl.float8e4nv)
    tl.store(q_out_ptr + tok * stride_qt + cols, q)
    tl.store(scale_out_ptr + tok * stride_st + grp * stride_sg, scale)


def hy4_silu_and_mul_quantize_fp8(
    gate_up: torch.Tensor,
    *,
    mn_major_tma: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse ``silu(gate) * up`` with 1x128 FP8 E4M3 quantization."""
    num_tokens, two_inter = gate_up.shape
    inter = two_inter // 2
    if two_inter % 2 != 0 or inter % 128 != 0:
        raise ValueError(f"Expected even width with half % 128 == 0, got {two_inter}")
    num_groups = inter // 128
    q_out = torch.empty(
        (num_tokens, inter), dtype=torch.float8_e4m3fn, device=gate_up.device
    )
    if mn_major_tma:
        aligned_t = max(((num_tokens + 3) // 4) * 4, 4)
        scale_storage = torch.empty(
            (num_groups, aligned_t), dtype=torch.float32, device=gate_up.device
        )
        scales = scale_storage.t()[:num_tokens]
    else:
        scales = torch.empty(
            (num_tokens, num_groups), dtype=torch.float32, device=gate_up.device
        )
    if num_tokens == 0:
        return q_out, scales

    gu_c = gate_up if gate_up.stride(-1) == 1 else gate_up.contiguous()
    if not gu_c.is_cuda:
        g, u = gu_c.chunk(2, dim=-1)
        y = (
            (F.silu(g.float()) * u.float())
            .to(gu_c.dtype)
            .float()
            .view(num_tokens, num_groups, 128)
        )
        s = (y.abs().amax(dim=-1).clamp_min(1e-10) / 448.0).to(torch.float32)
        q = (y / s.unsqueeze(-1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        q_out.copy_(q.view(num_tokens, inter))
        scales.copy_(s)
        return q_out, scales

    _hy4_silu_and_mul_quantize_fp8_kernel[(num_tokens, num_groups)](
        gu_c,
        q_out,
        scales,
        num_tokens,
        num_groups,
        gu_c.stride(0),
        q_out.stride(0),
        scales.stride(0),
        scales.stride(1),
        INTER_SIZE=inter,
        ORIG_DTYPE=_torch_to_triton_dtype(gu_c.dtype),
        num_warps=4,
    )
    return q_out, scales


__all__ = [
    "hy4_sigmoid_bias_topk",
    "hy4_sigmoid_gate_quantize_fp8",
    "hy4_split_routed_and_shared_swiglu",
    "hy4_add_rmsnorm_quantize_fp8",
    "hy4_silu_and_mul_quantize_fp8",
]
