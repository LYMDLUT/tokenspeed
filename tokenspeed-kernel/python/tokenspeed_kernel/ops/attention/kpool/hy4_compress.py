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


"""Hy4-flash compressed indexer 4:1 gated pooling, M-RoPE, and KPool store ops."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.kvcache.index_k_scale import (
    encode_index_k_scale,
    index_k_scale_ue8m0,
    index_k_scale_word,
    round_index_k_scale,
)
from tokenspeed_kernel.platform import pdl_enabled

# Per-device atomic arrival counters for Split-N KPool projection (sized to 64 groups).
_KPOOL_SPLITN_COUNTERS: dict[torch.device, torch.Tensor] = {}


def apply_interleaved_rope_last_dim(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply interleaved (even/odd pair) RoPE in FP32 to ``x``.

    Matches ``HYV4FlashRotaryEmbedding`` + ``apply_rotary_pos_emb``:
    ``rotate_half`` maps ``(x0, x1, x2, x3, ...) -> (-x1, x0, -x3, x2, ...)``.
    ``cos`` and ``sin`` have shape ``[..., rope_dim // 2]`` (one entry per pair)
    or ``[..., rope_dim]`` (duplicated halves ``[c, c]``).
    """
    orig_dtype = x.dtype
    rope_dim = x.shape[-1]
    half = rope_dim // 2
    if cos.shape[-1] == rope_dim:
        cos_half = cos[..., :half]
        sin_half = sin[..., :half]
    elif cos.shape[-1] == half:
        cos_half = cos
        sin_half = sin
    else:
        raise ValueError(
            f"cos/sin last dim ({cos.shape[-1]}) incompatible with rope_dim={rope_dim}"
        )
    xf = x.float()
    x_even = xf[..., 0::2]
    x_odd = xf[..., 1::2]
    cf = cos_half.float()
    sf = sin_half.float()
    out_even = x_even * cf - x_odd * sf
    out_odd = x_odd * cf + x_even * sf
    return torch.stack((out_even, out_odd), dim=-1).flatten(-2).to(orig_dtype)


def compute_hy4_mrope_cos_sin(
    positions: torch.Tensor,
    inv_freq: torch.Tensor,
    mrope_section: tuple[int, int, int] | list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute FP32 interleaved 3D M-RoPE ``(cos, sin)`` of shape ``[T, half]``.

    Args:
        positions: ``[T]`` (1D text) or ``[3, T]`` (3D T/H/W M-RoPE).
        inv_freq: ``[half]`` inverse frequencies on the target device.
        mrope_section: Three counts ``(s0, s1, s2)`` summing to ``half``.

    Returns:
        ``(cos, sin)`` in FP32 with shape ``[T, half]``.
    """
    if positions.ndim == 1:
        pos_3d = positions.unsqueeze(0).expand(3, -1)
    elif positions.ndim == 2 and positions.shape[0] == 3:
        pos_3d = positions
    else:
        raise ValueError(
            f"positions must have shape [T] or [3, T], got {tuple(positions.shape)}"
        )
    freqs = pos_3d.float().unsqueeze(-1) * inv_freq.to(
        device=positions.device, dtype=torch.float32
    ).view(1, 1, -1)
    cos_3d = freqs.cos()
    sin_3d = freqs.sin()

    cos = cos_3d[0].clone()
    sin = sin_3d[0].clone()
    for dim_idx, offset in ((1, 1), (2, 2)):
        length = int(mrope_section[dim_idx]) * 3
        sl = slice(offset, length, 3)
        cos[..., sl] = cos_3d[dim_idx, ..., sl]
        sin[..., sl] = sin_3d[dim_idx, ..., sl]
    return cos, sin


@triton.jit(
    do_not_specialize=[
        "num_rows",
        "index_num_pages",
        "stride_iv_page",
        "stride_is_page",
    ]
)
def _hy4_index_k_quantize_scatter_kernel(
    keys_ptr,
    write_slots_ptr,
    index_values_ptr,
    index_scales_ptr,
    num_rows,
    index_num_pages,
    stride_kr,
    stride_iv_page,
    stride_iv_row,
    stride_is_page,
    stride_is_row,
    INDEX_ROWS_PER_PAGE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    UE8M0: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    if row >= num_rows:
        return
    offs = tl.arange(0, BLOCK_D)
    mask = offs < HEAD_DIM
    x = tl.load(keys_ptr + row * stride_kr + offs, mask=mask, other=0.0).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-4)
    scale, word = index_k_scale_word(absmax / 448.0, UE8M0)
    quantized = tl.minimum(tl.maximum(x / scale, -448.0), 448.0)

    slot = tl.load(write_slots_ptr + row).to(tl.int64)
    page = slot // INDEX_ROWS_PER_PAGE
    index_row = slot % INDEX_ROWS_PER_PAGE
    valid = (slot >= 0) & (page < index_num_pages)
    value_base = page * stride_iv_page + index_row * stride_iv_row
    scale_base = page * stride_is_page + index_row * stride_is_row
    tl.store(
        index_values_ptr + value_base + offs,
        quantized.to(index_values_ptr.dtype.element_ty),
        mask=mask & valid,
    )
    tl.store(index_scales_ptr + scale_base, word, mask=valid)


def hy4_pool_group_keys(
    grouped_token_keys: torch.Tensor,
    key_value_weight: torch.Tensor,
    key_gate_weight: torch.Tensor,
    intra_group_ape: torch.Tensor,
    key_norm_weight: torch.Tensor,
    *,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Pool ``[G, K, D]`` normalized token keys into ``[G, D]`` group keys.

    Exact match to ``HYV4FlashCompressedIndexer.pool_groups``:
    values and gates are projected in FP32, ``intra_group_ape`` is added to the
    gates, softmax is computed along the intra-group axis in FP32, and the
    weighted sum is cast back to ``grouped_token_keys.dtype`` before RMSNorm
    (``key_norm``).
    """
    orig_dtype = grouped_token_keys.dtype
    group_f = grouped_token_keys.float()
    values = F.linear(group_f, key_value_weight.float())
    gates = F.linear(
        group_f, key_gate_weight.float()
    ) + intra_group_ape.float().unsqueeze(0)
    weights = F.softmax(gates, dim=1)
    pooled = (values * weights).sum(dim=1).to(orig_dtype)
    pooled_f = pooled.float()
    normed = (
        pooled_f * torch.rsqrt(pooled_f.pow(2).mean(dim=-1, keepdim=True) + eps)
    ).to(orig_dtype)
    return key_norm_weight.to(orig_dtype) * normed


def hy4_quantize_and_store_index_k(
    group_keys: torch.Tensor,
    write_slots: torch.Tensor,
    index_values: torch.Tensor,
    index_scales: torch.Tensor,
) -> None:
    """Quantize ``[G, 128]`` Hy4-flash pooled group keys to FP8+scale and scatter."""
    num_rows, head_dim = group_keys.shape
    if num_rows == 0:
        return
    if head_dim != 128:
        raise ValueError(
            f"hy4_quantize_and_store_index_k expects head_dim=128, got {head_dim}"
        )
    if not group_keys.is_cuda:
        rows_per_page = index_values.shape[1]
        num_pages = index_values.shape[0]
        xf = group_keys.float()
        absmax = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)
        scale = round_index_k_scale(absmax / 448.0)
        q = (xf / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        encoded = encode_index_k_scale(scale)
        slots = write_slots.to(torch.int64)
        valid = slots >= 0
        pages = torch.div(slots, rows_per_page, rounding_mode="floor")
        row_ids = torch.remainder(slots, rows_per_page)
        valid = valid & (pages < num_pages)
        if bool(valid.any()):
            vp = pages[valid]
            vr = row_ids[valid]
            index_values.view(torch.uint8)[vp, vr] = q.view(torch.uint8)[valid]
            index_scales.view(torch.int32)[vp, vr, 0] = encoded.view(torch.int32)[
                valid, 0
            ]
        return

    keys_contig = group_keys.contiguous()
    slots_contig = write_slots.contiguous()
    # `index_k_scale_word` emits raw int32 scale bit patterns; view `index_scales`
    # as int32 so Triton stores raw bits instead of int-to-float casting on SM90.
    scales_i32 = index_scales.view(torch.int32)
    _hy4_index_k_quantize_scatter_kernel[(num_rows,)](
        keys_contig,
        slots_contig,
        index_values,
        scales_i32,
        num_rows,
        index_values.shape[0],
        keys_contig.stride(0),
        index_values.stride(0),
        index_values.stride(1),
        scales_i32.stride(0),
        scales_i32.stride(1),
        INDEX_ROWS_PER_PAGE=index_values.shape[1],
        HEAD_DIM=head_dim,
        UE8M0=index_k_scale_ue8m0(),
        BLOCK_D=128,
        num_warps=4,
    )


def _torch_to_triton_dtype(dtype: torch.dtype):
    if dtype == torch.bfloat16:
        return tl.bfloat16
    if dtype == torch.float16:
        return tl.float16
    if dtype == torch.float32:
        return tl.float32
    raise ValueError(f"Unsupported dtype for Hy4-flash Triton kernel: {dtype}")


@triton.jit(
    do_not_specialize=[
        "num_tokens",
        "num_heads",
        "stride_x_t",
        "stride_o_t",
        "rope_offset",
        "out_offset",
    ]
)
def _hy4_apply_interleaved_rope_inplace_kernel(
    x_ptr,
    out_ptr,
    cos_ptr,
    sin_ptr,
    num_tokens,
    num_heads,
    stride_x_t,
    stride_x_h,
    stride_o_t,
    stride_o_h,
    stride_c_t,
    stride_s_t,
    rope_offset,
    out_offset,
    HALF_DIM: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
    ENABLE_PDL: tl.constexpr = False,
):
    tok = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    if tok >= num_tokens or head >= num_heads:
        return
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()

    pair = tl.arange(0, BLOCK_HALF)
    mask = pair < HALF_DIM
    c = tl.load(cos_ptr + tok * stride_c_t + pair, mask=mask, other=1.0).to(tl.float32)
    s = tl.load(sin_ptr + tok * stride_s_t + pair, mask=mask, other=0.0).to(tl.float32)

    in_base = x_ptr + tok * stride_x_t + head * stride_x_h + rope_offset
    x_e = tl.load(in_base + 2 * pair, mask=mask, other=0.0).to(tl.float32)
    x_o = tl.load(in_base + 2 * pair + 1, mask=mask, other=0.0).to(tl.float32)

    out_base = out_ptr + tok * stride_o_t + head * stride_o_h + out_offset
    out_e = (x_e * c - x_o * s).to(out_ptr.dtype.element_ty)
    out_o = (x_o * c + x_e * s).to(out_ptr.dtype.element_ty)
    tl.store(out_base + 2 * pair, out_e, mask=mask)
    tl.store(out_base + 2 * pair + 1, out_o, mask=mask)
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def hy4_apply_interleaved_rope_inplace(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply interleaved RoPE to the trailing ``2 * cos.shape[-1]`` channels.

    When ``out`` is ``None`` (default), mutates ``x[..., -2*half:]`` in-place and
    returns ``x``. When ``out`` is provided (shape ``[*x.shape[:-1], 2*half]`` or
    matching ``x.shape``), writes the rotated channels to ``out[..., -2*half:]``
    without mutating ``x`` and returns ``out``.
    """
    half = cos.shape[-1]
    rope_dim = 2 * half
    if x.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D tensor for x, got shape {tuple(x.shape)}")
    if x.shape[-1] < rope_dim:
        raise ValueError(
            f"x last dim ({x.shape[-1]}) smaller than rope_dim ({rope_dim})"
        )
    dst = x if out is None else out
    if dst.ndim != x.ndim or dst.shape[:-1] != x.shape[:-1] or dst.shape[-1] < rope_dim:
        raise ValueError(
            f"out shape {tuple(dst.shape)} incompatible with x shape {tuple(x.shape)} "
            f"and rope_dim={rope_dim}"
        )
    num_tokens = x.shape[0]
    if num_tokens == 0:
        return dst
    rope_offset = x.shape[-1] - rope_dim
    out_offset = dst.shape[-1] - rope_dim
    if (
        not x.is_cuda
        or not dst.is_cuda
        or x.stride(-1) != 1
        or dst.stride(-1) != 1
        or cos.stride(-1) != 1
        or sin.stride(-1) != 1
    ):
        cos_view = cos.unsqueeze(1) if x.ndim == 3 else cos
        sin_view = sin.unsqueeze(1) if x.ndim == 3 else sin
        dst[..., out_offset:].copy_(
            apply_interleaved_rope_last_dim(x[..., rope_offset:], cos_view, sin_view)
        )
        return dst

    num_heads = x.shape[1] if x.ndim == 3 else 1
    stride_x_t = x.stride(0)
    stride_x_h = x.stride(1) if x.ndim == 3 else 0
    stride_o_t = dst.stride(0)
    stride_o_h = dst.stride(1) if dst.ndim == 3 else 0
    block_half = triton.next_power_of_2(half)
    enable_pdl = pdl_enabled()
    _hy4_apply_interleaved_rope_inplace_kernel[(num_tokens, num_heads)](
        x,
        dst,
        cos,
        sin,
        num_tokens,
        num_heads,
        stride_x_t,
        stride_x_h,
        stride_o_t,
        stride_o_h,
        cos.stride(0),
        sin.stride(0),
        rope_offset,
        out_offset,
        HALF_DIM=half,
        BLOCK_HALF=block_half,
        ENABLE_PDL=enable_pdl,
        num_warps=1 if block_half <= 64 else 4,
        **({"launch_pdl": True} if enable_pdl else {}),
    )
    return dst


@triton.jit(
    do_not_specialize=[
        "num_tokens",
        "stride_x_t",
        "stride_c_t",
        "stride_s_t",
    ]
)
def _hy4_kv_latent_norm_and_rope_inplace_kernel(
    x_ptr,
    weight_ptr,
    cos_ptr,
    sin_ptr,
    num_tokens,
    stride_x_t,
    stride_c_t,
    stride_s_t,
    eps,
    KV_LORA_RANK: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HALF_ROPE: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
):
    tok = tl.program_id(0).to(tl.int64)
    if tok >= num_tokens:
        return

    row_ptr = x_ptr + tok * stride_x_t
    kv_offs = tl.arange(0, BLOCK_KV)
    kv_mask = kv_offs < KV_LORA_RANK
    kv = tl.load(row_ptr + kv_offs, mask=kv_mask, other=0.0).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(kv * kv, axis=0) / KV_LORA_RANK + eps)
    w = tl.load(weight_ptr + kv_offs, mask=kv_mask, other=0.0).to(
        x_ptr.dtype.element_ty
    )
    normed = (kv * inv_rms).to(x_ptr.dtype.element_ty) * w
    tl.store(row_ptr + kv_offs, normed, mask=kv_mask)

    pair = tl.arange(0, BLOCK_HALF)
    rope_mask = pair < HALF_ROPE
    c = tl.load(cos_ptr + tok * stride_c_t + pair, mask=rope_mask, other=1.0).to(
        tl.float32
    )
    s = tl.load(sin_ptr + tok * stride_s_t + pair, mask=rope_mask, other=0.0).to(
        tl.float32
    )
    rope_base = row_ptr + KV_LORA_RANK
    x_e = tl.load(rope_base + 2 * pair, mask=rope_mask, other=0.0).to(tl.float32)
    x_o = tl.load(rope_base + 2 * pair + 1, mask=rope_mask, other=0.0).to(tl.float32)
    out_e = (x_e * c - x_o * s).to(x_ptr.dtype.element_ty)
    out_o = (x_o * c + x_e * s).to(x_ptr.dtype.element_ty)
    tl.store(rope_base + 2 * pair, out_e, mask=rope_mask)
    tl.store(rope_base + 2 * pair + 1, out_o, mask=rope_mask)


def hy4_kv_latent_norm_and_rope_inplace(
    raw_latent: torch.Tensor,
    norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Fused in-place RMSNorm on ``[:, :kv_lora_rank]`` + interleaved RoPE on ``[:, kv_lora_rank:]``."""
    num_tokens, total_dim = raw_latent.shape
    kv_lora_rank = norm_weight.shape[0]
    half_rope = cos.shape[-1]
    if total_dim != kv_lora_rank + 2 * half_rope:
        raise ValueError(
            f"raw_latent dim {total_dim} != kv_lora_rank ({kv_lora_rank}) + rope_dim ({2 * half_rope})"
        )
    if num_tokens == 0:
        return raw_latent
    if (
        not raw_latent.is_cuda
        or raw_latent.stride(-1) != 1
        or cos.stride(-1) != 1
        or sin.stride(-1) != 1
    ):
        kv_pass = raw_latent[:, :kv_lora_rank]
        xf = kv_pass.float()
        normed = (xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)).to(
            raw_latent.dtype
        ) * norm_weight.to(raw_latent.dtype)
        raw_latent[:, :kv_lora_rank].copy_(normed)
        raw_latent[:, kv_lora_rank:].copy_(
            apply_interleaved_rope_last_dim(raw_latent[:, kv_lora_rank:], cos, sin)
        )
        return raw_latent

    _hy4_kv_latent_norm_and_rope_inplace_kernel[(num_tokens,)](
        raw_latent,
        norm_weight,
        cos,
        sin,
        num_tokens,
        raw_latent.stride(0),
        cos.stride(0),
        sin.stride(0),
        eps,
        KV_LORA_RANK=kv_lora_rank,
        BLOCK_KV=triton.next_power_of_2(kv_lora_rank),
        HALF_ROPE=half_rope,
        BLOCK_HALF=triton.next_power_of_2(half_rope),
        num_warps=4,
    )
    return raw_latent


@triton.jit(
    do_not_specialize=[
        "num_reqs",
        "max_index_pages",
        "stride_pos_axis",
        "stride_pos_tok",
        "stride_it_req",
        "stride_ogp_axis",
        "stride_ogp_grp",
    ]
)
def _hy4_kpool_decode_update_gather_kernel(
    token_keys_ptr,
    positions_3d_ptr,
    seq_lens_ptr,
    req_slots_ptr,
    index_table_ptr,
    tail_token_keys_ptr,
    group_pos0_ptr,
    out_grouped_ptr,
    out_group_pos_ptr,
    out_write_slots_ptr,
    num_reqs,
    max_index_pages,
    stride_tk_row,
    stride_pos_axis,
    stride_pos_tok,
    stride_it_req,
    stride_it_col,
    stride_tail_req,
    stride_tail_col,
    stride_gp0_req,
    stride_og_grp,
    stride_og_k,
    stride_ogp_axis,
    stride_ogp_grp,
    Q_LEN: tl.constexpr,
    KPOOL: tl.constexpr,
    MAX_GROUPS: tl.constexpr,
    TAIL_WIDTH: tl.constexpr,
    INDEX_ROWS_PER_PAGE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ENABLE_PDL: tl.constexpr = False,
):
    req = tl.program_id(0).to(tl.int64)
    if req >= num_reqs:
        return
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < HEAD_DIM
    seq_len = tl.load(seq_lens_ptr + req).to(tl.int64)
    req_slot = tl.load(req_slots_ptr + req).to(tl.int64)
    tail_dtype = tail_token_keys_ptr.dtype.element_ty
    zeros_d = tl.zeros((BLOCK_D,), dtype=tail_dtype)

    # Pre-initialize candidate slots to empty (-1); each KPOOL-sized slice of
    # Q_LEN contains at most one group end, mapped deterministically to step // KPOOL.
    for g in tl.static_range(MAX_GROUPS):
        out_idx = req * MAX_GROUPS + g
        tl.store(out_write_slots_ptr + out_idx, -1)
        tl.store(out_group_pos_ptr + 0 * stride_ogp_axis + out_idx * stride_ogp_grp, 0)
        tl.store(out_group_pos_ptr + 1 * stride_ogp_axis + out_idx * stride_ogp_grp, 0)
        tl.store(out_group_pos_ptr + 2 * stride_ogp_axis + out_idx * stride_ogp_grp, 0)
        for k in tl.static_range(KPOOL):
            tl.store(
                out_grouped_ptr + out_idx * stride_og_grp + k * stride_og_k + offs_d,
                zeros_d,
                mask=mask_d,
            )

    if seq_len <= 0:
        return

    gp0_base = group_pos0_ptr + req_slot * stride_gp0_req
    cur_p0 = tl.load(gp0_base + 0).to(tl.int64)
    cur_p1 = tl.load(gp0_base + 1).to(tl.int64)
    cur_p2 = tl.load(gp0_base + 2).to(tl.int64)

    for step in tl.static_range(Q_LEN):
        tok_pos = seq_len - Q_LEN + step
        if tok_pos >= 0:
            tok_row = req * Q_LEN + step
            tok_key = tl.load(
                token_keys_ptr + tok_row * stride_tk_row + offs_d,
                mask=mask_d,
                other=0.0,
            ).to(tail_dtype)
            tail_col = tok_pos % TAIL_WIDTH
            tail_row_ptr = (
                tail_token_keys_ptr
                + req_slot * stride_tail_req
                + tail_col * stride_tail_col
            )
            tl.store(tail_row_ptr + offs_d, tok_key, mask=mask_d)

            mod_k = tok_pos % KPOOL
            if mod_k == 0:
                cur_p0 = tl.load(
                    positions_3d_ptr + 0 * stride_pos_axis + tok_row * stride_pos_tok
                ).to(tl.int64)
                cur_p1 = tl.load(
                    positions_3d_ptr + 1 * stride_pos_axis + tok_row * stride_pos_tok
                ).to(tl.int64)
                cur_p2 = tl.load(
                    positions_3d_ptr + 2 * stride_pos_axis + tok_row * stride_pos_tok
                ).to(tl.int64)
                tl.store(gp0_base + 0, cur_p0)
                tl.store(gp0_base + 1, cur_p1)
                tl.store(gp0_base + 2, cur_p2)

            if mod_k == KPOOL - 1:
                out_idx = req * MAX_GROUPS + (step // KPOOL)
                group_base = tok_pos - (KPOOL - 1)
                for k in tl.static_range(KPOOL):
                    if k == KPOOL - 1:
                        k_vec = tok_key
                    else:
                        src_col = (group_base + k) % TAIL_WIDTH
                        k_vec = tl.load(
                            tail_token_keys_ptr
                            + req_slot * stride_tail_req
                            + src_col * stride_tail_col
                            + offs_d,
                            mask=mask_d,
                            other=0.0,
                        )
                    tl.store(
                        out_grouped_ptr
                        + out_idx * stride_og_grp
                        + k * stride_og_k
                        + offs_d,
                        k_vec,
                        mask=mask_d,
                    )
                tl.store(
                    out_group_pos_ptr + 0 * stride_ogp_axis + out_idx * stride_ogp_grp,
                    cur_p0,
                )
                tl.store(
                    out_group_pos_ptr + 1 * stride_ogp_axis + out_idx * stride_ogp_grp,
                    cur_p1,
                )
                tl.store(
                    out_group_pos_ptr + 2 * stride_ogp_axis + out_idx * stride_ogp_grp,
                    cur_p2,
                )
                pool_id = tok_pos // KPOOL
                table_col = tl.minimum(
                    tl.maximum(pool_id // INDEX_ROWS_PER_PAGE, 0),
                    max_index_pages - 1,
                )
                write_page = tl.load(
                    index_table_ptr + req * stride_it_req + table_col * stride_it_col
                ).to(tl.int64)
                write_slot = write_page * INDEX_ROWS_PER_PAGE + (
                    pool_id % INDEX_ROWS_PER_PAGE
                )
                tl.store(out_write_slots_ptr + out_idx, write_slot)
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def hy4_kpool_decode_update_and_gather(
    token_keys: torch.Tensor,
    positions_3d: torch.Tensor,
    seq_lens: torch.Tensor,
    req_slots: torch.Tensor,
    index_table: torch.Tensor,
    tail_token_keys: torch.Tensor,
    group_pos0: torch.Tensor,
    *,
    index_rows_per_page: int,
    kpool: int,
    q_len_per_req: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Update KPool tail ring + ``group_pos0`` for all decode steps and gather closed groups.

    Each request can close at most ``max_groups = (q_len_per_req + kpool - 1) // kpool``
    groups across ``q_len_per_req`` consecutive tokens. Unused candidate slots are
    marked with ``write_slots = -1``.
    """
    num_reqs = seq_lens.shape[0]
    head_dim = token_keys.shape[-1]
    max_groups = (q_len_per_req + kpool - 1) // kpool
    total_candidates = num_reqs * max_groups
    device = token_keys.device

    out_grouped = torch.empty(
        (total_candidates, kpool, head_dim),
        dtype=tail_token_keys.dtype,
        device=device,
    )
    out_group_pos = torch.empty((3, total_candidates), dtype=torch.int64, device=device)
    out_write_slots = torch.empty((total_candidates,), dtype=torch.int64, device=device)
    if num_reqs == 0 or total_candidates == 0:
        return out_grouped, out_group_pos, out_write_slots

    if not token_keys.is_cuda:
        tail_width = tail_token_keys.shape[1]
        max_pages = index_table.shape[1]
        out_grouped.zero_()
        out_group_pos.zero_()
        out_write_slots.fill_(-1)
        keys_view = token_keys.view(num_reqs, q_len_per_req, head_dim)
        pos_view = positions_3d.view(3, num_reqs, q_len_per_req)
        for r in range(num_reqs):
            seq_len = int(seq_lens[r].item())
            if seq_len <= 0:
                continue
            rslot = int(req_slots[r].item())
            for step in range(q_len_per_req):
                tok_pos = seq_len - q_len_per_req + step
                if tok_pos < 0:
                    continue
                tail_col = tok_pos % tail_width
                tail_token_keys[rslot, tail_col] = keys_view[r, step].to(
                    tail_token_keys.dtype
                )
                if tok_pos % kpool == 0:
                    group_pos0[rslot] = pos_view[:, r, step]
                if tok_pos % kpool == kpool - 1:
                    out_idx = r * max_groups + (step // kpool)
                    base = tok_pos - (kpool - 1)
                    for k in range(kpool):
                        out_grouped[out_idx, k] = tail_token_keys[
                            rslot, (base + k) % tail_width
                        ]
                    out_group_pos[:, out_idx] = group_pos0[rslot]
                    pool_id = tok_pos // kpool
                    tcol = min(max(pool_id // index_rows_per_page, 0), max_pages - 1)
                    wpage = int(index_table[r, tcol].item())
                    out_write_slots[out_idx] = (
                        wpage * index_rows_per_page + pool_id % index_rows_per_page
                    )
        return out_grouped, out_group_pos, out_write_slots

    keys_contig = token_keys if token_keys.stride(-1) == 1 else token_keys.contiguous()
    pos_contig = positions_3d
    seq_lens_contig = seq_lens if seq_lens.is_contiguous() else seq_lens.contiguous()
    req_slots_contig = (
        req_slots if req_slots.is_contiguous() else req_slots.contiguous()
    )
    enable_pdl = pdl_enabled()
    _hy4_kpool_decode_update_gather_kernel[(num_reqs,)](
        keys_contig,
        pos_contig,
        seq_lens_contig,
        req_slots_contig,
        index_table,
        tail_token_keys,
        group_pos0,
        out_grouped,
        out_group_pos,
        out_write_slots,
        num_reqs,
        index_table.shape[1],
        keys_contig.stride(0),
        pos_contig.stride(0),
        pos_contig.stride(1),
        index_table.stride(0),
        index_table.stride(1),
        tail_token_keys.stride(0),
        tail_token_keys.stride(1),
        group_pos0.stride(0),
        out_grouped.stride(0),
        out_grouped.stride(1),
        out_group_pos.stride(0),
        out_group_pos.stride(1),
        Q_LEN=q_len_per_req,
        KPOOL=kpool,
        MAX_GROUPS=max_groups,
        TAIL_WIDTH=tail_token_keys.shape[1],
        INDEX_ROWS_PER_PAGE=index_rows_per_page,
        HEAD_DIM=head_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
        ENABLE_PDL=enable_pdl,
        num_warps=4,
        **({"launch_pdl": True} if enable_pdl else {}),
    )
    return out_grouped, out_group_pos, out_write_slots


@triton.jit(
    do_not_specialize=[
        "num_groups",
        "index_num_pages",
        "stride_vg_grp",
        "stride_vg_k",
        "stride_ape_k",
        "stride_gp_axis",
        "stride_gp_g",
        "stride_iv_page",
        "stride_iv_row",
        "stride_is_page",
        "stride_is_row",
    ]
)
def _hy4_kpool_finalize_rope_store_kernel(
    val_gate_ptr,
    ape_ptr,
    key_norm_w_ptr,
    group_pos_ptr,
    inv_freq_ptr,
    write_slots_ptr,
    index_values_ptr,
    index_scales_ptr,
    num_groups,
    index_num_pages,
    stride_vg_grp,
    stride_vg_k,
    stride_ape_k,
    stride_gp_axis,
    stride_gp_g,
    stride_iv_page,
    stride_iv_row,
    stride_is_page,
    stride_is_row,
    eps,
    MROPE_S1: tl.constexpr,
    MROPE_S2: tl.constexpr,
    INDEX_ROWS_PER_PAGE: tl.constexpr,
    UE8M0: tl.constexpr,
    ORIG_DTYPE: tl.constexpr,
):
    """Fused KPool=4 softmax + RMSNorm + trailing 64-dim 3D M-RoPE + FP8 scatter for D=128."""
    g = tl.program_id(0).to(tl.int64)
    if g >= num_groups:
        return

    slot = tl.load(write_slots_ptr + g).to(tl.int64)
    if slot < 0:
        return
    page = slot // INDEX_ROWS_PER_PAGE
    if page >= index_num_pages:
        return
    index_row = slot % INDEX_ROWS_PER_PAGE

    pair = tl.arange(0, 64)
    even = 2 * pair
    odd = even + 1
    vg_base = val_gate_ptr + g * stride_vg_grp

    # Load 4 intra-group (value, gate + ape) vectors as 64 even/odd pairs.
    v0_e = tl.load(vg_base + 0 * stride_vg_k + even).to(tl.float32)
    v0_o = tl.load(vg_base + 0 * stride_vg_k + odd).to(tl.float32)
    g0_e = tl.load(vg_base + 0 * stride_vg_k + 128 + even).to(tl.float32) + tl.load(
        ape_ptr + 0 * stride_ape_k + even
    ).to(tl.float32)
    g0_o = tl.load(vg_base + 0 * stride_vg_k + 128 + odd).to(tl.float32) + tl.load(
        ape_ptr + 0 * stride_ape_k + odd
    ).to(tl.float32)

    v1_e = tl.load(vg_base + 1 * stride_vg_k + even).to(tl.float32)
    v1_o = tl.load(vg_base + 1 * stride_vg_k + odd).to(tl.float32)
    g1_e = tl.load(vg_base + 1 * stride_vg_k + 128 + even).to(tl.float32) + tl.load(
        ape_ptr + 1 * stride_ape_k + even
    ).to(tl.float32)
    g1_o = tl.load(vg_base + 1 * stride_vg_k + 128 + odd).to(tl.float32) + tl.load(
        ape_ptr + 1 * stride_ape_k + odd
    ).to(tl.float32)

    v2_e = tl.load(vg_base + 2 * stride_vg_k + even).to(tl.float32)
    v2_o = tl.load(vg_base + 2 * stride_vg_k + odd).to(tl.float32)
    g2_e = tl.load(vg_base + 2 * stride_vg_k + 128 + even).to(tl.float32) + tl.load(
        ape_ptr + 2 * stride_ape_k + even
    ).to(tl.float32)
    g2_o = tl.load(vg_base + 2 * stride_vg_k + 128 + odd).to(tl.float32) + tl.load(
        ape_ptr + 2 * stride_ape_k + odd
    ).to(tl.float32)

    v3_e = tl.load(vg_base + 3 * stride_vg_k + even).to(tl.float32)
    v3_o = tl.load(vg_base + 3 * stride_vg_k + odd).to(tl.float32)
    g3_e = tl.load(vg_base + 3 * stride_vg_k + 128 + even).to(tl.float32) + tl.load(
        ape_ptr + 3 * stride_ape_k + even
    ).to(tl.float32)
    g3_o = tl.load(vg_base + 3 * stride_vg_k + 128 + odd).to(tl.float32) + tl.load(
        ape_ptr + 3 * stride_ape_k + odd
    ).to(tl.float32)

    gm_e = tl.maximum(tl.maximum(g0_e, g1_e), tl.maximum(g2_e, g3_e))
    e0_e = tl.exp(g0_e - gm_e)
    e1_e = tl.exp(g1_e - gm_e)
    e2_e = tl.exp(g2_e - gm_e)
    e3_e = tl.exp(g3_e - gm_e)
    p_e = (v0_e * e0_e + v1_e * e1_e + v2_e * e2_e + v3_e * e3_e) / (
        e0_e + e1_e + e2_e + e3_e
    )

    gm_o = tl.maximum(tl.maximum(g0_o, g1_o), tl.maximum(g2_o, g3_o))
    e0_o = tl.exp(g0_o - gm_o)
    e1_o = tl.exp(g1_o - gm_o)
    e2_o = tl.exp(g2_o - gm_o)
    e3_o = tl.exp(g3_o - gm_o)
    p_o = (v0_o * e0_o + v1_o * e1_o + v2_o * e2_o + v3_o * e3_o) / (
        e0_o + e1_o + e2_o + e3_o
    )

    # Match hy4_pool_group_keys rounding: cast pooled to ORIG_DTYPE before RMSNorm.
    pf_e = p_e.to(ORIG_DTYPE).to(tl.float32)
    pf_o = p_o.to(ORIG_DTYPE).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(pf_e * pf_e + pf_o * pf_o, axis=0) / 128.0 + eps)
    w_e = tl.load(key_norm_w_ptr + even).to(ORIG_DTYPE)
    w_o = tl.load(key_norm_w_ptr + odd).to(ORIG_DTYPE)
    y_e = ((pf_e * inv_rms).to(ORIG_DTYPE) * w_e).to(tl.float32)
    y_o = ((pf_o * inv_rms).to(ORIG_DTYPE) * w_o).to(tl.float32)

    # Trailing 64 channels (pairs 32..63) receive interleaved 3D M-RoPE.
    p0 = tl.load(group_pos_ptr + 0 * stride_gp_axis + g * stride_gp_g).to(tl.float32)
    p1 = tl.load(group_pos_ptr + 1 * stride_gp_axis + g * stride_gp_g).to(tl.float32)
    p2 = tl.load(group_pos_ptr + 2 * stride_gp_axis + g * stride_gp_g).to(tl.float32)
    is_rope = pair >= 32
    rope_pair = tl.where(is_rope, pair - 32, 0)
    inv_f = tl.load(inv_freq_ptr + rope_pair, mask=is_rope, other=0.0).to(tl.float32)
    use_p1 = ((rope_pair % 3) == 1) & (rope_pair < MROPE_S1 * 3)
    use_p2 = ((rope_pair % 3) == 2) & (rope_pair < MROPE_S2 * 3)
    pos_sel = tl.where(use_p1, p1, tl.where(use_p2, p2, p0))
    freq = pos_sel * inv_f
    cos_v = tl.cos(freq).to(ORIG_DTYPE).to(tl.float32)
    sin_v = tl.sin(freq).to(ORIG_DTYPE).to(tl.float32)
    rot_e = tl.where(
        is_rope, (y_e * cos_v - y_o * sin_v).to(ORIG_DTYPE).to(tl.float32), y_e
    )
    rot_o = tl.where(
        is_rope, (y_o * cos_v + y_e * sin_v).to(ORIG_DTYPE).to(tl.float32), y_o
    )

    absmax = tl.maximum(tl.max(tl.maximum(tl.abs(rot_e), tl.abs(rot_o)), axis=0), 1e-4)
    scale, word = index_k_scale_word(absmax / 448.0, UE8M0)
    q_e = tl.minimum(tl.maximum(rot_e / scale, -448.0), 448.0).to(
        index_values_ptr.dtype.element_ty
    )
    q_o = tl.minimum(tl.maximum(rot_o / scale, -448.0), 448.0).to(
        index_values_ptr.dtype.element_ty
    )
    value_base = page * stride_iv_page + index_row * stride_iv_row
    scale_base = page * stride_is_page + index_row * stride_is_row
    tl.store(index_values_ptr + value_base + even, q_e)
    tl.store(index_values_ptr + value_base + odd, q_o)
    tl.store(index_scales_ptr + scale_base, word)


def hy4_kpool_finalize_rope_store(
    val_gate: torch.Tensor,
    intra_group_ape: torch.Tensor,
    key_norm_weight: torch.Tensor,
    group_pos: torch.Tensor,
    inv_freq: torch.Tensor,
    mrope_section: tuple[int, int, int] | list[int],
    write_slots: torch.Tensor,
    index_values: torch.Tensor,
    index_scales: torch.Tensor,
    *,
    orig_dtype: torch.dtype,
    eps: float = 1e-5,
) -> None:
    """Fused KPool softmax + RMSNorm + trailing 3D M-RoPE + FP8 index-K scatter."""
    num_groups, kpool, two_d = val_gate.shape
    if num_groups == 0:
        return
    head_dim = two_d // 2
    rope_dim = 2 * inv_freq.shape[0]
    if (
        not val_gate.is_cuda
        or kpool != 4
        or head_dim != 128
        or rope_dim != 64
        or orig_dtype not in (torch.bfloat16, torch.float16, torch.float32)
    ):
        values, gates = val_gate.split(head_dim, dim=-1)
        weights = F.softmax(gates + intra_group_ape.float().unsqueeze(0), dim=1)
        pooled = (values * weights).sum(dim=1).to(orig_dtype)
        pf = pooled.float()
        normed = (pf * torch.rsqrt(pf.pow(2).mean(dim=-1, keepdim=True) + eps)).to(
            orig_dtype
        )
        pooled = key_norm_weight.to(orig_dtype) * normed
        cos, sin = compute_hy4_mrope_cos_sin(group_pos, inv_freq, mrope_section)
        hy4_apply_interleaved_rope_inplace(
            pooled, cos.to(orig_dtype), sin.to(orig_dtype)
        )
        hy4_quantize_and_store_index_k(pooled, write_slots, index_values, index_scales)
        return

    vg_contig = val_gate.contiguous()
    ape_contig = intra_group_ape.contiguous()
    gp_contig = group_pos.contiguous()
    slots_contig = write_slots.contiguous()
    scales_i32 = index_scales.view(torch.int32)
    _hy4_kpool_finalize_rope_store_kernel[(num_groups,)](
        vg_contig,
        ape_contig,
        key_norm_weight,
        gp_contig,
        inv_freq,
        slots_contig,
        index_values,
        scales_i32,
        num_groups,
        index_values.shape[0],
        vg_contig.stride(0),
        vg_contig.stride(1),
        ape_contig.stride(0),
        gp_contig.stride(0),
        gp_contig.stride(1),
        index_values.stride(0),
        index_values.stride(1),
        scales_i32.stride(0),
        scales_i32.stride(1),
        eps,
        MROPE_S1=int(mrope_section[1]),
        MROPE_S2=int(mrope_section[2]),
        INDEX_ROWS_PER_PAGE=index_values.shape[1],
        UE8M0=index_k_scale_ue8m0(),
        ORIG_DTYPE=_torch_to_triton_dtype(orig_dtype),
        num_warps=4,
    )


@triton.jit(
    do_not_specialize=[
        "num_tokens",
        "stride_qsg",
    ]
)
def _hy4_split_qkv_indexer_norm_rope_kernel(
    packed_ptr,
    q_norm_w_ptr,
    kv_norm_w_ptr,
    tk_norm_w_ptr,
    cos_ptr,
    sin_ptr,
    q_resid_out_ptr,
    q_fp8_out_ptr,
    q_scale_out_ptr,
    latent_out_ptr,
    kv_slots_ptr,
    tk_out_ptr,
    hw_out_ptr,
    num_tokens,
    stride_pt,
    stride_ct,
    stride_st,
    stride_qft,
    stride_qsg,
    stride_kv_page,
    stride_kv_tok,
    q_eps,
    kv_eps,
    tk_eps,
    weights_scale,
    Q_LORA_RANK: tl.constexpr,
    KV_LORA_RANK: tl.constexpr,
    HALF_ROPE: tl.constexpr,
    TK_DIM: tl.constexpr,
    HW_DIM: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
    BLOCK_TK: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    HAS_INDEXER: tl.constexpr,
    QUANT_Q_FP8: tl.constexpr,
    WRITE_Q_BF16: tl.constexpr,
    SCATTER_KV: tl.constexpr,
    KV_PAGE_SIZE: tl.constexpr,
    ENABLE_PDL: tl.constexpr = False,
):
    """Fused split + RMSNorm + M-RoPE (+ optional FP8 quant and direct KV scatter)."""
    tok = tl.program_id(0).to(tl.int64)
    branch = tl.program_id(1)
    if tok >= num_tokens:
        return
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()

    row_ptr = packed_ptr + tok * stride_pt
    if branch == 0:
        q_offs = tl.arange(0, BLOCK_Q)
        q_mask = q_offs < Q_LORA_RANK
        q_val = tl.load(row_ptr + q_offs, mask=q_mask, other=0.0).to(tl.float32)
        q_inv_rms = tl.rsqrt(tl.sum(q_val * q_val, axis=0) / Q_LORA_RANK + q_eps)
        q_w = tl.load(q_norm_w_ptr + q_offs, mask=q_mask, other=0.0).to(
            q_resid_out_ptr.dtype.element_ty
        )
        q_out = (q_val * q_inv_rms).to(q_resid_out_ptr.dtype.element_ty) * q_w
        if WRITE_Q_BF16:
            tl.store(q_resid_out_ptr + tok * Q_LORA_RANK + q_offs, q_out, mask=q_mask)
        if QUANT_Q_FP8:
            qf_in = tl.where(q_mask, q_out.to(tl.float32), 0.0)
            qg_vals = tl.reshape(qf_in, [BLOCK_Q // 128, 128])
            qg_absmax = tl.maximum(tl.max(tl.abs(qg_vals), axis=1), 1e-10)
            qg_scale = qg_absmax / 448.0
            qg_q = tl.minimum(
                tl.maximum(qg_vals / qg_scale[:, None], -448.0), 448.0
            ).to(tl.float8e4nv)
            tl.store(
                q_fp8_out_ptr + tok * stride_qft + q_offs,
                tl.reshape(qg_q, [BLOCK_Q]),
                mask=q_mask,
            )
            qg_idx = tl.arange(0, BLOCK_Q // 128)
            tl.store(
                q_scale_out_ptr + qg_idx * stride_qsg + tok,
                qg_scale,
                mask=qg_idx < (Q_LORA_RANK // 128),
            )
    elif branch == 1:
        if SCATTER_KV:
            slot = tl.load(kv_slots_ptr + tok).to(tl.int64)
            if slot < 0:
                return
            out_base = (
                latent_out_ptr
                + (slot // KV_PAGE_SIZE) * stride_kv_page
                + (slot % KV_PAGE_SIZE) * stride_kv_tok
            )
        else:
            out_base = latent_out_ptr + tok * (KV_LORA_RANK + 2 * HALF_ROPE)
        kv_base = row_ptr + Q_LORA_RANK
        kv_offs = tl.arange(0, BLOCK_KV)
        kv_mask = kv_offs < KV_LORA_RANK
        kv_val = tl.load(kv_base + kv_offs, mask=kv_mask, other=0.0).to(tl.float32)
        kv_inv_rms = tl.rsqrt(tl.sum(kv_val * kv_val, axis=0) / KV_LORA_RANK + kv_eps)
        kv_w = tl.load(kv_norm_w_ptr + kv_offs, mask=kv_mask, other=0.0).to(
            latent_out_ptr.dtype.element_ty
        )
        tl.store(
            out_base + kv_offs,
            (kv_val * kv_inv_rms).to(latent_out_ptr.dtype.element_ty) * kv_w,
            mask=kv_mask,
        )

        pair = tl.arange(0, BLOCK_HALF)
        r_mask = pair < HALF_ROPE
        c = tl.load(cos_ptr + tok * stride_ct + pair, mask=r_mask, other=1.0).to(
            tl.float32
        )
        s = tl.load(sin_ptr + tok * stride_st + pair, mask=r_mask, other=0.0).to(
            tl.float32
        )
        rope_in = kv_base + KV_LORA_RANK
        rope_out = out_base + KV_LORA_RANK
        x_e = tl.load(rope_in + 2 * pair, mask=r_mask, other=0.0).to(tl.float32)
        x_o = tl.load(rope_in + 2 * pair + 1, mask=r_mask, other=0.0).to(tl.float32)
        tl.store(
            rope_out + 2 * pair,
            (x_e * c - x_o * s).to(latent_out_ptr.dtype.element_ty),
            mask=r_mask,
        )
        tl.store(
            rope_out + 2 * pair + 1,
            (x_o * c + x_e * s).to(latent_out_ptr.dtype.element_ty),
            mask=r_mask,
        )
    else:
        if HAS_INDEXER:
            tk_base = row_ptr + Q_LORA_RANK + KV_LORA_RANK + 2 * HALF_ROPE
            tk_offs = tl.arange(0, BLOCK_TK)
            tk_mask = tk_offs < TK_DIM
            tk_val = tl.load(tk_base + tk_offs, mask=tk_mask, other=0.0).to(tl.float32)
            tk_inv_rms = tl.rsqrt(tl.sum(tk_val * tk_val, axis=0) / TK_DIM + tk_eps)
            tk_w = tl.load(tk_norm_w_ptr + tk_offs, mask=tk_mask, other=0.0).to(
                tk_out_ptr.dtype.element_ty
            )
            tl.store(
                tk_out_ptr + tok * TK_DIM + tk_offs,
                (tk_val * tk_inv_rms).to(tk_out_ptr.dtype.element_ty) * tk_w,
                mask=tk_mask,
            )

            hw_base = tk_base + TK_DIM
            hw_offs = tl.arange(0, BLOCK_HW)
            hw_mask = hw_offs < HW_DIM
            hw_val = tl.load(hw_base + hw_offs, mask=hw_mask, other=0.0).to(tl.float32)
            tl.store(
                hw_out_ptr + tok * HW_DIM + hw_offs,
                hw_val * weights_scale,
                mask=hw_mask,
            )
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def _kv_cache_scatter_layout(
    kv_cache_out: torch.Tensor, expected_width: int
) -> tuple[int, int, int]:
    """Return ``(page_size, stride_kv_page, stride_kv_tok)`` for a latent KV buffer."""
    if kv_cache_out.shape[-1] != expected_width or kv_cache_out.stride(-1) != 1:
        raise ValueError(
            f"kv_cache_out must have contiguous last dim {expected_width}, "
            f"got shape {tuple(kv_cache_out.shape)} stride {tuple(kv_cache_out.stride())}"
        )
    if kv_cache_out.ndim == 4:
        if kv_cache_out.shape[2] != 1:
            raise ValueError(
                f"Expected 4D latent KV cache [pages, page_size, 1, D], got {tuple(kv_cache_out.shape)}"
            )
        return kv_cache_out.shape[1], kv_cache_out.stride(0), kv_cache_out.stride(1)
    if kv_cache_out.ndim == 3:
        if kv_cache_out.shape[1] == 1:
            return 1, kv_cache_out.stride(0), 0
        return kv_cache_out.shape[1], kv_cache_out.stride(0), kv_cache_out.stride(1)
    if kv_cache_out.ndim == 2:
        return 1, kv_cache_out.stride(0), 0
    raise ValueError(f"Unsupported kv_cache_out ndim={kv_cache_out.ndim}")


def hy4_split_qkv_indexer_norm_rope(
    packed: torch.Tensor,
    q_norm_weight: torch.Tensor,
    kv_norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    q_eps: float,
    kv_eps: float,
    tk_norm_weight: torch.Tensor | None = None,
    tk_eps: float = 1e-5,
    hw_dim: int = 0,
    weights_scale: float = 1.0,
    quantize_q_fp8_mn_major: bool = False,
    write_q_bf16: bool = True,
    kv_cache_out: torch.Tensor | None = None,
    kv_slots: torch.Tensor | None = None,
):
    """Split packed ``[q_a, kv_a(, token_keys, head_weights)]`` and apply norms + M-RoPE.

    When ``kv_cache_out`` and ``kv_slots`` are both provided, scatters normalized
    ``kv_latent`` and rotated ``k_pe`` directly into ``kv_cache_out[kv_slots]``
    and returns ``None`` for the temporary ``latent`` slot.
    """
    num_tokens = packed.shape[0]
    q_lora = q_norm_weight.shape[0]
    kv_lora = kv_norm_weight.shape[0]
    half_rope = cos.shape[-1]
    kv_total = kv_lora + 2 * half_rope
    has_indexer = tk_norm_weight is not None and hw_dim > 0
    tk_dim = tk_norm_weight.shape[0] if has_indexer else 0
    scatter_kv = kv_cache_out is not None and kv_slots is not None
    if (kv_cache_out is None) != (kv_slots is None):
        raise ValueError(
            "kv_cache_out and kv_slots must either both be set or both be None"
        )

    need_q_bf16 = write_q_bf16 or not quantize_q_fp8_mn_major
    q_resid = (
        torch.empty((num_tokens, q_lora), dtype=packed.dtype, device=packed.device)
        if need_q_bf16
        else None
    )
    latent = (
        None
        if scatter_kv
        else torch.empty(
            (num_tokens, kv_total), dtype=packed.dtype, device=packed.device
        )
    )
    token_keys = (
        torch.empty((num_tokens, tk_dim), dtype=packed.dtype, device=packed.device)
        if has_indexer
        else None
    )
    head_weights = (
        torch.empty((num_tokens, hw_dim), dtype=torch.float32, device=packed.device)
        if has_indexer
        else None
    )
    if num_tokens == 0:
        if quantize_q_fp8_mn_major:
            return (
                q_resid,
                latent,
                token_keys,
                head_weights,
                torch.empty(
                    (0, q_lora), dtype=torch.float8_e4m3fn, device=packed.device
                ),
                torch.empty(
                    (0, q_lora // 128), dtype=torch.float32, device=packed.device
                ),
            )
        return q_resid, latent, token_keys, head_weights

    kv_page_size = 1
    stride_kv_page = kv_total
    stride_kv_tok = kv_total
    if scatter_kv:
        assert kv_cache_out is not None and kv_slots is not None
        kv_page_size, stride_kv_page, stride_kv_tok = _kv_cache_scatter_layout(
            kv_cache_out, kv_total
        )

    can_fuse_q_fp8 = (
        quantize_q_fp8_mn_major
        and q_lora % 128 == 0
        and packed.dtype in (torch.bfloat16, torch.float16)
    )
    if (
        not packed.is_cuda
        or packed.stride(-1) != 1
        or cos.stride(-1) != 1
        or sin.stride(-1) != 1
        or (scatter_kv and (not kv_cache_out.is_cuda or kv_slots.stride(0) != 1))
    ):
        q_raw = packed[:, :q_lora].contiguous()
        qf = q_raw.float()
        q_normed = (qf * torch.rsqrt(qf.pow(2).mean(dim=-1, keepdim=True) + q_eps)).to(
            packed.dtype
        ) * q_norm_weight.to(packed.dtype)
        if q_resid is not None:
            q_resid.copy_(q_normed)
        kv_raw = packed[:, q_lora : q_lora + kv_total].contiguous()
        hy4_kv_latent_norm_and_rope_inplace(
            kv_raw, kv_norm_weight, cos, sin, eps=kv_eps
        )
        if scatter_kv:
            assert kv_cache_out is not None and kv_slots is not None
            valid = kv_slots >= 0
            if valid.any():
                s_idx = kv_slots[valid].to(torch.int64)
                kv_vals = kv_raw[valid].to(kv_cache_out.dtype)
                if kv_cache_out.ndim == 4:
                    kv_cache_out[s_idx // kv_page_size, s_idx % kv_page_size, 0] = (
                        kv_vals
                    )
                elif kv_cache_out.ndim == 3 and kv_cache_out.shape[1] == 1:
                    kv_cache_out[s_idx, 0] = kv_vals
                elif kv_cache_out.ndim == 3:
                    kv_cache_out[s_idx // kv_page_size, s_idx % kv_page_size] = kv_vals
                else:
                    kv_cache_out[s_idx] = kv_vals
        else:
            assert latent is not None
            latent.copy_(kv_raw)
        if has_indexer:
            assert (
                token_keys is not None
                and head_weights is not None
                and tk_norm_weight is not None
            )
            tk_raw = packed[:, q_lora + kv_total : q_lora + kv_total + tk_dim]
            tf = tk_raw.float()
            token_keys.copy_(
                (tf * torch.rsqrt(tf.pow(2).mean(dim=-1, keepdim=True) + tk_eps)).to(
                    packed.dtype
                )
                * tk_norm_weight.to(packed.dtype)
            )
            hw_raw = packed[:, q_lora + kv_total + tk_dim :]
            head_weights.copy_(hw_raw.float() * weights_scale)
        if quantize_q_fp8_mn_major:
            num_groups = q_lora // 128
            aligned_t = max(((num_tokens + 3) // 4) * 4, 4)
            g = q_normed.float().view(num_tokens, num_groups, 128)
            sc = g.abs().amax(dim=-1).clamp_min(1e-10) / 448.0
            q_fp8 = (
                (g / sc.unsqueeze(-1))
                .clamp(-448.0, 448.0)
                .to(torch.float8_e4m3fn)
                .view(num_tokens, q_lora)
            )
            sc_buf = torch.empty(
                (num_groups, aligned_t), dtype=torch.float32, device=packed.device
            )
            sc_buf[:, :num_tokens].copy_(sc.t())
            return (
                q_resid,
                latent,
                token_keys,
                head_weights,
                q_fp8,
                sc_buf[:, :num_tokens].t(),
            )
        return q_resid, latent, token_keys, head_weights

    latent_ptr_arg = kv_cache_out if scatter_kv else latent
    assert latent_ptr_arg is not None
    if can_fuse_q_fp8:
        aligned_t = max(((num_tokens + 3) // 4) * 4, 4)
        q_num_groups = q_lora // 128
        q_fp8 = torch.empty(
            (num_tokens, q_lora), dtype=torch.float8_e4m3fn, device=packed.device
        )
        q_scale_storage = torch.empty(
            (q_num_groups, aligned_t), dtype=torch.float32, device=packed.device
        )
        q_fp8_arg = q_fp8
        q_scale_arg = q_scale_storage
        stride_qft = q_fp8.stride(0)
        stride_qsg = aligned_t
    else:
        q_fp8 = None
        q_scale_storage = None
        q_fp8_arg = latent_ptr_arg
        q_scale_arg = cos
        stride_qft = 0
        stride_qsg = 0

    enable_pdl = pdl_enabled()
    _hy4_split_qkv_indexer_norm_rope_kernel[(num_tokens, 3 if has_indexer else 2)](
        packed,
        q_norm_weight,
        kv_norm_weight,
        tk_norm_weight if has_indexer else kv_norm_weight,
        cos,
        sin,
        q_resid if q_resid is not None else latent_ptr_arg,
        q_fp8_arg,
        q_scale_arg,
        latent_ptr_arg,
        kv_slots if scatter_kv else cos,
        token_keys if has_indexer else latent_ptr_arg,
        head_weights if has_indexer else cos,
        num_tokens,
        packed.stride(0),
        cos.stride(0),
        sin.stride(0),
        stride_qft,
        stride_qsg,
        stride_kv_page,
        stride_kv_tok,
        float(q_eps),
        float(kv_eps),
        float(tk_eps),
        float(weights_scale),
        Q_LORA_RANK=q_lora,
        KV_LORA_RANK=kv_lora,
        HALF_ROPE=half_rope,
        TK_DIM=tk_dim if has_indexer else 1,
        HW_DIM=hw_dim if has_indexer else 1,
        BLOCK_Q=triton.next_power_of_2(q_lora),
        BLOCK_KV=triton.next_power_of_2(kv_lora),
        BLOCK_HALF=triton.next_power_of_2(half_rope),
        BLOCK_TK=triton.next_power_of_2(tk_dim) if has_indexer else 1,
        BLOCK_HW=triton.next_power_of_2(hw_dim) if has_indexer else 1,
        HAS_INDEXER=has_indexer,
        QUANT_Q_FP8=can_fuse_q_fp8,
        WRITE_Q_BF16=q_resid is not None,
        SCATTER_KV=scatter_kv,
        KV_PAGE_SIZE=kv_page_size,
        ENABLE_PDL=enable_pdl,
        num_warps=4,
        **({"launch_pdl": True} if enable_pdl else {}),
    )
    if quantize_q_fp8_mn_major:
        assert q_fp8 is not None and q_scale_storage is not None
        return (
            q_resid,
            latent,
            token_keys,
            head_weights,
            q_fp8,
            q_scale_storage[:, :num_tokens].t(),
        )
    return q_resid, latent, token_keys, head_weights


@triton.jit(
    do_not_specialize=[
        "num_groups",
        "num_pages",
        "stride_gp_axis",
        "stride_gp_g",
    ]
)
def _hy4_kpool_project_finalize_rope_store_kernel(
    gk_ptr,
    wt_ptr,
    ape_ptr,
    key_norm_w_ptr,
    group_pos_ptr,
    inv_freq_ptr,
    write_slots_ptr,
    index_values_ptr,
    index_scales_ptr,
    num_groups,
    num_pages,
    stride_gk_g,
    stride_gk_k,
    stride_wt_k,
    stride_ape_k,
    stride_gp_axis,
    stride_gp_g,
    stride_iv_page,
    stride_iv_row,
    stride_is_page,
    stride_is_row,
    eps,
    MROPE_S1: tl.constexpr,
    MROPE_S2: tl.constexpr,
    INDEX_ROWS_PER_PAGE: tl.constexpr,
    UE8M0: tl.constexpr,
    ORIG_DTYPE: tl.constexpr,
    BLOCK_K: tl.constexpr = 32,
    ENABLE_PDL: tl.constexpr = False,
):
    """Fused KPool [w_val; w_gate] FP32 projection + softmax + RMSNorm + M-RoPE + FP8 scatter.

    Exits immediately when ``write_slots[g] < 0`` so idle decode steps skip the
    projection entirely without host sync.
    """
    g = tl.program_id(0).to(tl.int64)
    if g >= num_groups:
        return
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
    slot = tl.load(write_slots_ptr + g).to(tl.int64)
    if slot < 0:
        if ENABLE_PDL:
            tl.extra.cuda.gdc_launch_dependents()
        return
    page = slot // INDEX_ROWS_PER_PAGE
    index_row = slot % INDEX_ROWS_PER_PAGE
    if page >= num_pages:
        if ENABLE_PDL:
            tl.extra.cuda.gdc_launch_dependents()
        return

    r = tl.arange(0, 16)
    r_mask = r < 4
    kk = tl.arange(0, 128)
    c128 = tl.arange(0, 128)
    # Single-shot high-precision Tensor-Core projection [16, 128] @ [128, 128]:
    # for BF16 keys, exact 2-pass BF16 weight residual split achieves ~1e-5 FP32
    # accuracy at full BF16 WGMMA throughput; other dtypes use tf32x3 (~2e-6).
    wv = tl.load(wt_ptr + kk[:, None] * stride_wt_k + c128[None, :]).to(tl.float32)
    wg = tl.load(wt_ptr + kk[:, None] * stride_wt_k + (128 + c128)[None, :]).to(
        tl.float32
    )
    if ORIG_DTYPE == tl.bfloat16:
        x_bf = tl.load(
            gk_ptr + g * stride_gk_g + r[:, None] * stride_gk_k + kk[None, :],
            mask=r_mask[:, None],
            other=0.0,
        ).to(tl.bfloat16)
        wv_hi = wv.to(tl.bfloat16)
        wv_lo = (wv - wv_hi.to(tl.float32)).to(tl.bfloat16)
        wg_hi = wg.to(tl.bfloat16)
        wg_lo = (wg - wg_hi.to(tl.float32)).to(tl.bfloat16)
        v = tl.dot(x_bf, wv_lo, tl.dot(x_bf, wv_hi))
        ga = tl.dot(x_bf, wg_lo, tl.dot(x_bf, wg_hi))
    else:
        x = tl.load(
            gk_ptr + g * stride_gk_g + r[:, None] * stride_gk_k + kk[None, :],
            mask=r_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        v = tl.dot(x, wv, input_precision="tf32x3")
        ga = tl.dot(x, wg, input_precision="tf32x3")

    ape = tl.load(
        ape_ptr + r[:, None] * stride_ape_k + c128[None, :],
        mask=r_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    gl = tl.where(r_mask[:, None], ga + ape, -float("inf"))
    gm = tl.max(gl, axis=0)
    el = tl.where(r_mask[:, None], tl.exp(gl - gm[None, :]), 0.0)
    p = tl.sum(v * el, axis=0) / tl.sum(el, axis=0)

    pf = p.to(ORIG_DTYPE).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(pf * pf, axis=0) / 128.0 + eps)
    nw = tl.load(key_norm_w_ptr + c128).to(ORIG_DTYPE)
    y = ((pf * inv_rms).to(ORIG_DTYPE) * nw).to(tl.float32)

    # Split first 64 and trailing 64 channels; apply 3D M-RoPE to trailing 64.
    y_lo, y_hi = tl.split(tl.trans(tl.reshape(y, [2, 64])))
    y_hi_e, y_hi_o = tl.split(tl.reshape(y_hi, [32, 2]))
    c = tl.arange(0, 64)
    p0 = tl.load(group_pos_ptr + 0 * stride_gp_axis + g * stride_gp_g).to(tl.float32)
    p1 = tl.load(group_pos_ptr + 1 * stride_gp_axis + g * stride_gp_g).to(tl.float32)
    p2 = tl.load(group_pos_ptr + 2 * stride_gp_axis + g * stride_gp_g).to(tl.float32)
    rope_pair = tl.arange(0, 32)
    inv_f = tl.load(inv_freq_ptr + rope_pair).to(tl.float32)
    use_p1 = ((rope_pair % 3) == 1) & (rope_pair < MROPE_S1 * 3)
    use_p2 = ((rope_pair % 3) == 2) & (rope_pair < MROPE_S2 * 3)
    pos_sel = tl.where(use_p1, p1, tl.where(use_p2, p2, p0))
    freq = pos_sel * inv_f
    cos_v = tl.cos(freq).to(ORIG_DTYPE).to(tl.float32)
    sin_v = tl.sin(freq).to(ORIG_DTYPE).to(tl.float32)
    rot_e = (y_hi_e * cos_v - y_hi_o * sin_v).to(ORIG_DTYPE).to(tl.float32)
    rot_o = (y_hi_o * cos_v + y_hi_e * sin_v).to(ORIG_DTYPE).to(tl.float32)
    rot_hi = tl.reshape(tl.join(rot_e, rot_o), [64])

    absmax = tl.maximum(
        tl.maximum(tl.max(tl.abs(y_lo), axis=0), tl.max(tl.abs(rot_hi), axis=0)),
        1e-4,
    )
    scale, word = index_k_scale_word(absmax / 448.0, UE8M0)
    q_lo = tl.minimum(tl.maximum(y_lo / scale, -448.0), 448.0).to(
        index_values_ptr.dtype.element_ty
    )
    q_hi = tl.minimum(tl.maximum(rot_hi / scale, -448.0), 448.0).to(
        index_values_ptr.dtype.element_ty
    )
    value_base = page * stride_iv_page + index_row * stride_iv_row
    scale_base = page * stride_is_page + index_row * stride_is_row
    tl.store(index_values_ptr + value_base + c, q_lo)
    tl.store(index_values_ptr + value_base + 64 + c, q_hi)
    tl.store(index_scales_ptr + scale_base, word)
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit(
    do_not_specialize=[
        "num_groups",
        "num_pages",
        "stride_gp_axis",
        "stride_gp_g",
    ]
)
def _hy4_kpool_project_finalize_rope_store_splitn_kernel(
    gk_ptr,
    wt_ptr,
    ape_ptr,
    key_norm_w_ptr,
    group_pos_ptr,
    inv_freq_ptr,
    write_slots_ptr,
    index_values_ptr,
    index_scales_ptr,
    p_buf_ptr,
    cnt_ptr,
    num_groups,
    num_pages,
    stride_gk_g,
    stride_gk_k,
    stride_wt_k,
    stride_ape_k,
    stride_gp_axis,
    stride_gp_g,
    stride_iv_page,
    stride_iv_row,
    stride_is_page,
    stride_is_row,
    eps,
    MROPE_S1: tl.constexpr,
    MROPE_S2: tl.constexpr,
    INDEX_ROWS_PER_PAGE: tl.constexpr,
    UE8M0: tl.constexpr,
    ORIG_DTYPE: tl.constexpr,
    BN: tl.constexpr,
    NUM_NTILES: tl.constexpr,
    ENABLE_PDL: tl.constexpr = False,
):
    """Split-N KPool BF16 projection + softmax + last-CTA RMSNorm/M-RoPE/FP8 scatter.

    Spreads the 131 KB cold FP32 weight matrix across ``NUM_NTILES`` CTAs per
    group (BN=32 -> 4 CTAs/group), cutting register spills from 298 to 8 and
    halving decode latency while preserving bitwise-identical FP8 outputs.
    """
    g = tl.program_id(0).to(tl.int64)
    nt = tl.program_id(1).to(tl.int64)
    if g >= num_groups:
        return
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
    slot = tl.load(write_slots_ptr + g).to(tl.int64)
    if slot < 0:
        if ENABLE_PDL and nt == 0:
            tl.extra.cuda.gdc_launch_dependents()
        return
    page = slot // INDEX_ROWS_PER_PAGE
    index_row = slot % INDEX_ROWS_PER_PAGE
    if page >= num_pages:
        if ENABLE_PDL and nt == 0:
            tl.extra.cuda.gdc_launch_dependents()
        return

    r = tl.arange(0, 16)
    r_mask = r < 4
    kk = tl.arange(0, 128)
    cn = nt * BN + tl.arange(0, BN)
    x_bf = tl.load(
        gk_ptr + g * stride_gk_g + r[:, None] * stride_gk_k + kk[None, :],
        mask=r_mask[:, None],
        other=0.0,
    ).to(tl.bfloat16)
    wv = tl.load(
        wt_ptr + kk[:, None] * stride_wt_k + cn[None, :],
        eviction_policy="evict_first",
    ).to(tl.float32)
    wv_hi = wv.to(tl.bfloat16)
    wv_lo = (wv - wv_hi.to(tl.float32)).to(tl.bfloat16)
    v = tl.dot(x_bf, wv_lo, tl.dot(x_bf, wv_hi))

    wg = tl.load(
        wt_ptr + kk[:, None] * stride_wt_k + (128 + cn)[None, :],
        eviction_policy="evict_first",
    ).to(tl.float32)
    wg_hi = wg.to(tl.bfloat16)
    wg_lo = (wg - wg_hi.to(tl.float32)).to(tl.bfloat16)
    ga = tl.dot(x_bf, wg_lo, tl.dot(x_bf, wg_hi))

    ape = tl.load(
        ape_ptr + r[:, None] * stride_ape_k + cn[None, :],
        mask=r_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    gl = tl.where(r_mask[:, None], ga + ape, -float("inf"))
    gm = tl.max(gl, axis=0)
    el = tl.where(r_mask[:, None], tl.exp(gl - gm[None, :]), 0.0)
    p = (tl.sum(v * el, axis=0) / tl.sum(el, axis=0)).to(ORIG_DTYPE)
    tl.store(p_buf_ptr + g * 128 + cn, p, cache_modifier=".cg")

    if tl.atomic_add(cnt_ptr + g, 1, sem="acq_rel", scope="gpu") != NUM_NTILES - 1:
        return
    tl.store(cnt_ptr + g, 0)

    c128 = tl.arange(0, 128)
    pf = tl.load(p_buf_ptr + g * 128 + c128, cache_modifier=".cg").to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(pf * pf, axis=0) / 128.0 + eps)
    nw = tl.load(key_norm_w_ptr + c128).to(ORIG_DTYPE)
    y = ((pf * inv_rms).to(ORIG_DTYPE) * nw).to(tl.float32)

    y_lo, y_hi = tl.split(tl.trans(tl.reshape(y, [2, 64])))
    y_hi_e, y_hi_o = tl.split(tl.reshape(y_hi, [32, 2]))
    c = tl.arange(0, 64)
    p0 = tl.load(group_pos_ptr + 0 * stride_gp_axis + g * stride_gp_g).to(tl.float32)
    p1 = tl.load(group_pos_ptr + 1 * stride_gp_axis + g * stride_gp_g).to(tl.float32)
    p2 = tl.load(group_pos_ptr + 2 * stride_gp_axis + g * stride_gp_g).to(tl.float32)
    rope_pair = tl.arange(0, 32)
    inv_f = tl.load(inv_freq_ptr + rope_pair).to(tl.float32)
    use_p1 = ((rope_pair % 3) == 1) & (rope_pair < MROPE_S1 * 3)
    use_p2 = ((rope_pair % 3) == 2) & (rope_pair < MROPE_S2 * 3)
    pos_sel = tl.where(use_p1, p1, tl.where(use_p2, p2, p0))
    freq = pos_sel * inv_f
    cos_v = tl.cos(freq).to(ORIG_DTYPE).to(tl.float32)
    sin_v = tl.sin(freq).to(ORIG_DTYPE).to(tl.float32)
    rot_e = (y_hi_e * cos_v - y_hi_o * sin_v).to(ORIG_DTYPE).to(tl.float32)
    rot_o = (y_hi_o * cos_v + y_hi_e * sin_v).to(ORIG_DTYPE).to(tl.float32)
    rot_hi = tl.reshape(tl.join(rot_e, rot_o), [64])

    absmax = tl.maximum(
        tl.maximum(tl.max(tl.abs(y_lo), axis=0), tl.max(tl.abs(rot_hi), axis=0)),
        1e-4,
    )
    scale, word = index_k_scale_word(absmax / 448.0, UE8M0)
    q_lo = tl.minimum(tl.maximum(y_lo / scale, -448.0), 448.0).to(
        index_values_ptr.dtype.element_ty
    )
    q_hi = tl.minimum(tl.maximum(rot_hi / scale, -448.0), 448.0).to(
        index_values_ptr.dtype.element_ty
    )
    value_base = page * stride_iv_page + index_row * stride_iv_row
    scale_base = page * stride_is_page + index_row * stride_is_row
    tl.store(index_values_ptr + value_base + c, q_lo)
    tl.store(index_values_ptr + value_base + 64 + c, q_hi)
    tl.store(index_scales_ptr + scale_base, word)
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def hy4_kpool_project_finalize_rope_store(
    grouped_token_keys: torch.Tensor,
    packed_w_t: torch.Tensor,
    intra_group_ape: torch.Tensor,
    key_norm_weight: torch.Tensor,
    group_pos: torch.Tensor,
    inv_freq: torch.Tensor,
    mrope_section: tuple[int, int, int] | list[int],
    write_slots: torch.Tensor,
    index_values: torch.Tensor,
    index_scales: torch.Tensor,
    *,
    eps: float = 1e-5,
) -> None:
    """Fused KPool projection + softmax + RMSNorm + M-RoPE + FP8 scatter with early exit."""
    num_groups, kpool, head_dim = grouped_token_keys.shape
    if num_groups == 0:
        return
    if (
        not grouped_token_keys.is_cuda
        or kpool != 4
        or head_dim != 128
        or inv_freq.shape[0] != 32
        or grouped_token_keys.dtype
        not in (torch.bfloat16, torch.float16, torch.float32)
    ):
        val_gate = torch.matmul(
            grouped_token_keys.reshape(num_groups * kpool, head_dim).float(),
            packed_w_t.float(),
        ).view(num_groups, kpool, 2 * head_dim)
        hy4_kpool_finalize_rope_store(
            val_gate,
            intra_group_ape,
            key_norm_weight,
            group_pos,
            inv_freq,
            mrope_section,
            write_slots,
            index_values,
            index_scales,
            orig_dtype=grouped_token_keys.dtype,
            eps=eps,
        )
        return

    gk_contig = grouped_token_keys.contiguous()
    wt_contig = packed_w_t.contiguous()
    ape_contig = intra_group_ape.contiguous()
    gp_contig = group_pos.contiguous()
    slots_contig = write_slots.contiguous()
    scales_i32 = index_scales.view(torch.int32)
    enable_pdl = pdl_enabled()
    if num_groups <= 64 and grouped_token_keys.dtype == torch.bfloat16:
        p_buf = torch.empty(
            (num_groups, head_dim),
            dtype=grouped_token_keys.dtype,
            device=grouped_token_keys.device,
        )
        counters = _KPOOL_SPLITN_COUNTERS.get(grouped_token_keys.device)
        if counters is None:
            counters = _KPOOL_SPLITN_COUNTERS[grouped_token_keys.device] = torch.zeros(
                64, dtype=torch.int32, device=grouped_token_keys.device
            )
        _hy4_kpool_project_finalize_rope_store_splitn_kernel[(num_groups, 4)](
            gk_contig,
            wt_contig,
            ape_contig,
            key_norm_weight,
            gp_contig,
            inv_freq,
            slots_contig,
            index_values,
            scales_i32,
            p_buf,
            counters,
            num_groups,
            index_values.shape[0],
            gk_contig.stride(0),
            gk_contig.stride(1),
            wt_contig.stride(0),
            ape_contig.stride(0),
            gp_contig.stride(0),
            gp_contig.stride(1),
            index_values.stride(0),
            index_values.stride(1),
            scales_i32.stride(0),
            scales_i32.stride(1),
            eps,
            MROPE_S1=int(mrope_section[1]),
            MROPE_S2=int(mrope_section[2]),
            INDEX_ROWS_PER_PAGE=index_values.shape[1],
            UE8M0=index_k_scale_ue8m0(),
            ORIG_DTYPE=tl.bfloat16,
            BN=32,
            NUM_NTILES=4,
            ENABLE_PDL=enable_pdl,
            num_warps=4,
            **({"launch_pdl": True} if enable_pdl else {}),
        )
        return

    _hy4_kpool_project_finalize_rope_store_kernel[(num_groups,)](
        gk_contig,
        wt_contig,
        ape_contig,
        key_norm_weight,
        gp_contig,
        inv_freq,
        slots_contig,
        index_values,
        scales_i32,
        num_groups,
        index_values.shape[0],
        gk_contig.stride(0),
        gk_contig.stride(1),
        wt_contig.stride(0),
        ape_contig.stride(0),
        gp_contig.stride(0),
        gp_contig.stride(1),
        index_values.stride(0),
        index_values.stride(1),
        scales_i32.stride(0),
        scales_i32.stride(1),
        eps,
        MROPE_S1=int(mrope_section[1]),
        MROPE_S2=int(mrope_section[2]),
        INDEX_ROWS_PER_PAGE=index_values.shape[1],
        UE8M0=index_k_scale_ue8m0(),
        ORIG_DTYPE=_torch_to_triton_dtype(grouped_token_keys.dtype),
        BLOCK_K=128,
        ENABLE_PDL=enable_pdl,
        num_warps=8,
        num_stages=2,
        **({"launch_pdl": True} if enable_pdl else {}),
    )


__all__ = [
    "apply_interleaved_rope_last_dim",
    "compute_hy4_mrope_cos_sin",
    "hy4_pool_group_keys",
    "hy4_quantize_and_store_index_k",
    "hy4_apply_interleaved_rope_inplace",
    "hy4_kv_latent_norm_and_rope_inplace",
    "hy4_kpool_decode_update_and_gather",
    "hy4_kpool_finalize_rope_store",
    "hy4_split_qkv_indexer_norm_rope",
    "hy4_kpool_project_finalize_rope_store",
]
