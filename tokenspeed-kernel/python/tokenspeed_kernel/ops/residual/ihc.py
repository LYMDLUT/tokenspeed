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

"""Identity Hyper-Connections (iHC) kernels for Hy4-flash.

iHC maintains ``hc_mult`` (N=4) residual streams per token without Sinkhorn
stream-mixing:
- ``ihc_pre``: computes FP32 RMS-normalized linear logits from flattened
  streams, derives sigmoid ``pre`` and ``post`` weights, and reduces the N
  streams into one branch input ``[T, D]``.
- ``ihc_post``: adds ``post[..., None] * branch_output[..., None, :]`` into the
  residual stream tensor ``[T, N, D]`` in FP32 and casts back to stream dtype.
- ``ihc_head``: collapses the final N streams into one ``[T, D]`` hidden tensor
  before the shared final RMSNorm.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import pdl_enabled

# Per-device atomic arrival counters for single-kernel ihc_post_and_pre (sized to max T=256).
_IHC_FUSED_COUNTERS: dict[torch.device, torch.Tensor] = {}


@triton.jit(
    do_not_specialize=[
        "num_tokens",
        "stride_sg",
    ]
)
def _ihc_pre_reduce_kernel(
    x_ptr,
    mixes_ptr,
    scale_ptr,
    base_ptr,
    norm_w_ptr,
    out_ptr,
    post_ptr,
    q_out_ptr,
    q_scale_ptr,
    num_tokens,
    stride_xt,
    stride_xn,
    stride_ot,
    stride_mt,
    stride_pt,
    stride_qt,
    stride_sg,
    rms_norm_eps: tl.constexpr,
    out_norm_eps: tl.constexpr,
    hc_magnitude: tl.constexpr,
    hc_eps: tl.constexpr,
    HC_MULT: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_NORM: tl.constexpr,
    QUANT_FP8: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    if token >= num_tokens:
        return

    # Pass 1: unweighted RMS norm scalar over the flattened [HC_MULT * HIDDEN_SIZE] row.
    sum_sq = tl.zeros((), dtype=tl.float32)
    offs = tl.arange(0, BLOCK_D)
    for n in tl.static_range(0, HC_MULT):
        base_x = token * stride_xt + n * stride_xn
        for start in range(0, HIDDEN_SIZE, BLOCK_D):
            cols = start + offs
            mask = cols < HIDDEN_SIZE
            val = tl.load(x_ptr + base_x + cols, mask=mask, other=0.0).to(tl.float32)
            sum_sq += tl.sum(val * val, axis=0)
    inv_rms = tl.rsqrt(sum_sq / (HC_MULT * HIDDEN_SIZE) + rms_norm_eps)

    pre_scale = tl.load(scale_ptr + 0).to(tl.float32)
    post_scale = tl.load(scale_ptr + 1).to(tl.float32)

    n_idx = tl.arange(0, HC_MULT)
    pre_logits = tl.load(mixes_ptr + token * stride_mt + n_idx).to(tl.float32) * inv_rms
    post_logits = (
        tl.load(mixes_ptr + token * stride_mt + HC_MULT + n_idx).to(tl.float32)
        * inv_rms
    )
    pre_b = tl.load(base_ptr + n_idx).to(tl.float32)
    post_b = tl.load(base_ptr + HC_MULT + n_idx).to(tl.float32)

    pre = tl.sigmoid(pre_logits * pre_scale + pre_b) + hc_eps
    post = hc_magnitude * tl.sigmoid(post_logits * post_scale + post_b) + hc_eps
    tl.store(post_ptr + token * stride_pt + n_idx, post)

    # Pass 2: weighted stream reduction -> [HIDDEN_SIZE], optionally fused with RMSNorm + FP8 quant.
    if (HAS_NORM or QUANT_FP8) and BLOCK_D >= HIDDEN_SIZE:
        mask = offs < HIDDEN_SIZE
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for n in tl.static_range(0, HC_MULT):
            w_n = tl.sum(tl.where(n_idx == n, pre, 0.0), axis=0)
            val = tl.load(
                x_ptr + token * stride_xt + n * stride_xn + offs,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            acc += w_n * val
        acc_cast = acc.to(out_ptr.dtype.element_ty)
        if HAS_NORM:
            acc_f = tl.where(mask, acc_cast.to(tl.float32), 0.0)
            out_inv_rms = tl.rsqrt(
                tl.sum(acc_f * acc_f, axis=0) / HIDDEN_SIZE + out_norm_eps
            )
            nw = tl.load(norm_w_ptr + offs, mask=mask, other=0.0).to(
                out_ptr.dtype.element_ty
            )
            acc_cast = (acc_f * out_inv_rms).to(out_ptr.dtype.element_ty) * nw
        tl.store(out_ptr + token * stride_ot + offs, acc_cast, mask=mask)
        if QUANT_FP8:
            q_in = tl.where(mask, acc_cast.to(tl.float32), 0.0)
            g_vals = tl.reshape(q_in, [BLOCK_D // 128, 128])
            g_absmax = tl.maximum(tl.max(tl.abs(g_vals), axis=1), 1e-10)
            g_scale = g_absmax / 448.0
            g_q = tl.minimum(tl.maximum(g_vals / g_scale[:, None], -448.0), 448.0).to(
                tl.float8e4nv
            )
            tl.store(
                q_out_ptr + token * stride_qt + offs,
                tl.reshape(g_q, [BLOCK_D]),
                mask=mask,
            )
            g_idx = tl.arange(0, BLOCK_D // 128)
            tl.store(
                q_scale_ptr + g_idx * stride_sg + token,
                g_scale,
                mask=g_idx < (HIDDEN_SIZE // 128),
            )
    else:
        for start in range(0, HIDDEN_SIZE, BLOCK_D):
            cols = start + offs
            mask = cols < HIDDEN_SIZE
            acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
            for n in tl.static_range(0, HC_MULT):
                w_n = tl.sum(tl.where(n_idx == n, pre, 0.0), axis=0)
                val = tl.load(
                    x_ptr + token * stride_xt + n * stride_xn + cols,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
                acc += w_n * val
            tl.store(
                out_ptr + token * stride_ot + cols,
                acc.to(out_ptr.dtype.element_ty),
                mask=mask,
            )


@triton.jit(do_not_specialize=["num_tokens"])
def _ihc_bf16_mix_splitk_kernel(
    x_ptr,
    fn_ptr,
    partial_ptr,
    num_tokens,
    stride_xt,
    stride_fn_r,
    stride_pt,
    stride_ps,
    TOTAL_K: tl.constexpr,
    M_OUT: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    """Split-K BF16-input FP32-weight projection for iHC mixes + stream sum_sq.

    Reads BF16 ``hidden_streams`` directly (eliminating the FP32 HBM cast copy)
    and accumulates ``M_OUT`` linear projections plus ``sum_sq`` into
    ``partial[token, split, M_OUT + 1]`` for deterministic single-CTA reduction.
    """
    tok = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1).to(tl.int64)
    if tok >= num_tokens:
        return

    k_per_split = tl.cdiv(TOTAL_K, NUM_SPLITS)
    k_start = split * k_per_split
    k_end = tl.minimum(k_start + k_per_split, TOTAL_K)
    offs = tl.arange(0, BLOCK_K)
    m_idx = tl.arange(0, M_OUT)

    acc_mix = tl.zeros((M_OUT,), dtype=tl.float32)
    acc_sq = tl.zeros((), dtype=tl.float32)
    x_base = x_ptr + tok * stride_xt

    for k_off in range(k_start, k_end, BLOCK_K):
        cols = k_off + offs
        mask = cols < k_end
        xv = tl.load(x_base + cols, mask=mask, other=0.0).to(tl.float32)
        acc_sq += tl.sum(xv * xv, axis=0)
        w = tl.load(
            fn_ptr + m_idx[:, None] * stride_fn_r + cols[None, :],
            mask=mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc_mix += tl.sum(w * xv[None, :], axis=1)

    p_base = partial_ptr + tok * stride_pt + split * stride_ps
    tl.store(p_base + m_idx, acc_mix)
    tl.store(p_base + M_OUT, acc_sq)


@triton.jit(
    do_not_specialize=[
        "num_tokens",
        "stride_sg",
    ]
)
def _ihc_pre_finalize_from_partials_kernel(
    x_ptr,
    partial_ptr,
    scale_ptr,
    base_ptr,
    norm_w_ptr,
    out_ptr,
    post_ptr,
    q_out_ptr,
    q_scale_ptr,
    num_tokens,
    stride_xt,
    stride_xn,
    stride_ot,
    stride_pt,
    stride_part_t,
    stride_part_s,
    stride_qt,
    stride_sg,
    rms_norm_eps: tl.constexpr,
    out_norm_eps: tl.constexpr,
    hc_magnitude: tl.constexpr,
    hc_eps: tl.constexpr,
    HC_MULT: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    HAS_NORM: tl.constexpr,
    QUANT_FP8: tl.constexpr,
):
    """Deterministic reduction of split-K mixes/sum_sq + single-pass stream reduce, RMSNorm & FP8 quant."""
    token = tl.program_id(0).to(tl.int64)
    if token >= num_tokens:
        return

    m_out: tl.constexpr = 2 * HC_MULT
    s_idx = tl.arange(0, NUM_SPLITS)
    n_idx = tl.arange(0, HC_MULT)
    p_base = partial_ptr + token * stride_part_t

    sum_sq = tl.sum(tl.load(p_base + s_idx * stride_part_s + m_out), axis=0)
    inv_rms = tl.rsqrt(sum_sq / (HC_MULT * HIDDEN_SIZE) + rms_norm_eps)

    pre_logits = (
        tl.sum(
            tl.load(p_base + s_idx[:, None] * stride_part_s + n_idx[None, :]),
            axis=0,
        )
        * inv_rms
    )
    post_logits = (
        tl.sum(
            tl.load(
                p_base + s_idx[:, None] * stride_part_s + (HC_MULT + n_idx)[None, :]
            ),
            axis=0,
        )
        * inv_rms
    )
    pre_scale = tl.load(scale_ptr + 0).to(tl.float32)
    post_scale = tl.load(scale_ptr + 1).to(tl.float32)
    pre_b = tl.load(base_ptr + n_idx).to(tl.float32)
    post_b = tl.load(base_ptr + HC_MULT + n_idx).to(tl.float32)

    pre = tl.sigmoid(pre_logits * pre_scale + pre_b) + hc_eps
    post = hc_magnitude * tl.sigmoid(post_logits * post_scale + post_b) + hc_eps
    tl.store(post_ptr + token * stride_pt + n_idx, post)

    offs = tl.arange(0, BLOCK_D)
    if BLOCK_D >= HIDDEN_SIZE:
        mask = offs < HIDDEN_SIZE
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for n in tl.static_range(0, HC_MULT):
            w_n = tl.sum(tl.where(n_idx == n, pre, 0.0), axis=0)
            val = tl.load(
                x_ptr + token * stride_xt + n * stride_xn + offs,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            acc += w_n * val
        acc_cast = acc.to(out_ptr.dtype.element_ty)
        if HAS_NORM:
            acc_f = tl.where(mask, acc_cast.to(tl.float32), 0.0)
            out_inv_rms = tl.rsqrt(
                tl.sum(acc_f * acc_f, axis=0) / HIDDEN_SIZE + out_norm_eps
            )
            nw = tl.load(norm_w_ptr + offs, mask=mask, other=0.0).to(
                out_ptr.dtype.element_ty
            )
            acc_cast = (acc_f * out_inv_rms).to(out_ptr.dtype.element_ty) * nw
        tl.store(out_ptr + token * stride_ot + offs, acc_cast, mask=mask)
        if QUANT_FP8:
            q_in = tl.where(mask, acc_cast.to(tl.float32), 0.0)
            g_vals = tl.reshape(q_in, [BLOCK_D // 128, 128])
            g_absmax = tl.maximum(tl.max(tl.abs(g_vals), axis=1), 1e-10)
            g_scale = g_absmax / 448.0
            g_q = tl.minimum(tl.maximum(g_vals / g_scale[:, None], -448.0), 448.0).to(
                tl.float8e4nv
            )
            tl.store(
                q_out_ptr + token * stride_qt + offs,
                tl.reshape(g_q, [BLOCK_D]),
                mask=mask,
            )
            g_idx = tl.arange(0, BLOCK_D // 128)
            tl.store(
                q_scale_ptr + g_idx * stride_sg + token,
                g_scale,
                mask=g_idx < (HIDDEN_SIZE // 128),
            )
    else:
        for start in range(0, HIDDEN_SIZE, BLOCK_D):
            cols = start + offs
            mask = cols < HIDDEN_SIZE
            acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
            for n in tl.static_range(0, HC_MULT):
                w_n = tl.sum(tl.where(n_idx == n, pre, 0.0), axis=0)
                val = tl.load(
                    x_ptr + token * stride_xt + n * stride_xn + cols,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32)
                acc += w_n * val
            tl.store(
                out_ptr + token * stride_ot + cols,
                acc.to(out_ptr.dtype.element_ty),
                mask=mask,
            )


@triton.jit(
    do_not_specialize=[
        "num_tokens",
        "stride_sg",
    ]
)
def _ihc_post_and_pre_fused_kernel(
    residual_ptr,
    branch_ptr,
    post_norm_w_ptr,
    post_w_ptr,
    out_res_ptr,
    next_fn_ptr,
    partial_ptr,
    counter_ptr,
    scale_ptr,
    base_ptr,
    pre_norm_w_ptr,
    out_ptr,
    next_post_ptr,
    q_out_ptr,
    q_scale_ptr,
    num_tokens,
    stride_rt,
    stride_rn,
    stride_bt,
    stride_pt,
    stride_ort,
    stride_orn,
    stride_fn_r,
    stride_part_t,
    stride_part_s,
    stride_ot,
    stride_npt,
    stride_qt,
    stride_sg,
    post_norm_eps: tl.constexpr,
    rms_norm_eps: tl.constexpr,
    out_norm_eps: tl.constexpr,
    hc_magnitude: tl.constexpr,
    hc_eps: tl.constexpr,
    HC_MULT: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    M_OUT: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    PAD_SPLITS: tl.constexpr,
    HAS_POST_NORM: tl.constexpr,
    HAS_PRE_NORM: tl.constexpr,
    QUANT_FP8: tl.constexpr,
    ENABLE_PDL: tl.constexpr = False,
):
    """Single-kernel iHC post update + split-K mix + last-CTA finalize (+ norm/FP8)."""
    tok = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1).to(tl.int64)
    if tok >= num_tokens:
        return
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()

    inv_rms_post = 1.0
    if HAS_POST_NORM:
        d_full = tl.arange(0, BLOCK_D)
        d_mask = d_full < HIDDEN_SIZE
        br_full = tl.load(
            branch_ptr + tok * stride_bt + d_full, mask=d_mask, other=0.0
        ).to(tl.float32)
        inv_rms_post = tl.rsqrt(
            tl.sum(br_full * br_full, axis=0) / HIDDEN_SIZE + post_norm_eps
        )

    d_per_split = tl.cdiv(HIDDEN_SIZE, NUM_SPLITS)
    d_start = split * d_per_split
    d_end = tl.minimum(d_start + d_per_split, HIDDEN_SIZE)
    offs = tl.arange(0, BLOCK_K)
    m_idx = tl.arange(0, M_OUT)
    acc_mix = tl.zeros((M_OUT,), dtype=tl.float32)
    acc_sq = tl.zeros((), dtype=tl.float32)

    for d_off in range(d_start, d_end, BLOCK_K):
        cols = d_off + offs
        mask = cols < d_end
        br_s = tl.load(branch_ptr + tok * stride_bt + cols, mask=mask, other=0.0).to(
            tl.float32
        )
        if HAS_POST_NORM:
            nw_s = tl.load(post_norm_w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            normed_s = (
                ((br_s * inv_rms_post) * nw_s)
                .to(out_res_ptr.dtype.element_ty)
                .to(tl.float32)
            )
        else:
            normed_s = br_s

        for n in tl.static_range(0, HC_MULT):
            p = tl.load(post_w_ptr + tok * stride_pt + n).to(tl.float32)
            res = tl.load(
                residual_ptr + tok * stride_rt + n * stride_rn + cols,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            upd = (res + p * normed_s).to(out_res_ptr.dtype.element_ty)
            tl.store(
                out_res_ptr + tok * stride_ort + n * stride_orn + cols,
                upd,
                mask=mask,
            )
            uf = tl.where(mask, upd.to(tl.float32), 0.0)
            acc_sq += tl.sum(uf * uf, axis=0)
            w = tl.load(
                next_fn_ptr
                + m_idx[:, None] * stride_fn_r
                + (n * HIDDEN_SIZE + cols)[None, :],
                mask=mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc_mix += tl.sum(w * uf[None, :], axis=1)

    n_idx = tl.arange(0, HC_MULT)
    pre_scale = tl.load(scale_ptr + 0).to(tl.float32)
    post_scale = tl.load(scale_ptr + 1).to(tl.float32)

    if NUM_SPLITS == 1:
        inv_rms = tl.rsqrt(acc_sq / (HC_MULT * HIDDEN_SIZE) + rms_norm_eps)
        pre_logits = (
            tl.sum(
                tl.where(m_idx[:, None] == n_idx[None, :], acc_mix[:, None], 0.0),
                axis=0,
            )
            * inv_rms
        )
        post_logits = (
            tl.sum(
                tl.where(
                    m_idx[:, None] == (HC_MULT + n_idx)[None, :], acc_mix[:, None], 0.0
                ),
                axis=0,
            )
            * inv_rms
        )
        pre_b = tl.load(base_ptr + n_idx).to(tl.float32)
        post_b = tl.load(base_ptr + HC_MULT + n_idx).to(tl.float32)
        pre = tl.sigmoid(pre_logits * pre_scale + pre_b) + hc_eps
        next_post = (
            hc_magnitude * tl.sigmoid(post_logits * post_scale + post_b) + hc_eps
        )
        tl.store(next_post_ptr + tok * stride_npt + n_idx, next_post)

        d_full = tl.arange(0, BLOCK_D)
        d_mask = d_full < HIDDEN_SIZE
        acc_out = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for n in tl.static_range(0, HC_MULT):
            w_n = tl.sum(tl.where(n_idx == n, pre, 0.0), axis=0)
            val = tl.load(
                out_res_ptr + tok * stride_ort + n * stride_orn + d_full,
                mask=d_mask,
                other=0.0,
            ).to(tl.float32)
            acc_out += w_n * val
    else:
        p_base = partial_ptr + tok * stride_part_t + split * stride_part_s
        tl.store(p_base + m_idx, acc_mix)
        tl.store(p_base + M_OUT, acc_sq)
        prev = tl.atomic_add(counter_ptr + tok, 1, sem="acq_rel", scope="gpu")
        if prev != NUM_SPLITS - 1:
            return
        tl.store(counter_ptr + tok, 0)

        s_idx = tl.arange(0, PAD_SPLITS)
        s_mask = s_idx < NUM_SPLITS
        p_tok = partial_ptr + tok * stride_part_t + s_idx * stride_part_s
        sum_sq = tl.sum(
            tl.load(p_tok + M_OUT, mask=s_mask, other=0.0, cache_modifier=".cg"),
            axis=0,
        )
        inv_rms = tl.rsqrt(sum_sq / (HC_MULT * HIDDEN_SIZE) + rms_norm_eps)
        post_sum = tl.sum(
            tl.load(
                partial_ptr
                + tok * stride_part_t
                + s_idx[:, None] * stride_part_s
                + (HC_MULT + n_idx)[None, :],
                mask=s_mask[:, None],
                other=0.0,
                cache_modifier=".cg",
            ),
            axis=0,
        )
        post_b = tl.load(base_ptr + HC_MULT + n_idx).to(tl.float32)
        next_post = (
            hc_magnitude * tl.sigmoid(post_sum * inv_rms * post_scale + post_b) + hc_eps
        )
        tl.store(next_post_ptr + tok * stride_npt + n_idx, next_post)

        d_full = tl.arange(0, BLOCK_D)
        d_mask = d_full < HIDDEN_SIZE
        acc_out = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for n in tl.static_range(0, HC_MULT):
            pre_n = (
                tl.sum(
                    tl.load(p_tok + n, mask=s_mask, other=0.0, cache_modifier=".cg"),
                    axis=0,
                )
                * inv_rms
            )
            w_n = (
                tl.sigmoid(pre_n * pre_scale + tl.load(base_ptr + n).to(tl.float32))
                + hc_eps
            )
            val = tl.load(
                out_res_ptr + tok * stride_ort + n * stride_orn + d_full,
                mask=d_mask,
                other=0.0,
                cache_modifier=".cg",
            ).to(tl.float32)
            acc_out += w_n * val

    acc_cast = acc_out.to(out_ptr.dtype.element_ty)
    if HAS_PRE_NORM:
        acc_f = tl.where(d_mask, acc_cast.to(tl.float32), 0.0)
        out_inv = tl.rsqrt(tl.sum(acc_f * acc_f, axis=0) / HIDDEN_SIZE + out_norm_eps)
        nw = tl.load(pre_norm_w_ptr + d_full, mask=d_mask, other=0.0).to(
            out_ptr.dtype.element_ty
        )
        acc_cast = (acc_f * out_inv).to(out_ptr.dtype.element_ty) * nw
    tl.store(out_ptr + tok * stride_ot + d_full, acc_cast, mask=d_mask)
    if QUANT_FP8:
        q_in = tl.where(d_mask, acc_cast.to(tl.float32), 0.0)
        g_vals = tl.reshape(q_in, [BLOCK_D // 128, 128])
        g_absmax = tl.maximum(tl.max(tl.abs(g_vals), axis=1), 1e-10)
        g_scale = g_absmax / 448.0
        g_q = tl.minimum(tl.maximum(g_vals / g_scale[:, None], -448.0), 448.0).to(
            tl.float8e4nv
        )
        tl.store(
            q_out_ptr + tok * stride_qt + d_full,
            tl.reshape(g_q, [BLOCK_D]),
            mask=d_mask,
        )
        g_idx = tl.arange(0, BLOCK_D // 128)
        tl.store(
            q_scale_ptr + g_idx * stride_sg + tok,
            g_scale,
            mask=g_idx < (HIDDEN_SIZE // 128),
        )
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit(do_not_specialize=["num_tokens"])
def _ihc_post_fused_norm_kernel(
    residual_ptr,
    branch_ptr,
    norm_w_ptr,
    post_ptr,
    out_ptr,
    num_tokens,
    stride_rt,
    stride_rn,
    stride_bt,
    stride_pt,
    stride_ot,
    stride_on,
    norm_eps: tl.constexpr,
    HC_MULT: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fused RMSNorm(branch) + residual update across all HC_MULT streams in one pass."""
    token = tl.program_id(0).to(tl.int64)
    if token >= num_tokens:
        return
    offs = tl.arange(0, BLOCK_D)
    mask = offs < HIDDEN_SIZE
    br = tl.load(branch_ptr + token * stride_bt + offs, mask=mask, other=0.0).to(
        tl.float32
    )
    inv_rms = tl.rsqrt(tl.sum(br * br, axis=0) / HIDDEN_SIZE + norm_eps)
    nw = tl.load(norm_w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    normed = ((br * inv_rms) * nw).to(out_ptr.dtype.element_ty).to(tl.float32)
    for n in tl.static_range(0, HC_MULT):
        p = tl.load(post_ptr + token * stride_pt + n).to(tl.float32)
        res = tl.load(
            residual_ptr + token * stride_rt + n * stride_rn + offs,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        updated = res + p * normed
        tl.store(
            out_ptr + token * stride_ot + n * stride_on + offs,
            updated.to(out_ptr.dtype.element_ty),
            mask=mask,
        )


@triton.jit(do_not_specialize=["num_tokens"])
def _ihc_post_kernel(
    residual_ptr,
    branch_ptr,
    post_ptr,
    out_ptr,
    num_tokens,
    stride_rt,
    stride_rn,
    stride_bt,
    stride_pt,
    stride_ot,
    stride_on,
    HC_MULT: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1).to(tl.int64)
    if token >= num_tokens:
        return
    cols = block * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = cols < HIDDEN_SIZE
    branch = tl.load(branch_ptr + token * stride_bt + cols, mask=mask, other=0.0).to(
        tl.float32
    )
    for n in tl.static_range(0, HC_MULT):
        p = tl.load(post_ptr + token * stride_pt + n).to(tl.float32)
        res = tl.load(
            residual_ptr + token * stride_rt + n * stride_rn + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        updated = res + p * branch
        tl.store(
            out_ptr + token * stride_ot + n * stride_on + cols,
            updated.to(out_ptr.dtype.element_ty),
            mask=mask,
        )


@triton.jit(do_not_specialize=["num_tokens"])
def _ihc_head_reduce_kernel(
    x_ptr,
    mixes_ptr,
    scale_ptr,
    base_ptr,
    out_ptr,
    num_tokens,
    stride_xt,
    stride_xn,
    stride_ot,
    stride_mt,
    rms_norm_eps: tl.constexpr,
    hc_eps: tl.constexpr,
    HC_MULT: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    if token >= num_tokens:
        return

    sum_sq = tl.zeros((), dtype=tl.float32)
    offs = tl.arange(0, BLOCK_D)
    for n in tl.static_range(0, HC_MULT):
        base_x = token * stride_xt + n * stride_xn
        for start in range(0, HIDDEN_SIZE, BLOCK_D):
            cols = start + offs
            mask = cols < HIDDEN_SIZE
            val = tl.load(x_ptr + base_x + cols, mask=mask, other=0.0).to(tl.float32)
            sum_sq += tl.sum(val * val, axis=0)
    inv_rms = tl.rsqrt(sum_sq / (HC_MULT * HIDDEN_SIZE) + rms_norm_eps)

    head_scale = tl.load(scale_ptr + 0).to(tl.float32)
    n_idx = tl.arange(0, HC_MULT)
    logits = tl.load(mixes_ptr + token * stride_mt + n_idx).to(tl.float32) * inv_rms
    base = tl.load(base_ptr + n_idx).to(tl.float32)
    pre = tl.sigmoid(logits * head_scale + base) + hc_eps

    for start in range(0, HIDDEN_SIZE, BLOCK_D):
        cols = start + offs
        mask = cols < HIDDEN_SIZE
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for n in tl.static_range(0, HC_MULT):
            w_n = tl.sum(tl.where(n_idx == n, pre, 0.0), axis=0)
            val = tl.load(
                x_ptr + token * stride_xt + n * stride_xn + cols,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            acc += w_n * val
        tl.store(
            out_ptr + token * stride_ot + cols,
            acc.to(out_ptr.dtype.element_ty),
            mask=mask,
        )


def _ihc_pre_torch(
    hidden_streams: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    rms_norm_eps: float,
    hc_magnitude: float,
    hc_eps: float,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact FP32 reference matching HYV4FlashHyperConnection (+ optional RMSNorm)."""
    orig_dtype = hidden_streams.dtype
    num_tokens, hc_mult, hidden_size = hidden_streams.shape
    flat = hidden_streams.reshape(num_tokens, hc_mult * hidden_size).float()
    inv_rms = torch.rsqrt(flat.pow(2).mean(dim=-1, keepdim=True) + rms_norm_eps)
    mixes = F.linear(flat, fn.float()) * inv_rms
    pre_logits, post_logits = mixes.split(hc_mult, dim=-1)
    pre_b, post_b = base.float().split(hc_mult, dim=-1)
    pre_scale, post_scale = scale.float().unbind(0)
    pre = torch.sigmoid(pre_logits * pre_scale + pre_b) + hc_eps
    post = hc_magnitude * torch.sigmoid(post_logits * post_scale + post_b) + hc_eps
    branch_input = (
        (pre.unsqueeze(-1) * hidden_streams.float()).sum(dim=1).to(orig_dtype)
    )
    if norm_weight is not None:
        eps = rms_norm_eps if norm_eps is None else norm_eps
        bf = branch_input.float()
        normed = (bf * torch.rsqrt(bf.pow(2).mean(dim=-1, keepdim=True) + eps)).to(
            orig_dtype
        )
        branch_input = normed * norm_weight.to(orig_dtype)
    return post, branch_input


def ihc_pre(
    hidden_streams: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    rms_norm_eps: float = 1e-5,
    hc_magnitude: float = 2.0,
    hc_eps: float = 1e-6,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float | None = None,
    return_fp8_mn_major: bool = False,
):
    """Compute iHC post weights (FP32) and reduced (optionally RMS-normed & FP8-quantized) input."""
    if hidden_streams.ndim != 3:
        raise ValueError(
            f"ihc_pre expects [T, hc_mult, D], got {tuple(hidden_streams.shape)}"
        )
    num_tokens, hc_mult, hidden_size = hidden_streams.shape
    if num_tokens == 0:
        empty_post = torch.empty(
            (0, hc_mult), dtype=torch.float32, device=hidden_streams.device
        )
        empty_out = torch.empty(
            (0, hidden_size),
            dtype=hidden_streams.dtype,
            device=hidden_streams.device,
        )
        if return_fp8_mn_major:
            return (
                empty_post,
                empty_out,
                torch.empty(
                    (0, hidden_size),
                    dtype=torch.float8_e4m3fn,
                    device=hidden_streams.device,
                ),
                torch.empty(
                    (0, hidden_size // 128),
                    dtype=torch.float32,
                    device=hidden_streams.device,
                ),
            )
        return empty_post, empty_out
    if (
        not hidden_streams.is_cuda
        or (hc_mult & (hc_mult - 1)) != 0
        or hidden_streams.stride(-1) != 1
    ):
        post, out = _ihc_pre_torch(
            hidden_streams,
            fn,
            scale,
            base,
            rms_norm_eps=rms_norm_eps,
            hc_magnitude=hc_magnitude,
            hc_eps=hc_eps,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
        )
        if return_fp8_mn_major:
            num_groups = hidden_size // 128
            aligned_t = max(((num_tokens + 3) // 4) * 4, 4)
            g = out.float().view(num_tokens, num_groups, 128)
            sc = g.abs().amax(dim=-1).clamp_min(1e-10) / 448.0
            q = (g / sc.unsqueeze(-1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            sc_buf = torch.empty(
                (num_groups, aligned_t), dtype=torch.float32, device=out.device
            )
            sc_buf[:, :num_tokens].copy_(sc.t())
            return (
                post,
                out,
                q.view(num_tokens, hidden_size),
                sc_buf[:, :num_tokens].t(),
            )
        return post, out

    out = torch.empty(
        (num_tokens, hidden_size),
        dtype=hidden_streams.dtype,
        device=hidden_streams.device,
    )
    post = torch.empty(
        (num_tokens, hc_mult),
        dtype=torch.float32,
        device=hidden_streams.device,
    )
    eff_norm_eps = float(rms_norm_eps if norm_eps is None else norm_eps)
    can_fuse_norm = norm_weight is not None and hidden_size <= 4096
    can_fuse_fp8 = (
        return_fp8_mn_major
        and hidden_size <= 4096
        and hidden_size % 128 == 0
        and hidden_streams.dtype in (torch.bfloat16, torch.float16)
    )
    norm_w_arg = norm_weight if can_fuse_norm else out
    if can_fuse_fp8:
        aligned_t = max(((num_tokens + 3) // 4) * 4, 4)
        num_groups = hidden_size // 128
        q_out = torch.empty(
            (num_tokens, hidden_size),
            dtype=torch.float8_e4m3fn,
            device=hidden_streams.device,
        )
        q_scale_storage = torch.empty(
            (num_groups, aligned_t),
            dtype=torch.float32,
            device=hidden_streams.device,
        )
        q_out_arg = q_out
        q_scale_arg = q_scale_storage
        stride_qt = q_out.stride(0)
        stride_sg = aligned_t
    else:
        q_out = None
        q_scale_storage = None
        q_out_arg = out
        q_scale_arg = post
        stride_qt = 0
        stride_sg = 0

    # Decode/verify batches (T <= 256): split-K BF16-input GEMV eliminates the
    # FP32 cast copy of hidden_streams, cuBLAS splitKreduce, and the second HBM read.
    if (
        num_tokens <= 256
        and hidden_streams.is_contiguous()
        and fn.is_contiguous()
        and hidden_size <= 4096
    ):
        num_splits = 16
        partials = torch.empty(
            (num_tokens, num_splits, 2 * hc_mult + 1),
            dtype=torch.float32,
            device=hidden_streams.device,
        )
        _ihc_bf16_mix_splitk_kernel[(num_tokens, num_splits)](
            hidden_streams,
            fn,
            partials,
            num_tokens,
            hc_mult * hidden_size,
            fn.stride(0),
            partials.stride(0),
            partials.stride(1),
            TOTAL_K=hc_mult * hidden_size,
            M_OUT=2 * hc_mult,
            BLOCK_K=256,
            NUM_SPLITS=num_splits,
            num_warps=4,
        )
        block_d = triton.next_power_of_2(hidden_size)
        _ihc_pre_finalize_from_partials_kernel[(num_tokens,)](
            hidden_streams,
            partials,
            scale,
            base,
            norm_w_arg,
            out,
            post,
            q_out_arg,
            q_scale_arg,
            num_tokens,
            hidden_streams.stride(0),
            hidden_streams.stride(1),
            out.stride(0),
            post.stride(0),
            partials.stride(0),
            partials.stride(1),
            stride_qt,
            stride_sg,
            rms_norm_eps=float(rms_norm_eps),
            out_norm_eps=eff_norm_eps,
            hc_magnitude=float(hc_magnitude),
            hc_eps=float(hc_eps),
            HC_MULT=hc_mult,
            HIDDEN_SIZE=hidden_size,
            BLOCK_D=block_d,
            NUM_SPLITS=num_splits,
            HAS_NORM=can_fuse_norm,
            QUANT_FP8=can_fuse_fp8,
            num_warps=8 if block_d >= 2048 else 4,
        )
        if can_fuse_fp8:
            assert q_out is not None and q_scale_storage is not None
            return post, out, q_out, q_scale_storage[:, :num_tokens].t()
        return post, out

    flat = hidden_streams.reshape(num_tokens, hc_mult * hidden_size).float()
    mixes = F.linear(flat, fn.float())
    block_d = (
        triton.next_power_of_2(hidden_size)
        if (can_fuse_norm or can_fuse_fp8)
        else min(1024, triton.next_power_of_2(hidden_size))
    )
    _ihc_pre_reduce_kernel[(num_tokens,)](
        hidden_streams,
        mixes,
        scale,
        base,
        norm_w_arg,
        out,
        post,
        q_out_arg,
        q_scale_arg,
        num_tokens,
        hidden_streams.stride(0),
        hidden_streams.stride(1),
        out.stride(0),
        mixes.stride(0),
        post.stride(0),
        stride_qt,
        stride_sg,
        rms_norm_eps=float(rms_norm_eps),
        out_norm_eps=eff_norm_eps,
        hc_magnitude=float(hc_magnitude),
        hc_eps=float(hc_eps),
        HC_MULT=hc_mult,
        HIDDEN_SIZE=hidden_size,
        BLOCK_D=block_d,
        HAS_NORM=can_fuse_norm,
        QUANT_FP8=can_fuse_fp8,
        num_warps=8 if block_d >= 2048 else 4,
    )
    if norm_weight is not None and not can_fuse_norm:
        bf = out.float()
        out = (
            bf * torch.rsqrt(bf.pow(2).mean(dim=-1, keepdim=True) + eff_norm_eps)
        ).to(out.dtype) * norm_weight.to(out.dtype)
    if return_fp8_mn_major:
        if can_fuse_fp8:
            assert q_out is not None and q_scale_storage is not None
            return post, out, q_out, q_scale_storage[:, :num_tokens].t()
        num_groups = hidden_size // 128
        aligned_t = max(((num_tokens + 3) // 4) * 4, 4)
        g = out.float().view(num_tokens, num_groups, 128)
        sc = g.abs().amax(dim=-1).clamp_min(1e-10) / 448.0
        q = (g / sc.unsqueeze(-1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        sc_buf = torch.empty(
            (num_groups, aligned_t), dtype=torch.float32, device=out.device
        )
        sc_buf[:, :num_tokens].copy_(sc.t())
        return post, out, q.view(num_tokens, hidden_size), sc_buf[:, :num_tokens].t()
    return post, out


def ihc_post(
    residual: torch.Tensor,
    branch_output: torch.Tensor,
    post_weights: torch.Tensor,
    *,
    inplace: bool = False,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-5,
) -> torch.Tensor:
    """Update residual streams ``(post[..., None] * norm(branch)[..., None, :] + residual)``."""
    if residual.ndim != 3:
        raise ValueError(
            f"ihc_post expects [T, hc_mult, D], got {tuple(residual.shape)}"
        )
    num_tokens, hc_mult, hidden_size = residual.shape
    if num_tokens == 0:
        return residual if inplace else residual.clone()
    if (
        not residual.is_cuda
        or residual.stride(-1) != 1
        or branch_output.stride(-1) != 1
        or (norm_weight is not None and hidden_size > 4096)
    ):
        br = branch_output
        if norm_weight is not None:
            bf = br.float()
            br = (
                bf
                * torch.rsqrt(bf.pow(2).mean(dim=-1, keepdim=True) + float(norm_eps))
                * norm_weight.float()
            ).to(br.dtype)
        updated = (
            post_weights.float().unsqueeze(-1) * br.float().unsqueeze(1)
            + residual.float()
        ).to(residual.dtype)
        if inplace:
            residual.copy_(updated)
            return residual
        return updated

    out = residual if inplace else torch.empty_like(residual)
    if norm_weight is not None:
        block_d = triton.next_power_of_2(hidden_size)
        _ihc_post_fused_norm_kernel[(num_tokens,)](
            residual,
            branch_output,
            norm_weight,
            post_weights,
            out,
            num_tokens,
            residual.stride(0),
            residual.stride(1),
            branch_output.stride(0),
            post_weights.stride(0),
            out.stride(0),
            out.stride(1),
            norm_eps=float(norm_eps),
            HC_MULT=hc_mult,
            HIDDEN_SIZE=hidden_size,
            BLOCK_D=block_d,
            num_warps=8 if block_d >= 2048 else 4,
        )
        return out
    block_d = min(512, triton.next_power_of_2(hidden_size))
    grid = (num_tokens, triton.cdiv(hidden_size, block_d))
    _ihc_post_kernel[grid](
        residual,
        branch_output,
        post_weights,
        out,
        num_tokens,
        residual.stride(0),
        residual.stride(1),
        branch_output.stride(0),
        post_weights.stride(0),
        out.stride(0),
        out.stride(1),
        HC_MULT=hc_mult,
        HIDDEN_SIZE=hidden_size,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return out


def ihc_head(
    hidden_streams: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    rms_norm_eps: float = 1e-5,
    hc_eps: float = 1e-6,
) -> torch.Tensor:
    """Collapse ``[T, hc_mult, D]`` streams into ``[T, D]`` before final RMSNorm."""
    if hidden_streams.ndim != 3:
        raise ValueError(
            f"ihc_head expects [T, hc_mult, D], got {tuple(hidden_streams.shape)}"
        )
    num_tokens, hc_mult, hidden_size = hidden_streams.shape
    if num_tokens == 0:
        return torch.empty(
            (0, hidden_size),
            dtype=hidden_streams.dtype,
            device=hidden_streams.device,
        )
    if (
        not hidden_streams.is_cuda
        or (hc_mult & (hc_mult - 1)) != 0
        or hidden_streams.stride(-1) != 1
    ):
        flat = hidden_streams.reshape(num_tokens, hc_mult * hidden_size).float()
        inv_rms = torch.rsqrt(flat.pow(2).mean(dim=-1, keepdim=True) + rms_norm_eps)
        mixes = F.linear(flat, hc_fn.float()) * inv_rms
        pre = torch.sigmoid(mixes * hc_scale.float() + hc_base.float()) + hc_eps
        return (
            (pre.unsqueeze(-1) * hidden_streams.float())
            .sum(dim=1)
            .to(hidden_streams.dtype)
        )

    flat = hidden_streams.reshape(num_tokens, hc_mult * hidden_size).float()
    mixes = F.linear(flat, hc_fn.float())
    out = torch.empty(
        (num_tokens, hidden_size),
        dtype=hidden_streams.dtype,
        device=hidden_streams.device,
    )
    block_d = min(1024, triton.next_power_of_2(hidden_size))
    _ihc_head_reduce_kernel[(num_tokens,)](
        hidden_streams,
        mixes,
        hc_scale,
        hc_base,
        out,
        num_tokens,
        hidden_streams.stride(0),
        hidden_streams.stride(1),
        out.stride(0),
        mixes.stride(0),
        rms_norm_eps=float(rms_norm_eps),
        hc_eps=float(hc_eps),
        HC_MULT=hc_mult,
        HIDDEN_SIZE=hidden_size,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return out


def ihc_post_and_pre(
    residual: torch.Tensor,
    branch_output: torch.Tensor,
    post_weights: torch.Tensor,
    next_fn: torch.Tensor,
    next_scale: torch.Tensor,
    next_base: torch.Tensor,
    *,
    inplace: bool = False,
    post_norm_weight: torch.Tensor | None = None,
    post_norm_eps: float = 1e-5,
    rms_norm_eps: float = 1e-5,
    hc_magnitude: float = 2.0,
    hc_eps: float = 1e-6,
    pre_norm_weight: torch.Tensor | None = None,
    pre_norm_eps: float | None = None,
    return_fp8_mn_major: bool = False,
):
    """Fused iHC post update + next sublayer iHC pre collapse (2 kernels instead of 3).

    Returns ``(updated_streams, next_post, next_branch_in)`` or, when
    ``return_fp8_mn_major=True``, ``(updated_streams, next_post, next_branch_in,
    q_fp8, q_scale)``.
    """
    num_tokens, hc_mult, hidden_size = residual.shape
    eff_pre_eps = float(rms_norm_eps if pre_norm_eps is None else pre_norm_eps)
    can_fuse_pre_fp8 = (
        return_fp8_mn_major
        and hidden_size <= 4096
        and hidden_size % 128 == 0
        and residual.dtype in (torch.bfloat16, torch.float16)
    )
    if (
        num_tokens == 0
        or num_tokens > 256
        or not residual.is_cuda
        or (hc_mult & (hc_mult - 1)) != 0
        or hidden_size > 4096
        or not residual.is_contiguous()
        or branch_output.stride(-1) != 1
        or post_weights.stride(-1) != 1
        or not next_fn.is_contiguous()
        or (return_fp8_mn_major and not can_fuse_pre_fp8)
    ):
        updated = ihc_post(
            residual,
            branch_output,
            post_weights,
            inplace=inplace,
            norm_weight=post_norm_weight,
            norm_eps=post_norm_eps,
        )
        pre_res = ihc_pre(
            updated,
            next_fn,
            next_scale,
            next_base,
            rms_norm_eps=rms_norm_eps,
            hc_magnitude=hc_magnitude,
            hc_eps=hc_eps,
            norm_weight=pre_norm_weight,
            norm_eps=pre_norm_eps,
            return_fp8_mn_major=return_fp8_mn_major,
        )
        return (updated, *pre_res)

    out_res = residual if inplace else torch.empty_like(residual)
    out = torch.empty(
        (num_tokens, hidden_size), dtype=residual.dtype, device=residual.device
    )
    next_post = torch.empty(
        (num_tokens, hc_mult), dtype=torch.float32, device=residual.device
    )
    # Adaptive split-K schedule by batch size:
    # - small T (<=8): 12 splits x 256 cols increases SM occupancy;
    # - medium T (<=128): 6 splits x 512 cols cuts launch/atomic traffic;
    # - large T (>128): 1 split x 1024 cols skips atomics/partials entirely.
    if num_tokens <= 8:
        num_splits, block_k = 12, 256
    elif num_tokens <= 128:
        num_splits, block_k = 6, 512
    else:
        num_splits, block_k = 1, 1024
    pad_splits = triton.next_power_of_2(num_splits)
    counters = _IHC_FUSED_COUNTERS.get(residual.device)
    if counters is None:
        counters = _IHC_FUSED_COUNTERS[residual.device] = torch.zeros(
            256, dtype=torch.int32, device=residual.device
        )
    if num_splits > 1:
        partials = torch.empty(
            (num_tokens, num_splits, 2 * hc_mult + 1),
            dtype=torch.float32,
            device=residual.device,
        )
    else:
        partials = next_post
    block_d = triton.next_power_of_2(hidden_size)
    has_post_norm = post_norm_weight is not None
    post_nw_arg = post_norm_weight if has_post_norm else out
    can_fuse_pre_norm = pre_norm_weight is not None
    pre_nw_arg = pre_norm_weight if can_fuse_pre_norm else out
    if can_fuse_pre_fp8:
        aligned_t = max(((num_tokens + 3) // 4) * 4, 4)
        num_groups = hidden_size // 128
        q_out = torch.empty(
            (num_tokens, hidden_size),
            dtype=torch.float8_e4m3fn,
            device=residual.device,
        )
        q_scale_storage = torch.empty(
            (num_groups, aligned_t),
            dtype=torch.float32,
            device=residual.device,
        )
        q_out_arg = q_out
        q_scale_arg = q_scale_storage
        stride_qt = q_out.stride(0)
        stride_sg = aligned_t
    else:
        q_out = None
        q_scale_storage = None
        q_out_arg = out
        q_scale_arg = next_post
        stride_qt = 0
        stride_sg = 0

    enable_pdl = pdl_enabled()
    _ihc_post_and_pre_fused_kernel[(num_tokens, num_splits)](
        residual,
        branch_output,
        post_nw_arg,
        post_weights,
        out_res,
        next_fn,
        partials,
        counters,
        next_scale,
        next_base,
        pre_nw_arg,
        out,
        next_post,
        q_out_arg,
        q_scale_arg,
        num_tokens,
        residual.stride(0),
        residual.stride(1),
        branch_output.stride(0),
        post_weights.stride(0),
        out_res.stride(0),
        out_res.stride(1),
        next_fn.stride(0),
        partials.stride(0) if num_splits > 1 else 0,
        partials.stride(1) if num_splits > 1 else 0,
        out.stride(0),
        next_post.stride(0),
        stride_qt,
        stride_sg,
        post_norm_eps=float(post_norm_eps),
        rms_norm_eps=float(rms_norm_eps),
        out_norm_eps=eff_pre_eps,
        hc_magnitude=float(hc_magnitude),
        hc_eps=float(hc_eps),
        HC_MULT=hc_mult,
        HIDDEN_SIZE=hidden_size,
        M_OUT=2 * hc_mult,
        BLOCK_D=block_d,
        BLOCK_K=block_k,
        NUM_SPLITS=num_splits,
        PAD_SPLITS=pad_splits,
        HAS_POST_NORM=has_post_norm,
        HAS_PRE_NORM=can_fuse_pre_norm,
        QUANT_FP8=can_fuse_pre_fp8,
        ENABLE_PDL=enable_pdl,
        num_warps=8 if block_d >= 2048 else 4,
        **({"launch_pdl": True} if enable_pdl else {}),
    )
    if can_fuse_pre_fp8:
        assert q_out is not None and q_scale_storage is not None
        return out_res, next_post, out, q_out, q_scale_storage[:, :num_tokens].t()
    return out_res, next_post, out


__all__ = ["ihc_head", "ihc_post", "ihc_post_and_pre", "ihc_pre"]
