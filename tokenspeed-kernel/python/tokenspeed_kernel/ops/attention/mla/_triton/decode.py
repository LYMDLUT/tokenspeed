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

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import pdl_enabled

_FP8_DTYPES = frozenset({torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e4m3fnuz})
_MLA_DECODE_DTYPES = frozenset({torch.float16, torch.bfloat16}) | _FP8_DTYPES


@triton.jit
def tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _mla_decode_kernel(
    Q,
    KV_Cache,
    Out,
    LSE,
    page_table,
    cache_seqlens,
    sm_scale,
    stride_qb,
    stride_qq,
    stride_qh,
    stride_kv_page,
    stride_kv_token,
    stride_kv_head,
    stride_ob,
    stride_oq,
    stride_oh,
    stride_lse_b,
    stride_lse_q,
    stride_lse_h,
    # Page-table width follows the batch; runtime so every batch shape
    # shares one binary.
    page_table_stride_b,
    PAGE_SIZE: tl.constexpr,
    # Unused by this kernel; the per-batch longest context must not key its
    # compilation.
    MAX_SEQLEN_K,
    logit_cap: tl.constexpr,
    KV_LORA_RANK: tl.constexpr,
    QK_ROPE_HEAD_DIM: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_ROPE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_LSE: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    NONCAUSAL_BLOCK_SIZE: tl.constexpr,
    Q_LEN: tl.constexpr,
):
    cur_batch = tl.program_id(0).to(tl.int64)
    cur_q = tl.program_id(1).to(tl.int64)
    cur_head = tl.program_id(2).to(tl.int64)

    offs_r = tl.arange(0, BLOCK_R)
    offs_rope = tl.arange(0, BLOCK_ROPE)
    mask_r = offs_r < KV_LORA_RANK
    mask_rope = offs_rope < QK_ROPE_HEAD_DIM

    q_base = cur_batch * stride_qb + cur_q * stride_qq + cur_head * stride_qh
    q_latent = tl.load(Q + q_base + offs_r, mask=mask_r, other=0.0)
    q_rope = tl.load(
        Q + q_base + KV_LORA_RANK + offs_rope,
        mask=mask_rope,
        other=0.0,
    )

    cache_len = tl.load(cache_seqlens + cur_batch).to(tl.int32)
    if NONCAUSAL_BLOCK_SIZE > 1:
        q_cache_len = cache_len
        if WINDOW_LEFT >= 0:
            # Non-causal draft block: every row sees the whole proposal block,
            # while historical context obeys query_position - key_position <= window_left.
            block_position = (
                cur_q.to(tl.int32)
                if Q_LEN > 1
                else (cur_batch.to(tl.int32) % NONCAUSAL_BLOCK_SIZE)
            )
            context_len = cache_len - NONCAUSAL_BLOCK_SIZE
            window_start = tl.maximum(
                0,
                context_len - WINDOW_LEFT + block_position,
            )
            loop_start = (window_start // BLOCK_N) * BLOCK_N
        else:
            window_start = 0
            loop_start = 0
    else:
        # Causal decode or multi-token verify (Q_LEN >= 1): query index cur_q
        # sees prefixes up to cache_len - (Q_LEN - 1 - cur_q).
        q_cache_len = tl.maximum(0, cache_len - (Q_LEN - 1 - cur_q.to(tl.int32)))
        if WINDOW_LEFT >= 0:
            window_start = tl.maximum(0, q_cache_len - 1 - WINDOW_LEFT)
            loop_start = (window_start // BLOCK_N) * BLOCK_N
        else:
            window_start = 0
            loop_start = 0
    offs_n = tl.arange(0, BLOCK_N)
    acc = tl.zeros([BLOCK_R], dtype=tl.float32)
    e_sum = 0.0
    e_max = -float("inf")

    for start_n in tl.range(loop_start, q_cache_len, BLOCK_N, num_stages=2):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        token_offsets = start_n + offs_n
        mask_n = (token_offsets >= window_start) & (token_offsets < q_cache_len)
        page_indices = token_offsets // PAGE_SIZE
        page_offsets = (token_offsets - page_indices * PAGE_SIZE).to(tl.int64)
        physical_pages = tl.load(
            page_table + cur_batch * page_table_stride_b + page_indices.to(tl.int64),
            mask=mask_n,
            other=0,
        ).to(tl.int64)
        cache_base = (
            physical_pages * stride_kv_page
            + page_offsets * stride_kv_token
            + 0 * stride_kv_head
        )

        k_latent = tl.load(
            KV_Cache + cache_base[:, None] + offs_r[None, :],
            mask=mask_n[:, None] & mask_r[None, :],
            other=0.0,
        )
        k_rope = tl.load(
            KV_Cache + cache_base[:, None] + KV_LORA_RANK + offs_rope[None, :],
            mask=mask_n[:, None] & mask_rope[None, :],
            other=0.0,
        )

        qk = tl.sum(k_latent.to(tl.float32) * q_latent[None, :].to(tl.float32), axis=1)
        qk += tl.sum(k_rope.to(tl.float32) * q_rope[None, :].to(tl.float32), axis=1)
        qk *= sm_scale

        if logit_cap > 0:
            qk = logit_cap * tanh(qk / logit_cap)

        qk = tl.where(mask_n, qk, float("-inf"))
        block_max = tl.max(qk, axis=0)
        block_max_fixed = tl.where(block_max == float("-inf"), -1e20, block_max)
        n_e_max = tl.maximum(block_max_fixed, e_max)
        old_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max)
        acc = acc * old_scale + tl.sum(p[:, None] * k_latent.to(tl.float32), axis=0)
        e_sum = e_sum * old_scale + tl.sum(p, axis=0)
        e_max = n_e_max

    safe_sum = tl.where(e_sum > 0.0, e_sum, 1.0)
    out_base = cur_batch * stride_ob + cur_q * stride_oq + cur_head * stride_oh
    tl.store(Out + out_base + offs_r, acc / safe_sum, mask=mask_r)

    if HAS_LSE:
        lse = tl.where(e_sum > 0.0, tl.log(e_sum) + e_max, float("-inf"))
        tl.store(
            LSE
            + cur_batch * stride_lse_b
            + cur_q * stride_lse_q
            + cur_head * stride_lse_h,
            lse,
        )


@triton.jit(
    do_not_specialize=[
        "page_table_stride_b",
    ]
)
def _mla_decode_tc_kernel(
    Q,
    KV_Cache,
    Out,
    LSE,
    PartialOut,
    PartialLSE,
    Sinks,
    page_table,
    cache_seqlens,
    sm_scale,
    stride_qb,
    stride_qq,
    stride_qh,
    stride_kv_page,
    stride_kv_tok,
    stride_ob,
    stride_oq,
    stride_oh,
    stride_lse_b,
    stride_lse_q,
    stride_lse_h,
    page_table_stride_b,
    NUM_HEADS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    KV_LORA_RANK: tl.constexpr,
    QK_ROPE_HEAD_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    HAS_LSE: tl.constexpr,
    HAS_SINKS: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    NONCAUSAL_BLOCK_SIZE: tl.constexpr,
    Q_LEN: tl.constexpr,
    ENABLE_PDL: tl.constexpr = False,
):
    """Grouped-head Tensor-Core split-K MLA decode supporting SWA and multi-q verify."""
    cur_batch = tl.program_id(0).to(tl.int64)
    cur_q = tl.program_id(1).to(tl.int64)
    pid2 = tl.program_id(2)
    split = pid2 % NUM_SPLITS
    h_blk = pid2 // NUM_SPLITS
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()

    heads = h_blk * BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = heads < NUM_HEADS
    dims = tl.arange(0, KV_LORA_RANK)
    rope_dims = tl.arange(0, QK_ROPE_HEAD_DIM)

    q_base = cur_batch * stride_qb + cur_q * stride_qq + heads[:, None] * stride_qh
    q_lat = tl.load(Q + q_base + dims[None, :], mask=head_mask[:, None], other=0.0)
    q_rope = tl.load(
        Q + q_base + KV_LORA_RANK + rope_dims[None, :],
        mask=head_mask[:, None],
        other=0.0,
    )
    dot_dtype = q_lat.dtype

    cache_len = tl.load(cache_seqlens + cur_batch).to(tl.int32)
    if NONCAUSAL_BLOCK_SIZE > 1:
        q_cache_len = cache_len
        if WINDOW_LEFT >= 0:
            block_position = (
                cur_q.to(tl.int32)
                if Q_LEN > 1
                else (cur_batch.to(tl.int32) % NONCAUSAL_BLOCK_SIZE)
            )
            context_len = cache_len - NONCAUSAL_BLOCK_SIZE
            window_start = tl.maximum(
                0,
                context_len - WINDOW_LEFT + block_position,
            )
        else:
            window_start = 0
    else:
        q_cache_len = tl.maximum(0, cache_len - (Q_LEN - 1 - cur_q.to(tl.int32)))
        if WINDOW_LEFT >= 0:
            window_start = tl.maximum(0, q_cache_len - 1 - WINDOW_LEFT)
        else:
            window_start = 0

    loop_start = (window_start // BLOCK_N) * BLOCK_N
    total_blocks = tl.cdiv(tl.maximum(0, q_cache_len - loop_start), BLOCK_N)
    blocks_per_split = tl.cdiv(total_blocks, NUM_SPLITS)
    s_blk = split * blocks_per_split
    e_blk = tl.minimum(total_blocks, s_blk + blocks_per_split)
    kv_start = loop_start + s_blk * BLOCK_N
    kv_end = tl.minimum(q_cache_len, loop_start + e_blk * BLOCK_N)

    scale_log2 = sm_scale * 1.4426950408889634
    m_i = tl.full([BLOCK_H], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_H], tl.float32)
    acc = tl.zeros([BLOCK_H, KV_LORA_RANK], tl.float32)
    n_offs = tl.arange(0, BLOCK_N)

    for pos0 in range(kv_start, kv_end, BLOCK_N):
        pos = pos0 + n_offs
        valid = (pos >= window_start) & (pos < q_cache_len)
        # BLOCK_N divides PAGE_SIZE and pos0 is BLOCK_N-aligned, so all tokens
        # in this tile lie within one contiguous physical KV page.
        page_idx = pos0 // PAGE_SIZE
        page_off0 = pos0 % PAGE_SIZE
        phys_page = tl.load(page_table + cur_batch * page_table_stride_b + page_idx).to(
            tl.int64
        )
        kv_base = (
            phys_page * stride_kv_page
            + (page_off0 + n_offs).to(tl.int64) * stride_kv_tok
        )

        k_lat = tl.load(
            KV_Cache + kv_base[:, None] + dims[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(dot_dtype)
        scores = tl.dot(q_lat, tl.trans(k_lat))
        k_rope = tl.load(
            KV_Cache + kv_base[:, None] + KV_LORA_RANK + rope_dims[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(dot_dtype)
        scores = tl.dot(q_rope, tl.trans(k_rope), scores)
        scores = tl.where(valid[None, :], scores * scale_log2, -float("inf"))

        block_max = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, block_max)
        safe_m = tl.where(m_new == -float("inf"), 0.0, m_new)
        p = tl.exp2(scores - safe_m[:, None])
        alpha = tl.where(m_i == -float("inf"), 0.0, tl.exp2(m_i - safe_m))
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(dot_dtype), k_lat, acc)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    res = tl.where(l_i[:, None] > 0.0, acc / l_i[:, None], 0.0)
    if NUM_SPLITS == 1:
        lse_val = tl.where(
            l_i > 0.0, (m_i + tl.log2(l_i)) * 0.6931471805599453, -float("inf")
        )
        if HAS_SINKS:
            sink_val = tl.load(Sinks + heads, mask=head_mask, other=0.0).to(tl.float32)
            res = res * tl.sigmoid(lse_val - sink_val)[:, None]
        o_base = cur_batch * stride_ob + cur_q * stride_oq + heads[:, None] * stride_oh
        tl.store(
            Out + o_base + dims[None, :],
            res.to(Out.dtype.element_ty),
            mask=head_mask[:, None],
        )
        if HAS_LSE:
            tl.store(
                LSE
                + cur_batch * stride_lse_b
                + cur_q * stride_lse_q
                + heads * stride_lse_h,
                lse_val,
                mask=head_mask,
            )
    else:
        row_idx = (cur_batch * Q_LEN + cur_q) * NUM_HEADS + heads
        p_row = row_idx * NUM_SPLITS + split
        tl.store(
            PartialOut + p_row[:, None] * KV_LORA_RANK + dims[None, :],
            res,
            mask=head_mask[:, None],
        )
        lse_val = tl.where(l_i > 0.0, m_i + tl.log2(l_i), -float("inf"))
        tl.store(PartialLSE + p_row, lse_val, mask=head_mask)
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit(
    do_not_specialize=[
        "page_table_stride_b",
    ]
)
def _mla_decode_multiq_tc_kernel(
    Q,
    KV_Cache,
    Out,
    LSE,
    PartialOut,
    PartialLSE,
    Sinks,
    page_table,
    cache_seqlens,
    sm_scale,
    stride_qb,
    stride_qq,
    stride_qh,
    stride_kv_page,
    stride_kv_tok,
    stride_ob,
    stride_oq,
    stride_oh,
    stride_lse_b,
    stride_lse_q,
    stride_lse_h,
    page_table_stride_b,
    NUM_HEADS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    KV_LORA_RANK: tl.constexpr,
    QK_ROPE_HEAD_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    Q_GROUP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    HAS_LSE: tl.constexpr,
    HAS_SINKS: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    Q_LEN: tl.constexpr,
    ENABLE_PDL: tl.constexpr = False,
):
    """Multi-query shared-KV Tensor-Core MLA decode for causal multi-token verify."""
    cur_batch = tl.program_id(0).to(tl.int64)
    q_blk = tl.program_id(1).to(tl.int64)
    pid2 = tl.program_id(2)
    split = pid2 % NUM_SPLITS
    h_blk = pid2 // NUM_SPLITS
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()

    m_offs = tl.arange(0, BLOCK_M)
    q_local = m_offs // BLOCK_H
    h_local = m_offs % BLOCK_H
    cur_q = q_blk * Q_GROUP + q_local
    heads = h_blk * BLOCK_H + h_local
    m_mask = (q_local < Q_GROUP) & (cur_q < Q_LEN) & (heads < NUM_HEADS)

    dims = tl.arange(0, KV_LORA_RANK)
    rope_dims = tl.arange(0, QK_ROPE_HEAD_DIM)
    q_safe = tl.minimum(cur_q, Q_LEN - 1).to(tl.int64)
    h_safe = tl.minimum(heads, NUM_HEADS - 1).to(tl.int64)
    q_base = (
        cur_batch * stride_qb
        + q_safe[:, None] * stride_qq
        + h_safe[:, None] * stride_qh
    )
    q_lat = tl.load(Q + q_base + dims[None, :], mask=m_mask[:, None], other=0.0)
    q_rope = tl.load(
        Q + q_base + KV_LORA_RANK + rope_dims[None, :],
        mask=m_mask[:, None],
        other=0.0,
    )
    dot_dtype = q_lat.dtype

    cache_len = tl.load(cache_seqlens + cur_batch).to(tl.int32)
    row_q_end = tl.maximum(0, cache_len - (Q_LEN - 1 - cur_q.to(tl.int32)))
    if WINDOW_LEFT >= 0:
        row_win_start = tl.maximum(0, row_q_end - 1 - WINDOW_LEFT)
    else:
        row_win_start = tl.zeros([BLOCK_M], dtype=tl.int32)

    q_first = (q_blk * Q_GROUP).to(tl.int32)
    q_last = tl.minimum(Q_LEN - 1, q_first + Q_GROUP - 1)
    union_end = tl.maximum(0, cache_len - (Q_LEN - 1 - q_last))
    if WINDOW_LEFT >= 0:
        first_end = tl.maximum(0, cache_len - (Q_LEN - 1 - q_first))
        union_start = tl.maximum(0, first_end - 1 - WINDOW_LEFT)
    else:
        union_start = 0

    loop_start = (union_start // BLOCK_N) * BLOCK_N
    total_blocks = tl.cdiv(tl.maximum(0, union_end - loop_start), BLOCK_N)
    blocks_per_split = tl.cdiv(total_blocks, NUM_SPLITS)
    s_blk = split * blocks_per_split
    e_blk = tl.minimum(total_blocks, s_blk + blocks_per_split)
    kv_start = loop_start + s_blk * BLOCK_N
    kv_end = tl.minimum(union_end, loop_start + e_blk * BLOCK_N)

    scale_log2 = sm_scale * 1.4426950408889634
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, KV_LORA_RANK], tl.float32)
    n_offs = tl.arange(0, BLOCK_N)

    for pos0 in range(kv_start, kv_end, BLOCK_N):
        pos = pos0 + n_offs
        valid_k = (pos >= union_start) & (pos < union_end)
        page_idx = pos0 // PAGE_SIZE
        page_off0 = pos0 % PAGE_SIZE
        phys_page = tl.load(page_table + cur_batch * page_table_stride_b + page_idx).to(
            tl.int64
        )
        kv_base = (
            phys_page * stride_kv_page
            + (page_off0 + n_offs).to(tl.int64) * stride_kv_tok
        )
        k_lat = tl.load(
            KV_Cache + kv_base[:, None] + dims[None, :],
            mask=valid_k[:, None],
            other=0.0,
        ).to(dot_dtype)
        scores = tl.dot(q_lat, tl.trans(k_lat))
        k_rope = tl.load(
            KV_Cache + kv_base[:, None] + KV_LORA_RANK + rope_dims[None, :],
            mask=valid_k[:, None],
            other=0.0,
        ).to(dot_dtype)
        scores = tl.dot(q_rope, tl.trans(k_rope), scores)
        valid_2d = (pos[None, :] >= row_win_start[:, None]) & (
            pos[None, :] < row_q_end[:, None]
        )
        scores = tl.where(valid_2d, scores * scale_log2, -float("inf"))

        block_max = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, block_max)
        safe_m = tl.where(m_new == -float("inf"), 0.0, m_new)
        p = tl.exp2(scores - safe_m[:, None])
        alpha = tl.where(m_i == -float("inf"), 0.0, tl.exp2(m_i - safe_m))
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(dot_dtype), k_lat, acc)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    res = tl.where(l_i[:, None] > 0.0, acc / l_i[:, None], 0.0)
    if NUM_SPLITS == 1:
        lse_val = tl.where(
            l_i > 0.0, (m_i + tl.log2(l_i)) * 0.6931471805599453, -float("inf")
        )
        if HAS_SINKS:
            sink_val = tl.load(Sinks + h_safe, mask=m_mask, other=0.0).to(tl.float32)
            res = res * tl.sigmoid(lse_val - sink_val)[:, None]
        o_base = (
            cur_batch * stride_ob
            + q_safe[:, None] * stride_oq
            + h_safe[:, None] * stride_oh
        )
        tl.store(
            Out + o_base + dims[None, :],
            res.to(Out.dtype.element_ty),
            mask=m_mask[:, None],
        )
        if HAS_LSE:
            tl.store(
                LSE
                + cur_batch * stride_lse_b
                + q_safe * stride_lse_q
                + h_safe * stride_lse_h,
                lse_val,
                mask=m_mask,
            )
    else:
        row_idx = (cur_batch * Q_LEN + q_safe) * NUM_HEADS + h_safe
        p_row = row_idx * NUM_SPLITS + split
        tl.store(
            PartialOut + p_row[:, None] * KV_LORA_RANK + dims[None, :],
            res,
            mask=m_mask[:, None],
        )
        lse_val = tl.where(l_i > 0.0, m_i + tl.log2(l_i), -float("inf"))
        tl.store(PartialLSE + p_row, lse_val, mask=m_mask)
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit(
    do_not_specialize=[
        "stride_ob",
        "stride_oq",
        "stride_oh",
        "stride_lse_b",
        "stride_lse_q",
        "stride_lse_h",
    ]
)
def _mla_decode_merge_splits_kernel(
    PartialOut,
    PartialLSE,
    Out,
    LSE,
    Sinks,
    stride_ob,
    stride_oq,
    stride_oh,
    stride_lse_b,
    stride_lse_q,
    stride_lse_h,
    Q_LEN: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HAS_LSE: tl.constexpr,
    HAS_SINKS: tl.constexpr,
    KV_LORA_RANK: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ENABLE_PDL: tl.constexpr = False,
):
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
    row = tl.program_id(0).to(tl.int64)
    dims = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    s_rows = row * NUM_SPLITS + tl.arange(0, NUM_SPLITS)
    lse = tl.load(PartialLSE + s_rows)
    m = tl.max(lse, axis=0)
    w = tl.exp2(lse - tl.where(m == -float("inf"), 0.0, m))
    tot = tl.sum(w, axis=0)
    part = tl.load(PartialOut + s_rows[:, None] * KV_LORA_RANK + dims[None, :])
    merged = tl.sum(part * w[:, None], axis=0)

    head = row % NUM_HEADS
    rem = row // NUM_HEADS
    cur_q = rem % Q_LEN
    cur_batch = rem // Q_LEN
    o_ptr = Out + cur_batch * stride_ob + cur_q * stride_oq + head * stride_oh + dims
    out_val = tl.where(tot > 0.0, merged / tot, 0.0)
    lse_nat = tl.where(
        tot > 0.0, (m + tl.log2(tot)) * 0.6931471805599453, -float("inf")
    )
    if HAS_SINKS:
        sink_val = tl.load(Sinks + head).to(tl.float32)
        out_val = out_val * tl.sigmoid(lse_nat - sink_val)
    tl.store(o_ptr, out_val.to(Out.dtype.element_ty))
    if HAS_LSE and tl.program_id(1) == 0:
        lse_ptr = (
            LSE + cur_batch * stride_lse_b + cur_q * stride_lse_q + head * stride_lse_h
        )
        tl.store(lse_ptr, lse_nat)
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def _normalize_kv_cache(kv_cache: torch.Tensor) -> torch.Tensor:
    if kv_cache.dim() == 3:
        return kv_cache.unsqueeze(2)
    if kv_cache.dim() != 4:
        raise ValueError(f"kv_cache must be 3D or 4D, got {kv_cache.dim()}D")
    if kv_cache.shape[2] == 1:
        return kv_cache
    if kv_cache.shape[1] == 1:
        return kv_cache[:, 0].unsqueeze(2)
    raise ValueError(f"unsupported kv_cache shape {tuple(kv_cache.shape)}")


def mla_decode_fwd(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    out: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_k: int,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    softmax_scale: float,
    *,
    logit_cap: float = 0.0,
    lse: torch.Tensor | None = None,
    window_left: int = -1,
    noncausal_block_size: int = 1,
    sinks: torch.Tensor | None = None,
) -> None:
    if q.dim() != 4:
        raise ValueError(
            f"q must have shape [B, q_len, H, R + rope], got {tuple(q.shape)}"
        )
    q_len = int(q.shape[1])
    if q_len < 1:
        raise ValueError(f"q_len must be positive, got {q_len}")
    if q.shape[-1] != kv_lora_rank + qk_rope_head_dim:
        raise ValueError(
            f"q head dim must be {kv_lora_rank + qk_rope_head_dim}, got {q.shape[-1]}"
        )
    if out.shape != q.shape[:-1] + (kv_lora_rank,):
        raise ValueError(
            f"out shape must be {q.shape[:-1] + (kv_lora_rank,)}, "
            f"got {tuple(out.shape)}"
        )
    if q.stride(-1) != 1 or out.stride(-1) != 1:
        raise ValueError("q and out must have contiguous last dimension")
    if lse is not None and lse.shape != q.shape[:-1]:
        raise ValueError(f"lse shape must be {q.shape[:-1]}, got {tuple(lse.shape)}")
    if window_left < -1:
        raise ValueError(f"window_left must be -1 or non-negative, got {window_left}")
    if noncausal_block_size <= 0:
        raise ValueError(
            f"noncausal_block_size must be positive, got {noncausal_block_size}"
        )
    if 0 <= window_left < noncausal_block_size - 1:
        raise ValueError(
            "window_left must cover the complete non-causal block; got "
            f"window_left={window_left}, block_size={noncausal_block_size}"
        )
    if (
        window_left >= 0
        and noncausal_block_size > 1
        and q_len == 1
        and q.shape[0] % noncausal_block_size
    ):
        raise ValueError(
            "sliding MLA decode rows must contain complete non-causal blocks: "
            f"batch={q.shape[0]}, block_size={noncausal_block_size}"
        )
    if noncausal_block_size > 1 and q_len > 1 and q_len != noncausal_block_size:
        raise ValueError(
            f"noncausal_block_size={noncausal_block_size} must match q_len={q_len} "
            "when q_len > 1"
        )

    kv_cache = _normalize_kv_cache(kv_cache)
    if kv_cache.shape[2] != 1:
        raise ValueError(f"MLA kv_cache must have one KV head, got {kv_cache.shape[2]}")
    if kv_cache.shape[-1] != kv_lora_rank + qk_rope_head_dim:
        raise ValueError(
            f"kv_cache head dim must be {kv_lora_rank + qk_rope_head_dim}, "
            f"got {kv_cache.shape[-1]}"
        )
    if kv_cache.stride(-1) != 1:
        raise ValueError("kv_cache must have contiguous last dimension")

    if q.shape[0] == 0:
        return

    num_heads = int(q.shape[2])
    page_size = int(kv_cache.shape[1])
    lse_arg = lse if lse is not None else out

    # Multi-query shared-KV Tensor-Core path for causal multi-token verify
    # when num_heads < 32 (for H >= 32, BLOCK_H=16 BLOCK_N=64 single-Q TC
    # eliminates register spills and merge kernels, outperforming multi-Q).
    enable_pdl = pdl_enabled()
    pdl_kwargs = {"launch_pdl": True} if enable_pdl else {}
    if (
        q.is_cuda
        and q.dtype in (torch.bfloat16, torch.float16)
        and kv_cache.dtype in (torch.bfloat16, torch.float16)
        and logit_cap == 0.0
        and window_left >= 0
        and noncausal_block_size == 1
        and q_len >= 2
        and q.shape[0] >= 2
        and kv_lora_rank == 512
        and qk_rope_head_dim == 64
        and 16 <= num_heads < 32
        and num_heads % 16 == 0
        and page_size >= 32
        and page_size % 32 == 0
    ):
        batch_size = int(q.shape[0])
        block_h = 16
        block_n = 32
        if batch_size <= 8:
            q_group, block_m, num_splits, num_warps = 2, 32, 4, 4
        elif batch_size <= 24 and q_len >= 3:
            q_group, block_m, num_splits, num_warps = 3, 64, 1, 8
        else:
            q_group, block_m, num_splits, num_warps = 2, 32, 1, 4
        total_rows = batch_size * q_len * num_heads
        if num_splits > 1:
            partial_out = torch.empty(
                (total_rows, num_splits, kv_lora_rank),
                dtype=torch.float32,
                device=q.device,
            )
            partial_lse = torch.empty(
                (total_rows, num_splits),
                dtype=torch.float32,
                device=q.device,
            )
        else:
            partial_out = out
            partial_lse = lse_arg
        sinks_arg = sinks.contiguous() if sinks is not None else out
        has_sinks = sinks is not None
        q_blks = triton.cdiv(q_len, q_group)
        grid = (batch_size, q_blks, (num_heads // block_h) * num_splits)
        _mla_decode_multiq_tc_kernel[grid](
            q,
            kv_cache,
            out,
            lse_arg,
            partial_out,
            partial_lse,
            sinks_arg,
            page_table,
            cache_seqlens,
            softmax_scale,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            kv_cache.stride(0),
            kv_cache.stride(1),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            lse_arg.stride(0),
            lse_arg.stride(1),
            lse_arg.stride(2),
            page_table.stride(0),
            NUM_HEADS=num_heads,
            PAGE_SIZE=page_size,
            KV_LORA_RANK=kv_lora_rank,
            QK_ROPE_HEAD_DIM=qk_rope_head_dim,
            BLOCK_H=block_h,
            Q_GROUP=q_group,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            NUM_SPLITS=num_splits,
            HAS_LSE=lse is not None,
            HAS_SINKS=has_sinks,
            WINDOW_LEFT=window_left,
            Q_LEN=q_len,
            ENABLE_PDL=enable_pdl,
            num_warps=num_warps,
            num_stages=2,
            **pdl_kwargs,
        )
        if num_splits > 1:
            _mla_decode_merge_splits_kernel[(total_rows, kv_lora_rank // 128)](
                partial_out,
                partial_lse,
                out,
                lse_arg,
                sinks_arg,
                out.stride(0),
                out.stride(1),
                out.stride(2),
                lse_arg.stride(0),
                lse_arg.stride(1),
                lse_arg.stride(2),
                Q_LEN=q_len,
                NUM_HEADS=num_heads,
                HAS_LSE=lse is not None,
                HAS_SINKS=has_sinks,
                KV_LORA_RANK=kv_lora_rank,
                NUM_SPLITS=num_splits,
                BLOCK_D=128,
                ENABLE_PDL=enable_pdl,
                num_warps=4,
                **pdl_kwargs,
            )
        return

    # Grouped-head Tensor-Core split-K path: shares each KV page across BLOCK_H
    # query heads via tl.dot instead of reloading KV once per head with scalar FMA.
    if (
        q.is_cuda
        and q.dtype in (torch.bfloat16, torch.float16)
        and kv_cache.dtype in (torch.bfloat16, torch.float16)
        and logit_cap == 0.0
        and kv_lora_rank == 512
        and qk_rope_head_dim == 64
        and num_heads >= 16
        and page_size >= 32
        and page_size % 32 == 0
    ):
        # BLOCK_H=16 halves the FP32 [BLOCK_H, 512] accumulator footprint,
        # eliminating the 255-register spills of BLOCK_H=32 while doubling CTA
        # parallelism across SMs.
        block_h = 16
        total_q_rows = int(q.shape[0]) * q_len
        can_bn64 = page_size % 64 == 0
        if window_left >= 0:
            if total_q_rows >= 24:
                # 48+ CTAs saturate Hopper SMs with single-pass full-page loads
                # and skip partial buffers and the merge kernel completely.
                block_n = 64 if can_bn64 else 32
                num_splits = 1
            elif total_q_rows <= 2:
                block_n = 32
                num_splits = 16
            elif total_q_rows <= 4:
                block_n = 64 if can_bn64 else 32
                num_splits = 8
            elif total_q_rows <= 12:
                block_n = 64 if can_bn64 else 32
                num_splits = 4
            else:
                block_n = 64 if can_bn64 else 32
                num_splits = 2
        else:
            block_n = 64 if can_bn64 else 32
            num_splits = 4
        total_rows = total_q_rows * num_heads
        if num_splits > 1:
            partial_out = torch.empty(
                (total_rows, num_splits, kv_lora_rank),
                dtype=torch.float32,
                device=q.device,
            )
            partial_lse = torch.empty(
                (total_rows, num_splits),
                dtype=torch.float32,
                device=q.device,
            )
        else:
            partial_out = out
            partial_lse = lse_arg
        sinks_arg = sinks.contiguous() if sinks is not None else out
        has_sinks = sinks is not None
        grid = (q.shape[0], q_len, triton.cdiv(num_heads, block_h) * num_splits)
        _mla_decode_tc_kernel[grid](
            q,
            kv_cache,
            out,
            lse_arg,
            partial_out,
            partial_lse,
            sinks_arg,
            page_table,
            cache_seqlens,
            softmax_scale,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            kv_cache.stride(0),
            kv_cache.stride(1),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            lse_arg.stride(0),
            lse_arg.stride(1),
            lse_arg.stride(2),
            page_table.stride(0),
            NUM_HEADS=num_heads,
            PAGE_SIZE=page_size,
            KV_LORA_RANK=kv_lora_rank,
            QK_ROPE_HEAD_DIM=qk_rope_head_dim,
            BLOCK_H=block_h,
            BLOCK_N=block_n,
            NUM_SPLITS=num_splits,
            HAS_LSE=lse is not None,
            HAS_SINKS=has_sinks,
            WINDOW_LEFT=window_left,
            NONCAUSAL_BLOCK_SIZE=noncausal_block_size,
            Q_LEN=q_len,
            ENABLE_PDL=enable_pdl,
            num_warps=4,
            num_stages=2,
            **pdl_kwargs,
        )
        if num_splits > 1:
            _mla_decode_merge_splits_kernel[(total_rows, kv_lora_rank // 128)](
                partial_out,
                partial_lse,
                out,
                lse_arg,
                sinks_arg,
                out.stride(0),
                out.stride(1),
                out.stride(2),
                lse_arg.stride(0),
                lse_arg.stride(1),
                lse_arg.stride(2),
                Q_LEN=q_len,
                NUM_HEADS=num_heads,
                HAS_LSE=lse is not None,
                HAS_SINKS=has_sinks,
                KV_LORA_RANK=kv_lora_rank,
                NUM_SPLITS=num_splits,
                BLOCK_D=128,
                ENABLE_PDL=enable_pdl,
                num_warps=4,
                **pdl_kwargs,
            )
        return

    need_fallback_lse = sinks is not None and lse is None
    fallback_lse = (
        torch.empty(q.shape[:-1], dtype=torch.float32, device=q.device)
        if need_fallback_lse
        else lse
    )
    fallback_lse_arg = fallback_lse if fallback_lse is not None else out
    block_n = 16
    block_r = triton.next_power_of_2(kv_lora_rank)
    block_rope = triton.next_power_of_2(qk_rope_head_dim)
    grid = (q.shape[0], q.shape[1], q.shape[2])

    _mla_decode_kernel[grid](
        q,
        kv_cache,
        out,
        fallback_lse_arg,
        page_table,
        cache_seqlens,
        softmax_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        fallback_lse_arg.stride(0),
        fallback_lse_arg.stride(1),
        fallback_lse_arg.stride(2),
        page_table.stride(0),
        kv_cache.shape[1],
        max_seqlen_k,
        logit_cap=logit_cap,
        KV_LORA_RANK=kv_lora_rank,
        QK_ROPE_HEAD_DIM=qk_rope_head_dim,
        BLOCK_R=block_r,
        BLOCK_ROPE=block_rope,
        BLOCK_N=block_n,
        HAS_LSE=fallback_lse is not None,
        WINDOW_LEFT=window_left,
        NONCAUSAL_BLOCK_SIZE=noncausal_block_size,
        Q_LEN=q_len,
        num_warps=8,
        num_stages=2,
    )
    if sinks is not None:
        from tokenspeed_kernel.ops.attention.sink import apply_attention_sink

        assert fallback_lse is not None
        apply_attention_sink(
            out.reshape(-1, num_heads, kv_lora_rank),
            fallback_lse.reshape(-1, num_heads),
            sinks,
            inplace=True,
        )


def _triton_mla_decode_with_kvcache_impl(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_k: int,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    softmax_scale: float,
    *,
    logit_cap: float = 0.0,
    return_lse: bool = False,
    out: torch.Tensor | None = None,
    window_left: int = -1,
    noncausal_block_size: int = 1,
    sinks: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if out is None:
        out_dtype = torch.bfloat16 if q.dtype in _FP8_DTYPES else q.dtype
        out = torch.empty(
            q.shape[:-1] + (kv_lora_rank,), dtype=out_dtype, device=q.device
        )

    lse = (
        torch.empty(q.shape[:-1], dtype=torch.float32, device=q.device)
        if return_lse
        else None
    )
    mla_decode_fwd(
        q,
        kv_cache,
        out,
        page_table,
        cache_seqlens,
        max_seqlen_k,
        qk_nope_head_dim,
        kv_lora_rank,
        qk_rope_head_dim,
        softmax_scale,
        logit_cap=logit_cap,
        lse=lse,
        window_left=window_left,
        noncausal_block_size=noncausal_block_size,
        sinks=sinks,
    )
    if return_lse:
        return out, lse
    return out
