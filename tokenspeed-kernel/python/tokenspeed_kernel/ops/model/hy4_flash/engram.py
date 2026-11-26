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


"""Hy4-flash fused Engram n-gram hashing, embedding lookup, and gating Triton kernels."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.attention.kpool.hy4_compress import (
    _torch_to_triton_dtype,
)


@triton.jit(
    do_not_specialize=[
        "num_tokens",
        "table_size",
        "pad_id",
        "row_start",
        "row_end",
        "stride_nt_t",
        "stride_nt_k",
        "stride_w_row",
        "stride_o_t",
        "stride_o_h",
    ]
)
def _hy4_engram_hash_embed_kernel(
    ngram_tokens_ptr,
    lookup_table_ptr,
    multipliers_ptr,
    moduli_ptr,
    offsets_ptr,
    weight_ptr,
    out_ptr,
    num_tokens,
    table_size,
    pad_id,
    row_start,
    row_end,
    stride_nt_t,
    stride_nt_k,
    stride_w_row,
    stride_o_t,
    stride_o_h,
    MAX_NGRAM: tl.constexpr,
    N_HEAD_PER_NG: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    DO_EMBED: tl.constexpr,
):
    """Fused Engram n-gram hashing with optional GPU row-sharded embedding lookup."""
    tok = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    if tok >= num_tokens:
        return

    order = head // N_HEAD_PER_NG
    ngram_len = order + 2
    pad_comp = tl.load(lookup_table_ptr + pad_id).to(tl.int64)

    mix = tl.zeros((), dtype=tl.int64)
    for k in tl.static_range(MAX_NGRAM):
        if k < ngram_len:
            tid = tl.load(ngram_tokens_ptr + tok * stride_nt_t + k * stride_nt_k).to(
                tl.int64
            )
            valid = (tid >= 0) & (tid < table_size)
            safe_tid = tl.where(valid, tid, 0)
            comp = tl.where(
                valid,
                tl.load(lookup_table_ptr + safe_tid).to(tl.int64),
                pad_comp,
            )
            term = comp * tl.load(multipliers_ptr + k).to(tl.int64)
            mix = term if k == 0 else (mix ^ term)

    mod = tl.load(moduli_ptr + head).to(tl.int64)
    rem = mix % mod
    hash_id = tl.where(rem < 0, rem + mod, rem)

    if not DO_EMBED:
        if row_end > row_start:
            global_id = hash_id + tl.load(offsets_ptr + head).to(tl.int64)
            in_shard = (global_id >= row_start) & (global_id < row_end)
            hash_id = tl.where(in_shard, global_id - row_start, -1)
        tl.store(out_ptr + tok * stride_o_t + head, hash_id)
        return

    global_id = hash_id + tl.load(offsets_ptr + head).to(tl.int64)
    in_shard = (global_id >= row_start) & (global_id < row_end)
    local_id = tl.where(in_shard, global_id - row_start, 0)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < HEAD_DIM
    vec = tl.load(
        weight_ptr + local_id * stride_w_row + offs_d,
        mask=mask_d & in_shard,
        other=0.0,
    ).to(out_ptr.dtype.element_ty)
    tl.store(
        out_ptr + tok * stride_o_t + head * stride_o_h + offs_d,
        vec,
        mask=mask_d,
    )


def hy4_engram_hash_ngrams(
    ngram_tokens: torch.Tensor,
    lookup_table: torch.Tensor,
    hash_multipliers: torch.Tensor,
    head_vocab_sizes: torch.Tensor,
    *,
    pad_id: int,
    max_ngram_size: int,
    n_head_per_ngram: int,
    offsets: torch.Tensor | None = None,
    row_start: int = 0,
    row_end: int = 0,
) -> torch.Tensor:
    """Fused single-kernel Engram n-gram hashing -> ``[T, num_heads]`` int64.

    When ``offsets`` is provided with ``row_end > row_start``, directly emits
    shard-local embedding row indices (with ``-1`` for out-of-shard entries) so
    host-table UVA ``uint8_row_gather`` consumes them with zero intermediate ops.
    """
    num_tokens = ngram_tokens.shape[0]
    num_heads = head_vocab_sizes.shape[0]
    out = torch.empty(
        (num_tokens, num_heads), dtype=torch.int64, device=ngram_tokens.device
    )
    if num_tokens == 0 or num_heads == 0:
        return out
    nt_c = ngram_tokens if ngram_tokens.stride(-1) == 1 else ngram_tokens.contiguous()
    use_shard = offsets is not None and row_end > row_start
    _hy4_engram_hash_embed_kernel[(num_tokens, num_heads)](
        nt_c,
        lookup_table,
        hash_multipliers,
        head_vocab_sizes,
        offsets if use_shard else head_vocab_sizes,
        out,
        out,
        num_tokens,
        lookup_table.shape[0],
        int(pad_id),
        int(row_start) if use_shard else 0,
        int(row_end) if use_shard else 0,
        nt_c.stride(0),
        nt_c.stride(1),
        0,
        out.stride(0),
        1,
        MAX_NGRAM=max_ngram_size,
        N_HEAD_PER_NG=n_head_per_ngram,
        HEAD_DIM=1,
        BLOCK_D=1,
        DO_EMBED=False,
        num_warps=1,
    )
    return out


def hy4_engram_hash_and_embed(
    ngram_tokens: torch.Tensor,
    lookup_table: torch.Tensor,
    hash_multipliers: torch.Tensor,
    head_vocab_sizes: torch.Tensor,
    offsets: torch.Tensor,
    weight: torch.Tensor,
    *,
    pad_id: int,
    max_ngram_size: int,
    n_head_per_ngram: int,
    row_start: int,
    row_end: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Fused Engram n-gram hashing + GPU sharded embedding lookup -> ``[T, H, D]``."""
    num_tokens = ngram_tokens.shape[0]
    num_heads = head_vocab_sizes.shape[0]
    head_dim = weight.shape[-1]
    out = torch.empty(
        (num_tokens, num_heads, head_dim),
        dtype=out_dtype,
        device=ngram_tokens.device,
    )
    if num_tokens == 0 or num_heads == 0:
        return out
    nt_c = ngram_tokens if ngram_tokens.stride(-1) == 1 else ngram_tokens.contiguous()
    block_d = triton.next_power_of_2(head_dim)
    _hy4_engram_hash_embed_kernel[(num_tokens, num_heads)](
        nt_c,
        lookup_table,
        hash_multipliers,
        head_vocab_sizes,
        offsets,
        weight,
        out,
        num_tokens,
        lookup_table.shape[0],
        int(pad_id),
        int(row_start),
        int(row_end),
        nt_c.stride(0),
        nt_c.stride(1),
        weight.stride(0),
        out.stride(0),
        out.stride(1),
        MAX_NGRAM=max_ngram_size,
        N_HEAD_PER_NG=n_head_per_ngram,
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
        DO_EMBED=True,
        num_warps=2 if block_d <= 128 else 4,
    )
    return out


@triton.jit(
    do_not_specialize=[
        "num_tokens",
        "stride_et",
        "stride_eh",
        "stride_wh",
        "stride_ot",
    ]
)
def _hy4_engram_lookup_norm_flatten_kernel(
    emb_ptr,
    norm_w_ptr,
    out_ptr,
    num_tokens,
    stride_et,
    stride_eh,
    stride_wh,
    stride_ot,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ORIG_DTYPE: tl.constexpr,
    FP8_ROUND_TRIP: tl.constexpr,
):
    """Fused (optional FP8 round-trip) + per-head RMSNorm + flatten for Engram."""
    tok = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    if tok >= num_tokens:
        return

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < HEAD_DIM
    x = tl.load(
        emb_ptr + tok * stride_et + head * stride_eh + offs_d,
        mask=mask_d,
        other=0.0,
    ).to(tl.float32)

    if FP8_ROUND_TRIP:
        absmax = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-12)
        scale = (absmax / 448.0).to(ORIG_DTYPE).to(tl.float32)
        q_in = (x / scale).to(ORIG_DTYPE).to(tl.float32)
        q = tl.minimum(tl.maximum(q_in, -448.0), 448.0).to(tl.float8e4nv)
        x = (q.to(ORIG_DTYPE).to(tl.float32) * scale).to(ORIG_DTYPE).to(tl.float32)

    # Match PyTorch F.rms_norm(x, (head_dim,)) default eps = finfo(float32).eps.
    inv_rms = tl.rsqrt(tl.sum(x * x, axis=0) / HEAD_DIM + 1.1920928955078125e-07)
    w = tl.load(norm_w_ptr + head * stride_wh + offs_d, mask=mask_d, other=0.0).to(
        ORIG_DTYPE
    )
    y = ((x * inv_rms).to(ORIG_DTYPE) * w).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + tok * stride_ot + head * HEAD_DIM + offs_d, y, mask=mask_d)


def hy4_engram_lookup_norm_flatten(
    embeddings: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    fp8_round_trip: bool = False,
) -> torch.Tensor:
    """Fused Engram per-head RMSNorm + weight scale + flatten -> ``[T, H * D]``."""
    num_tokens, num_heads, head_dim = embeddings.shape
    out = torch.empty(
        (num_tokens, num_heads * head_dim),
        dtype=embeddings.dtype,
        device=embeddings.device,
    )
    if num_tokens == 0 or num_heads == 0:
        return out
    emb_c = embeddings if embeddings.stride(-1) == 1 else embeddings.contiguous()
    nw = norm_weight.reshape(num_heads, head_dim)
    nw_c = nw if nw.stride(-1) == 1 else nw.contiguous()
    block_d = triton.next_power_of_2(head_dim)
    _hy4_engram_lookup_norm_flatten_kernel[(num_tokens, num_heads)](
        emb_c,
        nw_c,
        out,
        num_tokens,
        emb_c.stride(0),
        emb_c.stride(1),
        nw_c.stride(0),
        out.stride(0),
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
        ORIG_DTYPE=_torch_to_triton_dtype(emb_c.dtype),
        FP8_ROUND_TRIP=fp8_round_trip,
        num_warps=2 if block_d <= 128 else 4,
    )
    return out


@triton.jit(
    do_not_specialize=[
        "num_tokens",
        "inv_sqrt_d",
    ]
)
def _hy4_engram_gate_combine_kernel(
    kv_proj_ptr,
    hidden_ptr,
    norm_key_w_ptr,
    norm_query_w_ptr,
    pre_gamma_ptr,
    branch_scales_ptr,
    engram_mask_ptr,
    out_ptr,
    num_tokens,
    inv_sqrt_d,
    stride_kvt,
    stride_ht,
    stride_hh,
    stride_nkh,
    stride_nqh,
    stride_bsh,
    stride_ot,
    stride_oh,
    HC_MULT: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ORIG_DTYPE: tl.constexpr,
    ADD_TO_HIDDEN: tl.constexpr,
):
    """Fused Engram key/query/value RMSNorm + signed-sqrt sigmoid gate + branch combine."""
    tok = tl.program_id(0).to(tl.int64)
    hc = tl.program_id(1).to(tl.int64)
    if tok >= num_tokens:
        return

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < HIDDEN_SIZE
    out_base = out_ptr + tok * stride_ot + hc * stride_oh
    h_base = hidden_ptr + tok * stride_ht + hc * stride_hh

    active = tl.load(engram_mask_ptr + tok) != 0
    if not active:
        if ADD_TO_HIDDEN:
            h_orig = tl.load(h_base + offs_d, mask=mask_d, other=0.0).to(
                out_ptr.dtype.element_ty
            )
            tl.store(out_base + offs_d, h_orig, mask=mask_d)
        else:
            tl.store(
                out_base + offs_d,
                tl.zeros((BLOCK_D,), dtype=out_ptr.dtype.element_ty),
                mask=mask_d,
            )
        return

    kv_row = kv_proj_ptr + tok * stride_kvt
    k_raw = tl.load(kv_row + hc * HIDDEN_SIZE + offs_d, mask=mask_d, other=0.0).to(
        tl.float32
    )
    q_raw = tl.load(h_base + offs_d, mask=mask_d, other=0.0).to(tl.float32)

    fp32_eps: tl.constexpr = 1.1920928955078125e-07
    k_inv = tl.rsqrt(tl.sum(k_raw * k_raw, axis=0) / HIDDEN_SIZE + fp32_eps)
    q_inv = tl.rsqrt(tl.sum(q_raw * q_raw, axis=0) / HIDDEN_SIZE + fp32_eps)
    nk_w = tl.load(
        norm_key_w_ptr + hc * stride_nkh + offs_d, mask=mask_d, other=0.0
    ).to(ORIG_DTYPE)
    nq_w = tl.load(
        norm_query_w_ptr + hc * stride_nqh + offs_d, mask=mask_d, other=0.0
    ).to(ORIG_DTYPE)
    k_norm = ((k_raw * k_inv).to(ORIG_DTYPE) * nk_w).to(ORIG_DTYPE)
    q_norm = ((q_raw * q_inv).to(ORIG_DTYPE) * nq_w).to(ORIG_DTYPE)
    prod = (k_norm * q_norm).to(ORIG_DTYPE).to(tl.float32)
    gate_raw = tl.sum(prod, axis=0) * inv_sqrt_d
    gate_sign = tl.where(gate_raw > 0.0, 1.0, tl.where(gate_raw < 0.0, -1.0, 0.0))
    gate_val = tl.sigmoid(tl.sqrt(tl.maximum(tl.abs(gate_raw), 1e-6)) * gate_sign)

    v_raw = tl.load(kv_row + HC_MULT * HIDDEN_SIZE + offs_d, mask=mask_d, other=0.0).to(
        tl.float32
    )
    v_inv = tl.rsqrt(tl.sum(v_raw * v_raw, axis=0) / HIDDEN_SIZE + fp32_eps)
    pre_g = tl.load(pre_gamma_ptr + offs_d, mask=mask_d, other=0.0).to(ORIG_DTYPE)
    main_sig = (pre_g * (v_raw * v_inv).to(ORIG_DTYPE)).to(ORIG_DTYPE)
    b_scale = tl.load(
        branch_scales_ptr + hc * stride_bsh + offs_d, mask=mask_d, other=0.0
    ).to(ORIG_DTYPE)
    main_f = main_sig.to(tl.float32)
    engram_val = (
        gate_val * main_f + (main_sig * b_scale).to(ORIG_DTYPE).to(tl.float32)
    ).to(ORIG_DTYPE)
    if ADD_TO_HIDDEN:
        out_val = (q_raw + engram_val.to(tl.float32)).to(out_ptr.dtype.element_ty)
    else:
        out_val = engram_val.to(out_ptr.dtype.element_ty)
    tl.store(out_base + offs_d, out_val, mask=mask_d)


def hy4_engram_gate_combine(
    kv_proj_out: torch.Tensor,
    hidden_states: torch.Tensor,
    norm_key_weight: torch.Tensor,
    norm_query_weight: torch.Tensor,
    pre_gamma: torch.Tensor,
    branch_scales: torch.Tensor,
    engram_mask: torch.Tensor,
    *,
    inv_sqrt_d: float,
    add_to_hidden: bool = False,
) -> torch.Tensor:
    """Fused Engram key/query/value RMSNorm + gating + branch combine + mask."""
    num_tokens, hc_mult, hidden_size = hidden_states.shape
    out = torch.empty_like(hidden_states)
    if num_tokens == 0 or hc_mult == 0:
        return out
    kv_c = kv_proj_out if kv_proj_out.stride(-1) == 1 else kv_proj_out.contiguous()
    h_c = hidden_states if hidden_states.stride(-1) == 1 else hidden_states.contiguous()
    nk = norm_key_weight.reshape(hc_mult, hidden_size)
    nq = norm_query_weight.reshape(hc_mult, hidden_size)
    bs = branch_scales.reshape(hc_mult, hidden_size)
    nk_c = nk if nk.stride(-1) == 1 else nk.contiguous()
    nq_c = nq if nq.stride(-1) == 1 else nq.contiguous()
    bs_c = bs if bs.stride(-1) == 1 else bs.contiguous()
    pg_c = pre_gamma if pre_gamma.is_contiguous() else pre_gamma.contiguous()
    mask_c = engram_mask if engram_mask.is_contiguous() else engram_mask.contiguous()
    mask_u8 = (
        mask_c.view(torch.uint8)
        if mask_c.dtype == torch.bool
        else mask_c.to(torch.uint8)
    )
    block_d = triton.next_power_of_2(hidden_size)
    _hy4_engram_gate_combine_kernel[(num_tokens, hc_mult)](
        kv_c,
        h_c,
        nk_c,
        nq_c,
        pg_c,
        bs_c,
        mask_u8,
        out,
        num_tokens,
        float(inv_sqrt_d),
        kv_c.stride(0),
        h_c.stride(0),
        h_c.stride(1),
        nk_c.stride(0),
        nq_c.stride(0),
        bs_c.stride(0),
        out.stride(0),
        out.stride(1),
        HC_MULT=hc_mult,
        HIDDEN_SIZE=hidden_size,
        BLOCK_D=block_d,
        ORIG_DTYPE=_torch_to_triton_dtype(h_c.dtype),
        ADD_TO_HIDDEN=add_to_hidden,
        num_warps=8 if block_d >= 2048 else 4,
    )
    return out


@triton.jit(
    do_not_specialize=[
        "num_tokens",
        "num_reqs",
        "pad_slot",
        "cap",
        "pad_id",
        "image_token_id",
        "video_token_id",
        "stride_hist_slot",
        "stride_ng_t",
    ]
)
def _hy4_build_engram_inputs_history_kernel(
    input_ids_ptr,
    history_ptr,
    committed_ptr,
    req_slots_ptr,
    starts_ptr,
    active_reqs_ptr,
    out_ngrams_ptr,
    out_mask_ptr,
    num_tokens,
    num_reqs,
    pad_slot,
    cap,
    pad_id,
    image_token_id,
    video_token_id,
    stride_hist_slot,
    stride_ng_t,
    MAX_NGRAM: tl.constexpr,
):
    """Single-kernel Engram history update + n-gram lookback + active/special mask."""
    tok = tl.program_id(0).to(tl.int64)
    if tok >= num_tokens:
        return

    # Branchless binary search over starts[1..num_reqs] for the request owning tok.
    lo = tl.zeros((), dtype=tl.int64)
    hi = tl.zeros((), dtype=tl.int64) + (num_reqs - 1)
    for _ in tl.static_range(12):
        active_bs = lo < hi
        mid = (lo + hi) >> 1
        s_next = tl.load(starts_ptr + mid + 1, mask=active_bs, other=0).to(tl.int64)
        lo = tl.where(active_bs & (tok >= s_next), mid + 1, lo)
        hi = tl.where(active_bs & (tok < s_next), mid, hi)
    req_row = lo
    req_start = tl.load(starts_ptr + req_row).to(tl.int64)
    local_j = tok - req_start
    slot = tl.load(req_slots_ptr + req_row).to(tl.int64)
    active = tl.load(active_reqs_ptr + req_row) != 0
    hist_pos = tl.load(committed_ptr + slot).to(tl.int64) + local_j

    cur_id = tl.load(input_ids_ptr + tok).to(tl.int64)
    safe_slot = tl.where(active, slot, pad_slot)
    clamped_w = tl.minimum(tl.maximum(hist_pos, 0), cap - 1)
    safe_write_pos = tl.where(active, clamped_w, 0)
    tl.store(
        history_ptr + safe_slot * stride_hist_slot + safe_write_pos,
        cur_id.to(history_ptr.dtype.element_ty),
    )

    ng_base = out_ngrams_ptr + tok * stride_ng_t
    tl.store(ng_base + 0, tl.where(active, cur_id, pad_id))
    for offset in tl.static_range(1, MAX_NGRAM):
        # Within the current batch, read directly from read-only input_ids to
        # avoid cross-CTA read-after-write hazards on history_ptr.
        has_prev = active & (hist_pos >= offset)
        in_batch = has_prev & (local_j >= offset)
        from_hist = has_prev & (local_j < offset)
        batch_tok = tl.load(
            input_ids_ptr + tl.where(in_batch, tok - offset, 0),
            mask=in_batch,
            other=pad_id,
        ).to(tl.int64)
        rpos = tl.minimum(tl.maximum(hist_pos - offset, 0), cap - 1)
        hist_tok = tl.load(
            history_ptr + safe_slot * stride_hist_slot + rpos,
            mask=from_hist,
            other=pad_id,
        ).to(tl.int64)
        prev_tok = tl.where(in_batch, batch_tok, tl.where(from_hist, hist_tok, pad_id))
        tl.store(ng_base + offset, prev_tok)

    mask_ok = (
        active
        & (cur_id != image_token_id)
        & (cur_id != video_token_id)
        & (cur_id != pad_id)
    )
    tl.store(out_mask_ptr + tok, mask_ok.to(tl.uint8))


def hy4_build_engram_inputs_from_history(
    input_ids: torch.Tensor,
    history_token_ids: torch.Tensor,
    committed_lengths: torch.Tensor,
    req_pool_indices: torch.Tensor,
    input_start_offsets: torch.Tensor,
    active_request_mask: torch.Tensor,
    *,
    max_ngram_size: int,
    pad_id: int,
    image_token_id: int,
    video_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Graph-safe single-kernel Engram history append + n-gram gather + mask."""
    num_tokens = input_ids.shape[0]
    num_reqs = req_pool_indices.shape[0]
    device = input_ids.device
    out_ngrams = torch.empty(
        (num_tokens, max_ngram_size), dtype=torch.int64, device=device
    )
    out_mask_u8 = torch.empty((num_tokens,), dtype=torch.uint8, device=device)
    if num_tokens == 0 or num_reqs == 0:
        out_mask_u8.zero_()
        return out_ngrams, out_mask_u8.view(torch.bool)
    if not input_ids.is_cuda or num_reqs > 4096:
        raise ValueError(
            "hy4_build_engram_inputs_from_history requires CUDA and bs <= 4096"
        )

    ids_c = input_ids if input_ids.is_contiguous() else input_ids.contiguous()
    comm_c = (
        committed_lengths
        if committed_lengths.is_contiguous()
        else committed_lengths.contiguous()
    )
    slots_c = (
        req_pool_indices
        if req_pool_indices.is_contiguous()
        else req_pool_indices.contiguous()
    )
    starts_c = (
        input_start_offsets
        if input_start_offsets.is_contiguous()
        else input_start_offsets.contiguous()
    )
    act_c = (
        active_request_mask
        if active_request_mask.is_contiguous()
        else active_request_mask.contiguous()
    )
    act_u8 = (
        act_c.view(torch.uint8) if act_c.dtype == torch.bool else act_c.to(torch.uint8)
    )
    _hy4_build_engram_inputs_history_kernel[(num_tokens,)](
        ids_c,
        history_token_ids,
        comm_c,
        slots_c,
        starts_c,
        act_u8,
        out_ngrams,
        out_mask_u8,
        num_tokens,
        num_reqs,
        history_token_ids.shape[0] - 1,
        history_token_ids.shape[1],
        int(pad_id),
        int(image_token_id),
        int(video_token_id),
        history_token_ids.stride(0),
        out_ngrams.stride(0),
        MAX_NGRAM=max_ngram_size,
        num_warps=1,
    )
    return out_ngrams, out_mask_u8.view(torch.bool)


__all__ = [
    "hy4_engram_hash_ngrams",
    "hy4_engram_hash_and_embed",
    "hy4_engram_lookup_norm_flatten",
    "hy4_engram_gate_combine",
    "hy4_build_engram_inputs_from_history",
]
