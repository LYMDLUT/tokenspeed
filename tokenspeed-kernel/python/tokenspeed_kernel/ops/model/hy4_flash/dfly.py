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


"""Hy4-flash DFly2 speculative drafter fused Triton kernels."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import pdl_enabled


@triton.jit(do_not_specialize=["num_rows", "num_heads"])
def _hy4_merge_attention_and_sink_kernel(
    out_a_ptr,
    lse_a_ptr,
    out_b_ptr,
    lse_b_ptr,
    sink_ptr,
    out_ptr,
    num_rows,
    num_heads,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_SINK: tl.constexpr,
):
    """Single-pass online-softmax merge of two partitions with optional sink rescale."""
    row = tl.program_id(0).to(tl.int64)
    if row >= num_rows:
        return

    la = tl.load(lse_a_ptr + row).to(tl.float32)
    lb = tl.load(lse_b_ptr + row).to(tl.float32)
    max_lse = tl.maximum(la, lb)
    both_empty = max_lse == -float("inf")
    safe_max = tl.where(both_empty, 0.0, max_lse)

    exp_a = tl.where(la == -float("inf"), 0.0, tl.exp(la - safe_max))
    exp_b = tl.where(lb == -float("inf"), 0.0, tl.exp(lb - safe_max))
    denom = exp_a + exp_b
    safe_denom = tl.where(both_empty, 1.0, denom)
    w_a = exp_a / safe_denom
    w_b = exp_b / safe_denom

    if HAS_SINK:
        merged_lse = tl.where(both_empty, -float("inf"), safe_max + tl.log(safe_denom))
        sink_v = tl.load(sink_ptr + (row % num_heads)).to(tl.float32)
        rescale = tl.sigmoid(merged_lse - sink_v)
        w_a = w_a * rescale
        w_b = w_b * rescale

    offs = tl.arange(0, BLOCK_D)
    mask = offs < HEAD_DIM
    base = row * HEAD_DIM + offs
    oa = tl.load(out_a_ptr + base, mask=mask, other=0.0).to(tl.float32)
    ob = tl.load(out_b_ptr + base, mask=mask, other=0.0).to(tl.float32)
    merged = (w_a * oa + w_b * ob).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + base, merged, mask=mask)


def hy4_merge_attention_and_sink(
    out_a: torch.Tensor,
    lse_a: torch.Tensor,
    out_b: torch.Tensor,
    lse_b: torch.Tensor,
    sink: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused online-softmax merge of ``(out_a, lse_a)`` + ``(out_b, lse_b)`` with optional sink.

    Args:
        out_a: ``[B, S, H, D]`` first attention output partition.
        lse_a: ``[B, S, H]`` FP32 natural-log LSE for partition A (may be ``-inf``).
        out_b: ``[B, S, H, D]`` second attention output partition.
        lse_b: ``[B, S, H]`` FP32 natural-log LSE for partition B (may be ``-inf``).
        sink: Optional ``[H]`` learnable attention sink vector.

    Returns:
        Merged (and optionally sink-rescaled) tensor matching ``out_b.shape`` and ``out_b.dtype``.
    """
    head_dim = out_b.shape[-1]
    num_heads = out_b.shape[-2]
    num_rows = out_b.numel() // head_dim
    out = torch.empty_like(out_b)
    if num_rows == 0:
        return out
    if not out_b.is_cuda:
        la = lse_a.float()
        lb = lse_b.float()
        max_lse = torch.maximum(la, lb)
        both_empty = torch.isneginf(max_lse)
        safe_max = torch.where(both_empty, torch.zeros_like(max_lse), max_lse)
        exp_a = torch.where(
            torch.isneginf(la), torch.zeros_like(la), torch.exp(la - safe_max)
        )
        exp_b = torch.where(
            torch.isneginf(lb), torch.zeros_like(lb), torch.exp(lb - safe_max)
        )
        denom = exp_a + exp_b
        safe_denom = torch.where(both_empty, torch.ones_like(denom), denom)
        w_a = (exp_a / safe_denom).unsqueeze(-1)
        w_b = (exp_b / safe_denom).unsqueeze(-1)
        merged = (w_a * out_a.float() + w_b * out_b.float()).to(out_b.dtype)
        if sink is not None:
            merged_lse = torch.where(
                both_empty,
                torch.full_like(max_lse, float("-inf")),
                safe_max + torch.log(safe_denom),
            )
            rescale = torch.sigmoid(merged_lse - sink.float().view(1, 1, -1)).unsqueeze(
                -1
            )
            merged = (merged.float() * rescale).to(out_b.dtype)
        out.copy_(merged)
        return out

    oa_c = out_a.contiguous()
    ob_c = out_b.contiguous()
    la_c = lse_a.float().contiguous()
    lb_c = lse_b.float().contiguous()
    sink_c = sink.contiguous() if sink is not None else lb_c
    block_d = triton.next_power_of_2(head_dim)
    _hy4_merge_attention_and_sink_kernel[(num_rows,)](
        oa_c,
        la_c,
        ob_c,
        lb_c,
        sink_c,
        out,
        num_rows,
        num_heads,
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
        HAS_SINK=sink is not None,
        num_warps=4 if block_d <= 512 else 8,
    )
    return out


@triton.jit(
    do_not_specialize=[
        "bs",
        "max_hidden_row",
        "max_pos_row",
        "pos_stride_axis",
    ]
)
def _hy4_dfly_build_decode_round_inputs_kernel(
    output_tokens_ptr,
    accept_lengths_ptr,
    seq_lens_ptr,
    pos_buf_ptr,
    anchor_out_ptr,
    last_rows_out_ptr,
    ctx_lens_out_ptr,
    block_pos_out_ptr,
    bs,
    max_hidden_row,
    max_pos_row,
    pos_stride_axis,
    VERIFY_WIDTH: tl.constexpr,
    IS_MROPE: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """Build pure-decode DFly2 round metadata and 3D block positions in one launch."""
    b = tl.program_id(0).to(tl.int64)
    if b >= bs:
        return
    acc = tl.load(accept_lengths_ptr + b).to(tl.int64)
    acc = tl.minimum(tl.maximum(acc, 1), VERIFY_WIDTH)
    row = b * VERIFY_WIDTH + (acc - 1)
    safe_row = tl.minimum(tl.maximum(row, 0), max_hidden_row)
    pos_row = tl.minimum(safe_row, max_pos_row)

    tok = tl.load(output_tokens_ptr + row).to(tl.int32)
    seq_len = tl.load(seq_lens_ptr + b).to(tl.int32)
    ctx_len = tl.maximum(seq_len - VERIFY_WIDTH + acc.to(tl.int32), 0)

    tl.store(anchor_out_ptr + b, tok)
    tl.store(last_rows_out_ptr + b, safe_row)
    tl.store(ctx_lens_out_ptr + b, ctx_len)

    w = tl.arange(0, BLOCK_W)
    w_mask = w < VERIFY_WIDTH
    offsets = (w + 1).to(tl.int64)
    if IS_MROPE:
        for axis in tl.static_range(0, 3):
            base_p = tl.load(pos_buf_ptr + axis * pos_stride_axis + pos_row).to(
                tl.int64
            )
            dst = (axis * bs + b) * VERIFY_WIDTH + w
            tl.store(block_pos_out_ptr + dst, base_p + offsets, mask=w_mask)
    else:
        base_p = tl.load(pos_buf_ptr + pos_row).to(tl.int64)
        vals = base_p + offsets
        for axis in tl.static_range(0, 3):
            dst = (axis * bs + b) * VERIFY_WIDTH + w
            tl.store(block_pos_out_ptr + dst, vals, mask=w_mask)


def hy4_dfly_build_decode_round_inputs(
    output_tokens: torch.Tensor,
    accept_lengths: torch.Tensor,
    seq_lens: torch.Tensor,
    pos_buf: torch.Tensor,
    *,
    bs: int,
    verify_width: int,
    hidden_rows: int,
    is_mrope: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused pure-decode DFly2 round inputs: ``(anchor, safe_rows, ctx_lens, block_pos)``."""
    device = output_tokens.device
    anchor_tokens = torch.empty((bs,), dtype=torch.int32, device=device)
    safe_rows = torch.empty((bs,), dtype=torch.int64, device=device)
    ctx_lens = torch.empty((bs,), dtype=torch.int32, device=device)
    block_positions = torch.empty(
        (3, bs, verify_width), dtype=torch.int64, device=device
    )
    if bs == 0:
        return anchor_tokens, safe_rows, ctx_lens, block_positions
    max_hidden_row = max(hidden_rows - 1, 0)
    max_pos_row = max((pos_buf.shape[1] if is_mrope else pos_buf.shape[0]) - 1, 0)
    pos_stride_axis = pos_buf.stride(0) if is_mrope else 0
    _hy4_dfly_build_decode_round_inputs_kernel[(bs,)](
        output_tokens,
        accept_lengths,
        seq_lens,
        pos_buf,
        anchor_tokens,
        safe_rows,
        ctx_lens,
        block_positions,
        bs,
        max_hidden_row,
        max_pos_row,
        pos_stride_axis,
        VERIFY_WIDTH=verify_width,
        IS_MROPE=is_mrope,
        BLOCK_W=triton.next_power_of_2(verify_width),
        num_warps=1,
    )
    return anchor_tokens, safe_rows, ctx_lens, block_positions


@triton.jit(do_not_specialize=["vocab_size", "vocab_offset", "stride_lb", "stride_pb"])
def _hy4_vocab_argmax_stage1_kernel(
    logits_ptr,
    part_val_ptr,
    part_idx_ptr,
    vocab_size,
    vocab_offset,
    stride_lb,
    stride_pb,
    NUM_SPLITS: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Parallel split-V stage 1 argmax over ``[B, V]`` FP32/BF16 logits."""
    b = tl.program_id(0).to(tl.int64)
    s = tl.program_id(1).to(tl.int64)
    chunk = tl.cdiv(vocab_size, NUM_SPLITS)
    start = s * chunk
    end = tl.minimum(start + chunk, vocab_size)
    best_v = -float("inf")
    best_i = tl.zeros((), dtype=tl.int32)
    offs = tl.arange(0, BLOCK_V)
    row_base = logits_ptr + b * stride_lb
    for pos in range(start, end, BLOCK_V):
        cols = pos + offs
        mask = cols < end
        vals = tl.load(row_base + cols, mask=mask, other=-float("inf")).to(tl.float32)
        cur_v = tl.max(vals, axis=0)
        cur_i = (pos + tl.argmax(vals, axis=0)).to(tl.int32)
        better = cur_v > best_v
        best_v = tl.where(better, cur_v, best_v)
        best_i = tl.where(better, cur_i, best_i)
    tl.store(part_val_ptr + b * stride_pb + s, best_v)
    tl.store(part_idx_ptr + b * stride_pb + s, best_i + vocab_offset.to(tl.int32))


@triton.jit(do_not_specialize=["stride_pb", "stride_out", "max_token_id"])
def _hy4_vocab_argmax_stage2_kernel(
    part_val_ptr,
    part_idx_ptr,
    out_ptr,
    out_contig_ptr,
    stride_pb,
    stride_out,
    max_token_id,
    PACK_KEY: tl.constexpr,
    CLAMP_MAX: tl.constexpr,
    WRITE_CONTIG: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    """Reduce ``NUM_SPLITS`` partial argmax results into an int32 token or packed int64 key."""
    b = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, NUM_SPLITS)
    vals = tl.load(part_val_ptr + b * stride_pb + offs)
    idxs = tl.load(part_idx_ptr + b * stride_pb + offs)
    win = tl.argmax(vals, axis=0)
    win_mask = offs == win
    best_v = tl.sum(tl.where(win_mask, vals, 0.0), axis=0)
    best_i = tl.sum(tl.where(win_mask, idxs, 0), axis=0).to(tl.int32)
    if PACK_KEY:
        # Pack (ordered_f32_bits << 32) | (0xFFFFFFFF - global_idx) as bitcast int64
        # so unsigned 64-bit max breaks value ties toward the smallest token index.
        norm_v = tl.where(best_v == 0.0, 0.0, best_v)
        v_bits = norm_v.to(tl.uint32, bitcast=True)
        sign = tl.full((), 0x80000000, tl.uint32)
        full = tl.full((), 0xFFFFFFFF, tl.uint32)
        ordered = v_bits ^ tl.where((v_bits & sign) != 0, full, sign)
        inv_idx = (full - best_i.to(tl.uint32)).to(tl.uint64)
        packed = ((ordered.to(tl.uint64) << 32) | inv_idx).to(tl.int64, bitcast=True)
        tl.store(out_ptr + b * stride_out, packed)
    else:
        if CLAMP_MAX:
            best_i = tl.minimum(tl.maximum(best_i, 0), max_token_id)
        tl.store(out_ptr + b * stride_out, best_i)
        if WRITE_CONTIG:
            tl.store(out_contig_ptr + b, best_i)


def _select_argmax_splits(batch_size: int) -> int:
    if batch_size <= 4:
        return 64
    if batch_size <= 16:
        return 32
    return 16


def hy4_vocab_argmax_into(
    logits: torch.Tensor,
    out: torch.Tensor,
    *,
    out_contig: torch.Tensor | None = None,
    vocab_offset: int = 0,
    max_token_id: int | None = None,
) -> torch.Tensor:
    """Compute multi-CTA argmax of ``[B, V]`` logits directly into 1D ``int32`` ``out``."""
    bsz, vocab_size = logits.shape
    if bsz == 0:
        return out
    if not logits.is_cuda or logits.stride(-1) != 1 or out.dtype != torch.int32:
        tok = (logits.argmax(dim=-1) + vocab_offset).to(torch.int32)
        if max_token_id is not None:
            tok = tok.clamp(0, int(max_token_id))
        out.copy_(tok)
        if out_contig is not None:
            out_contig.copy_(tok)
        return out
    num_splits = _select_argmax_splits(bsz)
    part_val = torch.empty((bsz, num_splits), dtype=torch.float32, device=logits.device)
    part_idx = torch.empty((bsz, num_splits), dtype=torch.int32, device=logits.device)
    _hy4_vocab_argmax_stage1_kernel[(bsz, num_splits)](
        logits,
        part_val,
        part_idx,
        vocab_size,
        int(vocab_offset),
        logits.stride(0),
        part_val.stride(0),
        NUM_SPLITS=num_splits,
        BLOCK_V=1024,
        num_warps=4,
    )
    _hy4_vocab_argmax_stage2_kernel[(bsz,)](
        part_val,
        part_idx,
        out,
        out_contig if out_contig is not None else out,
        part_val.stride(0),
        out.stride(0),
        int(max_token_id) if max_token_id is not None else 0,
        PACK_KEY=False,
        CLAMP_MAX=max_token_id is not None,
        WRITE_CONTIG=out_contig is not None,
        NUM_SPLITS=num_splits,
        num_warps=2,
    )
    return out


@triton.jit(do_not_specialize=["stride_ab", "stride_bb", "stride_ob"])
def _hy4_dual_rmsnorm_cat_kernel(
    a_ptr,
    b_ptr,
    wa_ptr,
    wb_ptr,
    out_ptr,
    stride_ab,
    stride_bb,
    stride_ob,
    eps: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ENABLE_PDL: tl.constexpr = False,
):
    """Fused ``cat([rmsnorm(a, wa), rmsnorm(b, wb)], dim=-1)`` in a single launch."""
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
    row = tl.program_id(0).to(tl.int64)
    branch = tl.program_id(1)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < HIDDEN_SIZE
    if branch == 0:
        x = tl.load(a_ptr + row * stride_ab + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(wa_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        inv = tl.rsqrt(tl.sum(x * x, axis=0) / HIDDEN_SIZE + eps)
        y = ((x * inv) * w).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + row * stride_ob + offs, y, mask=mask)
    else:
        x = tl.load(b_ptr + row * stride_bb + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(wb_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        inv = tl.rsqrt(tl.sum(x * x, axis=0) / HIDDEN_SIZE + eps)
        y = ((x * inv) * w).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + row * stride_ob + HIDDEN_SIZE + offs, y, mask=mask)
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def hy4_dual_rmsnorm_cat(
    a: torch.Tensor,
    b: torch.Tensor,
    weight_a: torch.Tensor,
    weight_b: torch.Tensor,
    *,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Compute ``torch.cat((rmsnorm(a, weight_a), rmsnorm(b, weight_b)), dim=-1)``."""
    bsz, hidden_size = a.shape
    out = torch.empty((bsz, 2 * hidden_size), dtype=a.dtype, device=a.device)
    if bsz == 0:
        return out
    if not a.is_cuda or hidden_size > 8192 or a.stride(-1) != 1 or b.stride(-1) != 1:
        af = a.float()
        bf = b.float()
        an = (
            af
            * torch.rsqrt(af.pow(2).mean(dim=-1, keepdim=True) + eps)
            * weight_a.float()
        ).to(a.dtype)
        bn = (
            bf
            * torch.rsqrt(bf.pow(2).mean(dim=-1, keepdim=True) + eps)
            * weight_b.float()
        ).to(a.dtype)
        return torch.cat((an, bn), dim=-1)
    block_d = triton.next_power_of_2(hidden_size)
    enable_pdl = pdl_enabled()
    _hy4_dual_rmsnorm_cat_kernel[(bsz, 2)](
        a,
        b,
        weight_a,
        weight_b,
        out,
        a.stride(0),
        b.stride(0),
        out.stride(0),
        eps=float(eps),
        HIDDEN_SIZE=hidden_size,
        BLOCK_D=block_d,
        ENABLE_PDL=enable_pdl,
        num_warps=8 if block_d >= 2048 else 4,
        **({"launch_pdl": True} if enable_pdl else {}),
    )
    return out


@triton.jit(
    do_not_specialize=[
        "num_heads",
        "block_size",
        "scaling",
        "stride_qb",
        "stride_lb",
        "stride_ocb",
        "stride_lcb",
        "stride_ob",
    ]
)
def _hy4_dfly_block_attn_merge_sink_kernel(
    q_ptr,
    latent_ptr,
    out_ctx_ptr,
    lse_ctx_ptr,
    ctx_lens_ptr,
    sink_ptr,
    out_ptr,
    num_heads,
    block_size,
    scaling,
    stride_qb,
    stride_qs,
    stride_qh,
    stride_lb,
    stride_ls,
    stride_ocb,
    stride_ocs,
    stride_och,
    stride_lcb,
    stride_lcs,
    stride_lch,
    stride_ob,
    stride_os,
    stride_oh,
    D_QK: tl.constexpr,
    D_V: tl.constexpr,
    BLOCK_S: tl.constexpr,
    CAUSAL: tl.constexpr,
    HAS_SINK: tl.constexpr,
    HAS_CTX_LENS: tl.constexpr,
):
    """Fused DFly2 intra-block absorbed attention + context merge + attention sink."""
    pid = tl.program_id(0).to(tl.int64)
    b = pid // num_heads
    h = pid % num_heads

    offs_s = tl.arange(0, BLOCK_S)
    mask_s = offs_s < block_size

    q_base = q_ptr + b * stride_qb + h * stride_qh
    lat_base = latent_ptr + b * stride_lb

    scores = tl.zeros((BLOCK_S, BLOCK_S), dtype=tl.float32)
    for k0 in tl.static_range(0, D_QK, 64):
        offs_k = k0 + tl.arange(0, 64)
        mask_k = offs_k < D_QK
        q_blk = tl.load(
            q_base + offs_s[:, None] * stride_qs + offs_k[None, :],
            mask=mask_s[:, None] & mask_k[None, :],
            other=0.0,
        )
        k_blk = tl.load(
            lat_base + offs_s[:, None] * stride_ls + offs_k[None, :],
            mask=mask_s[:, None] & mask_k[None, :],
            other=0.0,
        )
        scores = tl.dot(q_blk, tl.trans(k_blk), acc=scores, out_dtype=tl.float32)

    scores = scores * scaling
    valid_t = offs_s[None, :] < block_size
    if CAUSAL:
        valid_st = valid_t & (offs_s[None, :] <= offs_s[:, None])
    else:
        valid_st = valid_t
    masked_scores = tl.where(valid_st, scores, -float("inf"))
    row_max = tl.max(masked_scores, axis=1)
    safe_row_max = tl.where(row_max == -float("inf"), 0.0, row_max)
    exp_s = tl.where(valid_st, tl.exp(masked_scores - safe_row_max[:, None]), 0.0)
    row_sum = tl.sum(exp_s, axis=1)
    safe_row_sum = tl.where(row_sum == 0.0, 1.0, row_sum)
    lb = tl.where(row_sum == 0.0, -float("inf"), safe_row_max + tl.log(safe_row_sum))
    probs = exp_s / safe_row_sum[:, None]

    has_ctx = True
    if HAS_CTX_LENS:
        has_ctx = tl.load(ctx_lens_ptr + b) > 0

    la = tl.load(
        lse_ctx_ptr + b * stride_lcb + offs_s * stride_lcs + h * stride_lch,
        mask=mask_s & has_ctx,
        other=-float("inf"),
    ).to(tl.float32)
    max_lse = tl.maximum(la, lb)
    both_empty = max_lse == -float("inf")
    s_max = tl.where(both_empty, 0.0, max_lse)
    exp_a = tl.where(la == -float("inf"), 0.0, tl.exp(la - s_max))
    exp_b = tl.where(lb == -float("inf"), 0.0, tl.exp(lb - s_max))
    denom = exp_a + exp_b
    s_denom = tl.where(both_empty, 1.0, denom)
    w_a = exp_a / s_denom
    w_b = exp_b / s_denom

    if HAS_SINK:
        merged_lse = tl.where(both_empty, -float("inf"), s_max + tl.log(s_denom))
        sink_v = tl.load(sink_ptr + h).to(tl.float32)
        rescale = tl.sigmoid(merged_lse - sink_v)
        w_a = w_a * rescale
        w_b = w_b * rescale

    oc_base = out_ctx_ptr + b * stride_ocb + h * stride_och
    o_base = out_ptr + b * stride_ob + h * stride_oh
    for d0 in tl.static_range(0, D_V, 128):
        offs_d = d0 + tl.arange(0, 128)
        mask_d = offs_d < D_V
        v_blk = tl.load(
            lat_base + offs_s[:, None] * stride_ls + offs_d[None, :],
            mask=mask_s[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        out_draft_blk = tl.dot(
            probs, v_blk, input_precision="ieee", out_dtype=tl.float32
        )
        out_ctx_blk = tl.load(
            oc_base + offs_s[:, None] * stride_ocs + offs_d[None, :],
            mask=(mask_s[:, None] & mask_d[None, :]) & has_ctx,
            other=0.0,
        ).to(tl.float32)
        merged_blk = (w_a[:, None] * out_ctx_blk + w_b[:, None] * out_draft_blk).to(
            out_ptr.dtype.element_ty
        )
        tl.store(
            o_base + offs_s[:, None] * stride_os + offs_d[None, :],
            merged_blk,
            mask=mask_s[:, None] & mask_d[None, :],
        )


def hy4_dfly_block_attn_merge_sink(
    q_absorbed: torch.Tensor,
    latent_draft: torch.Tensor,
    out_ctx: torch.Tensor,
    lse_ctx: torch.Tensor,
    sink: torch.Tensor | None,
    *,
    scaling: float,
    causal: bool = True,
    ctx_lens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused DFly2 intra-block absorbed attention + context merge + attention sink.

    Replaces ``_attend_draft_block_absorbed`` (~12 PyTorch ops) plus
    ``hy4_merge_attention_and_sink`` with a single Triton launch when
    ``block_size <= 32`` on CUDA. When ``ctx_lens`` is supplied, requests with
    ``ctx_lens[b] <= 0`` skip reading ``out_ctx``/``lse_ctx`` and treat context
    LSE as ``-inf`` with zero extra kernels.
    """
    bsz, block_size, num_heads, d_qk = q_absorbed.shape
    d_v = out_ctx.shape[-1]
    out = torch.empty(
        (bsz, block_size, num_heads, d_v),
        dtype=q_absorbed.dtype,
        device=q_absorbed.device,
    )
    if bsz == 0 or block_size == 0 or num_heads == 0:
        return out
    if (
        not q_absorbed.is_cuda
        or block_size > 32
        or q_absorbed.dtype not in (torch.bfloat16, torch.float16)
        or q_absorbed.stride(-1) != 1
        or latent_draft.stride(-1) != 1
        or out_ctx.stride(-1) != 1
    ):
        if ctx_lens is not None:
            zero_ctx = (ctx_lens <= 0).view(bsz, 1, 1)
            out_ctx = torch.where(
                zero_ctx.unsqueeze(-1), torch.zeros_like(out_ctx), out_ctx
            )
            lse_ctx = torch.where(
                zero_ctx,
                torch.full_like(lse_ctx.float(), float("-inf")),
                lse_ctx.float(),
            )
        scores = (
            torch.einsum("bshd,btd->bhst", q_absorbed.float(), latent_draft.float())
            * scaling
        )
        if causal:
            q_idx = torch.arange(block_size, device=scores.device).unsqueeze(1)
            k_idx = torch.arange(block_size, device=scores.device).unsqueeze(0)
            scores = scores.masked_fill(
                (k_idx > q_idx).unsqueeze(0).unsqueeze(0), float("-inf")
            )
        lse_draft = torch.logsumexp(scores, dim=-1).transpose(1, 2).contiguous()
        probs = torch.softmax(scores, dim=-1)
        out_draft = torch.einsum(
            "bhst,btm->bshm", probs, latent_draft[..., :d_v].float()
        ).to(q_absorbed.dtype)
        return hy4_merge_attention_and_sink(
            out_ctx, lse_ctx, out_draft, lse_draft, sink
        )

    lse_c = lse_ctx if lse_ctx.dtype == torch.float32 else lse_ctx.float()
    ctx_lens_c = (
        (ctx_lens if ctx_lens.is_contiguous() else ctx_lens.contiguous())
        if ctx_lens is not None
        else lse_c
    )
    sink_c = sink.contiguous() if sink is not None else lse_c
    block_s = max(16, triton.next_power_of_2(block_size))
    _hy4_dfly_block_attn_merge_sink_kernel[(bsz * num_heads,)](
        q_absorbed,
        latent_draft,
        out_ctx,
        lse_c,
        ctx_lens_c,
        sink_c,
        out,
        num_heads,
        block_size,
        float(scaling),
        q_absorbed.stride(0),
        q_absorbed.stride(1),
        q_absorbed.stride(2),
        latent_draft.stride(0),
        latent_draft.stride(1),
        out_ctx.stride(0),
        out_ctx.stride(1),
        out_ctx.stride(2),
        lse_c.stride(0),
        lse_c.stride(1),
        lse_c.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        D_QK=d_qk,
        D_V=d_v,
        BLOCK_S=block_s,
        CAUSAL=causal,
        HAS_SINK=sink is not None,
        HAS_CTX_LENS=ctx_lens is not None,
        num_warps=4,
    )
    return out


@triton.jit(
    do_not_specialize=[
        "bsz",
        "max_ctx_k",
        "stride_slots_b",
        "softmax_scale",
    ]
)
def _hy4_dfly_shared_kv_dsa_kernel(
    q_ptr,
    kv_ptr,
    ctx_slots_ptr,
    ctx_lens_ptr,
    out_ptr,
    partial_out_ptr,
    partial_lse_ptr,
    lse_ptr,
    bsz,
    max_ctx_k,
    stride_qb,
    stride_qs,
    stride_qh,
    stride_slots_b,
    stride_kv_page,
    stride_kv_tok,
    stride_kv_c,
    stride_ob,
    stride_os,
    stride_oh,
    stride_lb,
    stride_ls,
    stride_lh,
    softmax_scale,
    Q_LEN: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    KV_LORA_RANK: tl.constexpr,
    QK_ROPE_HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    Q_GROUP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ENABLE_PDL: tl.constexpr = False,
):
    """Zero-copy DFly context sparse attention sharing per-request ctx_slots across Q_GROUP."""
    cur_b = tl.program_id(0).to(tl.int64)
    q_blk = tl.program_id(1).to(tl.int64)
    pid2 = tl.program_id(2)
    split = pid2 % NUM_SPLITS
    h_blk = pid2 // NUM_SPLITS
    if cur_b >= bsz:
        return
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()

    m_offs = tl.arange(0, BLOCK_M)
    q_local = m_offs // BLOCK_H
    h_local = m_offs % BLOCK_H
    cur_q = q_blk * Q_GROUP + q_local
    heads = h_blk * BLOCK_H + h_local
    m_mask = (q_local < Q_GROUP) & (cur_q < Q_LEN) & (heads < NUM_HEADS)

    q_safe = tl.minimum(cur_q, Q_LEN - 1).to(tl.int64)
    h_safe = tl.minimum(heads, NUM_HEADS - 1).to(tl.int64)
    q_base = (
        cur_b * stride_qb + q_safe[:, None] * stride_qs + h_safe[:, None] * stride_qh
    )

    dims = tl.arange(0, KV_LORA_RANK)
    q_nope = tl.load(q_ptr + q_base + dims[None, :], mask=m_mask[:, None], other=0.0)
    dot_dtype = q_nope.dtype
    if QK_ROPE_HEAD_DIM > 0:
        rope_dims = tl.arange(0, QK_ROPE_HEAD_DIM)
        q_rope = tl.load(
            q_ptr + q_base + KV_LORA_RANK + rope_dims[None, :],
            mask=m_mask[:, None],
            other=0.0,
        )

    valid_len = tl.load(ctx_lens_ptr + cur_b).to(tl.int32)
    valid_len = tl.minimum(tl.maximum(valid_len, 0), max_ctx_k)
    split_len = tl.cdiv(tl.cdiv(valid_len, NUM_SPLITS), BLOCK_N) * BLOCK_N
    start = split * split_len
    end = tl.minimum(start + split_len, valid_len)

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, KV_LORA_RANK], tl.float32)
    scale_log2 = softmax_scale * 1.4426950408889634
    n_offs = tl.arange(0, BLOCK_N)

    for block_start in range(start, end, BLOCK_N):
        cols = block_start + n_offs
        slots = tl.load(
            ctx_slots_ptr + cur_b * stride_slots_b + cols,
            mask=cols < end,
            other=-1,
        ).to(tl.int64)
        valid = slots >= 0
        kv_rows = (slots // PAGE_SIZE) * stride_kv_page + (
            slots % PAGE_SIZE
        ) * stride_kv_tok
        latent = tl.load(
            kv_ptr + kv_rows[:, None] + dims[None, :] * stride_kv_c,
            mask=valid[:, None],
            other=0.0,
        ).to(dot_dtype)
        scores = tl.dot(q_nope, tl.trans(latent))
        if QK_ROPE_HEAD_DIM > 0:
            k_rope = tl.load(
                kv_ptr
                + kv_rows[:, None]
                + (KV_LORA_RANK + rope_dims[None, :]) * stride_kv_c,
                mask=valid[:, None],
                other=0.0,
            ).to(dot_dtype)
            scores = tl.dot(q_rope, tl.trans(k_rope), scores)
        scores = tl.where(valid[None, :], scores * scale_log2, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        m_off = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.where(m_i == -float("inf"), 0.0, tl.exp2(m_i - m_off))
        p = tl.exp2(scores - m_off[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = tl.dot(p.to(dot_dtype), latent, acc * alpha[:, None])
        m_i = m_new

    res = tl.where(l_i[:, None] > 0.0, acc / l_i[:, None], 0.0)
    if NUM_SPLITS == 1:
        lse_nat = tl.where(
            l_i > 0.0, (m_i + tl.log2(l_i)) * 0.6931471805599453, -float("inf")
        )
        o_base = (
            cur_b * stride_ob
            + q_safe[:, None] * stride_os
            + h_safe[:, None] * stride_oh
        )
        tl.store(
            out_ptr + o_base + dims[None, :],
            res.to(out_ptr.dtype.element_ty),
            mask=m_mask[:, None],
        )
        tl.store(
            lse_ptr + cur_b * stride_lb + q_safe * stride_ls + h_safe * stride_lh,
            lse_nat,
            mask=m_mask,
        )
    else:
        row_idx = (cur_b * Q_LEN + q_safe) * NUM_HEADS + h_safe
        p_row = row_idx * NUM_SPLITS + split
        tl.store(
            partial_out_ptr + p_row[:, None] * KV_LORA_RANK + dims[None, :],
            res,
            mask=m_mask[:, None],
        )
        lse_b2 = tl.where(l_i > 0.0, m_i + tl.log2(l_i), -float("inf"))
        tl.store(partial_lse_ptr + p_row, lse_b2, mask=m_mask)
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _hy4_dfly_shared_kv_dsa_merge_kernel(
    partial_out_ptr,
    partial_lse_ptr,
    out_ptr,
    lse_ptr,
    KV_LORA_RANK: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ENABLE_PDL: tl.constexpr = False,
):
    """Merge split-K partial outputs and natural-log LSE for DFly context DSA."""
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
    row = tl.program_id(0).to(tl.int64)
    dims = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    s_rows = row * NUM_SPLITS + tl.arange(0, NUM_SPLITS)
    lse = tl.load(partial_lse_ptr + s_rows)
    m = tl.max(lse, axis=0)
    w = tl.exp2(lse - tl.where(m == -float("inf"), 0.0, m))
    tot = tl.sum(w, axis=0)
    part = tl.load(partial_out_ptr + s_rows[:, None] * KV_LORA_RANK + dims[None, :])
    merged = tl.sum(part * w[:, None], axis=0)
    out_val = tl.where(tot > 0.0, merged / tot, 0.0)
    tl.store(out_ptr + row * KV_LORA_RANK + dims, out_val.to(out_ptr.dtype.element_ty))
    if tl.program_id(1) == 0:
        lse_nat = tl.where(
            tot > 0.0, (m + tl.log2(tot)) * 0.6931471805599453, -float("inf")
        )
        tl.store(lse_ptr + row, lse_nat)
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def hy4_dfly_shared_kv_dsa(
    q_absorbed: torch.Tensor,
    target_kv_buffer: torch.Tensor,
    ctx_slots: torch.Tensor,
    ctx_lens: torch.Tensor,
    *,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-copy DFly context DSA sharing ``ctx_slots [B, K]`` across ``q_absorbed [B, S, H, D]``."""
    bsz, q_len, num_heads, _ = q_absorbed.shape
    max_ctx_k = int(ctx_slots.shape[1])
    out = torch.empty(
        (bsz, q_len, num_heads, kv_lora_rank),
        dtype=q_absorbed.dtype,
        device=q_absorbed.device,
    )
    lse = torch.empty(
        (bsz, q_len, num_heads),
        dtype=torch.float32,
        device=q_absorbed.device,
    )
    if bsz == 0 or q_len == 0:
        return out, lse
    if target_kv_buffer.ndim == 4:
        page_size = int(target_kv_buffer.shape[1])
        stride_kv_page = target_kv_buffer.stride(0)
        stride_kv_tok = target_kv_buffer.stride(1)
        stride_kv_c = target_kv_buffer.stride(3)
    elif target_kv_buffer.ndim == 3:
        page_size = 1
        stride_kv_page = target_kv_buffer.stride(0)
        stride_kv_tok = 0
        stride_kv_c = target_kv_buffer.stride(2)
    else:
        page_size = 1
        stride_kv_page = target_kv_buffer.stride(0)
        stride_kv_tok = 0
        stride_kv_c = target_kv_buffer.stride(1)

    block_h = 16
    if bsz == 1 or q_len == 1:
        q_group, block_m, num_warps = 1, 16, 4
        num_splits = 8 if bsz * q_len <= 8 else (4 if bsz * q_len <= 24 else 1)
    elif bsz >= 24 and q_len >= 3:
        q_group, block_m, num_splits, num_warps = 3, 64, 1, 8
    elif bsz >= 12:
        q_group, block_m, num_splits, num_warps = 2, 32, 2, 4
    else:
        q_group, block_m, num_splits, num_warps = 2, 32, 4, 4

    total_rows = bsz * q_len * num_heads
    if num_splits > 1:
        partial_out = torch.empty(
            (total_rows, num_splits, kv_lora_rank),
            dtype=torch.float32,
            device=q_absorbed.device,
        )
        partial_lse = torch.empty(
            (total_rows, num_splits),
            dtype=torch.float32,
            device=q_absorbed.device,
        )
    else:
        partial_out = out
        partial_lse = lse

    q_blks = triton.cdiv(q_len, q_group)
    grid = (bsz, q_blks, (num_heads // block_h) * num_splits)
    enable_pdl = pdl_enabled()
    pdl_kwargs = {"launch_pdl": True} if enable_pdl else {}
    _hy4_dfly_shared_kv_dsa_kernel[grid](
        q_absorbed,
        target_kv_buffer,
        ctx_slots,
        ctx_lens,
        out,
        partial_out,
        partial_lse,
        lse,
        bsz,
        max_ctx_k,
        q_absorbed.stride(0),
        q_absorbed.stride(1),
        q_absorbed.stride(2),
        ctx_slots.stride(0),
        stride_kv_page,
        stride_kv_tok,
        stride_kv_c,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
        float(softmax_scale),
        Q_LEN=q_len,
        NUM_HEADS=num_heads,
        KV_LORA_RANK=kv_lora_rank,
        QK_ROPE_HEAD_DIM=qk_rope_head_dim,
        PAGE_SIZE=page_size,
        NUM_SPLITS=num_splits,
        BLOCK_H=block_h,
        Q_GROUP=q_group,
        BLOCK_M=block_m,
        BLOCK_N=32,
        ENABLE_PDL=enable_pdl,
        num_warps=num_warps,
        num_stages=2,
        **pdl_kwargs,
    )
    if num_splits > 1:
        block_d = min(kv_lora_rank, 128)
        _hy4_dfly_shared_kv_dsa_merge_kernel[(total_rows, kv_lora_rank // block_d)](
            partial_out,
            partial_lse,
            out,
            lse,
            KV_LORA_RANK=kv_lora_rank,
            NUM_SPLITS=num_splits,
            BLOCK_D=block_d,
            ENABLE_PDL=enable_pdl,
            num_warps=4,
            **pdl_kwargs,
        )
    return out, lse


__all__ = [
    "hy4_merge_attention_and_sink",
    "hy4_dfly_build_decode_round_inputs",
    "hy4_vocab_argmax_into",
    "hy4_dual_rmsnorm_cat",
    "hy4_dfly_block_attn_merge_sink",
    "hy4_dfly_shared_kv_dsa",
]
