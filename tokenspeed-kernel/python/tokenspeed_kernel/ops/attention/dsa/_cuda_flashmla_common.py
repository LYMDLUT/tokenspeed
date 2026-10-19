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


"""Shared FlashMLA sparse-attention layout adapters, independent of its loader."""

from __future__ import annotations

from collections.abc import Callable

import torch
from tokenspeed_kernel.platform import current_platform

platform = current_platform()
_sm_counts: dict[int, int] = {}
_FLASHMLA_SPARSE_TOPK_ALIGN = 128


def _sm_count(device: torch.device) -> int:
    index = device.index if device.index is not None else torch.cuda.current_device()
    count = _sm_counts.get(index)
    if count is None:
        count = torch.cuda.get_device_properties(index).multi_processor_count
        _sm_counts[index] = count
    return count


def _flashmla_sparse_column_splits(
    rows: int, padded_heads: int, width: int, device: torch.device
) -> int:
    """Column splits per query row for a FlashMLA sparse prefill launch.

    The kernel attends every ``_flashmla_sparse_prefill_head_multiple()``
    heads of a query row in one CTA that walks the row's selected columns
    serially, so a launch with fewer CTAs than SMs (decode, verify, short
    prompts) is bound by one CTA's walk while most SMs idle. Slicing the
    columns into independent rows, merged afterwards by their LSE, spreads
    that walk over the idle SMs. Splitting stops while the split launch,
    ``rows * ctas_per_row * splits`` CTAs, still fits one wave, past which it
    adds work without shortening the launch.
    """
    sms = _sm_count(device)
    ctas_per_row = padded_heads // _flashmla_sparse_prefill_head_multiple()
    splits = 1
    while (
        rows * ctas_per_row * splits * 2 <= sms
        and width % (splits * 2 * _FLASHMLA_SPARSE_TOPK_ALIGN) == 0
    ):
        splits *= 2
    return splits


def _flashmla_sparse_prefill_head_multiple() -> int:
    return 128 if platform.is_nvidia and platform.is_blackwell_plus else 64


def _flashmla_sparse_prefill_padded_heads(num_heads: int) -> int:
    head_multiple = _flashmla_sparse_prefill_head_multiple()
    return ((int(num_heads) + head_multiple - 1) // head_multiple) * head_multiple


def _pad_query_heads(q: torch.Tensor, padded_heads: int) -> torch.Tensor:
    # Token counts vary per batch, so a per-shape workspace cache would grow
    # without bound; the caching allocator (or the graph pool under capture)
    # recycles these instead.
    actual_heads = q.shape[-2]
    q_padded = q.new_empty((*q.shape[:-2], padded_heads, q.shape[-1]))
    q_padded[..., :actual_heads, :].copy_(q)
    q_padded[..., actual_heads:, :].zero_()
    return q_padded


def _pad_prefill_query(q: torch.Tensor) -> tuple[torch.Tensor, int]:
    q = q.reshape(-1, q.shape[-2], q.shape[-1]).contiguous()
    actual_heads = q.shape[1]
    padded_heads = _flashmla_sparse_prefill_padded_heads(actual_heads)
    if padded_heads == actual_heads:
        return q, actual_heads
    return _pad_query_heads(q, padded_heads), actual_heads


def _flatten_regular_kv_cache(kv_cache: torch.Tensor, page_size: int) -> torch.Tensor:
    if kv_cache.dim() == 2:
        return kv_cache.view(-1, 1, kv_cache.shape[-1])
    if kv_cache.dim() == 3:
        return kv_cache.reshape(-1, kv_cache.shape[-2], kv_cache.shape[-1])
    if kv_cache.dim() == 4:
        if kv_cache.shape[1] != int(page_size):
            raise ValueError(
                f"paged kv_cache page size mismatch: got {kv_cache.shape[1]}, "
                f"expected {page_size}"
            )
        return kv_cache.reshape(-1, kv_cache.shape[-2], kv_cache.shape[-1])
    raise ValueError(f"unsupported kv_cache shape {tuple(kv_cache.shape)}")


def _mask_sparse_slots(
    slots: torch.Tensor, lengths: torch.Tensor | None
) -> torch.Tensor:
    if lengths is None:
        return slots
    columns = torch.arange(slots.shape[-1], device=slots.device)
    return slots.masked_fill(columns >= lengths.reshape(-1, 1), -1)


def _pad_sink_for_heads(
    sink: torch.Tensor,
    padded_heads: int,
    actual_heads: int,
) -> torch.Tensor:
    """Widen a per-head sink to the padded query head count.

    The sparse FlashMLA kernels require the query to be padded to a multiple of
    64 (128 on Blackwell). The padded lanes produce discarded output, but they
    still take part in the softmax, so their sink must be ``-inf``: that makes
    ``exp(sink - max)`` zero and leaves the real heads' normalization untouched.

    Args:
        sink: FP32 sink logits covering the layer's ``actual_heads``.
        padded_heads: Head count the kernel will run with.
        actual_heads: Heads the sink actually describes.

    Returns:
        The sink narrowed to the padded width, padded with ``-inf`` when the
        kernel runs wider.
    """
    if sink.dtype != torch.float32:
        raise ValueError(f"attention sink must be float32, got {sink.dtype}")
    if sink.numel() < actual_heads:
        raise ValueError(
            f"attention sink covers {sink.numel()} heads, need {actual_heads}"
        )
    if padded_heads == actual_heads:
        return sink[:actual_heads].contiguous()
    padded = sink.new_full((padded_heads,), float("-inf"))
    padded[:actual_heads].copy_(sink[:actual_heads])
    return padded


def _finish_sparse_attention(
    result: torch.Tensor,
    lse: torch.Tensor,
    slots: torch.Tensor,
    out: torch.Tensor | None,
    return_lse: bool,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    # Empty ranks contribute the identity element to the shared DCP reduction.
    empty = ~(slots >= 0).any(dim=-1)
    result.masked_fill_(empty[:, None, None], 0)
    if out is not None:
        if (
            out.numel() != result.numel()
            or out.dtype != result.dtype
            or not out.is_contiguous()
        ):
            raise ValueError(
                "FlashMLA sparse attention output buffer must be a contiguous "
                f"{result.dtype} tensor of {result.numel()} elements, got "
                f"{out.dtype} {tuple(out.shape)} with strides {out.stride()}"
            )
        out.view(result.shape).copy_(result)
        result = out
    if not return_lse:
        return result
    lse = lse.contiguous().masked_fill(empty[:, None], -float("inf"))
    return result, lse


def _flashmla_sparse_prefill_split(
    *,
    q: torch.Tensor,
    kv: torch.Tensor,
    slots: torch.Tensor,
    topk_lens: torch.Tensor | None,
    sm_scale: float,
    d_v: int,
    attn_sink: torch.Tensor | None,
    actual_heads: int,
    splits: int,
    out: torch.Tensor | None,
    return_lse: bool,
    sparse_fwd: Callable,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """FlashMLA sparse prefill with every row's columns split ``splits`` ways.

    Split ``s`` of row ``r`` becomes kernel row ``r * splits + s`` over the
    columns ``[s * width, (s + 1) * width)``. The kernel runs without the sink
    (it would scale each partial by its own denominator); the merge applies it
    once to the combined softmax.
    """
    from tokenspeed_kernel.ops.attention.dsa._triton.split_partials import (
        merge_split_partials,
    )

    rows, padded_heads, head_dim = q.shape
    width = slots.shape[-1] // splits
    q_split = (
        q.unsqueeze(1)
        .expand(rows, splits, padded_heads, head_dim)
        .reshape(rows * splits, padded_heads, head_dim)
    )
    starts = torch.arange(splits, device=q.device, dtype=torch.int32) * width
    if topk_lens is None:
        lens = torch.full(
            (rows, 1), slots.shape[-1], device=q.device, dtype=torch.int32
        )
    else:
        lens = topk_lens.reshape(rows, 1).to(torch.int32)
    split_lens = (lens - starts).clamp_(0, width).reshape(-1)
    partial, _, partial_lse = sparse_fwd(
        q=q_split,
        kv=kv,
        indices=slots.reshape(rows * splits, 1, width),
        sm_scale=sm_scale,
        d_v=d_v,
        attn_sink=None,
        topk_length=split_lens,
    )
    return merge_split_partials(
        partial,
        partial_lse,
        split_lens,
        splits=splits,
        heads=actual_heads,
        sink=None if attn_sink is None else attn_sink[:actual_heads].contiguous(),
        out=out,
        return_lse=return_lse,
    )


def _flashmla_sparse_prefill(
    q: torch.Tensor,
    kv_cache: torch.Tensor | None,
    sparse_kv_cache: torch.Tensor | None,
    topk_slots: torch.Tensor,
    topk_lens: torch.Tensor | None,
    max_seqlen_k: int,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    softmax_scale: float,
    page_size: int,
    q_len_per_req: int,
    kv_seq_lens: torch.Tensor | None,
    logit_cap: float,
    k_scale: float,
    return_lse: bool,
    out: torch.Tensor | None,
    enable_pdl: bool,
    attn_sink: torch.Tensor | None = None,
    *,
    sparse_fwd: Callable,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    del enable_pdl
    if kv_cache is None:
        raise RuntimeError("FlashMLA sparse prefill requires kv_cache")
    if logit_cap != 0.0:
        raise RuntimeError("FlashMLA sparse prefill does not support logit_cap")
    q_kernel, actual_heads = _pad_prefill_query(q)
    kv = _flatten_regular_kv_cache(kv_cache, page_size)
    slots = _mask_sparse_slots(topk_slots, topk_lens)
    # Keep sink normalization inside FlashMLA before its BF16 output cast.
    # Merging BF16 partials would round once before applying the sink and
    # once again afterwards, even when only one split has live keys.
    splits = (
        1
        if attn_sink is not None
        else _flashmla_sparse_column_splits(
            q_kernel.shape[0], q_kernel.shape[1], slots.shape[-1], q_kernel.device
        )
    )
    if splits > 1:
        return _flashmla_sparse_prefill_split(
            q=q_kernel,
            kv=kv,
            slots=slots,
            topk_lens=topk_lens,
            sm_scale=float(softmax_scale) * float(k_scale),
            d_v=int(kv_lora_rank),
            attn_sink=attn_sink,
            actual_heads=actual_heads,
            splits=splits,
            out=out,
            return_lse=return_lse,
            sparse_fwd=sparse_fwd,
        )
    sink = (
        None
        if attn_sink is None
        else _pad_sink_for_heads(attn_sink, q_kernel.shape[1], actual_heads)
    )
    # Every slot at or past topk_lens is masked to -1 above, so topk_length
    # lets the kernel skip those columns instead of iterating them.
    result, _, lse = sparse_fwd(
        q=q_kernel,
        kv=kv,
        indices=slots.unsqueeze(1),
        sm_scale=float(softmax_scale) * float(k_scale),
        d_v=int(kv_lora_rank),
        attn_sink=sink,
        topk_length=(
            None
            if topk_lens is None
            else topk_lens.reshape(-1).to(torch.int32).contiguous()
        ),
    )
    result = result[:, :actual_heads, :]
    lse = lse[:, :actual_heads]
    if return_lse and attn_sink is not None:
        # flash_mla_sparse_fwd folds the sink into the output normalization
        # but not into the returned LSE; add it so DCP merges see the same
        # denominator as the output.
        lse = torch.logaddexp(lse, attn_sink[:actual_heads].to(lse.dtype))
    return _finish_sparse_attention(result, lse, slots, out, return_lse)
