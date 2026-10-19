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


"""Explicit SM90 DSA cores built from the pinned public FlashMLA source."""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.attention.dsa._cuda_flashmla_common import (
    _finish_sparse_attention,
    _flashmla_sparse_prefill,
)
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

_CAPABILITY = CapabilityRequirement(
    min_arch_version=ArchVersion(9, 0),
    max_arch_version=ArchVersion(9, 0),
    vendors=frozenset({"nvidia"}),
)
_SIGNATURES = frozenset({format_signature(q=dense_tensor_format(torch.bfloat16))})
_TRAITS = {
    "qk_nope_head_dim": frozenset({128, 192}),
    "kv_lora_rank": frozenset({512}),
    "qk_rope_head_dim": frozenset({64}),
    "page_size": frozenset({64}),
    "topk": frozenset({512, 1024, 2048}),
    "has_kv_cache": frozenset({True}),
    "has_sparse_kv_cache": frozenset({False}),
    "topk_layout": frozenset({"global_slots"}),
    "logit_cap": frozenset({False}),
    "return_lse": frozenset({False, True}),
    "sinks": frozenset({False, True}),
    "slot_order": frozenset({"selection"}),
}


@register_kernel(
    "attention",
    "dsa_prefill",
    name="flashmla_public_sm90_dsa_prefill",
    solution="flashmla_public",
    capability=_CAPABILITY,
    signatures=_SIGNATURES,
    traits={**_TRAITS, "q_len": frozenset({1})},
    priority=Priority.REFERENCE,
)
def flashmla_public_sm90_dsa_prefill(
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
    slot_order: str,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Serve the DSA core contract through public-source BF16 sparse prefill.

    Query rows may have 1..128 heads and are padded to 64 or 128. The regular
    cache is BF16, with one 576-wide latent KV head. Global INT32 slots carry
    512, 1024 or 2048 columns; optional integer lengths mask their tail first.
    The common adapter preserves the existing flattening/stride, sink, LSE,
    empty-row and caller-output-buffer semantics. Decode uses this same entry.
    The raw provider returns output, max logits and LSE; max logits are not
    exposed by the DSA facade. A sink affects raw output but not raw LSE.
    """
    platform = current_platform()
    if not platform.is_nvidia or platform.arch_version != ArchVersion(9, 0):
        raise ValueError("public FlashMLA DSA requires SM90")
    if q.ndim not in (3, 4) or q.dtype != torch.bfloat16 or not q.is_cuda:
        raise ValueError("public FlashMLA DSA requires rank-3/4 CUDA BF16 queries")
    heads = q.shape[-2]
    if not 1 <= heads <= 128 or q.shape[-1] != 576:
        raise ValueError("public FlashMLA DSA requires 1..128 query heads of width 576")
    rows = q.numel() // (heads * q.shape[-1])
    if (
        kv_cache is None
        or sparse_kv_cache is not None
        or kv_cache.dtype != torch.bfloat16
        or kv_cache.device != q.device
        or kv_cache.ndim not in (2, 3, 4)
        or kv_cache.shape[-1] != 576
        or (kv_cache.ndim >= 3 and kv_cache.shape[-2] != 1)
    ):
        raise ValueError("public FlashMLA DSA requires regular colocated BF16 KV")
    if (
        qk_nope_head_dim not in (128, 192)
        or kv_lora_rank != 512
        or qk_rope_head_dim != 64
        or page_size != 64
        or q_len_per_req not in (1, 2, 3, 4, 5, 6)
        or logit_cap != 0.0
        or slot_order != "selection"
    ):
        raise ValueError("unsupported public FlashMLA DSA geometry or slot order")
    if (
        topk_slots.ndim != 2
        or topk_slots.shape[0] != rows
        or topk_slots.shape[1] not in (512, 1024, 2048)
        or topk_slots.dtype != torch.int32
        or topk_slots.device != q.device
        or topk_slots.stride(-1) != 1
    ):
        raise ValueError(
            "public FlashMLA DSA requires colocated INT32 [rows, topk] slots"
        )
    if topk_lens is not None and (
        topk_lens.numel() != rows
        or topk_lens.dtype not in (torch.int32, torch.int64)
        or topk_lens.device != q.device
    ):
        raise ValueError(
            "public FlashMLA DSA requires one integer length per query row"
        )
    if attn_sink is not None and (
        attn_sink.ndim != 1
        or attn_sink.dtype != torch.float32
        or attn_sink.device != q.device
        or attn_sink.numel() < heads
    ):
        raise ValueError("public FlashMLA DSA requires colocated FP32 per-head sinks")
    if out is not None and (
        out.device != q.device
        or out.dtype != q.dtype
        or out.numel() != rows * heads * kv_lora_rank
        or not out.is_contiguous()
    ):
        raise ValueError(
            "public FlashMLA DSA output buffer must be contiguous colocated BF16 "
            "with the result's element count"
        )
    if rows == 0:
        result = q.new_empty((0, heads, kv_lora_rank))
        lse = q.new_empty((0, heads), dtype=torch.float32)
        return _finish_sparse_attention(result, lse, topk_slots, out, return_lse)

    # Independent from the optional installed flash_mla extension. The source
    # package is imported only for this explicitly selected nonempty call.
    from tokenspeed_kernel.thirdparty.flashmla_public_sm90 import flash_mla_sparse_fwd

    return _flashmla_sparse_prefill(
        q=q,
        kv_cache=kv_cache,
        sparse_kv_cache=sparse_kv_cache,
        topk_slots=topk_slots,
        topk_lens=topk_lens,
        max_seqlen_k=max_seqlen_k,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        softmax_scale=softmax_scale,
        page_size=page_size,
        q_len_per_req=q_len_per_req,
        kv_seq_lens=kv_seq_lens,
        logit_cap=logit_cap,
        k_scale=k_scale,
        return_lse=return_lse,
        out=out,
        enable_pdl=enable_pdl,
        attn_sink=attn_sink,
        sparse_fwd=flash_mla_sparse_fwd,
    )


@register_kernel(
    "attention",
    "dsa_decode",
    name="flashmla_public_sm90_dsa_decode",
    solution="flashmla_public",
    capability=_CAPABILITY,
    signatures=_SIGNATURES,
    traits={**_TRAITS, "q_len": frozenset({1, 2, 3, 4, 5, 6})},
    priority=Priority.REFERENCE,
)
def flashmla_public_sm90_dsa_decode(**kwargs):
    """Dense BF16 decode and verification rows use the same public core."""
    return flashmla_public_sm90_dsa_prefill(**kwargs)
