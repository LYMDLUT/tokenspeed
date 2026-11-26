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

"""Standalone kernel tests for Hy4-flash ops (no runtime imports)."""

from __future__ import annotations

import math
from test.utils import assert_no_triton_compile

import pytest
import torch
import torch.nn.functional as F
from tokenspeed_kernel.ops.attention.kpool.hy4_compress import (
    _hy4_apply_interleaved_rope_inplace_kernel,
    _hy4_index_k_quantize_scatter_kernel,
    _hy4_kpool_decode_update_gather_kernel,
    _hy4_kpool_finalize_rope_store_kernel,
    _hy4_kpool_project_finalize_rope_store_kernel,
    _hy4_kpool_project_finalize_rope_store_splitn_kernel,
    _hy4_kv_latent_norm_and_rope_inplace_kernel,
    _hy4_split_qkv_indexer_norm_rope_kernel,
    apply_interleaved_rope_last_dim,
    compute_hy4_mrope_cos_sin,
    hy4_apply_interleaved_rope_inplace,
    hy4_kpool_decode_update_and_gather,
    hy4_kpool_finalize_rope_store,
    hy4_kpool_project_finalize_rope_store,
    hy4_kv_latent_norm_and_rope_inplace,
    hy4_pool_group_keys,
    hy4_quantize_and_store_index_k,
    hy4_split_qkv_indexer_norm_rope,
)
from tokenspeed_kernel.ops.attention.mla._triton.decode import mla_decode_fwd
from tokenspeed_kernel.ops.attention.sink import (
    _apply_attention_sink_kernel,
    apply_attention_sink,
)
from tokenspeed_kernel.ops.gemm import mm
from tokenspeed_kernel.ops.kvcache.index_k_scale import (
    decode_index_k_scale,
    encode_index_k_scale,
    index_k_scale_dtype,
    round_index_k_scale,
)
from tokenspeed_kernel.ops.model.hy4_flash.dfly import (
    _hy4_dfly_build_decode_round_inputs_kernel,
    _hy4_merge_attention_and_sink_kernel,
    hy4_dfly_build_decode_round_inputs,
    hy4_merge_attention_and_sink,
)
from tokenspeed_kernel.ops.model.hy4_flash.fusion import (
    _hy4_sigmoid_bias_topk_kernel,
    _hy4_sigmoid_gate_quantize_fp8_kernel,
    _hy4_split_routed_and_shared_swiglu_kernel,
    hy4_sigmoid_bias_topk,
    hy4_sigmoid_gate_quantize_fp8,
    hy4_split_routed_and_shared_swiglu,
)
from tokenspeed_kernel.ops.residual.ihc import (
    _ihc_bf16_mix_splitk_kernel,
    _ihc_head_reduce_kernel,
    _ihc_post_and_pre_fused_kernel,
    _ihc_post_kernel,
    _ihc_pre_finalize_from_partials_kernel,
    _ihc_pre_reduce_kernel,
    ihc_head,
    ihc_post,
    ihc_post_and_pre,
    ihc_pre,
)
from tokenspeed_kernel.platform import Platform

_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _fp8_e4m3fn_step_distance(actual: torch.Tensor, expected: torch.Tensor) -> int:
    """Return max representable E4M3FN step distance between two FP8 tensors."""
    if actual.numel() == 0:
        assert expected.numel() == 0
        return 0

    def _rank(t: torch.Tensor) -> torch.Tensor:
        bits = t.contiguous().view(torch.uint8)
        mag = (bits & 0x7F).to(torch.int16)
        assert not bool((mag == 0x7F).any()), "unexpected FP8 NaN"
        return torch.where((bits & 0x80) != 0, 0x7E - mag, 0x7E + mag)

    return int((_rank(actual) - _rank(expected)).abs().max().item())


def _ref_attention_sink_fp64(
    output: torch.Tensor, lse: torch.Tensor, sink: torch.Tensor
) -> torch.Tensor:
    scale = torch.sigmoid(lse.double() - sink.double().unsqueeze(0)).unsqueeze(-1)
    return (output.double() * scale).to(output.dtype)


def _ref_ihc_pre_fp64(
    streams: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    rms_norm_eps: float = 1e-5,
    hc_magnitude: float = 2.0,
    hc_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    t, n, d = streams.shape
    flat = streams.reshape(t, n * d).double()
    inv_rms = torch.rsqrt(flat.pow(2).mean(dim=-1, keepdim=True) + rms_norm_eps)
    mixes = F.linear(flat, fn.double()) * inv_rms
    pre_logits, post_logits = mixes.split(n, dim=-1)
    pre_b, post_b = base.double().split(n, dim=-1)
    pre_s, post_s = scale.double().unbind(0)
    pre = torch.sigmoid(pre_logits * pre_s + pre_b) + hc_eps
    post = hc_magnitude * torch.sigmoid(post_logits * post_s + post_b) + hc_eps
    reduced = (pre.unsqueeze(-1) * streams.double()).sum(dim=1)
    return post.float(), reduced.to(streams.dtype)


def _ref_ihc_head_fp64(
    streams: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    rms_norm_eps: float = 1e-5,
    hc_eps: float = 1e-6,
) -> torch.Tensor:
    t, n, d = streams.shape
    flat = streams.reshape(t, n * d).double()
    inv_rms = torch.rsqrt(flat.pow(2).mean(dim=-1, keepdim=True) + rms_norm_eps)
    mixes = F.linear(flat, fn.double()) * inv_rms
    pre = torch.sigmoid(mixes * scale.double() + base.double()) + hc_eps
    return (pre.unsqueeze(-1) * streams.double()).sum(dim=1).to(streams.dtype)


def test_hy4_mrope_and_pool_group_keys_cpu() -> None:
    torch.manual_seed(0)
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64.0))
    pos_3d = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 2], [0, 1, 0, 1]], dtype=torch.long)
    cos, sin = compute_hy4_mrope_cos_sin(pos_3d, inv_freq, (12, 10, 10))
    assert cos.shape == (4, 32)
    assert sin.shape == (4, 32)
    x = torch.randn(4, 2, 64, dtype=torch.bfloat16)
    rotated = apply_interleaved_rope_last_dim(x, cos.unsqueeze(1), sin.unsqueeze(1))
    assert rotated.shape == x.shape
    assert rotated.dtype == x.dtype
    # Rotation preserves per-vector L2 norm up to BF16 rounding.
    torch.testing.assert_close(
        rotated.float().norm(dim=-1),
        x.float().norm(dim=-1),
        rtol=2e-2,
        atol=2e-2,
    )

    grouped = torch.randn(6, 4, 128, dtype=torch.bfloat16)
    kv_w = torch.randn(128, 128, dtype=torch.float32) * 0.1
    kg_w = torch.randn(128, 128, dtype=torch.float32) * 0.1
    ape = torch.randn(4, 128, dtype=torch.float32) * 0.05
    norm_w = torch.ones(128, dtype=torch.bfloat16)
    pooled = hy4_pool_group_keys(grouped, kv_w, kg_w, ape, norm_w, eps=1e-5)
    assert pooled.shape == (6, 128)
    assert pooled.dtype == torch.bfloat16
    assert torch.isfinite(pooled).all()
    gf = grouped.float()
    vals = F.linear(gf, kv_w)
    gates = F.linear(gf, kg_w) + ape.unsqueeze(0)
    raw = (vals * F.softmax(gates, dim=1)).sum(dim=1)
    ref = norm_w.float() * (
        raw * torch.rsqrt(raw.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
    )
    torch.testing.assert_close(pooled.float(), ref, rtol=2e-2, atol=2e-2)


def test_apply_attention_sink_matches_augmented_softmax_cpu() -> None:
    torch.manual_seed(1)
    q_len, kv_len, heads, dim = 6, 12, 4, 16
    scores = torch.randn(q_len, heads, kv_len)
    values = torch.randn(heads, kv_len, dim)
    sinks = torch.randn(heads)

    lse = torch.logsumexp(scores, dim=-1)
    probs = torch.softmax(scores, dim=-1)
    plain_out = torch.einsum("qhk,hkd->qhd", probs, values)
    rescaled = apply_attention_sink(plain_out.clone(), lse, sinks, inplace=False)

    aug_scores = torch.cat(
        [scores, sinks.view(1, heads, 1).expand(q_len, heads, 1)], dim=-1
    )
    aug_probs = torch.softmax(aug_scores, dim=-1)[..., :-1]
    ref_out = torch.einsum("qhk,hkd->qhd", aug_probs, values)
    torch.testing.assert_close(rescaled, ref_out, rtol=1e-5, atol=1e-5)


def test_hy4_quantize_and_store_index_k_strided_arena_view() -> None:
    """Verify strided paged arena views with storage_offset preserve decoded scale/value."""
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_pages, num_layers, rows_per_page, head_dim = 8, 3, 16, 128
    row_bytes = head_dim + 4
    arena = torch.zeros(
        (num_pages, num_layers, rows_per_page, row_bytes),
        dtype=torch.uint8,
        device=device,
    )
    buf = arena[:, 1]
    page_stride = buf.stride(0)
    values = torch.as_strided(
        buf, (num_pages, rows_per_page, head_dim), (page_stride, head_dim, 1)
    ).view(torch.float8_e4m3fn)
    scales = torch.as_strided(
        buf,
        (num_pages, rows_per_page, 4),
        (page_stride, 4, 1),
        buf.storage_offset() + rows_per_page * head_dim,
    ).view(index_k_scale_dtype())

    keys = torch.randn(24, head_dim, dtype=torch.bfloat16, device=device)
    slots = torch.tensor(
        [16 + i for i in range(12)] + [64 + i for i in range(12)],
        dtype=torch.int64,
        device=device,
    )
    hy4_quantize_and_store_index_k(keys, slots, values, scales)

    pages = slots // rows_per_page
    rows = slots % rows_per_page
    decoded_scales = decode_index_k_scale(scales[pages, rows])
    dequantized = values[pages, rows].float() * decoded_scales
    expected_scales = (
        keys.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-4) / 448.0
    )
    assert torch.isfinite(decoded_scales).all()
    assert (decoded_scales >= expected_scales * 0.99).all()
    assert (decoded_scales <= expected_scales * 2.01).all()
    torch.testing.assert_close(dequantized, keys.float(), atol=0.25, rtol=0.1)


@_CUDA
def test_hy4_quantize_and_store_index_k_cuda_bits_and_no_recompile() -> None:
    torch.manual_seed(11)
    device = torch.device("cuda")
    scale_dtype = index_k_scale_dtype()
    rows_per_page = 16
    head_dim = 128

    def run_once(num_rows: int, num_pages: int) -> None:
        keys = (
            torch.randn(num_rows, head_dim, dtype=torch.bfloat16, device=device) * 0.6
        )
        slots = torch.arange(num_rows, dtype=torch.int32, device=device)
        if num_rows >= 3:
            slots[1] = -1
            slots[-1] = num_pages * rows_per_page + 5  # OOB must be skipped
        values = torch.zeros(
            (num_pages, rows_per_page, head_dim),
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        scales = torch.zeros(
            (num_pages, rows_per_page, 1), dtype=scale_dtype, device=device
        )
        hy4_quantize_and_store_index_k(keys, slots, values, scales)

        # Verify CPU and GPU agree on raw scale encoding (exact for UE8M0 int32,
        # <= 1 ULP for FP32 SM90 division; catches any int-to-float bitcast bug).
        cpu_values = torch.zeros_like(values, device="cpu")
        cpu_scales = torch.zeros_like(scales, device="cpu")
        hy4_quantize_and_store_index_k(keys.cpu(), slots.cpu(), cpu_values, cpu_scales)
        if scale_dtype == torch.int32:
            assert torch.equal(
                scales.view(torch.int32).cpu(), cpu_scales.view(torch.int32)
            )
        else:
            torch.testing.assert_close(scales.cpu(), cpu_scales, rtol=1e-5, atol=1e-7)
            ulp_diff = (
                scales.view(torch.int32).cpu() - cpu_scales.view(torch.int32)
            ).abs()
            assert int(ulp_diff.max().item()) <= 1
        assert _fp8_e4m3fn_step_distance(values.cpu(), cpu_values) <= 1

        # Check active row 0 scale matches round_index_k_scale(absmax/448) within 1 ULP.
        expected_scale = round_index_k_scale(
            keys[0:1].float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-4) / 448.0
        )
        expected_word = encode_index_k_scale(expected_scale).view(torch.int32)[0, 0]
        assert (
            abs(
                int(scales.view(torch.int32)[0, 0, 0].item())
                - int(expected_word.item())
            )
            <= 1
        )
        # Masked negative slot must remain untouched.
        if num_rows >= 3:
            assert int(scales.view(torch.int32)[0, 1, 0].item()) == 0

    run_once(7, 4)
    with assert_no_triton_compile(_hy4_index_k_quantize_scatter_kernel):
        run_once(16, 4)
        run_once(29, 8)
        run_once(64, 16)


@_CUDA
def test_hy4_quantize_and_store_index_k_int64_large_page_offset() -> None:
    """Verify page * stride_iv_page stays in int64 past 2^31 bytes."""
    device = torch.device("cuda")
    rows_per_page = 16
    head_dim = 128
    target_page = (1 << 20) + 17  # 1,048,593
    page_stride_bytes = rows_per_page * head_dim  # 2048; target_page * 2048 > 2^31

    free_bytes, _ = torch.cuda.mem_get_info(device)
    needed_bytes = (target_page + 1) * page_stride_bytes + 128 * 1024 * 1024
    if free_bytes < needed_bytes:
        pytest.skip(
            f"insufficient free VRAM for >2GiB int64 offset test ({free_bytes})"
        )

    big_vals = torch.empty(
        (target_page + 1, rows_per_page, head_dim),
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    big_vals[target_page].zero_()
    big_scales = torch.empty(
        (target_page + 1, rows_per_page, 1),
        dtype=index_k_scale_dtype(),
        device=device,
    )
    big_scales[target_page].zero_()
    assert target_page * big_vals.stride(0) > (1 << 31)

    keys = torch.randn((1, head_dim), dtype=torch.bfloat16, device=device) * 0.5
    slot = torch.tensor(
        [target_page * rows_per_page + 3], dtype=torch.int64, device=device
    )
    hy4_quantize_and_store_index_k(keys, slot, big_vals, big_scales)

    expected_scale = round_index_k_scale(
        keys.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-4) / 448.0
    )
    expected_word = encode_index_k_scale(expected_scale).view(torch.int32)[0, 0]
    assert (
        abs(
            int(big_scales.view(torch.int32)[target_page, 3, 0].item())
            - int(expected_word.item())
        )
        <= 1
    )
    del big_vals, big_scales


@_CUDA
def test_apply_attention_sink_parity_and_no_recompile() -> None:
    torch.manual_seed(3)
    device = torch.device("cuda")

    def check(num_tokens: int, *, inplace: bool) -> None:
        out = torch.randn(num_tokens, 32, 128, dtype=torch.bfloat16, device=device)
        lse = torch.randn(num_tokens, 32, dtype=torch.float32, device=device) * 4.0
        lse[0, 0] = float("-inf")
        sink = torch.randn(32, dtype=torch.float32, device=device)
        expected = _ref_attention_sink_fp64(out, lse, sink)
        inp = out.clone() if inplace else out
        got = apply_attention_sink(inp, lse, sink, inplace=inplace)
        if inplace:
            assert got.data_ptr() == inp.data_ptr()
        torch.testing.assert_close(got, expected, rtol=1e-2, atol=1e-2)
        assert float(got[0, 0].abs().max().item()) == 0.0

    check(5, inplace=False)
    with assert_no_triton_compile(_apply_attention_sink_kernel):
        check(16, inplace=True)
        check(33, inplace=False)
        check(128, inplace=True)


@_CUDA
def test_ihc_kernels_parity_and_no_recompile() -> None:
    torch.manual_seed(5)
    device = torch.device("cuda")
    hc_mult = 4
    hidden = 512

    fn_pre = (
        torch.randn(2 * hc_mult, hc_mult * hidden, dtype=torch.float32, device=device)
        * 0.02
    )
    scale_pre = torch.tensor([0.1, 0.2], dtype=torch.float32, device=device)
    base_pre = torch.randn(2 * hc_mult, dtype=torch.float32, device=device) * 0.1

    fn_head = (
        torch.randn(hc_mult, hc_mult * hidden, dtype=torch.float32, device=device)
        * 0.02
    )
    scale_head = torch.tensor([0.15], dtype=torch.float32, device=device)
    base_head = torch.randn(hc_mult, dtype=torch.float32, device=device) * 0.1

    norm_w = torch.randn(hidden, dtype=torch.bfloat16, device=device) * 0.2 + 1.0

    def check(num_tokens: int) -> None:
        streams = torch.randn(
            num_tokens, hc_mult, hidden, dtype=torch.bfloat16, device=device
        )
        branch = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=device)

        post_got, reduced_got = ihc_pre(streams, fn_pre, scale_pre, base_pre)
        post_ref, reduced_ref = _ref_ihc_pre_fp64(streams, fn_pre, scale_pre, base_pre)
        torch.testing.assert_close(post_got, post_ref, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(reduced_got, reduced_ref, rtol=2e-2, atol=2e-2)

        post_norm_got, normed_got = ihc_pre(
            streams,
            fn_pre,
            scale_pre,
            base_pre,
            norm_weight=norm_w,
            norm_eps=1e-5,
        )
        rf = reduced_ref.double()
        normed_ref = (rf * torch.rsqrt(rf.pow(2).mean(dim=-1, keepdim=True) + 1e-5)).to(
            streams.dtype
        ) * norm_w
        torch.testing.assert_close(post_norm_got, post_ref, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(normed_got, normed_ref, rtol=3e-2, atol=3e-2)

        updated_got = ihc_post(streams, branch, post_got)
        streams_ip = streams.clone()
        updated_ip = ihc_post(streams_ip, branch, post_got, inplace=True)
        assert updated_ip.data_ptr() == streams_ip.data_ptr()
        updated_ref = (
            post_ref.double().unsqueeze(-1) * branch.double().unsqueeze(1)
            + streams.double()
        ).to(streams.dtype)
        torch.testing.assert_close(updated_got, updated_ref, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(updated_ip, updated_ref, rtol=2e-2, atol=2e-2)

        head_got = ihc_head(updated_got, fn_head, scale_head, base_head)
        head_ref = _ref_ihc_head_fp64(updated_got, fn_head, scale_head, base_head)
        torch.testing.assert_close(head_got, head_ref, rtol=2e-2, atol=2e-2)

    # Warm both split-K (<=256) and large-M (>256) regimes before checking zero recompile.
    check(3)
    check(280)
    with assert_no_triton_compile(
        _ihc_bf16_mix_splitk_kernel,
        _ihc_pre_finalize_from_partials_kernel,
        _ihc_pre_reduce_kernel,
        _ihc_post_kernel,
        _ihc_head_reduce_kernel,
    ):
        check(1)
        check(16)
        check(64)
        check(192)
        check(300)


@_CUDA
@pytest.mark.parametrize("q_len,window_left", [(1, -1), (6, -1), (6, 31)])
def test_triton_mla_decode_multi_q_len_and_swa_parity(
    q_len: int, window_left: int
) -> None:
    """Verify multi-token verify (q_len > 1) and SWA windowing in Triton MLA decode."""
    torch.manual_seed(19)
    device = torch.device("cuda")
    bs, num_heads, kv_lora_rank, rope_dim, page_size = 2, 8, 512, 64, 64
    num_pages = 8
    sm_scale = 1.0 / math.sqrt(128 + rope_dim)

    q = (
        torch.randn(
            bs,
            q_len,
            num_heads,
            kv_lora_rank + rope_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.2
    )
    kv_cache = (
        torch.randn(
            num_pages,
            page_size,
            1,
            kv_lora_rank + rope_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.2
    )
    page_table = torch.tensor(
        [[0, 2, 4, 6], [1, 3, 5, 7]], dtype=torch.int32, device=device
    )
    cache_seqlens = torch.tensor([90, 140], dtype=torch.int32, device=device)

    out = torch.empty(
        (bs, q_len, num_heads, kv_lora_rank), dtype=torch.bfloat16, device=device
    )
    lse = torch.empty((bs, q_len, num_heads), dtype=torch.float32, device=device)
    mla_decode_fwd(
        q,
        kv_cache,
        out,
        page_table,
        cache_seqlens,
        max_seqlen_k=140,
        qk_nope_head_dim=128,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=rope_dim,
        softmax_scale=sm_scale,
        lse=lse,
        window_left=window_left,
    )

    # Exact FP64 reference per (batch, q_idx)
    ref_out = torch.zeros_like(out, dtype=torch.float64)
    ref_lse = torch.zeros_like(lse, dtype=torch.float64)
    for b in range(bs):
        full_len = int(cache_seqlens[b].item())
        pages = page_table[b].long()
        flat_kv = (
            kv_cache[pages, :, 0, :]
            .reshape(-1, kv_lora_rank + rope_dim)[:full_len]
            .double()
        )
        for qi in range(q_len):
            q_cache_len = full_len - (q_len - 1 - qi)
            w_start = max(0, q_cache_len - 1 - window_left) if window_left >= 0 else 0
            k_slice = flat_kv[w_start:q_cache_len]  # [L, R+rope]
            v_slice = k_slice[:, :kv_lora_rank]  # [L, R]
            q_vec = q[b, qi].double()  # [H, R+rope]
            scores = (q_vec @ k_slice.T) * sm_scale  # [H, L]
            lse_row = torch.logsumexp(scores, dim=-1)
            probs = torch.softmax(scores, dim=-1)
            ref_out[b, qi] = probs @ v_slice
            ref_lse[b, qi] = lse_row

    torch.testing.assert_close(out, ref_out.to(out.dtype), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(lse, ref_lse.float(), rtol=1e-3, atol=1e-3)


@_CUDA
def test_hy4_fused_rope_and_kv_latent_cuda() -> None:
    """Verify fused in-place interleaved RoPE + KV latent RMSNorm/RoPE and zero recompile."""
    torch.manual_seed(23)
    device = torch.device("cuda")
    mrope_section = (11, 11, 10)
    inv_freq = 1.0 / (
        1e7 ** (torch.arange(0, 64, 2, dtype=torch.float32, device=device) / 64)
    )

    for num_tokens, check_no_compile in ((7, False), (19, True)):
        pos_3d = torch.stack(
            [
                torch.arange(num_tokens, device=device),
                torch.arange(num_tokens, device=device) * 2 + 3,
                torch.arange(num_tokens, device=device) * 3 + 5,
            ],
            dim=0,
        )
        cos, sin = compute_hy4_mrope_cos_sin(pos_3d, inv_freq, mrope_section)
        cos_bf16 = cos.to(torch.bfloat16).contiguous()
        sin_bf16 = sin.to(torch.bfloat16).contiguous()

        q = torch.randn(num_tokens, 16, 192, dtype=torch.bfloat16, device=device)
        q_ref = q.clone()
        q_ref[..., 128:] = apply_interleaved_rope_last_dim(
            q_ref[..., 128:], cos_bf16.unsqueeze(1), sin_bf16.unsqueeze(1)
        )

        raw_kv = torch.randn(num_tokens, 576, dtype=torch.bfloat16, device=device)
        norm_w = torch.randn(512, dtype=torch.bfloat16, device=device)
        eps = 1e-5
        kv_f = raw_kv[:, :512].float()
        ref_kv_pass = (
            kv_f * torch.rsqrt(kv_f.pow(2).mean(dim=-1, keepdim=True) + eps)
        ).to(torch.bfloat16) * norm_w
        ref_k_pe = apply_interleaved_rope_last_dim(raw_kv[:, 512:], cos_bf16, sin_bf16)
        kv_ref = torch.cat((ref_kv_pass, ref_k_pe), dim=-1)

        if check_no_compile:
            with assert_no_triton_compile(
                _hy4_apply_interleaved_rope_inplace_kernel,
                _hy4_kv_latent_norm_and_rope_inplace_kernel,
            ):
                hy4_apply_interleaved_rope_inplace(q, cos_bf16, sin_bf16)
                hy4_kv_latent_norm_and_rope_inplace(
                    raw_kv, norm_w, cos_bf16, sin_bf16, eps=eps
                )
        else:
            hy4_apply_interleaved_rope_inplace(q, cos_bf16, sin_bf16)
            hy4_kv_latent_norm_and_rope_inplace(
                raw_kv, norm_w, cos_bf16, sin_bf16, eps=eps
            )

        assert torch.equal(q[..., :128], q_ref[..., :128])
        torch.testing.assert_close(q[..., 128:], q_ref[..., 128:], rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(raw_kv, kv_ref, rtol=1e-2, atol=1e-2)


@_CUDA
@pytest.mark.parametrize("q_len_per_req", [1, 6])
def test_hy4_fused_kpool_decode_pipeline_cuda(q_len_per_req: int) -> None:
    """Verify fused decode KPool update + finalize/M-RoPE/FP8 store against exact reference."""
    torch.manual_seed(31 + q_len_per_req)
    device = torch.device("cuda")
    kpool = 4
    tail_width = 8
    head_dim = 128
    index_rows_per_page = 16
    num_pages = 8
    mrope_section = (11, 11, 10)
    inv_freq = 1.0 / (
        1e7 ** (torch.arange(0, 64, 2, dtype=torch.float32, device=device) / 64)
    )

    w_val = torch.randn(head_dim, head_dim, dtype=torch.float32, device=device) * 0.1
    w_gate = torch.randn(head_dim, head_dim, dtype=torch.float32, device=device) * 0.1
    packed_w = torch.cat((w_val, w_gate), dim=0).contiguous()
    ape = torch.randn(kpool, head_dim, dtype=torch.float32, device=device) * 0.2
    key_norm_w = torch.randn(head_dim, dtype=torch.bfloat16, device=device) + 1.0

    def _run_case(seq_lens_list: list[int], check_no_compile: bool) -> None:
        num_reqs = len(seq_lens_list)
        num_req_slots = num_reqs + 2
        tail_fused = (
            torch.randn(
                num_req_slots, tail_width, head_dim, dtype=torch.bfloat16, device=device
            )
            * 0.3
        )
        gp0_fused = torch.randint(
            0, 50, (num_req_slots, 3), dtype=torch.int64, device=device
        )
        iv_fused = torch.zeros(
            num_pages,
            index_rows_per_page,
            head_dim,
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        is_fused = torch.zeros(
            num_pages,
            index_rows_per_page,
            1,
            dtype=index_k_scale_dtype(),
            device=device,
        )

        tail_ref = tail_fused.clone()
        gp0_ref = gp0_fused.clone()
        iv_ref = iv_fused.clone()
        is_ref = is_fused.clone()

        token_keys = (
            torch.randn(
                num_reqs * q_len_per_req, head_dim, dtype=torch.bfloat16, device=device
            )
            * 0.5
        )
        positions_3d = torch.randint(
            0, 200, (3, num_reqs * q_len_per_req), dtype=torch.int64, device=device
        )
        seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)
        req_slots = torch.tensor(
            [0 if s <= 0 else i + 1 for i, s in enumerate(seq_lens_list)],
            dtype=torch.int32,
            device=device,
        )
        index_table = (
            torch.arange(num_reqs * 2, dtype=torch.int32, device=device).view(
                num_reqs, 2
            )
            % num_pages
        )

        # Reference step-by-step path matching original write_decode_cache.
        keys_v = token_keys.view(num_reqs, q_len_per_req, head_dim)
        pos_v = positions_3d.view(3, num_reqs, q_len_per_req)
        sl64 = seq_lens.to(torch.int64)
        rs64 = req_slots.to(torch.int64)
        it64 = index_table.to(torch.int64)
        goff = torch.arange(kpool, dtype=torch.int64, device=device)
        r_arange = torch.arange(num_reqs, dtype=torch.int64, device=device)
        for step in range(q_len_per_req):
            tok_pos = (sl64 - q_len_per_req + step).clamp_min(0)
            valid_tok = (sl64 - q_len_per_req + step) >= 0
            tcol = torch.remainder(tok_pos, tail_width)
            tail_ref[rs64, tcol] = torch.where(
                valid_tok.unsqueeze(1), keys_v[:, step], tail_ref[rs64, tcol]
            )
            is_start = valid_tok & (torch.remainder(tok_pos, kpool) == 0)
            gp0_ref[rs64] = torch.where(
                is_start.unsqueeze(1), pos_v[:, :, step].T, gp0_ref[rs64]
            )
            is_end = valid_tok & (torch.remainder(tok_pos, kpool) == kpool - 1)
            gcols = torch.remainder(
                (tok_pos - (kpool - 1)).unsqueeze(1) + goff.unsqueeze(0), tail_width
            )
            grouped = tail_ref[rs64.unsqueeze(1), gcols]
            pooled = hy4_pool_group_keys(
                grouped, w_val, w_gate, ape, key_norm_w, eps=1e-5
            )
            cos, sin = compute_hy4_mrope_cos_sin(
                gp0_ref[rs64].T, inv_freq, mrope_section
            )
            pooled_rot = torch.cat(
                (
                    pooled[:, :64],
                    apply_interleaved_rope_last_dim(
                        pooled[:, 64:], cos.to(pooled.dtype), sin.to(pooled.dtype)
                    ),
                ),
                dim=-1,
            )
            pool_ids = torch.div(tok_pos, kpool, rounding_mode="floor")
            tcols = torch.div(
                pool_ids, index_rows_per_page, rounding_mode="floor"
            ).clamp(0, 1)
            raw_slots = it64[r_arange, tcols] * index_rows_per_page + torch.remainder(
                pool_ids, index_rows_per_page
            )
            masked_slots = torch.where(
                is_end, raw_slots, torch.full_like(raw_slots, -1)
            )
            hy4_quantize_and_store_index_k(pooled_rot, masked_slots, iv_ref, is_ref)

        # Fused path.
        if check_no_compile:
            with assert_no_triton_compile(
                _hy4_kpool_decode_update_gather_kernel,
                _hy4_kpool_finalize_rope_store_kernel,
            ):
                out_g, out_gp, out_slots = hy4_kpool_decode_update_and_gather(
                    token_keys,
                    positions_3d,
                    seq_lens,
                    req_slots,
                    index_table,
                    tail_fused,
                    gp0_fused,
                    index_rows_per_page=index_rows_per_page,
                    kpool=kpool,
                    q_len_per_req=q_len_per_req,
                )
                vg = F.linear(out_g.reshape(-1, head_dim).float(), packed_w).view(
                    out_g.shape[0], kpool, 2 * head_dim
                )
                hy4_kpool_finalize_rope_store(
                    vg,
                    ape,
                    key_norm_w,
                    out_gp,
                    inv_freq,
                    mrope_section,
                    out_slots,
                    iv_fused,
                    is_fused,
                    orig_dtype=token_keys.dtype,
                    eps=1e-5,
                )
        else:
            out_g, out_gp, out_slots = hy4_kpool_decode_update_and_gather(
                token_keys,
                positions_3d,
                seq_lens,
                req_slots,
                index_table,
                tail_fused,
                gp0_fused,
                index_rows_per_page=index_rows_per_page,
                kpool=kpool,
                q_len_per_req=q_len_per_req,
            )
            vg = F.linear(out_g.reshape(-1, head_dim).float(), packed_w).view(
                out_g.shape[0], kpool, 2 * head_dim
            )
            hy4_kpool_finalize_rope_store(
                vg,
                ape,
                key_norm_w,
                out_gp,
                inv_freq,
                mrope_section,
                out_slots,
                iv_fused,
                is_fused,
                orig_dtype=token_keys.dtype,
                eps=1e-5,
            )

        assert torch.equal(tail_fused, tail_ref)
        assert torch.equal(gp0_fused, gp0_ref)
        assert _fp8_e4m3fn_step_distance(iv_fused, iv_ref) <= 1
        torch.testing.assert_close(
            decode_index_k_scale(is_fused),
            decode_index_k_scale(is_ref),
            rtol=1e-2,
            atol=1e-4,
        )

    # First call compiles for q_len_per_req; second call varies num_reqs & exercises
    # padded rows (0), 0/1/2 closed groups, and page wrap with zero recompilation.
    _run_case([4, 12, 0], check_no_compile=False)
    _run_case([0, 5, 10, 12, 19, 24], check_no_compile=True)


@_CUDA
def test_hy4_sigmoid_bias_topk_cuda() -> None:
    """Verify fused 512-expert top-16 sigmoid+bias router parity and zero recompile."""
    torch.manual_seed(31)
    device = torch.device("cuda")
    num_experts, top_k = 512, 16
    bias = torch.randn(num_experts, dtype=torch.float32, device=device) * 0.05

    def _run(num_tokens: int) -> None:
        logits = torch.randn(
            num_tokens, num_experts, dtype=torch.float32, device=device
        )
        if num_tokens >= 2:
            # Exercise exact score tie-break and NaN squashing.
            logits[0, 5] = logits[0, 20] = 3.5
            bias[5] = bias[20] = 0.0
            logits[1, 3] = float("nan")
        got_w, got_ids = hy4_sigmoid_bias_topk(
            logits, bias, top_k=top_k, routed_scaling_factor=3.998, norm_topk_prob=True
        )
        scores = logits.sigmoid()
        choice = torch.where(
            torch.isnan(scores + bias),
            torch.full_like(scores, float("-inf")),
            scores + bias,
        )
        ref_ids = torch.topk(choice, k=top_k, dim=-1, sorted=True).indices
        ref_w = scores.gather(1, ref_ids)
        ref_w = (ref_w / (ref_w.sum(dim=-1, keepdim=True) + 1e-20)) * 3.998
        # Compare as sorted sets per row to allow identical-score ordering differences.
        for r in range(num_tokens):
            g_order = torch.argsort(got_ids[r])
            r_order = torch.argsort(ref_ids[r].to(torch.int32))
            assert torch.equal(got_ids[r][g_order], ref_ids[r].to(torch.int32)[r_order])
            torch.testing.assert_close(
                got_w[r][g_order], ref_w[r][r_order], rtol=1e-5, atol=1e-5
            )

    _run(3)
    with assert_no_triton_compile(_hy4_sigmoid_bias_topk_kernel):
        _run(1)
        _run(16)
        _run(65)


@_CUDA
@pytest.mark.parametrize("mn_major_tma", [False, True])
def test_hy4_sigmoid_gate_quantize_fp8_cuda(mn_major_tma: bool) -> None:
    """Verify fused sigmoid(gate)*attn_out + 1x128 FP8 quant (and Hopper DeepGEMM)."""
    torch.manual_seed(37)
    device = torch.device("cuda")
    dim = 4096

    def _run(num_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
        attn = torch.randn(num_tokens, dim, dtype=torch.bfloat16, device=device)
        gate = torch.randn(num_tokens, dim, dtype=torch.bfloat16, device=device)
        q_got, s_got = hy4_sigmoid_gate_quantize_fp8(
            attn, gate, mn_major_tma=mn_major_tma
        )
        y_ref = (attn * torch.sigmoid(gate)).float().view(num_tokens, dim // 128, 128)
        s_ref = (y_ref.abs().amax(dim=-1).clamp_min(1e-10) / 448.0).to(torch.float32)
        q_ref = (
            (y_ref / s_ref.unsqueeze(-1))
            .clamp(-448.0, 448.0)
            .to(torch.float8_e4m3fn)
            .view(num_tokens, dim)
        )
        torch.testing.assert_close(s_got, s_ref, rtol=1e-5, atol=1e-6)
        assert _fp8_e4m3fn_step_distance(q_got, q_ref) <= 1
        if mn_major_tma:
            assert s_got.stride(0) == 1
            assert s_got.stride(1) == max(((num_tokens + 3) // 4) * 4, 4)
        else:
            assert s_got.is_contiguous()
        return q_got, s_got

    _run(3)
    with assert_no_triton_compile(_hy4_sigmoid_gate_quantize_fp8_kernel):
        q6, s6 = _run(6)
        _run(32)

    if mn_major_tma and Platform.get().is_hopper:
        w = torch.randn(3072, dim, dtype=torch.bfloat16, device=device).to(
            torch.float8_e4m3fn
        )
        ws = (
            torch.rand(3072 // 128, dim // 128, dtype=torch.float32, device=device)
            * 0.05
            + 0.01
        )
        out_dg = mm(
            q6,
            w,
            A_scales=s6,
            B_scales=ws,
            out_dtype=torch.bfloat16,
            quant="mxfp8",
            block_size=[128, 128],
            solution="deepgemm",
        )
        out_tri = mm(
            q6,
            w,
            A_scales=s6.contiguous(),
            B_scales=ws,
            out_dtype=torch.bfloat16,
            quant="mxfp8",
            block_size=[128, 128],
        )
        torch.testing.assert_close(out_dg, out_tri, rtol=2e-2, atol=2e-2)


@_CUDA
@pytest.mark.parametrize("with_sink", [False, True])
def test_hy4_merge_attention_and_sink_cuda(with_sink: bool) -> None:
    """Verify fused DFly2 online-softmax merge + optional sink kernel."""
    torch.manual_seed(41)
    device = torch.device("cuda")
    heads, dim = 16, 512
    sink = torch.randn(heads, dtype=torch.float32, device=device) if with_sink else None

    def _run(bsz: int, seq_len: int) -> None:
        out_a = torch.randn(
            bsz, seq_len, heads, dim, dtype=torch.bfloat16, device=device
        )
        out_b = torch.randn(
            bsz, seq_len, heads, dim, dtype=torch.bfloat16, device=device
        )
        lse_a = (
            torch.randn(bsz, seq_len, heads, dtype=torch.float32, device=device) * 3.0
        )
        lse_b = (
            torch.randn(bsz, seq_len, heads, dtype=torch.float32, device=device) * 3.0
        )
        lse_a[0, 0, 0] = float("-inf")
        lse_a[0, 0, 1] = float("-inf")
        lse_b[0, 0, 1] = float("-inf")
        got = hy4_merge_attention_and_sink(out_a, lse_a, out_b, lse_b, sink)
        ref = hy4_merge_attention_and_sink(
            out_a.cpu(),
            lse_a.cpu(),
            out_b.cpu(),
            lse_b.cpu(),
            sink.cpu() if sink is not None else None,
        ).to(device)
        torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)
        assert not torch.isnan(got).any()

    _run(1, 6)
    with assert_no_triton_compile(_hy4_merge_attention_and_sink_kernel):
        _run(4, 6)
        _run(8, 6)


@_CUDA
@pytest.mark.parametrize("with_indexer", [False, True])
def test_hy4_split_qkv_indexer_norm_rope_cuda(with_indexer: bool) -> None:
    """Verify fused packed [q_a, kv_a(, tk, hw)] split + RMSNorm + M-RoPE."""
    torch.manual_seed(43)
    device = torch.device("cuda")
    q_lora, kv_lora, rope_dim = 1024, 512, 64
    tk_dim, hw_dim = (128, 16) if with_indexer else (0, 0)
    total_dim = q_lora + kv_lora + rope_dim + tk_dim + hw_dim
    qw = torch.randn(q_lora, dtype=torch.bfloat16, device=device) * 0.2 + 1.0
    kvw = torch.randn(kv_lora, dtype=torch.bfloat16, device=device) * 0.2 + 1.0
    tkw = (
        torch.randn(tk_dim, dtype=torch.bfloat16, device=device) * 0.2 + 1.0
        if with_indexer
        else None
    )

    def _run(num_tokens: int) -> None:
        packed = torch.randn(num_tokens, total_dim, dtype=torch.bfloat16, device=device)
        angles = torch.randn(
            num_tokens, rope_dim // 2, dtype=torch.float32, device=device
        )
        cos = torch.cos(angles).to(torch.bfloat16).contiguous()
        sin = torch.sin(angles).to(torch.bfloat16).contiguous()
        got = hy4_split_qkv_indexer_norm_rope(
            packed,
            qw,
            kvw,
            cos,
            sin,
            q_eps=1e-5,
            kv_eps=1e-5,
            tk_norm_weight=tkw,
            tk_eps=1e-5,
            hw_dim=hw_dim,
            weights_scale=0.022,
        )
        ref = hy4_split_qkv_indexer_norm_rope(
            packed.cpu(),
            qw.cpu(),
            kvw.cpu(),
            cos.cpu(),
            sin.cpu(),
            q_eps=1e-5,
            kv_eps=1e-5,
            tk_norm_weight=tkw.cpu() if tkw is not None else None,
            tk_eps=1e-5,
            hw_dim=hw_dim,
            weights_scale=0.022,
        )
        torch.testing.assert_close(got[0], ref[0].to(device), rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(got[1], ref[1].to(device), rtol=2e-2, atol=2e-2)
        if with_indexer:
            assert got[2] is not None and ref[2] is not None
            assert got[3] is not None and ref[3] is not None
            torch.testing.assert_close(got[2], ref[2].to(device), rtol=2e-2, atol=2e-2)
            torch.testing.assert_close(got[3], ref[3].to(device), rtol=1e-4, atol=1e-5)

    _run(4)
    with assert_no_triton_compile(_hy4_split_qkv_indexer_norm_rope_kernel):
        _run(1)
        _run(33)


@_CUDA
def test_hy4_split_routed_and_shared_swiglu_cuda() -> None:
    """Verify fused [routed_down; shared_gate_up] split + SiLU*up kernel."""
    torch.manual_seed(47)
    device = torch.device("cuda")
    routed_dim, shared_inter = 1536, 768

    def _run(num_tokens: int) -> None:
        packed = torch.randn(
            num_tokens,
            routed_dim + 2 * shared_inter,
            dtype=torch.bfloat16,
            device=device,
        )
        r_got, s_got = hy4_split_routed_and_shared_swiglu(packed, routed_dim)
        r_ref = packed[:, :routed_dim]
        g = packed[:, routed_dim : routed_dim + shared_inter].float()
        u = packed[:, routed_dim + shared_inter :].float()
        s_ref = (F.silu(g) * u).to(packed.dtype)
        assert torch.equal(r_got, r_ref)
        torch.testing.assert_close(s_got, s_ref, rtol=2e-2, atol=2e-2)

    _run(3)
    with assert_no_triton_compile(_hy4_split_routed_and_shared_swiglu_kernel):
        _run(1)
        _run(24)
        _run(96)


@_CUDA
def test_ihc_post_fused_norm_and_pre_fp8_quant_cuda() -> None:
    """Verify fused RMSNorm in ihc_post and fused 1x128 FP8 TMA quant in ihc_pre."""
    from tokenspeed_kernel.ops.residual.ihc import (
        _ihc_post_fused_norm_kernel,
        _ihc_pre_finalize_from_partials_kernel,
        _ihc_pre_reduce_kernel,
        ihc_post,
        ihc_pre,
    )

    torch.manual_seed(51)
    device = torch.device("cuda")
    hc_mult, hidden = 4, 3072
    fn = (
        torch.randn(2 * hc_mult, hc_mult * hidden, dtype=torch.float32, device=device)
        * 0.02
    )
    scale = torch.tensor([0.1, 0.2], dtype=torch.float32, device=device)
    base = torch.randn(2 * hc_mult, dtype=torch.float32, device=device) * 0.1
    nw = torch.randn(hidden, dtype=torch.bfloat16, device=device) * 0.2 + 1.0

    def _run_pre(num_tokens: int) -> None:
        streams = torch.randn(
            num_tokens, hc_mult, hidden, dtype=torch.bfloat16, device=device
        )
        post_ref, out_ref = ihc_pre(
            streams, fn, scale, base, norm_weight=nw, norm_eps=1e-5
        )
        post_got, out_got, q_fp8, q_scale = ihc_pre(
            streams,
            fn,
            scale,
            base,
            norm_weight=nw,
            norm_eps=1e-5,
            return_fp8_mn_major=True,
        )
        torch.testing.assert_close(post_got, post_ref, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(out_got, out_ref, rtol=0, atol=0)
        aligned_t = max(((num_tokens + 3) // 4) * 4, 4)
        assert q_scale.stride() == (1, aligned_t)
        g = out_ref.float().view(num_tokens, hidden // 128, 128)
        sc_ref = g.abs().amax(dim=-1).clamp_min(1e-10) / 448.0
        q_ref = (g / sc_ref.unsqueeze(-1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        torch.testing.assert_close(q_scale, sc_ref, rtol=1e-5, atol=1e-6)
        assert torch.equal(
            q_fp8.view(torch.uint8), q_ref.view(num_tokens, hidden).view(torch.uint8)
        )

    _run_pre(6)
    with assert_no_triton_compile(_ihc_pre_finalize_from_partials_kernel):
        _run_pre(1)
        _run_pre(24)
    _run_pre(48)
    with assert_no_triton_compile(_ihc_pre_reduce_kernel):
        _run_pre(64)

    def _run_post(num_tokens: int) -> None:
        res = torch.randn(
            num_tokens, hc_mult, hidden, dtype=torch.bfloat16, device=device
        )
        br = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=device)
        pw = torch.rand(num_tokens, hc_mult, dtype=torch.float32, device=device)
        bf = br.float()
        br_norm = (
            bf * torch.rsqrt(bf.pow(2).mean(dim=-1, keepdim=True) + 1e-5) * nw.float()
        ).to(br.dtype)
        ref = ihc_post(res.clone(), br_norm, pw, inplace=True)
        got = ihc_post(res.clone(), br, pw, inplace=True, norm_weight=nw, norm_eps=1e-5)
        torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)

    _run_post(6)
    with assert_no_triton_compile(_ihc_post_fused_norm_kernel):
        _run_post(1)
        _run_post(32)


@_CUDA
@pytest.mark.parametrize("with_indexer", [False, True])
def test_hy4_split_qkv_fused_fp8_quant_cuda(with_indexer: bool) -> None:
    """Verify fused q_resid 1x128 FP8 TMA quantization inside hy4_split_qkv_indexer_norm_rope."""
    torch.manual_seed(53)
    device = torch.device("cuda")
    q_lora, kv_lora, rope_dim = 1536, 512, 64
    tk_dim, hw_dim = (128, 16) if with_indexer else (0, 0)
    total_dim = q_lora + kv_lora + rope_dim + tk_dim + hw_dim
    qw = torch.randn(q_lora, dtype=torch.bfloat16, device=device) * 0.2 + 1.0
    kvw = torch.randn(kv_lora, dtype=torch.bfloat16, device=device) * 0.2 + 1.0
    tkw = (
        torch.randn(tk_dim, dtype=torch.bfloat16, device=device) * 0.2 + 1.0
        if with_indexer
        else None
    )

    def _run(num_tokens: int) -> None:
        packed = torch.randn(num_tokens, total_dim, dtype=torch.bfloat16, device=device)
        angles = torch.randn(
            num_tokens, rope_dim // 2, dtype=torch.float32, device=device
        )
        cos = torch.cos(angles).to(torch.bfloat16).contiguous()
        sin = torch.sin(angles).to(torch.bfloat16).contiguous()
        ref_q, ref_lat, ref_tk, ref_hw = hy4_split_qkv_indexer_norm_rope(
            packed,
            qw,
            kvw,
            cos,
            sin,
            q_eps=1e-5,
            kv_eps=1e-5,
            tk_norm_weight=tkw,
            tk_eps=1e-5,
            hw_dim=hw_dim,
            weights_scale=0.022,
        )
        got_q, got_lat, got_tk, got_hw, q_fp8, q_scale = (
            hy4_split_qkv_indexer_norm_rope(
                packed,
                qw,
                kvw,
                cos,
                sin,
                q_eps=1e-5,
                kv_eps=1e-5,
                tk_norm_weight=tkw,
                tk_eps=1e-5,
                hw_dim=hw_dim,
                weights_scale=0.022,
                quantize_q_fp8_mn_major=True,
                write_q_bf16=with_indexer,
            )
        )
        if with_indexer:
            assert got_q is not None
            torch.testing.assert_close(got_q, ref_q, rtol=0, atol=0)
        else:
            assert got_q is None
        torch.testing.assert_close(got_lat, ref_lat, rtol=0, atol=0)
        aligned_t = max(((num_tokens + 3) // 4) * 4, 4)
        assert q_scale.stride() == (1, aligned_t)
        g = ref_q.float().view(num_tokens, q_lora // 128, 128)
        sc_ref = g.abs().amax(dim=-1).clamp_min(1e-10) / 448.0
        q_ref = (g / sc_ref.unsqueeze(-1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        torch.testing.assert_close(q_scale, sc_ref, rtol=1e-5, atol=1e-6)
        assert torch.equal(
            q_fp8.view(torch.uint8), q_ref.view(num_tokens, q_lora).view(torch.uint8)
        )

    _run(6)
    with assert_no_triton_compile(_hy4_split_qkv_indexer_norm_rope_kernel):
        _run(1)
        _run(24)


@_CUDA
def test_hy4_kpool_project_finalize_rope_store_cuda() -> None:
    """Verify fused KPool FP32 projection + finalize + M-RoPE + FP8 scatter with early exit."""
    torch.manual_seed(57)
    device = torch.device("cuda")
    head_dim, kpool = 128, 4
    packed_w = (
        torch.randn(2 * head_dim, head_dim, dtype=torch.float32, device=device) * 0.1
    )
    packed_wt = packed_w.t().contiguous()
    ape = torch.randn(kpool, head_dim, dtype=torch.bfloat16, device=device) * 0.1
    knw = torch.randn(head_dim, dtype=torch.bfloat16, device=device) * 0.2 + 1.0
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, 64, 2, device=device, dtype=torch.float32) / 64.0)
    )
    mrope_section = (11, 11, 10)

    def _run(num_groups: int) -> None:
        gk = torch.randn(
            num_groups, kpool, head_dim, dtype=torch.bfloat16, device=device
        )
        gp = torch.randint(0, 500, (3, num_groups), dtype=torch.int64, device=device)
        slots = torch.tensor(
            [(-1 if i % 2 == 0 else i + 3) for i in range(num_groups)],
            dtype=torch.int32,
            device=device,
        )
        iv_ref = torch.zeros(4, 16, head_dim, dtype=torch.float8_e4m3fn, device=device)
        is_ref = torch.zeros(4, 16, 1, dtype=torch.float32, device=device)
        vg = F.linear(gk.reshape(-1, head_dim).float(), packed_w).view(
            num_groups, kpool, 2 * head_dim
        )
        hy4_kpool_finalize_rope_store(
            vg,
            ape,
            knw,
            gp,
            inv_freq,
            mrope_section,
            slots,
            iv_ref,
            is_ref,
            orig_dtype=gk.dtype,
            eps=1e-5,
        )
        iv_got = torch.zeros_like(iv_ref)
        is_got = torch.zeros_like(is_ref)
        hy4_kpool_project_finalize_rope_store(
            gk,
            packed_wt,
            ape,
            knw,
            gp,
            inv_freq,
            mrope_section,
            slots,
            iv_got,
            is_got,
            eps=1e-5,
        )
        torch.testing.assert_close(
            iv_got.float() * is_got,
            iv_ref.float() * is_ref,
            rtol=5e-2,
            atol=5e-2,
        )

    _run(4)
    with assert_no_triton_compile(
        _hy4_kpool_project_finalize_rope_store_kernel,
        _hy4_kpool_project_finalize_rope_store_splitn_kernel,
    ):
        _run(2)
        _run(8)


@_CUDA
@pytest.mark.parametrize("is_mrope", [False, True])
def test_hy4_dfly_build_decode_round_inputs_cuda(is_mrope: bool) -> None:
    """Verify fused pure-decode DFly2 round metadata kernel."""
    torch.manual_seed(59)
    device = torch.device("cuda")
    verify_width = 6

    def _run(bs: int) -> None:
        out_toks = torch.randint(
            0, 1000, (bs * verify_width,), dtype=torch.int32, device=device
        )
        acc = torch.randint(
            1, verify_width + 1, (bs,), dtype=torch.int32, device=device
        )
        seq_lens = torch.randint(
            verify_width, 500, (bs,), dtype=torch.int32, device=device
        )
        if is_mrope:
            pos_buf = torch.randint(
                0, 500, (3, bs * verify_width), dtype=torch.int64, device=device
            )
        else:
            pos_buf = torch.randint(
                0, 500, (bs * verify_width,), dtype=torch.int64, device=device
            )
        a_got, r_got, c_got, bp_got = hy4_dfly_build_decode_round_inputs(
            out_toks,
            acc,
            seq_lens,
            pos_buf,
            bs=bs,
            verify_width=verify_width,
            hidden_rows=bs * verify_width,
            is_mrope=is_mrope,
        )
        dec_arange = torch.arange(bs, dtype=torch.int64, device=device)
        dec_acc = acc.to(torch.int64).clamp(1, verify_width)
        r_ref = dec_arange * verify_width + (dec_acc - 1)
        a_ref = out_toks[r_ref].to(torch.int32)
        c_ref = (seq_lens - verify_width + dec_acc.to(torch.int32)).clamp_min(0)
        offsets = torch.arange(
            1, verify_width + 1, dtype=torch.int64, device=device
        ).view(1, 1, verify_width)
        last_pos = (
            pos_buf[:, r_ref] if is_mrope else pos_buf[r_ref].unsqueeze(0).expand(3, -1)
        )
        bp_ref = last_pos.unsqueeze(-1) + offsets
        assert torch.equal(a_got, a_ref)
        assert torch.equal(r_got, r_ref)
        assert torch.equal(c_got, c_ref)
        assert torch.equal(bp_got, bp_ref)

    _run(4)
    with assert_no_triton_compile(_hy4_dfly_build_decode_round_inputs_kernel):
        _run(1)
        _run(16)


@_CUDA
@pytest.mark.parametrize(
    ("num_heads", "q_len", "seq_len", "window_left"),
    [
        (32, 6, 38, 511),
        (32, 6, 600, 511),
        (32, 1, 700, 511),
        (16, 6, 1024, -1),
    ],
)
def test_mla_decode_tc_matches_reference_cuda(
    num_heads: int, q_len: int, seq_len: int, window_left: int
) -> None:
    """Verify Tensor-Core grouped-head split-K MLA decode matches exact FP32 reference."""
    from tokenspeed_kernel.ops.attention.mla import mla_decode_with_kvcache
    from tokenspeed_kernel.ops.attention.mla._triton.decode import (
        _mla_decode_merge_splits_kernel,
        _mla_decode_tc_kernel,
    )

    torch.manual_seed(61)
    device = torch.device("cuda")
    bs, page_size = 2, 64
    num_pages = (seq_len + page_size - 1) // page_size + 1
    kv = (
        torch.randn(
            bs * num_pages, page_size, 1, 576, dtype=torch.bfloat16, device=device
        )
        * 0.2
    )
    bt = torch.arange(bs * num_pages, dtype=torch.int32, device=device).view(
        bs, num_pages
    )
    seql = torch.tensor(
        [seq_len, max(q_len, seq_len // 2)], dtype=torch.int32, device=device
    )
    q = (
        torch.randn(bs, q_len, num_heads, 576, dtype=torch.bfloat16, device=device)
        * 0.2
    )
    scale = 576**-0.5

    got_o, got_lse = mla_decode_with_kvcache(
        q,
        kv,
        bt,
        seql,
        seq_len,
        128,
        512,
        64,
        scale,
        return_lse=True,
        window_left=window_left,
        solution="triton",
    )
    # Exact FP32 reference
    ref_o = torch.zeros_like(got_o, dtype=torch.float32)
    ref_lse = torch.full_like(got_lse, float("-inf"))
    for b in range(bs):
        clen = int(seql[b].item())
        k_seq = kv[bt[b].long()].reshape(-1, 576)[:clen].float()
        for qi in range(q_len):
            q_clen = max(0, clen - (q_len - 1 - qi))
            w_start = max(0, q_clen - 1 - window_left) if window_left >= 0 else 0
            k_win = k_seq[w_start:q_clen]
            scores = (q[b, qi].float() @ k_win.t()) * scale
            lse_v = torch.logsumexp(scores, dim=-1)
            probs = torch.softmax(scores, dim=-1)
            ref_o[b, qi] = probs @ k_win[:, :512]
            ref_lse[b, qi] = lse_v

    torch.testing.assert_close(got_o.float(), ref_o, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(got_lse, ref_lse, rtol=1e-3, atol=2e-3)

    # Zero-recompilation across varying batch/seqlens
    seql2 = torch.tensor([max(q_len, seq_len - 7), 0], dtype=torch.int32, device=device)
    with assert_no_triton_compile(
        _mla_decode_tc_kernel, _mla_decode_merge_splits_kernel
    ):
        o2, lse2 = mla_decode_with_kvcache(
            q,
            kv,
            bt,
            seql2,
            seq_len,
            128,
            512,
            64,
            scale,
            return_lse=True,
            window_left=window_left,
            solution="triton",
        )
    assert (o2[1] == 0).all()
    assert torch.isneginf(lse2[1]).all()


@_CUDA
@pytest.mark.parametrize("has_post_norm", [False, True])
@pytest.mark.parametrize("return_fp8", [False, True])
def test_ihc_post_and_pre_cuda(has_post_norm: bool, return_fp8: bool) -> None:
    """Verify fused ihc_post_and_pre matches separate ihc_post + ihc_pre with zero recompile."""
    torch.manual_seed(71)
    device = torch.device("cuda")
    hc_mult, hidden = 4, 3072
    post_nw = (
        torch.randn(hidden, dtype=torch.bfloat16, device=device) * 0.2 + 1.0
        if has_post_norm
        else None
    )
    pre_nw = torch.randn(hidden, dtype=torch.bfloat16, device=device) * 0.2 + 1.0
    next_fn = (
        torch.randn(2 * hc_mult, hc_mult * hidden, dtype=torch.bfloat16, device=device)
        * 0.02
    )
    next_scale = torch.tensor([0.3, 0.4], dtype=torch.float32, device=device)
    next_base = torch.randn(2 * hc_mult, dtype=torch.float32, device=device) * 0.1

    def _run(num_tokens: int) -> None:
        streams = torch.randn(
            num_tokens, hc_mult, hidden, dtype=torch.bfloat16, device=device
        )
        branch = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=device)
        post_w = (
            torch.rand(num_tokens, hc_mult, dtype=torch.float32, device=device) + 0.1
        )

        ref_streams = ihc_post(
            streams.clone(),
            branch,
            post_w,
            inplace=True,
            norm_weight=post_nw,
            norm_eps=1e-5,
        )
        ref_pre = ihc_pre(
            ref_streams,
            next_fn,
            next_scale,
            next_base,
            norm_weight=pre_nw,
            norm_eps=1e-5,
            return_fp8_mn_major=return_fp8,
        )

        got_streams = streams.clone()
        got = ihc_post_and_pre(
            got_streams,
            branch,
            post_w,
            next_fn,
            next_scale,
            next_base,
            inplace=True,
            post_norm_weight=post_nw,
            post_norm_eps=1e-5,
            pre_norm_weight=pre_nw,
            pre_norm_eps=1e-5,
            return_fp8_mn_major=return_fp8,
        )
        if num_tokens > 0:
            assert got[0].data_ptr() == got_streams.data_ptr()
        if has_post_norm:
            torch.testing.assert_close(got[0], ref_streams, rtol=1.6e-2, atol=1e-2)
        else:
            torch.testing.assert_close(got[0], ref_streams, rtol=0, atol=0)
        torch.testing.assert_close(got[1], ref_pre[0], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(got[2], ref_pre[1], rtol=2e-2, atol=2e-2)
        if return_fp8:
            max_step = 2 if has_post_norm else 1
            assert _fp8_e4m3fn_step_distance(got[3], ref_pre[2]) <= max_step
            torch.testing.assert_close(got[4], ref_pre[3], rtol=1e-3, atol=1e-3)

    _run(6)
    _run(24)
    _run(192)
    _run(0)
    with assert_no_triton_compile(_ihc_post_and_pre_fused_kernel):
        _run(1)
        _run(4)
        _run(16)
        _run(96)
        _run(160)

    # Verify CUDA Graph multi-replay safety of atomic arrival counters.
    streams_g = torch.randn(6, hc_mult, hidden, dtype=torch.bfloat16, device=device)
    branch_g = torch.randn(6, hidden, dtype=torch.bfloat16, device=device)
    post_g = torch.rand(6, hc_mult, dtype=torch.float32, device=device) + 0.1
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        g_out = ihc_post_and_pre(
            streams_g,
            branch_g,
            post_g,
            next_fn,
            next_scale,
            next_base,
            inplace=False,
            post_norm_weight=post_nw,
            post_norm_eps=1e-5,
            pre_norm_weight=pre_nw,
            pre_norm_eps=1e-5,
            return_fp8_mn_major=return_fp8,
        )
    for scale in (0.5, -1.2, 1.8):
        streams_g.copy_(
            torch.randn(6, hc_mult, hidden, dtype=torch.bfloat16, device=device) * scale
        )
        branch_g.copy_(
            torch.randn(6, hidden, dtype=torch.bfloat16, device=device) * scale
        )
        g.replay()
        ref_s = ihc_post(
            streams_g.clone(),
            branch_g,
            post_g,
            inplace=True,
            norm_weight=post_nw,
            norm_eps=1e-5,
        )
        ref_p = ihc_pre(
            ref_s,
            next_fn,
            next_scale,
            next_base,
            norm_weight=pre_nw,
            norm_eps=1e-5,
            return_fp8_mn_major=return_fp8,
        )
        torch.testing.assert_close(g_out[1], ref_p[0], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(g_out[2], ref_p[1], rtol=2e-2, atol=2e-2)


@_CUDA
def test_mla_decode_tc_swa_single_pass_large_batch_cuda() -> None:
    """Verify SWA MLA decode single-pass path (num_splits=1 when bs*q_len >= 64)."""
    from tokenspeed_kernel.ops.attention.mla import mla_decode_with_kvcache

    torch.manual_seed(73)
    device = torch.device("cuda")
    bs, q_len, num_heads, page_size, seq_len, window_left = 12, 6, 32, 64, 192, 127
    num_pages = seq_len // page_size
    kv = (
        torch.randn(
            bs * num_pages, page_size, 1, 576, dtype=torch.bfloat16, device=device
        )
        * 0.2
    )
    bt = torch.arange(bs * num_pages, dtype=torch.int32, device=device).view(
        bs, num_pages
    )
    seql = torch.full((bs,), seq_len, dtype=torch.int32, device=device)
    seql[1] = 17
    seql[2] = 0
    q = (
        torch.randn(bs, q_len, num_heads, 576, dtype=torch.bfloat16, device=device)
        * 0.2
    )
    scale = 576**-0.5

    got_o, got_lse = mla_decode_with_kvcache(
        q,
        kv,
        bt,
        seql,
        seq_len,
        128,
        512,
        64,
        scale,
        return_lse=True,
        window_left=window_left,
        solution="triton",
    )
    assert (got_o[2] == 0).all()
    assert torch.isneginf(got_lse[2]).all()
    for b in (0, 1):
        clen = int(seql[b].item())
        k_seq = kv[bt[b]].reshape(-1, 576)[:clen].float()
        for qi in range(q_len):
            q_clen = max(0, clen - (q_len - 1 - qi))
            w_start = max(0, q_clen - 1 - window_left)
            k_win = k_seq[w_start:q_clen]
            scores = (q[b, qi].float() @ k_win.t()) * scale
            ref_lse = torch.logsumexp(scores, dim=-1)
            ref_o = torch.softmax(scores, dim=-1) @ k_win[:, :512]
            torch.testing.assert_close(
                got_o[b, qi].float(), ref_o, rtol=2e-2, atol=2e-2
            )
            torch.testing.assert_close(got_lse[b, qi], ref_lse, rtol=1e-3, atol=2e-3)


@_CUDA
def test_hy4_vocab_argmax_cuda() -> None:
    """Verify multi-CTA full and TP-distributed vocab argmax plus tie-breaking."""
    from tokenspeed_kernel.ops.model.hy4_flash.dfly import (
        hy4_vocab_argmax_into,
    )

    torch.manual_seed(79)
    device = torch.device("cuda")
    vocab_size = 120832
    half_v = vocab_size // 2

    for bsz in (0, 1, 4, 16, 32):
        logits = torch.randn(bsz, vocab_size, dtype=torch.float32, device=device)
        if bsz >= 4:
            # Force exact cross-shard value ties (positive, negative, -0.0/+0.0).
            logits[0, :] = -10.0
            logits[0, 500] = 3.5
            logits[0, half_v + 100] = 3.5
            logits[1, :] = -10.0
            logits[1, 100] = -1.25
            logits[1, half_v + 100] = -1.25
            logits[2, :] = -10.0
            logits[2, 200] = -0.0
            logits[2, half_v + 5] = 0.0

        ref = logits.argmax(dim=-1).to(torch.int32)
        proposals = torch.full((bsz, 6), -1, dtype=torch.int32, device=device)
        contig = torch.full((bsz,), -1, dtype=torch.int32, device=device)
        hy4_vocab_argmax_into(
            logits, proposals[:, 2], out_contig=contig, max_token_id=vocab_size - 1
        )
        assert torch.equal(proposals[:, 2], ref)
        assert torch.equal(contig, ref)


@_CUDA
def test_hy4_dual_rmsnorm_cat_cuda() -> None:
    """Verify fused dual RMSNorm + concat matches separate RMSNorm + torch.cat."""
    from tokenspeed_kernel.ops.model.hy4_flash.dfly import hy4_dual_rmsnorm_cat

    torch.manual_seed(83)
    device = torch.device("cuda")
    hidden = 3072
    wa = torch.randn(hidden, dtype=torch.bfloat16, device=device) * 0.2 + 1.0
    wb = torch.randn(hidden, dtype=torch.bfloat16, device=device) * 0.2 + 1.0
    for bsz in (0, 1, 6, 32):
        a = torch.randn(bsz, hidden, dtype=torch.bfloat16, device=device)
        b = torch.randn(bsz, hidden, dtype=torch.bfloat16, device=device)
        got = hy4_dual_rmsnorm_cat(a, b, wa, wb, eps=1e-5)
        af, bf = a.float(), b.float()
        an = (
            af * torch.rsqrt(af.pow(2).mean(dim=-1, keepdim=True) + 1e-5) * wa.float()
        ).to(a.dtype)
        bn = (
            bf * torch.rsqrt(bf.pow(2).mean(dim=-1, keepdim=True) + 1e-5) * wb.float()
        ).to(b.dtype)
        ref = torch.cat((an, bn), dim=-1)
        torch.testing.assert_close(got, ref, rtol=1.6e-2, atol=1e-2)


@_CUDA
@pytest.mark.parametrize("bs", [2, 12])
def test_mla_decode_tc_fused_sink_cuda(bs: int) -> None:
    """Verify fused sink rescaling in Triton MLA decode (split-K and single-pass)."""
    from tokenspeed_kernel.ops.attention.mla._triton.decode import (
        _triton_mla_decode_with_kvcache_impl,
    )

    torch.manual_seed(89)
    device = torch.device("cuda")
    q_len, num_heads, page_size, seq_len, window_left = 6, 32, 64, 192, 127
    num_pages = seq_len // page_size
    kv = (
        torch.randn(
            bs * num_pages, page_size, 1, 576, dtype=torch.bfloat16, device=device
        )
        * 0.2
    )
    bt = torch.arange(bs * num_pages, dtype=torch.int32, device=device).view(
        bs, num_pages
    )
    seql = torch.full((bs,), seq_len, dtype=torch.int32, device=device)
    seql[1] = 0
    q = (
        torch.randn(bs, q_len, num_heads, 576, dtype=torch.bfloat16, device=device)
        * 0.2
    )
    sinks = torch.randn(num_heads, dtype=torch.float32, device=device) * 2.0
    scale = 576**-0.5

    ref_o, ref_lse = _triton_mla_decode_with_kvcache_impl(
        q,
        kv,
        bt,
        seql,
        seq_len,
        128,
        512,
        64,
        scale,
        return_lse=True,
        window_left=window_left,
    )
    apply_attention_sink(
        ref_o.reshape(-1, num_heads, 512),
        ref_lse.reshape(-1, num_heads),
        sinks,
        inplace=True,
    )
    got_o = _triton_mla_decode_with_kvcache_impl(
        q,
        kv,
        bt,
        seql,
        seq_len,
        128,
        512,
        64,
        scale,
        return_lse=False,
        window_left=window_left,
        sinks=sinks,
    )
    assert (got_o[1] == 0).all()
    torch.testing.assert_close(got_o, ref_o, rtol=1.6e-2, atol=1e-2)


@_CUDA
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("with_sink", [False, True])
def test_hy4_dfly_block_attn_merge_sink_cuda(causal: bool, with_sink: bool) -> None:
    """Verify fused DFly2 intra-block absorbed attention + context merge + sink."""
    from tokenspeed_kernel.ops.model.hy4_flash.dfly import (
        hy4_dfly_block_attn_merge_sink,
    )

    torch.manual_seed(107)
    device = torch.device("cuda")
    heads, d_v, d_qk = 32, 512, 576
    scaling = 192**-0.5
    sink = torch.randn(heads, dtype=torch.float32, device=device) if with_sink else None
    for bsz, block_size in ((1, 6), (4, 6), (2, 16), (1, 1)):
        q_abs = (
            torch.randn(
                bsz, block_size, heads, d_qk, dtype=torch.bfloat16, device=device
            )
            * 0.2
        )
        latent = (
            torch.randn(bsz, block_size, d_qk, dtype=torch.bfloat16, device=device)
            * 0.2
        )
        out_ctx = (
            torch.randn(
                bsz, block_size, heads, d_v, dtype=torch.bfloat16, device=device
            )
            * 0.2
        )
        lse_ctx = torch.randn(
            bsz, block_size, heads, dtype=torch.float32, device=device
        )
        lse_ctx[0, 0, 0] = float("-inf")

        got = hy4_dfly_block_attn_merge_sink(
            q_abs, latent, out_ctx, lse_ctx, sink, scaling=scaling, causal=causal
        )
        ref = hy4_dfly_block_attn_merge_sink(
            q_abs.cpu(),
            latent.cpu(),
            out_ctx.cpu(),
            lse_ctx.cpu(),
            sink.cpu() if sink is not None else None,
            scaling=scaling,
            causal=causal,
        ).to(device)
        torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)
        assert not torch.isnan(got).any()


@_CUDA
@pytest.mark.parametrize("with_residual", [False, True])
@pytest.mark.parametrize("write_bf16", [False, True])
def test_hy4_add_rmsnorm_and_silu_mul_quantize_fp8_cuda(
    with_residual: bool, write_bf16: bool
) -> None:
    """Verify fused DFly2 add+RMSNorm+FP8 and SiLU-mul+FP8 quantization."""
    from tokenspeed_kernel.ops.model.hy4_flash.fusion import (
        hy4_add_rmsnorm_quantize_fp8,
        hy4_silu_and_mul_quantize_fp8,
    )

    torch.manual_seed(113)
    device = torch.device("cuda")
    dim = 3072
    w = torch.randn(dim, dtype=torch.bfloat16, device=device) * 0.2 + 1.0
    for bsz, seq in ((1, 6), (4, 6), (0, 6)):
        x = torch.randn(bsz, seq, dim, dtype=torch.bfloat16, device=device)
        res = (
            torch.randn(bsz, seq, dim, dtype=torch.bfloat16, device=device)
            if with_residual
            else None
        )
        got_res, got_norm, got_q, got_s = hy4_add_rmsnorm_quantize_fp8(
            x, w, residual=res, eps=1e-5, mn_major_tma=True, write_bf16=write_bf16
        )
        ref_res, ref_norm, ref_q, ref_s = hy4_add_rmsnorm_quantize_fp8(
            x.cpu(),
            w.cpu(),
            residual=res.cpu() if res is not None else None,
            eps=1e-5,
            mn_major_tma=True,
            write_bf16=write_bf16,
        )
        if with_residual:
            assert got_res is not None and ref_res is not None
            torch.testing.assert_close(got_res, ref_res.to(device), rtol=0, atol=0)
        if write_bf16:
            assert got_norm is not None and ref_norm is not None
            torch.testing.assert_close(
                got_norm, ref_norm.to(device), rtol=1.6e-2, atol=1e-2
            )
        if bsz > 0:
            torch.testing.assert_close(got_s, ref_s.to(device), rtol=2e-2, atol=1e-4)
            assert _fp8_e4m3fn_step_distance(got_q, ref_q.to(device)) <= 1

    gu = torch.randn(17, 2 * dim, dtype=torch.bfloat16, device=device)
    gq, gs = hy4_silu_and_mul_quantize_fp8(gu, mn_major_tma=True)
    rq, rs = hy4_silu_and_mul_quantize_fp8(gu.cpu(), mn_major_tma=True)
    torch.testing.assert_close(gs, rs.to(device), rtol=2e-2, atol=1e-4)
    assert _fp8_e4m3fn_step_distance(gq, rq.to(device)) <= 1


@_CUDA
@pytest.mark.parametrize("fp8_rt", [False, True])
def test_hy4_engram_fused_kernels_cuda(fp8_rt: bool) -> None:
    """Verify fused Engram hash/embed, lookup norm flatten, and gate combine kernels."""
    from tokenspeed_kernel.ops.model.hy4_flash.engram import (
        hy4_engram_gate_combine,
        hy4_engram_hash_and_embed,
        hy4_engram_hash_ngrams,
        hy4_engram_lookup_norm_flatten,
    )

    torch.manual_seed(127)
    device = torch.device("cuda")
    t, max_ng, n_h_per_ng, head_dim, hc_mult, hidden = 9, 3, 8, 96, 4, 3072
    num_heads = (max_ng - 1) * n_h_per_ng
    table_size, pad_id = 512, 17
    lookup = torch.randint(0, 256, (table_size,), dtype=torch.int32, device=device)
    mult = torch.tensor([1000003, 2000003, 3000017], dtype=torch.int64, device=device)
    moduli = torch.tensor(
        [31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101],
        dtype=torch.int64,
        device=device,
    )
    offsets = torch.cat(
        (torch.zeros(1, dtype=torch.int64, device=device), moduli[:-1].cumsum(0))
    )
    total_rows = int(moduli.sum().item())
    row_start, row_end = total_rows // 3, (2 * total_rows) // 3
    weight = torch.randn(
        row_end - row_start, head_dim, dtype=torch.bfloat16, device=device
    )

    ngram_tokens = torch.randint(
        -5, table_size + 5, (t, max_ng), dtype=torch.int64, device=device
    )
    got_ids = hy4_engram_hash_ngrams(
        ngram_tokens,
        lookup,
        mult,
        moduli,
        pad_id=pad_id,
        max_ngram_size=max_ng,
        n_head_per_ngram=n_h_per_ng,
    )
    clamped = ngram_tokens.clamp(0, table_size - 1)
    valid = (ngram_tokens >= 0) & (ngram_tokens < table_size)
    comp = torch.where(valid, lookup[clamped], lookup[pad_id]).to(torch.int64)
    mixed = comp * mult
    ref_parts = []
    for ng in range(2, max_ng + 1):
        m_acc = mixed[..., 0]
        for off in range(1, ng):
            m_acc = torch.bitwise_xor(m_acc, mixed[..., off])
        s = (ng - 2) * n_h_per_ng
        ref_parts.append(m_acc.unsqueeze(-1) % moduli[s : s + n_h_per_ng])
    ref_ids = torch.cat(ref_parts, dim=-1)
    assert torch.equal(got_ids, ref_ids)

    got_emb = hy4_engram_hash_and_embed(
        ngram_tokens,
        lookup,
        mult,
        moduli,
        offsets,
        weight,
        pad_id=pad_id,
        max_ngram_size=max_ng,
        n_head_per_ngram=n_h_per_ng,
        row_start=row_start,
        row_end=row_end,
        out_dtype=torch.bfloat16,
    )
    gids = ref_ids + offsets
    in_shard = (gids >= row_start) & (gids < row_end)
    ref_emb = weight[(gids - row_start).clamp(0, weight.shape[0] - 1)].masked_fill(
        ~in_shard.unsqueeze(-1), 0.0
    )
    assert torch.equal(got_emb, ref_emb)

    lnw = (
        torch.randn(1, num_heads, head_dim, dtype=torch.bfloat16, device=device) * 0.2
        + 1.0
    )
    got_flat = hy4_engram_lookup_norm_flatten(got_emb, lnw, fp8_round_trip=fp8_rt)
    emb_ref_in = got_emb
    if fp8_rt:
        fmax = torch.finfo(torch.float8_e4m3fn).max
        sc = emb_ref_in.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / fmax
        emb_ref_in = (emb_ref_in / sc).clamp(-fmax, fmax).to(torch.float8_e4m3fn).to(
            emb_ref_in.dtype
        ) * sc
    ref_flat = (torch.nn.functional.rms_norm(emb_ref_in, (head_dim,)) * lnw).flatten(
        start_dim=-2
    )
    torch.testing.assert_close(got_flat, ref_flat, rtol=2e-2, atol=2e-2)

    kv_out = torch.randn(t, (hc_mult + 1) * hidden, dtype=torch.bfloat16, device=device)
    hidden_st = torch.randn(t, hc_mult, hidden, dtype=torch.bfloat16, device=device)
    nkw = (
        torch.randn(1, hc_mult, hidden, dtype=torch.bfloat16, device=device) * 0.2 + 1.0
    )
    nqw = (
        torch.randn(1, hc_mult, hidden, dtype=torch.bfloat16, device=device) * 0.2 + 1.0
    )
    pg = torch.randn(hidden, dtype=torch.bfloat16, device=device) * 0.2 + 1.0
    bs = torch.randn(1, hc_mult, hidden, dtype=torch.bfloat16, device=device) * 0.2
    mask = torch.tensor(
        [True, False, True, True, False, True, False, True, True], device=device
    )

    got_out = hy4_engram_gate_combine(
        kv_out, hidden_st, nkw, nqw, pg, bs, mask, inv_sqrt_d=hidden**-0.5
    )
    keys = (
        torch.nn.functional.rms_norm(
            kv_out[:, : hc_mult * hidden].view(t, hc_mult, hidden), (hidden,)
        )
        * nkw
    )
    queries = torch.nn.functional.rms_norm(hidden_st, (hidden,)) * nqw
    gate = (keys * queries).sum(dim=-1) * (hidden**-0.5)
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    main_sig = pg * torch.nn.functional.rms_norm(
        kv_out[:, hc_mult * hidden :], (hidden,)
    )
    ref_out = (
        gate.sigmoid().unsqueeze(-1) * main_sig.unsqueeze(1)
        + main_sig.unsqueeze(1) * bs
    ).masked_fill(~mask[:, None, None], 0.0)
    assert (got_out[~mask] == 0).all()
    torch.testing.assert_close(got_out, ref_out, rtol=2e-2, atol=2e-2)

    got_added = hy4_engram_gate_combine(
        kv_out,
        hidden_st,
        nkw,
        nqw,
        pg,
        bs,
        mask,
        inv_sqrt_d=hidden**-0.5,
        add_to_hidden=True,
    )
    torch.testing.assert_close(got_added[~mask], hidden_st[~mask], rtol=0, atol=0)
    torch.testing.assert_close(got_added, hidden_st + ref_out, rtol=2e-2, atol=2e-2)


@_CUDA
def test_hy4_dfly_block_attn_merge_sink_zero_ctx_lens_cuda() -> None:
    """Verify ctx_lens<=0 skips reading uninitialized/NaN out_ctx rows safely."""
    from tokenspeed_kernel.ops.model.hy4_flash.dfly import (
        hy4_dfly_block_attn_merge_sink,
    )

    torch.manual_seed(137)
    device = torch.device("cuda")
    bsz, block_size, heads, d_v, d_qk = 3, 6, 16, 512, 576
    scaling = 192**-0.5
    q_abs = (
        torch.randn(bsz, block_size, heads, d_qk, dtype=torch.bfloat16, device=device)
        * 0.2
    )
    latent = (
        torch.randn(bsz, block_size, d_qk, dtype=torch.bfloat16, device=device) * 0.2
    )
    out_ctx = (
        torch.randn(bsz, block_size, heads, d_v, dtype=torch.bfloat16, device=device)
        * 0.2
    )
    lse_ctx = torch.randn(bsz, block_size, heads, dtype=torch.float32, device=device)
    sink = torch.randn(heads, dtype=torch.float32, device=device)
    ctx_lens = torch.tensor([128, 0, 64], dtype=torch.int32, device=device)
    # Poison the zero-length context row with NaN to ensure the kernel skips it.
    out_ctx[1].fill_(float("nan"))
    lse_ctx[1].fill_(float("nan"))

    got = hy4_dfly_block_attn_merge_sink(
        q_abs,
        latent,
        out_ctx,
        lse_ctx,
        sink,
        scaling=scaling,
        causal=True,
        ctx_lens=ctx_lens,
    )
    out_ctx_clean = out_ctx.clone()
    lse_ctx_clean = lse_ctx.clone()
    out_ctx_clean[1].zero_()
    lse_ctx_clean[1].fill_(float("-inf"))
    ref = hy4_dfly_block_attn_merge_sink(
        q_abs.cpu(),
        latent.cpu(),
        out_ctx_clean.cpu(),
        lse_ctx_clean.cpu(),
        sink.cpu(),
        scaling=scaling,
        causal=True,
    ).to(device)
    assert not torch.isnan(got).any()
    torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)


@_CUDA
def test_hy4_build_engram_inputs_from_history_cuda() -> None:
    """Verify single-kernel Engram history append + n-gram lookback + mask matches PyTorch."""
    from tokenspeed_kernel.ops.model.hy4_flash.engram import (
        hy4_build_engram_inputs_from_history,
    )

    torch.manual_seed(139)
    device = torch.device("cuda")
    pad_id, img_id, vid_id, max_ng = 120002, 120010, 120011, 3
    num_slots, cap = 8, 32
    history = torch.randint(0, 1000, (num_slots, cap), dtype=torch.int32, device=device)
    history_ref = history.clone()
    committed = torch.tensor(
        [0, 5, 1, 10, 2, 0, 0, 0], dtype=torch.int32, device=device
    )
    req_slots = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32, device=device)
    starts = torch.tensor([0, 6, 12, 18, 24, 30], dtype=torch.int32, device=device)
    active_reqs = torch.tensor(
        [True, True, False, True, True], dtype=torch.bool, device=device
    )
    input_ids = torch.randint(0, 1000, (30,), dtype=torch.int32, device=device)
    input_ids[2] = img_id
    input_ids[7] = vid_id
    input_ids[19] = pad_id

    got_ng, got_mask = hy4_build_engram_inputs_from_history(
        input_ids,
        history,
        committed,
        req_slots,
        starts,
        active_reqs,
        max_ngram_size=max_ng,
        pad_id=pad_id,
        image_token_id=img_id,
        video_token_id=vid_id,
    )

    # Exact reference matching build_hy4_engram_inputs
    bs = req_slots.shape[0]
    pad_slot = num_slots - 1
    token_idx = torch.arange(30, device=device, dtype=starts.dtype)
    req_row = torch.searchsorted(starts[1:], token_idx, right=True).clamp_max(bs - 1)
    local_j = token_idx - starts[req_row]
    slot = req_slots[req_row].to(torch.int64)
    active = active_reqs[req_row]
    hist_pos = committed[slot].to(torch.int64) + local_j.to(torch.int64)
    safe_slot = torch.where(active, slot, pad_slot)
    safe_wpos = torch.where(active, hist_pos.clamp(0, cap - 1), 0)
    history_ref[safe_slot, safe_wpos] = input_ids
    ids64 = input_ids.to(torch.int64)
    cols = [torch.where(active, ids64, pad_id)]
    for off in range(1, max_ng):
        rpos = (hist_pos - off).clamp(0, cap - 1)
        fetched = history_ref[safe_slot, rpos].to(torch.int64)
        cols.append(torch.where(active & (hist_pos >= off), fetched, pad_id))
    ref_ng = torch.stack(cols, dim=-1)
    ref_mask = (
        (input_ids != img_id) & (input_ids != vid_id) & (input_ids != pad_id) & active
    )
    assert torch.equal(got_ng, ref_ng)
    assert torch.equal(got_mask, ref_mask)
    assert torch.equal(history[:pad_slot], history_ref[:pad_slot])


@_CUDA
@pytest.mark.parametrize("bs,seq", [(4, 128), (4, 512), (16, 256), (32, 512)])
def test_mla_decode_multiq_shared_kv_swa_cuda(bs: int, seq: int) -> None:
    """Verify Multi-Q Shared-KV SWA MLA decode matches per-query reference across batch tiers."""
    from tokenspeed_kernel.ops.attention.mla._triton.decode import (
        _mla_decode_kernel,
        mla_decode_fwd,
    )

    torch.manual_seed(141 + bs + seq)
    device = torch.device("cuda")
    q_len, num_heads, d_lat, d_rope, page_size = 6, 16, 512, 64, 64
    sm_scale = 192**-0.5
    pages_per_req = (seq + page_size - 1) // page_size + 1
    total_pages = bs * pages_per_req
    kv = (
        torch.randn(
            total_pages,
            page_size,
            1,
            d_lat + d_rope,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.2
    )
    q = (
        torch.randn(
            bs, q_len, num_heads, d_lat + d_rope, dtype=torch.bfloat16, device=device
        )
        * 0.2
    )
    pt = torch.arange(total_pages, dtype=torch.int32, device=device).view(
        bs, pages_per_req
    )
    sl = torch.randint(
        max(q_len, seq // 2), seq + 1, (bs,), dtype=torch.int32, device=device
    )
    sinks = torch.randn(num_heads, dtype=torch.float32, device=device)
    out = torch.empty(bs, q_len, num_heads, d_lat, dtype=torch.bfloat16, device=device)
    lse = torch.empty(bs, q_len, num_heads, dtype=torch.float32, device=device)
    mla_decode_fwd(
        q,
        kv,
        out,
        pt,
        sl,
        seq,
        128,
        d_lat,
        d_rope,
        sm_scale,
        lse=lse,
        window_left=512,
        sinks=sinks,
    )

    # Scalar reference kernel without multi-q packing (then apply sinks).
    ref = torch.empty_like(out)
    ref_lse = torch.empty_like(lse)
    _mla_decode_kernel[(bs, q_len, num_heads)](
        q,
        kv,
        ref,
        ref_lse,
        pt,
        sl,
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv.stride(0),
        kv.stride(1),
        kv.stride(2),
        ref.stride(0),
        ref.stride(1),
        ref.stride(2),
        ref_lse.stride(0),
        ref_lse.stride(1),
        ref_lse.stride(2),
        pt.stride(0),
        PAGE_SIZE=page_size,
        MAX_SEQLEN_K=seq,
        logit_cap=0.0,
        KV_LORA_RANK=d_lat,
        QK_ROPE_HEAD_DIM=d_rope,
        BLOCK_R=d_lat,
        BLOCK_ROPE=d_rope,
        BLOCK_N=64,
        HAS_LSE=True,
        WINDOW_LEFT=512,
        NONCAUSAL_BLOCK_SIZE=1,
        Q_LEN=q_len,
    )
    ref = (
        ref.float() * torch.sigmoid(ref_lse - sinks[None, None, :]).unsqueeze(-1)
    ).to(out.dtype)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(lse, ref_lse, rtol=2e-2, atol=2e-2)


@_CUDA
@pytest.mark.parametrize("bs", [1, 4, 16, 32])
def test_hy4_dfly_shared_kv_dsa_cuda(bs: int) -> None:
    """Verify zero-copy Multi-Q Shared-KV DFly context DSA matches _run_dense_kv including ctx_len=0."""
    from tokenspeed_kernel.ops.attention.dsa.triton import _run_dense_kv
    from tokenspeed_kernel.ops.model.hy4_flash.dfly import hy4_dfly_shared_kv_dsa

    torch.manual_seed(149 + bs)
    device = torch.device("cuda")
    q_len, num_heads, d_lat, d_rope, page_size, topk = 6, 16, 512, 64, 64, 512
    sm_scale = 192**-0.5
    n_pages = bs * 10
    kv = (
        torch.randn(
            n_pages, page_size, 1, d_lat + d_rope, dtype=torch.bfloat16, device=device
        )
        * 0.2
    )
    q = (
        torch.randn(
            bs, q_len, num_heads, d_lat + d_rope, dtype=torch.bfloat16, device=device
        )
        * 0.2
    )
    ctx_slots = torch.randint(
        0, n_pages * page_size, (bs, topk), dtype=torch.int64, device=device
    )
    ctx_lens = torch.randint(32, topk + 1, (bs,), dtype=torch.int32, device=device)
    if bs > 1:
        ctx_lens[1] = 0
    got_out, got_lse = hy4_dfly_shared_kv_dsa(
        q,
        kv,
        ctx_slots,
        ctx_lens,
        kv_lora_rank=d_lat,
        qk_rope_head_dim=d_rope,
        softmax_scale=sm_scale,
    )
    slots_rep = (
        ctx_slots.unsqueeze(1)
        .expand(bs, q_len, topk)
        .reshape(bs * q_len, topk)
        .to(torch.int32)
        .contiguous()
    )
    lens_rep = (
        ctx_lens.unsqueeze(1)
        .expand(bs, q_len)
        .reshape(bs * q_len)
        .to(torch.int32)
        .contiguous()
    )
    ref_out, ref_lse = _run_dense_kv(
        q.reshape(bs * q_len, num_heads, d_lat + d_rope),
        kv,
        slots_rep,
        lens_rep,
        softmax_scale=sm_scale,
        kv_lora_rank=d_lat,
        qk_rope_head_dim=d_rope,
        return_lse=True,
        sinks=None,
    )
    torch.testing.assert_close(got_out, ref_out.view_as(got_out), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(got_lse, ref_lse.view_as(got_lse), rtol=2e-2, atol=2e-2)
    if bs > 1:
        assert torch.equal(got_out[1], torch.zeros_like(got_out[1]))
        assert torch.isneginf(got_lse[1]).all()


@_CUDA
@pytest.mark.parametrize("num_heads", [16, 32])
def test_hy4_apply_interleaved_rope_out_of_place_cuda(num_heads: int) -> None:
    """Verify out-of-place RoPE writes rotated q_pe directly into absorbed query tail."""
    torch.manual_seed(201 + num_heads)
    device = torch.device("cuda")
    num_tokens, nope_dim, rope_dim, kv_lora = 6, 192, 64, 512
    q = torch.randn(
        num_tokens, num_heads, nope_dim + rope_dim, dtype=torch.bfloat16, device=device
    )
    q_orig = q.clone()
    cos = torch.randn(num_tokens, rope_dim // 2, dtype=torch.bfloat16, device=device)
    sin = torch.randn(num_tokens, rope_dim // 2, dtype=torch.bfloat16, device=device)

    absorbed = torch.zeros(
        num_tokens, num_heads, kv_lora + rope_dim, dtype=torch.bfloat16, device=device
    )
    q_pe = q[..., nope_dim:]
    hy4_apply_interleaved_rope_inplace(q_pe, cos, sin, out=absorbed[..., kv_lora:])

    # Input q must not be mutated when out is provided.
    assert torch.equal(q, q_orig)
    # Output slice must match in-place reference identically.
    q_ref = q_orig.clone()
    hy4_apply_interleaved_rope_inplace(q_ref, cos, sin)
    assert torch.equal(absorbed[..., kv_lora:], q_ref[..., nope_dim:])
    assert torch.equal(
        absorbed[..., :kv_lora], torch.zeros_like(absorbed[..., :kv_lora])
    )


@_CUDA
@pytest.mark.parametrize("layout", ["paged4d", "flat3d"])
def test_hy4_split_qkv_indexer_norm_rope_direct_kv_scatter_cuda(layout: str) -> None:
    """Verify direct KV scatter in hy4_split_qkv_indexer_norm_rope matches latent output."""
    torch.manual_seed(211 if layout == "paged4d" else 212)
    device = torch.device("cuda")
    num_tokens, q_lora, kv_lora, rope_dim, tk_dim, hw_dim = 6, 1024, 512, 64, 128, 16
    kv_total = kv_lora + rope_dim
    packed = torch.randn(
        num_tokens,
        q_lora + kv_total + tk_dim + hw_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    q_w = torch.randn(q_lora, dtype=torch.bfloat16, device=device)
    kv_w = torch.randn(kv_lora, dtype=torch.bfloat16, device=device)
    tk_w = torch.randn(tk_dim, dtype=torch.bfloat16, device=device)
    cos = torch.randn(num_tokens, rope_dim // 2, dtype=torch.bfloat16, device=device)
    sin = torch.randn(num_tokens, rope_dim // 2, dtype=torch.bfloat16, device=device)

    q_ref, lat_ref, tk_ref, hw_ref, qfp8_ref, qsc_ref = hy4_split_qkv_indexer_norm_rope(
        packed,
        q_w,
        kv_w,
        cos,
        sin,
        q_eps=1e-5,
        kv_eps=1e-5,
        tk_norm_weight=tk_w,
        tk_eps=1e-5,
        hw_dim=hw_dim,
        weights_scale=0.125,
        quantize_q_fp8_mn_major=True,
        write_q_bf16=True,
    )
    assert lat_ref is not None

    if layout == "paged4d":
        kv_cache = torch.zeros(4, 64, 1, kv_total, dtype=torch.bfloat16, device=device)
        slots = torch.tensor(
            [3, 65, -1, 130, 191, 255], dtype=torch.int32, device=device
        )
    else:
        kv_cache = torch.zeros(256, 1, kv_total, dtype=torch.bfloat16, device=device)
        slots = torch.tensor(
            [3, 65, -1, 130, 191, 255], dtype=torch.int32, device=device
        )

    q_got, lat_got, tk_got, hw_got, qfp8_got, qsc_got = hy4_split_qkv_indexer_norm_rope(
        packed,
        q_w,
        kv_w,
        cos,
        sin,
        q_eps=1e-5,
        kv_eps=1e-5,
        tk_norm_weight=tk_w,
        tk_eps=1e-5,
        hw_dim=hw_dim,
        weights_scale=0.125,
        quantize_q_fp8_mn_major=True,
        write_q_bf16=True,
        kv_cache_out=kv_cache,
        kv_slots=slots,
    )
    assert lat_got is None
    assert torch.equal(q_got, q_ref)
    assert torch.equal(tk_got, tk_ref)
    assert torch.equal(hw_got, hw_ref)
    assert torch.equal(qfp8_got, qfp8_ref)
    assert torch.equal(qsc_got, qsc_ref)
    for tok, slot in enumerate(slots.tolist()):
        if slot < 0:
            continue
        if layout == "paged4d":
            got_row = kv_cache[slot // 64, slot % 64, 0]
        else:
            got_row = kv_cache[slot, 0]
        assert torch.equal(got_row, lat_ref[tok])


@_CUDA
def test_hy4_add_rmsnorm_quantize_fp8_pre_norm_cuda() -> None:
    """Verify fused pre_norm_weight in hy4_add_rmsnorm_quantize_fp8 matches 2-step RMSNorm."""
    from tokenspeed_kernel.ops.model.hy4_flash.fusion import (
        hy4_add_rmsnorm_quantize_fp8,
    )

    torch.manual_seed(223)
    device = torch.device("cuda")
    x = torch.randn(6, 3072, dtype=torch.bfloat16, device=device)
    res = torch.randn(6, 3072, dtype=torch.bfloat16, device=device)
    pre_w = torch.randn(3072, dtype=torch.bfloat16, device=device)
    post_w = torch.randn(3072, dtype=torch.bfloat16, device=device)

    xf = x.float()
    x_pre = (
        xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + 1e-5) * pre_w.float()
    ).to(torch.bfloat16)
    res_ref, _, q_ref, sc_ref = hy4_add_rmsnorm_quantize_fp8(
        x_pre, post_w, residual=res, eps=1e-5, mn_major_tma=True, write_bf16=False
    )
    res_got, _, q_got, sc_got = hy4_add_rmsnorm_quantize_fp8(
        x,
        post_w,
        residual=res,
        eps=1e-5,
        pre_norm_weight=pre_w,
        pre_eps=1e-5,
        mn_major_tma=True,
        write_bf16=False,
    )
    torch.testing.assert_close(res_got, res_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(sc_got, sc_ref, rtol=1e-2, atol=1e-3)
    torch.testing.assert_close(
        q_got.float() * sc_got.repeat_interleave(128, dim=-1),
        q_ref.float() * sc_ref.repeat_interleave(128, dim=-1),
        rtol=2e-2,
        atol=2e-2,
    )


@_CUDA
def test_hy4_split_routed_and_shared_swiglu_strided_out_cuda() -> None:
    """Verify strided shared_out writes directly into the right half of cat_buf."""
    from tokenspeed_kernel.ops.model.hy4_flash.fusion import (
        hy4_split_routed_and_shared_swiglu,
    )

    torch.manual_seed(241)
    device = torch.device("cuda")
    num_tokens, routed_dim, shared_inter = 6, 1536, 1536
    packed = torch.randn(
        num_tokens, routed_dim + 2 * shared_inter, dtype=torch.bfloat16, device=device
    )
    routed_ref, shared_ref = hy4_split_routed_and_shared_swiglu(packed, routed_dim)

    cat_buf = torch.full(
        (num_tokens, routed_dim + shared_inter),
        -7.5,
        dtype=torch.bfloat16,
        device=device,
    )
    routed_got, shared_got = hy4_split_routed_and_shared_swiglu(
        packed, routed_dim, shared_out=cat_buf[:, routed_dim:]
    )
    assert shared_got.data_ptr() == cat_buf[:, routed_dim:].data_ptr()
    assert torch.equal(routed_got, routed_ref)
    assert torch.equal(shared_got, shared_ref)
    assert torch.equal(cat_buf[:, routed_dim:], shared_ref)
    assert torch.all(cat_buf[:, :routed_dim] == -7.5)


@_CUDA
@pytest.mark.parametrize("m", [1, 6, 16])
def test_hy4_sigmoid_bias_topk_wait_after_pdl_chain_cuda(m: int) -> None:
    """Verify PDL-reordered MoE entry (router -> packed_down -> topk(wait_after) -> split)."""
    from tokenspeed_kernel.ops.gemm.triton_skinny import skinny_mm

    torch.manual_seed(317 + m)
    device = torch.device("cuda")
    h = torch.randn(m, 3072, dtype=torch.bfloat16, device=device) * 0.2
    w_router = torch.randn(512, 3072, dtype=torch.bfloat16, device=device) * 0.2
    bias = torch.randn(512, dtype=torch.float32, device=device) * 0.05
    w_down = torch.randn(4608, 3072, dtype=torch.bfloat16, device=device) * 0.2

    # Reference order: router -> topk -> packed_down -> split.
    logits_ref = skinny_mm(h, w_router, None, out_dtype=torch.float32, compact_m=True)
    tw_ref, ti_ref = hy4_sigmoid_bias_topk(
        logits_ref, bias, top_k=16, routed_scaling_factor=3.998, norm_topk_prob=True
    )
    pk_ref = skinny_mm(h, w_down, None, out_dtype=torch.bfloat16, compact_m=True)
    r_ref, s_ref = hy4_split_routed_and_shared_swiglu(pk_ref, 1536)

    # PDL-reordered sequence: router -> packed_down -> topk(wait_after_compute=True) -> split.
    logits_got = skinny_mm(h, w_router, None, out_dtype=torch.float32, compact_m=True)
    pk_got = skinny_mm(h, w_down, None, out_dtype=torch.bfloat16, compact_m=True)
    tw_got, ti_got = hy4_sigmoid_bias_topk(
        logits_got,
        bias,
        top_k=16,
        routed_scaling_factor=3.998,
        norm_topk_prob=True,
        wait_after_compute=True,
    )
    r_got, s_got = hy4_split_routed_and_shared_swiglu(pk_got, 1536)

    assert torch.equal(ti_got, ti_ref)
    assert torch.equal(tw_got, tw_ref)
    assert torch.equal(r_got, r_ref)
    assert torch.equal(s_got, s_ref)


@_CUDA
@pytest.mark.parametrize("num_groups", [1, 2, 32, 64, 70])
def test_hy4_kpool_splitn_and_fallback_cuda(num_groups: int) -> None:
    """Verify Split-N KPool (G <= 64) and fallback (G > 64) match reference FP8 scatter."""
    torch.manual_seed(331 + num_groups)
    device = torch.device("cuda")
    kpool, head_dim = 4, 128
    rows_per_page = 64
    num_pages = 4
    gk = (
        torch.randn(num_groups, kpool, head_dim, dtype=torch.bfloat16, device=device)
        * 0.3
    )
    wt = torch.randn(head_dim, 2 * head_dim, dtype=torch.float32, device=device) * 0.1
    ape = torch.randn(kpool, head_dim, dtype=torch.bfloat16, device=device) * 0.1
    knw = torch.randn(head_dim, dtype=torch.bfloat16, device=device) * 0.5 + 1.0
    gpos = torch.randint(0, 128, (3, num_groups), dtype=torch.int64, device=device)
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, 64, 2, device=device).float() / 64.0))
    mrope_section = (12, 10, 10)
    # Mix valid slots and -1 early-exit slots.
    slots = torch.tensor(
        [i if (i % 3 != 1 or i == 0) else -1 for i in range(num_groups)],
        dtype=torch.int64,
        device=device,
    )

    iv_ref = torch.zeros(
        (num_pages, rows_per_page, head_dim), dtype=torch.float8_e4m3fn, device=device
    )
    is_ref = torch.zeros(
        (num_pages, rows_per_page, 1), dtype=torch.float32, device=device
    )
    vg_ref = torch.matmul(gk.reshape(-1, head_dim).float(), wt).view(
        num_groups, kpool, 2 * head_dim
    )
    hy4_kpool_finalize_rope_store(
        vg_ref,
        ape,
        knw,
        gpos,
        inv_freq,
        mrope_section,
        slots,
        iv_ref,
        is_ref,
        orig_dtype=torch.bfloat16,
    )

    iv_got = torch.zeros_like(iv_ref)
    is_got = torch.zeros_like(is_ref)
    hy4_kpool_project_finalize_rope_store(
        gk, wt, ape, knw, gpos, inv_freq, mrope_section, slots, iv_got, is_got
    )
    torch.testing.assert_close(
        iv_got.float() * is_got,
        iv_ref.float() * is_ref,
        rtol=5e-2,
        atol=5e-2,
    )


@_CUDA
@pytest.mark.parametrize(
    ("bsz", "q_len"),
    [(1, 1), (4, 1), (1, 6), (16, 1), (4, 6)],
)
def test_mla_decode_swa_bh16_all_buckets_cuda(bsz: int, q_len: int) -> None:
    """Verify BLOCK_H=16 SWA MLA decode buckets (splits 16/8/4/2/1) with sinks."""
    from tokenspeed_kernel.ops.attention.mla._triton.decode import mla_decode_fwd

    torch.manual_seed(401 + bsz * 17 + q_len)
    device = torch.device("cuda")
    num_heads = 32
    kv_lora_rank = 512
    qk_rope_dim = 64
    page_size = 64
    window_left = 511
    softmax_scale = (192.0) ** -0.5

    max_seqlen = 640
    pages_per_req = (max_seqlen + page_size - 1) // page_size
    total_pages = bsz * pages_per_req
    q = (
        torch.randn(
            bsz,
            q_len,
            num_heads,
            kv_lora_rank + qk_rope_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.25
    )
    kv_cache = (
        torch.randn(
            total_pages,
            page_size,
            1,
            kv_lora_rank + qk_rope_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.25
    )
    page_table = torch.arange(total_pages, dtype=torch.int32, device=device).view(
        bsz, pages_per_req
    )
    # Mix short (< window) and long (> window) sequence lengths across requests.
    cache_seqlens = torch.tensor(
        [120 + (i * 137) % (max_seqlen - 120) for i in range(bsz)],
        dtype=torch.int32,
        device=device,
    )
    sinks = torch.randn(num_heads, dtype=torch.float32, device=device) * 0.3

    out = torch.empty(
        bsz, q_len, num_heads, kv_lora_rank, dtype=torch.bfloat16, device=device
    )
    lse = torch.empty(bsz, q_len, num_heads, dtype=torch.float32, device=device)
    mla_decode_fwd(
        q,
        kv_cache,
        out,
        page_table,
        cache_seqlens,
        max_seqlen,
        128,
        kv_lora_rank,
        qk_rope_dim,
        softmax_scale,
        lse=lse,
        window_left=window_left,
        sinks=sinks,
    )

    out_ref = torch.empty_like(out)
    lse_ref = torch.empty_like(lse)
    for b in range(bsz):
        seq_len = int(cache_seqlens[b].item())
        kv_flat = (
            kv_cache[page_table[b]]
            .reshape(-1, kv_lora_rank + qk_rope_dim)[:seq_len]
            .float()
        )
        v_flat = kv_flat[:, :kv_lora_rank]
        for qi in range(q_len):
            q_pos = seq_len - q_len + qi
            win_start = max(0, q_pos - window_left)
            k_win = kv_flat[win_start : q_pos + 1]
            v_win = v_flat[win_start : q_pos + 1]
            logits = (q[b, qi].float() @ k_win.T) * softmax_scale
            # Include attention sink logit in denominator without adding to V.
            sink_logits = sinks.unsqueeze(-1)
            all_logits = torch.cat([logits, sink_logits], dim=-1)
            lse_row = torch.logsumexp(all_logits, dim=-1)
            probs = torch.exp(logits - lse_row.unsqueeze(-1))
            out_ref[b, qi] = (probs @ v_win).to(torch.bfloat16)
            lse_ref[b, qi] = lse_row

    torch.testing.assert_close(out, out_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(lse, lse_ref, rtol=1e-2, atol=1e-2)
