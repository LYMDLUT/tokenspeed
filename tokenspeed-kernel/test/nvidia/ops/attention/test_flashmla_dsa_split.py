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

"""FlashMLA sparse prefill split over the selected columns of few-row launches.

A launch with fewer rows than SMs attends every row's column slices as rows of
their own and merges the partials by LSE. It must agree with one launch over
the whole rows, sink included, on HY V4's TP32 shape: 2 real heads padded to
the kernel's 64.
"""

import pytest
import torch
from tokenspeed_kernel.platform import current_platform

platform = current_platform()

pytestmark = pytest.mark.skipif(
    not (platform.is_nvidia and platform.is_hopper_plus),
    reason="The flashmla DSA solution registers on Hopper+ NVIDIA GPUs only",
)

if platform.is_nvidia and platform.is_hopper_plus:
    from tokenspeed_kernel.ops.attention.dsa import cuda as dsa_cuda
    from tokenspeed_kernel.ops.attention.dsa._triton.split_partials import (
        merge_split_partials,
    )
    from tokenspeed_kernel.ops.attention.mla.cuda import flash_mla_sparse_fwd

HEADS = 2
PADDED_HEADS = 64
ROW = 576
KV_LORA_RANK = 512
TOPK = 2048
SLOTS = 8192
SCALE = ROW**-0.5

CASES = {
    "decode": dict(lens=[TOPK]),
    "mtp3": dict(lens=[TOPK - 3, TOPK - 2, TOPK - 1, TOPK]),
    "short": dict(lens=[1, 63, 64, 65, 700, 1500]),
    "empty_rows": dict(lens=[0, 5, 0, TOPK]),
    "invalid_slots": dict(lens=[TOPK, 1200, 300, 64], invalid=0.4),
}


def _inputs(lens, invalid):
    generator = torch.Generator(device="cuda").manual_seed(len(lens))
    rows = len(lens)
    kv = (torch.randn(SLOTS, 1, ROW, device="cuda", generator=generator) * 0.5).to(
        torch.bfloat16
    )
    q = torch.zeros(rows, PADDED_HEADS, ROW, device="cuda", dtype=torch.bfloat16)
    q[:, :HEADS] = torch.randn(rows, HEADS, ROW, device="cuda", generator=generator).to(
        torch.bfloat16
    )
    slots = torch.stack(
        [
            torch.randperm(SLOTS, device="cuda", generator=generator)[:TOPK]
            for _ in range(rows)
        ]
    ).to(torch.int32)
    if invalid:
        slots[torch.rand(slots.shape, device="cuda", generator=generator) < invalid] = (
            -1
        )
    lens = torch.tensor(lens, device="cuda", dtype=torch.int32)
    sink = torch.randn(HEADS, device="cuda", generator=generator)
    return q, kv, dsa_cuda._mask_sparse_slots(slots, lens), lens, sink


def _one_launch(q, kv, slots, lens, sink):
    padded_sink = (
        None
        if sink is None
        else dsa_cuda._pad_sink_for_heads(sink, PADDED_HEADS, HEADS)
    )
    result, _, lse = flash_mla_sparse_fwd(
        q=q,
        kv=kv,
        indices=slots.unsqueeze(1),
        sm_scale=SCALE,
        d_v=KV_LORA_RANK,
        attn_sink=padded_sink,
        topk_length=lens,
    )
    result, lse = result[:, :HEADS], lse[:, :HEADS]
    if sink is not None:
        lse = torch.logaddexp(lse, sink)
    return dsa_cuda._finish_sparse_attention(result, lse, slots, None, True)


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("splits", [2, 4, 16])
@pytest.mark.parametrize("with_sink", [True, False], ids=["sink", "no_sink"])
def test_split_matches_one_launch(case, splits, with_sink):
    spec = CASES[case]
    q, kv, slots, lens, sink = _inputs(spec["lens"], spec.get("invalid", 0.0))
    sink = sink if with_sink else None
    ref_out, ref_lse = _one_launch(q, kv, slots, lens, sink)
    out, lse = dsa_cuda._flashmla_sparse_prefill_split(
        q=q,
        kv=kv,
        slots=slots,
        topk_lens=lens,
        sm_scale=SCALE,
        d_v=KV_LORA_RANK,
        attn_sink=sink,
        actual_heads=HEADS,
        splits=splits,
        out=None,
        return_lse=True,
    )
    assert out.shape == ref_out.shape and out.dtype == torch.bfloat16
    # The partials are BF16 before the FP32 merge: one more rounding than the
    # single launch, which rounds only its final output.
    torch.testing.assert_close(out.float(), ref_out.float(), rtol=0.0, atol=4e-3)
    finite = torch.isfinite(ref_lse)
    torch.testing.assert_close(lse[finite], ref_lse[finite], rtol=0.0, atol=1e-4)
    assert torch.isneginf(lse[~finite]).all()
    assert (out[~finite.all(dim=1)] == 0).all()


def test_split_writes_the_supplied_output_buffer():
    q, kv, slots, lens, sink = _inputs([TOPK, 900], 0.0)
    out = torch.full((2, HEADS * KV_LORA_RANK), 7.0, device="cuda", dtype=q.dtype)
    result = dsa_cuda._flashmla_sparse_prefill_split(
        q=q,
        kv=kv,
        slots=slots,
        topk_lens=lens,
        sm_scale=SCALE,
        d_v=KV_LORA_RANK,
        attn_sink=sink,
        actual_heads=HEADS,
        splits=8,
        out=out,
        return_lse=False,
    )
    ref_out, _ = _one_launch(q, kv, slots, lens, sink)
    assert result is out
    torch.testing.assert_close(
        out.view_as(ref_out).float(), ref_out.float(), rtol=0.0, atol=4e-3
    )


@pytest.mark.parametrize("rows", [1, 2, 4, 8, 31, 39, 40, 64, 78, 200])
@pytest.mark.parametrize("width", [512, 1024, 2048])
def test_splits_fill_one_wave(rows, width):
    sms = dsa_cuda._sm_count(torch.device("cuda"))
    splits = dsa_cuda._flashmla_sparse_column_splits(rows, width, torch.device("cuda"))
    assert splits >= 1 and width % (splits * dsa_cuda._FLASHMLA_SPARSE_TOPK_ALIGN) == 0
    if splits > 1:
        assert rows * splits <= sms
    # Doubling again would overflow the wave or the kernel's column alignment.
    assert (
        rows * splits * 2 > sms
        or width % (splits * 2 * dsa_cuda._FLASHMLA_SPARSE_TOPK_ALIGN) != 0
    )


def test_merge_treats_reported_empty_partials_as_empty():
    # FlashMLA reports a row that attended to nothing with +inf LSE; such a
    # split must not take part even when its column count is nonzero.
    partial = torch.randn(4, PADDED_HEADS, KV_LORA_RANK, device="cuda").to(
        torch.bfloat16
    )
    lse = torch.tensor([[1.0], [float("inf")], [0.5], [float("inf")]], device="cuda")
    lse = lse.expand(4, PADDED_HEADS).contiguous()
    split_lens = torch.tensor([64, 64, 64, 64], device="cuda", dtype=torch.int32)
    out, merged = merge_split_partials(
        partial,
        lse,
        split_lens,
        splits=2,
        heads=HEADS,
        sink=None,
        out=None,
        return_lse=True,
    )
    torch.testing.assert_close(out[0], partial[0, :HEADS])
    torch.testing.assert_close(out[1], partial[2, :HEADS])
    torch.testing.assert_close(
        merged, torch.tensor([[1.0, 1.0], [0.5, 0.5]], device="cuda")
    )
