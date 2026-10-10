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

"""DSA over a dense BF16 KV cache with a per-head learnable sink.

Shapes follow HY V4 (64 heads, 576-wide latent+rope rows, top-2048). The sink is
one extra softmax candidate without a value vector, so the reference appends it
as a logit column and drops its probability from the output.
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.attention.dsa import dsa_decode, dsa_prefill

HEADS = 64
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_NOPE_HEAD_DIM = 192
ROW = KV_LORA_RANK + QK_ROPE_HEAD_DIM
TOPK = 2048
SLOTS = 8192
SCALE = ROW**-0.5


def _reference(q, kv, slots, lens, sink):
    out = torch.zeros(q.shape[0], HEADS, KV_LORA_RANK, device=q.device)
    lse = torch.full((q.shape[0], HEADS), float("-inf"), device=q.device)
    for row in range(q.shape[0]):
        selected = slots[row, : int(lens[row])]
        selected = selected[selected >= 0].long()
        if selected.numel() == 0:
            continue
        keys = kv[selected].float()
        logits = q[row].float() @ keys.T * SCALE
        if sink is not None:
            logits = torch.cat([logits, sink[:, None]], dim=1)
        lse[row] = torch.logsumexp(logits, dim=1)
        probs = torch.softmax(logits, dim=1)[:, : selected.numel()]
        out[row] = probs @ keys[:, :KV_LORA_RANK]
    return out, lse


CASES = {
    "short": dict(lens=[1, 2, 5, 13, 20, 33, 64, 7]),
    "empty_rows": dict(lens=[0, 3, 0, 100]),
    "invalid_slots": dict(lens=[2048, 1500, 900, 64, 20, 2048, 300, 1], invalid=0.3),
    "full": dict(lens=[TOPK] * 8),
    "mtp": dict(lens=[10, 11, 500, 501, 2047, 2048, 1, 2], q_len=2),
}


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("with_sink", [True, False], ids=["sink", "no_sink"])
@pytest.mark.parametrize("mode", ["decode", "prefill"])
@pytest.mark.parametrize("solution", ["triton", "flashmla"])
def test_dsa_dense_kv_sink(device, require, solution, mode, with_sink, case):
    spec = CASES[case]
    q_len = spec.get("q_len", 1)
    if mode == "prefill" and q_len != 1:
        pytest.skip("prefill rows are single-token")
    if solution == "flashmla" and mode == "decode" and not with_sink:
        pytest.skip("FlashMLA dense-KV decode is selected only with a sink")
    api_name = f"dsa_{mode}"
    require("attention", api_name, solution, torch.bfloat16, "q")

    generator = torch.Generator(device=device).manual_seed(0)
    kv = (torch.randn(SLOTS, ROW, device=device, generator=generator) * 0.5).to(
        torch.bfloat16
    )
    sink = (
        torch.randn(HEADS, device=device, dtype=torch.float32, generator=generator)
        if with_sink
        else None
    )
    rows = len(spec["lens"])
    q = torch.randn(rows, HEADS, ROW, device=device, generator=generator).to(
        torch.bfloat16
    )
    slots = torch.stack(
        [
            torch.randperm(SLOTS, device=device, generator=generator)[:TOPK]
            for _ in range(rows)
        ]
    ).to(torch.int32)
    if spec.get("invalid"):
        mask = torch.rand(slots.shape, device=device, generator=generator)
        slots[mask < spec["invalid"]] = -1
    lens = torch.tensor(spec["lens"], device=device, dtype=torch.int32)

    kwargs = dict(
        q=q,
        kv_cache=kv,
        sparse_kv_cache=None,
        topk_slots=slots,
        topk_lens=lens,
        max_seqlen_k=SLOTS,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        softmax_scale=SCALE,
        page_size=64,
        return_lse=True,
        solution=solution,
        attn_sink=sink,
    )
    if mode == "decode":
        out, lse = dsa_decode(q_len_per_req=q_len, **kwargs)
    else:
        out, lse = dsa_prefill(**kwargs)

    ref_out, ref_lse = _reference(q, kv, slots, lens, sink)
    finite = torch.isfinite(ref_lse)
    assert out.shape == (rows, HEADS, KV_LORA_RANK)
    torch.testing.assert_close(out.float(), ref_out, rtol=0.0, atol=2e-2)
    torch.testing.assert_close(lse[finite], ref_lse[finite], rtol=0.0, atol=1e-2)
    # Rows with nothing selected must stay empty even when the sink is set.
    assert torch.isneginf(lse[~finite]).all()
    assert (out[~finite.all(dim=1)] == 0).all()
