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

import pytest
import torch
from tokenspeed_kernel.ops.attention.dsa import dsa_decode, dsa_prefill
from tokenspeed_kernel.ops.kvcache.flashmla import set_flashmla_kv_buffer
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform().is_hopper, reason="Hopper FlashMLA FP8 KV path"
)


def test_packed_attention_rejects_a_mismatched_page_size():
    with pytest.raises(ValueError, match="sparse KV page size mismatch"):
        dsa_decode(
            q=torch.zeros((1, 8, 576), dtype=torch.bfloat16, device="cuda"),
            kv_cache=None,
            sparse_kv_cache=torch.zeros(
                (1, 128, 1, 656), dtype=torch.uint8, device="cuda"
            ),
            topk_slots=torch.full((1, 512), -1, dtype=torch.int32, device="cuda"),
            topk_lens=torch.zeros(1, dtype=torch.int32, device="cuda"),
            max_seqlen_k=128,
            qk_nope_head_dim=128,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            softmax_scale=576**-0.5,
            page_size=64,
            solution="flashmla",
            slot_order="selection",
        )


@pytest.mark.parametrize("rope_dim", [0, 64])
@pytest.mark.parametrize("topk", [512, 2051])
@pytest.mark.parametrize("phase,q_len", [("prefill", 1), ("decode", 1), ("decode", 3)])
def test_packed_attention_matches_dequantized_reference(rope_dim, topk, phase, q_len):
    torch.manual_seed(123)
    keys = torch.randn((128, 1, 512 + rope_dim), device="cuda", dtype=torch.bfloat16)
    query = torch.randn((3, 8, 512 + rope_dim), device="cuda", dtype=torch.bfloat16)
    cache = torch.zeros((128, 1, 656), device="cuda", dtype=torch.uint8)
    locations = torch.arange(128, dtype=torch.int64, device="cuda")
    set_flashmla_kv_buffer(
        cache,
        locations,
        keys[..., :512],
        keys[..., 512:],
        sanitize=False,
        write_mask=None,
    )
    # Decode the public row format independently of the gather kernel.
    values = cache[:, 0, :512].view(torch.float8_e4m3fn).float().reshape(128, 4, 128)
    scales = cache[:, 0, 512:528].view(torch.float32)
    latent = (values * scales[..., None]).reshape(128, 512)
    key = torch.cat((latent, keys[:, 0, 512:].float()), dim=-1)
    slots = torch.full((3, topk), -1, dtype=torch.int32, device="cuda")
    slots[0, :4] = torch.tensor([0, 63, 64, 127], device="cuda")
    slots[1, :2] = torch.tensor([15, 100], device="cuda")
    lengths = torch.tensor([4, 1, 0], dtype=torch.int32, device="cuda")
    scale = (512 + rope_dim) ** -0.5
    args = dict(
        q=query,
        kv_cache=None,
        sparse_kv_cache=cache,
        topk_slots=slots,
        topk_lens=lengths,
        max_seqlen_k=128,
        qk_nope_head_dim=256 if rope_dim == 0 else 128,
        kv_lora_rank=512,
        qk_rope_head_dim=rope_dim,
        softmax_scale=scale,
        page_size=64,
        return_lse=True,
        solution="flashmla",
        slot_order="selection",
    )
    attention = dsa_prefill if phase == "prefill" else dsa_decode
    if phase == "decode":
        args["q_len_per_req"] = q_len
    actual, lse = attention(**args)
    valid = (slots >= 0) & (torch.arange(topk, device="cuda") < lengths[:, None])
    gathered = slots.clamp_min(0).long()
    logits = torch.einsum("thd,tkd->thk", query.float(), key[gathered]) * scale
    logits.masked_fill_(~valid[:, None], -float("inf"))
    reference_lse = torch.logsumexp(logits, -1)
    reference = torch.einsum(
        "thk,tkd->thd", logits.softmax(-1).nan_to_num(), latent[gathered]
    )
    torch.testing.assert_close(actual.float(), reference, atol=0.015, rtol=0.015)
    torch.testing.assert_close(lse, reference_lse, atol=2e-3, rtol=2e-3)
    assert not actual[2].any() and torch.isneginf(lse[2]).all()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replay_out, replay_lse = attention(**args)
    # Both the selection and packed values must be read again on replay.
    slots.fill_(9)
    lengths.fill_(1)
    query.mul_(0.5)
    keys.fill_(2)
    set_flashmla_kv_buffer(
        cache,
        locations,
        keys[..., :512],
        keys[..., 512:],
        sanitize=False,
        write_mask=None,
    )
    graph.replay()
    torch.testing.assert_close(
        replay_out.float(), torch.full_like(replay_out.float(), 2), atol=0.01, rtol=0
    )
    assert torch.isfinite(replay_lse).all()
    lengths.zero_()
    graph.replay()
    assert not replay_out.any() and torch.isneginf(replay_lse).all()
