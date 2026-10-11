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

"""Numerical check: Marlin W8A16 MXFP8 MoE apply vs a bf16 dequant reference.

E4M3 times an E8M0 power of two is exact in bf16, so the Marlin output must
track bf16 GEMMs over the dequantized experts. Runs on SM90+.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from tokenspeed_kernel.ops.moe.marlin import mxfp8
from tokenspeed_kernel.ops.moe.marlin.mxfp8 import (
    dequantize_marlin_mxfp8,
    marlin_mxfp8_moe_weights,
    marlin_mxfp8_precomputed_moe_apply,
)
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.thirdparty.cuda.marlin_moe import moe_align_block_size

BLOCK = 32


def _requires_sm90():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not current_platform().is_nvidia:
        pytest.skip("NVIDIA required")
    if current_platform().arch_version < type(current_platform().arch_version)(9, 0):
        pytest.skip("SM90+ required")


def _make_weights(num_experts, hidden, ispp, generator, scale_range=(118, 124)):
    """Random E4M3 experts with raw E8M0 (uint8) scales per 1x32 block."""

    def fp8(*shape):
        values = torch.randn(*shape, generator=generator, device="cuda") * 2
        return values.clamp(-448, 448).to(torch.float8_e4m3fn)

    def e8m0(*shape):
        lo, hi = scale_range
        return torch.randint(
            lo, hi + 1, shape, generator=generator, device="cuda", dtype=torch.uint8
        )

    return {
        "w13_weight": fp8(num_experts, 2 * ispp, hidden),
        "w13_weight_scale_inv": e8m0(num_experts, 2 * ispp, hidden // BLOCK),
        "w2_weight": fp8(num_experts, hidden, ispp),
        "w2_weight_scale_inv": e8m0(num_experts, hidden, ispp // BLOCK),
    }


def _dequant(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    s = torch.exp2(scale.float() - 127).repeat_interleave(BLOCK, dim=-1)
    return (weight.float() * s).to(torch.bfloat16)


def _reference(x, raw, topk_ids, topk_weights, limit):
    """bf16 GEMMs over dequantized experts with the clamped SwiGLU epilogue."""
    out = torch.zeros_like(x, dtype=torch.float32)
    for expert in range(raw["w13_weight"].shape[0]):
        token_ids, slot_ids = (topk_ids == expert).nonzero(as_tuple=True)
        if not token_ids.numel():
            continue
        w13 = _dequant(raw["w13_weight"][expert], raw["w13_weight_scale_inv"][expert])
        w2 = _dequant(raw["w2_weight"][expert], raw["w2_weight_scale_inv"][expert])
        gate_up = (x[token_ids].float() @ w13.float().T).to(torch.bfloat16)
        gate, up = gate_up.float().chunk(2, dim=-1)
        if limit is not None:
            gate = gate.clamp(max=limit)
            up = up.clamp(-limit, limit)
        hidden = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
        down = (hidden.float() @ w2.float().T).to(torch.bfloat16)
        out.index_add_(
            0, token_ids, down.float() * topk_weights[token_ids, slot_ids][:, None]
        )
    return out


class _Experts(torch.nn.Module):
    def __init__(self, raw, ep_size, ep_rank, limit):
        super().__init__()
        for name, value in raw.items():
            setattr(self, name, torch.nn.Parameter(value, requires_grad=False))
        self.num_local_experts = raw["w13_weight"].shape[0]
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.tp_size = 1
        self.tp_rank = 0
        self.activation = "swiglu"
        self.swiglu_arg = SimpleNamespace(alpha=None, limit=limit)
        self.swiglu_beta = None
        self._moe_backend_state = None


def _routing(num_tokens, num_experts, top_k, generator):
    scores = torch.rand(num_tokens, num_experts, generator=generator, device="cuda")
    topk_weights, topk_ids = scores.topk(top_k, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(-1, keepdim=True)
    return topk_ids.to(torch.int32), topk_weights.float()


def _run(raw, x, topk_ids, topk_weights, ep_size, limit):
    num_experts = raw["w13_weight"].shape[0]
    num_local = num_experts // ep_size
    plan = {"activation": "swiglu", "deterministic_finalize": True}
    acc = torch.zeros_like(x, dtype=torch.float32)
    for ep_rank in range(ep_size):
        lo = ep_rank * num_local
        shard = {k: v[lo : lo + num_local].clone() for k, v in raw.items()}
        w = _Experts(shard, ep_size, ep_rank, limit)
        marlin_mxfp8_moe_weights(plan, w)
        acc += marlin_mxfp8_precomputed_moe_apply(
            plan, x, w, None, topk_weights=topk_weights, topk_ids=topk_ids
        ).float()
    return acc


def _assert_close(actual, expected):
    magnitude = expected.abs().max().item()
    torch.testing.assert_close(actual, expected, atol=2e-2 * magnitude, rtol=2e-2)


@pytest.mark.parametrize("num_tokens", [1, 7, 64, 300])
@pytest.mark.parametrize("ep_size", [1, 2])
def test_marlin_mxfp8_matches_dequant_reference(num_tokens: int, ep_size: int):
    _requires_sm90()
    generator = torch.Generator(device="cuda").manual_seed(7)
    num_experts, top_k, hidden, ispp = 8, 3, 512, 256
    raw = _make_weights(num_experts, hidden, ispp, generator)
    x = (torch.randn(num_tokens, hidden, generator=generator, device="cuda")).to(
        torch.bfloat16
    )
    topk_ids, topk_weights = _routing(num_tokens, num_experts, top_k, generator)

    expected = _reference(x, raw, topk_ids, topk_weights, limit=None)
    actual = _run(raw, x, topk_ids, topk_weights, ep_size, limit=None)
    _assert_close(actual, expected)


@pytest.mark.parametrize("ep_size", [1, 2])
def test_marlin_mxfp8_honors_swiglu_clamp(ep_size: int):
    """The clamp must change the answer by far more than the tolerance, so an
    epilogue that drops it fails instead of passing by accident."""
    _requires_sm90()
    generator = torch.Generator(device="cuda").manual_seed(13)
    num_experts, top_k, hidden, ispp = 8, 2, 256, 128
    limit = 3.0
    raw = _make_weights(num_experts, hidden, ispp, generator, scale_range=(124, 126))
    x = torch.randn(32, hidden, generator=generator, device="cuda").to(torch.bfloat16)
    topk_ids, topk_weights = _routing(32, num_experts, top_k, generator)

    expected = _reference(x, raw, topk_ids, topk_weights, limit)
    unclamped = _reference(x, raw, topk_ids, topk_weights, None)
    assert (expected - unclamped).abs().max() > 0.5 * expected.abs().max()
    actual = _run(raw, x, topk_ids, topk_weights, ep_size, limit)
    _assert_close(actual, expected)


@pytest.mark.parametrize("num_tokens", [5, 300])
def test_marlin_mxfp8_without_ep_skips_invalid_and_repeated_ids(num_tokens: int):
    """Without EP, ``-1`` slots contribute zero and a repeated id counts once
    per slot. 300 tokens (1200 routes) outgrow the small-batch alignment
    kernel; both batches stay below the dense threshold, so the Marlin GEMMs
    see the -1 blocks."""
    _requires_sm90()
    generator = torch.Generator(device="cuda").manual_seed(37)
    num_experts, top_k, hidden, ispp = 8, 4, 512, 256
    assert num_tokens * top_k / num_experts < mxfp8._DENSE_MIN_ROUTES_PER_EXPERT
    raw = _make_weights(num_experts, hidden, ispp, generator)
    x = torch.randn(num_tokens, hidden, generator=generator, device="cuda").to(
        torch.bfloat16
    )
    topk_ids, topk_weights = _routing(num_tokens, num_experts, top_k, generator)
    topk_ids[:, 1] = topk_ids[:, 0]
    topk_ids[::2, 2] = -1
    topk_ids[1] = -1

    expected = _reference(x, raw, topk_ids, topk_weights, limit=None)
    actual = _run(raw, x, topk_ids, topk_weights, ep_size=1, limit=None)
    _assert_close(actual, expected)
    assert torch.count_nonzero(actual[1]) == 0


@pytest.mark.parametrize("num_experts,ep_size", [(16, 1), (300, 1), (64, 8), (1100, 1)])
def test_deterministic_align_is_reproducible(num_experts: int, ep_size: int):
    """The deterministic alignment is bitwise stable across calls (EP-masked ids
    included), matches the atomic scatter up to the order inside a segment,
    and past the small-batch kernel (>= 1024 routes or > 64 expert lanes)
    lists every expert's routes in route order. Global ids aligned against
    the rank's expert_start lay out exactly like pre-remapped local ids."""
    _requires_sm90()
    generator = torch.Generator(device="cuda").manual_seed(5)
    num_local = num_experts // ep_size
    expert_start = num_local if ep_size > 1 else 0
    for num_tokens in (100, 300, 4096):
        topk_ids, _ = _routing(num_tokens, num_experts, 8, generator)
        topk_ids[0, 0] = -1
        local = topk_ids - expert_start
        local = torch.where((local >= 0) & (local < num_local), local, -1)
        local = local.to(torch.int32)

        def align(ids, start, deterministic):
            return moe_align_block_size(
                ids, 16, num_local, expert_start=start, deterministic=deterministic
            )

        stable = align(local, 0, True)
        total = int(stable[2].item())
        for ids, start in ((local, 0), (local, 0), (topk_ids, expert_start)):
            again = align(ids, start, True)
            assert int(again[2].item()) == total
            assert torch.equal(again[0][:total], stable[0][:total])
            assert torch.equal(again[1][: total // 16], stable[1][: total // 16])
        fast = align(topk_ids, expert_start, False)
        assert total == int(fast[2].item())
        assert torch.equal(stable[1][: total // 16], fast[1][: total // 16])
        sorted_ids = stable[0][:total]
        numel = local.numel()
        if numel >= 1024 or num_local + 1 > 64:
            # Each block holds one expert's routes followed by padding.
            blocks = sorted_ids.view(-1, 16)
            valid = blocks < numel
            assert torch.all(valid[:, :-1] | ~valid[:, 1:])
            assert torch.all((blocks[:, 1:] > blocks[:, :-1]) | ~valid[:, 1:])
        assert torch.equal(
            sorted_ids[sorted_ids < numel].sort().values,
            torch.arange(numel, device="cuda", dtype=torch.int32),
        )
        assert torch.equal(
            stable[0][:total].sort().values, fast[0][:total].sort().values
        )


def test_marlin_mxfp8_is_bitwise_reproducible():
    """deterministic_finalize: repeated applies on one batch are bitwise equal,
    including batches whose routes exceed the small-batch alignment kernel and
    the prefill-sized one that takes the dense path."""
    _requires_sm90()
    generator = torch.Generator(device="cuda").manual_seed(17)
    num_experts, top_k, hidden, ispp = 16, 4, 1024, 512
    raw = _make_weights(num_experts, hidden, ispp, generator)
    w = _Experts({k: v.clone() for k, v in raw.items()}, 1, 0, 7.0)
    plan = {"activation": "swiglu", "deterministic_finalize": True}
    marlin_mxfp8_moe_weights(plan, w)
    for num_tokens in (5, 256, 512, 2048):
        x = torch.randn(num_tokens, hidden, generator=generator, device="cuda").to(
            torch.bfloat16
        )
        topk_ids, topk_weights = _routing(num_tokens, num_experts, top_k, generator)
        first = marlin_mxfp8_precomputed_moe_apply(
            plan, x, w, None, topk_weights=topk_weights, topk_ids=topk_ids
        )
        for _ in range(3):
            again = marlin_mxfp8_precomputed_moe_apply(
                plan, x, w, None, topk_weights=topk_weights, topk_ids=topk_ids
            )
            assert torch.equal(first, again)


@pytest.mark.parametrize("num_tokens", [4, 1024])
def test_marlin_mxfp8_production_geometry(num_tokens: int):
    """hidden 6144 / ispp 2048 / top-8 over 256 experts, one EP32 rank's eight
    local experts; routes to other ranks must contribute nothing."""
    _requires_sm90()
    generator = torch.Generator(device="cuda").manual_seed(19)
    num_global, ep_size, top_k, hidden, ispp = 256, 32, 8, 6144, 2048
    num_local = num_global // ep_size
    ep_rank = 5
    raw = _make_weights(num_local, hidden, ispp, generator)
    x = torch.randn(num_tokens, hidden, generator=generator, device="cuda").to(
        torch.bfloat16
    )
    topk_ids, topk_weights = _routing(num_tokens, num_global, top_k, generator)
    # Route token 0 to this rank so even the 4-token batch exercises the GEMMs.
    topk_ids[0, 0] = ep_rank * num_local

    local_ids = topk_ids - ep_rank * num_local
    local_ids = torch.where((local_ids >= 0) & (local_ids < num_local), local_ids, -1)
    expected = _reference(x, raw, local_ids, topk_weights, limit=7.0)

    w = _Experts(raw, ep_size, ep_rank, 7.0)
    plan = {"activation": "swiglu", "deterministic_finalize": True}
    marlin_mxfp8_moe_weights(plan, w)
    actual = marlin_mxfp8_precomputed_moe_apply(
        plan, x, w, None, topk_weights=topk_weights, topk_ids=topk_ids
    ).float()
    _assert_close(actual, expected)
    untouched = (local_ids < 0).all(dim=1)
    assert torch.count_nonzero(actual[untouched]) == 0


@pytest.mark.parametrize("num_experts,hidden,ispp", [(3, 256, 128), (2, 6144, 2048)])
def test_marlin_dequant_inverts_the_repack(num_experts: int, hidden: int, ispp: int):
    """Dequantizing the Marlin layout reproduces the bf16 experts bitwise, with
    w13's gate/up halves swapped into the CUTLASS order."""
    _requires_sm90()
    generator = torch.Generator(device="cuda").manual_seed(23)
    raw = _make_weights(num_experts, hidden, ispp, generator, scale_range=(100, 160))
    w = _Experts({k: v.clone() for k, v in raw.items()}, 1, 0, None)
    marlin_mxfp8_moe_weights(
        {"activation": "swiglu", "deterministic_finalize": True}, w
    )

    w13 = torch.empty(
        num_experts, 2 * ispp, hidden, dtype=torch.bfloat16, device="cuda"
    )
    w2 = torch.empty(num_experts, hidden, ispp, dtype=torch.bfloat16, device="cuda")
    dequantize_marlin_mxfp8(w.w13_weight, w.w13_weight_scale_inv, w13, swap_halves=True)
    dequantize_marlin_mxfp8(w.w2_weight, w.w2_weight_scale_inv, w2, swap_halves=False)

    expected_w13 = _dequant(raw["w13_weight"], raw["w13_weight_scale_inv"])
    assert torch.equal(w13, torch.cat(expected_w13.chunk(2, dim=1)[::-1], dim=1))
    assert torch.equal(w2, _dequant(raw["w2_weight"], raw["w2_weight_scale_inv"]))


@pytest.mark.parametrize("limit", [None, 3.0])
@pytest.mark.parametrize("ep_size", [1, 2])
def test_dense_prefill_path_matches_reference(
    ep_size: int, limit: float | None, monkeypatch
):
    """Prefill-sized batches take the dequant + CUTLASS bf16 path; it must match
    the reference and the Marlin path it replaces, ``-1`` slots included."""
    _requires_sm90()
    generator = torch.Generator(device="cuda").manual_seed(29)
    num_experts, top_k, hidden, ispp, num_tokens = 16, 4, 512, 256, 1024
    assert num_tokens * top_k / num_experts >= mxfp8._DENSE_MIN_ROUTES_PER_EXPERT
    raw = _make_weights(num_experts, hidden, ispp, generator, scale_range=(124, 126))
    x = torch.randn(num_tokens, hidden, generator=generator, device="cuda").to(
        torch.bfloat16
    )
    topk_ids, topk_weights = _routing(num_tokens, num_experts, top_k, generator)
    topk_ids[::3, 1] = -1

    expected = _reference(x, raw, topk_ids, topk_weights, limit)
    dense = _run(raw, x, topk_ids, topk_weights, ep_size, limit)
    _assert_close(dense, expected)
    monkeypatch.setattr(mxfp8, "_DENSE_MIN_ROUTES_PER_EXPERT", float("inf"))
    marlin = _run(raw, x, topk_ids, topk_weights, ep_size, limit)
    _assert_close(dense, marlin)


def test_prefill_stays_on_marlin_when_dequant_cannot_tile_ispp():
    """ispp 192 meets Marlin's alignment but not the dequant's K width, so a
    prefill-sized batch must keep the Marlin GEMMs."""
    _requires_sm90()
    generator = torch.Generator(device="cuda").manual_seed(31)
    num_experts, top_k, hidden, ispp, num_tokens = 8, 4, 256, 192, 512
    assert num_tokens * top_k / num_experts >= mxfp8._DENSE_MIN_ROUTES_PER_EXPERT
    raw = _make_weights(num_experts, hidden, ispp, generator, scale_range=(124, 126))
    w = _Experts({k: v.clone() for k, v in raw.items()}, 1, 0, None)
    plan = {"activation": "swiglu", "deterministic_finalize": True}
    marlin_mxfp8_moe_weights(plan, w)
    assert not w._moe_backend_state.dense_prefill
    x = torch.randn(num_tokens, hidden, generator=generator, device="cuda").to(
        torch.bfloat16
    )
    topk_ids, topk_weights = _routing(num_tokens, num_experts, top_k, generator)
    actual = marlin_mxfp8_precomputed_moe_apply(
        plan, x, w, None, topk_weights=topk_weights, topk_ids=topk_ids
    ).float()
    _assert_close(actual, _reference(x, raw, topk_ids, topk_weights, limit=None))


def test_marlin_mxfp8_rejects_malformed_scales():
    _requires_sm90()
    generator = torch.Generator(device="cuda").manual_seed(3)
    raw = _make_weights(2, 256, 128, generator)
    raw["w2_weight_scale_inv"] = raw["w2_weight_scale_inv"][..., :-1].contiguous()
    w = _Experts(raw, 1, 0, None)
    with pytest.raises(ValueError, match="1x32 blocks"):
        marlin_mxfp8_moe_weights(
            {"activation": "swiglu", "deterministic_finalize": True}, w
        )
