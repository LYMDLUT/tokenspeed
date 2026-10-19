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


"""Public SM90 FlashMLA: real selection, adapter contracts and CUDA execution."""

import subprocess
import sys
import textwrap
from dataclasses import replace

import pytest
import torch
from tokenspeed_kernel.ops.attention.dsa import dsa_decode, dsa_prefill
from tokenspeed_kernel.platform import ArchVersion, current_platform
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.selection import NoKernelFoundError, select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

_PLATFORM = current_platform()
pytestmark = pytest.mark.skipif(
    not (_PLATFORM.is_nvidia and _PLATFORM.arch_version == ArchVersion(9, 0)),
    reason="The public sparse provider requires NVIDIA SM90",
)


def _traits(mode, heads, sink):
    return {
        "q_len": 1,
        "num_heads": heads,
        "qk_nope_head_dim": 192,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "page_size": 64,
        "topk": 2048,
        "has_kv_cache": True,
        "has_sparse_kv_cache": False,
        "logit_cap": False,
        "return_lse": True,
        "sinks": sink,
        "topk_layout": "global_slots",
        "slot_order": "selection",
    }


@pytest.mark.parametrize("mode", ["dsa_prefill", "dsa_decode"])
@pytest.mark.parametrize("heads", [2, 64, 128])
@pytest.mark.parametrize("sink", [False, True])
def test_registering_public_provider_preserves_actual_auto_selection(
    monkeypatch, mode, heads, sink
):
    registry = KernelRegistry.get()
    baseline = KernelRegistry()
    for spec in registry.list_kernels():
        if spec.solution != "flashmla_public":
            baseline.register(spec, registry.get_impl(spec.name))
    monkeypatch.delenv(
        f"TOKENSPEED_KERNEL_OVERRIDE_ATTENTION_{mode.upper()}", raising=False
    )
    kwargs = dict(
        traits=_traits(mode, heads, sink),
        platform=_PLATFORM,
    )
    signature = format_signature(q=dense_tensor_format(torch.bfloat16))
    with monkeypatch.context() as patch:
        patch.setattr(KernelRegistry, "_instance", baseline)
        previous = select_kernel("attention", mode, signature, **kwargs).name
    registry.clear_cache()
    current = select_kernel("attention", mode, signature, **kwargs).name
    forced = select_kernel(
        "attention", mode, signature, solution="flashmla_public", **kwargs
    ).name
    assert current == previous
    assert current != forced
    assert forced == f"flashmla_public_sm90_{mode}"


def test_public_registration_does_not_need_the_legacy_extension():
    # A separate interpreter has no cached DSA/DSV4/MLA modules to hide an
    # optional-dependency import failure. This imports only, without a forward.
    script = textwrap.dedent("""
        import builtins
        original_import = builtins.__import__
        def without_flashmla(name, *args, **kwargs):
            if name == "flash_mla" or name.startswith("flash_mla."):
                raise ImportError("legacy extension intentionally unavailable")
            return original_import(name, *args, **kwargs)
        builtins.__import__ = without_flashmla
        import tokenspeed_kernel.ops.attention.dsa
        import tokenspeed.runtime.layers.attention.backends
        from tokenspeed_kernel.registry import KernelRegistry
        registry = KernelRegistry.get()
        assert registry.get_by_name("flashmla_public_sm90_dsa_prefill") is not None
        assert registry.get_by_name("flashmla_public_sm90_dsa_decode") is not None
        assert registry.get_by_name("flashmla_dsa_prefill") is None
        print("cold import without legacy FlashMLA passed")
        """)
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_public_selection_refuses_sm100_fp8_and_packed_kv():
    traits = _traits("dsa_decode", 64, True)
    for platform, dtype, updates in (
        (replace(_PLATFORM, arch_version=ArchVersion(10, 0)), torch.bfloat16, {}),
        (_PLATFORM, torch.float8_e4m3fn, {}),
        (_PLATFORM, torch.bfloat16, {"has_sparse_kv_cache": True}),
    ):
        signature = format_signature(q=dense_tensor_format(dtype))
        with pytest.raises(NoKernelFoundError):
            select_kernel(
                "attention",
                "dsa_decode",
                signature,
                platform=platform,
                traits=traits | updates,
                solution="flashmla_public",
            )


def _kwargs(q, kv, slots, lens, sink):
    return dict(
        q=q,
        kv_cache=kv,
        sparse_kv_cache=None,
        topk_slots=slots,
        topk_lens=lens,
        max_seqlen_k=kv.shape[0],
        qk_nope_head_dim=192,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        softmax_scale=0.0625,
        page_size=64,
        k_scale=1.0,
        return_lse=True,
        solution="flashmla_public",
        slot_order="selection",
        attn_sink=sink,
    )


@pytest.mark.parametrize("phase", ["prefill", "decode"])
@pytest.mark.parametrize("with_lengths", [False, True])
def test_adapter_preserves_strides_padding_output_and_lse(
    monkeypatch, phase, with_lengths
):
    from tokenspeed_kernel.thirdparty import flashmla_public_sm90 as raw

    q = torch.zeros(1, 3, 2, 1152, device="cuda", dtype=torch.bfloat16)[..., ::2]
    kv = torch.zeros(128, 2, 576, device="cuda", dtype=torch.bfloat16)[:, :1]
    slots = torch.arange(1024, device="cuda", dtype=torch.int32).repeat(3, 1)[:, :512]
    slots[0].fill_(-1)
    lens = (
        torch.tensor([0, 1, 17], device="cuda", dtype=torch.int64)
        if with_lengths
        else None
    )
    sink = torch.tensor([-0.5, 0.5], device="cuda")
    out = torch.empty(3, 2 * 512, device="cuda", dtype=torch.bfloat16)
    calls = []

    def raw_call(**kwargs):
        calls.append(kwargs)
        assert kwargs["q"].shape == (3, 64, 576)
        assert kwargs["q"].is_contiguous()
        assert kwargs["kv"].data_ptr() == kv.data_ptr()
        assert kwargs["kv"].stride() == kv.stride()
        assert kwargs["indices"].shape == (3, 1, 512)
        assert kwargs["sm_scale"] == 0.0625
        assert kwargs["d_v"] == 512
        assert bool(torch.isneginf(kwargs["attn_sink"][2:]).all())
        if with_lengths:
            assert kwargs["topk_length"].dtype == torch.int32
            assert bool((kwargs["indices"][1, :, 1:] == -1).all())
        else:
            assert kwargs["topk_length"] is None
            assert kwargs["indices"].stride(0) == slots.stride(0)
        return (
            torch.full((3, 64, 512), 3.0, dtype=torch.bfloat16, device="cuda"),
            torch.full((3, 64), 999.0, device="cuda"),
            torch.full((3, 64), 2.0, device="cuda"),
        )

    monkeypatch.setattr(raw, "flash_mla_sparse_fwd", raw_call)
    kwargs = _kwargs(q, kv, slots, lens, sink)
    kwargs["out"] = out
    if phase == "decode":
        kwargs["q_len_per_req"] = 3
    result, lse = (dsa_prefill if phase == "prefill" else dsa_decode)(**kwargs)
    assert len(calls) == 1
    assert result is out
    assert bool((result[0] == 0).all())
    assert bool((result[1:] == 3).all())
    assert bool(torch.isneginf(lse[0]).all())
    expected = torch.logaddexp(torch.full((2, 2), 2.0, device="cuda"), sink)
    torch.testing.assert_close(lse[1:], expected, rtol=0, atol=0)
    kwargs["out"] = torch.empty(out.shape, dtype=out.dtype, device="cpu")
    with pytest.raises(ValueError, match="output buffer"):
        (dsa_prefill if phase == "prefill" else dsa_decode)(**kwargs)
    assert len(calls) == 1


def test_public_raw_values_max_lse_and_graph_replay():
    from tokenspeed_kernel.thirdparty.flashmla_public_sm90 import flash_mla_sparse_fwd

    q = torch.zeros(4, 64, 576, device="cuda", dtype=torch.bfloat16)
    values = torch.arange(64, device="cuda", dtype=torch.float32) / 128
    kv = values[:, None, None].expand(64, 1, 576).contiguous().bfloat16()
    slots = torch.arange(512, device="cuda", dtype=torch.int32).repeat(4, 1)
    lengths = torch.tensor([0, 1, 17, 64], device="cuda", dtype=torch.int32)
    slots.masked_fill_(torch.arange(512, device="cuda")[None] >= lengths[:, None], -1)
    sink = torch.zeros(64, device="cuda")
    args = dict(
        q=q,
        kv=kv,
        indices=slots[:, None],
        sm_scale=0.0625,
        d_v=512,
        attn_sink=sink,
        topk_length=lengths,
    )
    actual, max_logits, raw_lse = flash_mla_sparse_fwd(**args)
    expected = torch.zeros(4, device="cuda")
    for row, count in enumerate((0, 1, 17, 64)):
        expected[row] = values[:count].sum() / (count + 1)
    torch.testing.assert_close(
        actual.float(), expected[:, None, None].expand_as(actual), rtol=0, atol=1e-3
    )
    torch.testing.assert_close(
        max_logits[1:], torch.zeros_like(max_logits[1:]), rtol=0, atol=0
    )
    torch.testing.assert_close(
        raw_lse[1:], lengths[1:, None].float().log().expand(3, 64), rtol=0, atol=1e-6
    )
    assert bool(torch.isneginf(max_logits[0]).all())
    assert bool(torch.isposinf(raw_lse[0]).all())
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    facade_args = _kwargs(q, kv, slots, lengths, sink)
    dsa_decode(**facade_args)
    with torch.cuda.graph(graph):
        captured = dsa_decode(**facade_args)
    q.fill_(0.003)
    graph.replay()
    eager = dsa_decode(**facade_args)
    for got, wanted in zip(captured, eager):
        torch.testing.assert_close(got, wanted, rtol=0, atol=0)
    # Nonzero Q/K oracle. Every channel of a KV row is its value, allowing
    # FP32 logits without a TF32-sensitive reference GEMM.
    repeated, max_logits, raw_lse = flash_mla_sparse_fwd(**args)
    logits = q.float().sum(-1, keepdim=True) * values * args["sm_scale"]
    live = torch.arange(64, device="cuda")[None, :] < lengths[:, None]
    logits = logits.masked_fill(~live[:, None, :], -float("inf"))
    denominator = torch.logsumexp(
        torch.cat((logits, sink[None, :, None].expand(4, -1, 1)), -1), -1
    )
    probabilities = (logits - denominator[:, :, None]).exp()
    expected = (probabilities * values).sum(-1)
    torch.testing.assert_close(
        repeated.float(), expected[:, :, None].expand_as(repeated), rtol=0.01, atol=1e-3
    )
    torch.testing.assert_close(max_logits[1:], logits[1:].amax(-1), rtol=0, atol=1e-5)
    torch.testing.assert_close(
        raw_lse[1:], torch.logsumexp(logits[1:], -1), rtol=0, atol=1e-5
    )


def test_public_sinkless_split_uses_public_provider_and_empty_identity(monkeypatch):
    from tokenspeed_kernel.thirdparty import flashmla_public_sm90 as raw

    original = raw.flash_mla_sparse_fwd
    calls = []

    def public_call(**kwargs):
        calls.append(kwargs["q"].shape[0])
        return original(**kwargs)

    monkeypatch.setattr(raw, "flash_mla_sparse_fwd", public_call)
    q = torch.zeros(2, 2, 576, device="cuda", dtype=torch.bfloat16)
    kv = torch.full((64, 1, 576), 0.25, device="cuda", dtype=torch.bfloat16)
    slots = torch.zeros(2, 512, device="cuda", dtype=torch.int32)
    lengths = torch.tensor([0, 17], device="cuda", dtype=torch.int32)
    result, lse = dsa_decode(**_kwargs(q, kv, slots, lengths, None))
    assert len(calls) == 1 and calls[0] > q.shape[0]
    assert bool((result[0] == 0).all()) and bool(torch.isneginf(lse[0]).all())
    torch.testing.assert_close(
        result[1], torch.full_like(result[1], 0.25), rtol=0, atol=0
    )
    torch.testing.assert_close(
        lse[1], torch.full_like(lse[1], 17.0).log(), rtol=0, atol=1e-6
    )
