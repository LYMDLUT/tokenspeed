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


"""Hy4-flash model-specific fused Triton operators."""

from tokenspeed_kernel.ops.model.hy4_flash.dfly import (
    hy4_dfly_block_attn_merge_sink,
    hy4_dfly_build_decode_round_inputs,
    hy4_dfly_shared_kv_dsa,
    hy4_dual_rmsnorm_cat,
    hy4_merge_attention_and_sink,
    hy4_vocab_argmax_into,
)
from tokenspeed_kernel.ops.model.hy4_flash.engram import (
    hy4_build_engram_inputs_from_history,
    hy4_engram_gate_combine,
    hy4_engram_hash_and_embed,
    hy4_engram_hash_ngrams,
    hy4_engram_lookup_norm_flatten,
)
from tokenspeed_kernel.ops.model.hy4_flash.fusion import (
    hy4_add_rmsnorm_quantize_fp8,
    hy4_sigmoid_bias_topk,
    hy4_sigmoid_gate_quantize_fp8,
    hy4_silu_and_mul_quantize_fp8,
    hy4_split_routed_and_shared_swiglu,
)

__all__ = [
    "hy4_merge_attention_and_sink",
    "hy4_dfly_build_decode_round_inputs",
    "hy4_vocab_argmax_into",
    "hy4_dual_rmsnorm_cat",
    "hy4_dfly_block_attn_merge_sink",
    "hy4_dfly_shared_kv_dsa",
    "hy4_engram_hash_ngrams",
    "hy4_engram_hash_and_embed",
    "hy4_engram_lookup_norm_flatten",
    "hy4_engram_gate_combine",
    "hy4_build_engram_inputs_from_history",
    "hy4_sigmoid_bias_topk",
    "hy4_sigmoid_gate_quantize_fp8",
    "hy4_split_routed_and_shared_swiglu",
    "hy4_add_rmsnorm_quantize_fp8",
    "hy4_silu_and_mul_quantize_fp8",
]
