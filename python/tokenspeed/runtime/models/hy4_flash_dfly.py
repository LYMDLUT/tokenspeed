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

"""Hy4-flash DFly2 block-parallel speculative draft model.

Architecture (matching released ``tencent/Hy4-flash-FP8`` ``model.dfly_layers.*``
tensors and ``AngelSpec`` ``DFly2DraftModel``):

* Shares the target model's ``embed_tokens`` and FP32 ``lm_head`` (no duplicate
  vocab tables in memory).
* ``mtp_fusion`` (``mode="slot0"``): fuses the target's post-norm final hidden
  state at the last accepted position ``a - 1`` with the anchor embedding
  ``E(x_a)`` and adds the gated residual to block slot 0.
* 5 causal dual-source Gated-MLA layers (32 heads, ``learnable_sink_param``,
  ``linear_gate``, ``post_proj_norm``, Dense SwiGLU MLP, no iHC):
  - Each draft layer ``i`` reuses the target's compressed MLA KV cache from
    ``target_kv_layers[i]`` (``[25, 29, 33, 37, 41]``) and the top-2048 context
    slots selected by ``target_index_layer[i]`` (``[25, 29, 33, 37, 41]``) when
    context length exceeds ``target_index_topk``.
  - The draft block's own compressed latents are projected by the draft layer
    itself and combined with the target context through exact online softmax
    merging plus a single learnable-sink rescaling.
* ``final_norm`` -> sequential ``hidden_correction`` chain walk conditioned on
  the previous proposed token embedding -> target FP32 ``lm_head``.
* Optional ``confidence_head`` predicting per-slot acceptance logits.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from tokenspeed_kernel.ops.activation import silu_and_mul
from tokenspeed_kernel.ops.attention import apply_attention_sink
from tokenspeed_kernel.ops.attention.dsa import dsa_decode
from tokenspeed_kernel.ops.attention.kpool.hy4_compress import (
    apply_interleaved_rope_last_dim,
    compute_hy4_mrope_cos_sin,
    hy4_apply_interleaved_rope_inplace,
    hy4_kv_latent_norm_and_rope_inplace,
    hy4_split_qkv_indexer_norm_rope,
)
from tokenspeed_kernel.ops.attention.mla import mla_project_value
from tokenspeed_kernel.ops.gemm import bmm
from tokenspeed_kernel.ops.gemm.triton_skinny import skinny_mm, skinny_mm_supported
from tokenspeed_kernel.ops.model.hy4_flash.dfly import (
    hy4_dfly_block_attn_merge_sink,
    hy4_dfly_shared_kv_dsa,
    hy4_dual_rmsnorm_cat,
    hy4_merge_attention_and_sink,
)
from tokenspeed_kernel.ops.model.hy4_flash.fusion import (
    hy4_add_rmsnorm_quantize_fp8,
    hy4_sigmoid_gate_quantize_fp8,
    hy4_silu_and_mul_quantize_fp8,
)
from torch import nn

from tokenspeed.runtime.configs.hy4_flash_config import (
    HYV4FlashConfig,
    HYV4FlashDFlyConfig,
)
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.layers.attention.kv_cache.hybrid_hy4_flash import (
    HybridHy4FlashTokenToKVPool,
)
from tokenspeed.runtime.layers.layernorm import RMSNorm
from tokenspeed.runtime.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from tokenspeed.runtime.layers.mlp import gated_activation
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from tokenspeed.runtime.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from tokenspeed.runtime.models.base import BaseCausalLM
from tokenspeed.runtime.models.hy4_flash import (
    HYV4FlashLogitsProcessor,
    _apply_hy4_rmsnorm,
    _dequantize_fp8_blockwise,
    _enable_hopper_deepgemm,
    _is_unquantized_no_bias_linear,
    _linear_cpu_or_cuda,
    _o_proj_fp8_block_mode,
    build_hy4_lm_head,
)
from tokenspeed.runtime.utils import add_prefix

logger = logging.getLogger(__name__)

_DFLY_CHECKPOINT_PREFIX = "model.dfly_layers."


def _normalize_block_positions_3d(
    positions: torch.Tensor, *, batch_size: int, block_size: int
) -> torch.Tensor:
    """Normalize block positions to ``[3, B, block_size]`` int64 M-RoPE coordinates."""
    if positions.ndim == 3 and positions.shape[0] == 3:
        return positions.to(torch.int64)
    if positions.ndim == 3 and positions.shape[1] == 3:
        return positions.transpose(0, 1).to(torch.int64)
    if positions.ndim == 2:
        return positions.to(torch.int64).unsqueeze(0).expand(3, -1, -1)
    if positions.ndim == 1:
        pos2d = positions.to(torch.int64).view(batch_size, block_size)
        return pos2d.unsqueeze(0).expand(3, -1, -1)
    raise ValueError(
        f"Unsupported DFly2 block positions shape {tuple(positions.shape)}"
    )


def _merge_two_attention_states(
    out_a: torch.Tensor,
    lse_a: torch.Tensor,
    out_b: torch.Tensor,
    lse_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Numerically exact FP32 online softmax merge for two attention partitions.

    Handles empty partitions where ``lse_a == -inf`` or ``lse_b == -inf``
    without producing NaNs.

    Args:
        out_a: ``[..., D]`` first partition value sum (already normalized by ``lse_a``).
        lse_a: ``[...]`` FP32 log-sum-exp (natural log) of the first partition.
        out_b: ``[..., D]`` second partition value sum.
        lse_b: ``[...]`` FP32 log-sum-exp (natural log) of the second partition.

    Returns:
        ``(merged_out, merged_lse)`` with ``merged_out`` in ``out_b.dtype`` and
        ``merged_lse`` in ``torch.float32``.
    """
    lse_a_f = lse_a.float()
    lse_b_f = lse_b.float()
    max_lse = torch.maximum(lse_a_f, lse_b_f)
    both_empty = torch.isneginf(max_lse)
    safe_max = torch.where(both_empty, torch.zeros_like(max_lse), max_lse)

    exp_a = torch.where(
        torch.isneginf(lse_a_f),
        torch.zeros_like(lse_a_f),
        torch.exp(lse_a_f - safe_max),
    )
    exp_b = torch.where(
        torch.isneginf(lse_b_f),
        torch.zeros_like(lse_b_f),
        torch.exp(lse_b_f - safe_max),
    )
    denom = exp_a + exp_b
    safe_denom = torch.where(both_empty, torch.ones_like(denom), denom)

    w_a = (exp_a / safe_denom).unsqueeze(-1)
    w_b = (exp_b / safe_denom).unsqueeze(-1)
    merged_out = (w_a * out_a.float() + w_b * out_b.float()).to(out_b.dtype)
    merged_lse = torch.where(
        both_empty,
        torch.full_like(max_lse, float("-inf")),
        safe_max + torch.log(safe_denom),
    )
    return merged_out, merged_lse


def _apply_sink_to_latent(
    latent_out: torch.Tensor,
    lse: torch.Tensor,
    sink: torch.Tensor,
) -> torch.Tensor:
    """Apply learnable attention sink rescale ``sigmoid(lse - sink)`` to ``[B, S, H, D]``.

    ``lse`` has shape ``[B, S, H]`` in natural-log base and ``sink`` has shape ``[H]``.
    """
    if latent_out.is_cuda:
        bsz, seq_len, num_heads, dim = latent_out.shape
        out_flat = latent_out.reshape(bsz * seq_len, num_heads, dim).contiguous()
        lse_flat = lse.reshape(bsz * seq_len, num_heads).contiguous()
        apply_attention_sink(out_flat, lse_flat, sink, inplace=True)
        return out_flat.view(bsz, seq_len, num_heads, dim)
    sink_f = sink.float().view(1, 1, -1)
    rescale = torch.sigmoid(lse.float() - sink_f).unsqueeze(-1)
    return (latent_out.float() * rescale).to(latent_out.dtype)


# ===----------------------------------------------------------------------=== #
# DFly2 Ancillary Modules (MTPFusion, HiddenCorrection, ConfidenceHead)
# ===----------------------------------------------------------------------=== #


class HYV4FlashDFlyMTPFusion(nn.Module):
    """MTP input fusion matching ``model.dfly_layers.mtp_fusion.*``."""

    def __init__(
        self,
        hidden_size: int,
        mode: str,
        rms_norm_eps: float,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if mode not in ("slot0", "slot1", "broadcast"):
            raise ValueError(
                f"Unsupported DFly2 mtp_fusion mode={mode!r}; expected slot0/slot1/broadcast"
            )
        self.mode = mode
        self.gate_slot = 1 if mode == "slot1" else 0
        self.enorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.hnorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.eh_proj = ReplicatedLinear(
            2 * hidden_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("eh_proj", prefix),
        )
        self.gate = nn.Parameter(torch.zeros(hidden_size))
        self.mask_gate: nn.Parameter | None = (
            nn.Parameter(torch.zeros(hidden_size)) if mode == "broadcast" else None
        )

    def fuse(
        self, anchor_embeds: torch.Tensor, prev_hidden: torch.Tensor
    ) -> torch.Tensor:
        """Compute ``eh_proj(cat([enorm(E(x_a)), hnorm(h_{a-1})]))``."""
        if (
            anchor_embeds.is_cuda
            and prev_hidden.dtype == anchor_embeds.dtype
            and self.enorm.variance_epsilon == self.hnorm.variance_epsilon
        ):
            concat = hy4_dual_rmsnorm_cat(
                anchor_embeds,
                prev_hidden,
                self.enorm.weight,
                self.hnorm.weight,
                eps=self.enorm.variance_epsilon,
            )
            return _linear_cpu_or_cuda(self.eh_proj, concat)
        norm_e = _apply_hy4_rmsnorm(anchor_embeds, self.enorm)
        norm_h = _apply_hy4_rmsnorm(prev_hidden.to(anchor_embeds.dtype), self.hnorm)
        concat = torch.cat((norm_e, norm_h), dim=-1)
        return _linear_cpu_or_cuda(self.eh_proj, concat)

    def apply_to_block(
        self,
        block_embeds: torch.Tensor,
        prev_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Add gated MTP fusion signal onto ``block_embeds [B, block_size, D]``."""
        anchor_embeds = block_embeds[:, 0, :]
        fused = self.fuse(anchor_embeds, prev_hidden)
        out = block_embeds.clone()
        if self.mask_gate is not None:
            out[:, 0, :] = out[:, 0, :] + self.gate.to(fused.dtype) * fused
            out[:, 1:, :] = out[:, 1:, :] + self.mask_gate.to(fused.dtype).unsqueeze(
                0
            ).unsqueeze(0) * fused.unsqueeze(1)
            return out
        slot = self.gate_slot
        if slot < out.shape[1]:
            out[:, slot, :] = out[:, slot, :] + self.gate.to(fused.dtype) * fused
        return out


class HYV4FlashDFlyHiddenCorrection(nn.Module):
    """TreeFlash residual hidden correction matching ``model.dfly_layers.hidden_correction.*``."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.hidden_norm = RMSNorm(self.hidden_size, eps=rms_norm_eps)
        self.embed_norm = RMSNorm(self.hidden_size, eps=rms_norm_eps)
        in_features = 2 * self.hidden_size
        self.gate_proj = ReplicatedLinear(
            in_features,
            self.intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_proj", prefix),
        )
        self.up_proj = ReplicatedLinear(
            in_features,
            self.intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("up_proj", prefix),
        )
        self.down_proj = ReplicatedLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        self._packed_gate_up_weight: torch.Tensor | None = None

    def prepare_packed_weights(self) -> None:
        """Pack unquantized ``[gate_proj; up_proj]`` weights into one contiguous GEMM."""
        if (
            _is_unquantized_no_bias_linear(self.gate_proj)
            and _is_unquantized_no_bias_linear(self.up_proj)
            and self.gate_proj.weight.dtype == self.up_proj.weight.dtype
            and self.gate_proj.weight.device == self.up_proj.weight.device
        ):
            self._packed_gate_up_weight = torch.cat(
                (self.gate_proj.weight.detach(), self.up_proj.weight.detach()), dim=0
            ).contiguous()
        else:
            self._packed_gate_up_weight = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_token_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``h + down(silu(gate(cat(hnorm(h), enorm(e)))) * up(...))``."""
        packed_gu = self._packed_gate_up_weight
        if (
            hidden_states.is_cuda
            and packed_gu is not None
            and packed_gu.device == hidden_states.device
            and packed_gu.dtype == hidden_states.dtype
            and prev_token_embeds.dtype == hidden_states.dtype
            and self.hidden_norm.variance_epsilon == self.embed_norm.variance_epsilon
        ):
            gate_in = hy4_dual_rmsnorm_cat(
                hidden_states,
                prev_token_embeds,
                self.hidden_norm.weight,
                self.embed_norm.weight,
                eps=self.hidden_norm.variance_epsilon,
            )
            if skinny_mm_supported(gate_in, packed_gu):
                gate_up = skinny_mm(
                    gate_in,
                    packed_gu,
                    None,
                    out_dtype=gate_in.dtype,
                    compact_m=True,
                )
            else:
                gate_up = F.linear(gate_in, packed_gu)
            act = silu_and_mul(gate_up)
            if _is_unquantized_no_bias_linear(self.down_proj) and skinny_mm_supported(
                act, self.down_proj.weight
            ):
                delta = skinny_mm(
                    act,
                    self.down_proj.weight,
                    None,
                    out_dtype=act.dtype,
                    compact_m=True,
                )
            else:
                delta = _linear_cpu_or_cuda(self.down_proj, act)
            return hidden_states + delta.to(hidden_states.dtype)
        h_norm = _apply_hy4_rmsnorm(hidden_states, self.hidden_norm)
        e_norm = _apply_hy4_rmsnorm(
            prev_token_embeds.to(hidden_states.dtype), self.embed_norm
        )
        gate_in = torch.cat((h_norm, e_norm), dim=-1)
        gate_val = _linear_cpu_or_cuda(self.gate_proj, gate_in)
        up_val = _linear_cpu_or_cuda(self.up_proj, gate_in)
        delta = _linear_cpu_or_cuda(self.down_proj, F.silu(gate_val) * up_val)
        return hidden_states + delta.to(hidden_states.dtype)


class HYV4FlashDFlyConfidenceHead(nn.Module):
    """Per-slot acceptance-probability logit head (``model.dfly_layers.confidence_head.*``)."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(int(hidden_size), 1, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states.to(self.proj.weight.dtype)).squeeze(-1)


# ===----------------------------------------------------------------------=== #
# DFly2 Dual-Source Gated-MLA Attention + Dense SwiGLU MLP + Layer
# ===----------------------------------------------------------------------=== #


class HYV4FlashDFlyAttention(nn.Module):
    """Dual-source Gated MLA reading target KV cache + draft block KV."""

    def __init__(
        self,
        config: HYV4FlashDFlyConfig,
        layer_id: int,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = int(layer_id)
        self.mapping = mapping
        self.target_kv_layer = int(config.target_kv_layers[layer_id])
        self.target_index_layer = int(config.target_index_layer[layer_id])
        self.attn_causal = bool(config.attn_causal)

        self.num_heads = int(config.num_attention_heads)
        self.qk_nope_head_dim = int(config.qk_nope_head_dim)
        self.qk_rope_head_dim = int(config.qk_rope_head_dim)
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = int(config.v_head_dim)
        self.q_lora_rank = int(config.q_lora_rank)
        self.kv_lora_rank = int(config.kv_lora_rank)
        self.scaling = self.qk_head_dim ** (-0.5)

        tp_rank = mapping.attn.tp_rank
        tp_size = mapping.attn.tp_size
        tp_group = mapping.attn.tp_group
        if self.num_heads % tp_size != 0:
            raise ValueError(
                f"DFly2 layer {layer_id}: num_heads={self.num_heads} not divisible by attn tp_size={tp_size}"
            )
        self.num_local_heads = self.num_heads // tp_size
        self.tp_rank = tp_rank
        self.tp_size = tp_size

        self.q_a_proj = ReplicatedLinear(
            config.hidden_size,
            self.q_lora_rank,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("q_a_proj", prefix),
        )
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(
            self.q_lora_rank,
            self.num_heads * self.qk_head_dim,
            bias=False,
            quant_config=quant_config,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            prefix=add_prefix("q_b_proj", prefix),
        )
        self.kv_a_proj_with_mqa = ReplicatedLinear(
            config.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("kv_a_proj_with_mqa", prefix),
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            prefix=add_prefix("kv_b_proj", prefix),
        )
        self.linear_gate: ColumnParallelLinear | None = (
            ColumnParallelLinear(
                config.hidden_size,
                self.num_heads * self.v_head_dim,
                bias=False,
                quant_config=quant_config,
                tp_rank=tp_rank,
                tp_size=tp_size,
                tp_group=tp_group,
                prefix=add_prefix("linear_gate", prefix),
            )
            if config.gated_mla
            else None
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            prefix=add_prefix("o_proj", prefix),
        )
        self.post_proj_norm: RMSNorm | None = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.attn_post_proj_norm
            else None
        )
        self.learnable_sink_param: nn.Parameter | None = None
        if config.learnable_sink:
            self.learnable_sink_param = nn.Parameter(
                torch.full(
                    (self.num_local_heads,),
                    float(config.learnable_sink_init),
                    dtype=torch.get_default_dtype(),
                )
            )
            self.learnable_sink_param.weight_loader = sharded_weight_loader(0, tp_rank)

        rope_params = config.rope_parameters
        self.mrope_section = tuple(int(x) for x in rope_params["mrope_section"])
        rope_theta = float(rope_params["rope_theta"])
        inv_freq = 1.0 / (
            rope_theta
            ** (
                torch.arange(0, self.qk_rope_head_dim, 2, dtype=torch.float32)
                / self.qk_rope_head_dim
            )
        )
        self.register_buffer("rope_inv_freq", inv_freq, persistent=False)

        # Absorbed MLA weight slices populated by prepare_absorbed_mla_weights().
        self.w_kc: torch.Tensor | None = None
        self.w_vc: torch.Tensor | None = None
        self._packed_in_proj_weight: torch.Tensor | None = None
        self._causal_mask_cache: dict[tuple[int, torch.device], torch.Tensor] = {}

    def prepare_packed_bf16_projections(self) -> None:
        """Pack ``[q_a_proj; kv_a_proj_with_mqa]`` into one BF16 GEMM matrix."""
        if (
            _is_unquantized_no_bias_linear(self.q_a_proj)
            and _is_unquantized_no_bias_linear(self.kv_a_proj_with_mqa)
            and self.q_a_proj.weight.dtype == self.kv_a_proj_with_mqa.weight.dtype
            and self.q_a_proj.weight.device == self.kv_a_proj_with_mqa.weight.device
        ):
            self._packed_in_proj_weight = torch.cat(
                (
                    self.q_a_proj.weight.detach(),
                    self.kv_a_proj_with_mqa.weight.detach(),
                ),
                dim=0,
            ).contiguous()
        else:
            self._packed_in_proj_weight = None

    def prepare_absorbed_mla_weights(self) -> None:
        """Materialize absorbed ``w_kc`` and ``w_vc`` from ``kv_b_proj``."""
        weight = self.kv_b_proj.weight
        if weight.dtype == torch.float8_e4m3fn:
            scale_inv = getattr(self.kv_b_proj, "weight_scale_inv", None)
            if scale_inv is not None:
                weight = _dequantize_fp8_blockwise(weight, scale_inv).to(torch.bfloat16)
            else:
                weight = weight.to(torch.bfloat16)
        w_full = weight.view(
            self.num_local_heads,
            self.qk_nope_head_dim + self.v_head_dim,
            self.kv_lora_rank,
        )
        self.w_kc = w_full[:, : self.qk_nope_head_dim, :].contiguous()
        self.w_vc = w_full[:, self.qk_nope_head_dim :, :].contiguous()

    def _ensure_absorbed_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.w_kc is None or self.w_vc is None:
            self.prepare_absorbed_mla_weights()
        assert self.w_kc is not None and self.w_vc is not None
        return self.w_kc, self.w_vc

    def _project_draft_q_and_latent(
        self,
        draft_hidden: torch.Tensor,
        positions_3d: torch.Tensor,
        *,
        rope_cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None,
        apply_q_rope: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project draft hidden ``[B, S, D]`` to ``(q_nope, q_pe, latent_draft)``."""
        bsz, block_size, hidden_dim = draft_hidden.shape
        flat_hidden = draft_hidden.reshape(bsz * block_size, hidden_dim)
        if rope_cos_sin is not None:
            cos_cast, sin_cast = rope_cos_sin
        else:
            flat_pos = positions_3d.reshape(3, bsz * block_size)
            cos, sin = compute_hy4_mrope_cos_sin(
                flat_pos, self.rope_inv_freq, self.mrope_section
            )
            cos_cast = cos.to(draft_hidden.dtype).contiguous()
            sin_cast = sin.to(draft_hidden.dtype).contiguous()

        packed_w = self._packed_in_proj_weight
        if (
            flat_hidden.is_cuda
            and packed_w is not None
            and packed_w.device == flat_hidden.device
            and packed_w.dtype == flat_hidden.dtype
        ):
            if skinny_mm_supported(flat_hidden, packed_w):
                packed_out = skinny_mm(
                    flat_hidden,
                    packed_w,
                    None,
                    out_dtype=flat_hidden.dtype,
                    compact_m=True,
                )
            else:
                packed_out = F.linear(flat_hidden, packed_w)
            q_b_fp8, q_b_mn = _o_proj_fp8_block_mode(self.q_b_proj)
            if q_b_fp8 and q_b_mn:
                _, latent_flat, _, _, q_fp8, q_scale = hy4_split_qkv_indexer_norm_rope(
                    packed_out,
                    self.q_a_layernorm.weight,
                    self.kv_a_layernorm.weight,
                    cos_cast,
                    sin_cast,
                    q_eps=self.q_a_layernorm.variance_epsilon,
                    kv_eps=self.kv_a_layernorm.variance_epsilon,
                    quantize_q_fp8_mn_major=True,
                    write_q_bf16=False,
                )
                q_states = self.q_b_proj(
                    q_fp8, block_scale=q_scale, output_dtype=flat_hidden.dtype
                )[0].view(bsz * block_size, self.num_local_heads, self.qk_head_dim)
            else:
                q_resid, latent_flat, _, _ = hy4_split_qkv_indexer_norm_rope(
                    packed_out,
                    self.q_a_layernorm.weight,
                    self.kv_a_layernorm.weight,
                    cos_cast,
                    sin_cast,
                    q_eps=self.q_a_layernorm.variance_epsilon,
                    kv_eps=self.kv_a_layernorm.variance_epsilon,
                )
                q_states = _linear_cpu_or_cuda(self.q_b_proj, q_resid).view(
                    bsz * block_size, self.num_local_heads, self.qk_head_dim
                )
            if apply_q_rope:
                hy4_apply_interleaved_rope_inplace(q_states, cos_cast, sin_cast)
            q_nope, q_pe = q_states.split(
                [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
            )
            latent_draft = latent_flat.view(
                bsz, block_size, self.kv_lora_rank + self.qk_rope_head_dim
            )
        else:
            q_resid = _apply_hy4_rmsnorm(
                _linear_cpu_or_cuda(self.q_a_proj, flat_hidden),
                self.q_a_layernorm,
            )
            q_states = _linear_cpu_or_cuda(self.q_b_proj, q_resid).view(
                bsz * block_size, self.num_local_heads, self.qk_head_dim
            )
            if apply_q_rope:
                hy4_apply_interleaved_rope_inplace(q_states, cos_cast, sin_cast)
            q_nope, q_pe = q_states.split(
                [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
            )

            kv_raw = _linear_cpu_or_cuda(self.kv_a_proj_with_mqa, flat_hidden)
            if kv_raw.is_cuda:
                latent_draft = hy4_kv_latent_norm_and_rope_inplace(
                    kv_raw,
                    self.kv_a_layernorm.weight,
                    cos_cast,
                    sin_cast,
                    eps=self.kv_a_layernorm.variance_epsilon,
                ).view(bsz, block_size, self.kv_lora_rank + self.qk_rope_head_dim)
            else:
                kv_pass, k_pe = kv_raw.split(
                    [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
                )
                k_pass = _apply_hy4_rmsnorm(kv_pass, self.kv_a_layernorm)
                k_pe = apply_interleaved_rope_last_dim(k_pe, cos_cast, sin_cast)
                latent_draft = torch.cat((k_pass, k_pe), dim=-1).view(
                    bsz, block_size, self.kv_lora_rank + self.qk_rope_head_dim
                )
        return (
            q_nope.view(bsz, block_size, self.num_local_heads, self.qk_nope_head_dim),
            q_pe.view(bsz, block_size, self.num_local_heads, self.qk_rope_head_dim),
            latent_draft,
        )

    def _project_output_tail(
        self,
        latent_out: torch.Tensor,
        draft_hidden: torch.Tensor,
        *,
        linear_gate_fp8: tuple[torch.Tensor, torch.Tensor] | None = None,
        skip_post_norm: bool = False,
    ) -> torch.Tensor:
        """Up-project absorbed latent ``[B, S, H, kv_lora]`` -> gate -> o_proj -> norm."""
        bsz, block_size, num_heads, _ = latent_out.shape
        _, w_vc = self._ensure_absorbed_weights()
        num_tokens = bsz * block_size
        if latent_out.is_cuda:
            v_out = torch.empty(
                (num_tokens, num_heads * self.v_head_dim),
                dtype=latent_out.dtype,
                device=latent_out.device,
            )
            mla_project_value(
                latent_out.reshape(num_tokens, num_heads, self.kv_lora_rank),
                w_vc.transpose(1, 2),
                gate=None,
                out=v_out,
            )
            attn_out = v_out.view(num_tokens, num_heads, self.v_head_dim)
        else:
            attn_out = torch.einsum(
                "nhm,hdm->nhd",
                latent_out.reshape(num_tokens, num_heads, self.kv_lora_rank).float(),
                w_vc.float(),
            ).to(latent_out.dtype)

        flat_hidden = draft_hidden.reshape(num_tokens, draft_hidden.shape[-1])
        flat_attn = attn_out.reshape(num_tokens, num_heads * self.v_head_dim)
        if self.linear_gate is not None:
            if linear_gate_fp8 is not None:
                gate = self.linear_gate(
                    linear_gate_fp8[0],
                    block_scale=linear_gate_fp8[1],
                    output_dtype=flat_hidden.dtype,
                )[0]
            else:
                gate = _linear_cpu_or_cuda(self.linear_gate, flat_hidden)
            fused_fp8_gate, use_mn_major = _o_proj_fp8_block_mode(self.o_proj)
            if (
                flat_attn.is_cuda
                and fused_fp8_gate
                and flat_attn.dtype in (torch.bfloat16, torch.float16)
            ):
                q_fp8, q_scale = hy4_sigmoid_gate_quantize_fp8(
                    flat_attn, gate, mn_major_tma=use_mn_major
                )
                projected = self.o_proj(q_fp8, scale=q_scale)[0]
            else:
                flat_attn = flat_attn * torch.sigmoid(gate)
                projected = _linear_cpu_or_cuda(self.o_proj, flat_attn)
        else:
            projected = _linear_cpu_or_cuda(self.o_proj, flat_attn)
        if self.post_proj_norm is not None and not skip_post_norm:
            projected = _apply_hy4_rmsnorm(projected, self.post_proj_norm)
        return projected.view(bsz, block_size, -1)

    def _absorb_q(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        *,
        rope_cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Compute absorbed query ``[B, S, H, kv_lora + qk_rope]``."""
        w_kc, _ = self._ensure_absorbed_weights()
        if q_nope.is_cuda:
            bsz, block_size, num_heads, _ = q_nope.shape
            num_tokens = bsz * block_size
            q_abs = torch.empty(
                (num_tokens, num_heads, self.kv_lora_rank + self.qk_rope_head_dim),
                dtype=q_nope.dtype,
                device=q_nope.device,
            )
            if rope_cos_sin is not None:
                hy4_apply_interleaved_rope_inplace(
                    q_pe.reshape(num_tokens, num_heads, self.qk_rope_head_dim),
                    rope_cos_sin[0],
                    rope_cos_sin[1],
                    out=q_abs[..., self.kv_lora_rank :],
                )
            else:
                q_abs[..., self.kv_lora_rank :].copy_(
                    q_pe.reshape(num_tokens, num_heads, self.qk_rope_head_dim)
                )
            if num_tokens > 0:
                bmm(
                    q_nope.reshape(
                        num_tokens, num_heads, self.qk_nope_head_dim
                    ).transpose(0, 1),
                    w_kc.to(q_nope.dtype).transpose(1, 2),
                    out=q_abs[..., : self.kv_lora_rank].transpose(0, 1),
                )
            return q_abs.view(bsz, block_size, num_heads, -1)
        q_lat = torch.einsum("bshd,hdm->bshm", q_nope.float(), w_kc.float()).to(
            q_nope.dtype
        )
        if rope_cos_sin is not None:
            bsz, block_size, num_heads, _ = q_pe.shape
            q_pe = apply_interleaved_rope_last_dim(
                q_pe.reshape(bsz * block_size, num_heads, self.qk_rope_head_dim),
                rope_cos_sin[0].unsqueeze(1),
                rope_cos_sin[1].unsqueeze(1),
            ).view(bsz, block_size, num_heads, self.qk_rope_head_dim)
        return torch.cat((q_lat, q_pe), dim=-1)

    def _get_causal_mask(self, block_size: int, device: torch.device) -> torch.Tensor:
        key = (block_size, device)
        mask = self._causal_mask_cache.get(key)
        if mask is None:
            q_idx = torch.arange(block_size, device=device).unsqueeze(1)
            k_idx = torch.arange(block_size, device=device).unsqueeze(0)
            mask = (k_idx > q_idx).unsqueeze(0).unsqueeze(0)
            self._causal_mask_cache[key] = mask
        return mask

    def _attend_draft_block_absorbed(
        self,
        q_absorbed: torch.Tensor,
        latent_draft: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute exact intra-block absorbed attention ``(out_latent, lse)``."""
        _, block_size, _, _ = q_absorbed.shape
        # Attention logits shaped [B, H, S, S].
        scores = (
            torch.einsum(
                "bshd,btd->bhst",
                q_absorbed.float(),
                latent_draft.float(),
            )
            * self.scaling
        )
        if self.attn_causal:
            scores = scores.masked_fill(
                self._get_causal_mask(block_size, scores.device), float("-inf")
            )
        lse = torch.logsumexp(scores, dim=-1)  # [B, H, S]
        probs = torch.softmax(scores, dim=-1)  # [B, H, S, S]
        v_latent = latent_draft[..., : self.kv_lora_rank].float()  # [B, S, kv_lora]
        out_latent = torch.einsum("bhst,btm->bshm", probs, v_latent).to(
            q_absorbed.dtype
        )
        return out_latent, lse.transpose(1, 2).contiguous()  # [B, S, H, M], [B, S, H]

    def _attend_context_absorbed_from_slots(
        self,
        q_absorbed: torch.Tensor,
        target_kv_buffer: torch.Tensor,
        ctx_slots: torch.Tensor,
        ctx_lens: torch.Tensor | None,
        *,
        skip_zero_ctx_mask: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Attend absorbed query ``[B, S, H, D]`` over target paged KV slots ``[B, K]``."""
        bsz, block_size, num_heads, head_dim = q_absorbed.shape
        max_ctx_k = ctx_slots.shape[1]
        if max_ctx_k == 0:
            empty_out = torch.zeros(
                (bsz, block_size, num_heads, self.kv_lora_rank),
                dtype=q_absorbed.dtype,
                device=q_absorbed.device,
            )
            empty_lse = torch.full(
                (bsz, block_size, num_heads),
                float("-inf"),
                dtype=torch.float32,
                device=q_absorbed.device,
            )
            return empty_out, empty_lse

        # Fast zero-copy CUDA path sharing per-request ctx_slots [B, K] across
        # all block_size queries without broadcasting slots/lens or copying output.
        if (
            q_absorbed.is_cuda
            and ctx_lens is not None
            and q_absorbed.dtype in (torch.bfloat16, torch.float16)
            and target_kv_buffer.dtype in (torch.bfloat16, torch.float16)
            and target_kv_buffer.ndim >= 2
            and q_absorbed.stride(-1) == 1
            and ctx_slots.stride(-1) == 1
            and ctx_lens.is_contiguous()
            and self.kv_lora_rank in (128, 512)
            and self.qk_rope_head_dim in (0, 64)
            and num_heads >= 16
            and num_heads % 16 == 0
        ):
            return hy4_dfly_shared_kv_dsa(
                q_absorbed,
                target_kv_buffer,
                ctx_slots,
                ctx_lens,
                kv_lora_rank=self.kv_lora_rank,
                qk_rope_head_dim=self.qk_rope_head_dim,
                softmax_scale=self.scaling,
            )

        # On CUDA, use Triton sparse MLA decode directly against the target KV
        # buffer without gathering a [B, K, 576] temporary tensor whenever the
        # geometry matches triton_dsa_decode's registered kernel traits (all
        # real Hy4-flash decode / CUDA-graph rounds with topk=2048/2051).
        if (
            q_absorbed.is_cuda
            and ctx_lens is not None
            and target_kv_buffer.ndim >= 2
            and self.kv_lora_rank in (128, 512)
            and self.qk_nope_head_dim in (128, 192, 256)
            and self.qk_rope_head_dim in (0, 64)
            and max_ctx_k in (512, 1024, 2048, 2049, 2050, 2051)
        ):
            # Broadcast per-request ctx_slots [B, K] -> [B * S, K] and ctx_lens [B] -> [B * S].
            q_flat = q_absorbed.reshape(
                bsz * block_size, num_heads, head_dim
            ).contiguous()
            topk_indices = (
                ctx_slots.unsqueeze(1)
                .expand(bsz, block_size, max_ctx_k)
                .reshape(bsz * block_size, max_ctx_k)
                .to(torch.int32)
                .contiguous()
            )
            topk_lens_rep = (
                ctx_lens.unsqueeze(1)
                .expand(bsz, block_size)
                .reshape(bsz * block_size)
                .to(torch.int32)
                .contiguous()
            )
            out_buf = (
                torch.empty(
                    (bsz * block_size, num_heads, self.kv_lora_rank),
                    dtype=q_absorbed.dtype,
                    device=q_absorbed.device,
                )
                if skip_zero_ctx_mask
                else torch.zeros(
                    (bsz * block_size, num_heads, self.kv_lora_rank),
                    dtype=q_absorbed.dtype,
                    device=q_absorbed.device,
                )
            )
            page_size = (
                int(target_kv_buffer.shape[1]) if target_kv_buffer.ndim == 4 else 64
            )
            out_ctx, lse_ctx = dsa_decode(
                q=q_flat,
                kv_cache=target_kv_buffer,
                sparse_kv_cache=None,
                topk_slots=topk_indices,
                topk_lens=topk_lens_rep,
                max_seqlen_k=max_ctx_k,
                qk_nope_head_dim=self.qk_nope_head_dim,
                kv_lora_rank=self.kv_lora_rank,
                qk_rope_head_dim=self.qk_rope_head_dim,
                softmax_scale=self.scaling,
                page_size=page_size,
                q_len_per_req=1,
                return_lse=True,
                out=out_buf,
                solution="triton",
                slot_order="selection",
            )
            out_ctx_4d = out_ctx.view(bsz, block_size, num_heads, self.kv_lora_rank)
            lse_ctx_3d = (
                lse_ctx.view(bsz, block_size, num_heads)
                if lse_ctx.dtype == torch.float32
                else lse_ctx.view(bsz, block_size, num_heads).float()
            )
            if skip_zero_ctx_mask:
                return out_ctx_4d, lse_ctx_3d
            # For requests with ctx_lens == 0, ensure lse_ctx is -inf so merge ignores context.
            zero_ctx = (topk_lens_rep <= 0).view(bsz, block_size, 1)
            out_ctx_4d = torch.where(
                zero_ctx.unsqueeze(-1), torch.zeros_like(out_ctx_4d), out_ctx_4d
            )
            lse_ctx_3d = torch.where(
                zero_ctx,
                torch.full_like(lse_ctx_3d, float("-inf")),
                lse_ctx_3d,
            )
            return out_ctx_4d, lse_ctx_3d

        # Exact gathered path (CPU or small-K fallback): read latents at ctx_slots.
        valid = ctx_slots >= 0
        if target_kv_buffer.ndim == 4:
            page_size = target_kv_buffer.shape[1]
            num_slots = target_kv_buffer.shape[0] * page_size
            safe_slots = ctx_slots.clamp(min=0, max=num_slots - 1).to(torch.int64)
            ctx_latent = target_kv_buffer[
                safe_slots // page_size, safe_slots % page_size, 0
            ]
        elif target_kv_buffer.ndim == 3 and target_kv_buffer.shape[1] != 1:
            page_size = target_kv_buffer.shape[1]
            num_slots = target_kv_buffer.shape[0] * page_size
            safe_slots = ctx_slots.clamp(min=0, max=num_slots - 1).to(torch.int64)
            ctx_latent = target_kv_buffer[
                safe_slots // page_size, safe_slots % page_size
            ]
        else:
            kv_2d = target_kv_buffer.reshape(-1, head_dim)
            safe_slots = ctx_slots.clamp(min=0, max=kv_2d.shape[0] - 1).to(torch.int64)
            ctx_latent = kv_2d[safe_slots]  # [B, K, D]
        scores = (
            torch.einsum("bshd,bkd->bhsk", q_absorbed.float(), ctx_latent.float())
            * self.scaling
        )
        scores = scores.masked_fill(~valid.unsqueeze(1).unsqueeze(2), float("-inf"))
        has_any = valid.any(dim=-1)  # [B]
        safe_scores = torch.where(
            has_any.view(bsz, 1, 1, 1),
            scores,
            torch.zeros_like(scores),
        )
        lse = torch.logsumexp(safe_scores, dim=-1)  # [B, H, S]
        lse = torch.where(
            has_any.view(bsz, 1, 1),
            lse,
            torch.full_like(lse, float("-inf")),
        )
        probs = torch.where(
            has_any.view(bsz, 1, 1, 1),
            torch.softmax(safe_scores, dim=-1),
            torch.zeros_like(safe_scores),
        )
        out_ctx = torch.einsum(
            "bhsk,bkm->bshm",
            probs,
            ctx_latent[..., : self.kv_lora_rank].float(),
        ).to(q_absorbed.dtype)
        return out_ctx, lse.transpose(1, 2).contiguous()

    def forward_with_target_kv(
        self,
        draft_hidden: torch.Tensor,
        draft_positions: torch.Tensor,
        target_kv_buffer: torch.Tensor,
        ctx_slots: torch.Tensor,
        ctx_lens: torch.Tensor | None = None,
        *,
        rope_cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None,
        linear_gate_fp8: tuple[torch.Tensor, torch.Tensor] | None = None,
        skip_post_norm: bool = False,
    ) -> torch.Tensor:
        """Serving forward reusing target KV buffer at ``ctx_slots [B, K]``."""
        bsz, block_size, _ = draft_hidden.shape
        positions_3d = _normalize_block_positions_3d(
            draft_positions, batch_size=bsz, block_size=block_size
        )
        if rope_cos_sin is None:
            flat_pos = positions_3d.reshape(3, bsz * block_size)
            cos, sin = compute_hy4_mrope_cos_sin(
                flat_pos, self.rope_inv_freq, self.mrope_section
            )
            rope_cos_sin = (
                cos.to(draft_hidden.dtype).contiguous(),
                sin.to(draft_hidden.dtype).contiguous(),
            )
        q_nope, q_pe, latent_draft = self._project_draft_q_and_latent(
            draft_hidden,
            positions_3d,
            rope_cos_sin=rope_cos_sin,
            apply_q_rope=False,
        )
        q_absorbed = self._absorb_q(q_nope, q_pe, rope_cos_sin=rope_cos_sin)

        use_fused_merge = q_absorbed.is_cuda and block_size <= 32
        out_ctx, lse_ctx = self._attend_context_absorbed_from_slots(
            q_absorbed,
            target_kv_buffer,
            ctx_slots,
            ctx_lens,
            skip_zero_ctx_mask=use_fused_merge and ctx_lens is not None,
        )
        if use_fused_merge:
            merged_out = hy4_dfly_block_attn_merge_sink(
                q_absorbed,
                latent_draft,
                out_ctx,
                lse_ctx,
                self.learnable_sink_param,
                scaling=self.scaling,
                causal=self.attn_causal,
                ctx_lens=ctx_lens,
            )
        else:
            out_draft, lse_draft = self._attend_draft_block_absorbed(
                q_absorbed, latent_draft
            )
            if out_draft.is_cuda:
                merged_out = hy4_merge_attention_and_sink(
                    out_ctx, lse_ctx, out_draft, lse_draft, self.learnable_sink_param
                )
            else:
                merged_out, merged_lse = _merge_two_attention_states(
                    out_ctx, lse_ctx, out_draft, lse_draft
                )
                if self.learnable_sink_param is not None:
                    merged_out = _apply_sink_to_latent(
                        merged_out, merged_lse, self.learnable_sink_param
                    )
        return self._project_output_tail(
            merged_out,
            draft_hidden,
            linear_gate_fp8=linear_gate_fp8,
            skip_post_norm=skip_post_norm,
        )


class HYV4FlashDFlyMLP(nn.Module):
    """Dense SwiGLU MLP for DFly2 layers (FP8 ``128x128`` in FP8 checkpoints)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_act = hidden_act
        tp_rank = mapping.dense.tp_rank
        tp_size = mapping.dense.tp_size
        tp_group = mapping.dense.tp_group
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            prefix=add_prefix("down_proj", prefix),
        )

        self.act_fn = gated_activation(hidden_act, swiglu_limit=None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        gate_up_fp8: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        orig_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, orig_shape[-1])
        if gate_up_fp8 is not None:
            gate_up = self.gate_up_proj(
                gate_up_fp8[0],
                block_scale=gate_up_fp8[1],
                output_dtype=flat.dtype,
            )[0]
        else:
            gate_up = _linear_cpu_or_cuda(self.gate_up_proj, flat)
        if flat.is_cuda:
            fused_down, down_mn = _o_proj_fp8_block_mode(self.down_proj)
            if (
                self.hidden_act == "silu"
                and fused_down
                and gate_up.dtype in (torch.bfloat16, torch.float16)
                and (gate_up.shape[-1] // 2) % 128 == 0
            ):
                q_fp8, q_scale = hy4_silu_and_mul_quantize_fp8(
                    gate_up, mn_major_tma=down_mn
                )
                out = self.down_proj(q_fp8, scale=q_scale)[0]
                return out.view(orig_shape)
            activated = self.act_fn(gate_up)
        else:
            gate, up = gate_up.chunk(2, dim=-1)
            activated = F.silu(gate) * up
        out = _linear_cpu_or_cuda(self.down_proj, activated)
        return out.view(orig_shape)


class HYV4FlashDFlyLayer(nn.Module):
    """Single DFly2 decoder layer (pre-RMSNorm dual-source MLA + Dense SwiGLU MLP)."""

    def __init__(
        self,
        config: HYV4FlashDFlyConfig,
        layer_id: int,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_id = int(layer_id)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = HYV4FlashDFlyAttention(
            config=config,
            layer_id=layer_id,
            mapping=mapping,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = HYV4FlashDFlyMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            mapping=mapping,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

    def _forward_with_residual(
        self,
        draft_hidden: torch.Tensor,
        residual: torch.Tensor | None,
        draft_positions: torch.Tensor,
        target_kv_buffer: torch.Tensor,
        ctx_slots: torch.Tensor,
        ctx_lens: torch.Tensor | None = None,
        *,
        rope_cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one DFly2 layer returning ``(mlp_out, residual)`` for cross-layer add+norm fusion."""
        linear_gate_fp8: tuple[torch.Tensor, torch.Tensor] | None = None
        if draft_hidden.is_cuda and draft_hidden.dtype in (
            torch.bfloat16,
            torch.float16,
        ):
            orig_shape = draft_hidden.shape
            dim = orig_shape[-1]
            fused_gate, gate_mn = (
                _o_proj_fp8_block_mode(self.self_attn.linear_gate)
                if self.self_attn.linear_gate is not None
                else (False, False)
            )
            if fused_gate:
                new_res, normed, gate_q, gate_s = hy4_add_rmsnorm_quantize_fp8(
                    draft_hidden,
                    self.input_layernorm.weight,
                    residual=residual,
                    eps=self.input_layernorm.variance_epsilon,
                    mn_major_tma=gate_mn,
                    write_bf16=True,
                )
                assert normed is not None
                res_after_in = draft_hidden if new_res is None else new_res
                linear_gate_fp8 = (gate_q, gate_s)
            elif residual is not None:
                normed_2d, res_2d = self.input_layernorm(
                    draft_hidden.reshape(-1, dim), residual.reshape(-1, dim)
                )
                normed = normed_2d.view(orig_shape)
                res_after_in = res_2d.view(orig_shape)
            else:
                res_after_in = draft_hidden
                normed = self.input_layernorm(draft_hidden.reshape(-1, dim)).view(
                    orig_shape
                )
        else:
            res_after_in = (
                draft_hidden if residual is None else (draft_hidden + residual)
            )
            normed = _apply_hy4_rmsnorm(res_after_in, self.input_layernorm)

        can_fuse_mlp_norm = (
            normed.is_cuda
            and normed.dtype in (torch.bfloat16, torch.float16)
            and _o_proj_fp8_block_mode(self.mlp.gate_up_proj)[0]
        )
        fuse_post_proj_norm = (
            can_fuse_mlp_norm and self.self_attn.post_proj_norm is not None
        )
        attn_out = self.self_attn.forward_with_target_kv(
            normed,
            draft_positions,
            target_kv_buffer,
            ctx_slots,
            ctx_lens=ctx_lens,
            rope_cos_sin=rope_cos_sin,
            linear_gate_fp8=linear_gate_fp8,
            skip_post_norm=fuse_post_proj_norm,
        )

        if attn_out.is_cuda and attn_out.dtype in (torch.bfloat16, torch.float16):
            orig_shape = attn_out.shape
            dim = orig_shape[-1]
            fused_mlp, mlp_mn = _o_proj_fp8_block_mode(self.mlp.gate_up_proj)
            if fused_mlp:
                post_norm = (
                    self.self_attn.post_proj_norm if fuse_post_proj_norm else None
                )
                hidden, _, mlp_q, mlp_s = hy4_add_rmsnorm_quantize_fp8(
                    attn_out,
                    self.post_attention_layernorm.weight,
                    residual=res_after_in,
                    eps=self.post_attention_layernorm.variance_epsilon,
                    pre_norm_weight=(
                        post_norm.weight if post_norm is not None else None
                    ),
                    pre_eps=(
                        post_norm.variance_epsilon if post_norm is not None else 1e-5
                    ),
                    mn_major_tma=mlp_mn,
                    write_bf16=False,
                )
                assert hidden is not None
                mlp_out = self.mlp(hidden, gate_up_fp8=(mlp_q, mlp_s))
                return mlp_out, hidden
            mlp_in_2d, hidden_2d = self.post_attention_layernorm(
                attn_out.reshape(-1, dim), res_after_in.reshape(-1, dim)
            )
            mlp_out = self.mlp(mlp_in_2d.view(orig_shape))
            return mlp_out, hidden_2d.view(orig_shape)

        hidden = res_after_in + attn_out
        mlp_out = self.mlp(_apply_hy4_rmsnorm(hidden, self.post_attention_layernorm))
        return mlp_out, hidden

    def forward_with_target_kv(
        self,
        draft_hidden: torch.Tensor,
        draft_positions: torch.Tensor,
        target_kv_buffer: torch.Tensor,
        ctx_slots: torch.Tensor,
        ctx_lens: torch.Tensor | None = None,
        *,
        rope_cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        mlp_out, residual = self._forward_with_residual(
            draft_hidden,
            None,
            draft_positions,
            target_kv_buffer,
            ctx_slots,
            ctx_lens=ctx_lens,
            rope_cos_sin=rope_cos_sin,
        )
        return residual + mlp_out


# ===----------------------------------------------------------------------=== #
# Top-Level DFly2 Backbone & CausalLM Entry Point
# ===----------------------------------------------------------------------=== #


class HYV4FlashDFlyModel(nn.Module):
    """5-layer Hy4-flash DFly2 draft backbone sharing target embeddings and KV cache."""

    def __init__(
        self,
        config: HYV4FlashDFlyConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.mapping = mapping
        self.hidden_size = int(config.hidden_size)
        self.mask_token_id = int(config.mask_token_id)

        # Checkpoint modules_to_not_convert keys live under "model.dfly_layers.*".
        dfly_prefix = (
            add_prefix("dfly_layers", prefix) if prefix else "model.dfly_layers"
        )

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.mtp_fusion: HYV4FlashDFlyMTPFusion | None = (
            HYV4FlashDFlyMTPFusion(
                hidden_size=config.hidden_size,
                mode=config.mtp_fusion,
                rms_norm_eps=config.rms_norm_eps,
                quant_config=quant_config,
                prefix=add_prefix("mtp_fusion", dfly_prefix),
            )
            if config.mtp_fusion is not None
            else None
        )
        self.layers = nn.ModuleList(
            [
                HYV4FlashDFlyLayer(
                    config=config,
                    layer_id=layer_id,
                    mapping=mapping,
                    quant_config=quant_config,
                    prefix=add_prefix(str(layer_id), dfly_prefix),
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_correction: HYV4FlashDFlyHiddenCorrection | None = (
            HYV4FlashDFlyHiddenCorrection(
                hidden_size=config.hidden_size,
                intermediate_size=config.hidden_correction_intermediate_size,
                rms_norm_eps=config.rms_norm_eps,
                quant_config=quant_config,
                prefix=add_prefix("hidden_correction", dfly_prefix),
            )
            if config.enable_hidden_correction
            else None
        )
        self.confidence_head: HYV4FlashDFlyConfidenceHead | None = (
            HYV4FlashDFlyConfidenceHead(config.hidden_size)
            if config.enable_confidence_head
            else None
        )

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def build_block_inputs(
        self,
        anchor_tokens: torch.Tensor,
        prev_hidden: torch.Tensor | None,
        block_size: int,
    ) -> torch.Tensor:
        """Build ``[B, block_size, D]`` noise embeddings with optional MTP fusion."""
        bsz = anchor_tokens.shape[0]
        block_ids = torch.full(
            (bsz, block_size),
            self.mask_token_id,
            dtype=torch.int64,
            device=anchor_tokens.device,
        )
        block_ids[:, 0] = anchor_tokens.to(torch.int64).clamp(
            0, self.config.vocab_size - 1
        )
        block_embeds = self.embed_tokens(block_ids.reshape(-1)).view(
            bsz, block_size, self.hidden_size
        )
        if self.mtp_fusion is not None and prev_hidden is not None:
            block_embeds = self.mtp_fusion.apply_to_block(block_embeds, prev_hidden)
        return block_embeds

    def forward_with_target_kv(
        self,
        anchor_tokens: torch.Tensor,
        prev_hidden: torch.Tensor | None,
        draft_positions: torch.Tensor,
        target_kv_pool: HybridHy4FlashTokenToKVPool,
        ctx_slots_per_layer: list[torch.Tensor],
        ctx_lens_per_layer: list[torch.Tensor | None] | None = None,
        *,
        block_size: int,
    ) -> torch.Tensor:
        """Run all 5 DFly2 layers against the target's shared DSA KV pool."""
        hidden = self.build_block_inputs(anchor_tokens, prev_hidden, block_size)
        rope_cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None
        if len(self.layers) > 0 and hidden.shape[0] > 0:
            bsz = hidden.shape[0]
            positions_3d = _normalize_block_positions_3d(
                draft_positions, batch_size=bsz, block_size=block_size
            )
            first_attn = self.layers[0].self_attn
            cos, sin = compute_hy4_mrope_cos_sin(
                positions_3d.reshape(3, bsz * block_size),
                first_attn.rope_inv_freq,
                first_attn.mrope_section,
            )
            rope_cos_sin = (
                cos.to(hidden.dtype).contiguous(),
                sin.to(hidden.dtype).contiguous(),
            )
        residual: torch.Tensor | None = None
        for idx, layer in enumerate(self.layers):
            target_kv_buf = target_kv_pool.get_key_buffer(
                layer.self_attn.target_kv_layer
            )
            ctx_lens = None if ctx_lens_per_layer is None else ctx_lens_per_layer[idx]
            hidden, residual = layer._forward_with_residual(
                draft_hidden=hidden,
                residual=residual,
                draft_positions=draft_positions,
                target_kv_buffer=target_kv_buf,
                ctx_slots=ctx_slots_per_layer[idx],
                ctx_lens=ctx_lens,
                rope_cos_sin=rope_cos_sin,
            )
        if residual is not None:
            if hidden.is_cuda:
                orig_shape = hidden.shape
                dim = orig_shape[-1]
                normed_2d, _ = self.final_norm(
                    hidden.reshape(-1, dim), residual.reshape(-1, dim)
                )
                return normed_2d.view(orig_shape)
            hidden = hidden + residual
        return _apply_hy4_rmsnorm(hidden, self.final_norm)

    def correct_hidden(
        self,
        slot_hidden: torch.Tensor,
        prev_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Apply ``hidden_correction`` for one chain step (pass-through if disabled)."""
        if self.hidden_correction is None:
            return slot_hidden
        safe_prev = prev_token_ids.to(torch.int64).clamp(0, self.config.vocab_size - 1)
        prev_embeds = self.embed_tokens(safe_prev)
        return self.hidden_correction(slot_hidden, prev_embeds)


def _canonicalize_dfly_weight_name(raw_name: str) -> str | None:
    """Map embedded ``model.dfly_layers.*`` or standalone DFly2 keys to runtime names.

    Returns ``None`` for non-DFly weights (target backbone, ViT, Engram, etc.)
    so loading from the shared Hy4-flash checkpoint skips them cleanly.
    """
    if raw_name.startswith(_DFLY_CHECKPOINT_PREFIX):
        suffix = raw_name[len(_DFLY_CHECKPOINT_PREFIX) :]
        # Layer weights in released checkpoint sit directly under model.dfly_layers.{i}.*
        head = suffix.split(".", 1)[0]
        if head.isdigit():
            return f"model.layers.{suffix}"
        return f"model.{suffix}"

    # Standalone AngelSpec DFly2 checkpoint keys: layers.{i}.*, mtp_fusion.*,
    # hidden_correction.*, confidence_head.*, final_norm.*
    if raw_name.startswith("layers."):
        return f"model.{raw_name}"
    if raw_name.startswith(
        ("mtp_fusion.", "hidden_correction.", "confidence_head.", "final_norm.")
    ):
        return f"model.{raw_name}"
    return None


def load_hy4_flash_dfly_weights(
    module: HYV4FlashForCausalLMDFly,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> set[str]:
    """Load DFly2 draft weights (BF16 or block-FP8 with BF16 ``scale_inv``)."""
    params_dict: dict[str, nn.Parameter] = dict(module.named_parameters())
    stacked_params_mapping = (
        (".mlp.gate_up_proj.", ".mlp.gate_proj.", 0),
        (".mlp.gate_up_proj.", ".mlp.up_proj.", 1),
    )
    loaded_params: set[str] = set()

    for raw_name, loaded_weight in weights:
        name = _canonicalize_dfly_weight_name(raw_name)
        if name is None:
            continue

        # Released FP8 checkpoints store block weight_scale_inv in BF16; cast to FP32.
        if name.endswith("_scale_inv") and loaded_weight.dtype != torch.float32:
            loaded_weight = loaded_weight.to(torch.float32)

        for param_key, ckpt_key, shard_id in stacked_params_mapping:
            if ckpt_key in name:
                target_name = name.replace(ckpt_key, param_key)
                if target_name not in params_dict:
                    raise KeyError(
                        f"DFly2 stacked parameter {target_name!r} not found (from {raw_name!r})"
                    )
                param = params_dict[target_name]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(target_name)
                break
        else:
            if name not in params_dict:
                raise KeyError(
                    f"Unexpected DFly2 weight {raw_name!r} (mapped to {name!r})"
                )
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

    return loaded_params


class HYV4FlashForCausalLMDFly(BaseCausalLM):
    """Hy4-flash DFly2 speculative draft model entry point."""

    model_cls = HYV4FlashDFlyModel
    logits_processor_cls = HYV4FlashLogitsProcessor

    def __init__(
        self,
        config: HYV4FlashDFlyConfig | HYV4FlashConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        encoder_only: bool = False,
    ) -> None:
        dfly_config = (
            HYV4FlashDFlyConfig.from_target(config)
            if isinstance(config, HYV4FlashConfig)
            else config
        )
        super().__init__(
            dfly_config,
            mapping=mapping,
            quant_config=quant_config,
            prefix=prefix,
            encoder_only=encoder_only,
        )

    def resolve_lm_head(
        self,
        config: HYV4FlashDFlyConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> nn.Module:
        return build_hy4_lm_head(
            config=config,
            mapping=self.mapping,
            quant_config=quant_config,
            prefix=prefix,
        )

    def set_embed_and_head_module(
        self,
        embed_weight: torch.Tensor | None,
        lm_head: nn.Module,
    ) -> None:
        """Bind target's shared ``embed_tokens`` weight and FP32 ``lm_head`` module."""
        if embed_weight is not None and self.model is not None:
            self.model.embed_tokens.weight = embed_weight
        self.lm_head = lm_head

    def get_embed_and_head(self):
        embed = self.model.embed_tokens.weight if self.model is not None else None
        head = self.lm_head.weight if self.lm_head is not None else None
        return embed, head

    @staticmethod
    def checkpoint_weight_name_filter(name: str) -> bool:
        """Read only DFly2 shards when loading from an embedded Hy4-flash checkpoint."""
        return _canonicalize_dfly_weight_name(name) is not None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded = load_hy4_flash_dfly_weights(self, weights)
        self.post_load_weights()
        return loaded

    def post_load_weights(self) -> None:
        if self.model is None:
            return
        _enable_hopper_deepgemm(self)
        for layer in self.model.layers:
            layer.self_attn.prepare_absorbed_mla_weights()
            layer.self_attn.prepare_packed_bf16_projections()
        if self.model.hidden_correction is not None:
            self.model.hidden_correction.prepare_packed_weights()


EntryClass = [HYV4FlashForCausalLMDFly]

__all__ = [
    "EntryClass",
    "HYV4FlashDFlyAttention",
    "HYV4FlashDFlyConfidenceHead",
    "HYV4FlashDFlyHiddenCorrection",
    "HYV4FlashDFlyLayer",
    "HYV4FlashDFlyMLP",
    "HYV4FlashDFlyMTPFusion",
    "HYV4FlashDFlyModel",
    "HYV4FlashForCausalLMDFly",
    "load_hy4_flash_dfly_weights",
]
