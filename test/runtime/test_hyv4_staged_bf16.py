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

"""CPU contracts of HY V4's opt-in BF16 implementation and graph padding."""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from tokenspeed.runtime.configs.hy_v4_config import HYV4Config
from tokenspeed.runtime.configs.model_config import _resolve_attention_family
from tokenspeed.runtime.layers.attention.backends.paged.dsa import DSABackend
from tokenspeed.runtime.layers.attention.configs.dsa import DSAConfig
from tokenspeed.runtime.layers.moe.topk import TopKOutputFormat
from tokenspeed.runtime.models import hy_v4
from tokenspeed.runtime.utils.server_args import ServerArgs


class HYV4StagedLaunchTests(unittest.TestCase):
    def test_mode_resolves_target_and_draft_without_changing_auto(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            original = ServerArgs(model="x")
            staged = ServerArgs(
                model="x", hyv4_numerics="staged-bf16", draft_moe_backend="auto"
            )
        self.assertEqual(original.hyv4_numerics, "auto")
        self.assertEqual(original.moe_backend, "auto")
        self.assertEqual(original.layer_boundary_norm, "fused")
        self.assertEqual(staged.numerics, "auto")
        self.assertEqual(staged.moe_backend, "triton_staged")
        self.assertEqual(staged.draft_moe_backend, "triton_staged")
        self.assertIsNone(staged.ep_dispatch_algorithm)
        self.assertEqual(staged.layer_boundary_norm, "unfused")
        self.assertFalse(staged.enable_allreduce_fusion)
        self.assertTrue(staged.disable_tf32)

    def test_conflicting_launch_contracts_are_rejected(self):
        for changed in (
            {"numerics": "rl-bitwise"},
            {"dtype": "float16"},
            {"quantization": "fp8"},
            {"kv_cache_dtype": "fp8_e4m3"},
            {"kv_cache_quant_method": "per_token_head"},
            {"moe_backend": "triton"},
            {"draft_moe_backend": "flashinfer_cutlass"},
            {"all2all_backend": "deepep"},
            {"moe_combine_order": "slot"},
            {"init_expert_location": "placement.json"},
            {"enable_eplb": True},
            {"ep_dispatch_algorithm": "static"},
            {"ep_dispatch_algorithm": "dynamic"},
        ):
            with (
                self.subTest(changed=changed),
                mock.patch.dict(os.environ, {}, clear=True),
            ):
                with self.assertRaisesRegex(ValueError, "staged-bf16"):
                    ServerArgs(model="x", hyv4_numerics="staged-bf16", **changed)
        for key in ("NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"):
            with (
                self.subTest(key=key),
                mock.patch.dict(os.environ, {key: "1"}, clear=True),
            ):
                with self.assertRaisesRegex(ValueError, key):
                    ServerArgs(model="x", hyv4_numerics="staged-bf16")

    def test_worker_checks_resolved_dtype_quantization_and_router_domain(self):
        options = {
            "hyv4_numerics": "staged-bf16",
            "all2all_backend": "none",
            "moe_combine_order": "rank",
            "init_expert_location": "trivial",
            "ep_num_redundant_experts": 0,
            "enable_eplb": False,
            "ep_dispatch_algorithm": None,
            "layer_boundary_norm": "unfused",
            "enable_allreduce_fusion": False,
        }
        config = SimpleNamespace(
            n_routed_experts=512,
            num_experts_per_tok=22,
            n_group=1,
            topk_group=1,
            hidden_size=16,
            q_lora_rank=8,
            kv_lora_rank=8,
        )
        mapping = SimpleNamespace(moe=SimpleNamespace(tp_size=1, ep_size=1))
        with (
            mock.patch.dict(hy_v4.global_server_args_dict, options),
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(
                hy_v4,
                "current_platform",
                return_value=SimpleNamespace(
                    is_nvidia=True, arch_version=hy_v4.ArchVersion(9, 0)
                ),
            ),
            mock.patch.object(
                hy_v4,
                "get_moe_backend",
                return_value=SimpleNamespace(value="triton_staged"),
            ),
            mock.patch.object(torch, "get_default_dtype", return_value=torch.bfloat16),
            mock.patch.object(torch, "set_float32_matmul_precision"),
            mock.patch.object(
                torch, "get_float32_matmul_precision", return_value="highest"
            ),
            mock.patch.object(
                torch.backends.cuda, "matmul", SimpleNamespace(allow_tf32=False)
            ),
        ):
            hy_v4.configure_hyv4_numerics(config, mapping, None)
            for experts, topk in ((512, 23), (768, 17), (1025, 8)):
                config.n_routed_experts, config.num_experts_per_tok = experts, topk
                with self.subTest(experts=experts, topk=topk):
                    with self.assertRaisesRegex(ValueError, "single-group"):
                        hy_v4.configure_hyv4_numerics(config, mapping, None)
            config.n_routed_experts, config.num_experts_per_tok = 768, 16
            hy_v4.configure_hyv4_numerics(config, mapping, None)
            with self.assertRaisesRegex(ValueError, "quantized"):
                hy_v4.configure_hyv4_numerics(config, mapping, object())
            with mock.patch.object(
                torch, "get_default_dtype", return_value=torch.float16
            ):
                with self.assertRaisesRegex(ValueError, "BF16 model worker"):
                    hy_v4.configure_hyv4_numerics(config, mapping, None)

    def test_ambient_kernel_override_cannot_replace_the_named_provider(self):
        key = "TOKENSPEED_KERNEL_OVERRIDE_MOE_SIGMOID_BIAS_TOPK"
        with mock.patch.dict(os.environ, {key: "torch_sigmoid_bias_topk"}):
            with self.assertRaisesRegex(ValueError, "effective kernel override"):
                hy_v4._require_hyv4_kernel(
                    "moe", "sigmoid_bias_topk", hy_v4._HYV4_TOPK_KERNEL
                )


class HYV4StagedBoundaryTests(unittest.TestCase):
    def test_qkv_norm_preserves_rope_and_honors_strided_outputs(self):
        class Norm(torch.nn.Module):
            def forward(self, x, *, out):
                out.copy_(x + 1)
                return out

        adapter = hy_v4.HYV4StagedQKVNorm(Norm(), Norm())
        qkv = torch.arange(3 * 24, dtype=torch.float32).reshape(3, 24).bfloat16()
        original = qkv.clone()
        q, latent = qkv.split((8, 16), dim=-1)
        kv, rope = latent.split((8, 8), dim=-1)
        q_out = torch.empty_like(q)
        actual_q, actual_kv = adapter(q, kv, output_q_a=q_out)
        self.assertIs(actual_q, q_out)
        self.assertIs(actual_kv, kv)
        torch.testing.assert_close(q, original[:, :8], rtol=0, atol=0)
        torch.testing.assert_close(q_out, original[:, :8] + 1, rtol=0, atol=0)
        torch.testing.assert_close(kv, original[:, 8:16] + 1, rtol=0, atol=0)
        torch.testing.assert_close(rope, original[:, 16:], rtol=0, atol=0)
        separate = torch.empty_like(kv)
        adapter(q, kv, output_q_a=q_out, output_kv_a=separate)
        torch.testing.assert_close(kv, original[:, 8:16] + 1, rtol=0, atol=0)
        torch.testing.assert_close(separate, original[:, 8:16] + 2, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "separate all-gather"):
            adapter.forward_with_allgather_fusion()

    def test_ihc_materializes_before_the_same_norm_with_and_without_pending_post(self):
        layer = object.__new__(hy_v4.HYV4DecoderLayer)
        torch.nn.Module.__init__(layer)
        layer.staged_bf16 = True
        layer.config = SimpleNamespace(hc_magnitude=2.0, rms_norm_eps=1e-5, hc_eps=1e-6)
        boundary = SimpleNamespace(hc_fn=object(), hc_scale=object(), hc_base=object())
        hidden = torch.ones(2, 4, 8, dtype=torch.bfloat16)
        reduced = torch.full((2, 8), 3, dtype=torch.bfloat16)
        updated = hidden + 1
        gates = torch.ones(2, 4)
        pending = hy_v4.IhcPost(reduced, gates)
        norm = mock.Mock(
            weight=object(), variance_epsilon=1e-5, return_value=reduced + 1
        )
        with (
            mock.patch.object(
                hy_v4.residual_ops, "ihc_pre", return_value=(reduced, gates)
            ) as pre,
            mock.patch.object(
                hy_v4.residual_ops,
                "ihc_fused_hc",
                return_value=(updated, reduced, gates),
            ) as fused,
        ):
            for previous, residual in ((None, hidden), (pending, updated)):
                result = layer._ihc_pre(hidden, previous, boundary, norm)
                self.assertIs(result[0], residual)
                self.assertIs(result[2], gates)
                norm.assert_called_with(reduced)
                call = pre.call_args if previous is None else fused.call_args
                self.assertIsNone(call.kwargs["norm_weight"])
                self.assertIsNone(call.kwargs["norm_eps"])
                self.assertEqual(call.kwargs["solution"], "torch")
        self.assertEqual(norm.call_count, 2)
        with mock.patch.dict(
            os.environ, {"TOKENSPEED_KERNEL_OVERRIDE_RESIDUAL_IHC_PRE": "other"}
        ):
            with self.assertRaisesRegex(ValueError, "effective kernel override"):
                layer._ihc_pre(hidden, None, boundary, norm)

    def test_norm_factory_is_scoped_to_hy_and_selects_at_construction(self):
        for mode, expected in (
            ("auto", hy_v4.RMSNorm),
            ("staged-bf16", hy_v4.HYV4StagedRMSNorm),
        ):
            with mock.patch.dict(
                hy_v4.global_server_args_dict, {"hyv4_numerics": mode}
            ):
                norm = hy_v4.hyv4_rmsnorm(8, eps=1e-5)
            self.assertIs(type(norm), expected)

    def test_staged_norm_materializes_residual_and_passes_the_output(self):
        norm = hy_v4.HYV4StagedRMSNorm(8)
        x = torch.full((2, 8), 1.0, dtype=torch.bfloat16)
        residual = torch.full_like(x, 0.00390625)
        out = torch.empty_like(x)

        def normalize(value, weight, eps, *, out):
            torch.testing.assert_close(value, x + residual, rtol=0, atol=0)
            out.copy_(value)
            return out

        with mock.patch(
            "tokenspeed_kernel.ops.layernorm.cuda_staged.rmsnorm", side_effect=normalize
        ):
            result, updated = norm(x, residual, out=out)
        self.assertIs(result, out)
        torch.testing.assert_close(updated, x + residual, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "fused all-reduce"):
            norm.forward_with_allreduce_fusion()


class HYV4StagedRoutingTests(unittest.TestCase):
    def test_padding_keeps_the_same_provider_and_masks_before_observation(self):
        router = hy_v4.HYV4StagedTopK(
            top_k=2,
            use_grouped_topk=True,
            num_expert_group=1,
            topk_group=1,
            correction_bias=torch.zeros(4),
            routed_scaling_factor=2.827,
            output_format=TopKOutputFormat.STANDARD,
        )
        hidden = torch.zeros(3, 8, dtype=torch.bfloat16)
        logits = torch.zeros(3, 4)
        weights = torch.arange(6, dtype=torch.float32).reshape(3, 2)
        ids = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.int32)
        observed = []
        with (
            mock.patch.object(
                hy_v4,
                "moe_topk",
                side_effect=lambda *a, **kw: (weights.clone(), ids.clone()),
            ) as select,
            mock.patch.object(
                hy_v4,
                "record_expert_load",
                side_effect=lambda load, routed: observed.append(routed.clone()),
            ),
        ):
            eager = router(hidden, logits)
            padded = router(hidden, logits, num_token_non_padded=torch.tensor(1))
        self.assertEqual(select.call_args_list[0], select.call_args_list[1])
        self.assertEqual(
            select.call_args.kwargs["override"], "cuda_public_v029_sigmoid_bias_topk"
        )
        self.assertIs(padded.router_logits, logits)
        torch.testing.assert_close(eager.topk_ids, ids, rtol=0, atol=0)
        torch.testing.assert_close(padded.topk_weights, weights, rtol=0, atol=0)
        self.assertTrue(bool((padded.topk_ids[1:] == -1).all()))
        torch.testing.assert_close(observed[1], padded.topk_ids, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "expert placement"):
            router(hidden, logits, expert_location_dispatch_info=object())
        with self.assertRaisesRegex(ValueError, "STANDARD"):
            router(hidden, logits, output_format=TopKOutputFormat.BYPASSED)


class HYV4PublicAttentionConfigTests(unittest.TestCase):
    def test_only_hy_target_and_nextn_hooks_select_the_public_core(self):
        for architecture, family_name, is_draft in (
            ("HYV4ForCausalLM", "HY V4", False),
            ("HYV4ForCausalLMNextN", "HY V4", True),
            ("GlmMoeDsaForCausalLM", "GLM", False),
        ):
            for mode in ("auto", "staged-bf16"):
                with self.subTest(architecture=architecture, mode=mode):
                    with mock.patch.dict(os.environ, {}, clear=True):
                        args = ServerArgs(model="x", hyv4_numerics=mode)
                    hf = HYV4Config(architectures=[architecture])
                    config = SimpleNamespace(
                        hf_config=hf,
                        hf_text_config=hf,
                        num_attention_layers=(
                            hf.num_nextn_predict_layers
                            if is_draft
                            else hf.num_hidden_layers
                        ),
                        num_attention_heads=hf.num_attention_heads,
                        num_key_value_heads=hf.num_key_value_heads,
                    )
                    family = _resolve_attention_family(hf, hf)
                    self.assertIsNotNone(family)
                    self.assertEqual(family.name, family_name)
                    family.configure(config, args)
                    expected = (
                        "flashmla_public"
                        if family_name == "HY V4" and mode == "staged-bf16"
                        else None
                    )
                    self.assertEqual(config.dsa_core_kernel_solution, expected)
                    spec = DSAConfig(**DSAConfig._spec_kwargs(args, config, is_draft))
                    self.assertEqual(spec.core_kernel_solution, expected)

    def test_resolved_cache_and_device_are_rejected_before_backend_allocation(self):
        config = SimpleNamespace(
            dtype=torch.bfloat16,
            kv_cache_dtype=torch.bfloat16,
            kv_cache_quant_method="none",
            kv_cache_mxfp8=False,
        )
        spec = SimpleNamespace(core_kernel_solution="flashmla_public")
        from tokenspeed_kernel.platform import ArchVersion

        for changes, arch in (
            ({"kv_cache_dtype": torch.float8_e4m3fn}, ArchVersion(9, 0)),
            ({"kv_cache_quant_method": "per_token_head"}, ArchVersion(9, 0)),
            ({}, ArchVersion(10, 0)),
        ):
            changed = SimpleNamespace(**(vars(config) | changes))
            with (
                self.subTest(changes=changes, arch=arch),
                mock.patch(
                    "tokenspeed.runtime.layers.attention.backends.paged.dsa.current_platform",
                    return_value=SimpleNamespace(is_nvidia=True, arch_version=arch),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "SM90.*BF16"):
                    DSABackend(changed, spec, kernel_page_size=64)

    def test_public_core_rejects_late_override_without_changing_indexer(self):
        backend = object.__new__(DSABackend)
        backend.kernel_solution = None
        backend.core_kernel_solution = "flashmla_public"
        for mode in ("dsa_prefill", "dsa_decode"):
            with self.subTest(mode=mode):
                self.assertEqual(
                    backend._core_kernel_kwargs(mode), {"solution": "flashmla_public"}
                )
                self.assertIsNone(backend.kernel_solution)
                key = f"TOKENSPEED_KERNEL_OVERRIDE_ATTENTION_{mode.upper()}"
                with mock.patch.dict(os.environ, {key: "triton_dsa_decode"}):
                    with self.assertRaisesRegex(
                        ValueError, "effective kernel override"
                    ):
                        backend._core_kernel_kwargs(mode)


if __name__ == "__main__":
    unittest.main()
