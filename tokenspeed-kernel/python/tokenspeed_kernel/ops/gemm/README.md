# GEMM operations

## Weight-only MXFP8 dense linears

`prepare_mxfp8_linear` selects a registered `gemm.mxfp8_w8a16` preparation
factory using raw E4M3 `[N, K]` weights, raw E8M0 `[N, K/32]` scale bytes,
activation dtype, and the explicit reduction policies. It returns an
`Mxfp8LinearPlan`; runtime code does not construct a vendor implementation.

The current factory is `marlin_mxfp8_linear` in `marlin.py`. It owns repacking
and native-module warmup. Its CUDA adapter and pinned upstream implementation
live in `thirdparty/cuda/`. The packed representation remains backend-owned;
the selection signature describes the **raw preparation inputs**, not the
packed INT32 weights used internally.

`solution` restricts normal selection. `override` uses the standard precedence:
`TOKENSPEED_KERNEL_OVERRIDE_GEMM_MXFP8_W8A16`, scoped `kernel_override`, then
call-site override. An unknown override or unsupported solution raises; no
BF16 dense-GEMM fallback is allowed. Selection is pinned at preparation because
another backend may need a different packed layout; changing it requires a new
plan/model load. Eager and CUDA-graph execution use the same prepared apply
operation and expose its selected name in kernel profiling.

The Marlin factory accepts BF16 activations, E4M3 weights, and E8M0 group-32
scales, with explicit `use_atomic_add` and `use_fp32_reduce`. It rejects
unsupported devices/dtypes even under a forced override. Code capability is
limited to SM90, SM100, SM103, and SM107; hardware validation currently covers
H20/SM90 only. Device kernels retain their pinned upstream source bytes.

Checkpoint-excluded projections retain their original precision. HY V4's
indexer separately pairs and dequantizes its MXFP8 `wk` during loading, then
uses the reference BF16 `wk_weights_proj` fusion. Ordinary dense projections
keep their Marlin plan. Absorbed MLA also materializes K/V weights for its
tensor contractions; neither exception is a fallback for ordinary dense
projections. See the HY V4 MXFP8 recipe for model-specific loading details.
