# HY V4 MXFP8 dense Marlin

ModelOpt MXFP8 dense projections retain the checkpoint E4M3 weight and E8M0
scales (one per 32 input channels), then run `MarlinMxfp8LinearKernel` with
BF16 activations, FP32 split-K reduction and atomic reduction disabled. No
vLLM installation is required. The CUDA arithmetic and launch selection are
vendored from public vLLM v0.29.0 commit
`98dff2a81d747d1dba01a47f939f48c3526d4206`; provenance and Apache-2.0 license
are shipped alongside the optional JIT source.

## Loading and execution

The loader keeps both weight and scale parameters through TP sharding and
fused Q/KV or gate/up loading. Preparation repacks to Marlin's int32 layout
and permutes E8M0 bytes without converting dense weights to BF16. The native
module is warmed during post-load preparation, before CUDA graph capture.
Matrices that miss a thread tile use zero-padded weight/activation dimensions
and output slicing. Each invocation owns its locks and temporary reduction
buffer, including under concurrent streams and graph capture.

Checkpoint-excluded modules (including HY V4's BF16 attention `linear_gate`)
remain unquantized. The indexer is an explicit exception to ordinary dense
MXFP8 execution: its `wk` weight and E8M0 scale are paired and dequantized once
at loading, then combined with BF16 `weights_proj` in `wk_weights_proj`.
Only that fused projection is allocated and executed; no unused standalone
`wk` Marlin weights are prepared. This preserves the reference model's
indexer computation. Ordinary QKV, output and MLP projections remain Marlin.

The shared GLM loader must handle checkpoint shards before testing whether a
standalone parameter exists, and report the fused parameter as loaded only
after both projection shards arrive. The MLA absorbed K/V contractions
separately materialize their BF16 tensor weights; that is not a fallback for
the quantized `kv_b_proj` linear forward. Dense preparation does not select
the routed MoE backend.

For the routed MXFP8 GEMV fast path, GEMM2's reduced value and its routing
weight round to BF16 separately, their product rounds to BF16, and the top-k
sum accumulates those products in FP32. This matches the in-tree native
Marlin epilogue; a single FP32 product followed by one cast is not the same
numerical contract. It does not guarantee bitwise agreement of the upstream
GEMM accumulations or of the large-batch CUTLASS branch.

## Supported contract

The initial adapter supports NVIDIA SM90/SM100/SM103/SM107, BF16 activations,
E4M3 weights and uint8/E8M0 group-32 scales. Every TP input partition must be
a multiple of 32. Unsupported dtypes/devices, missing or duplicate scales,
nonfinite weights, and scale encodings 0/255 are rejected explicitly. The
public Marlin device conversion assumes E8M0 encoding 0 does not occur; do
not silently accept that underflow. BF16 refits into a prepared MXFP8 model
are rejected and require a fresh model load. An explicit override selects a
registered `gemm.mxfp8_w8a16` preparation factory, not a BF16 `gemm.mm` kernel;
unknown overrides fail rather than silently changing the numerical path.
Fused gate/up interleaving is not accepted for this method.

## Validation

Run `test/runtime/test_modelopt_mxfp8.py` and
`tokenspeed-kernel/test/nvidia/ops/gemm/test_marlin_mxfp8.py` with the normal
runtime/kernel environment. Tests cover original byte preservation, TP
weight/scale slicing, fused loading, padded dimensions, zero/short/646/704-row
inputs, bias, repeated execution, CUDA graph replay and absorbed-weight
materialization. Numerical comparisons use an FP64 product reference with
explicit BF16 ULP and relative-L2 bounds. A passing kernel test is not a
full-model quality or speed acceptance result: rerun the same-checkpoint
paired model regression before claiming accuracy parity.

Host-only regressions in `test/runtime/test_glm_indexer_fused_loading.py`
exercise the shared loader, missing standalone parameters, FP8 pair arrival
across calls, and incomplete updates. They execute production Python loader
code with real CPU tensors and replace only the model/GPU boundaries; they
are not a substitute for GPU or full-model validation.

`test/runtime/test_mxfp8_moe_host_contract.py` checks host-only GEMV/Marlin/
CUTLASS dispatch boundaries and the BF16 routing arithmetic. The GPU suite's
`test_gemv_weighted_finalize_matches_native_bf16_epilogue` uses exact equality,
multiple split counts, and NaN-filled invalid/nonlocal routes. Run it on
supported NVIDIA hardware before accepting a GEMV epilogue change.

For a CPU-only check of the residual/gate integration, run from the repository
root with a Python environment containing PyTorch and pytest:

```sh
python -m pytest -q test/runtime/test_hyv4_residual_gate_cpu.py
```

This executes production source nodes for iHC parameter construction/checkpoint
loading, Q/KV norm epsilon wiring, headwise/elementwise gating, and fused
post-to-pre boundaries with BF16 and FP32 residuals. It checks that iHC parameters
retain FP32 checkpoint precision and that the attention sigmoid rounds in the
gate dtype before multiplication. The attention base constructor, linear backend
and unrelated MoE loader are substituted; GPU kernel dispatch, quantized linear
arithmetic, CUDA graphs and full-model quality remain separate validation steps.

`test/runtime/test_dsa_indexer_cpu_boundaries.py` covers immutable workspace
index-to-slot conversion, prefix/chunk causal metadata through 128K logical
positions, and long-input fused projection row preservation. Its CPU tensor
checks do not establish GPU TopK/quantization or scheduler cache-hit parity.
