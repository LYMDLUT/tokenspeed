# Public vLLM v0.29 single-group sigmoid TopK

This independent candidate extracts the single-group implementation from the
public vLLM v0.29.0 commit
[`98dff2a81d747d1dba01a47f939f48c3526d4206`](https://github.com/vllm-project/vllm/commit/98dff2a81d747d1dba01a47f939f48c3526d4206).
It does not replace the separate public v0.12 candidate or establish the source
revision or build identity of another installed binary.

## Source and license

- [`grouped_topk_kernels.cu`](https://github.com/vllm-project/vllm/blob/98dff2a81d747d1dba01a47f939f48c3526d4206/csrc/libtorch_stable/moe/grouped_topk_kernels.cu): retain the full license header, the cast/finite/scoring helpers at lines 376-431, and the complete `single_group_topk` namespace at lines 851-1277.
- [`moeTopKFuncs.cuh`](https://github.com/vllm-project/vllm/blob/98dff2a81d747d1dba01a47f939f48c3526d4206/csrc/libtorch_stable/moe/moeTopKFuncs.cuh): copied verbatim as `public_v029_moeTopKFuncs.cuh`, including its vLLM/NVIDIA Apache-2.0 notice and TensorRT-LLM/FlashInfer source links.
- The repository Apache-2.0 license is copied verbatim as `public_v029_topk.LICENSE`.
- `public_v029_topk.SOURCE.json` records exact public URLs, commit, source hashes, extraction ranges, build policy, and adaptations.

Device functions, tier order, and warp/block dispatch are unchanged. Local host
adaptations replace Torch stable with TVM FFI, validate input/output metadata,
check both CUDA launch return values and dispatch success, and return early for
empty batches. The public `N <= 16` PDL policy is retained. The header requires
only CUDA cooperative groups, CUB, and standard headers; Torch stable headers
are not needed by this isolated entry.

## Supported contract

`tokenspeed_kernel.thirdparty.cuda.public_v029_topk.single_group_sigmoid_topk`
accepts colocated contiguous CUDA FP32 logits `[N, E]` and bias `[E]` for one
expert group. Finite logits and bias are the value contract; the hot path does
not scan GPU values. The explicit scalar arguments are `topk`, `renormalize`,
and `routed_scaling_factor`.

The first matching `(maximum experts, maximum topk)` tier is selected:

```text
(128,8), (256,8), (384,8), (512,8), (512,22),
(768,16), (896,16), (1024,16)
```

Thus `1 <= E <= 512` permits `1 <= topk <= min(E,22)`, and
`512 < E <= 1024` permits `1 <= topk <= 16`. Other shapes are explicitly
rejected instead of falling back to another implementation. `N` is limited to
`INT32_MAX-32768` to protect the public warp loop's final INT32 increment.
Outputs are contiguous FP32 weights and INT32 ids with shape `[N, topk]`.

The public sigmoid remains `0.5f*tanhf(0.5f*x)+0.5f`; biased-score ties prefer
lower expert ids. The warp path recovers the weight numerator by subtracting
bias from the selected FP32 biased score. The block path retains the unbiased
sigmoid separately. Consequently, a large bias can lose low score bits in the
warp path. This behavior is retained and should not be described as ideal
mathematical sigmoid routing.

Both paths round the scale to FP32, divide it by `sum + 1e-20F` when normalizing,
and then multiply the numerator by that FP32 scale. Unnormalized weights also
apply the FP32 scale. The wrapper accepts scales within the finite FP32 range;
it does not require exact FP32 representability. The public block/warp choice
is retained, including the token-count condition and the tuned 896/1024 tiers.

The registered provider
`tokenspeed_kernel.ops.moe.cuda_public_v029_topk.cuda_public_v029_sigmoid_bias_topk`
is NVIDIA/FP32-only, uses the existing `moe/sigmoid_bias_topk` family with
solution `cuda`, and preserves an explicitly requested floating output dtype.
Its priority is below the existing optimized default. Select it explicitly by
its registered name for candidate evaluation.

## Build and validation

The lazy FlashInfer JIT module uses C++20/CUDA20, `use_fast_math=False`,
`--ftz=false`, `--prec-div=true`, `--prec-sqrt=true`, and `--fmad=true`.
The public v0.29 CMake target inherits common CUDA flags and does not attach
fast-math flags to grouped TopK. These explicit settings define this candidate;
they do not prove equality with every public wheel or external build.

The cache key includes the public commit, CUDA source, helper header, shared
TVM FFI header, and explicit compiler flags. The first nonempty call must be
warmed before CUDA graph capture; a cold call during capture is rejected
before compiler import. Empty batches validate metadata but do not load or
compile the optional module. No installed AOT library is rebuilt.

The dedicated NVIDIA tests cover tier boundaries, both dispatch paths, ties,
bias subtraction, FP32 scaling order, output dtype, validation, empty batches,
and graph capture/replay. Compilation, tests, and application acceptance still
require execution on supported NVIDIA hardware.
