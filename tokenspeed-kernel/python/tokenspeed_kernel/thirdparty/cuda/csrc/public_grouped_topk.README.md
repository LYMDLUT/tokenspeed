# Public grouped sigmoid TopK

`public_grouped_topk.cu` vendors the self-contained CUDA algorithm from
[vLLM v0.12.0](https://github.com/vllm-project/vllm/releases/tag/v0.12.0), pinned to
commit `4fd9d6a85c00ac0186aa9abbeff73fc2ac6c721e`:

- [Original source](https://github.com/vllm-project/vllm/blob/4fd9d6a85c00ac0186aa9abbeff73fc2ac6c721e/csrc/moe/grouped_topk_kernels.cu)
- [Original Apache-2.0 license](https://github.com/vllm-project/vllm/blob/4fd9d6a85c00ac0186aa9abbeff73fc2ac6c721e/LICENSE), copied verbatim as `public_grouped_topk.LICENSE`
- Original source SHA-256: `35e13affcf3fee889d8bb6a1a1dabfd7f4d0de02424e59aa24601d1adf5b874f`
- License SHA-256: `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`

The original header attributes the implementation to vLLM and NVIDIA and
identifies TensorRT-LLM v0.21.0 as its source. That header remains intact.
The public device functions and arithmetic are unchanged. Local adaptations
replace the PyTorch/c10 includes and host binding with TVM FFI, validate tensor
metadata and integer bounds, and check both CUDA launch return values. Only the
FP32 sigmoid entry is exposed. The public templates remain in the source.

The optional Python loader builds this source separately from the installed
AOT library. Its module key includes the pinned commit, source/header contents,
and explicit CUDA arithmetic flags. It requires FlashInfer `gen_jit_spec` with
`use_fast_math=False` support and passes `--ftz=false`, `--prec-div=true`,
`--prec-sqrt=true`, and `--fmad=false`. These flags define this candidate's build
contract; they do not establish bitwise equality with an unknown external build.

## API and arithmetic

`tokenspeed_kernel.thirdparty.cuda.public_grouped_topk.grouped_sigmoid_topk`
accepts contiguous CUDA FP32 logits `[N, E]` and bias `[E]` on the same device.
All behavior choices are explicit keyword arguments. `G=num_expert_group` must
be in `[1, 32]`, divide `E`, and leave at least two experts per group.
`1 <= topk_group <= G` and
`1 <= topk <= min(32, topk_group * E/G)` are required. The adapter also bounds
`E <= INT32_MAX-31` and `N*G <= INT32_MAX` for the public indexing arithmetic.
Inputs and resulting group scores must be finite; no GPU value scan is added.
The scale must be finite. Empty batches return empty outputs without building.

The public kernel computes sigmoid as `0.5f*tanhf(0.5f*x)+0.5f`, selects groups
by the sum of their top two biased scores, and selects experts by biased score.
Ties favor lower group/expert indices. Output weights use the selected unbiased
sigmoid scores, with optional FP32 sum normalization starting at `1e-20`,
followed by multiplication by the **double** scale and storage as FP32. Both
normalized and unnormalized modes apply the scale. Output ids are INT32.

`tokenspeed_kernel.ops.moe.cuda_public_topk.cuda_public_sigmoid_bias_topk`
registers the ungrouped case under `moe/sigmoid_bias_topk`, with solution `cuda`
and explicit name `cuda_public_sigmoid_bias_topk`. It passes `G=topk_group=1`
and requires the caller's requested floating weight dtype explicitly. Its priority is
below the existing optimized implementation. This new module is not imported
by existing registration files until the caller explicitly integrates it.

Warm the first nonempty call before CUDA graph capture; a cold call during
capture is rejected before compilation. GPU correctness,
compilation, graph replay, and performance must be validated on supported
NVIDIA hardware; the source and metadata checks alone do not establish them.
