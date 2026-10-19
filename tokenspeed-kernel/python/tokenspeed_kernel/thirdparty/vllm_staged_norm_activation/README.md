# Explicit CUDA Norm and SiLU candidates

These optional functions are not registered or imported by the normal kernel
entry points. They require an explicit import from
`tokenspeed_kernel.ops.layernorm.cuda_staged` or
`tokenspeed_kernel.ops.activation.cuda_staged`. Importing does not compile;
the first nonempty call builds a separate TVM-FFI module with FlashInfer
`gen_jit_spec`. Call `load_module()` here before CUDA graph capture.
The module name includes a content hash of `kernels.cu`, `tvm_ffi_utils.h`,
and the explicit compiler flags, including the disabled fast-math setting.

The public source is vLLM v0.29.0, commit
`98dff2a81d747d1dba01a47f939f48c3526d4206`; exact URLs and original byte hashes
are in `SOURCE.json`. The adapted CUDA source retains Apache-2.0 attribution
and the complete upstream `LICENSE`. No precompiled vLLM library is loaded.

## Numerical scope

* `rmsnorm(x, weight, eps, *, out=None)` takes BF16 rank 2–4 input, one
  contiguous BF16 or FP32 weight vector, and positive FP32 epsilon. Input and
  output rows are nonoverlapping, dense, and 16-byte aligned, with hidden size
  divisible by eight. It preserves the source's aligned eight-element
  per-thread accumulation, CUB `BlockReduce<float, 1024>`, `rsqrtf(sum / H + eps)`,
  and `BF16(float(x) * inverse_rms * float(weight))` evaluation order.
  Its launch follows the public default: at most 1024 threads for fewer than
  256 rows, otherwise 256 threads. It does not implement batch-invariant mode.
* `silu_and_mul(x, *, out=None)` takes BF16 packed `[gate | up]` rows. Its scalar
  adaptation preserves `BF16(gate / (1 + expf(-gate * 1)))` before multiplying
  by `float(up) + 0` and rounding again. It is the ordinary unclamped operation
  for dense/shared feed-forwards, not the routed expert clamp implementation.

Explicit adaptations include independent TVM-FFI bindings, BF16-only input,
FP32 norm weights, strided output, safe exact in-place norm, and scalar SiLU
memory access. The public norm originally has same-dtype input/weight, more
dtypes, unweighted and grouped-weight forms, and wider layout handling. The
public SiLU also has packed/vectorized memory paths, which are not copied.
Partial input/output overlap is rejected; only norm accepts a complete alias
with identical pointer, shape, and strides. Address-span overlap checks are
conservative and may reject disjoint interleaved views.

The loader disables FlashInfer fast math and explicitly fixes `ftz=false`,
`prec-div=true`, `prec-sqrt=true`, and `fmad=true`. This records the candidate's
build contract; it does not establish the flags or CUB/toolchain version used
by any separately built reference binary. Inputs must contain finite values
for the numerical tests' tolerance claims; no exceptional-value equivalence
is asserted.

## Validation boundary

The dedicated NVIDIA tests check the formula and intermediate cast,
repeatability, strides, alias rejection, empty rows, current CUDA stream, and
warmed graph replay. They do not load a reference implementation. Before
adoption, compile on the target stack, run these tests, and compare actual
same-input outputs against independently collected reference results. Until
then, source review and Python AST parsing establish neither CUDA compilation
nor numerical equivalence. Runtime selection and deployment are separate.
