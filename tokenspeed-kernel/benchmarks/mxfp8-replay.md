# MXFP8 captured-layer replay

`python -m tokenspeed_kernel.benchmark.mxfp8_replay` compares canonical
group32 E4M3/E8M0 weights without requantizing the checkpoint. Run each backend
in a separate process against the same artifact. A linear replay includes
online activation quantization and GEMM. A routed-expert replay includes
routing preparation, both GEMMs, the routed SwiGLU clamp, intermediate BF16
rounding, online activation quantization, route weighting and finalization.
It returns one rank's contribution and does not time communication, shared
experts, attention or scheduling.

## Capture format

Call `export_artifact(directory, manifest, tensors)` from the replay module at
a layer boundary. Supply the original checkpoint payloads before backend
repacking, and activations and routing from a real model forward. The exporter
does not install hooks or create synthetic workloads. It writes `manifest.json`
and a tensor-only `tensors.pt`; loading verifies the payload SHA256. Export is
outside all timing and never replaces an existing directory.

The manifest requires `schema_version: 1`, `kind: "linear"` or `"moe"`, public
`checkpoint`, `checkpoint_revision`, `layer`, `capture_revision`, and explicit
`tp_size`, `tp_rank`, `ep_size`, `ep_rank`. The MoE manifest additionally records
global `num_experts` and `activation_limit: 10.0`. Expert shards must be equal,
contiguous ranges. These identifiers describe the captured source; record the
capture model configuration and input dataset revision with the experiment.

| Tensor | Linear shape | MoE shape | Storage |
| --- | --- | --- | --- |
| `x` | `[T, K]` | `[T, H]` | BF16 |
| `weight` | `[N, K]` | — | E4M3 |
| `weight_scale` | `[N, K/32]` | — | uint8 E8M0 |
| `w13_weight` | — | `[E_local, 2I, H]`, gate then up | E4M3 |
| `w13_weight_scale_inv` | — | `[E_local, 2I, H/32]` | uint8 E8M0 |
| `w2_weight` | — | `[E_local, H, I]` | E4M3 |
| `w2_weight_scale_inv` | — | `[E_local, H, I/32]` | uint8 E8M0 |
| `topk_ids` | — | `[T, top_k]`, global IDs, `-1` for padding | int32/int64 |
| `topk_weights` | — | `[T, top_k]` | FP32 |

The shared Marlin/Humming comparison accepts scale bytes 1 through 254.
Bytes 0 and 255 are rejected, rather than silently changing their value.
Humming additionally requires its documented Hopper dimension alignments.
Capture multiple actual layers and routing loads; a single layer does not
represent a full model's mixture of shapes or expert imbalance.

## Run

Choose error tolerances before inspecting the candidate result. The example
passes them as shell variables to avoid imposing an unvalidated acceptance
threshold. For a routed-expert capture, run each command separately:

```bash
python -m tokenspeed_kernel.benchmark.mxfp8_replay \
  --artifact /path/to/capture --output marlin.json --backend marlin \
  --timing cuda-graph --cache cold --reference-device cuda \
  --atol "$ATOL" --rtol "$RTOL" --warmup 5 --samples 50

python -m tokenspeed_kernel.benchmark.mxfp8_replay \
  --artifact /path/to/capture --output humming.json --backend humming \
  --timing cuda-graph --cache cold --reference-device cuda \
  --atol "$ATOL" --rtol "$RTOL" --warmup 5 --samples 50
```

Linear captures use `--backend marlin` for W8A16 and `--backend triton` for
the existing group32 W8A8 GEMM. `--timing eager` invokes the same public
operation without graph capture; it is a separate comparison. A graph error
fails the run instead of falling back to eager. MoE explicitly requires
deterministic finalization, and Humming's adapter disables Stream-K.
PDL is disabled for every comparator because the Humming adapter rejects it;
the script restores the caller's PDL and TF32 settings on exit.

Compare only matching artifact hashes, topology, timing mode, cache state,
warmup counts and GPU configuration. Preserve every raw timing sample. Report
the ratio of medians together with p50/p95 and the actual selected kernel.
Package versions, Git revision and dirty state are recorded. Preparation,
compile/warmup/first-call/readback wall time and warmed device timings are separate.
The first-call figure depends on existing compiler caches and is not a claim
of an empty-cache build. PyTorch allocator peaks include graph and kernel
workspaces allocated through PyTorch, but exclude CUDA context/module memory
and allocations made directly by native libraries. Measure total process
device memory separately for deployment capacity. In these allocator peaks,
cold-cache timing also includes the timer's cache-clearing buffer in memory
statistics, but excludes clearing from device timing.

Precision failure or a repeated-call byte mismatch marks `accepted_timing`
false and returns a nonzero exit status. Keep such timing samples only as
diagnostics; exclude them from accepted speedup comparisons.

An older composite implementation that selects GEMV or expands weights to
BF16 must have a separately labeled report from its own pinned revision.
Never label that result "pure Marlin", import its implementation into this
replay, or use it as the only W8A16 baseline. Different resident buffers and
preparation costs are part of that comparison.

## Precision and acceptance

The reference uses FP32 GEMMs with TF32 disabled. It preserves BF16 gate/up
outputs, clamps routed gate to at most 10 and up to `[-10, 10]`, then rounds
SwiGLU to BF16. GEMM2 output, route weight and their product are BF16; top-k
slots accumulate in order in FP32 and the result rounds to BF16.

The W8A8 reference additionally quantizes each 32-value activation group to
E4M3 using `scale = ceil_pow2(max(amax, 1e-4) / 448)`. Humming stores those
same scale values in FP32. The report separates:

- The W8A8 reference against the W8A16 reference: activation quantization's
  effect under the stated reference arithmetic.
- The implementation against its matching reference: full-chain execution
  deviation. FC1 rounding differences can cross FC2 quantization thresholds,
  so this is not labeled isolated GEMM error.
- The implementation against the W8A16 reference, plus repeated-call byte
  equality and measured-output equality with eager. CUDA Graph runs read the
  actual captured output after replay, rather than checking only eager calls.

Fixed-operand GEMM tests must separately feed identical E4M3 payloads and
scales to the candidate and FP32 oracle, for both projections. That isolates
kernel arithmetic from online quantization. The replay's CPU tests certify
artifact validation and reference operation order; they do not certify H20
kernels. Layer results likewise do not establish end-to-end quality or speed:
complete the multi-rank load, eager/graph, long-context, teacher-forced
logprob/NLL and serving-latency checks against the same model/tokenizer/input
revision before changing deployment defaults. No GSM8K run is part of this
replay.
