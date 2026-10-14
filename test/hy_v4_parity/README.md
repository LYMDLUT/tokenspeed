# HY V4 cross-engine parity harness

Compares TokenSpeed against vLLM 0.29.0, the first vLLM release that ships
`vllm/models/hy_v4`, layer by layer, on a small truncated HY V4 checkpoint.

A full 78-layer HY V4 is ~1.5 TB in BF16, so the harness builds a checkpoint
that is bit-identical to the released weights on the layers it keeps (see
`hy_v4_fixtures.py`) and runs both engines on it. Every sub-block's input and
output is dumped, so a mismatch is reported at the first sub-block that produced
it rather than only at the final logits.

## Quick start

```bash
python -m test.hy_v4_parity.hy_v4_parity \
    --reference-repo /path/to/vllm \
    --reference-python /path/to/vllm-venv/bin/python \
    --tokenspeed-python $(which python)
```

Exit status is 0 only when both the layer dumps and the generated logprobs are
within tolerance, so this can be wired into CI as a gate for the HY V4 port.

Useful switches:

- `--truncation 3:8` — layers to keep : experts to keep (`all` for all 256).
  Layer 0 is dense, layer 1 the first MoE layer, layer 2 the first shared
  indexer, so 3 layers cover every layer kind.
- `--tokenspeed-args "--attention-backend trtllm_mla"` — sweep a backend.
- `--skip-reference` — smoke-test the harness with only TokenSpeed.
- `--rebuild` — rebuild the truncated checkpoint instead of reusing the cache.

Both engines receive fixed token ids and skip tokenizer initialization. The
reference also disables detokenization; the truncated checkpoint therefore
needs no vocabulary or tokenizer model files for this comparison. Dtype labels
remain diagnostic; finite activation values are judged using the explicit
numerical tolerances, not a new cross-engine storage-dtype equality rule.

## Which taps are compared

Callable sub-blocks are hooked with their named activation argument, including
keyword-only calls. Parameter-only iHC boundaries are observed at the actual
`ihc_pre`/`ihc_fused_hc` operation, without replacing the fused calculation.
For the fused operation, the recorded input is its returned, updated residual
channels, the input of the next pre boundary. Both engines record the iHC
output at the following attention/MLP input, after RMSNorm whether it is fused
or separate. Raw pre-normalization output is not compared with normalized
input. Missing expected taps fail at collection completion. iHC token counts
exclude the residual-channel axis.

Per decoder layer, in execution order: `hc_attn`, `attn`, `hc_mlp`, `mlp`, each
as input and output. That ordering matters when diagnosing:

- `hc_attn`/`hc_mlp` divergence — the iHC boundary formula (gates, eps, channel
  reduction, fused RMSNorm).
- `attn` divergence — the sparse MLA path: indexer, rope slice, learnable sink,
  or the output gate.
- `mlp` divergence — the clamped SwiGLU on the routed experts, or the router.

Each tap stores its shape, whole-tensor norm and peak, and an 8x8 corner sample.
The comparison checks all three signals. Empty or malformed evidence,
non-finite values and missing per-token logprobs fail. Tolerances must be
finite and non-negative; zero requests an exact comparison. Samples and
statistics are diagnostic, not proof of full-tensor equality: an unsampled
permutation can preserve all three. Use complete tensor comparisons for a
bitwise or elementwise guarantee.

## Dumping a live server

For a running engine (not the harness), inject the dump via `PYTHONPATH`:

```bash
mkdir -p /tmp/hy4inject
cp test/hy_v4_parity/hy_v4_dump_sitecustomize.py /tmp/hy4inject/sitecustomize.py
HY_V4_DUMP_PATH=/tmp/ts.json \
HY_V4_DUMP_TAG=tokenspeed \
PYTHONPATH=/tmp/hy4inject:$PYTHONPATH \
  ./your_normal_launch.sh
```

The launcher installs this injection for the parent and spawned workers,
preserving additional `PYTHONPATH` entries. Each worker writes
`<stem>.worker-<pid>.json` and a completion sidecar before its selected forward
returns; it does not rely on interpreter exit. Direct `.forward()` calls are
covered too. `HY_V4_DUMP_PROMPT_IDS` selects exact token ids rather than a
warmup batch of the same length. Without that filter, live-server dumps are
diagnostic snapshots of the latest complete forward.

Every harness invocation creates a fresh engine-run directory. The default
single-GPU check requires exactly one complete worker dump, the requested
layers and input, and a summary with the same run identity. Missing evidence
cannot reuse an older dump, and multiple worker dumps are not silently merged.
The instrumentation is inert unless `HY_V4_DUMP_PATH` is set.

| Variable | Meaning |
| --- | --- |
| `HY_V4_DUMP_PATH` | Base dump name; each worker appends its pid. Unset means disabled. |
| `HY_V4_DUMP_PROMPT_IDS` | JSON token-id list selecting one exact prefill; the harness sets it. |
| `HY_V4_DUMP_LAYERS` | Layers to tap; default is all in the config. |
| `HY_V4_DUMP_TAG` | Free-form label written to `<path>.meta.json`. |
| `HY_V4_DUMP_TARGETS` | Extra `module:class` targets, for a reference checkout laid out differently. |

Compare two dumps with:

```python
from test.hy_v4_parity.hy_v4_dump import compare_dumps

passed, report = compare_dumps("reference.json", "tokenspeed.json")
print(report)
```

## Testing the harness itself

`test_hy_v4_parity_harness.py` covers the hook, the comparator and the
truncation rewrite on CPU, with no checkpoint download. A parity run that
silently compares nothing is worse than no run at all, so these tests include
cases for a missing tap, a shape mismatch, and a divergence that hides outside
the sampled corner.

```bash
python -m unittest test.hy_v4_parity.test_hy_v4_parity_harness
```

## Draft step-0 graph check

`draft_graph_check.py` checks the MTP draft's step-0 prefill graph against the
eager step on real weights. It truncates the checkpoint (keeping the MTP
block), serves one greedy MTP batch twice on one GPU, with the draft graph and
with step 0 eager, and compares what the draft proposed at every served extend
or mixed step:

```bash
python -m test.hy_v4_parity.draft_graph_check --source /path/to/hy_v4
```

Only actual graph replays count as graph evidence. `--enable-mixed-batch`
also lets prompt chunks share batches with decodes; those mixed rounds keep
the eager step 0 in both runs and cannot dilute the replay verdict. Every
engine process loads `draft_graph_sitecustomize.py` as its `sitecustomize`,
which toggles the draft graph (`HY_V4_DRAFT_GRAPH`) and records the actual
step-0 path, an input fingerprint, proposals and full tensor snapshots.

The checker requires identical live inputs/frontiers and checks every replay
proposal and per-row top-2 gap (`--max-gap-delta`), full logits, written latent
KV and dequantized index-K with explicit numerical tolerances. Missing or
non-finite tensors, incomplete traces and different input sequences fail.
Generated token sequences must also match. Later draft ids and verification
counts remain diagnostic; matching those counts alone cannot establish parity.

`--min-match` is retained only for command-line compatibility and must be `1.0`.
The checker grants neither an arbitrary percentage of errors nor a median
that can hide a bad replay. Near ties or divergent schedules need a same-input
reproduction; a failed comparison is not silently reclassified as rounding.
Snapshots remain next to the trace for diagnosis; they may be large, and the
GPU synchronization makes this unsuitable for timing.

The oracle and its negative tests ship with the graph change under
`test/runtime/draft_graph_validation.py` and
`test/runtime/test_draft_graph_validation.py`. The CPU harness tests also
exercise the saved evidence reader: identical `.pt` payloads pass, while a
missing file, corrupted latent KV, missing tensor or mismatched input frontier
fails. Run them without a checkpoint or GPU:

```bash
python -m pytest -q test/runtime/test_draft_graph_validation.py \
  test/hy_v4_parity/test_hy_v4_parity_harness.py
```

These tests validate the checker and artifact reader, not the GPU recorder
or captured inference. The latter still needs a real-checkpoint run at the
tested commit.

## Suggested order

1. `gpu_kernel_check.py` on any CUDA GPU (no Hopper, no checkpoint needed).
2. `--skip-reference --load-format dummy` to confirm instrumentation attaches.
3. Full run at `--truncation 3:8`: catches wiring and boundary bugs cheaply.
4. `--truncation 6:8` (adds another full indexer layer) to catch indexer reuse.
5. Full-layer run at real scale, then a GSM8K comparison and MTP acceptance rate.

## Kernel-level GPU check

`gpu_kernel_check.py` executes the new iHC Triton kernels and the learnable-sink
branch of the DSA Triton kernel on a real GPU and compares them with reference
formulas. It loads only the two op modules through a shadow package, so it runs
on a much lighter stack than the full runtime:

```bash
PYTHONPATH=$PWD/tokenspeed-kernel/python:$PWD/python \
  python test/hy_v4_parity/gpu_kernel_check.py --tokens 128 --hc 4 --hidden 6144
```

Notes on the tolerances, since they are deliberate:

- Gates are FP32 end to end and are held to `1e-4`.
- BF16 outputs use a bound of one BF16 ulp at the reference's peak magnitude.
  A fixed bound fails spuriously as outputs grow (the iHC head sums `hc_mult`
  channels).
- The iHC test uses large epsilons (`hc_eps=0.05`, `rms_eps=0.5`). With the
  released ~1e-6 values a kernel that drops an epsilon differs by less than BF16
  rounding and would pass.

The FlashMLA and Triton sink paths over a dense BF16 KV cache are covered by
`tokenspeed-kernel/test/ops/test_attention_dsa_sink.py`, which goes through the
public `dsa_decode` / `dsa_prefill` dispatchers. Its strong-sink regressions also
cover the portable Triton FP8 packed-KV path, true empty rows and invalid
slots, in both prefill and decode. These are kernel checks, not model-level
numerical validation.

## Serving benchmark metric contract

`serve_bench.py` emits `metrics_version: 2`. TTFT is measured at the first
nonempty text frame (whitespace counts); TPOT uses the interval between the
first and last nonempty text frames divided by the reported completion token
count minus one. It is a client-observed estimate when frames coalesce tokens,
not a measurement of every GPU token step. End-to-end time includes the
remaining protocol tail, while delayed usage/footer frames do not inflate TPOT.

A response must finish with `[DONE]`, contain visible output, and report valid
usage with the requested completion length. Missing usage, truncated streams,
HTTP-200 error events and short generations fail instead of borrowing the
requested token count to manufacture throughput. Do not mix older metric
records with version 2 without rerunning the same workload. Existing published
benchmark values are not recomputed or relabeled by this change.

Checkpoint rebuilds refuse a destination equal to the source checkpoint or
containing it, including a symlink alias. Symlink output directories are also
rejected. Without `--rebuild`, only a matching export with its checkpoint files
present is reused; a nonempty destination with a missing, different or
incomplete marker is rejected instead of silently overwriting files. An empty
directory may be used for a first export. Configuration and selected source
tensors are validated before an existing output is removed. `--rebuild`
replaces an export, never its source checkpoint; use a separate output path.

The exported index's `metadata.total_size` counts tensor storage bytes,
including the different element sizes of BF16, FP32 and FP8 weights.
