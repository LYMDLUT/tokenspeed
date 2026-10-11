# Deep sparse attention kernels

The DSA operators separate sparse-history selection from attention:

- `dsa_prefill_topk` and `dsa_decode_topk` score the index cache and produce
  padded global KV slots plus a live length for every query.
- `dsa_prefill` and `dsa_decode` consume those selected slots and read the
  latent KV cache.
- `dsa_plan` owns optional decode planning state.

AMD CDNA4/gfx950 and CDNA5/gfx1250 provide Gluon implementations for full
selected attention. The gfx1250 implementation supports BF16, E4M3, and E5M2
queries; dense or packed KV rows; page size 64; latent ranks 128 and 512; and
optional 64-wide RoPE. A zero RoPE width selects the same kernel path with the
RoPE storage, gathers, and score term compiled out.

GLM-5.3-Flash uses the zero-RoPE specialization with
`qk_nope_head_dim=256`, `kv_lora_rank=512`, and 16 local heads at TP4. Its
KPool selection has a configured width of 2048 and can append three tail
positions, so both prefill and decode accept padded slot widths 2048 through
2051. The live `topk_lens` value determines how many entries participate in
the online softmax; unused entries remain `-1`.

Portable Triton implementations remain the fallback for trait combinations
without a matching native registration.


The portable Triton attention implementations support `return_lse=True` for
both dense and packed latent caches. They return `(output, lse)`, with FP32
natural-log LSE shaped `[tokens, heads]`; all-invalid rows produce zero output
and negative-infinite LSE. This permits exact softmax-weighted merging of
context-partitioned sparse attention. Supplying `out` preserves the supplied
output buffer even when returning LSE. Omitting `topk_lens` uses the full padded
slot width, with negative slots still excluded.

DeepGEMM index scoring is registered for 16, 32, or 64 index heads. The
16-head case pads queries and weights to its native 32-head ABI with zeros;
caller-provided scoring scales and forced initial/local candidate policies
remain unchanged. Other head counts are excluded by kernel traits.

FlashMLA sparse prefill (regular BF16 KV) and sparse decode (packed FP8 KV)
also support `return_lse=True`. Both return natural-log LSE `[tokens, heads]`
and normalize empty rows to zero output / negative-infinite LSE for the shared
DCP reduction. `topk_lens` masks excluded columns before dispatch. Decode
transposes the vendor's `[batch, heads, query]` LSE to token-major order. Both
vendor kernels apply a sink to the output normalization only, so the adapters
add it to the returned LSE (`logaddexp(lse, sink)`), as the Triton kernels
report it. A supplied output buffer must be contiguous, hold the result's
element count and match its dtype; it retains its identity and anything else
raises. Regular BF16 KV is never reinterpreted as packed FP8.

Decode over regular BF16 KV runs the sparse prefill kernel, one query row per
token, in two registrations. `flashmla_dense_dsa_decode` takes calls with a
sink. `flashmla_dense_dsa_partial_decode` takes sinkless calls that return the
LSE on Hopper: that is how the context-parallel ranks of every DSA model
produce partials, applying any sink once to the merged result. At the DCP 8
shapes of HY V4, GLM-5 and DeepSeek V3.2 (16, 64 and 128 gathered heads, 256
owned columns, batch 1 to 64) it runs 9 to 740 times faster than Triton on H20,
with matching output and LSE. Blackwell keeps Triton for these calls until they
are measured there. Sinkless decode without LSE stays on Triton (Hopper) and
FlashInfer TRT-LLM (Blackwell), because this kernel pads every row's heads to
its 64-head tile. Context-parallel callers pack each rank's owned slots first
and pass their count as `topk_lens`, so the kernel walks only those columns.

FlashMLA sparse prefill attends each query row with one CTA per 64 padded
heads (128 on Blackwell); each CTA walks the row's selected columns serially,
so a launch with fewer CTAs than SMs (regular-KV decode and verify, short
prompts) is bound by one row's walk while most SMs idle. Such launches split
every row's columns into slices that keep the kernel's 128-column alignment,
run the slices as independent rows and merge them by their LSE; the sink is
applied once, to the merged softmax. The split count is the largest power of
two for which the split launch, `rows * ctas_per_row * splits` CTAs, fits one
wave of the device's SMs and every slice keeps that alignment; launches that
cannot take two splits stay whole. HY V4 at TP32 runs 2 heads padded to one CTA
per row. The partials are rounded to BF16 before the FP32 merge, so results
match a single launch to within that rounding. Because the split count follows
the launch's row count, results are reproducible run to run but not
batch-invariant.

## Sharded Index-K candidates

`dsa_index_candidates` scores a bounded query tile against rank-local Index-K
for prefill and decode. Inputs are a position-preserving page table (`-1` for
absent pages), request IDs and global causal lengths. Outputs are global logical
offsets and FP32 scores: invalid candidates use `-1`/`-inf`, and forced
initial/local candidates use `+inf`. Cross-rank merging belongs to runtime DCP.

DeepGEMM compacts owned, causally visible pages and maps results back to global
offsets, masking padding and handling partial tails and empty shards. Query
quantization and head padding match the unsharded path; portable Triton provides
the same interface. Query tiling bounds scratch memory, and fixed-shape GPU
metadata supports CUDA graph replay without host reads.
