# Bounded Hy4 runtime validation

This plan covers frontend control contracts and a bounded investigation of
generation differences between cold and warm cache execution. It separates
source synchronization, CPU regression coverage, instrumentation, deployment
identity, GPU evidence collection, and acceptance decisions.

Deployment addresses, account names, local paths, fixed input fingerprints,
approval records, and raw run evidence belong in a caller-supplied run manifest.
They are not part of this reusable repository document. The caller must supply
the expected input identity and the applicable authorization and execution-rule
references explicitly.

## Current implementation boundary

The frontend changes preserve the scheduler's flush result and wake the frontend
response waiter after an explicit abort is successfully enqueued. The diagnostic
supervisor provides structured flush validation, serial request admission,
submission accounting, a shared deadline, and capture-receipt association checks.

The diagnostic command line currently supports `--selftest` and `--plan`. A live
HTTP adapter, current preflight evidence, source-bound tensor capture, tensor-byte
validation, numerical analysis, and evidence-directed follow-up orchestration
still require implementation. A valid capture receipt is not a numerical verdict.

Repository synchronization does not deploy the code. CPU success does not close
the online cancellation issue or authorize a full evaluation. Keep evidence
completeness, observed behavior, root-cause conclusions, cancellation integration
status, and evaluation authorization as separate result fields.

The supervisor requires explicit `sealed_input_sha256`,
`authorization_reference`, `execution_rules_reference`, and `clock` arguments.
Provide the real input fingerprint and controlling references through the private
run configuration, and use a monotonic clock for deadline control. Synthetic
fixtures in the unit tests are only test data and must not be submitted to a
service as the sealed experiment input. A reference string identifies a
controlling artifact; it does not itself prove permission or satisfy preflight.

## Stage map

| Stage | Entry condition | Work | Exit evidence |
| --- | --- | --- | --- |
| 0. Source identity | Reviewed file scope | Commit the source, relevant tests, and reusable diagnostics; verify the remote branch | Local and remote commit IDs agree; changed-file inventory is recorded |
| 1. CPU contracts | Identified dependency environment | Run diagnostic tests and relevant frontend tests; validate actual interface schemas | Commands, exit codes, source hashes, dependency identity, and explicit coverage limits |
| 2. Instrumentation | Actual deployed sampler paths identified | Implement capture, adapters, byte validation, and numerical analysis | Reviewed patch and schema; meaningful synthetic-data and failure-path tests |
| 3. Run preflight | Stages 1 and 2 complete | Verify deployed files, input, topology, in-flight state, and a single budget window | Complete run manifest and current request-correlated cleanup evidence |
| 4. Initial capture | Authorized window active; preflight valid | Run two serial cold/hot/hot cycles, initially at most six submissions | Raw responses, flush receipts, two-position snapshots, and a durable ledger |
| 5. Conditional comparison | A specific hypothesis and remaining budget exist | Use at most six remaining submissions for a discriminating comparison | A stated single variable, matched contexts, and the same shared budget |
| 6. Closeout | New admission has stopped | Verify frontend and scheduler state, analyze evidence, and archive results | Reproducible analysis, unresolved hypotheses, final consumption, and an explicit next decision |

Pure CPU work in stages 1 and 2 can proceed in parallel. Any diagnostic warmup,
capture, or replay verification that uses a GPU belongs to the same bounded GPU
window as stage 4. Generation-based probes consume submission budget even when
called preflight or adapter checks.

## 0. Synchronization and deployment identity

Confirm the exact submitted file set and query the actual remote branch commit.
The relevant local files must match that commit. Preserve unrelated working-tree
changes; do not interpret an up-to-date branch as proof that uncommitted edits
have been uploaded.

Before deployment, record a mapping from commit ID and diagnostic patch hash to
each node's actual source files, imported package paths, native dependencies, and
launch arguments. A branch name or a clean directory alone does not identify the
code already loaded by a running process. Keep a defined restoration procedure
for the identified diagnostic version.

## 1. CPU and frontend contract checks

From the repository root, the implemented diagnostic checks are:

```bash
python3 test/diagnostics/hy4_r6/c4_followup_r6_driver.py --selftest
python3 test/diagnostics/hy4_r6/c4_followup_r6_driver.py --plan
```

These commands perform no live service generation. For the source-boundary/ASGI
harness, with its dependencies installed, run:

```bash
TOKENSPEED_CPU_TEST_ROOT="$PWD" python3 -m unittest discover \
  -s test/diagnostics/hy4_r6 -p focused_cpu_tests.py -v
```

The source-boundary harness compiles selected production bodies and repository
tests without importing unrelated GPU startup modules. Report this scope
explicitly. In the target dependency environment, also verify normal module
imports and run the relevant complete frontend test modules. Do not replace a
failing normal import with a successful extracted-body test and call the runtime
validated.

Retain these regression contracts:

| Contract | Required observation |
| --- | --- |
| Flush success | GET and POST preserve a successful scheduler result |
| Flush refusal | A false result is not wrapped as success; the direct route guard returns its documented error status |
| Flush exception | The caller receives an error and the supervisor admits no next request |
| Explicit abort | A successful IPC send marks the request and wakes its waiter, which exits with an explicit abort error |
| Failed direct abort IPC send | The error propagates, the active request state remains available, and it is not falsely marked aborted |
| Waiter exits while abort IPC fails | Cleanup still removes only that waiter's state; a failed scheduler send must not leave an abandoned frontend state behind |
| Ordering and repetition | Repeated aborts are idempotent; abort before first iteration is visible; an already received natural finish wins |
| Late output | A late scheduler frame cannot replace the frontend abort outcome |
| State ownership | Closing an old generator cannot remove or abort a new state with the same request ID |

The direct engine route and the aggregation gateway can have different response
schemas. Bind the HTTP adapter to the actual deployed interface. For aggregated
flush, require explicit structured success and complete expected worker counts.
The aggregation worker count is not the tensor-parallel rank count.

## 2. Instrumentation work still required

Recheck the following symbols against the selected source version. Historical
line numbers are not an interface contract.

| Source location | Capture or verification responsibility |
| --- | --- |
| `python/tokenspeed/runtime/execution/model_executor.py::_forward_step` | Actual forward identity, output position, execution path, and logits processing stage |
| `model_executor.py::_run_sampling` | Both prefill `sample` and ordinary decode `verify(N=1)` |
| `python/tokenspeed/runtime/sampling/backends/base.py::prepare_step` | Request ID, parameter-pool row, slot/offset, effective parameters, and reuse |
| `sampling/backends/base.py::maybe_broadcast` | Sampling owner, TP group, broadcast source, and pre/post-broadcast tokens |
| `sampling/backends/greedy.py::sample` or the actual backend | Effective selection logits and the backend's actual choice |
| `sampling/backends/triton.py::sample / verify` | Parameter normalization, pool routing, and the decode path when this backend is active |
| `python/tokenspeed/runtime/layers/logits_processor.py::_get_logits` | Whether logits are already global, real-vocabulary slicing, and token ranges |
| `python/tokenspeed/runtime/sampling/utils.py::gather_token_logprobs` and `_write_logprob_outputs` | `logprob_order`, distribution input, and output semantics |
| Proposed HTTP/preflight/trace adapter | Complete raw responses, request-correlated terminal state, current in-flight evidence, trace delivery, and error propagation |
| Proposed tensor reader and analyzer | Snapshot byte/hash/shape/dtype checks, global vocabulary, owner alignment, and independent numerical calculations |
| Proposed conditional controller | Evidence-directed follow-ups sharing the initial window's total ledger and deadline |

Temperature zero does not by itself prove that the standalone greedy backend is
used. Identify the actual backend and effective normalized parameters. Hooking
only `sample` misses the ordinary decode path needed for the fifth output token.

### Capture association and lifecycle

Associate every record with the authorized window, run, request, actual
forward/replay event, output position, rank/TP group, and parameter-pool row. Also
record the prompt fingerprint, actual generated prefix, effective sequence length,
physical bucket, Graph/eager path, backend, effective parameters, and source
identity.

Positions 0 and 4 must refer to different actual forward events. Make an
independent snapshot before the next step reuses the working buffer. Record the
copy-completion event and synchronize the reader with that event. Holding a
tensor reference or overwriting the same snapshot path does not preserve prior
evidence.

Two independently captured tensors can legitimately have identical bytes and
hashes. Detect improper reuse through event association, paths, ownership, and
copy lifecycle; do not require different events to produce different numerical
hashes.

The current receipt validator does not read the tensor. The future reader must
open the actual snapshot bytes and verify them rather than trusting JSON claims
such as `copy_complete`, `argmax`, `top2`, or `logprob` as numerical proof.

### CUDA Graph and replay

A Python hook can run only during capture. The implementation must demonstrate a
fresh event and snapshot for each actual replay. Use capture-compatible
preallocated buffers and completion events, then read at a safe point. Avoid
adding arbitrary `.cpu()` calls or global synchronization inside capture.

Record whether instrumentation changes the execution path or timing. If capture
forces a fallback to eager, label the evidence as eager. Do not report it as a
Graph replay check. A Graph/eager comparison must identify its controlled
variable and any restart, capture, cache, or scheduling differences; the
supervisor must not switch modes automatically.

### CPU and synthetic-evidence negative cases

- Wrong request, forward, position, prompt fingerprint, or generated prefix.
- Pool-row, owner, or broadcast-source mismatch; duplicate events; both capture
  positions incorrectly sharing one forward.
- Missing ranks/shards, overlapping or incomplete vocabulary ranges, or repeated
  concatenation of an already global tensor.
- Incomplete copies, corrupted bytes, incorrect hashes/shapes/dtypes, and
  overwritten snapshots.
- Missing fields, invalid non-finite values, and token disagreement between the
  raw response and capture evidence.
- Timeout, disconnect, non-JSON response, partial flush, and missing worker
  counts.
- Failure to write the raw response or trace, followed by no further admission.
- Attempts to reset the total budget by changing run IDs, restarting the
  controller, or entering the conditional branch.

These tests verify schema, reader, and failure-handling behavior. They do not
prove actual replay freshness; preserve that distinction in the result.

## 3. Preflight before GPU activity

Complete offline preparation before consuming the GPU window:

1. Read explicit authorization and execution-rule references, limits, one window
   ID, code identity, expected input fingerprint, model/tokenizer identity, and
   topology from the controlled run manifest.
2. Verify actual files and imports on every node, launch arguments, backend,
   sampler owner/broadcast configuration, and Graph state.
3. Verify the sealed 4000-token input. Retokenizing into another prompt while
   retaining the old input identity is invalid.
4. Correlate frontend response tasks and scheduler state with outstanding request
   IDs. A healthy endpoint, no new logs, an old Finish message, or absence from a
   request dictionary cannot independently prove that a waiter has exited.
5. Prepare evidence storage, validate durable writes and snapshot capacity, and
   record the restoration procedure for the diagnostic version.
6. Create a persistent ledger for the single authorized window. Use a monotonic
   deadline for control and record wall-clock timestamps for review. Persist each
   submission count before entering the transport.
7. Record the first diagnostic GPU activity. Warmup/capture/replay consume the
   window; generation probes consume submission count.

Do not admit a request when identity, in-flight state, or the remaining budget is
unknown. Report incomplete evidence separately from a numerical model failure.

## 4. Initial two-cycle capture

### Shared hard limits

| Item | Limit |
| --- | --- |
| GPU window | At most 1800 seconds from the first diagnostic GPU activity |
| Actual generation submissions | At most 12, including failed/timed-out submissions and generation probes |
| Output length | At most five new tokens per request; the current two-position contract requests five |
| Initial sequence | Two cold/hot/hot cycles, at most six submissions |
| Reserved submissions | At most six, only for evidence-directed comparisons |
| Automatic retries or resume | None |
| Active abort in this sequence | None |
| Flush | One attempt per cycle; stop on failure |

An early EOS or shorter output preserves whatever first-position evidence exists
but leaves the fifth-position capture incomplete. Do not silently resubmit to
fill the missing position.

### Per-cycle order

1. Recheck remaining time, count, and source identity.
2. Send POST flush and preserve the complete raw receipt. Admit A only when HTTP,
   structured status, and every expected aggregation worker satisfy the contract.
3. Submit cold A. Preserve its complete response immediately. Validate request ID,
   typed finish reason, token/logprob correspondence, `cached_tokens == 0`, and
   valid evidence for both positions.
4. Only after A passes, submit hot B. Repeat validation with `cached_tokens > 0`.
5. Only after B passes, submit hot C. Mark the cycle valid only after C passes.
6. Run the second cycle only if its entry conditions still hold. Any failure stops
   the remaining letters and later cycles.

Use the fixed-cycle supervisor for the initial two cycles. Reserved submissions
require the proposed conditional controller and one shared ledger/deadline. A
new output directory or run object must not create a fresh allowance for the
same authorized diagnostic window.

### Comparable capture positions

Position 0 is the first output token after the real 4000-token prefill. Position
4 is the fifth output token after four actual generated tokens in the same
request. Requests compared at position 4 must have the same first four generated
token IDs.

A single 4004-token prefill is not a substitute for this decode path. Once
generated prefixes diverge, later positions are no longer matched-context
comparisons even if some later token IDs happen to agree.

## 5. Numerical analysis order

1. **Validate identity and bytes.** Check association keys, snapshot hash,
   shape/dtype, vocabulary range, and copy events. Reject a complete JSON receipt
   whose snapshot is missing or invalid.
2. **Identify the decision owner.** Align that rank's effective logits, actual
   pre-broadcast choice, and the post-broadcast token. Another rank's local argmax
   is not the decision reference.
3. **Establish the global vocabulary.** Do not concatenate an already global
   tensor. If sharded, verify complete, non-overlapping ranges before global
   candidate selection or logsumexp.
4. **Identify the distribution stage.** Separate raw logits, selection logits
   after applicable masks/penalties, top-k/top-p processing, and the distribution
   used for API logprob. A top-k=1 tensor with one finite entry cannot establish
   the pre-truncation top-two gap.
5. **Handle non-finite values.** NaNs and unexpected positive infinity invalidate
   numerical evidence. Interpret legitimate masked/padded negative infinity with
   the effective vocabulary and mask.
6. **Check actual selection semantics.** Calculate the true maximum set, top-two
   candidates, and gap. Check membership first, then the actual backend's
   tie-break and broadcast behavior. Membership alone does not prove that a
   deterministic selection contract was followed.
7. **Recompute selected logprob independently.** Match the distribution meaning
   and `logprob_order`; use a stable logsumexp calculation and record analysis
   dtype and order. Do not assume raw-logit softmax is always the API contract.
8. **Compare requests last.** Require identical complete contexts. Predefine
   tolerances from dtype, order, and semantics. Report exact equality separately
   from numerical closeness; an existing API tolerance is not a universal
   internal-tensor threshold.

Archive the analysis script version and input snapshot inventory. Similar
selected logprobs from different forward passes remain descriptive observations;
they do not establish a near tie within either pass.

## 6. Conditional use of the remaining budget

Before each comparison, state the evidence already obtained, the competing
explanations, what remains fixed, the single intended change, the request/time
cost, and which result would distinguish the explanations. Stop and archive when
that design is not available; do not fill the allowance with repeated black-box
calls.

| Evidence | Next discriminating check | Limit on the conclusion |
| --- | --- | --- |
| Owner logits do not correspond to the reported choice | Parameter-pool row, broadcast source, response routing, and capture association | Do not attribute the mismatch to ordinary rounding |
| Matched first-token logits differ | Trace the first divergence through prefill, LM Head, attention, MoE, or indexer | Output logits alone do not identify a specific kernel |
| Matched Graph/eager results differ | Replay metadata, active rows, bucket, buffer lifetime, and synchronization | Record timing/cache/restart differences; the comparison is not automatic proof of a Graph defect |
| A hot request appears to change a shared prefix page | Page/slot ownership and write range, excluding expected private-tail updates | Not every KV change is invalid |
| True same-forward top candidates are close | Record the gap and investigate the source and effect of its change | A near tie does not authorize full evaluation |
| No reproduction, or instrumentation changes the path | Report coverage, input, and instrumentation effects | Do not declare the original issue fixed |

Any permitted restart, recapture, or additional probe remains inside the same
window and submission accounting. The controller must not expand its authority
to perform an unrecorded mode change.

## 7. Stop conditions and closeout

Stop new admission on invalid/partial flush, incorrect cache behavior, timeout,
disconnect, non-JSON response, unknown terminal state, identity drift, capture
association failure, missing/corrupt snapshots, evidence-write failure, a new
unexpected fault, or exhausted count/time.

Preserve request IDs, send times, counts, the last response, errors, and evidence
locations. Follow the applicable restricted error procedure to observe frontend
and scheduler completion. Do not automatically retry, abort, send another
generation probe, switch service modes, or begin another cycle. Preserve an
explicit unknown-in-flight status when cleanup cannot be proven.

Reserve closeout time and stop early when sufficient evidence is available.
Pure CPU integrity checks and offline analysis can continue after the GPU
window; additional GPU activity to fill missing evidence cannot.

Choose a dispatch cutoff before the GPU deadline using measured request latency
and an explicit allowance for passive completion checks. A client timeout does
not stop GPU work. If a request is still in flight at the deadline, record the
run as incomplete and window compliance as unproven, or exceeded when the traces
show that it exceeded the limit. Stopping new submissions alone is not evidence
that the GPU window was respected. Follow the authorized cleanup procedure
without inventing an additional abort or retry allowance.

## 8. Cancellation integration validation is separate

CPU abort tests establish the frontend state contract. Online cancellation needs
deployment-level evidence and is not closed by a normal generation, HTTP 200,
successful slot reuse, or a scheduler Finish message alone.

Record the scope, request allowance, and time window for cancellation integration
tests separately. Active abort testing does not belong to the zero-active-abort
cold/hot sequence above. Do not create an unrecorded extra window for it.

| Observation point | Acceptance evidence |
| --- | --- |
| Request/abort association | Same request ID, explicit ordering, no duplicate scheduler abort on repetition |
| Frontend waiter | The task receives an explicit abort outcome and exits; streaming and non-streaming paths are covered |
| Transport | Real client evidence that the HTTP response or stream terminates |
| Scheduler | Request-correlated acceptance, completion, and state-reclamation evidence |
| Resources and state | Expected request-state and slot/resource behavior; successful reuse is supporting evidence |
| Late frames and disconnects | No resurrection, no deletion of a replacement state, and no second abort |

A local abort marker is not proof of scheduler or GPU drain. Until these checks
are complete, keep cancellation integration open and enforce the applicable
restricted execution and cleanup conditions for every diagnostic run.

## 9. Required archive and evaluation gate

Preserve the authorization reference; code/environment/model/tokenizer identity;
node mapping; sealed input; actual requests; pre-submission ledger; monotonic and
wall-clock times; raw flush receipts; complete responses and errors; capture
index; independent snapshots; request-correlated frontend/scheduler state; and
terminal/in-flight evidence.

Include the integrity report, independent analysis code/results, conditional
comparison rationale, instrumentation effects, unresolved hypotheses, final
consumption, and the next decision. Pairwise comparisons of twelve requests are
still comparisons within one sample of twelve requests, not sixty-six
independent experiments. An invalid request remains counted even when excluded
from numerical analysis.

Full evaluation still requires a supported explanation or verified numerical
fix, cancellation integration or an explicitly accepted restricted mode with
cleanup evidence, actual deployment identity, request-correlated raw-response
collection, isolated scoring, preserved frozen evaluation settings, and an
explicit acceptance decision. Resolve any ambiguity between retry count and
total attempts in the evaluation protocol before implementing its executor. The
bounded diagnostic here uses zero automatic retries.

## Next implementation task

Complete the capture and adapters in stage 2: cover actual `sample` and
`verify(N=1)`, make independent snapshot bytes and replay freshness verifiable,
and connect transport/preflight/trace to one shared budget. Only then start the
initial six submissions. No live command is provided because that integration
does not yet exist.
