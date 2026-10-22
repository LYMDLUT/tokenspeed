# Frontend request completion

`AsyncLLM` owns the request state consumed by its response generator. Scheduler
completion and frontend completion are separate: enqueueing an abort does not
prove that scheduler work or GPU resources have already drained.

An explicit abort follows this contract:

- A completed request keeps its natural terminal response. An unknown request or
  an already requested abort is a no-op.
- The scheduler abort is sent before the frontend state changes. An IPC failure
  propagates to the caller and leaves the request state intact.
- After the send succeeds, `abort_requested` is set and the request event wakes
  its waiter. The state remains addressable until that waiter exits, including
  an abort before the response generator's first iteration.
- The waiter checks the abort state before consuming pending output and raises
  `EngineGenerateError`. Late scheduler output cannot replace that local outcome.
- Generator cleanup removes only the exact state object it owns. Closing an old
  generator must not remove or abort a new request that reused the same ID.
- Client task cancellation still sends one scheduler abort and removes its state.
  Cancellation after an explicit abort does not send a second abort. If the IPC
  send fails while a waiter is exiting, its frontend state is still removed and
  the error propagates; scheduler cleanup remains unconfirmed.

The SGLang-compatible GET and POST `/flush_cache` routes preserve the scheduler's
`FlushCacheReqOutput.success`. A refusal is returned as `success=false` with HTTP
400 by the existing route guard; backend exceptions remain HTTP 500. A successful
HTTP wrapper must never substitute for a successful scheduler flush.

The response-generator regression tests in `test/runtime/test_async_llm_cancel.py`
and the control-route tests in `test/runtime/test_rl_control_endpoint.py` cover
these contracts without requiring GPU execution. End-to-end gateway completion
and scheduler drain still require deployment-level validation.
