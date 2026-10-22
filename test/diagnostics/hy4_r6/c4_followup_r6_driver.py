#!/usr/bin/env python3
"""CPU-tested admission supervisor for the planned C4 r6 diagnostic.

This module deliberately contains no HTTP client or instrumentation installer.
The command line supports preparation inspection and CPU self-tests only. A
reviewed integration must supply transport, current preflight evidence, and an
internal-capture reader before ``Supervisor.run`` can be used on a service.

Capture receipt validation is not tensor/numerics validation. In particular,
greedy checks belong to the sampler owner rank's logits before token broadcast;
another rank's local argmax is not a valid substitute. Source hooks and the
independent numeric comparison remain the next implementation/validation step.
The next live integration should start with ``cycles=2`` (six requests). The
remaining six requests are reserved for evidence-dependent follow-up; this
fixed-cycle supervisor does not implement that conditional/adaptive controller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import unittest
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

SEALED_INPUT_TOKENS = 4000
HEX256 = re.compile(r"[0-9a-f]{64}\Z")


class InvalidEvidence(ValueError):
    """The evidence cannot support admission of another request."""


class BudgetExhausted(InvalidEvidence):
    """No more work may be admitted in this diagnostic window."""


def _integer(value: Any, label: str, *, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise InvalidEvidence(f"{label}: expected integer >= {minimum}")
    return value


def _finite(value: Any, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise InvalidEvidence(f"{label}: expected finite number")
    return float(value)


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or HEX256.fullmatch(value) is None:
        raise InvalidEvidence(f"{label}: expected lowercase SHA256")
    return value


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidEvidence(f"{label}: expected nonempty string")
    return value


def _json_body(response: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(response, Mapping):
        raise InvalidEvidence("response is not an object")
    if type(response.get("status")) is not int or response["status"] != 200:
        raise InvalidEvidence(f"HTTP status {response.get('status')!r}")
    if response.get("exc") is not None:
        raise InvalidEvidence(f"transport failure: {response['exc']}")
    text = response.get("body_text")
    if not isinstance(text, str):
        raise InvalidEvidence("missing complete raw response body")

    def reject_constant(value: str) -> None:
        raise InvalidEvidence(f"non-JSON numeric constant {value}")

    try:
        body = json.loads(text, parse_constant=reject_constant)
    except (ValueError, TypeError) as error:
        raise InvalidEvidence(f"invalid JSON response: {error}") from error
    if not isinstance(body, dict):
        raise InvalidEvidence("JSON response must be an object")
    return body


def flush_verdict(
    response: Mapping[str, Any], *, expected_workers: int
) -> tuple[bool, str]:
    """Validate the observed POST aggregation contract, not the GET adapter.

    ``expected_workers`` is the configured HTTP aggregator count, not
    the tensor-parallel rank count. Free-form message text is not a predicate.
    """
    try:
        _integer(expected_workers, "expected_workers", minimum=1)
        body = _json_body(response)
        if body.get("status") != "success":
            raise InvalidEvidence("flush body status is not success")
        total = _integer(body.get("total_workers"), "total_workers", minimum=1)
        flushed = _integer(body.get("workers_flushed"), "workers_flushed", minimum=1)
        if total != expected_workers or flushed != total:
            raise InvalidEvidence("flush worker counts do not match expected topology")
        if "success" in body and body["success"] is not True:
            raise InvalidEvidence("contradictory or non-boolean success field")
        if "workers_failed" in body:
            failed = _integer(body["workers_failed"], "workers_failed", minimum=0)
            if failed != 0:
                raise InvalidEvidence("flush reports failed workers")
        if "total_http_workers" in body or "total_grpc_workers" in body:
            http_workers = _integer(
                body.get("total_http_workers"), "total_http_workers", minimum=0
            )
            grpc_workers = _integer(
                body.get("total_grpc_workers"), "total_grpc_workers", minimum=0
            )
            if http_workers + grpc_workers != total:
                raise InvalidEvidence(
                    "flush component worker counts contradict total_workers"
                )
        if "total_zmq_workers_skipped" in body:
            skipped = _integer(
                body["total_zmq_workers_skipped"],
                "total_zmq_workers_skipped",
                minimum=0,
            )
            if skipped != 0:
                raise InvalidEvidence("flush skipped workers")
    except InvalidEvidence as error:
        return False, str(error)
    return True, "structured_all_workers_success"


def first_divergence(
    ids_a: list[int],
    ids_b: list[int],
    logprobs_a: list[float] | None = None,
    logprobs_b: list[float] | None = None,
) -> dict[str, Any]:
    """Describe output differences; cross-run probabilities imply no near tie."""
    position = 0
    while position < min(len(ids_a), len(ids_b)) and ids_a[position] == ids_b[position]:
        position += 1
    equal = ids_a == ids_b
    result: dict[str, Any] = {
        "ids_equal": equal,
        "common_prefix_len": position,
        "output_position": None if equal else position,
        "token_id_A": ids_a[position] if position < len(ids_a) else None,
        "token_id_B": ids_b[position] if position < len(ids_b) else None,
        "root_cause": "UNDETERMINED",
    }
    if not equal and logprobs_a is not None and logprobs_b is not None:
        if position < len(logprobs_a) and position < len(logprobs_b):
            left = _finite(logprobs_a[position], "selected_logprob_A")
            right = _finite(logprobs_b[position], "selected_logprob_B")
            result.update(
                {
                    "selected_logprob_A": left,
                    "selected_logprob_B": right,
                    "cross_run_selected_logprob_abs_delta": abs(left - right),
                    "delta_semantics": "different_forward_selected_logprobs_only",
                }
            )
    return result


@dataclass(frozen=True)
class Limits:
    """All correctness-relevant limits are explicit at the call site."""

    expected_workers: int
    vocab_size: int
    max_requests: int
    max_new_tokens: int
    wall_seconds: float
    request_timeout_seconds: float
    control_timeout_seconds: float
    trace_timeout_seconds: float

    def validate(self) -> None:
        _integer(self.expected_workers, "expected_workers", minimum=1)
        _integer(self.vocab_size, "vocab_size", minimum=2)
        if _integer(self.max_requests, "max_requests", minimum=1) > 12:
            raise InvalidEvidence("r6 permits at most 12 generate submissions")
        if _integer(self.max_new_tokens, "max_new_tokens", minimum=1) != 5:
            raise InvalidEvidence(
                "this two-position diagnostic requires exactly 5 output tokens"
            )
        wall = _finite(self.wall_seconds, "wall_seconds")
        if not 0 < wall <= 1800:
            raise InvalidEvidence("r6 wall budget must be in (0, 1800] seconds")
        for field, value in (
            ("request_timeout_seconds", self.request_timeout_seconds),
            ("control_timeout_seconds", self.control_timeout_seconds),
            ("trace_timeout_seconds", self.trace_timeout_seconds),
        ):
            if _finite(value, field) <= 0:
                raise InvalidEvidence(f"{field} must be positive")


def validate_generation(
    response: Mapping[str, Any],
    *,
    rid: str,
    cold: bool,
    limits: Limits,
) -> dict[str, Any]:
    body = _json_body(response)
    meta = body.get("meta_info")
    if not isinstance(meta, dict):
        raise InvalidEvidence("missing meta_info")
    response_id = meta.get("id", body.get("id"))
    if response_id != rid or any(
        value != rid for value in (body.get("id", rid), meta.get("id", rid))
    ):
        raise InvalidEvidence("response request ID does not match dispatched request")
    ids = body.get("output_ids")
    if not isinstance(ids, list) or len(ids) != limits.max_new_tokens:
        raise InvalidEvidence(
            "incomplete two-position output or output token budget exceeded"
        )
    for token in ids:
        if _integer(token, "output token", minimum=0) >= limits.vocab_size:
            raise InvalidEvidence("output token outside vocabulary")
    if "output_ids" in meta and meta["output_ids"] != ids:
        raise InvalidEvidence("conflicting output_ids representations")
    if (
        _integer(meta.get("prompt_tokens"), "prompt_tokens", minimum=1)
        != SEALED_INPUT_TOKENS
    ):
        raise InvalidEvidence("prompt length changed")
    if _integer(meta.get("completion_tokens"), "completion_tokens", minimum=0) != len(
        ids
    ):
        raise InvalidEvidence("completion count does not match output_ids")
    cached = _integer(meta.get("cached_tokens"), "cached_tokens", minimum=0)
    if (
        cached >= SEALED_INPUT_TOKENS
        or (cold and cached != 0)
        or (not cold and cached == 0)
    ):
        raise InvalidEvidence("cold/hot cache gate not satisfied")
    finish = meta.get("finish_reason")
    if not isinstance(finish, dict) or finish.get("type") not in ("length", "stop"):
        raise InvalidEvidence("generation lacks a successful typed terminal response")
    entries = meta.get("output_token_logprobs")
    if not isinstance(entries, list) or len(entries) != len(ids):
        raise InvalidEvidence("selected logprob count does not match output_ids")
    logprobs = []
    for index, item in enumerate(entries):
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            raise InvalidEvidence("malformed selected logprob tuple")
        probability = _finite(item[0], "selected logprob")
        token = _finite(item[1], "logprob token ID")
        if not token.is_integer() or int(token) != ids[index]:
            raise InvalidEvidence("selected logprob token ID/position mismatch")
        logprobs.append(probability)
    return {"id": rid, "ids": ids, "logprobs": logprobs, "cached_tokens": cached}


def validate_capture_receipt(
    evidence: Any,
    *,
    run_id: str,
    rid: str,
    output: dict[str, Any],
    diagnostic_source_sha256: str,
    sealed_input_sha256: str,
    limits: Limits,
) -> None:
    """Check identity/association of receipts, without accepting their numerics.

    A source-bound integration must validate and archive the referenced immutable
    tensor files. These schema checks alone do not establish that the tensor
    bytes, global aggregation, sampler decision, or logprob values are correct.
    """
    _hash(sealed_input_sha256, "sealed_input_sha256")
    if not isinstance(evidence, dict):
        raise InvalidEvidence("internal capture receipt missing")
    for name, expected in (
        ("run_id", run_id),
        ("rid", rid),
        ("diagnostic_source_sha256", diagnostic_source_sha256),
    ):
        if evidence.get(name) != expected:
            raise InvalidEvidence(f"internal capture {name} mismatch")
    captures = evidence.get("captures")
    if not isinstance(captures, list) or len(captures) != 2:
        raise InvalidEvidence("both internal capture positions are required")
    positions = [
        capture.get("output_position")
        for capture in captures
        if isinstance(capture, dict)
    ]
    if (
        len(positions) != 2
        or any(type(position) is not int for position in positions)
        or sorted(positions) != [0, 4]
    ):
        raise InvalidEvidence("internal captures must uniquely cover positions 0 and 4")
    forward_ids = set()
    snapshot_paths = set()
    for capture in captures:
        position = capture["output_position"]
        forward_id = _nonempty(capture.get("forward_id"), "forward_id")
        if forward_id in forward_ids:
            raise InvalidEvidence("capture positions must identify distinct forwards")
        forward_ids.add(forward_id)
        if (
            _hash(capture.get("prompt_ids_sha256"), "prompt_ids_sha256")
            != sealed_input_sha256
        ):
            raise InvalidEvidence(
                "capture prompt identity does not match the sealed input"
            )
        prefix = capture.get("generated_prefix_ids")
        if (
            not isinstance(prefix, list)
            or any(type(token) is not int for token in prefix)
            or prefix != output["ids"][:position]
        ):
            raise InvalidEvidence("capture did not follow the actual generated prefix")
        if (
            _integer(capture.get("model_input_tokens"), "model_input_tokens", minimum=1)
            != SEALED_INPUT_TOKENS + position
        ):
            raise InvalidEvidence("capture effective sequence length mismatch")
        if capture.get("forward_kind") != ("prefill" if position == 0 else "decode"):
            raise InvalidEvidence(
                "4004-token prefill cannot stand in for position-4 decode"
            )
        if (
            _integer(
                capture.get("post_broadcast_token_id"),
                "post_broadcast_token_id",
                minimum=0,
            )
            != output["ids"][position]
        ):
            raise InvalidEvidence("internal/output selected-token association mismatch")
        for name in (
            "sampler_owner_rank",
            "capture_rank",
            "request_pool_row",
            "global_logits_row",
        ):
            _integer(capture.get(name), name, minimum=0)
        if capture["capture_rank"] != capture["sampler_owner_rank"]:
            raise InvalidEvidence("capture is not from the sampler owner rank")
        local_selected = _integer(
            capture.get("pre_broadcast_token_id"), "pre_broadcast_token_id", minimum=0
        )
        if local_selected >= limits.vocab_size:
            raise InvalidEvidence("pre-broadcast token outside vocabulary")
        if capture.get("execution_mode") not in ("graph", "eager"):
            raise InvalidEvidence("missing execution mode")
        if (
            _integer(capture.get("global_vocab_size"), "global_vocab_size", minimum=2)
            != limits.vocab_size
        ):
            raise InvalidEvidence("global valid vocabulary evidence is missing")
        if capture.get("independent_copy") is not True:
            raise InvalidEvidence("capture must own an independent snapshot")
        snapshot_path = _nonempty(capture.get("snapshot_path"), "snapshot_path")
        if snapshot_path in snapshot_paths:
            raise InvalidEvidence("capture positions must not reuse a snapshot path")
        snapshot_paths.add(snapshot_path)
        for name in (
            "synchronization_boundary",
            "logprob_semantics",
            "sampler_backend",
        ):
            _nonempty(capture.get(name), name)
        _hash(capture.get("snapshot_sha256"), "snapshot_sha256")
        for name in ("effective_sampling_params", "physical_bucket", "slot_metadata"):
            if not isinstance(capture.get(name), dict) or not capture[name]:
                raise InvalidEvidence(f"missing {name}")


class Supervisor:
    """One non-resumable run; calls are serial and all failures stop admission.

    Injected callables must honor the supplied timeout and must not retry or
    generate requests internally. Transport receives (method, path, body,
    timeout); trace_reader receives (run_id, rid, timeout). Preflight receives
    (run_id, timeout). None may call abort. Their production integration is
    deliberately not supplied by this preparation module. ``output_root`` is
    the directory for one approved diagnostic window: an exclusive durable claim
    prevents a new process/instance from resetting that window's budget.
    The caller supplies the sealed prompt SHA256 and the authorization/rules
    references; these are never inferred from public test fixtures. A live
    integration must pass a monotonic clock such as ``time.monotonic``.
    """

    def __init__(
        self,
        *,
        limits: Limits,
        output_root: Path,
        sealed_input_sha256: str,
        authorization_reference: str,
        execution_rules_reference: str,
        transport: Callable[..., Mapping[str, Any]],
        trace_reader: Callable[..., Mapping[str, Any]],
        preflight: Callable[..., Mapping[str, Any]],
        clock: Callable[[], float],
    ) -> None:
        limits.validate()
        self.sealed_input_sha256 = _hash(sealed_input_sha256, "sealed_input_sha256")
        self.authorization_reference = _nonempty(
            authorization_reference, "authorization_reference"
        )
        self.execution_rules_reference = _nonempty(
            execution_rules_reference, "execution_rules_reference"
        )
        self.limits = limits
        self.transport = transport
        self.trace_reader = trace_reader
        self.preflight = preflight
        self.clock = clock
        self.run_id = f"c4fr6-{uuid.uuid4().hex}"
        self.output_root = Path(output_root)
        self.directory = self.output_root / self.run_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.started = False
        self.started_at = 0.0
        self.deadline = 0.0
        self.generated = 0
        self.event_count = 0
        self.halted = False
        self.outputs: list[dict[str, Any]] = []
        self.diagnostic_source_sha256 = ""

    def _claim_window(self) -> None:
        claim = self.output_root / "R6_WINDOW_CLAIM.json"
        try:
            with claim.open("x", encoding="utf-8") as stream:
                json.dump(
                    {
                        "run_id": self.run_id,
                        "limits": asdict(self.limits),
                        "resume_allowed": False,
                    },
                    stream,
                    indent=2,
                    allow_nan=False,
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError as error:
            raise InvalidEvidence(
                "this diagnostic window was already claimed; restarting cannot reset its budget"
            ) from error
        descriptor = os.open(self.output_root, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _save(self, name: str, value: Any) -> None:
        target = self.directory / name
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        descriptor = os.open(self.directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _event(self, kind: str, **fields: Any) -> None:
        self.event_count += 1
        self._save(
            f"event_{self.event_count:04d}.json",
            {
                "run_id": self.run_id,
                "kind": kind,
                "elapsed_seconds": self.clock() - self.started_at,
                "generate_attempts": self.generated,
                **fields,
            },
        )

    def _remaining(self, cap: float) -> float:
        if self.halted:
            raise InvalidEvidence("run already halted; resumption is forbidden")
        remaining = self.deadline - self.clock()
        if remaining <= 0:
            raise BudgetExhausted("r6 wall budget exhausted")
        return min(cap, remaining)

    def _request(
        self, path: str, body: dict[str, Any], *, generate: bool
    ) -> Mapping[str, Any]:
        cap = (
            self.limits.request_timeout_seconds
            if generate
            else self.limits.control_timeout_seconds
        )
        timeout = self._remaining(cap)
        if generate:
            if self.generated >= self.limits.max_requests:
                raise BudgetExhausted("r6 generate submission budget exhausted")
            # Reserve the attempt durably before entering transport. An uncertain
            # dispatch counts; a crash cannot silently reset a consumed attempt.
            self.generated += 1
        self._event(
            "dispatch_intent",
            method="POST",
            path=path,
            body=body,
            timeout_seconds=timeout,
        )
        # Durable bookkeeping itself consumes the same deadline.
        timeout = self._remaining(cap)
        try:
            response = self.transport("POST", path, body, timeout)
        except Exception as error:
            self._event(
                "transport_exception",
                path=path,
                error=f"{type(error).__name__}: {error}",
            )
            raise InvalidEvidence(
                "transport completion is uncertain; stop and establish in-flight state"
            ) from error
        # Raw text preserves even non-JSON/nonfinite backend output. Do not
        # serialize a transport's optional parsed convenience object instead.
        raw = (
            {name: response.get(name) for name in ("status", "exc", "body_text")}
            if isinstance(response, Mapping)
            else {"malformed_envelope": repr(response)}
        )
        self._event("raw_response", path=path, response=raw)
        self._remaining(cap)
        return response

    def _check_preflight(self, receipt: Mapping[str, Any]) -> None:
        if not isinstance(receipt, Mapping) or receipt.get("run_id") != self.run_id:
            raise InvalidEvidence("preflight receipt must bind to this fresh run")
        self.diagnostic_source_sha256 = _hash(
            receipt.get("diagnostic_source_sha256"), "diagnostic_source_sha256"
        )
        if (
            _integer(receipt.get("expected_workers"), "expected_workers", minimum=1)
            != self.limits.expected_workers
        ):
            raise InvalidEvidence("preflight aggregation topology mismatch")
        if receipt.get("instrumentation_ready") is not True:
            raise InvalidEvidence("internal instrumentation is not ready")
        _nonempty(receipt.get("source_identity_reference"), "source_identity_reference")
        drain = receipt.get("frontend_drain_evidence")
        if not isinstance(drain, dict) or drain.get("kind") not in (
            "request_terminal_receipts",
            "task_stream_snapshot",
            "controlled_cleanup",
        ):
            raise InvalidEvidence(
                "need frontend terminal/task/cleanup evidence; old logs or no rid are insufficient"
            )
        _nonempty(drain.get("reference"), "frontend_drain_evidence.reference")
        _hash(drain.get("sha256"), "frontend_drain_evidence.sha256")

    def run(self, input_ids: list[int], *, cycles: int) -> dict[str, Any]:
        if self.started:
            raise InvalidEvidence("Supervisor instances cannot restart or resume")
        self.started = True
        self.started_at = self.clock()
        self.deadline = self.started_at + self.limits.wall_seconds
        status = "INCOMPLETE"
        reason = None
        try:
            self._claim_window()
            _integer(cycles, "cycles", minimum=1)
            if cycles * 3 > self.limits.max_requests:
                raise InvalidEvidence(
                    "requested cold/hot/hot cycles exceed request budget"
                )
            if not isinstance(input_ids, list) or len(input_ids) != SEALED_INPUT_TOKENS:
                raise InvalidEvidence(
                    "sealed input must contain exactly 4000 token IDs"
                )
            for token in input_ids:
                if _integer(token, "input token", minimum=0) >= self.limits.vocab_size:
                    raise InvalidEvidence("input token outside vocabulary")
            digest = hashlib.sha256(json.dumps(input_ids).encode()).hexdigest()
            if digest != self.sealed_input_sha256:
                raise InvalidEvidence("sealed original input SHA256 mismatch")
            self._save("SEALED_INPUT.json", {"input_ids": input_ids, "sha256": digest})
            self._event(
                "run_start",
                limits=asdict(self.limits),
                ruling=self.authorization_reference,
                rules=self.execution_rules_reference,
            )
            receipt = self.preflight(
                self.run_id, self._remaining(self.limits.control_timeout_seconds)
            )
            self._save("PREFLIGHT.json", receipt)
            self._remaining(self.limits.control_timeout_seconds)
            self._check_preflight(receipt)
            for cycle in range(1, cycles + 1):
                flush = self._request("/flush_cache", {}, generate=False)
                valid, detail = flush_verdict(
                    flush, expected_workers=self.limits.expected_workers
                )
                if not valid:
                    raise InvalidEvidence(f"flush INVALID: {detail}")
                for label, cold in (
                    ("A-cold", True),
                    ("B-hot", False),
                    ("C-hot", False),
                ):
                    rid = f"{self.run_id}-c{cycle}-{label}"
                    body = {
                        "rid": rid,
                        "input_ids": input_ids,
                        "sampling_params": {
                            "temperature": 0.0,
                            "max_new_tokens": self.limits.max_new_tokens,
                        },
                        "return_logprob": True,
                        "top_logprobs_num": 0,
                    }
                    response = self._request("/generate", body, generate=True)
                    output = validate_generation(
                        response, rid=rid, cold=cold, limits=self.limits
                    )
                    evidence = self.trace_reader(
                        self.run_id,
                        rid,
                        self._remaining(self.limits.trace_timeout_seconds),
                    )
                    self._save(f"CAPTURE_c{cycle}_{label}.json", evidence)
                    self._remaining(self.limits.trace_timeout_seconds)
                    validate_capture_receipt(
                        evidence,
                        run_id=self.run_id,
                        rid=rid,
                        output=output,
                        diagnostic_source_sha256=self.diagnostic_source_sha256,
                        sealed_input_sha256=self.sealed_input_sha256,
                        limits=self.limits,
                    )
                    self.outputs.append({"cycle": cycle, "label": label, **output})
                    self._event(
                        "request_evidence_complete", rid=rid, numeric_review="PENDING"
                    )
            status = "CAPTURE_RECEIPTS_COLLECTED"
        except Exception as error:
            self.halted = True
            reason = f"{type(error).__name__}: {error}"
            # Any error ends this run. No retry, abort, new probe, or next cycle.
            self._event("admission_stopped", reason=reason)
        result = {
            "run_id": self.run_id,
            "status": status,
            "stop_reason": reason,
            "generate_attempts": self.generated,
            "outputs": self.outputs,
            "elapsed_seconds": self.clock() - self.started_at,
            "numeric_review": "PENDING",
            "root_cause": "UNDETERMINED",
            "p7_contract": "OPEN",
            "c5_execution_authorized": False,
        }
        self._save("VERDICT.json", result)
        return result


def preparation_status() -> dict[str, Any]:
    return {
        "status": "PREPARATION_ONLY",
        "live_entrypoint_available": False,
        "implemented": [
            "strict_POST_flush_gate",
            "descriptive_cross_run_logprob_delta",
            "serial_stop_before_next_request",
            "durable_attempt_ledger",
            "shared_wall_deadline",
            "internal_capture_receipt_association",
        ],
        "pending": [
            "source_bound_instrumentation",
            "reviewed_transport_and_preflight_adapter",
            "immutable_tensor_byte_validation",
            "owner_rank_numeric_analysis",
        ],
        "ceilings": {
            "generate_submissions": 12,
            "output_tokens_per_request": 5,
            "wall_seconds": 1800,
        },
        "automatic_retries": 0,
        "abort_calls": 0,
        "c5_execution_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--plan", action="store_true", help="show preparation status; no service calls"
    )
    mode.add_argument(
        "--selftest", action="store_true", help="run offline regression tests"
    )
    args = parser.parse_args()
    if args.plan:
        print(json.dumps(preparation_status(), ensure_ascii=False, indent=2))
        return 0
    suite = unittest.defaultTestLoader.discover(
        str(Path(__file__).parent), pattern="test_c4_followup_r6_driver.py"
    )
    if suite.countTestCases() == 0:
        print("No r6 regression tests found; self-test cannot pass.", file=sys.stderr)
        return 2
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 2


if __name__ == "__main__":
    sys.exit(main())
