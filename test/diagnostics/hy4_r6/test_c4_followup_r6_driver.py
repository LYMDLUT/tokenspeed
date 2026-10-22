"""Offline regression tests. No transport in this file opens a connection."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import c4_followup_r6_driver as r6

SYNTHETIC_INPUT_SHA256 = hashlib.sha256(json.dumps([1] * 4000).encode()).hexdigest()


def response(body, *, status=200, exc=None):
    return {"status": status, "exc": exc, "body_text": json.dumps(body)}


def good_flush():
    return {"status": "success", "workers_flushed": 1, "total_workers": 1}


class Clock:
    def __init__(self):
        self.value = 1000.0

    def __call__(self):
        return self.value


class SyntheticService:
    """Fake metadata is intentionally not proof of actual tensor correctness."""

    def __init__(self, clock):
        self.clock = clock
        self.calls = []
        self.trace_calls = []
        self.preflight_calls = []
        self.generated = []
        self.failure = None
        self.failure_at = 1
        self.flush_body = good_flush()
        self.preflight_delay = 0.0
        self.flush_delay = 0.0
        self.generate_delay = 0.0
        self.trace_edit = None
        self.preflight_edit = None
        self.ids = [8086, 2, 3, 4, 16]

    def preflight(self, run_id, timeout):
        self.preflight_calls.append((run_id, timeout))
        self.clock.value += self.preflight_delay
        receipt = {
            "run_id": run_id,
            "expected_workers": 1,
            "diagnostic_source_sha256": "a" * 64,
            "instrumentation_ready": True,
            "source_identity_reference": "synthetic-fixture-only",
            "frontend_drain_evidence": {
                "kind": "task_stream_snapshot",
                "reference": "synthetic-fixture-only",
                "sha256": "b" * 64,
            },
        }
        if self.preflight_edit:
            self.preflight_edit(receipt)
        return receipt

    def transport(self, method, path, body, timeout):
        self.calls.append((method, path, copy.deepcopy(body), timeout))
        if path == "/flush_cache":
            self.clock.value += self.flush_delay
            return response(self.flush_body)
        if path != "/generate":
            raise AssertionError(f"unexpected endpoint {path}")
        self.generated.append(body["rid"])
        self.clock.value += self.generate_delay
        failing = len(self.generated) == self.failure_at
        if failing and self.failure == "timeout":
            raise TimeoutError("synthetic timeout")
        if failing and self.failure == "non_json":
            return {
                "status": 200,
                "exc": None,
                "body_text": "<html>upstream failed</html>",
            }
        ids = list(self.ids)
        payload = {
            "output_ids": ids,
            "text": "synthetic output",
            "meta_info": {
                "id": body["rid"],
                "prompt_tokens": 4000,
                "completion_tokens": 5,
                "cached_tokens": 0 if body["rid"].endswith("A-cold") else 3968,
                "finish_reason": {"type": "length", "length": 5},
                "output_token_logprobs": [[-0.1, token] for token in ids],
            },
        }
        if failing:
            if self.failure == "bad_cache":
                payload["meta_info"]["cached_tokens"] = 3968
            if self.failure == "rid_mismatch":
                payload["meta_info"]["id"] = "another-request"
            if self.failure == "unterminated":
                payload["meta_info"].pop("finish_reason")
            if self.failure == "too_many_tokens":
                payload["output_ids"].append(99)
            if self.failure == "logprob_id_mismatch":
                payload["meta_info"]["output_token_logprobs"][0][1] = 99
            if self.failure == "nan_logprob":
                payload["meta_info"]["output_token_logprobs"][0][0] = float("nan")
            if self.failure == "infinite_logprob":
                payload["meta_info"]["output_token_logprobs"][0][0] = 1e309
        return response(payload)

    def trace(self, run_id, rid, timeout):
        self.trace_calls.append((run_id, rid, timeout))
        if self.failure == "trace_missing" and len(self.generated) == self.failure_at:
            return None
        captures = []
        for position in (0, 4):
            captures.append(
                {
                    "output_position": position,
                    "forward_id": f"{rid}-forward-{position}",
                    "prompt_ids_sha256": SYNTHETIC_INPUT_SHA256,
                    "generated_prefix_ids": self.ids[:position],
                    "model_input_tokens": 4000 + position,
                    "forward_kind": "prefill" if position == 0 else "decode",
                    "post_broadcast_token_id": self.ids[position],
                    "pre_broadcast_token_id": self.ids[position],
                    "sampler_owner_rank": 0,
                    "capture_rank": 0,
                    "request_pool_row": 2,
                    "global_logits_row": 0,
                    "execution_mode": "graph",
                    "global_vocab_size": 120000,
                    "independent_copy": True,
                    "snapshot_path": f"synthetic-not-real/{rid}-{position}.npy",
                    "snapshot_sha256": "c" * 64,
                    "synchronization_boundary": "synthetic-fixture-only",
                    "logprob_semantics": "untransformed_model_log_softmax",
                    "sampler_backend": "synthetic-triton-top-k-one",
                    "effective_sampling_params": {"top_k": 1, "temperature": 1.0},
                    "physical_bucket": {"batch": 1},
                    "slot_metadata": {"slot": 2},
                }
            )
        receipt = {
            "run_id": run_id,
            "rid": rid,
            "diagnostic_source_sha256": "a" * 64,
            "captures": captures,
        }
        if self.trace_edit:
            self.trace_edit(receipt)
        return receipt


class FlushTests(unittest.TestCase):
    def test_only_explicit_complete_aggregation_succeeds(self):
        self.assertTrue(r6.flush_verdict(response(good_flush()), expected_workers=1)[0])
        aggregated_success_body = {
            "status": "success",
            "message": "Successfully flushed cache on all 1 workers",
            "workers_flushed": 1,
            "total_http_workers": 0,
            "total_grpc_workers": 1,
            "total_zmq_workers_skipped": 0,
            "total_workers": 1,
        }
        self.assertTrue(
            r6.flush_verdict(response(aggregated_success_body), expected_workers=1)[0]
        )
        body = {
            **good_flush(),
            "message": "previous request failed; this flush succeeded",
        }
        self.assertTrue(r6.flush_verdict(response(body), expected_workers=1)[0])

    def test_original_false_success_regressions(self):
        cases = [
            {},
            {"status": "failure", "workers_flushed": 0, "total_workers": 1},
            {"success": True, "message": "Cache flushed."},
            {**good_flush(), "success": False},
            {**good_flush(), "workers_failed": 1},
            {**good_flush(), "total_http_workers": 1, "total_grpc_workers": 1},
            {**good_flush(), "total_zmq_workers_skipped": 1},
        ]
        for body in cases:
            with self.subTest(body=body):
                self.assertFalse(
                    r6.flush_verdict(response(body), expected_workers=1)[0]
                )

    def test_counts_are_strict_positive_integers_and_topology_bound(self):
        for field in ("workers_flushed", "total_workers"):
            for value in (
                None,
                True,
                False,
                1.0,
                "1",
                0,
                -1,
                2,
                float("nan"),
                float("inf"),
            ):
                with self.subTest(field=field, value=value):
                    self.assertFalse(
                        r6.flush_verdict(
                            response({**good_flush(), field: value}), expected_workers=1
                        )[0]
                    )
        self.assertFalse(
            r6.flush_verdict(response(good_flush()), expected_workers=32)[0]
        )
        for value in (True, 1.0, 0):
            self.assertFalse(
                r6.flush_verdict(response(good_flush()), expected_workers=value)[0]
            )

    def test_transport_errors_and_partial_success_are_invalid(self):
        for status in (206, 500, True, None):
            self.assertFalse(
                r6.flush_verdict(
                    response(good_flush(), status=status), expected_workers=1
                )[0]
            )
        self.assertFalse(
            r6.flush_verdict(response(good_flush(), exc="timeout"), expected_workers=1)[
                0
            ]
        )
        self.assertFalse(
            r6.flush_verdict(
                {"status": 200, "body_text": "not-json"}, expected_workers=1
            )[0]
        )


class DifferenceTests(unittest.TestCase):
    def test_identical_selected_logprobs_do_not_establish_near_tie(self):
        result = r6.first_divergence([16], [1411], [-0.01005], [-0.01005])
        self.assertEqual(result["cross_run_selected_logprob_abs_delta"], 0)
        self.assertEqual(result["root_cause"], "UNDETERMINED")
        self.assertNotIn("near_tie_hint", result)
        self.assertEqual(
            result["delta_semantics"], "different_forward_selected_logprobs_only"
        )

    def test_prefix_and_equal_sequence_cases(self):
        self.assertEqual(r6.first_divergence([1, 2], [1, 3])["common_prefix_len"], 1)
        self.assertIsNone(r6.first_divergence([1, 2], [1, 2])["output_position"])
        self.assertIsNone(r6.first_divergence([1], [1, 2])["token_id_A"])


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.ids = [1] * 4000
        self.clock = Clock()
        self.fake = SyntheticService(self.clock)
        self.window_index = 0
        self.limits = r6.Limits(
            expected_workers=1,
            vocab_size=120000,
            max_requests=12,
            max_new_tokens=5,
            wall_seconds=1800,
            request_timeout_seconds=30,
            control_timeout_seconds=10,
            trace_timeout_seconds=10,
        )

    def supervisor(self):
        self.window_index += 1
        return r6.Supervisor(
            limits=self.limits,
            output_root=Path(self.temp.name) / f"window-{self.window_index}",
            transport=self.fake.transport,
            trace_reader=self.fake.trace,
            preflight=self.fake.preflight,
            clock=self.clock,
            sealed_input_sha256=SYNTHETIC_INPUT_SHA256,
            authorization_reference="synthetic-authorization",
            execution_rules_reference="synthetic-execution-rules",
        )

    def test_success_records_receipts_but_does_not_accept_numerics_or_c5(self):
        supervisor = self.supervisor()
        result = supervisor.run(self.ids, cycles=1)
        self.assertEqual(result["status"], "CAPTURE_RECEIPTS_COLLECTED")
        self.assertEqual(len(self.fake.generated), 3)
        self.assertEqual(result["generate_attempts"], 3)
        self.assertEqual(result["numeric_review"], "PENDING")
        self.assertFalse(result["c5_execution_authorized"])
        self.assertEqual(result["p7_contract"], "OPEN")
        self.assertEqual(
            result, json.loads((supervisor.directory / "VERDICT.json").read_text())
        )
        self.assertEqual(len(list(supervisor.directory.glob("CAPTURE_*.json"))), 3)
        self.assertEqual(
            [call[1] for call in self.fake.calls],
            ["/flush_cache", "/generate", "/generate", "/generate"],
        )

    def test_every_failure_on_A_stops_before_B_or_C_without_retry(self):
        for failure in (
            "timeout",
            "non_json",
            "bad_cache",
            "rid_mismatch",
            "unterminated",
            "too_many_tokens",
            "logprob_id_mismatch",
            "nan_logprob",
            "infinite_logprob",
            "trace_missing",
        ):
            with self.subTest(failure=failure):
                self.fake = SyntheticService(self.clock)
                self.fake.failure = failure
                supervisor = self.supervisor()
                result = supervisor.run(self.ids, cycles=3)
                self.assertEqual(result["status"], "INCOMPLETE")
                self.assertEqual(len(self.fake.generated), 1)
                self.assertEqual(result["generate_attempts"], 1)
                self.assertEqual(len(self.fake.calls), 2)
                self.assertTrue((supervisor.directory / "VERDICT.json").exists())
                events = [
                    json.loads(path.read_text())
                    for path in supervisor.directory.glob("event_*.json")
                ]
                self.assertTrue(
                    any(
                        event["kind"] == "dispatch_intent"
                        and event["path"] == "/generate"
                        for event in events
                    )
                )
                self.assertTrue(
                    any(event["kind"] == "admission_stopped" for event in events)
                )

    def test_failure_on_B_stops_before_C_and_next_cycle(self):
        self.fake.failure = "timeout"
        self.fake.failure_at = 2
        result = self.supervisor().run(self.ids, cycles=3)
        self.assertEqual(result["generate_attempts"], 2)
        self.assertEqual(len(self.fake.generated), 2)
        self.assertEqual(len(result["outputs"]), 1)

    def test_invalid_flush_stops_before_first_generate_without_retry(self):
        self.fake.flush_body = {
            "status": "partial_success",
            "workers_flushed": 0,
            "total_workers": 1,
        }
        result = self.supervisor().run(self.ids, cycles=3)
        self.assertEqual(result["generate_attempts"], 0)
        self.assertEqual(len(self.fake.calls), 1)

    def test_all_generate_submissions_are_counted_and_bounded_at_twelve(self):
        result = self.supervisor().run(self.ids, cycles=4)
        self.assertEqual(result["generate_attempts"], 12)
        self.assertEqual(len(self.fake.generated), 12)
        self.fake = SyntheticService(self.clock)
        result = self.supervisor().run(self.ids, cycles=5)
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(self.fake.calls, [])
        self.assertEqual(self.fake.preflight_calls, [])

    def test_preflight_consumes_same_deadline(self):
        self.fake.preflight_delay = 1801
        result = self.supervisor().run(self.ids, cycles=1)
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(self.fake.calls, [])
        self.assertIn("BudgetExhausted", result["stop_reason"])

    def test_every_call_timeout_is_clamped_to_remaining_window(self):
        self.fake.preflight_delay = 1795
        self.fake.flush_delay = 2
        self.fake.generate_delay = 4
        result = self.supervisor().run(self.ids, cycles=1)
        self.assertEqual([call[3] for call in self.fake.calls], [5, 3])
        self.assertEqual(result["generate_attempts"], 1)
        self.assertEqual(self.fake.trace_calls, [])

    def test_restart_cannot_reset_request_or_time_budget(self):
        supervisor = self.supervisor()
        supervisor.run(self.ids, cycles=1)
        calls = len(self.fake.calls)
        with self.assertRaises(r6.InvalidEvidence):
            supervisor.run(self.ids, cycles=1)
        self.assertEqual(len(self.fake.calls), calls)

    def test_new_instance_cannot_reset_an_already_claimed_window(self):
        first = self.supervisor()
        first.run(self.ids, cycles=1)
        calls = len(self.fake.calls)
        second = r6.Supervisor(
            limits=self.limits,
            output_root=first.output_root,
            transport=self.fake.transport,
            trace_reader=self.fake.trace,
            preflight=self.fake.preflight,
            clock=self.clock,
            sealed_input_sha256=SYNTHETIC_INPUT_SHA256,
            authorization_reference="synthetic-authorization",
            execution_rules_reference="synthetic-execution-rules",
        )
        result = second.run(self.ids, cycles=1)
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertIn("already claimed", result["stop_reason"])
        self.assertEqual(len(self.fake.calls), calls)

    def test_invalid_sealed_prompt_is_rejected_before_preflight_or_http(self):
        altered = list(self.ids)
        altered[-1] = 2
        result = self.supervisor().run(altered, cycles=1)
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(self.fake.preflight_calls, [])
        self.assertEqual(self.fake.calls, [])

    def test_weak_drain_claim_or_unready_hooks_cannot_start(self):
        edits = [
            lambda receipt: receipt.update(instrumentation_ready=False),
            lambda receipt: receipt["frontend_drain_evidence"].update(
                kind="no_new_logs"
            ),
            lambda receipt: receipt.update(expected_workers=True),
        ]
        for edit in edits:
            with self.subTest(edit=edit):
                self.fake = SyntheticService(self.clock)
                self.fake.preflight_edit = edit
                result = self.supervisor().run(self.ids, cycles=1)
                self.assertEqual(result["status"], "INCOMPLETE")
                self.assertEqual(self.fake.calls, [])

    def test_trace_identity_prefix_owner_and_copy_requirements(self):
        edits = [
            lambda receipt: receipt.update(rid="another-rid"),
            lambda receipt: receipt["captures"][1].update(
                generated_prefix_ids=[9, 2, 3, 4]
            ),
            lambda receipt: receipt["captures"][1].update(forward_kind="prefill"),
            lambda receipt: receipt["captures"][0].update(capture_rank=1),
            lambda receipt: receipt["captures"][0].update(independent_copy=False),
            lambda receipt: receipt["captures"][0].pop("logprob_semantics"),
            lambda receipt: receipt["captures"][0].update(output_position=False),
        ]
        for edit in edits:
            with self.subTest(edit=edit):
                self.fake = SyntheticService(self.clock)
                self.fake.trace_edit = edit
                result = self.supervisor().run(self.ids, cycles=2)
                self.assertEqual(result["status"], "INCOMPLETE")
                self.assertEqual(len(self.fake.generated), 1)

    def test_capture_forward_prompt_and_snapshot_identity_requirements(self):
        edits = [
            lambda receipt: receipt["captures"][0].pop("forward_id"),
            lambda receipt: receipt["captures"][0].update(forward_id=" "),
            lambda receipt: receipt["captures"][1].update(
                forward_id=receipt["captures"][0]["forward_id"]
            ),
            lambda receipt: receipt["captures"][0].update(prompt_ids_sha256="d" * 64),
            lambda receipt: receipt["captures"][1].update(
                snapshot_path=receipt["captures"][0]["snapshot_path"]
            ),
        ]
        for edit in edits:
            with self.subTest(edit=edit):
                self.fake = SyntheticService(self.clock)
                self.fake.trace_edit = edit
                result = self.supervisor().run(self.ids, cycles=2)
                self.assertEqual(result["status"], "INCOMPLETE")
                self.assertEqual(len(self.fake.generated), 1)

    def test_evidence_write_oserror_stops_admission_before_next_request(self):
        for failed_write in ("raw_response", "capture"):
            with self.subTest(failed_write=failed_write):
                self.fake = SyntheticService(self.clock)
                supervisor = self.supervisor()
                save = supervisor._save

                def fail_selected_write(name, value):
                    raw_response = (
                        name.startswith("event_")
                        and value.get("kind") == "raw_response"
                        and value.get("path") == "/generate"
                    )
                    capture = name.startswith("CAPTURE_")
                    if (failed_write == "raw_response" and raw_response) or (
                        failed_write == "capture" and capture
                    ):
                        raise OSError("synthetic evidence disk failure")
                    save(name, value)

                with patch.object(supervisor, "_save", side_effect=fail_selected_write):
                    result = supervisor.run(self.ids, cycles=2)
                self.assertTrue(supervisor.halted)
                self.assertEqual(result["status"], "INCOMPLETE")
                self.assertIn("OSError", result["stop_reason"])
                self.assertEqual(len(self.fake.generated), 1)
                self.assertEqual(len(self.fake.calls), 2)
                self.assertEqual(result["generate_attempts"], 1)
                self.assertEqual(result["outputs"], [])

    def test_invalid_identity_configuration_is_rejected_before_any_work(self):
        output_root = Path(self.temp.name) / "invalid-configuration"
        baseline = {
            "limits": self.limits,
            "output_root": output_root,
            "transport": self.fake.transport,
            "trace_reader": self.fake.trace,
            "preflight": self.fake.preflight,
            "clock": self.clock,
            "sealed_input_sha256": SYNTHETIC_INPUT_SHA256,
            "authorization_reference": "synthetic-authorization",
            "execution_rules_reference": "synthetic-execution-rules",
        }
        changes = (
            {"sealed_input_sha256": "not-a-sha256"},
            {"authorization_reference": " "},
            {"execution_rules_reference": ""},
        )
        for change in changes:
            with self.subTest(change=change), self.assertRaises(r6.InvalidEvidence):
                r6.Supervisor(**{**baseline, **change})
            self.assertFalse(output_root.exists())
        self.assertEqual(self.fake.calls, [])
        self.assertEqual(self.fake.preflight_calls, [])

    def test_bad_limits_rejected_before_any_work(self):
        baseline = self.limits
        for change in (
            {"max_requests": 13},
            {"max_new_tokens": 6},
            {"wall_seconds": 1801},
            {"max_requests": True},
            {"trace_timeout_seconds": float("nan")},
        ):
            with self.subTest(change=change), self.assertRaises(r6.InvalidEvidence):
                r6.Supervisor(
                    limits=replace(baseline, **change),
                    output_root=Path(self.temp.name),
                    transport=self.fake.transport,
                    trace_reader=self.fake.trace,
                    preflight=self.fake.preflight,
                    clock=self.clock,
                    sealed_input_sha256=SYNTHETIC_INPUT_SHA256,
                    authorization_reference="synthetic-authorization",
                    execution_rules_reference="synthetic-execution-rules",
                )


if __name__ == "__main__":
    unittest.main()
