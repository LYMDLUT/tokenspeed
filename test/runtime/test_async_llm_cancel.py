# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Regression guard for ``AsyncLLM._wait_one_response`` cancellation.

Locks the contract Phase G.1 established: when the task driving a
streaming generator is cancelled (e.g. FastAPI cancelling its route
coroutine because the client disconnected), the generator's
``finally`` drops the rid from ``rid_to_state`` and fires exactly one
``AbortReq`` at the scheduler. No dangling per-request state. No
reliance on ``fastapi.Request.is_disconnected()``.

This test uses a stub that bypasses ZMQ / ModelConfig / HF tokenizer
bring-up — the same pattern as
``test/runtime/test_inline_detokenizer_receiver.py``.
"""

import os
import sys
import unittest

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=20, suite="runtime-1gpu")

import asyncio  # noqa: E402
import types  # noqa: E402
from typing import Any  # noqa: E402
from unittest import mock  # noqa: E402

from tokenspeed.runtime.engine.async_llm import AsyncLLM  # noqa: E402
from tokenspeed.runtime.engine.collector import RequestOutputCollector  # noqa: E402
from tokenspeed.runtime.engine.exceptions import EngineGenerateError  # noqa: E402
from tokenspeed.runtime.engine.io_struct import BatchEmbeddingOut  # noqa: E402
from tokenspeed.runtime.engine.output_processor import (  # noqa: E402
    OutputProcessor,
    ReqState,
)


class _FakeScheduler:
    """Captures ``AbortReq``s the stub AsyncLLM pushes on cancellation."""

    def __init__(self) -> None:
        self.aborts: list = []

    def send_pyobj(self, obj: Any) -> None:
        self.aborts.append(obj)


class _StubAsyncLLM(AsyncLLM):
    """Bypass ZMQ + ModelConfig + HF bring-up.

    Populates only the attributes ``_wait_one_response`` + ``abort_request``
    read: ``rid_to_state``, ``log_requests``, ``enable_metrics``,
    ``tokenizer``, ``model_config``, and an ``engine_core_client`` whose
    ``send_to_scheduler`` is a capture fake so the test can assert the
    ``AbortReq`` was sent.
    """

    def __init__(self) -> None:
        self.rid_to_state: dict[str, ReqState] = {}
        self.log_requests = False
        self.enable_metrics = False
        self.tokenizer = None
        self.dump_requests_folder = None
        self.server_args = types.SimpleNamespace(
            weight_version="test", speculative_algorithm=None
        )
        self.model_config = types.SimpleNamespace(is_multimodal_gen=False)
        self.engine_core_client = types.SimpleNamespace(
            send_to_scheduler=_FakeScheduler()
        )


class _StubReqObj:
    """Minimal stand-in for ``GenerateReqInput`` used by
    ``_wait_one_response``. Only the attributes actually read by the
    generator body are provided.
    """

    def __init__(self, *, rid: str = "r1", stream: bool = True, input_ids=None) -> None:
        self.rid = rid
        self.stream = stream
        self.input_ids = input_ids
        self.text = None
        self.sampling_params = {"skip_special_tokens": False}


def _fresh_state(obj: _StubReqObj) -> ReqState:
    return ReqState(
        RequestOutputCollector(),
        False,
        asyncio.Event(),
        obj,
        created_time=0.0,
    )


class TestWaitOneResponseCancellation(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_before_first_output_cleans_up(self) -> None:
        """Cancel while ``_wait_one_response`` is blocked on the first
        event. Expect: rid dropped from ``rid_to_state``; one ``AbortReq``
        sent to the scheduler.
        """
        mgr = _StubAsyncLLM()
        obj = _StubReqObj(rid="r-cancel-1", stream=True)
        state = _fresh_state(obj)
        mgr.rid_to_state[obj.rid] = state

        # Drive the generator via a task so we can cancel it.
        gen = mgr._wait_one_response(obj)

        async def drain() -> None:
            async for _ in gen:
                pass

        task = asyncio.create_task(drain())

        # Give the generator one event-loop tick to enter
        # ``await state.event.wait()``.
        await asyncio.sleep(0)

        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        # Finally block cleanup.
        self.assertNotIn(obj.rid, mgr.rid_to_state)
        aborts = mgr.engine_core_client.send_to_scheduler.aborts
        self.assertEqual(
            len(aborts),
            1,
            f"expected exactly one AbortReq on cancel, got {len(aborts)}",
        )
        # The AbortReq should carry the right rid.
        self.assertEqual(aborts[0].rid, obj.rid)

    async def test_cancelled_waiter_cleans_up_when_abort_ipc_fails(self) -> None:
        mgr = _StubAsyncLLM()
        obj = _StubReqObj(rid="cancel-ipc-failure", stream=True)
        mgr.rid_to_state[obj.rid] = _fresh_state(obj)
        gen = mgr._wait_one_response(obj)
        self.addAsyncCleanup(gen.aclose)

        async def drain() -> None:
            async for _ in gen:
                pass

        task = asyncio.create_task(drain())
        await asyncio.sleep(0)
        sender = mgr.engine_core_client.send_to_scheduler
        with mock.patch.object(
            sender, "send_pyobj", side_effect=RuntimeError("IPC failed")
        ) as abort:
            task.cancel()
            with self.assertRaisesRegex(RuntimeError, "IPC failed"):
                await task

        abort.assert_called_once()
        self.assertNotIn(obj.rid, mgr.rid_to_state)

    async def test_normal_finish_does_not_fire_abort(self) -> None:
        """When the generator exits on ``state.finished`` (scheduler's
        terminal frame), no ``AbortReq`` should fire — the scheduler
        already knows the request is done.
        """
        mgr = _StubAsyncLLM()
        obj = _StubReqObj(rid="r-finish-1", stream=True)
        state = _fresh_state(obj)
        mgr.rid_to_state[obj.rid] = state

        gen = mgr._wait_one_response(obj)

        async def drive() -> list:
            out = []
            async for chunk in gen:
                out.append(chunk)
            return out

        task = asyncio.create_task(drive())
        await asyncio.sleep(0)

        # Simulate a terminal frame from the scheduler: mark finished,
        # put the final out_dict into the collector, wake the generator.
        state.finished = True
        state.collector.put(
            {
                "text": "hello",
                "output_ids": [1, 2],
                "meta_info": {"id": obj.rid, "finish_reason": None},
            },
            stream=True,
            cumulative=False,
        )
        state.event.set()

        results = await task

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["text"], "hello")
        self.assertNotIn(obj.rid, mgr.rid_to_state)
        # Normal finish ⇒ no abort.
        self.assertEqual(
            len(mgr.engine_core_client.send_to_scheduler.aborts),
            0,
            "normal finish should not schedule an AbortReq",
        )


class TestExplicitAbort(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.mgr = _StubAsyncLLM()

    def _request(self, *, stream: bool):
        obj = _StubReqObj(rid="explicit-abort", stream=stream)
        state = _fresh_state(obj)
        self.mgr.rid_to_state[obj.rid] = state
        gen = self.mgr._wait_one_response(obj)
        self.addAsyncCleanup(gen.aclose)
        return obj, state, gen

    @staticmethod
    async def _drain(gen) -> list:
        return [chunk async for chunk in gen]

    async def _cancel_task(self, task) -> None:
        if not task.done():
            task.cancel()
        try:
            await task
        except (asyncio.CancelledError, EngineGenerateError):
            pass

    def _drive(self, gen):
        task = asyncio.create_task(self._drain(gen))
        self.addAsyncCleanup(self._cancel_task, task)
        return task

    async def _assert_aborted(self, task, rid: str) -> None:
        # wait() does not cancel the producer on timeout: a broken implementation
        # must fail for the hanging task, not pass through test-side cancellation.
        done, _ = await asyncio.wait({task}, timeout=1.0)
        self.assertIn(task, done, "explicit abort left the frontend waiter pending")
        with self.assertRaisesRegex(EngineGenerateError, "aborted"):
            task.result()
        self.assertNotIn(rid, self.mgr.rid_to_state)
        self.assertEqual(
            [req.rid for req in self.mgr.engine_core_client.send_to_scheduler.aborts],
            [rid],
        )

    async def test_abort_wakes_waiting_streaming_request(self) -> None:
        obj, _state, gen = self._request(stream=True)
        task = self._drive(gen)
        await asyncio.sleep(0)
        self.mgr.abort_request(obj.rid)
        await self._assert_aborted(task, obj.rid)

    async def test_abort_wakes_waiting_nonstreaming_request(self) -> None:
        obj, _state, gen = self._request(stream=False)
        task = self._drive(gen)
        await asyncio.sleep(0)
        self.mgr.abort_request(obj.rid)
        await self._assert_aborted(task, obj.rid)

    async def test_abort_before_first_iterator_resume(self) -> None:
        obj, _state, gen = self._request(stream=False)
        self.mgr.abort_request(obj.rid)
        await self._assert_aborted(self._drive(gen), obj.rid)

    async def test_repeated_abort_and_unknown_id_are_idempotent(self) -> None:
        obj, _state, gen = self._request(stream=True)
        task = self._drive(gen)
        await asyncio.sleep(0)
        self.mgr.abort_request("unknown")
        self.mgr.abort_request(obj.rid)
        self.mgr.abort_request(obj.rid)
        await self._assert_aborted(task, obj.rid)
        self.mgr.abort_request(obj.rid)
        self.assertEqual(len(self.mgr.engine_core_client.send_to_scheduler.aborts), 1)

    async def test_abort_supersedes_pending_nonterminal_output(self) -> None:
        obj, state, gen = self._request(stream=False)
        task = self._drive(gen)
        await asyncio.sleep(0)
        state.collector.put(
            {"text": "partial", "output_ids": [1], "meta_info": {"id": obj.rid}},
            stream=False,
            cumulative=False,
        )
        state.event.set()
        self.mgr.abort_request(obj.rid)
        await self._assert_aborted(task, obj.rid)

    async def test_abort_between_streamed_chunks(self) -> None:
        obj, state, gen = self._request(stream=True)
        state.collector.put(
            {"text": "partial", "output_ids": [1], "meta_info": {"id": obj.rid}},
            stream=True,
            cumulative=False,
        )
        state.event.set()
        chunk = await asyncio.wait_for(anext(gen), timeout=1.0)
        self.assertEqual(chunk["output_ids"], [1])
        task = self._drive(gen)
        await asyncio.sleep(0)
        self.mgr.abort_request(obj.rid)
        await self._assert_aborted(task, obj.rid)

    async def test_natural_finish_already_received_wins(self) -> None:
        obj, state, gen = self._request(stream=True)
        task = self._drive(gen)
        await asyncio.sleep(0)
        out = {
            "text": "done",
            "meta_info": {"id": obj.rid, "finish_reason": {"type": "length"}},
        }
        state.finished = True
        state.collector.put(out, stream=True, cumulative=False)
        state.event.set()
        self.mgr.abort_request(obj.rid)
        self.assertEqual(await asyncio.wait_for(task, timeout=1.0), [out])
        self.assertNotIn(obj.rid, self.mgr.rid_to_state)
        self.assertEqual(self.mgr.engine_core_client.send_to_scheduler.aborts, [])

    async def test_old_generator_cleanup_preserves_new_same_id_state(self) -> None:
        obj, state, gen = self._request(stream=True)
        state.finished = True
        state.collector.put(
            {"text": "done", "meta_info": {"id": obj.rid}},
            stream=True,
            cumulative=False,
        )
        state.event.set()
        await asyncio.wait_for(anext(gen), timeout=1.0)
        replacement = _fresh_state(obj)
        self.mgr.rid_to_state[obj.rid] = replacement
        await gen.aclose()
        self.assertIs(self.mgr.rid_to_state[obj.rid], replacement)
        self.assertEqual(self.mgr.engine_core_client.send_to_scheduler.aborts, [])
        self.mgr.rid_to_state.clear()

    async def test_disconnect_after_explicit_abort_sends_no_second_abort(self) -> None:
        obj, _state, gen = self._request(stream=True)
        task = self._drive(gen)
        await asyncio.sleep(0)
        self.mgr.abort_request(obj.rid)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertNotIn(obj.rid, self.mgr.rid_to_state)
        self.assertEqual(len(self.mgr.engine_core_client.send_to_scheduler.aborts), 1)

    async def test_ipc_failure_does_not_claim_successful_abort(self) -> None:
        obj, state, gen = self._request(stream=True)
        task = self._drive(gen)
        await asyncio.sleep(0)
        sender = self.mgr.engine_core_client.send_to_scheduler
        with mock.patch.object(
            sender, "send_pyobj", side_effect=RuntimeError("IPC failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "IPC failed"):
                self.mgr.abort_request(obj.rid)
        self.assertIs(self.mgr.rid_to_state[obj.rid], state)
        self.assertFalse(state.event.is_set())
        self.mgr.abort_request(obj.rid)
        await self._assert_aborted(task, obj.rid)

    async def test_late_output_does_not_revive_aborted_request(self) -> None:
        obj, state, gen = self._request(stream=False)
        task = self._drive(gen)
        await asyncio.sleep(0)
        self.mgr.abort_request(obj.rid)
        OutputProcessor(self.mgr).handle_batch_output(
            BatchEmbeddingOut(
                rids=[obj.rid],
                finished_reasons=[{"type": "length"}],
                embeddings=[[1.0]],
                prompt_tokens=[1],
            )
        )
        self.assertFalse(state.collector.has_pending())
        self.assertFalse(state.finished)
        await self._assert_aborted(task, obj.rid)


if __name__ == "__main__":
    unittest.main(verbosity=2)
