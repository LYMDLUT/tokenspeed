"""Run exact frontend bodies without importing GPU-only startup modules.

These are source-boundary/ASGI CPU tests, not full-runtime or gateway tests.
The production bodies and the repository regression tests are compiled intact.
Only AsyncLLM's unused base classes and unrelated import-time initialization
are omitted. Message types, collector and request state are production code.
"""

from __future__ import annotations

import ast
import asyncio
import dataclasses
import hashlib
import hmac
import importlib.util
import json
import logging
import os
import sys
import time
import types
import unittest
import uuid
from collections.abc import Awaitable, Callable
from http import HTTPStatus
from pathlib import Path
from typing import Any
from unittest import mock

import msgspec
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

ROOT = Path(os.environ["TOKENSPEED_CPU_TEST_ROOT"])
EVIDENCE = []
logger = logging.getLogger(__name__)


def load_leaf(rel, module_name):
    path = ROOT / rel
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    EVIDENCE.append(
        {
            "path": rel,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "whole_leaf": True,
        }
    )
    return module


def extract(rel, names, *, selected_methods=None, omit_bases=()):
    path = ROOT / rel
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = []
    found = set()
    for node in tree.body:
        if (
            not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            or node.name not in names
        ):
            continue
        found.add(node.name)
        EVIDENCE.append(
            {
                "path": rel,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "symbol": node.name,
                "line_range": [node.lineno, node.end_lineno],
            }
        )
        if isinstance(node, ast.ClassDef):
            if node.name in omit_bases:
                node.bases = []
            if selected_methods is not None and node.name in selected_methods:
                wanted = set(selected_methods[node.name])
                node.body = [
                    m
                    for m in node.body
                    if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and m.name in wanted
                ]
                assert {m.name for m in node.body} == wanted
        nodes.append(node)
    assert found == set(names), (rel, found, names)
    mod = ast.Module(
        body=[ast.parse("from __future__ import annotations").body[0], *nodes],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(mod), str(path), "exec"), globals())


RequestOutputCollector = load_leaf(
    "python/tokenspeed/runtime/engine/collector.py", "frontend_collector_under_test"
).RequestOutputCollector
EngineGenerateError = load_leaf(
    "python/tokenspeed/runtime/engine/exceptions.py", "frontend_exceptions_under_test"
).EngineGenerateError
LogprobsProcessor = load_leaf(
    "python/tokenspeed/runtime/engine/logprobs.py", "frontend_logprobs_under_test"
).LogprobsProcessor
extract(
    "python/tokenspeed/runtime/engine/io_struct.py",
    {"BaseReq", "BaseBatchReq", "AbortReq", "FlushCacheReqOutput", "BatchEmbeddingOut"},
)
extract(
    "python/tokenspeed/runtime/engine/output_processor.py",
    {"ReqState", "OutputProcessor"},
)
extract(
    "python/tokenspeed/runtime/engine/async_llm.py",
    {"AsyncLLM"},
    selected_methods={"AsyncLLM": {"_wait_one_response", "abort_request"}},
    omit_bases={"AsyncLLM"},
)
router = APIRouter()
extract("python/tokenspeed/runtime/entrypoints/rl_control.py", {"install_bearer_auth"})
extract(
    "python/tokenspeed/runtime/entrypoints/sglang_compat_http.py",
    {
        "_llm",
        "_guarded",
        "_json_body",
        "flush_cache",
        "abort_request",
        "build_sglang_compat_app",
    },
)
SimpleNamespace = types.SimpleNamespace
extract("test/runtime/rl_fakes.py", {"FakeLLM"})
extract(
    "test/runtime/test_async_llm_cancel.py",
    {
        "_FakeScheduler",
        "_StubAsyncLLM",
        "_StubReqObj",
        "_fresh_state",
        "TestWaitOneResponseCancellation",
        "TestExplicitAbort",
    },
)
extract(
    "test/runtime/test_rl_control_endpoint.py",
    {"TestRouteSemantics"},
    selected_methods={
        "TestRouteSemantics": {
            "_client",
            "test_flush_cache_accepts_get_and_post",
            "test_flush_cache_propagates_scheduler_refusal",
            "test_flush_cache_surfaces_backend_errors",
        }
    },
)


class TestAbortRouteWaiter(unittest.IsolatedAsyncioTestCase):
    async def test_http_abort_ends_same_loop_waiter(self):
        from httpx import ASGITransport, AsyncClient

        mgr = _StubAsyncLLM()
        mgr.server_args.rl_control_api_key = None
        obj = _StubReqObj(rid="http-abort", stream=False)
        mgr.rid_to_state[obj.rid] = _fresh_state(obj)
        gen = mgr._wait_one_response(obj)

        async def drive():
            return [item async for item in gen]

        task = asyncio.create_task(drive())
        try:
            await asyncio.sleep(0)
            async with AsyncClient(
                transport=ASGITransport(app=build_sglang_compat_app(mgr)),
                base_url="http://test",
            ) as client:
                response = await client.post("/abort_request", json={"rid": obj.rid})
            self.assertEqual(response.status_code, 200)
            done, _ = await asyncio.wait({task}, timeout=1)
            self.assertIn(task, done)
            with self.assertRaisesRegex(EngineGenerateError, "aborted"):
                task.result()
            self.assertNotIn(obj.rid, mgr.rid_to_state)
        finally:
            if not task.done():
                task.cancel()
            try:
                await task
            except (asyncio.CancelledError, EngineGenerateError):
                pass
            await gen.aclose()


out = os.environ.get("TOKENSPEED_CPU_TEST_EVIDENCE")
if out:
    Path(out).write_text(
        json.dumps(
            {
                "scope": __doc__,
                "source_bindings": EVIDENCE,
                "gpu_modules_imported": "torch" in sys.modules,
            },
            indent=2,
        )
    )
