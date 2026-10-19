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

"""CPU DSA metadata regressions; no GPU scoring or model initialization.

Load leaf helpers directly and extract one production metadata method to avoid
initializing the GPU runtime. Tensor arithmetic and page-table expansion are real.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

# CI discovers this marker without importing the GPU runtime.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def _load_leaf(relative_path):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


UTILS = _load_leaf("python/tokenspeed/runtime/layers/attention/dsa/utils.py")
PAGES = _load_leaf("python/tokenspeed/runtime/layers/attention/page_table.py")


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("strided", [False, True])
def test_workspace_conversion_preserves_shared_logical_indices(dtype, strided):
    base = torch.tensor([[2, -1, 0, -7], [1, 0, -1, 2]], dtype=dtype)
    indices = base[:, ::2] if strided else base
    original = base.clone()
    slots = torch.tensor([640, 129, 300], dtype=torch.int64)
    expected = torch.tensor(
        [[300, -1, 640, -7], [129, 640, -1, 300]], dtype=torch.int32
    )
    if strided:
        expected = expected[:, ::2]
    for _ in range(2):
        actual = UTILS.workspace_indices_to_kv_slots(indices, slots)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(base, original, rtol=0, atol=0)


def _prefill_metadata_method():
    path = ROOT / "python/tokenspeed/runtime/models/glm5.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "GlmMoeDsaAttention"
    )
    method = next(
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_compute_prefill_topk_indices"
    )
    namespace = {
        "torch": torch,
        "GlmDsaPrefillTopK": type("GlmDsaPrefillTopK", (), {}),
        "build_prefill_kv_workspace_slots": PAGES.build_prefill_kv_workspace_slots,
    }
    exec(
        compile(
            ast.Module(
                body=[ast.parse("from __future__ import annotations").body[0], method],
                type_ignores=[],
            ),
            str(path),
            "exec",
        ),
        namespace,
    )
    return namespace[method.name]


@pytest.mark.parametrize("prefix", [0, 2047, 2048, 8192, 131072])
def test_chunk_and_prefix_metadata_uses_logical_causal_lengths(prefix):
    page_size = 64
    prefixes = torch.tensor([prefix, 64], dtype=torch.int64)
    extends = torch.tensor([3, 2], dtype=torch.int64)
    totals = prefixes + extends
    width = (int(totals.max()) + page_size - 1) // page_size
    # Reverse physical pages: logical position must never be inferred from slot ID.
    table = torch.arange(2 * width, 0, -1, dtype=torch.int32).reshape(2, width)
    chunk = SimpleNamespace(
        extend_prefix_lens_cpu=prefixes, extend_seq_lens_cpu=extends, page_table=table
    )
    backend = SimpleNamespace(
        sparse_topk=SimpleNamespace(prefill=None), chunked_prefill_metadata=chunk
    )
    ctx = SimpleNamespace(
        num_extends=2,
        attn_backend=backend,
        token_to_kv_pool=SimpleNamespace(arena=SimpleNamespace(kv_page_size=page_size)),
    )
    owner = SimpleNamespace(
        index_topk=2048, _compute_prefill_topk_indices_portable=lambda **kwargs: kwargs
    )
    result = _prefill_metadata_method()(
        owner, SimpleNamespace(query=torch.empty(5, 2, 128)), ctx, 5
    )
    expected_lengths = torch.tensor(
        [prefix + 1, prefix + 2, prefix + 3, 65, 66], dtype=torch.int64
    )
    torch.testing.assert_close(
        result["candidate_lens_cpu"], expected_lengths, rtol=0, atol=0
    )
    starts = torch.tensor([0, 0, 0, prefix + 3, prefix + 3], dtype=torch.int32)
    torch.testing.assert_close(result["row_starts"], starts, rtol=0, atol=0)
    torch.testing.assert_close(
        result["row_ends"], starts + expected_lengths.int(), rtol=0, atol=0
    )
    assert result["topk"] == 2048
    assert result["max_seq_len"] == int(totals.max())
    assert result["kv_workspace_slots"].numel() == int(totals.sum())
    for req, start in [(0, 0), (1, prefix + 3)]:
        positions = torch.arange(int(totals[req]))
        expected_slots = (
            table[req, positions // page_size].long() * page_size
            + positions % page_size
        )
        torch.testing.assert_close(
            result["kv_workspace_slots"][start : start + positions.numel()],
            expected_slots,
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize(
    "model_file,class_name",
    [("glm5.py", "GlmDsaIndexer"), ("hy_v4.py", "HYV4DsaIndexer")],
)
@pytest.mark.parametrize("tokens", [1, 8193])
def test_fused_indexer_projection_preserves_all_long_input_rows(
    model_file, class_name, tokens
):
    path = ROOT / "python/tokenspeed/runtime/models" / model_file
    tree = ast.parse(path.read_text(), filename=str(path))
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    namespace = {"torch": torch, "GlmDsaIndexerOutput": SimpleNamespace}
    exec(
        compile(
            ast.Module(
                body=[ast.parse("from __future__ import annotations").body[0], method],
                type_ignores=[],
            ),
            str(path),
            "exec",
        ),
        namespace,
    )
    heads, head_dim, rope_dim, hidden_dim, rank = 2, 128, 64, 16, 8
    hidden = (
        (torch.arange(tokens * hidden_dim).reshape(tokens, hidden_dim) % 5 - 2) / 8
    ).bfloat16()
    q_lora = (
        (torch.arange(tokens * rank).reshape(tokens, rank) % 5 - 2) / 8
    ).bfloat16()
    wk = (
        (torch.arange(head_dim * hidden_dim).reshape(head_dim, hidden_dim) % 3 - 1) / 8
    ).bfloat16()
    ww = (
        (torch.arange(heads * hidden_dim).reshape(heads, hidden_dim) % 3 - 1) / 8
    ).bfloat16()
    wq = (
        (torch.arange(heads * head_dim * rank).reshape(heads * head_dim, rank) % 3 - 1)
        / 8
    ).bfloat16()

    def projection(weight):
        return lambda x: (torch.nn.functional.linear(x, weight), None)

    seen_positions = []

    def rotary(positions, query, key):
        # Replace only the GPU RoPE boundary. Real positions and strided slices
        # must reach it unchanged, for every row including those past Top-K.
        seen_positions.append(positions.clone())
        shift = (positions % 4).to(query.dtype).reshape(-1, 1, 1)
        return query + shift, key + shift

    state = SimpleNamespace(
        index_n_heads=heads,
        index_head_dim=head_dim,
        rope_head_dim=rope_dim,
        wq_b=projection(wq),
        wk=projection(wk),
        weights_proj=projection(ww),
        wk_weights_proj=projection(torch.cat((wk, ww))),
        _wk_weights_proj_loaded=False,
        k_norm=torch.nn.LayerNorm(head_dim, eps=1e-6, dtype=torch.bfloat16),
        rotary_emb=rotary,
    )
    positions = torch.arange(131072, 131072 + tokens)
    direct = namespace["forward"](state, hidden, q_lora, positions)
    state._wk_weights_proj_loaded = True
    fused = namespace["forward"](state, hidden, q_lora, positions)
    for name in ("query", "key", "weights"):
        torch.testing.assert_close(
            vars(fused)[name], vars(direct)[name], rtol=0, atol=0
        )
        assert vars(fused)[name].shape[0] == tokens
    assert fused.weights.stride(0) == head_dim + heads
    for seen in seen_positions:
        torch.testing.assert_close(seen, positions, rtol=0, atol=0)
