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

"""Sitecustomize that toggles and counts Eagle's draft step-0 prefill graph.

``draft_graph_check.py`` copies this file onto ``PYTHONPATH`` as
``sitecustomize.py`` so it reaches the engine's worker processes, which a patch
in the parent cannot. With ``HY_V4_DRAFT_GRAPH=off`` the drafter declares no
step-0 prefill graph, so the executor neither sizes nor captures one and step 0
runs eagerly while the target keeps its graphs; ``on`` leaves capture alone. With ``HY_V4_DRAFT_GRAPH_STATS`` set, every
process that runs the drafter also keeps ``HY_V4_DRAFT_GRAPH_STATS.<pid>`` up to
date with how many step-0 forwards replayed the graph and how many extends ran
eagerly, and appends one line per served eager-step draft call (batch shape and
proposed ids) to ``HY_V4_DRAFT_GRAPH_STATS.<pid>.trace``; decode steps replay
inside the decode graph and never reach Python. Both are written as they happen because
engine shutdown does not run ``atexit``; leave the variable unset when timing a
server, since the trace syncs the GPU on every draft call.
"""

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def _chain_next_sitecustomize() -> None:
    """Run the next ``sitecustomize`` on ``sys.path``; Python imports only one."""
    for entry in sys.path:
        if not entry or os.path.abspath(entry) == _HERE:
            continue
        candidate = os.path.join(entry, "sitecustomize.py")
        if os.path.isfile(candidate):
            with open(candidate) as handle:
                source = handle.read()
            namespace = {"__name__": "sitecustomize", "__file__": candidate}
            exec(compile(source, candidate, "exec"), namespace)
            return


_chain_next_sitecustomize()


def _install() -> None:
    mode = os.environ.get("HY_V4_DRAFT_GRAPH")
    if mode not in ("on", "off"):
        return
    from tokenspeed.runtime.execution.drafter import eagle

    if mode == "off":
        eagle.Eagle.captures_prefill_graph = property(lambda self: False)

    stats = os.environ.get("HY_V4_DRAFT_GRAPH_STATS")
    print(f"[draft_graph] mode={mode} stats={stats}", file=sys.stderr)
    if not stats:
        return
    from tokenspeed.runtime.execution import forward_step

    stats_path = f"{stats}.{os.getpid()}"
    trace_path = f"{stats_path}.trace"
    counts = {"replays": 0, "eager_extends": 0}

    def record(key: str) -> None:
        counts[key] += 1
        with open(stats_path, "w") as handle:
            json.dump(counts, handle)

    # The decode graph captures the whole step, drafter included: its warmups
    # run on dummy inputs and its replays never reach Python. Skipping the
    # graph phase leaves exactly the served eager steps (extend and mixed).
    forward_first_step = eagle.Eagle._forward_first_step

    def counted_first_step(self, ctx, *args, **kwargs):
        if not forward_step.get_is_cuda_graph_phase():
            graph = self.prefill_graph
            if graph is not None and graph.can_run(ctx):
                record("replays")
            elif ctx.num_extends > 0:
                record("eager_extends")
        return forward_first_step(self, ctx, *args, **kwargs)

    eagle.Eagle._forward_first_step = counted_first_step

    # The draft's logits processor picks each step's ids (eagerly, after any
    # replay); the top-2 logit gap per row tells a near-tie flip from a wrong
    # forward. A mid-chunk batch stops after step 0, so the later steps are
    # placeholder ids with no gap.
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessor

    argmax = LogitsProcessor._argmax
    margins = []
    drafting = [False]

    def recorded_argmax(self, logits):
        if drafting[0] and not forward_step.get_is_cuda_graph_phase():
            top = logits.float().topk(2, dim=-1).values
            margins.append((top[:, 0] - top[:, 1]).tolist())
        return argmax(self, logits)

    LogitsProcessor._argmax = recorded_argmax
    draft = eagle.Eagle.draft

    def traced_draft(self, draft_input):
        margins.clear()
        drafting[0] = True
        try:
            next_tokens = draft(self, draft_input)
        finally:
            drafting[0] = False
        if forward_step.get_is_cuda_graph_phase():
            return next_tokens
        bs, steps = next_tokens.shape[0], next_tokens.shape[1] - 1
        gaps = margins + [[None] * bs] * (steps - len(margins))
        entry = {
            "bs": int(bs),
            "num_extends": int(draft_input.num_extends),
            "accept": draft_input.accept_lengths.tolist(),
            "draft": next_tokens[:, 1:].tolist(),
            "margin": [list(row) for row in zip(*gaps)],
        }
        with open(trace_path, "a") as handle:
            handle.write(json.dumps(entry) + "\n")
        return next_tokens

    eagle.Eagle.draft = traced_draft


try:
    _install()
except Exception as error:  # pragma: no cover - a silent miss would fake a pass
    print(f"[draft_graph] install failed: {error}", file=sys.stderr)
    raise
