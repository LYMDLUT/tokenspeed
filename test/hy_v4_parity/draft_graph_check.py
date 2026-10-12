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

"""Check HY V4's draft step-0 prefill graph against eager on real weights.

    python -m test.hy_v4_parity.draft_graph_check --source /path/to/hy_v4

Builds a truncated checkpoint (released layers bit for bit, plus the MTP
block), then serves one greedy MTP batch twice on one GPU: with the draft's
step-0 prefill graph, and with step 0 eager. The target keeps its graphs in
both runs. A small chunked-prefill size splits the long prompts across
forwards; ``--enable-mixed-batch`` also lets those chunks share a batch with
decodes, so step 0 follows mixed batches. The graph declines those, so they
run eager in both runs and the check judges the pure extends among them.

The truncated target barely agrees with its draft, so the outputs say little
about the draft; the check reads the draft itself. Every served extend or
mixed step (decode steps replay the decode graph) records the ids each draft
step proposed and the top-2 logit gap behind each id. Step 0 is the graphed
forward and is judged: its ids must match and, where they match, its gaps must
agree to about bf16 rounding (the replay runs the bucket-padded shape, whose
GEMMs may tile differently). Later steps run eagerly in both runs on top of
step 0's cache, where that rounding compounds; they, the verify steps and the
outputs are reported. A proposal flipped at a near-tie changes what verify
accepts, and the same token then comes out of a different forward, so the
outputs can drift after it although verification is lossless. The graph run
must have replayed step 0 and the eager run must not have. Exit status 0 means
all of that held.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from test.hy_v4_parity.hy_v4_fixtures import (  # noqa: E402
    Truncation,
    build_truncated_checkpoint,
)


def _prompts(count: int, seed: int) -> list[list[int]]:
    """Token-id prompts spread from 9 to 1500 tokens; repeated spans let MTP accept."""
    rng = random.Random(seed)
    prompts = []
    for index in range(count):
        span = [rng.randrange(1000, 30000) for _ in range(rng.randrange(8, 40))]
        length = 9 + 1491 * index // max(count - 1, 1)
        prompts.append((span * (length // len(span) + 1))[:length])
    return prompts


def _run(args) -> int:
    """One engine run in this process; the parent sets the sitecustomize env."""
    from tokenspeed.runtime.entrypoints.engine import Engine

    engine = Engine(
        model=args.model,
        # The scheduler loads a tokenizer even with ids in and out; the
        # truncated checkpoint carries none.
        tokenizer=args.tokenizer,
        dtype="bfloat16",
        skip_tokenizer_init=True,
        speculative_algorithm="MTP",
        speculative_num_steps=args.steps,
        max_model_len=4096,
        max_num_seqs=args.prompts,
        chunked_prefill_size=256,
        enable_mixed_batch=args.enable_mixed_batch,
        enable_prefix_caching=False,
        disable_kvstore=True,
        disable_autotune=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        log_level="warning",
    )
    try:
        outputs = engine.generate(
            input_ids=_prompts(args.prompts, args.seed),
            sampling_params={"temperature": 0.0, "max_new_tokens": args.max_new_tokens},
        )
    finally:
        engine.shutdown()
    Path(args.out).write_text(
        json.dumps(
            [
                {
                    "output_ids": [int(token) for token in out["output_ids"]],
                    "verify_ct": int(out["meta_info"].get("spec_verify_ct", 0)),
                }
                for out in outputs
            ]
        )
    )
    return 0


def _spawn(mode: str, model: Path, workdir: Path, args) -> dict:
    """Serve the batch in a fresh interpreter; return outputs, counts and trace."""
    inject = workdir / "inject"
    inject.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_HERE / "draft_graph_sitecustomize.py", inject / "sitecustomize.py")
    stats = workdir / f"stats_{mode}"
    for stale in glob.glob(f"{stats}.*"):
        os.remove(stale)
    out = workdir / f"outputs_{mode}.json"
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [str(inject), str(_REPO_ROOT), os.environ.get("PYTHONPATH", "")]
        ),
        "HY_V4_DRAFT_GRAPH": mode,
        "HY_V4_DRAFT_GRAPH_STATS": str(stats),
    }
    command = [
        sys.executable,
        "-m",
        "test.hy_v4_parity.draft_graph_check",
        "--run",
        "--model",
        str(model),
        "--tokenizer",
        str(args.source),
        "--out",
        str(out),
        "--steps",
        str(args.steps),
        "--prompts",
        str(args.prompts),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--seed",
        str(args.seed),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ] + (["--enable-mixed-batch"] if args.enable_mixed_batch else [])
    print(f"\n== draft graph {mode}: {' '.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=str(_REPO_ROOT), env=env)
    if completed.returncode != 0:
        raise SystemExit(f"draft graph {mode} run exited with {completed.returncode}")
    counts = {"replays": 0, "eager_extends": 0}
    for path in glob.glob(f"{stats}.*[0-9]"):
        for key, value in json.loads(Path(path).read_text()).items():
            counts[key] += value
    trace = []
    for path in sorted(glob.glob(f"{stats}.*.trace")):
        trace += [json.loads(line) for line in Path(path).read_text().splitlines()]
    return {
        "outputs": json.loads(out.read_text()),
        "counts": counts,
        "trace": trace,
    }


def _output_summary(graph: list[dict], eager: list[dict]) -> str:
    """How many requests produced the same tokens and verify steps."""
    if len(graph) != len(eager):
        return f"request counts differ: {len(graph)} vs {len(eager)}"
    tokens = sum(g["output_ids"] == e["output_ids"] for g, e in zip(graph, eager))
    verify = sum(g["verify_ct"] == e["verify_ct"] for g, e in zip(graph, eager))
    return (
        f"outputs match for {tokens}/{len(graph)} requests, verify steps for "
        f"{verify}/{len(graph)}"
    )


def _compare_proposals(
    graph: list[dict], eager: list[dict], min_match: float, max_gap_delta: float
) -> tuple[list[str], str]:
    """Judge step 0, the graphed forward, on its ids and its logit gaps.

    The replay runs the bucket-padded shape, so its GEMMs may tile differently
    from eager: matching rows then differ in their top-2 logit gap by about
    bf16 rounding, and a near-tie can flip. A wrong graph moves the gap by the
    gap's own size. Later steps run eagerly in both runs on step 0's cache, so
    the rounding compounds there; they are reported, not judged.
    """
    if not graph:
        return ["the graph run traced no served draft calls"], ""
    shapes = [(call["bs"], call["num_extends"]) for call in graph]
    if shapes != [(call["bs"], call["num_extends"]) for call in eager]:
        return ["the two runs drafted different batches"], ""
    step0 = step0_same = later = later_same = 0
    deltas = []
    for g, e in zip(graph, eager):
        for ids_g, ids_e, gap_g, gap_e in zip(
            g["draft"], e["draft"], g["margin"], e["margin"]
        ):
            step0 += 1
            if ids_g[0] == ids_e[0]:
                step0_same += 1
                deltas.append(abs(gap_g[0] - gap_e[0]))
            # A mid-chunk batch stops after step 0: no gap, placeholder ids.
            drafted = [step for step in range(1, len(ids_g)) if gap_g[step] is not None]
            later += len(drafted)
            later_same += sum(ids_g[step] == ids_e[step] for step in drafted)
    deltas.sort()
    median = deltas[len(deltas) // 2] if deltas else float("inf")
    summary = (
        f"step-0 ids {step0_same}/{step0} match, median top-2 gap delta "
        f"{median:.4f}; later-step ids {later_same}/{later} match"
    )
    problems = []
    if step0_same < min_match * step0:
        problems.append(f"step-0 ids match below {min_match:.0%}: {summary}")
    if median > max_gap_delta:
        problems.append(f"step-0 logit gaps disagree beyond rounding: {summary}")
    return problems, summary


def _accept_length(outputs: list[dict]) -> float:
    tokens = sum(len(out["output_ids"]) for out in outputs)
    steps = sum(out["verify_ct"] for out in outputs)
    return tokens / steps if steps else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", help="HY V4 checkpoint directory to truncate.")
    parser.add_argument("--truncation", default="3:16", help="LAYERS:EXPERTS")
    parser.add_argument("--workdir", default="/tmp/hy_v4_draft_graph")
    parser.add_argument("--steps", type=int, default=3, help="MTP draft steps")
    parser.add_argument("--prompts", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument(
        "--enable-mixed-batch",
        action="store_true",
        help="Let the scheduler put prefill chunks and decodes in one batch.",
    )
    parser.add_argument(
        "--min-match",
        type=float,
        default=0.95,
        help="Least share of step-0 proposed ids that must match.",
    )
    parser.add_argument(
        "--max-gap-delta",
        type=float,
        default=0.05,
        help="Largest median step-0 top-2 logit gap difference between runs.",
    )
    parser.add_argument("--run", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model", help=argparse.SUPPRESS)
    parser.add_argument("--tokenizer", help=argparse.SUPPRESS)
    parser.add_argument("--out", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.run:
        return _run(args)
    if not args.source:
        parser.error("--source is required")

    layers, _, experts = args.truncation.partition(":")
    truncation = Truncation(
        num_layers=int(layers),
        num_experts=None if not experts or experts == "all" else int(experts),
    )
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    model = build_truncated_checkpoint(
        workdir / f"hy4-{truncation.describe().replace('/', '-')}",
        truncation,
        repo_id=args.source,
        log=lambda message: print(f"[fixture] {message}", flush=True),
    )

    graph = _spawn("on", model, workdir, args)
    eager = _spawn("off", model, workdir, args)
    print()
    for name, run in (("graph", graph), ("eager", eager)):
        trace = run["trace"]
        mixed = sum(0 < call["num_extends"] < call["bs"] for call in trace)
        print(
            f"{name} run: {run['counts']['replays']} step-0 replays, "
            f"{run['counts']['eager_extends']} eager extends, "
            f"{len(trace)} eager-step draft calls ({mixed} mixed), "
            f"accept length {_accept_length(run['outputs']):.3f}"
        )
    print(_output_summary(graph["outputs"], eager["outputs"]))
    problems, summary = _compare_proposals(
        graph["trace"], eager["trace"], args.min_match, args.max_gap_delta
    )
    if graph["counts"]["replays"] == 0:
        problems.append("the graph run never replayed the draft step-0 graph")
    if eager["counts"]["replays"] != 0 or eager["counts"]["eager_extends"] == 0:
        problems.append("the eager run did not run step 0 eagerly")
    for problem in problems:
        print(f"FAIL {problem}")
    if not problems:
        print(f"PASS {len(graph['trace'])} served draft calls: {summary}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
