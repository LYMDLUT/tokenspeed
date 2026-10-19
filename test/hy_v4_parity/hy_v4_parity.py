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

"""Compare TokenSpeed and the reference engine on a truncated HY V4, layer by layer.

    python -m test.hy_v4_parity.hy_v4_parity \
        --reference-repo /path/to/vllm \
        --reference-python /path/to/vllm-venv/bin/python

Steps: build a truncated checkpoint from the released weights (bit-identical on
the layers it keeps), run each engine over a fixed prompt collecting sampled
sub-block activation summaries, then compare those summaries and the generated
token logprobs. The summaries do not establish full-tensor equality.

Exit status is 0 only when both the layer dumps and the logprobs are within
tolerance, so this doubles as a regression gate for the HY V4 port.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from test.hy_v4_parity.hy_v4_dump import compare_dumps  # noqa: E402
from test.hy_v4_parity.hy_v4_fixtures import (  # noqa: E402
    HY_V4_REPO_ID,
    Truncation,
    build_truncated_checkpoint,
)


def _run_engine(
    *,
    python: str,
    engine: str,
    model_path: Path,
    workdir: Path,
    num_layers: int,
    extra: list[str],
) -> tuple[dict, Path]:
    from test.hy_v4_parity.run_engine_dump import worker_environment

    workdir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix=f"{engine}-", dir=workdir)).resolve()
    run_id = run_dir.name
    dump_path = run_dir / f"{engine}.dump.json"
    summary_path = run_dir / f"{engine}.summary.json"
    reference_parser = argparse.ArgumentParser(add_help=False)
    reference_parser.add_argument(
        "--reference-repo", default=os.environ.get("HY_V4_REFERENCE_REPO", "")
    )
    reference_args, _ = reference_parser.parse_known_args(extra)
    env = worker_environment(
        dump_path=dump_path,
        run_id=run_id,
        num_layers=num_layers,
        engine=engine,
        reference_repo=reference_args.reference_repo if engine == "reference" else "",
    )
    command = [
        python,
        "-m",
        "test.hy_v4_parity.run_engine_dump",
        "--engine",
        engine,
        "--model-path",
        str(model_path.resolve()),
        "--run-id",
        run_id,
        "--dump-path",
        str(dump_path),
        "--summary-path",
        str(summary_path),
        "--num-layers",
        str(num_layers),
        *extra,
    ]
    print(f"\n== running {engine}: {' '.join(command)}")
    completed = subprocess.run(
        command,
        cwd=str(_REPO_ROOT),
        env=env,
        text=True,
    )
    if completed.returncode != 0:
        raise SystemExit(
            f"{engine} exited with {completed.returncode}; see its output above"
        )
    if not dump_path.exists():
        raise SystemExit(
            f"{engine} produced no layer dump at {dump_path}. The model may not "
            "have been attached: confirm the checkpoint exposes model_type=hy_v4 "
            "and that BaseCausalLM.post_init ran."
        )
    summary = json.loads(summary_path.read_text())
    metadata = json.loads(
        dump_path.with_suffix(dump_path.suffix + ".meta.json").read_text()
    )
    if (
        summary.get("run_id") != run_id
        or metadata.get("run_id") != run_id
        or metadata.get("complete") is not True
    ):
        raise RuntimeError("dump and summary do not belong to this completed run")
    valid, detail = _compare_outputs(summary, summary, atol=0)
    if not valid:
        raise RuntimeError(detail)
    return summary, dump_path


def _compare_outputs(
    reference: dict, candidate: dict, *, atol: float
) -> tuple[bool, str]:
    if not math.isfinite(atol) or atol < 0:
        raise ValueError("logprob tolerance must be finite and non-negative")
    for name, summary in (("reference", reference), ("candidate", candidate)):
        if not isinstance(summary, dict):
            return False, f"{name}: missing output summary"
        ids, logprobs = summary.get("output_ids"), summary.get("logprobs")
        if (
            not isinstance(ids, list)
            or not ids
            or any(type(token) is not int or token < 0 for token in ids)
        ):
            return False, f"{name}: missing or invalid generated token ids"
        if (
            not isinstance(logprobs, list)
            or len(logprobs) != len(ids)
            or any(
                not isinstance(lp, (int, float)) or not math.isfinite(lp)
                for lp in logprobs
            )
        ):
            return False, f"{name}: require one finite logprob per generated token"
    lines: list[str] = []
    ok = True
    if reference["output_ids"] != candidate["output_ids"]:
        ok = False
        lines.append(
            "output ids differ:\n"
            f"  reference: {reference['output_ids']}\n"
            f"  candidate: {candidate['output_ids']}"
        )
    else:
        lines.append(f"output ids match: {candidate['output_ids']}")
    ref_lp, cand_lp = reference.get("logprobs", []), candidate.get("logprobs", [])
    if len(ref_lp) != len(cand_lp):
        ok = False
        lines.append(f"logprob counts differ: {len(ref_lp)} vs {len(cand_lp)}")
    elif ref_lp:
        deltas = [abs(a - b) for a, b in zip(ref_lp, cand_lp)]
        worst = max(deltas)
        # Prompts are short, so unlike the full model this tolerance is not
        # relaxed for scale: any real wiring bug blows past it.
        if worst > atol:
            ok = False
        lines.append(
            f"logprob max|diff| = {worst:.4e} over {len(deltas)} steps "
            f"(tolerance {atol:.1e}); diffs={[f'{d:.1e}' for d in deltas]}"
        )
    return ok, "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default=HY_V4_REPO_ID,
        help="Original checkpoint path or model ID used to build the truncation.",
    )
    parser.add_argument(
        "--tokenizer",
        help="TokenSpeed scheduler tokenizer; defaults to an explicit "
        "HY_V4_TS_ENGINE_KWARGS tokenizer, otherwise --source.",
    )
    parser.add_argument(
        "--truncation",
        default="3:8",
        help="LAYERS:EXPERTS to keep, e.g. '3:8'. Use 'all' for all experts.",
    )
    parser.add_argument(
        "--workdir",
        default="/tmp/hy_v4_parity",
        help="Where the truncated checkpoint, dumps and summaries are written.",
    )
    parser.add_argument(
        "--reference-repo",
        default=os.environ.get("HY_V4_REFERENCE_REPO", ""),
        help="vLLM checkout that ships vllm/models/hy_v4 (0.29.0).",
    )
    parser.add_argument(
        "--reference-python",
        default=sys.executable,
        help="Interpreter that can import that vLLM checkout.",
    )
    parser.add_argument(
        "--tokenspeed-python",
        default=sys.executable,
        help="Interpreter that can import TokenSpeed.",
    )
    parser.add_argument(
        "--tokenspeed-args",
        default="",
        help="Extra TokenSpeed engine args (e.g. a backend or quantization to sweep).",
    )
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--logprob-atol", type=float, default=1e-2)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the truncated checkpoint even if a cached one exists.",
    )
    parser.add_argument(
        "--skip-reference",
        action="store_true",
        help="Only run TokenSpeed (smoke test the harness itself).",
    )
    args = parser.parse_args()

    extra = args.tokenspeed_args.split()
    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument("--tokenizer")
    forwarded, _ = extra_parser.parse_known_args(extra)
    if forwarded.tokenizer is not None:
        parser.error("pass --tokenizer directly, not inside --tokenspeed-args")

    from test.hy_v4_parity.run_engine_dump import _tokenspeed_engine_kwargs

    engine_kwargs = _tokenspeed_engine_kwargs(
        args.tokenizer, default_tokenizer=args.source
    )
    layers, _, experts = args.truncation.partition(":")
    truncation = Truncation(
        num_layers=int(layers),
        num_experts=None if not experts or experts == "all" else int(experts),
    )
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    model_path = build_truncated_checkpoint(
        workdir / f"hy4-{truncation.describe().replace('/', '-')}",
        truncation,
        repo_id=args.source,
        overwrite=args.rebuild,
        log=lambda message: print(f"[fixture] {message}"),
    )

    candidate, candidate_dump = _run_engine(
        python=args.tokenspeed_python,
        engine="tokenspeed",
        model_path=model_path,
        workdir=workdir,
        num_layers=truncation.num_layers,
        extra=[
            "--tokenizer",
            engine_kwargs["tokenizer"],
            *extra,
        ],
    )
    if args.skip_reference:
        print(f"\n[skip] reference run skipped; TokenSpeed dump at {candidate_dump}")
        return 0

    if not args.reference_repo:
        raise SystemExit(
            "--reference-repo is required (or set HY_V4_REFERENCE_REPO); it must "
            "point at a checkout that provides vllm/models/hy_v4"
        )
    reference, reference_dump = _run_engine(
        python=args.reference_python,
        engine="reference",
        model_path=model_path,
        workdir=workdir,
        num_layers=truncation.num_layers,
        extra=["--reference-repo", args.reference_repo],
    )

    layers_ok, layer_report = compare_dumps(
        reference_dump, candidate_dump, atol=args.atol, rtol=args.rtol
    )
    outputs_ok, output_report = _compare_outputs(
        reference, candidate, atol=args.logprob_atol
    )

    print("\n== layer comparison (reference vs tokenspeed)")
    print(layer_report)
    print("\n== output comparison")
    print(output_report)

    verdict = layers_ok and outputs_ok
    print(f"\nVERDICT: {'PASS' if verdict else 'FAIL'}")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
