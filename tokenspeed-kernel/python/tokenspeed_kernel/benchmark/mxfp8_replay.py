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

"""Replay captured MXFP8 linear or routed-expert layers through public ops.

Run one backend per process so resident weights and CUDA graph pools are not
shared between competitors. The artifact contains canonical checkpoint weights
and captured inputs, never backend-specific packed weights.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch


@dataclass(frozen=True)
class ReplayArtifact:
    manifest: dict
    tensors: dict[str, torch.Tensor]
    sha256: str


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_artifact(manifest: Mapping, tensors: Mapping[str, torch.Tensor]) -> None:
    """Reject ambiguous layouts before transferring any replay data to a GPU."""
    if manifest.get("schema_version") != 1 or manifest.get("kind") not in {
        "linear",
        "moe",
    }:
        raise ValueError("Expected MXFP8 replay schema_version=1 and kind=linear/moe")
    for field in ("checkpoint", "checkpoint_revision", "layer", "capture_revision"):
        if not isinstance(manifest.get(field), str) or not manifest[field]:
            raise ValueError(f"Missing artifact provenance: {field}")
    for field in ("tp_size", "ep_size"):
        if type(manifest.get(field)) is not int or manifest[field] < 1:
            raise ValueError(f"{field} must be a positive integer")
    for axis in ("tp", "ep"):
        rank = manifest.get(f"{axis}_rank")
        if type(rank) is not int or not 0 <= rank < manifest[f"{axis}_size"]:
            raise ValueError(f"Invalid {axis}_rank")
    expected = {"x", "weight", "weight_scale"}
    if manifest["kind"] == "moe":
        expected = {
            "x",
            "w13_weight",
            "w13_weight_scale_inv",
            "w2_weight",
            "w2_weight_scale_inv",
            "topk_ids",
            "topk_weights",
        }
    if set(tensors) != expected:
        raise ValueError(
            f"Expected tensor keys {sorted(expected)}, got {sorted(tensors)}"
        )
    if any(not isinstance(value, torch.Tensor) for value in tensors.values()):
        raise TypeError("Replay payload must contain only tensors")
    if any(value.device.type != "cpu" for value in tensors.values()):
        raise ValueError("Artifact validation requires CPU tensors")
    x = tensors["x"]
    if x.ndim != 2 or x.dtype != torch.bfloat16 or not x.shape[0] or not x.shape[1]:
        raise ValueError("x must be a nonempty BF16 matrix [tokens, hidden]")
    if not torch.isfinite(x).all():
        raise ValueError("Captured activations must be finite")
    if manifest["kind"] == "linear":
        pairs = [("weight", "weight_scale")]
        weight = tensors["weight"]
        if weight.ndim != 2 or weight.shape[1] != x.shape[1]:
            raise ValueError("Linear weight must have shape [N, hidden]")
    else:
        pairs = [
            ("w13_weight", "w13_weight_scale_inv"),
            ("w2_weight", "w2_weight_scale_inv"),
        ]
        w13, w2 = tensors["w13_weight"], tensors["w2_weight"]
        if w13.ndim != 3 or w2.ndim != 3:
            raise ValueError("MoE weights must have shape [experts, N, K]")
        experts, twice_intermediate, hidden = w13.shape
        if (
            hidden != x.shape[1]
            or twice_intermediate % 2
            or w2.shape
            != (
                experts,
                hidden,
                twice_intermediate // 2,
            )
        ):
            raise ValueError("MoE gate/up and down geometry disagree")
        if manifest.get("num_experts") != experts * manifest["ep_size"]:
            raise ValueError("num_experts must describe equal contiguous EP shards")
        if manifest.get("activation_limit") != 10.0:
            raise ValueError("This routed-expert replay requires activation_limit=10")
        ids, weights = tensors["topk_ids"], tensors["topk_weights"]
        if ids.ndim != 2 or ids.shape[0] != x.shape[0] or not ids.shape[1]:
            raise ValueError("topk_ids must have shape [tokens, top_k > 0]")
        if ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("topk_ids must be int32 or int64")
        if weights.shape != ids.shape or weights.dtype != torch.float32:
            raise ValueError(
                "topk_weights must be FP32 with the same shape as topk_ids"
            )
        if not torch.isfinite(weights).all():
            raise ValueError("Routing weights must be finite")
        if ((ids < -1) | (ids >= manifest["num_experts"])).any():
            raise ValueError("Routing ids must be global expert ids or the -1 sentinel")
        ordered = ids.sort(dim=1).values
        if ((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0)).any():
            raise ValueError("Captured top-k must not repeat an expert within a token")
    for weight_name, scale_name in pairs:
        weight, scale = tensors[weight_name], tensors[scale_name]
        if weight.dtype != torch.float8_e4m3fn or scale.dtype != torch.uint8:
            raise TypeError("Weights must be original E4M3 and scales raw E8M0 uint8")
        if min(weight.shape) <= 0 or weight.shape[-1] % 32:
            raise ValueError("Weight dimensions must be positive and K divisible by 32")
        if scale.shape != (*weight.shape[:-1], weight.shape[-1] // 32):
            raise ValueError(f"Invalid group32 scale geometry for {weight_name}")
        if ((scale == 0) | (scale == 255)).any():
            raise ValueError("Comparable Marlin/Humming scales must be in [1, 254]")


def load_artifact(directory: Path) -> ReplayArtifact:
    manifest = json.loads((directory / "manifest.json").read_text())
    payload = directory / "tensors.pt"
    digest = file_sha256(payload)
    if manifest.get("tensors_sha256") != digest:
        raise ValueError("tensors.pt SHA256 differs from the capture manifest")
    tensors = torch.load(payload, map_location="cpu", weights_only=True)
    validate_artifact(manifest, tensors)
    return ReplayArtifact(manifest, tensors, digest)


def export_artifact(
    directory: Path, manifest: dict, tensors: dict[str, torch.Tensor]
) -> None:
    """Save a captured layer without replacing an existing artifact directory.

    The caller supplies original checkpoint payloads and layer input tensors;
    this function does not synthesize, requantize, or collect model data.
    """
    cpu = {
        name: value.detach().to("cpu").contiguous() for name, value in tensors.items()
    }
    validate_artifact(manifest, cpu)
    directory.mkdir(parents=True, exist_ok=False)
    payload = directory / "tensors.pt"
    torch.save(cpu, payload)
    metadata = {**manifest, "tensors_sha256": file_sha256(payload)}
    (directory / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")


def quantize_dequantize_group32(x: torch.Tensor) -> torch.Tensor:
    """Reference E4M3 quantization with the same power-of-two scale recipe."""
    groups = x.float().reshape(*x.shape[:-1], -1, 32)
    amax = groups.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax * (1.0 / 448.0))))
    return ((groups / scale).to(torch.float8_e4m3fn).float() * scale).reshape(x.shape)


def dequantize_weight(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    factors = torch.exp2(scale.float() - 127.0)
    return (weight.float().unflatten(-1, (-1, 32)) * factors[..., None]).flatten(-2)


def reference_output(
    artifact: ReplayArtifact,
    *,
    activation: str,
    device: torch.device,
    rows_per_chunk: int,
) -> torch.Tensor:
    """FP32 GEMM oracle with explicit BF16 layer and routed-product boundaries.

    Only one expert's dequantized weights and one row chunk are resident at a
    time. The returned output is on CPU, outside the benchmark memory boundary.
    """
    if activation not in {"bf16", "fp8"} or rows_per_chunk < 1:
        raise ValueError("Reference requires activation=bf16/fp8 and a positive chunk")
    tensors, manifest = artifact.tensors, artifact.manifest
    x = tensors["x"].to(device)
    quantize = (
        quantize_dequantize_group32 if activation == "fp8" else torch.Tensor.float
    )
    if manifest["kind"] == "linear":
        weight = dequantize_weight(
            tensors["weight"].to(device), tensors["weight_scale"].to(device)
        )
        chunks = [
            (quantize(rows) @ weight.T).to(torch.bfloat16).cpu()
            for rows in x.split(rows_per_chunk)
        ]
        return torch.cat(chunks)
    ids = tensors["topk_ids"]
    top_k = ids.shape[1]
    experts = tensors["w13_weight"].shape[0]
    offset = manifest["ep_rank"] * experts
    routes = torch.zeros((*ids.shape, x.shape[1]), dtype=torch.bfloat16)
    flat = routes.view(-1, x.shape[1])
    for expert in range(experts):
        route_ids = (ids.reshape(-1) == expert + offset).nonzero().flatten()
        if not route_ids.numel():
            continue
        w13 = dequantize_weight(
            tensors["w13_weight"][expert].to(device),
            tensors["w13_weight_scale_inv"][expert].to(device),
        )
        w2 = dequantize_weight(
            tensors["w2_weight"][expert].to(device),
            tensors["w2_weight_scale_inv"][expert].to(device),
        )
        for chosen in route_ids.split(rows_per_chunk):
            inputs = x[(chosen // top_k).to(device)]
            gate, up = (
                (quantize(inputs) @ w13.T).to(torch.bfloat16).float().chunk(2, dim=-1)
            )
            gate, up = gate.clamp_max(10.0), up.clamp(-10.0, 10.0)
            activated = (gate * torch.sigmoid(gate) * up).to(torch.bfloat16)
            flat[chosen] = (quantize(activated) @ w2.T).to(torch.bfloat16).cpu()
        del w13, w2
    weights = tensors["topk_weights"].to(torch.bfloat16).float()
    output = torch.zeros_like(tensors["x"], dtype=torch.float32)
    for slot in range(top_k):
        output += (
            (routes[:, slot].float() * weights[:, slot, None])
            .to(torch.bfloat16)
            .float()
        )
    return output.to(torch.bfloat16)


def error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    if actual.shape != expected.shape:
        raise ValueError("Cannot compare outputs with different shapes")
    a, b = actual.double(), expected.double()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("Non-finite output cannot produce a valid precision report")
    difference = a - b
    error_norm, reference_norm = (
        torch.linalg.vector_norm(difference).item(),
        torch.linalg.vector_norm(b).item(),
    )
    return {
        "relative_l2": (
            error_norm / reference_norm
            if reference_norm
            else (0.0 if not error_norm else None)
        ),
        "reference_l2": reference_norm,
        "error_l2": error_norm,
        "max_abs": difference.abs().max().item(),
        "mean_abs": difference.abs().mean().item(),
        "exact_fraction": (a == b).float().mean().item(),
    }


def bitwise_equal(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    return (
        actual.shape == expected.shape
        and actual.dtype == expected.dtype
        and torch.equal(
            actual.contiguous().view(torch.uint8),
            expected.contiguous().view(torch.uint8),
        )
    )


def precision_verdict(
    measured: torch.Tensor,
    first_eager: torch.Tensor,
    repeat_eager: torch.Tensor,
    reference: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> dict[str, bool]:
    """Validate the measured path as well as the surrounding eager calls."""
    reference_close = torch.allclose(
        measured.double(), reference.double(), atol=atol, rtol=rtol
    )
    repeat_equal = bitwise_equal(first_eager, repeat_eager)
    measured_equal = bitwise_equal(measured, first_eager)
    return {
        "passed": reference_close and repeat_equal and measured_equal,
        "reference_close": reference_close,
        "eager_repeat_bitwise_equal": repeat_equal,
        "measured_vs_eager_bitwise_equal": measured_equal,
    }


@dataclass(frozen=True)
class _SwigluArg:
    alpha: float
    limit: float


class ReplayMoeWeights(torch.nn.Module):
    def __init__(self, artifact: ReplayArtifact, device: torch.device):
        super().__init__()
        data, meta = artifact.tensors, artifact.manifest
        self.w13_weight = torch.nn.Parameter(
            data["w13_weight"].to(device), requires_grad=False
        )
        self.w2_weight = torch.nn.Parameter(
            data["w2_weight"].to(device), requires_grad=False
        )
        self.w13_weight_scale_inv = torch.nn.Parameter(
            data["w13_weight_scale_inv"].to(device), requires_grad=False
        )
        self.w2_weight_scale_inv = torch.nn.Parameter(
            data["w2_weight_scale_inv"].to(device), requires_grad=False
        )
        self.activation = "swiglu"
        self.swiglu_arg = _SwigluArg(alpha=1.0, limit=10.0)
        self.swiglu_beta = None
        self.w13_input_layout = "concatenated"
        self.w13_weight_bias = None
        self.w2_weight_bias = None
        self.ep_size, self.ep_rank = meta["ep_size"], meta["ep_rank"]
        self.tp_size, self.tp_rank = meta["tp_size"], meta["tp_rank"]
        self.num_experts = meta["num_experts"]
        self.num_local_experts = data["w13_weight"].shape[0]
        self.hidden_size = data["x"].shape[1]
        self.intermediate_size_per_partition = data["w2_weight"].shape[-1]
        self.top_k = data["topk_ids"].shape[1]
        self._moe_backend_state: object | None = None


def prepare_call(artifact: ReplayArtifact, *, backend: str, device: torch.device):
    """Return the complete public op, its warmup, and selected registration."""
    x = artifact.tensors["x"].to(device)
    if artifact.manifest["kind"] == "linear":
        from tokenspeed_kernel.ops.gemm import mm, prepare_mxfp8_linear

        weight = artifact.tensors["weight"].to(device)
        scale = artifact.tensors["weight_scale"].to(device)
        if backend == "marlin":
            plan = prepare_mxfp8_linear(
                weight,
                scale,
                input_dtype=torch.bfloat16,
                use_atomic_add=False,
                use_fp32_reduce=True,
                solution="marlin",
                override=None,
            )
            return lambda: plan.apply(x), lambda: None, plan.kernel_name
        if backend != "triton":
            raise ValueError("Linear replay supports marlin or triton")
        return (
            lambda: mm(
                x,
                weight,
                B_scales=scale,
                out_dtype=torch.bfloat16,
                quant="mxfp8",
                block_size=[1, 32],
                override="triton_mm_fp8_blockscale",
            ),
            lambda: None,
            "triton_mm_fp8_blockscale",
        )
    if backend not in {"marlin", "humming"}:
        raise ValueError("MoE replay supports marlin or humming")
    from tokenspeed_kernel.ops.moe import (
        moe_apply,
        moe_plan,
        moe_process_weights,
        moe_warmup,
    )

    w = ReplayMoeWeights(artifact, device)
    weights = artifact.tensors["topk_weights"].to(device)
    ids = artifact.tensors["topk_ids"].to(device)
    plan = moe_plan(
        weight_dtype="fp8",
        input_dtype=torch.bfloat16,
        activation="swiglu",
        requires_deferred_finalize=False,
        requires_deterministic_finalize=True,
        routing_mode="precomputed_topk",
        a2a_backend=None,
        ep_size=w.ep_size,
        ispp=w.intermediate_size_per_partition,
        hidden=w.hidden_size,
        swiglu_form="standard",
        activation_clamped=True,
        expert_id_repeats=False,
        num_experts=w.num_experts,
        top_k=w.top_k,
        fp8_scale_block_shape=(1, 32),
        internal_activation_dtype="fp8" if backend == "humming" else "input",
        with_bias=False,
        fast_math=False,
        combine_order="rank",
        features=frozenset(),
        solution=backend,
    )
    moe_process_weights(plan, w)
    return (
        lambda: moe_apply(plan, x, w, None, topk_weights=weights, topk_ids=ids),
        lambda: moe_warmup(plan, w, max_num_tokens=x.shape[0]),
        plan["apply_kernel_name"],
    )


def percentile(samples: list[float], fraction: float) -> float:
    if not samples or not 0 <= fraction <= 1:
        raise ValueError("Percentile requires samples and a fraction in [0, 1]")
    ordered = sorted(samples)
    index = (len(ordered) - 1) * fraction
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def environment() -> dict:
    versions = {}
    for package in (
        "torch",
        "tokenspeed-kernel",
        "tokenspeed-triton",
        "humming-kernels",
        "flashinfer-python",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    root = next(
        (
            parent
            for parent in Path(__file__).resolve().parents
            if (parent / ".git").exists()
        ),
        None,
    )
    git, dirty, diff_sha256 = None, None, None
    if root is not None:
        git = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                text=True,
                capture_output=True,
                check=True,
            ).stdout
        )
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=root,
            capture_output=True,
            check=True,
        ).stdout
        diff_sha256 = hashlib.sha256(diff).hexdigest()
    humming_commit = None
    try:
        direct_url = importlib.metadata.distribution("humming-kernels").read_text(
            "direct_url.json"
        )
        if direct_url:
            humming_commit = json.loads(direct_url).get("vcs_info", {}).get("commit_id")
    except importlib.metadata.PackageNotFoundError:
        pass
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "git_revision": git,
        "git_dirty": dirty,
        "git_diff_sha256": diff_sha256,
        "benchmark_source_sha256": file_sha256(Path(__file__)),
        "packages": versions,
        "humming_vcs_commit": humming_commit,
        "torch_cuda": torch.version.cuda,
        "gpu_name": properties.name,
        "gpu_uuid": str(properties.uuid),
        "gpu_capability": [properties.major, properties.minor],
        "gpu_total_memory": properties.total_memory,
    }


def measure_eager(
    call: Callable, *, warmup: int, samples: int, cold_cache: bool
) -> list[float]:
    from tokenspeed_kernel._triton import triton

    cache = (
        triton.runtime.driver.active.get_empty_cache_for_benchmark()
        if cold_cache
        else None
    )
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    values = []
    for _ in range(samples):
        if cache is not None:
            triton.runtime.driver.active.clear_cache(cache)
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        call()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000.0)
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--backend", choices=("marlin", "humming", "triton"), required=True
    )
    parser.add_argument("--timing", choices=("eager", "cuda-graph"), required=True)
    parser.add_argument("--cache", choices=("hot", "cold"), required=True)
    parser.add_argument("--reference-device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--atol", type=float, required=True)
    parser.add_argument("--rtol", type=float, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--reference-rows", type=int, default=64)
    args = parser.parse_args()
    if min(args.warmup, args.samples, args.reference_rows) < 1 or not all(
        math.isfinite(value) and value >= 0 for value in (args.atol, args.rtol)
    ):
        parser.error("Counts must be positive and tolerances nonnegative")
    if args.output.exists():
        parser.error("Refusing to replace an existing result")
    if not torch.cuda.is_available():
        parser.error("GPU timing requires CUDA")
    artifact = load_artifact(args.artifact)
    from tokenspeed_kernel.platform import pdl_enabled

    previous_pdl = pdl_enabled()
    previous_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        # The Humming adapter rejects PDL. Both competitors must use the same
        # launch policy through the public op, which reads this setting.
        pdl_enabled(False)
        _execute(args, artifact)
    finally:
        pdl_enabled(previous_pdl)
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32
        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32


def _execute(args: argparse.Namespace, artifact: ReplayArtifact) -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    baseline_allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    call, warm, registration = prepare_call(
        artifact, backend=args.backend, device=device
    )
    torch.cuda.synchronize()
    preparation_ms = (time.perf_counter() - started) * 1000
    preparation_peak = torch.cuda.max_memory_allocated()
    started = time.perf_counter()
    warm()
    first_eager = call().detach().cpu()
    torch.cuda.synchronize()
    first_call_ms = (time.perf_counter() - started) * 1000
    prepared_allocated = torch.cuda.memory_allocated()
    startup_peak_allocated = torch.cuda.max_memory_allocated()
    startup_peak_reserved = torch.cuda.max_memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    timer_details = None
    retained_output: list[torch.Tensor | None] = [None]

    def measured_call():
        output = call()
        retained_output[0] = output
        return output

    if args.timing == "cuda-graph":
        from tokenspeed_kernel.benchmark.graph import (
            GraphBenchmarkConfig,
            GraphTimer,
            PreparedInvocation,
        )

        measurement = GraphTimer(
            GraphBenchmarkConfig(
                eager_warmup_iterations=args.warmup,
                replay_warmup_iterations=args.warmup,
            )
        ).measure(
            PreparedInvocation(measured_call),
            cold_cache=args.cache == "cold",
            measurement_blocks=args.samples,
        )
        timer_details = asdict(measurement)
        samples = list(measurement.samples_us)
    else:
        samples = measure_eager(
            measured_call,
            warmup=args.warmup,
            samples=args.samples,
            cold_cache=args.cache == "cold",
        )
    if retained_output[0] is None:
        raise RuntimeError("The timer did not invoke the captured layer")
    # The wrapper runs during graph construction. Replay updates this same
    # destination without executing Python; read it before another eager call.
    actual = retained_output[0].detach().cpu()
    memory = {
        "scope": "pytorch_allocator",
        "baseline_allocated_bytes": baseline_allocated,
        "preparation_peak_allocated_bytes": preparation_peak,
        "prepared_allocated_bytes": prepared_allocated,
        "startup_peak_allocated_bytes": startup_peak_allocated,
        "startup_peak_reserved_bytes": startup_peak_reserved,
        "execution_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "execution_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "includes_timer_cache_buffer": args.cache == "cold",
    }
    repeat = call().detach().cpu()
    reference_device = torch.device(args.reference_device)
    a16 = reference_output(
        artifact,
        activation="bf16",
        device=reference_device,
        rows_per_chunk=args.reference_rows,
    )
    a8 = reference_output(
        artifact,
        activation="fp8",
        device=reference_device,
        rows_per_chunk=args.reference_rows,
    )
    expected = a16 if args.backend == "marlin" else a8
    verdict = precision_verdict(
        actual, first_eager, repeat, expected, atol=args.atol, rtol=args.rtol
    )
    passed = verdict["passed"]
    report = {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "linear"
            if artifact.manifest["kind"] == "linear"
            else "local_expert_chain_without_collectives"
        ),
        "artifact": artifact.manifest,
        "environment": environment(),
        "backend": args.backend,
        "registration": registration,
        "activation": "bf16" if args.backend == "marlin" else "group32_fp8_pow2",
        "pdl_enabled": False,
        "tf32_enabled": False,
        "timing": args.timing,
        "cache": args.cache,
        "warmup_iterations": args.warmup,
        "samples_us": samples,
        "p50_us": percentile(samples, 0.5),
        "p95_us": percentile(samples, 0.95),
        "accepted_timing": passed,
        "preparation_wall_ms": preparation_ms,
        "compile_warmup_first_call_and_readback_wall_ms": first_call_ms,
        "graph_timer": timer_details,
        "memory": memory,
        "precision": {
            "atol": args.atol,
            "rtol": args.rtol,
            **verdict,
            "implementation_vs_matching_reference": error_metrics(actual, expected),
            "activation_quantization_reference_vs_a16": error_metrics(a8, a16),
            "implementation_vs_a16_reference": error_metrics(actual, a16),
            "reference_device": args.reference_device,
            "reference_rows": args.reference_rows,
        },
        "limitations": [
            "PDL is disabled for every comparator because the Humming adapter does not support it.",
            "Local MoE replay excludes cross-rank communication, shared experts, attention and scheduling.",
            "Full-chain implementation error can include FC1 rounding crossing FC2 quantization thresholds; fixed-operand GEMM tests isolate kernel arithmetic.",
            "No throughput or model-quality conclusion follows from this layer replay.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "passed": passed,
                "p50_us": report["p50_us"],
                "p95_us": report["p95_us"],
            }
        )
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
