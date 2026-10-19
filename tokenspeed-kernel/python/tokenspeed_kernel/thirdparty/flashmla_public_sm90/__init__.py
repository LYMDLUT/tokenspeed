# MIT License
#
# Copyright (c) 2026 LightSeek Foundation <contact@lightseek.org>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Independent SM90 sparse prefill from pinned public FlashMLA sources.

This package does not import vLLM or a prebuilt FlashMLA extension. Its explicit
math policy describes this candidate, not the historical reference wheel build.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import os
import re
from functools import cache
from numbers import Real
from pathlib import Path

import torch

_FLASHMLA_COMMIT = "0397728d511c4e3d94ea3a01d8dda8654525a611"
_CUTLASS_COMMIT = "147f5673d0c1c3dcf66f78d677fd647e4a020219"
_CXX_FLAGS = ("-std=c++20", "-O3")
_CUDA_FLAGS = (
    "-std=c++20",
    "-O3",
    "-gencode=arch=compute_90a,code=sm_90a",
    "--ftz=false",
    "--prec-div=true",
    "--prec-sqrt=true",
    "--fmad=true",
)
_COMPILE_UNITS = (
    "csrc/public_flashmla_sm90.cu",
    "csrc/flashmla/sm90/prefill/sparse/fwd.cu",
    "csrc/flashmla/sm90/prefill/sparse/instantiations/phase1_k512.cu",
    "csrc/flashmla/sm90/prefill/sparse/instantiations/phase1_k512_topklen.cu",
    "csrc/flashmla/sm90/prefill/sparse/instantiations/phase1_k576.cu",
    "csrc/flashmla/sm90/prefill/sparse/instantiations/phase1_k576_topklen.cu",
)
_INCLUDE_DIRECTORIES = (
    "csrc",
    "csrc/flashmla",
    "csrc/flashmla/kerutils/include",
    "csrc/cutlass/include",
)
_OVERRIDE_ENV = (
    "NVCC_PREPEND_FLAGS",
    "NVCC_APPEND_FLAGS",
    "FLASHINFER_EXTRA_CUDAFLAGS",
    "FLASHINFER_EXTRA_CFLAGS",
    "FLASHINFER_EXTRA_LDFLAGS",
)
_RECORDED_ENV = (
    *_OVERRIDE_ENV,
    "CUDAFLAGS",
    "CXXFLAGS",
    "CMAKE_ARGS",
    "CUDA_NVCC_FLAGS",
    "TORCH_CUDA_ARCH_LIST",
    "FLASHINFER_CUDA_ARCH_LIST",
    "FLASHINFER_WORKSPACE_BASE",
    "FLASHINFER_NVCC",
    "FLASHINFER_NVCC_LAUNCHER",
    "FLASHINFER_CXX_LAUNCHER",
    "FLASHINFER_JIT_DEBUG",
    "FLASHINFER_JIT_VERBOSE",
    "FLASHINFER_JIT_LINEINFO",
    "FLASHINFER_NVCC_THREADS",
    "CUDA_HOME",
    "CUDA_PATH",
    "CC",
    "CXX",
    "MAX_JOBS",
)
_INT32_MAX = 2**31 - 1
_FLOAT32_MAX = 3.4028234663852886e38
_LOG2_E = 1.4426950216293335  # Exact FP32 value of public LOG_2_E = 1.44269504f.
_BUILD_STATE: dict | None = None


def _file_fact(path: Path) -> dict:
    data = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def public_source_identity() -> dict:
    """Verify every bundled file and return the deterministic build identity.

    This reads files only: it does not import FlashInfer, inspect a GPU, or build.
    The caller should bind SOURCE.json itself to its sealed staging manifest.
    """
    root = Path(__file__).resolve().parent
    manifest_path = root / "SOURCE.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest["flashmla_commit"] != _FLASHMLA_COMMIT:
        raise RuntimeError("Unexpected public FlashMLA source commit")
    if manifest["cutlass_commit"] != _CUTLASS_COMMIT:
        raise RuntimeError("Unexpected public CUTLASS source commit")
    if tuple(manifest["compile_units"]) != _COMPILE_UNITS:
        raise RuntimeError("Unexpected public FlashMLA compilation units")
    if tuple(manifest["include_directories"]) != _INCLUDE_DIRECTORIES:
        raise RuntimeError("Unexpected public FlashMLA include directories")
    expected_flags = {
        "cxx": list(_CXX_FLAGS),
        "cuda": list(_CUDA_FLAGS),
        "link": [],
        "use_fast_math": False,
    }
    if manifest["flags"] != expected_flags:
        raise RuntimeError("Unexpected public FlashMLA math/build policy")
    declared = set()
    files = []
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("Invalid path in public FlashMLA source manifest")
        path = root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise RuntimeError("Public FlashMLA sources must stay in their package")
        if entry["path"] in declared:
            raise RuntimeError("Duplicate public FlashMLA source manifest path")
        declared.add(entry["path"])
        actual = _file_fact(path)
        if any(actual[key] != entry[key] for key in ("bytes", "sha256")):
            raise RuntimeError(f"Public FlashMLA source changed: {relative}")
        files.append({**entry, "resolved_path": actual["path"]})
    actual_paths = {"__init__.py"}
    for directory in (root / "csrc", root / "licenses"):
        actual_paths.update(
            str(path.relative_to(root))
            for path in directory.rglob("*")
            if path.is_file()
        )
    if declared != actual_paths:
        raise RuntimeError("Public FlashMLA manifest does not cover its source tree")
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": digest,
        "flashmla_commit": _FLASHMLA_COMMIT,
        "cutlass_commit": _CUTLASS_COMMIT,
        "module_name": f"tokenspeed_public_flashmla_sm90_{digest[:16]}",
        "compile_units": [str(root / name) for name in _COMPILE_UNITS],
        "include_directories": [str(root / name) for name in _INCLUDE_DIRECTORIES],
        "flags": expected_flags,
        "files": files,
    }


def public_build_state() -> dict:
    """Report the exact JIT output paths and bytes after a successful load."""
    if _BUILD_STATE is None:
        return {"loaded": False, "source_identity": public_source_identity()}
    state = copy.deepcopy(_BUILD_STATE)
    state["current_source_identity"] = public_source_identity()
    for key in ("library", "build_ninja"):
        state[f"current_{key}"] = _file_fact(Path(state[key]["path"]))
    return state


def _validate_compile_commands(commands: list[dict], identity: dict) -> None:
    if {entry["file"] for entry in commands} != set(identity["compile_units"]):
        raise RuntimeError("JIT compile commands do not cover the pinned sources")
    wanted = {"ftz": "false", "prec-div": "true", "prec-sqrt": "true", "fmad": "true"}
    cutlass_include = f"-I{identity['include_directories'][-1]}"
    for entry in commands:
        command = entry["command"]
        if re.search(r"(?<!\S)-{1,2}use_fast_math(?:\s|=|$)", command):
            raise RuntimeError("JIT unexpectedly enabled fast math")
        seen = re.findall(
            r"(?<!\S)-{1,2}(ftz|prec-div|prec-sqrt|fmad)(?:=|\s+)(\S+)",
            command,
        )
        if set(seen) != set(wanted.items()):
            raise RuntimeError("JIT math flags differ from the candidate policy")
        if any(flag not in command for flag in _CUDA_FLAGS):
            raise RuntimeError("JIT omitted an explicit candidate compiler flag")
        if cutlass_include not in command:
            raise RuntimeError("JIT omitted the pinned CUTLASS include directory")
        system_positions = [m.start() for m in re.finditer(r"-isystem\s+", command)]
        if system_positions and command.index(cutlass_include) > min(system_positions):
            raise RuntimeError("Pinned CUTLASS must precede default system includes")


@cache
def _load_public_flashmla_sm90_module():
    global _BUILD_STATE
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Warm public SM90 FlashMLA before CUDA graph capture")
    environment = {name: os.environ.get(name) for name in _RECORDED_ENV}
    overrides = [name for name in _OVERRIDE_ENV if environment[name]]
    if overrides:
        raise RuntimeError(f"Explicit JIT flag overrides are unsupported: {overrides}")
    debug = environment["FLASHINFER_JIT_DEBUG"]
    if debug is None:
        debug = environment["FLASHINFER_JIT_VERBOSE"]
    if debug == "1":
        raise RuntimeError("The public SM90 candidate requires a non-debug JIT build")
    identity = public_source_identity()
    from flashinfer.jit import core, cpp_ext, env

    if "use_fast_math" not in inspect.signature(core.gen_jit_spec).parameters:
        raise RuntimeError("FlashInfer gen_jit_spec must support use_fast_math=False")
    spec = core.gen_jit_spec(
        identity["module_name"],
        [Path(path) for path in identity["compile_units"]],
        extra_include_paths=[Path(path) for path in identity["include_directories"]],
        extra_cflags=list(_CXX_FLAGS),
        extra_cuda_cflags=list(_CUDA_FLAGS),
        use_fast_math=False,
    )
    if spec.is_aot:
        raise RuntimeError("The public SM90 candidate must use its own JIT artifact")
    commands = spec.get_compile_commands()
    _validate_compile_commands(commands, identity)
    module = spec.build_and_load()
    _BUILD_STATE = {
        "loaded": True,
        "source_identity": identity,
        "environment": environment,
        "jit_builder_files": {
            name: _file_fact(Path(part.__file__))
            for name, part in (("core", core), ("cpp_ext", cpp_ext), ("env", env))
        },
        # FlashInfer's generated descriptions omit compiler-launcher wrappers,
        # dependency-generation arguments and the link command. Retain Ninja
        # identity separately; this is not an executed compiler transcript.
        "generated_compile_commands": commands,
        "build_directory": str(spec.build_dir.resolve()),
        "library": _file_fact(spec.jit_library_path),
        "build_ninja": _file_fact(spec.ninja_path),
    }
    return module


def flash_mla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: torch.Tensor | None = None,
    topk_length: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run unchanged public SM90 sparse prefill with separate output buffers.

    Q and KV are BF16 [N, Hq, Dq] and [M, 1, Dq]; indices are I32 [N, 1, K].
    Hq is 64 or 128, Dq is 512 or 576, Dv is 512, and K is a positive multiple
    of 128. Nonempty Q requires nonempty KV. Leading input strides may describe
    padded views; Q's TMA strides must be positive and 16-byte aligned.

    Optional sink is contiguous FP32 [Hq], and lengths are contiguous I32 [N].
    GPU length values must be in [0, K]; they are not scanned on the hot path.
    Indices outside [0, M) are masked by the unchanged public kernel. Sink -inf
    is supported. This API does not substitute another attention implementation.

    Returns contiguous (BF16 out [N,Hq,512], FP32 max_logits [N,Hq], FP32 lse
    [N,Hq]) in public API order. The public no-valid-token max/lse conventions
    are retained; callers may apply their established attention adapter rules.
    Empty Q validates metadata and returns empty outputs without loading JIT.
    """
    tensors = {"q": q, "kv": kv, "indices": indices}
    if any(not isinstance(tensor, torch.Tensor) for tensor in tensors.values()):
        raise TypeError("q, kv and indices must be tensors")
    if any(tensor.ndim != 3 for tensor in tensors.values()):
        raise ValueError("q, kv and indices must have three dimensions")
    n, heads, d_qk = q.shape
    m = kv.shape[0]
    topk = indices.shape[2]
    if heads not in (64, 128) or d_qk not in (512, 576):
        raise ValueError("Require Hq in {64,128} and Dq in {512,576}")
    if isinstance(d_v, bool) or not isinstance(d_v, int) or d_v != 512:
        raise ValueError("d_v must be the integer 512")
    if kv.shape != (m, 1, d_qk) or indices.shape != (n, 1, topk):
        raise ValueError("Require KV [M,1,Dq] and indices [N,1,K]")
    if topk <= 0 or topk % 128 or topk > _INT32_MAX - 127:
        raise ValueError("K must be a positive multiple of 128 within INT32 bounds")
    if n * heads > _INT32_MAX or m > _INT32_MAX or (n > 0 and m == 0):
        raise ValueError("Invalid or unsupported query/KV sequence extent")
    expected_dtypes = {
        "q": torch.bfloat16,
        "kv": torch.bfloat16,
        "indices": torch.int32,
    }
    for name, tensor in tensors.items():
        if tensor.dtype != expected_dtypes[name]:
            raise TypeError(f"Unexpected {name} dtype")
        if not tensor.is_cuda or tensor.device != q.device:
            raise ValueError("All inputs must be CUDA tensors on the same device")
        if tensor.stride(-1) != 1:
            raise ValueError(f"{name} must have a contiguous last dimension")
        if any(stride < 0 or stride > _INT32_MAX for stride in tensor.stride()):
            raise ValueError(f"{name} strides exceed the public INT32 contract")
    if any(stride <= 0 or (stride * 2) % 16 for stride in q.stride()[:2]):
        raise ValueError("Q leading strides must be positive and 16-byte aligned")
    if n and (kv.stride(0) * 2) % 16:
        raise ValueError("KV row byte stride must be a multiple of 16")
    for name, tensor, alignment in (
        ("q", q, 16),
        ("kv", kv, 16),
        ("indices", indices, 4),
    ):
        if tensor.numel() and tensor.data_ptr() % alignment:
            raise ValueError(f"{name} pointer alignment requirement is not met")
    if n and (n - 1) * indices.stride(0) + topk - 1 > _INT32_MAX:
        raise ValueError("Indices address arithmetic exceeds the public INT32 bound")
    if not isinstance(sm_scale, Real) or isinstance(sm_scale, bool):
        raise TypeError("sm_scale must be a real scalar")
    if not math.isfinite(sm_scale) or abs(sm_scale) > _FLOAT32_MAX / _LOG2_E:
        raise ValueError("sm_scale and its log2 conversion must fit finite FP32")
    for name, tensor, shape, dtype in (
        ("attn_sink", attn_sink, (heads,), torch.float32),
        ("topk_length", topk_length, (n,), torch.int32),
    ):
        if tensor is None:
            continue
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != dtype:
            raise TypeError(f"Unexpected {name} type or dtype")
        if tuple(tensor.shape) != shape or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous with shape {shape}")
        if not tensor.is_cuda or tensor.device != q.device:
            raise ValueError(f"{name} must share Q's CUDA device")
    out = torch.empty((n, heads, d_v), device=q.device, dtype=torch.bfloat16)
    max_logits = torch.empty((n, heads), device=q.device, dtype=torch.float32)
    lse = torch.empty_like(max_logits)
    with torch.cuda.device(q.device):
        if torch.cuda.get_device_capability(q.device) != (9, 0):
            raise ValueError("The public FlashMLA candidate supports SM90 only")
        if n == 0:
            return out, max_logits, lse
        module = _load_public_flashmla_sm90_module()
        module.public_flashmla_sparse_fwd(
            q,
            kv,
            indices,
            float(sm_scale),
            d_v,
            attn_sink,
            topk_length,
            out,
            max_logits,
            lse,
        )
    return out, max_logits, lse
