// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Copyright (c) 2026 LightSeek Foundation (TVM-FFI adaptation).
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Adapted from vLLM commit 98dff2a81d747d1dba01a47f939f48c3526d4206:
// csrc/libtorch_stable/{layernorm_kernels,activation_kernels}.cu.
// See SOURCE.json for original URLs/hashes and the explicit scope changes.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

#include <cub/block/block_reduce.cuh>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <tvm/ffi/extra/cuda/device_guard.h>

#include "tvm_ffi_utils.h"

namespace {

struct Rows {
  int rank;
  int64_t s2, s3, s4, n2, n3;

  __device__ int64_t offset(int64_t row) const {
    if (rank == 2) return row * s2;
    if (rank == 3) return (row / n2) * s3 + (row % n2) * s2;
    const int64_t batch = row / (n3 * n2);
    const int64_t rest = row % (n3 * n2);
    return batch * s4 + (rest / n2) * s3 + (rest % n2) * s2;
  }
};

Rows row_layout(TensorView x) {
  return {static_cast<int>(x.ndim()), x.stride(-2), x.ndim() >= 3 ? x.stride(-3) : 0,
          x.ndim() >= 4 ? x.stride(-4) : 0,
          x.ndim() >= 3 ? x.size(-2) : 0,
          x.ndim() >= 4 ? x.size(-3) : 0};
}

void check_rows(TensorView x, int vector_width) {
  CHECK_CUDA(x);
  CHECK_INPUT_TYPE(x, dl_bfloat16);
  TVM_FFI_ICHECK(x.ndim() >= 2 && x.ndim() <= 4);
  TVM_FFI_ICHECK(x.size(-1) > 0 && x.size(-1) % vector_width == 0);
  TVM_FFI_ICHECK_EQ(x.stride(-1), 1);
  TVM_FFI_ICHECK_LE(x.size(-1), std::numeric_limits<int>::max());
  TVM_FFI_ICHECK_LE(x.numel() / x.size(-1), std::numeric_limits<int>::max());
  if (x.numel() == 0) return;
  TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(x.data_ptr()) %
                       (vector_width * sizeof(__nv_bfloat16)), 0);
  std::pair<int64_t, int64_t> dimensions[4];
  int count = 0;
  for (int i = 0; i < x.ndim(); ++i) {
    if (x.size(i) > 1) {
      if (i != x.ndim() - 1) TVM_FFI_ICHECK_EQ(x.stride(i) % vector_width, 0);
      dimensions[count++] = {x.stride(i), x.size(i)};
    }
  }
  std::sort(dimensions, dimensions + count);
  int64_t span = 1;
  for (int i = 0; i < count; ++i) {
    TVM_FFI_ICHECK_GE(dimensions[i].first, span) << "overlapping rows/elements";
    span += (dimensions[i].second - 1) * dimensions[i].first;
  }
}

uintptr_t end_address(TensorView x) {
  int64_t span = 1;
  for (int i = 0; i < x.ndim(); ++i) span += (x.size(i) - 1) * x.stride(i);
  return reinterpret_cast<uintptr_t>(x.data_ptr()) + span * (x.dtype().bits / 8);
}

bool overlaps(TensorView a, TensorView b) {
  return a.numel() && b.numel() &&
         reinterpret_cast<uintptr_t>(a.data_ptr()) < end_address(b) &&
         reinterpret_cast<uintptr_t>(b.data_ptr()) < end_address(a);
}

bool exact_alias(TensorView a, TensorView b) {
  if (a.data_ptr() != b.data_ptr() || a.ndim() != b.ndim()) return false;
  for (int i = 0; i < a.ndim(); ++i) {
    if (a.size(i) != b.size(i) || a.stride(i) != b.stride(i)) return false;
  }
  return true;
}

struct FloatAdd {
  __device__ __forceinline__ float operator()(float a, float b) const {
    return a + b;
  }
};

template <typename Weight>
__global__ void rmsnorm_kernel(const __nv_bfloat16* input, const Weight* weight,
                               __nv_bfloat16* output, Rows in_rows, Rows out_rows,
                               int hidden, float eps) {
  // No restrict on input/output: a full row-preserving alias is supported.
  const __nv_bfloat16* in = input + in_rows.offset(blockIdx.x);
  __nv_bfloat16* out = output + out_rows.offset(blockIdx.x);
  float variance = 0.0f;
  // Preserve the public aligned VEC_SIZE=8 per-thread accumulation order.
  for (int i = threadIdx.x; i < hidden / 8; i += blockDim.x) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float x = static_cast<float>(in[i * 8 + j]);
      variance += x * x;
    }
  }
  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduce_store;
  __shared__ float inverse_rms;
  variance = BlockReduce(reduce_store).Reduce(variance, FloatAdd{}, blockDim.x);
  if (threadIdx.x == 0) inverse_rms = rsqrtf(variance / hidden + eps);
  __syncthreads();
  for (int i = threadIdx.x; i < hidden / 8; i += blockDim.x) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int index = i * 8 + j;
      const float x = static_cast<float>(in[index]);
      const float w = static_cast<float>(weight[index]);
      out[index] = __float2bfloat16_rn(x * inverse_rms * w);
    }
  }
}

__global__ void silu_kernel(const __nv_bfloat16* input, __nv_bfloat16* output,
                            Rows in_rows, Rows out_rows, int hidden) {
  const __nv_bfloat16* in = input + in_rows.offset(blockIdx.x);
  __nv_bfloat16* out = output + out_rows.offset(blockIdx.x);
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    const float gate = static_cast<float>(in[i]);
    const float up = static_cast<float>(in[hidden + i]);
    // Public silu_kernel<T> returns T before compute multiplies by float up.
    const __nv_bfloat16 activated =
        __float2bfloat16_rn(gate / (1.0f + expf(-gate * 1.0f)));
    out[i] = __float2bfloat16_rn(static_cast<float>(activated) * (up + 0.0f));
  }
}

}  // namespace

void rmsnorm(TensorView input, TensorView weight, TensorView output, double eps) {
  check_rows(input, 8);
  check_rows(output, 8);
  CHECK_INPUT(weight);
  CHECK_DEVICE(input, output);
  CHECK_DEVICE(input, weight);
  CHECK_SHAPE(input, output);
  TVM_FFI_ICHECK_EQ(weight.ndim(), 1);
  TVM_FFI_ICHECK_EQ(weight.numel(), input.size(-1));
  TVM_FFI_ICHECK(weight.dtype() == dl_bfloat16 || weight.dtype() == dl_float32);
  const float epsilon = static_cast<float>(eps);
  TVM_FFI_ICHECK(std::isfinite(epsilon) && epsilon > 0.0f);
  TVM_FFI_ICHECK(!overlaps(input, output) || exact_alias(input, output));
  TVM_FFI_ICHECK(!overlaps(weight, output));
  if (input.numel() == 0) return;
  ffi::CUDADeviceGuard guard(input.device().device_id);
  const auto stream = get_stream(input.device());
  const int hidden = input.size(-1);
  const int rows = input.numel() / hidden;
  // Public default (non-batch-invariant) launch, with the same reduction shape.
  const int threads = std::min(hidden / 8, rows < 256 ? 1024 : 256);
  const auto* in = static_cast<const __nv_bfloat16*>(input.data_ptr());
  auto* out = static_cast<__nv_bfloat16*>(output.data_ptr());
  if (weight.dtype() == dl_bfloat16) {
    rmsnorm_kernel<<<rows, threads, 0, stream>>>(
        in, static_cast<const __nv_bfloat16*>(weight.data_ptr()), out,
        row_layout(input), row_layout(output), hidden, epsilon);
  } else {
    rmsnorm_kernel<<<rows, threads, 0, stream>>>(
        in, static_cast<const float*>(weight.data_ptr()), out,
        row_layout(input), row_layout(output), hidden, epsilon);
  }
  TVM_FFI_ICHECK_EQ(cudaGetLastError(), cudaSuccess);
}

void silu_and_mul(TensorView input, TensorView output) {
  check_rows(input, 1);
  check_rows(output, 1);
  CHECK_DEVICE(input, output);
  TVM_FFI_ICHECK_EQ(input.ndim(), output.ndim());
  TVM_FFI_ICHECK_EQ(input.size(-1) % 2, 0);
  TVM_FFI_ICHECK_EQ(input.size(-1) / 2, output.size(-1));
  for (int i = 0; i < input.ndim() - 1; ++i) {
    TVM_FFI_ICHECK_EQ(input.size(i), output.size(i));
  }
  TVM_FFI_ICHECK(!overlaps(input, output));
  if (input.numel() == 0) return;
  ffi::CUDADeviceGuard guard(input.device().device_id);
  const auto stream = get_stream(input.device());
  const int hidden = output.size(-1);
  const int rows = output.numel() / hidden;
  silu_kernel<<<rows, std::min(hidden, 1024), 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input.data_ptr()),
      static_cast<__nv_bfloat16*>(output.data_ptr()), row_layout(input),
      row_layout(output), hidden);
  TVM_FFI_ICHECK_EQ(cudaGetLastError(), cudaSuccess);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(rmsnorm, rmsnorm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(silu_and_mul, silu_and_mul);
