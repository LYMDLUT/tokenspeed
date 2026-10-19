// MIT License
//
// Copyright (c) 2026 LightSeek Foundation <contact@lightseek.org>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Standalone TVM FFI host adapter for the public FlashMLA SM90 sparse-prefill
// implementation at 0397728d511c4e3d94ea3a01d8dda8654525a611.
// Device algorithms and the four-way SM90 dispatcher remain in vendored files.
// The vendored FlashMLA/CUTLASS license notices are retained separately.

#include <cuda_runtime.h>
#include <cuda.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include <tvm/ffi/optional.h>
#include <tvm/ffi/extra/cuda/device_guard.h>

#include "tvm_ffi_utils.h"
#include "flashmla/sm90/prefill/sparse/fwd.h"

namespace {

constexpr int64_t kIntMax = std::numeric_limits<int>::max();
constexpr float kLog2E = 1.44269504f;  // Public api/common.h: LOG_2_E.
static_assert(sizeof(int) == sizeof(int32_t));

void check_tensor(TensorView tensor, DLDataType dtype, int ndim,
                  int device_id, const char* name) {
  TVM_FFI_ICHECK_EQ(tensor.ndim(), ndim) << name << ": wrong rank";
  TVM_FFI_ICHECK_EQ(tensor.dtype(), dtype) << name << ": wrong dtype";
  TVM_FFI_ICHECK_EQ(tensor.device().device_type, kDLCUDA)
      << name << ": CUDA tensor required";
  TVM_FFI_ICHECK_EQ(tensor.device().device_id, device_id)
      << name << ": tensors must share one CUDA device";
  for (int axis = 0; axis < ndim; ++axis) {
    TVM_FFI_ICHECK(tensor.size(axis) >= 0) << name << ": negative dimension";
    TVM_FFI_ICHECK(tensor.stride(axis) >= 0) << name << ": negative stride";
  }
  TVM_FFI_ICHECK_EQ(tensor.stride(ndim - 1), 1)
      << name << ": last dimension must be contiguous";
}

int checked_int(int64_t value, const char* name) {
  TVM_FFI_ICHECK(value >= 0 && value <= kIntMax)
      << name << ": value exceeds nonnegative INT32 range";
  return static_cast<int>(value);
}

struct AddressSpan {
  uintptr_t begin;
  uintptr_t end;
  const char* name;
};

// Conservative byte-span overlap checking also covers strided input views.
// It may reject two interleaved views whose bounding spans overlap; it never
// treats disjoint base pointers alone as proof that buffers do not alias.
AddressSpan address_span(TensorView tensor, uint64_t item_bytes,
                         const char* name) {
  for (int axis = 0; axis < tensor.ndim(); ++axis) {
    if (tensor.size(axis) == 0) return {0, 0, name};
  }
  uint64_t last_element = 0;
  constexpr uint64_t kMax = std::numeric_limits<uint64_t>::max();
  for (int axis = 0; axis < tensor.ndim(); ++axis) {
    const auto extent = static_cast<uint64_t>(tensor.size(axis) - 1);
    const auto stride = static_cast<uint64_t>(tensor.stride(axis));
    TVM_FFI_ICHECK(extent == 0 || stride <= (kMax - last_element) / extent)
        << name << ": tensor span overflows";
    last_element += extent * stride;
  }
  TVM_FFI_ICHECK(last_element < kMax / item_bytes)
      << name << ": byte span overflows";
  const uint64_t bytes = (last_element + 1) * item_bytes;
  const uintptr_t begin = reinterpret_cast<uintptr_t>(tensor.data_ptr());
  TVM_FFI_ICHECK(begin != 0) << name << ": null nonempty tensor pointer";
  TVM_FFI_ICHECK(bytes <= std::numeric_limits<uintptr_t>::max() - begin)
      << name << ": address span overflows";
  return {begin, begin + static_cast<uintptr_t>(bytes), name};
}

void check_alignment(const AddressSpan& span, uintptr_t alignment) {
  if (span.begin == span.end) return;
  TVM_FFI_ICHECK_EQ(span.begin % alignment, 0)
      << span.name << ": pointer alignment requirement not met";
}

void check_disjoint(const AddressSpan& a, const AddressSpan& b) {
  if (a.begin == a.end || b.begin == b.end) return;
  TVM_FFI_ICHECK(a.end <= b.begin || b.end <= a.begin)
      << a.name << " overlaps " << b.name;
}

}  // namespace

// Python owns output allocation and returns (out, max_logits, lse).
// GPU value contracts, intentionally not checked by CPU copies or scans:
//   * every provided topk_length is in [0, indices.size(2)];
//   * indices outside [0, kv.size(0)) are invalid and are masked by the kernel;
//   * attn_sink may contain +/-infinity, as specified by the public API.
// This adapter rejects nonempty-Q/empty-KV and zero-capacity TopK cases.
void public_flashmla_sparse_fwd(
    TensorView q, TensorView kv, TensorView indices, double sm_scale,
    int64_t d_v, ffi::Optional<TensorView> attn_sink,
    ffi::Optional<TensorView> topk_length, TensorView out,
    TensorView max_logits, TensorView lse) {
  const int device_id = q.device().device_id;
  TVM_FFI_ICHECK(device_id >= 0) << "invalid CUDA device id";
  check_tensor(q, dl_bfloat16, 3, device_id, "q");
  check_tensor(kv, dl_bfloat16, 3, device_id, "kv");
  check_tensor(indices, dl_int32, 3, device_id, "indices");
  check_tensor(out, dl_bfloat16, 3, device_id, "out");
  check_tensor(max_logits, dl_float32, 2, device_id, "max_logits");
  check_tensor(lse, dl_float32, 2, device_id, "lse");

  const int s_q = checked_int(q.size(0), "s_q");
  const int s_kv = checked_int(kv.size(0), "s_kv");
  const int h_q = checked_int(q.size(1), "h_q");
  const int h_kv = checked_int(kv.size(1), "h_kv");
  const int d_qk = checked_int(q.size(2), "d_qk");
  const int topk = checked_int(indices.size(2), "topk");
  TVM_FFI_ICHECK(h_q == 64 || h_q == 128) << "h_q must be 64 or 128";
  TVM_FFI_ICHECK_EQ(h_kv, 1) << "SM90 sparse prefill requires h_kv=1";
  TVM_FFI_ICHECK(d_qk == 512 || d_qk == 576) << "d_qk must be 512 or 576";
  TVM_FFI_ICHECK_EQ(d_v, 512) << "d_v must be 512";
  TVM_FFI_ICHECK(topk > 0 && topk % 128 == 0)
      << "SM90 TopK capacity must be positive and divisible by 128";
  TVM_FFI_ICHECK(s_q == 0 || s_kv > 0)
      << "empty KV with nonempty Q is outside this candidate's contract";
  TVM_FFI_ICHECK_EQ(kv.size(2), d_qk);
  TVM_FFI_ICHECK_EQ(indices.size(0), s_q);
  TVM_FFI_ICHECK_EQ(indices.size(1), h_kv);
  TVM_FFI_ICHECK_EQ(out.size(0), s_q);
  TVM_FFI_ICHECK_EQ(out.size(1), h_q);
  TVM_FFI_ICHECK_EQ(out.size(2), d_v);
  TVM_FFI_ICHECK_EQ(max_logits.size(0), s_q);
  TVM_FFI_ICHECK_EQ(max_logits.size(1), h_q);
  TVM_FFI_ICHECK_EQ(lse.size(0), s_q);
  TVM_FFI_ICHECK_EQ(lse.size(1), h_q);
  TVM_FFI_ICHECK(out.IsContiguous() && max_logits.IsContiguous() && lse.IsContiguous())
      << "all output buffers must be contiguous";

  if (attn_sink.has_value()) {
    const TensorView sink = attn_sink.value();
    check_tensor(sink, dl_float32, 1, device_id, "attn_sink");
    TVM_FFI_ICHECK_EQ(sink.size(0), h_q);
  }
  if (topk_length.has_value()) {
    const TensorView lengths = topk_length.value();
    check_tensor(lengths, dl_int32, 1, device_id, "topk_length");
    TVM_FFI_ICHECK_EQ(lengths.size(0), s_q);
  }

  const int q_stride_0 = checked_int(q.stride(0), "q.stride(0)");
  const int q_stride_1 = checked_int(q.stride(1), "q.stride(1)");
  const int kv_stride_0 = checked_int(kv.stride(0), "kv.stride(0)");
  const int kv_stride_1 = checked_int(kv.stride(1), "kv.stride(1)");
  const int idx_stride_0 = checked_int(indices.stride(0), "indices.stride(0)");
  const int idx_stride_1 = checked_int(indices.stride(1), "indices.stride(1)");
  // The public kernel forms these offsets using INT32 expressions.
  TVM_FFI_ICHECK(s_q <= kIntMax / h_q) << "INT32 row/output offset overflow";
  if (s_q > 0 && idx_stride_0 > 0) {
    TVM_FFI_ICHECK(static_cast<int64_t>(s_q - 1) <=
                   (kIntMax - (topk - 1)) / idx_stride_0)
        << "INT32 index-row offset overflow";
  }
  if (s_q > 0) {
    TVM_FFI_ICHECK(q_stride_0 > 0 && q_stride_1 > 0 &&
                   q_stride_0 % 8 == 0 && q_stride_1 % 8 == 0)
        << "Q TMA leading byte strides must be positive multiples of 16";
    TVM_FFI_ICHECK_EQ(kv_stride_0 % 8, 0)
        << "KV row byte stride must be a multiple of 16 for cp.async";
  }

  // Preserve public api/sparse_fwd.h's double*float expression followed by
  // conversion to the two float fields. Do not first round sm_scale to float.
  const float scale = static_cast<float>(sm_scale);
  const float scale_log2 = static_cast<float>(sm_scale * kLog2E);
  TVM_FFI_ICHECK(std::isfinite(sm_scale) && std::isfinite(scale) &&
                 std::isfinite(scale_log2))
      << "softmax scale must remain finite in both public FP32 fields";

  std::vector<AddressSpan> inputs{
      address_span(q, 2, "q"), address_span(kv, 2, "kv"),
      address_span(indices, 4, "indices")};
  check_alignment(inputs[0], 16);
  check_alignment(inputs[1], 16);
  check_alignment(inputs[2], 4);
  if (attn_sink.has_value()) {
    inputs.push_back(address_span(attn_sink.value(), 4, "attn_sink"));
    check_alignment(inputs.back(), 4);
  }
  if (topk_length.has_value()) {
    inputs.push_back(address_span(topk_length.value(), 4, "topk_length"));
    check_alignment(inputs.back(), 4);
  }
  const std::array<AddressSpan, 3> outputs{
      address_span(out, 2, "out"), address_span(max_logits, 4, "max_logits"),
      address_span(lse, 4, "lse")};
  for (size_t i = 0; i < outputs.size(); ++i) {
    check_alignment(outputs[i], 16);  // TMA / bulk S2G destinations.
    for (const auto& input : inputs) check_disjoint(outputs[i], input);
    for (size_t j = 0; j < i; ++j) check_disjoint(outputs[i], outputs[j]);
  }

  ffi::CUDADeviceGuard guard(device_id);
  cudaDeviceProp properties{};
  const cudaError_t property_status = cudaGetDeviceProperties(&properties, device_id);
  TVM_FFI_ICHECK_EQ(property_status, cudaSuccess) << cudaGetErrorString(property_status);
  TVM_FFI_ICHECK(properties.major == 9 && properties.minor == 0)
      << "this candidate executes only the SM90a implementation";
  TVM_FFI_ICHECK(properties.multiProcessorCount > 0);
  const int64_t grid_x = static_cast<int64_t>(s_q) * (h_q / 64);
  TVM_FFI_ICHECK(grid_x <= properties.maxGridSize[0]) << "CUDA grid.x overflow";
  // No TMA descriptor construction or kernel launch for zero query rows.
  if (s_q == 0) return;

  SparseAttnFwdParams params{};
  params.s_q = s_q;
  params.s_kv = s_kv;
  params.h_q = h_q;
  params.h_kv = h_kv;
  params.d_qk = d_qk;
  params.d_v = static_cast<int>(d_v);
  params.topk = topk;
  params.sm_scale = scale;
  params.sm_scale_div_log2 = scale_log2;
  params.q = static_cast<cutlass::bfloat16_t*>(q.data_ptr());
  params.kv = static_cast<cutlass::bfloat16_t*>(kv.data_ptr());
  params.indices = static_cast<int*>(indices.data_ptr());
  params.attn_sink = attn_sink.has_value()
                         ? static_cast<float*>(attn_sink.value().data_ptr())
                         : nullptr;
  params.topk_length = topk_length.has_value()
                          ? static_cast<int*>(topk_length.value().data_ptr())
                          : nullptr;
  params.stride_q_s_q = q_stride_0;
  params.stride_q_h_q = q_stride_1;
  params.stride_kv_s_kv = kv_stride_0;
  params.stride_kv_h_kv = kv_stride_1;
  params.stride_indices_s_q = idx_stride_0;
  params.stride_indices_h_kv = idx_stride_1;
  params.out = static_cast<cutlass::bfloat16_t*>(out.data_ptr());
  params.max_logits = static_cast<float*>(max_logits.data_ptr());
  params.lse = static_cast<float*>(lse.data_ptr());
  params.num_sm = properties.multiProcessorCount;
  params.stream = get_stream(q.device());
  sm90::run_fwd_kernel(params);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(public_flashmla_sparse_fwd,
                              public_flashmla_sparse_fwd);
