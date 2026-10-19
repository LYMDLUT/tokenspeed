/*
 * Adapted from
 * https://github.com/NVIDIA/TensorRT-LLM/blob/v1.3.0rc2/cpp/tensorrt_llm/kernels/noAuxTcKernels.cu
 * Copyright (c) 2025, The vLLM team.
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// TokenSpeed adaptation of the public vLLM single-group path from commit
// 98dff2a81d747d1dba01a47f939f48c3526d4206. Device functions and tier dispatch
// are retained verbatim; host launches are checked and exposed through TVM FFI.
#include <cmath>
#include <cstdint>
#include <limits>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda/std/cmath>
#include <cuda/std/limits>
#include <tvm/ffi/extra/cuda/device_guard.h>

#include "public_v029_moeTopKFuncs.cuh"
#include "tvm_ffi_utils.h"

namespace cg = cooperative_groups;
namespace vllm {
namespace moe {
static constexpr int WARP_SIZE = 32;
static constexpr int PDLEnableTokens = 16;

// Public grouped_topk_kernels.cu lines 376-431.
template <typename T_OUT, typename T_IN>
__device__ inline T_OUT cuda_cast(T_IN val) {
  return val;
}

template <>
__device__ inline float cuda_cast<float, __nv_bfloat16>(__nv_bfloat16 val) {
  return __bfloat162float(val);
}

template <typename T>
__device__ inline T neg_inf() {
  // cuda::std::numeric_limits<T>::infinity() returns `0` for [T=bf16 or fp16]
  // so we need to cast from fp32
  return cuda_cast<T, float>(-cuda::std::numeric_limits<float>::infinity());
}

template <typename T>
__device__ inline bool is_finite(const T val) {
#if (__CUDACC_VER_MAJOR__ * 10000 + __CUDACC_VER_MINOR__ * 100 >= 120800)
  return cuda::std::isfinite(val);
#else
  return isfinite(cuda_cast<float, T>(val));
#endif
}

// Scoring function enums
enum ScoringFunc {
  SCORING_NONE = 0,    // no activation function
  SCORING_SIGMOID = 1  // apply sigmoid
};

// Adapted from
// https://github.com/NVIDIA/TensorRT-LLM/blob/v1.3.0rc2/cpp/tensorrt_llm/kernels/noAuxTcKernels.cu
__device__ inline float sigmoid_accurate(float x) {
  return 0.5f * tanhf(0.5f * x) + 0.5f;
}

template <typename T>
__device__ inline T apply_sigmoid(T val) {
  float f = cuda_cast<float, T>(val);
  return cuda_cast<T, float>(sigmoid_accurate(f));
}

template <ScoringFunc SF, typename T>
__device__ inline T apply_scoring(T val) {
  if constexpr (SF == SCORING_NONE) {
    return val;
  } else if constexpr (SF == SCORING_SIGMOID) {
    return apply_sigmoid(val);
  } else {
    static_assert(SF == SCORING_NONE || SF == SCORING_SIGMOID,
                  "Unsupported ScoringFunc in apply_scoring");
    return val;
  }
}

// Public grouped_topk_kernels.cu lines 851-1277. Only launch error checks
// have been added to the two host launch sites.
// Adapted from
// https://github.com/flashinfer-ai/flashinfer/blob/06400d062a2d51564bbe781f6f811d0b75ca593e/include/flashinfer/trtllm/fused_moe/RoutingKernelTopK.cuh
namespace single_group_topk {
namespace detail {

static constexpr int BlockDim = 256;
static constexpr uint32_t FullWarpMask = 0xffffffffU;
static constexpr float InvalidScore = -INFINITY;

// TopK-only tuning: use wider workers and keep these tiers on the block path.
template <int MaxNumExperts, int MaxNumTopExperts>
static constexpr bool UseTunedBlockPath =
    MaxNumTopExperts == 16 && (MaxNumExperts == 896 || MaxNumExperts == 1024);

template <typename T, typename BiasT, ScoringFunc SF>
__device__ __forceinline__ void preprocess_score(T input, BiasT correction_bias,
                                                 float& unbiased_score,
                                                 float& selection_score) {
  unbiased_score = 0.0F;
  selection_score = InvalidScore;
  float const input_float = cuda_cast<float, T>(input);
  float const bias = cuda_cast<float, BiasT>(correction_bias);
  if (!is_finite(input_float) || !is_finite(bias)) {
    return;
  }

  float const unbiased = apply_scoring<SF>(input_float);
  float const biased = unbiased + bias;
  if constexpr (SF == SCORING_NONE) {
    if (!is_finite(biased)) {
      return;
    }
  }
  unbiased_score = unbiased;
  selection_score = biased == 0.0F ? 0.0F : biased;
}

template <typename IdxT>
__device__ __forceinline__ void write_outputs(
    cg::thread_block_tile<WARP_SIZE> const& warp, float lane_selection_score,
    float lane_unbiased, int32_t lane_expert, int32_t lane, int32_t token,
    int32_t topk, float* topk_values, IdxT* topk_indices, bool renormalize,
    float routed_scaling_factor) {
  bool const finite_selection =
      lane < topk && lane_selection_score != InvalidScore;
  lane_unbiased = finite_selection ? lane_unbiased : 0.0F;
  unsigned const finite_mask = __ballot_sync(FullWarpMask, finite_selection);
  float const sum = cg::reduce(warp, lane_unbiased, cg::plus<float>{});

  if (lane < topk) {
    float output = 0.0F;
    if (finite_mask == 0) {
      if (renormalize) {
        output = 1.0F / static_cast<float>(topk);
      }
    } else if (finite_selection) {
      float scale = routed_scaling_factor;
      if (renormalize) {
        scale /= sum + 1e-20F;
      }
      output = lane_unbiased * scale;
    }

    int64_t const output_index = int64_t{token} * topk + lane;
    topk_values[output_index] = output;
    topk_indices[output_index] = static_cast<IdxT>(lane_expert);
  }
}

template <typename T, typename BiasT, typename IdxT, ScoringFunc SF,
          int MaxNumExperts, int MaxNumTopExperts>
__global__ void __launch_bounds__(BlockDim)
    single_group_topk_block_kernel(T const* scores, float* topk_values,
                                   IdxT* topk_indices, BiasT const* bias,
                                   int64_t num_experts, int64_t topk,
                                   bool renormalize,
                                   float routed_scaling_factor,
                                   bool enable_pdl) {
  static constexpr int NumChunks = (MaxNumExperts + WARP_SIZE - 1) / WARP_SIZE;
  static constexpr int WorkerValuesPerLane =
      UseTunedBlockPath<MaxNumExperts, MaxNumTopExperts> ? 8 : 4;
  static constexpr int ExpertsPerWorkerWarp = WorkerValuesPerLane * WARP_SIZE;
  using LaneOwnedRange =
      reduce_topk::HighExpertLaneOwnedTopKRange<MaxNumExperts,
                                                MaxNumTopExperts>;
  static constexpr int NumWorkerWarps =
      (MaxNumExperts + ExpertsPerWorkerWarp - 1) / ExpertsPerWorkerWarp;
  static constexpr int NumIntermediate = NumWorkerWarps * MaxNumTopExperts;
  static constexpr int MergeValuesPerLane =
      (NumIntermediate + WARP_SIZE - 1) / WARP_SIZE;
  static constexpr bool LaneOwnedResourcesFit =
      NumWorkerWarps <= BlockDim / WARP_SIZE && MergeValuesPerLane <= 64;
  static constexpr bool UseHierarchicalLaneTopK =
      LaneOwnedRange::kEnabled && LaneOwnedResourcesFit;

  static_assert(NumChunks <= 64);
  static_assert(MaxNumTopExperts <= WARP_SIZE);

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  if (enable_pdl) {
    cudaGridDependencySynchronize();
  }
#endif

  __shared__ float __attribute((aligned(128))) biased_scores[MaxNumExperts];
  __shared__ float __attribute((aligned(128))) unbiased_scores[MaxNumExperts];

  int32_t const token = static_cast<int32_t>(blockIdx.x);
  int32_t const lane = static_cast<int32_t>(threadIdx.x) % WARP_SIZE;
  int32_t const warp_id = static_cast<int32_t>(threadIdx.x) / WARP_SIZE;
  int32_t const num_experts_i32 = static_cast<int32_t>(num_experts);
  int32_t const topk_i32 = static_cast<int32_t>(topk);
  T const* token_scores = scores + int64_t{token} * num_experts;

  for (int32_t expert = static_cast<int32_t>(threadIdx.x);
       expert < num_experts_i32; expert += BlockDim) {
    preprocess_score<T, BiasT, SF>(token_scores[expert], bias[expert],
                                   unbiased_scores[expert],
                                   biased_scores[expert]);
  }
  __syncthreads();

  auto warp = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());

  if constexpr (UseHierarchicalLaneTopK) {
    __shared__ float
        __attribute((aligned(128))) intermediate_scores[NumIntermediate];
    __shared__ int32_t
        __attribute((aligned(128))) intermediate_indices[NumIntermediate];

    if (warp_id < NumWorkerWarps) {
      float local_scores[WorkerValuesPerLane];
      int32_t local_indices[WorkerValuesPerLane];
#pragma unroll
      for (int index = 0; index < WorkerValuesPerLane; ++index) {
        int32_t const expert =
            warp_id * ExpertsPerWorkerWarp + index * WARP_SIZE + lane;
        local_scores[index] =
            expert < num_experts_i32 ? biased_scores[expert] : InvalidScore;
        local_indices[index] = expert;
      }

      float lane_score;
      int32_t lane_expert;
      reduce_topk::reduceTopKForLane<MaxNumTopExperts>(
          warp, lane_score, lane_expert, local_scores, local_indices,
          InvalidScore, lane);
      if (lane < MaxNumTopExperts) {
        int32_t const intermediate = warp_id * MaxNumTopExperts + lane;
        bool const active = lane < topk_i32;
        intermediate_scores[intermediate] = active ? lane_score : InvalidScore;
        intermediate_indices[intermediate] =
            active ? lane_expert : MaxNumExperts;
      }
    }
    __syncthreads();

    if (warp_id != 0) {
      return;
    }

    float merge_scores[MergeValuesPerLane];
    int32_t merge_indices[MergeValuesPerLane];
#pragma unroll
    for (int index = 0; index < MergeValuesPerLane; ++index) {
      int32_t const intermediate = index * WARP_SIZE + lane;
      bool const active = intermediate < NumIntermediate;
      merge_scores[index] =
          active ? intermediate_scores[intermediate] : InvalidScore;
      merge_indices[index] =
          active ? intermediate_indices[intermediate] : MaxNumExperts;
    }

    float lane_score;
    int32_t lane_expert;
    reduce_topk::reduceTopKForLane<MaxNumTopExperts>(
        warp, lane_score, lane_expert, merge_scores, merge_indices,
        InvalidScore, lane);
    float const lane_unbiased =
        lane < topk_i32 && lane_expert >= 0 && lane_expert < num_experts_i32
            ? unbiased_scores[lane_expert]
            : 0.0F;
    write_outputs(warp, lane_score, lane_unbiased, lane_expert, lane, token,
                  topk_i32, topk_values, topk_indices, renormalize,
                  routed_scaling_factor);
  } else {
    if (warp_id != 0) {
      return;
    }

    float local_scores[NumChunks];
    int32_t local_indices[NumChunks];
#pragma unroll
    for (int index = 0; index < NumChunks; ++index) {
      int32_t const expert = index * WARP_SIZE + lane;
      local_scores[index] =
          expert < num_experts_i32 ? biased_scores[expert] : InvalidScore;
      local_indices[index] = expert;
    }

    float top_scores[MaxNumTopExperts];
    int32_t top_experts[MaxNumTopExperts];
    reduce_topk::reduceTopK(warp, top_scores, top_experts, local_scores,
                            local_indices, InvalidScore, topk_i32);
    float const lane_score = lane < topk_i32 ? top_scores[lane] : InvalidScore;
    int32_t const lane_expert = lane < topk_i32 ? top_experts[lane] : -1;
    float const lane_unbiased =
        lane < topk_i32 && lane_expert >= 0 && lane_expert < num_experts_i32
            ? unbiased_scores[lane_expert]
            : 0.0F;
    write_outputs(warp, lane_score, lane_unbiased, lane_expert, lane, token,
                  topk_i32, topk_values, topk_indices, renormalize,
                  routed_scaling_factor);
  }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  if (enable_pdl) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
#endif
}

template <int MaxNumExperts>
struct WarpTopKLaunchConfig {
  static constexpr int DefaultBlockDim =
      MaxNumExperts <= 1024 ? MaxNumExperts : 1024;
  static constexpr int BlockDim = DefaultBlockDim > 256 ? 256 : DefaultBlockDim;
  static constexpr int NumWarps = BlockDim / WARP_SIZE;
  static constexpr int MaxBlockScale =
      (DefaultBlockDim + BlockDim - 1) / BlockDim;
  static constexpr int MaxBlocks = 1024 * MaxBlockScale;

  static_assert(BlockDim % WARP_SIZE == 0);

  static uint32_t grid_dim(int64_t num_tokens) {
    int64_t const token_blocks = (num_tokens + NumWarps - 1) / NumWarps;
    int64_t const selected =
        token_blocks < MaxBlocks ? token_blocks : MaxBlocks;
    return static_cast<uint32_t>(selected > 0 ? selected : 1);
  }
};

template <typename T, typename BiasT, typename IdxT, ScoringFunc SF,
          int MaxNumExperts, int MaxNumTopExperts>
__global__ void __launch_bounds__(WarpTopKLaunchConfig<MaxNumExperts>::BlockDim)
    single_group_topk_warp_kernel(T const* scores, float* topk_values,
                                  IdxT* topk_indices, BiasT const* bias,
                                  int64_t num_tokens, int64_t num_experts,
                                  int64_t topk, bool renormalize,
                                  float routed_scaling_factor,
                                  bool enable_pdl) {
  static constexpr int NumChunks = (MaxNumExperts + WARP_SIZE - 1) / WARP_SIZE;
  static constexpr int WarpBlockDim =
      WarpTopKLaunchConfig<MaxNumExperts>::BlockDim;
  using LaneOwnedRange =
      reduce_topk::HighExpertLaneOwnedTopKRange<MaxNumExperts,
                                                MaxNumTopExperts>;

  static_assert(NumChunks <= 64);
  static_assert(MaxNumTopExperts <= WARP_SIZE);

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  if (enable_pdl) {
    cudaGridDependencySynchronize();
  }
#endif

  int32_t const lane = static_cast<int32_t>(threadIdx.x) % WARP_SIZE;
  int32_t const warp_id = static_cast<int32_t>(threadIdx.x) / WARP_SIZE;
  int32_t const global_warp =
      static_cast<int32_t>(blockIdx.x) * WarpBlockDim / WARP_SIZE + warp_id;
  int32_t const global_warp_stride =
      static_cast<int32_t>(gridDim.x) * WarpBlockDim / WARP_SIZE;
  int32_t const num_experts_i32 = static_cast<int32_t>(num_experts);
  int32_t const topk_i32 = static_cast<int32_t>(topk);
  auto warp = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());

  for (int32_t token = global_warp; token < num_tokens;
       token += global_warp_stride) {
    T const* token_scores = scores + int64_t{token} * num_experts;
    float local_scores[NumChunks];
    int32_t local_indices[NumChunks];
#pragma unroll
    for (int index = 0; index < NumChunks; ++index) {
      int32_t const expert = index * WARP_SIZE + lane;
      float unbiased;
      float selection;
      if (expert < num_experts_i32) {
        preprocess_score<T, BiasT, SF>(token_scores[expert], bias[expert],
                                       unbiased, selection);
      } else {
        selection = InvalidScore;
      }
      local_scores[index] = selection;
      local_indices[index] = expert;
    }

    float lane_score;
    int32_t lane_expert;
    if constexpr (LaneOwnedRange::kEnabled) {
      reduce_topk::reduceTopKForLane<MaxNumTopExperts>(
          warp, lane_score, lane_expert, local_scores, local_indices,
          InvalidScore, lane);
    } else {
      float top_scores[MaxNumTopExperts];
      int32_t top_experts[MaxNumTopExperts];
      reduce_topk::reduceTopK(warp, top_scores, top_experts, local_scores,
                              local_indices, InvalidScore, topk_i32);
      lane_score = lane < topk_i32 ? top_scores[lane] : InvalidScore;
      lane_expert = lane < topk_i32 ? top_experts[lane] : -1;
    }

    float lane_unbiased = 0.0F;
    if (lane < topk_i32 && lane_expert >= 0 && lane_expert < num_experts_i32) {
      lane_unbiased = lane_score - cuda_cast<float, BiasT>(bias[lane_expert]);
    }
    write_outputs(warp, lane_score, lane_unbiased, lane_expert, lane, token,
                  topk_i32, topk_values, topk_indices, renormalize,
                  routed_scaling_factor);
  }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  if (enable_pdl) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
#endif
}

template <int Experts, int TopK>
struct Tier {
  static constexpr int kExperts = Experts;
  static constexpr int kTopK = TopK;
};

template <typename... Tiers>
struct TierList {};

using SigmoidBiasTiers =
    TierList<Tier<128, 8>, Tier<256, 8>, Tier<384, 8>, Tier<512, 8>,
             Tier<512, 22>, Tier<768, 16>, Tier<896, 16>, Tier<1024, 16>>;

using PrecomputedSoftmaxBiasTiers =
    TierList<Tier<128, 4>, Tier<128, 8>, Tier<160, 8>, Tier<256, 8>,
             Tier<256, 16>, Tier<512, 8>, Tier<512, 16>, Tier<512, 22>,
             Tier<512, 32>, Tier<576, 8>, Tier<768, 16>, Tier<896, 16>,
             Tier<1024, 16>>;

template <typename T, typename BiasT, typename IdxT, ScoringFunc SF,
          int MaxNumExperts, int MaxNumTopExperts>
void launch(T* scores, float* topk_values, IdxT* topk_indices,
            BiasT const* bias, int64_t num_tokens, int64_t num_experts,
            int64_t topk, bool renormalize, double routed_scaling_factor,
            bool enable_pdl, cudaLaunchConfig_t& config) {
  config.dynamicSmemBytes = 0;
  bool const use_block_kernel =
      UseTunedBlockPath<MaxNumExperts, MaxNumTopExperts> ||
      MaxNumExperts > 1024 || num_experts >= 1024 ||
      (num_experts >= 256 && num_tokens <= 1024);
  if (use_block_kernel) {
    config.gridDim = static_cast<uint32_t>(num_tokens);
    config.blockDim = BlockDim;
    TVM_FFI_ICHECK_EQ(
        cudaLaunchKernelEx(
            &config,
            &single_group_topk_block_kernel<T, BiasT, IdxT, SF, MaxNumExperts,
                                            MaxNumTopExperts>,
            scores, topk_values, topk_indices, bias, num_experts, topk, renormalize,
            static_cast<float>(routed_scaling_factor), enable_pdl),
        cudaSuccess);
  } else {
    using WarpConfig = WarpTopKLaunchConfig<MaxNumExperts>;
    config.gridDim = WarpConfig::grid_dim(num_tokens);
    config.blockDim = WarpConfig::BlockDim;
    TVM_FFI_ICHECK_EQ(
        cudaLaunchKernelEx(
            &config,
            &single_group_topk_warp_kernel<T, BiasT, IdxT, SF, MaxNumExperts,
                                           MaxNumTopExperts>,
            scores, topk_values, topk_indices, bias, num_tokens, num_experts, topk,
            renormalize, static_cast<float>(routed_scaling_factor), enable_pdl),
        cudaSuccess);
  }
}

template <typename T, typename BiasT, typename IdxT, ScoringFunc SF>
bool dispatch(TierList<>*, T*, float*, IdxT*, BiasT const*, int64_t, int64_t,
              int64_t, bool, double, bool, cudaLaunchConfig_t&) {
  return false;
}

template <typename T, typename BiasT, typename IdxT, ScoringFunc SF,
          typename First, typename... Rest>
bool dispatch(TierList<First, Rest...>*, T* scores, float* topk_values,
              IdxT* topk_indices, BiasT const* bias, int64_t num_tokens,
              int64_t num_experts, int64_t topk, bool renormalize,
              double routed_scaling_factor, bool enable_pdl,
              cudaLaunchConfig_t& config) {
  if (num_experts <= First::kExperts && topk <= First::kTopK) {
    launch<T, BiasT, IdxT, SF, First::kExperts, First::kTopK>(
        scores, topk_values, topk_indices, bias, num_tokens, num_experts, topk,
        renormalize, routed_scaling_factor, enable_pdl, config);
    return true;
  }
  return dispatch<T, BiasT, IdxT, SF>(
      static_cast<TierList<Rest...>*>(nullptr), scores, topk_values,
      topk_indices, bias, num_tokens, num_experts, topk, renormalize,
      routed_scaling_factor, enable_pdl, config);
}

}  // namespace detail

template <typename T, typename BiasT, typename IdxT, ScoringFunc SF>
bool invoke(T* scores, float* topk_values, IdxT* topk_indices,
            BiasT const* bias, int64_t num_tokens, int64_t num_experts,
            int64_t topk, bool renormalize, double routed_scaling_factor,
            bool enable_pdl, cudaLaunchConfig_t& config) {
  static_assert(SF == SCORING_NONE || SF == SCORING_SIGMOID);
  if constexpr (SF == SCORING_SIGMOID) {
    return detail::dispatch<T, BiasT, IdxT, SF>(
        static_cast<detail::SigmoidBiasTiers*>(nullptr), scores, topk_values,
        topk_indices, bias, num_tokens, num_experts, topk, renormalize,
        routed_scaling_factor, enable_pdl, config);
  } else {
    return detail::dispatch<T, BiasT, IdxT, SF>(
        static_cast<detail::PrecomputedSoftmaxBiasTiers*>(nullptr), scores,
        topk_values, topk_indices, bias, num_tokens, num_experts, topk,
        renormalize, routed_scaling_factor, enable_pdl, config);
  }
}

}  // namespace single_group_topk

}  // namespace moe
}  // namespace vllm

void public_v029_single_group_sigmoid_topk(
    TensorView scores, TensorView bias, TensorView topk_values,
    TensorView topk_indices, int64_t topk, bool renormalize,
    double routed_scaling_factor) {
  CHECK_INPUT_AND_TYPE(scores, dl_float32);
  CHECK_INPUT_AND_TYPE(bias, dl_float32);
  CHECK_INPUT_AND_TYPE(topk_values, dl_float32);
  CHECK_INPUT_AND_TYPE(topk_indices, dl_int32);
  CHECK_DIM(2, scores);
  CHECK_DIM(1, bias);
  CHECK_DIM(2, topk_values);
  CHECK_DIM(2, topk_indices);
  CHECK_DEVICE(scores, bias);
  CHECK_DEVICE(scores, topk_values);
  CHECK_DEVICE(scores, topk_indices);

  const int64_t num_tokens = scores.size(0);
  const int64_t num_experts = scores.size(1);
  // The public warp path increments an INT32 token counter after each row.
  // Reserve its maximum grid stride, even for the final loop increment.
  constexpr int64_t kMaxTokens = std::numeric_limits<int32_t>::max() - 32768;
  TVM_FFI_ICHECK(num_tokens >= 0 && num_tokens <= kMaxTokens);
  TVM_FFI_ICHECK(num_experts >= 1 && num_experts <= 1024);
  TVM_FFI_ICHECK(topk >= 1 && topk <= num_experts);
  TVM_FFI_ICHECK((num_experts <= 512 && topk <= 22) ||
                 (num_experts > 512 && topk <= 16))
      << "Shape is outside the public sigmoid single-group tiers";
  TVM_FFI_ICHECK(std::isfinite(routed_scaling_factor) &&
                 std::abs(routed_scaling_factor) <=
                     static_cast<double>(std::numeric_limits<float>::max()))
      << "Scale must be within the finite FP32 range";
  TVM_FFI_ICHECK_EQ(bias.size(0), num_experts);
  TVM_FFI_ICHECK_EQ(topk_values.size(0), num_tokens);
  TVM_FFI_ICHECK_EQ(topk_values.size(1), topk);
  TVM_FFI_ICHECK_EQ(topk_indices.size(0), num_tokens);
  TVM_FFI_ICHECK_EQ(topk_indices.size(1), topk);
  if (num_tokens == 0) {
    return;
  }

  ffi::CUDADeviceGuard guard(scores.device().device_id);
  auto stream = get_stream(scores.device());
  const bool enable_pdl = num_tokens <= vllm::moe::PDLEnableTokens;
  cudaLaunchConfig_t config{};
  config.stream = stream;
  cudaLaunchAttribute attrs[1]{};
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  config.numAttrs = 1;
  config.attrs = attrs;
  const bool launched = vllm::moe::single_group_topk::invoke<
      float, float, int32_t, vllm::moe::SCORING_SIGMOID>(
      static_cast<float*>(scores.data_ptr()),
      static_cast<float*>(topk_values.data_ptr()),
      static_cast<int32_t*>(topk_indices.data_ptr()),
      static_cast<const float*>(bias.data_ptr()), num_tokens, num_experts,
      topk, renormalize, routed_scaling_factor, enable_pdl, config);
  TVM_FFI_ICHECK(launched) << "No public sigmoid single-group tier matched";
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(public_v029_single_group_sigmoid_topk,
                              public_v029_single_group_sigmoid_topk);
