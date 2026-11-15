/*
 * Copyright (c) 2026 by FlashInfer team.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <limits>
#include <map>
#include <memory>

#include "flashinfer/trtllm/batched_gemm/KernelRunner.h"
#include "flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/trtllm/gen/SfLayoutDecl.h"
#include "flashinfer/trtllm/fused_moe/runner.h"
#include "tvm_ffi_utils.h"

namespace flashinfer {
namespace {

namespace tk = tensorrt_llm::kernels;
namespace btg = batchedGemm::trtllm::gen;
using Runner = tk::TrtllmGenBatchedGemmRunner;
using tvm::ffi::Array;

struct RoutedGemmShape {
  int32_t tokens, experts, n, k, max_ctas, output_rows, activation_rows;
};

RoutedGemmShape checked_shape(int64_t tokens, int64_t experts, int64_t top_k, int64_t n, int64_t k,
                              int64_t tile) {
  constexpr auto limit = std::numeric_limits<int32_t>::max();
  for (auto value : {tokens, experts, top_k, n, k}) {
    TVM_FFI_ICHECK(value > 0 && value <= limit) << "GEMM dimensions must fit positive int32.";
  }
  TVM_FFI_ICHECK_LE(top_k, experts);
  TVM_FFI_ICHECK(tile == 8 || tile == 16 || tile == 32 || tile == 64);
  TVM_FFI_ICHECK(n % 128 == 0 && k % 128 == 0)
      << "Shuffled MXFP4 GEMM requires N and K divisible by 128.";
  auto const pairs = tokens * top_k;
  TVM_FFI_ICHECK_LE(pairs, limit);
  auto const filled = std::min(experts, pairs);
  auto const max_ctas = filled + (pairs - filled) / tile;
  auto const output_rows = (max_ctas * tile + 127) / 128 * 128;
  // Match Routing::maybeGetMinTokenCount: native activation TMA storage is at least 128 KiB.
  auto const minimum_rows = ((128 * 1024 + k - 1) / k + 127) / 128 * 128;
  auto const activation_rows = std::max(output_rows, minimum_rows);
  TVM_FFI_ICHECK_LE(activation_rows, limit);
  return {static_cast<int32_t>(tokens),
          static_cast<int32_t>(experts),
          static_cast<int32_t>(n),
          static_cast<int32_t>(k),
          static_cast<int32_t>(max_ctas),
          static_cast<int32_t>(output_rows),
          static_cast<int32_t>(activation_rows)};
}

Runner& get_runner(int device, int tile) {
  thread_local std::map<std::pair<int, int>, std::unique_ptr<Runner>> runners;
  auto& runner = runners[{device, tile}];
  if (!runner) {
    tk::TrtllmGenBatchedGemmRunnerOptions options;
    options.dtypeA = btg::Dtype::MxE2m1;
    options.dtypeB = btg::Dtype::MxE4m3;
    options.dtypeC = btg::Dtype::Bfloat16;
    options.transposeMmaOutput = true;
    options.useShuffledMatrix = true;
    options.tileSize = tile;
    runner = std::make_unique<Runner>(options);
  }
  return *runner;
}

void check_tensor(TensorView value, TensorView anchor, DLDataType dtype) {
  CHECK_INPUT(value);
  CHECK_DEVICE(value, anchor);
  TVM_FFI_ICHECK_EQ(value.dtype(), dtype);
}

}  // namespace

// Rows contain tactic, scale layout, workspace bytes, activation rows and output rows.
Array<Array<int64_t>> trtllm_mxfp8_routed_gemm_configs(TensorView device_anchor, int64_t tokens,
                                                       int64_t experts, int64_t top_k, int64_t n,
                                                       int64_t k, int64_t tile) {
  CHECK_CUDA(device_anchor);
  auto const shape = checked_shape(tokens, experts, top_k, n, k, tile);
  ffi::CUDADeviceGuard guard(device_anchor.device().device_id);
  auto& runner = get_runner(device_anchor.device().device_id, tile);
  Array<Array<int64_t>> result;
  for (auto tactic : runner.getValidConfigIndices(shape.tokens, shape.n, shape.k, {}, shape.tokens,
                                                  shape.experts, shape.max_ctas)) {
    auto const layout = static_cast<int64_t>(runner.getSfLayoutB(tactic));
    if (layout != 0 && layout != 1 && layout != 3) continue;
    auto const bytes = runner.getWorkspaceSizeInBytes(
        shape.tokens, shape.n, shape.k, {}, shape.tokens, shape.experts, shape.max_ctas, tactic);
    result.push_back(
        {tactic, layout, static_cast<int64_t>(bytes), shape.activation_rows, shape.output_rows});
  }
  return result;
}

void trtllm_mxfp8_routed_gemm(TensorView input, TensorView input_scale, TensorView weight,
                              TensorView weight_scale, TensorView output, TensorView workspace,
                              TensorView padded_count, TensorView cta_expert, TensorView cta_limit,
                              TensorView num_ctas, int64_t tokens, int64_t top_k, int64_t tile,
                              int64_t tactic) {
  CHECK_DIM(3, weight);
  auto const shape =
      checked_shape(tokens, weight.size(0), top_k, weight.size(1), weight.size(2) * 2, tile);
  for (auto tensor : {input, input_scale, weight, weight_scale, workspace}) {
    check_tensor(tensor, input, dl_uint8);
  }
  check_tensor(output, input, dl_bfloat16);
  for (auto tensor : {padded_count, cta_expert, cta_limit, num_ctas}) {
    check_tensor(tensor, input, dl_int32);
    CHECK_DIM(1, tensor);
  }
  CHECK_DIM(2, input);
  CHECK_DIM(2, output);
  TVM_FFI_ICHECK_EQ(input.size(1), shape.k);
  TVM_FFI_ICHECK_GE(input.size(0), shape.activation_rows);
  TVM_FFI_ICHECK_EQ(output.size(1), shape.n);
  TVM_FFI_ICHECK_GE(output.size(0), shape.output_rows);
  TVM_FFI_ICHECK_GE(input_scale.numel(), input.size(0) * (shape.k / 32));
  TVM_FFI_ICHECK_EQ(weight_scale.numel(), weight.size(0) * weight.size(1) * (shape.k / 32));
  TVM_FFI_ICHECK_EQ(padded_count.numel(), 1);
  TVM_FFI_ICHECK_EQ(num_ctas.numel(), 1);
  TVM_FFI_ICHECK_GE(cta_expert.numel(), shape.max_ctas);
  TVM_FFI_ICHECK_GE(cta_limit.numel(), shape.max_ctas);
  TVM_FFI_ICHECK(tactic >= 0 && tactic <= std::numeric_limits<int32_t>::max());
  ffi::CUDADeviceGuard guard(input.device().device_id);
  auto& runner = get_runner(input.device().device_id, tile);
  TVM_FFI_ICHECK(runner.isValidConfigIndex(tactic, shape.tokens, shape.n, shape.k, {}, shape.tokens,
                                           shape.experts, shape.max_ctas))
      << "Invalid MXFP4/MXFP8 routed GEMM tactic.";
  auto const layout = static_cast<int64_t>(runner.getSfLayoutB(tactic));
  TVM_FFI_ICHECK(layout == 0 || layout == 1 || layout == 3)
      << "Unsupported activation scale layout.";
  auto const bytes = runner.getWorkspaceSizeInBytes(
      shape.tokens, shape.n, shape.k, {}, shape.tokens, shape.experts, shape.max_ctas, tactic);
  TVM_FFI_ICHECK_GE(workspace.numel(), bytes);
  runner.run(shape.tokens, shape.n, shape.k, {}, shape.tokens, shape.experts, shape.max_ctas,
             input.data_ptr(), input_scale.data_ptr(), weight.data_ptr(), weight_scale.data_ptr(),
             nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
             output.data_ptr(), nullptr, nullptr, static_cast<int32_t*>(padded_count.data_ptr()),
             static_cast<int32_t*>(cta_expert.data_ptr()),
             static_cast<int32_t*>(cta_limit.data_ptr()),
             static_cast<int32_t*>(num_ctas.data_ptr()), nullptr, workspace.data_ptr(),
             get_stream(input.device()), input.device().device_id, tactic, false);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_mxfp8_routed_gemm_configs, trtllm_mxfp8_routed_gemm_configs);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_mxfp8_routed_gemm, trtllm_mxfp8_routed_gemm);

void trtllm_mxfp8_routed_gemm_route(TensorView ids, TensorView weights, TensorView histogram,
                                    TensorView padded_count, TensorView indices, TensorView inverse,
                                    TensorView expert_counts, TensorView cta_expert,
                                    TensorView cta_limit, TensorView num_ctas, int64_t experts,
                                    int64_t tile) {
  CHECK_DIM(2, ids);
  auto const shape = checked_shape(ids.size(0), experts, ids.size(1), 128, 128, tile);
  check_tensor(ids, ids, dl_int32);
  check_tensor(weights, ids, dl_bfloat16);
  CHECK_DIM(2, weights);
  TVM_FFI_ICHECK_EQ(weights.size(0), ids.size(0));
  TVM_FFI_ICHECK_EQ(weights.size(1), ids.size(1));
  for (auto tensor : {histogram, padded_count, indices, inverse, expert_counts, cta_expert,
                      cta_limit, num_ctas}) {
    check_tensor(tensor, ids, dl_int32);
    CHECK_DIM(1, tensor);
  }
  TVM_FFI_ICHECK_GE(histogram.numel(), std::max<int64_t>(experts * 2, 512));
  TVM_FFI_ICHECK_EQ(padded_count.numel(), 1);
  TVM_FFI_ICHECK_GE(indices.numel(), ids.numel());
  // The native routed GEMM can read one guard element beyond the inverse map.
  TVM_FFI_ICHECK_GE(inverse.numel(), shape.output_rows + 1);
  TVM_FFI_ICHECK_GE(expert_counts.numel(), experts);
  TVM_FFI_ICHECK_GE(cta_expert.numel(), shape.max_ctas);
  TVM_FFI_ICHECK_GE(cta_limit.numel(), shape.max_ctas);
  TVM_FFI_ICHECK_EQ(num_ctas.numel(), 1);
  ffi::CUDADeviceGuard guard(ids.device().device_id);
  tk::trtllmgen_moe::Routing::Runner runner(tile);
  auto ptr = [](TensorView tensor) { return static_cast<int32_t*>(tensor.data_ptr()); };
  runner.run(nullptr, nullptr, shape.tokens, shape.experts, ids.size(1), 0, 0, 0, 0, shape.experts,
             1.0f, ptr(ids), ptr(histogram), ptr(padded_count), ptr(indices), nullptr, ptr(inverse),
             ptr(ids), weights.data_ptr(), ptr(expert_counts), ptr(cta_expert), ptr(cta_limit),
             ptr(num_ctas), btg::Dtype::Bfloat16, btg::Dtype::Bfloat16, false, false,
             tk::trtllmgen_moe::Routing::RoutingMethodType::TopK, get_stream(ids.device()),
             btg::Dtype::Bfloat16, false, nullptr, false);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_mxfp8_routed_gemm_route, trtllm_mxfp8_routed_gemm_route);

}  // namespace flashinfer
