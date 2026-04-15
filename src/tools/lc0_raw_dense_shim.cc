#include "tools/lc0_raw_dense_shim.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>

#include "neural/factory.h"
#include "neural/network.h"
#include "neural/shared_params.h"
#include "utils/bititer.h"
#include "utils/exception.h"
#include "utils/optionsdict.h"
#include "utils/optionsparser.h"

namespace {

using lczero::Exception;
using lczero::InputPlanes;
using lczero::kInputPlanes;

constexpr int kPolicySize = LC0_RAW_DENSE_POLICY_SIZE;
constexpr int kInputSquares = LC0_RAW_DENSE_INPUT_SQUARES;
constexpr int kDenseInputSize =
    LC0_RAW_DENSE_INPUT_SQUARES * LC0_RAW_DENSE_INPUT_PLANES;

void SetError(char* err_buf, size_t err_buf_len, std::string_view message) {
  if (!err_buf || err_buf_len == 0) return;
  const size_t copy_len = std::min(err_buf_len - 1, message.size());
  if (copy_len > 0) {
    std::memcpy(err_buf, message.data(), copy_len);
  }
  err_buf[copy_len] = '\0';
}

bool SupportsRawDenseInput(lczero::NetworkCapabilities caps) {
  return caps.input_format == pblczero::NetworkFormat::INPUT_CLASSICAL_112_PLANE;
}

InputPlanes DenseTrainingBytesToInputPlanes(const uint8_t* dense_input) {
  InputPlanes result(kInputPlanes);
  for (int plane = 0; plane < kInputPlanes; ++plane) {
    uint64_t training_mask = 0ULL;
    std::optional<uint8_t> plane_value;
    for (int sq = 0; sq < kInputSquares; ++sq) {
      const uint8_t v = dense_input[sq * kInputPlanes + plane];
      if (v == 0) continue;
      if (!plane_value.has_value()) {
        plane_value = v;
      } else if (*plane_value != v) {
        throw Exception("Dense input plane " + std::to_string(plane) +
                        " has mixed non-zero values.");
      }
      training_mask |= (1ULL << sq);
    }
    result[plane].mask = lczero::ReverseBitsInBytes(training_mask);
    result[plane].value =
        plane_value.has_value() ? static_cast<float>(*plane_value) : 1.0f;
  }
  return result;
}

}  // namespace

struct lc0_raw_dense_handle {
  lc0_raw_dense_handle()
      : options_parser(), options(options_parser.GetMutableOptions()) {}

  std::mutex mutex;
  lczero::OptionsParser options_parser;
  lczero::OptionsDict* options;
  std::unique_ptr<lczero::Network> network;
  std::optional<lczero::NetworkCapabilities> caps;
};

extern "C" {

int lc0_raw_dense_create(const char* weights_path, const char* backend,
                         const char* backend_options,
                         lc0_raw_dense_handle** out_handle, char* err_buf,
                         size_t err_buf_len) {
  SetError(err_buf, err_buf_len, "");
  if (!out_handle) {
    SetError(err_buf, err_buf_len, "out_handle must not be null.");
    return LC0_RAW_DENSE_INVALID_ARGUMENT;
  }
  *out_handle = nullptr;

  try {
    auto handle = std::make_unique<lc0_raw_dense_handle>();
    lczero::SharedBackendParams::Populate(&handle->options_parser);

    if (weights_path && weights_path[0] != '\0') {
      handle->options->Set<std::string>(lczero::SharedBackendParams::kWeightsId,
                                        weights_path);
    }
    if (backend && backend[0] != '\0') {
      handle->options->Set<std::string>(lczero::SharedBackendParams::kBackendId,
                                        backend);
    }
    if (backend_options && backend_options[0] != '\0') {
      handle->options->Set<std::string>(
          lczero::SharedBackendParams::kBackendOptionsId, backend_options);
    }

    handle->network = lczero::NetworkFactory::LoadNetwork(*handle->options);
    if (!handle->network) {
      throw Exception("NetworkFactory::LoadNetwork returned null.");
    }

    handle->caps = handle->network->GetCapabilities();
    if (!SupportsRawDenseInput(*handle->caps)) {
      throw Exception("Unsupported network input format " +
                      std::to_string(static_cast<int>(handle->caps->input_format)) +
                      "; expected INPUT_CLASSICAL_112_PLANE.");
    }

    *out_handle = handle.release();
    return LC0_RAW_DENSE_OK;
  } catch (const std::exception& e) {
    SetError(err_buf, err_buf_len, e.what());
    return LC0_RAW_DENSE_INIT_FAILED;
  } catch (...) {
    SetError(err_buf, err_buf_len, "Unknown exception in lc0_raw_dense_create.");
    return LC0_RAW_DENSE_INTERNAL_ERROR;
  }
}

void lc0_raw_dense_destroy(lc0_raw_dense_handle* handle) { delete handle; }

int lc0_raw_dense_infer(lc0_raw_dense_handle* handle,
                        const uint8_t* input_planes_u8, size_t batch_size,
                        float* out_q, float* out_d, float* out_m,
                        float* out_policy, char* err_buf,
                        size_t err_buf_len) {
  SetError(err_buf, err_buf_len, "");
  if (!handle) {
    SetError(err_buf, err_buf_len, "handle must not be null.");
    return LC0_RAW_DENSE_INVALID_ARGUMENT;
  }
  if (!input_planes_u8) {
    SetError(err_buf, err_buf_len, "input_planes_u8 must not be null.");
    return LC0_RAW_DENSE_INVALID_ARGUMENT;
  }
  if (batch_size == 0) {
    SetError(err_buf, err_buf_len, "batch_size must be > 0.");
    return LC0_RAW_DENSE_INVALID_ARGUMENT;
  }
  if (batch_size > static_cast<size_t>(std::numeric_limits<int>::max())) {
    SetError(err_buf, err_buf_len, "batch_size exceeds int range.");
    return LC0_RAW_DENSE_INVALID_ARGUMENT;
  }

  try {
    std::lock_guard<std::mutex> lock(handle->mutex);
    auto computation = handle->network->NewComputation();
    for (size_t sample = 0; sample < batch_size; ++sample) {
      computation->AddInput(
          DenseTrainingBytesToInputPlanes(input_planes_u8 +
                                          sample * kDenseInputSize));
    }
    computation->ComputeBlocking();

    if (computation->GetBatchSize() != static_cast<int>(batch_size)) {
      throw Exception("Backend returned batch size " +
                      std::to_string(computation->GetBatchSize()) +
                      " for requested batch size " + std::to_string(batch_size) +
                      ".");
    }

    const auto& caps = *handle->caps;
    for (size_t sample = 0; sample < batch_size; ++sample) {
      const int sample_idx = static_cast<int>(sample);
      if (out_q) out_q[sample] = computation->GetQVal(sample_idx);
      if (out_d) {
        out_d[sample] = caps.has_wdl() ? computation->GetDVal(sample_idx) : 0.0f;
      }
      if (out_m) {
        out_m[sample] = caps.has_mlh() ? computation->GetMVal(sample_idx) : 0.0f;
      }
      if (out_policy) {
        float* dst = out_policy + sample * kPolicySize;
        for (int move = 0; move < kPolicySize; ++move) {
          dst[move] = computation->GetPVal(sample_idx, move);
        }
      }
    }
    return LC0_RAW_DENSE_OK;
  } catch (const std::exception& e) {
    SetError(err_buf, err_buf_len, e.what());
    return LC0_RAW_DENSE_INFER_FAILED;
  } catch (...) {
    SetError(err_buf, err_buf_len, "Unknown exception in lc0_raw_dense_infer.");
    return LC0_RAW_DENSE_INTERNAL_ERROR;
  }
}

}  // extern "C"
