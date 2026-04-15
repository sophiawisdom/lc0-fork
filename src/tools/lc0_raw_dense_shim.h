#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct lc0_raw_dense_handle lc0_raw_dense_handle;

enum {
  LC0_RAW_DENSE_OK = 0,
  LC0_RAW_DENSE_INVALID_ARGUMENT = 1,
  LC0_RAW_DENSE_INIT_FAILED = 2,
  LC0_RAW_DENSE_INFER_FAILED = 3,
  LC0_RAW_DENSE_INTERNAL_ERROR = 4,
};

enum {
  LC0_RAW_DENSE_INPUT_SQUARES = 64,
  LC0_RAW_DENSE_INPUT_PLANES = 112,
  LC0_RAW_DENSE_POLICY_SIZE = 1858,
};

// The dense input buffer is [batch_size, 64, 112] flattened row-major,
// square-major then plane-major. Square order is the LC0 training-data order
// used by wisechess/game_playing, i.e. files are flipped within each rank.
int lc0_raw_dense_create(const char* weights_path, const char* backend,
                         const char* backend_options,
                         lc0_raw_dense_handle** out_handle, char* err_buf,
                         size_t err_buf_len);

void lc0_raw_dense_destroy(lc0_raw_dense_handle* handle);

int lc0_raw_dense_infer(lc0_raw_dense_handle* handle,
                        const uint8_t* input_planes_u8, size_t batch_size,
                        float* out_q, float* out_d, float* out_m,
                        float* out_policy, char* err_buf,
                        size_t err_buf_len);

#ifdef __cplusplus
}
#endif
