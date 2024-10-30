#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void tensor_add(float *output, const float *input1, const float *input2,
                size_t size);
void tensor_div(float *output, const float *input1, const float *input2,
                size_t size);
void tensor_mat_mul(float *output, const float *input1, const float *input2,
                    const int M, const int N, const int K);
void tensor_mean(float *output, const float *input, const int size);
void tensor_mul(float *output, const float *input1, const float *input2,
                size_t size);
void tensor_pow(float *output, const float *input, float exponent, size_t size);
void tensor_scalar_add(float *output, const float *input, const float scalar,
                       const int size);
void tensor_scalar_div(float *output, const float *input, const float scalar,
                       const int size);
void tensor_scalar_mul(float *output, const float *input, const float scalar,
                       const int size);
void tensor_scalar_sub(float *output, const float *input, const float scalar,
                       const int size);
void tensor_sub(float *output, const float *input1, const float *input2,
                size_t size);
void tensor_sum(float *output, const float *input, const int size);
void tensor_transpose(float *output, const float *input, int rows, int cols);

#ifdef __cplusplus
}
#endif
