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
void tensor_mul(float *output, const float *input1, const float *input2,
                size_t size);
void tensor_pow(float *output, const float *input, float exponent, size_t size);
void tensor_scalar_mul(float *output, const float *input, const float scalar,
                       const int size);

#ifdef __cplusplus
}
#endif
