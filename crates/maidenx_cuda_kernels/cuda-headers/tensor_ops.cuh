#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void tensor_add(float *output, const float *input1, const float *input2,
                size_t size);
void tensor_mul(float *output, const float *input1, const float *input2,
                size_t size);
void tensor_matmul(float *output, const float *input1, const float *input2,
                   const int M, const int N, const int K);

#ifdef __cplusplus
}
#endif
