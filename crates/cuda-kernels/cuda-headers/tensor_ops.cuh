#pragma once

#ifdef __cplusplus
extern "C" {
#endif

int tensor_add(const float *a, const float *b, float *c, int n);
int tensor_multiply(const float *a, const float *b, float *c, int n);

#ifdef __cplusplus
}
#endif
