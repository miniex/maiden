#include "tensor_ops.cuh"

__global__ void add_kernel(const float *a, const float *b, float *c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

__global__ void multiply_kernel(const float *a, const float *b, float *c,
                                int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] * b[idx];
  }
}

extern "C" {
int tensor_add(const float *a, const float *b, float *c, int n) {
  int blockSize = 256;
  int numBlocks = (n + blockSize - 1) / blockSize;
  add_kernel<<<numBlocks, blockSize>>>(a, b, c, n);
  return cudaGetLastError();
}

int tensor_multiply(const float *a, const float *b, float *c, int n) {
  int blockSize = 256;
  int numBlocks = (n + blockSize - 1) / blockSize;
  multiply_kernel<<<numBlocks, blockSize>>>(a, b, c, n);
  return cudaGetLastError();
}
}
