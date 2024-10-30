#include "tensor_ops.cuh"
#include <cuda_runtime.h>

__global__ void pow_kernel(float *output, const float *input, float exponent,
                           size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = powf(input[idx], exponent);
  }
}

extern "C" {
void tensor_pow(float *output, const float *input, float exponent,
                size_t size) {
  constexpr int BLOCK_SIZE = 256;
  int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

  pow_kernel<<<num_blocks, BLOCK_SIZE>>>(output, input, exponent, size);
  cudaDeviceSynchronize();
}
}
