#include "tensor_ops.cuh"
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void scalar_add_kernel(float *output, const float *input,
                                  const float scalar, const int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < size) {
    output[idx] = input[idx] + scalar;
  }
}

extern "C" {
void tensor_scalar_add(float *output, const float *input, const float scalar,
                       const int size) {
  int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

  scalar_add_kernel<<<num_blocks, BLOCK_SIZE>>>(output, input, scalar, size);
  cudaDeviceSynchronize();
}
}
