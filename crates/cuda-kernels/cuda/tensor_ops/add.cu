#include "tensor_ops.cuh"
#include <cuda_runtime.h>

__global__ void add_kernel(float *output, const float *input1,
                           const float *input2, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = input1[idx] + input2[idx];
  }
}

extern "C" {
void tensor_add(float *output, const float *input1, const float *input2,
                size_t size) {
  constexpr int BLOCK_SIZE = 256;
  int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  add_kernel<<<num_blocks, BLOCK_SIZE>>>(output, input1, input2, size);
  cudaDeviceSynchronize();
}
}
