#include "tensor_ops.cuh"
#define BLOCK_SIZE 256

__global__ void sum_kernel(float *output, const float *input, const int size) {
  __shared__ float shared_sum[BLOCK_SIZE];

  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  shared_sum[tid] = 0.0f;
  if (idx < size) {
    shared_sum[tid] = input[idx];
  }
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared_sum[tid] += shared_sum[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(output, shared_sum[0]);
  }
}

extern "C" {
void tensor_sum(float *output, const float *input, const int size) {
  float zero = 0.0f;
  cudaMemcpy(output, &zero, sizeof(float), cudaMemcpyHostToDevice);

  int block_size = BLOCK_SIZE;
  int num_blocks = (size + block_size - 1) / block_size;
  sum_kernel<<<num_blocks, block_size>>>(output, input, size);
  cudaDeviceSynchronize();
}
}
