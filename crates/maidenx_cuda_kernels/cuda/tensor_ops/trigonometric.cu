#include "tensor_ops.cuh"
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void sin_kernel(float *output, const float *input, const int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = sinf(input[idx]);
  }
}

__global__ void cos_kernel(float *output, const float *input, const int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = cosf(input[idx]);
  }
}

__global__ void tan_kernel(float *output, const float *input, const int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = tanf(input[idx]);
  }
}

__global__ void asin_kernel(float *output, const float *input, const int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = asinf(input[idx]);
  }
}

__global__ void acos_kernel(float *output, const float *input, const int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = acosf(input[idx]);
  }
}

__global__ void atan_kernel(float *output, const float *input, const int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = atanf(input[idx]);
  }
}

extern "C" {
void tensor_sin(float *output, const float *input, const int size) {
  int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  sin_kernel<<<num_blocks, BLOCK_SIZE>>>(output, input, size);
  cudaDeviceSynchronize();
}

void tensor_cos(float *output, const float *input, const int size) {
  int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  cos_kernel<<<num_blocks, BLOCK_SIZE>>>(output, input, size);
  cudaDeviceSynchronize();
}

void tensor_tan(float *output, const float *input, const int size) {
  int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  tan_kernel<<<num_blocks, BLOCK_SIZE>>>(output, input, size);
  cudaDeviceSynchronize();
}

void tensor_asin(float *output, const float *input, const int size) {
  int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  asin_kernel<<<num_blocks, BLOCK_SIZE>>>(output, input, size);
  cudaDeviceSynchronize();
}

void tensor_acos(float *output, const float *input, const int size) {
  int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  acos_kernel<<<num_blocks, BLOCK_SIZE>>>(output, input, size);
  cudaDeviceSynchronize();
}

void tensor_atan(float *output, const float *input, const int size) {
  int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  atan_kernel<<<num_blocks, BLOCK_SIZE>>>(output, input, size);
  cudaDeviceSynchronize();
}
}
