#include "nn_ops.cuh"
#include <cuda_runtime.h>

__global__ void relu_forward_kernel(float *output, const float *input, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    output[idx] = fmaxf(input[idx], 0.0f);
  }
}

__global__ void relu_backward_kernel(float *grad_output, const float *input,
                                     float *grad_input, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    grad_input[idx] = input[idx] > 0.0f ? grad_output[idx] : 0.0f;
  }
}

extern "C" {
void relu_forward(float *output, const float *input, int n) {
  int block_size = 256;
  int grid_size = (n + block_size - 1) / block_size;
  relu_forward_kernel<<<grid_size, block_size>>>(output, input, n);
}

void relu_backward(float *grad_output, const float *input, float *grad_input,
                   int n) {
  int block_size = 256;
  int grid_size = (n + block_size - 1) / block_size;
  relu_backward_kernel<<<grid_size, block_size>>>(grad_output, input,
                                                  grad_input, n);
}
}
