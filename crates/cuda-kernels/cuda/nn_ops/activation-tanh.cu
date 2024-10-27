#include "nn_ops.cuh"
#include <cuda_runtime.h>

__global__ void tanh_forward_kernel(float *output, const float *input, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    output[idx] = tanhf(input[idx]);
  }
}

__global__ void tanh_backward_kernel(float *grad_output, const float *output,
                                     float *grad_input, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float tanh_x = output[idx];
    grad_input[idx] = grad_output[idx] * (1.0f - tanh_x * tanh_x);
  }
}

extern "C" {
void tanh_forward(float *output, const float *input, int n) {
  int block_size = 256;
  int grid_size = (n + block_size - 1) / block_size;
  tanh_forward_kernel<<<grid_size, block_size>>>(output, input, n);
}

void tanh_backward(float *grad_output, const float *output, float *grad_input,
                   int n) {
  int block_size = 256;
  int grid_size = (n + block_size - 1) / block_size;
  tanh_backward_kernel<<<grid_size, block_size>>>(grad_output, output,
                                                  grad_input, n);
}
}
