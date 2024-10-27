#include "nn_ops.cuh"
#include <cuda_runtime.h>

__global__ void sigmoid_forward_kernel(float *output, const float *input,
                                       int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    output[idx] = 1.0f / (1.0f + expf(-input[idx]));
  }
}

__global__ void sigmoid_backward_kernel(float *grad_output, const float *output,
                                        float *grad_input, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float sigmoid_x = output[idx];
    grad_input[idx] = grad_output[idx] * sigmoid_x * (1.0f - sigmoid_x);
  }
}

extern "C" {
void sigmoid_forward(float *output, const float *input, int n) {
  int block_size = 256;
  int grid_size = (n + block_size - 1) / block_size;
  sigmoid_forward_kernel<<<grid_size, block_size>>>(output, input, n);
}

void sigmoid_backward(float *grad_output, const float *output,
                      float *grad_input, int n) {
  int block_size = 256;
  int grid_size = (n + block_size - 1) / block_size;
  sigmoid_backward_kernel<<<grid_size, block_size>>>(grad_output, output,
                                                     grad_input, n);
}
}
