#include "nn_ops.cuh"
#include <cuda_runtime.h>

__global__ void linear_forward_kernel(float *output, const float *input,
                                      const float *weight, const float *bias,
                                      int batch_size, int out_features,
                                      int in_features) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < batch_size * out_features) {
    int batch = idx / out_features;
    int out_idx = idx % out_features;

    float sum = 0.0f;
    for (int i = 0; i < in_features; i++) {
      sum += input[batch * in_features + i] * weight[out_idx * in_features + i];
    }

    if (bias != nullptr) {
      sum += bias[out_idx];
    }

    output[idx] = sum;
  }
}

__global__ void linear_backward_kernel(float *grad_input, float *grad_weight,
                                       float *grad_bias,
                                       const float *grad_output,
                                       const float *input, const float *weight,
                                       int batch_size, int out_features,
                                       int in_features) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < batch_size * out_features) {
    int batch = idx / out_features;
    int out_idx = idx % out_features;

    for (int i = 0; i < in_features; i++) {
      atomicAdd(&grad_input[batch * in_features + i],
                grad_output[idx] * weight[out_idx * in_features + i]);
    }

    for (int i = 0; i < in_features; i++) {
      atomicAdd(&grad_weight[out_idx * in_features + i],
                grad_output[idx] * input[batch * in_features + i]);
    }

    if (grad_bias != nullptr) {
      atomicAdd(&grad_bias[out_idx], grad_output[idx]);
    }
  }
}

extern "C" {
void linear_forward(float *output, const float *input, const float *weight,
                    const float *bias, int batch_size, int out_features,
                    int in_features) {
  int total_threads = batch_size * out_features;
  int block_size = 256;
  int grid_size = (total_threads + block_size - 1) / block_size;

  linear_forward_kernel<<<grid_size, block_size>>>(
      output, input, weight, bias, batch_size, out_features, in_features);
}

void linear_backward(float *grad_input, float *grad_weight, float *grad_bias,
                     const float *grad_output, const float *input,
                     const float *weight, int batch_size, int out_features,
                     int in_features) {
  int total_threads = batch_size * out_features;
  int block_size = 256;
  int grid_size = (total_threads + block_size - 1) / block_size;

  linear_backward_kernel<<<grid_size, block_size>>>(
      grad_input, grad_weight, grad_bias, grad_output, input, weight,
      batch_size, out_features, in_features);
}
}
