#include "nn_ops.cuh"
#include <cuda_runtime.h>

__global__ void bilinear_forward_kernel(float *output, const float *input1,
                                        const float *input2,
                                        const float *weight, const float *bias,
                                        int batch_size, int out_features,
                                        int in1_features, int in2_features) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < batch_size * out_features) {
    int batch = idx / out_features;
    int out_idx = idx % out_features;

    float sum = 0.0f;
    for (int i = 0; i < in1_features; i++) {
      for (int j = 0; j < in2_features; j++) {
        float w = weight[(out_idx * in1_features + i) * in2_features + j];
        sum += input1[batch * in1_features + i] * w *
               input2[batch * in2_features + j];
      }
    }

    if (bias != nullptr) {
      sum += bias[out_idx];
    }

    output[idx] = sum;
  }
}

__global__ void bilinear_backward_kernel(
    float *grad_input1, float *grad_input2, float *grad_weight,
    float *grad_bias, const float *grad_output, const float *input1,
    const float *input2, const float *weight, int batch_size, int out_features,
    int in1_features, int in2_features) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < batch_size * out_features) {
    int batch = idx / out_features;
    int out_idx = idx % out_features;

    for (int i = 0; i < in1_features; i++) {
      for (int j = 0; j < in2_features; j++) {
        float w = weight[(out_idx * in1_features + i) * in2_features + j];
        float grad = grad_output[idx];

        atomicAdd(&grad_input1[batch * in1_features + i],
                  grad * w * input2[batch * in2_features + j]);

        atomicAdd(&grad_input2[batch * in2_features + j],
                  grad * w * input1[batch * in1_features + i]);

        atomicAdd(&grad_weight[(out_idx * in1_features + i) * in2_features + j],
                  grad * input1[batch * in1_features + i] *
                      input2[batch * in2_features + j]);
      }
    }

    if (grad_bias != nullptr) {
      atomicAdd(&grad_bias[out_idx], grad_output[idx]);
    }
  }
}

extern "C" {
void bilinear_forward(float *output, const float *input1, const float *input2,
                      const float *weight, const float *bias, int batch_size,
                      int out_features, int in1_features, int in2_features) {
  int total_threads = batch_size * out_features;
  int block_size = 256;
  int grid_size = (total_threads + block_size - 1) / block_size;

  bilinear_forward_kernel<<<grid_size, block_size>>>(
      output, input1, input2, weight, bias, batch_size, out_features,
      in1_features, in2_features);
}

void bilinear_backward(float *grad_input1, float *grad_input2,
                       float *grad_weight, float *grad_bias,
                       const float *grad_output, const float *input1,
                       const float *input2, const float *weight, int batch_size,
                       int out_features, int in1_features, int in2_features) {
  int total_threads = batch_size * out_features;
  int block_size = 256;
  int grid_size = (total_threads + block_size - 1) / block_size;

  bilinear_backward_kernel<<<grid_size, block_size>>>(
      grad_input1, grad_input2, grad_weight, grad_bias, grad_output, input1,
      input2, weight, batch_size, out_features, in1_features, in2_features);
}
}
