#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// activation
void relu_forward(float *output, const float *input, int n);
void relu_backward(float *grad_output, const float *input, float *grad_input,
                   int n);

void sigmoid_forward(float *output, const float *input, int n);
void sigmoid_backward(float *grad_output, const float *output,
                      float *grad_input, int n);

void tanh_forward(float *output, const float *input, int n);
void tanh_backward(float *grad_output, const float *output, float *grad_input,
                   int n);
// layer - linear
void linear_forward(float *output, const float *input, const float *weight,
                    const float *bias, int batch_size, int out_features,
                    int in_features);
void linear_backward(float *grad_input, float *grad_weight, float *grad_bias,
                     const float *grad_output, const float *input,
                     const float *weight, int batch_size, int out_features,
                     int in_features);

void bilinear_forward(float *output, const float *input1, const float *input2,
                      const float *weight, const float *bias, int batch_size,
                      int out_features, int in1_features, int in2_features);
void bilinear_backward(float *grad_input1, float *grad_input2,
                       float *grad_weight, float *grad_bias,
                       const float *grad_output, const float *input1,
                       const float *input2, const float *weight, int batch_size,
                       int out_features, int in1_features, int in2_features);

#ifdef __cplusplus
}
#endif
