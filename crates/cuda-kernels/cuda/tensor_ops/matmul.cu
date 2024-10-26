#include "tensor_ops.cuh"
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void matmul_kernel(float *output, const float *input1,
                              const float *input2, const int M, const int N,
                              const int K) {
  __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
  __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;
  float sum = 0.0f;

  for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
    if (row < M && t * TILE_SIZE + threadIdx.x < K) {
      tile_A[threadIdx.y][threadIdx.x] =
          input1[row * K + t * TILE_SIZE + threadIdx.x];
    } else {
      tile_A[threadIdx.y][threadIdx.x] = 0.0f;
    }

    if (t * TILE_SIZE + threadIdx.y < K && col < N) {
      tile_B[threadIdx.y][threadIdx.x] =
          input2[(t * TILE_SIZE + threadIdx.y) * N + col];
    } else {
      tile_B[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

    for (int i = 0; i < TILE_SIZE; i++) {
      sum += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < M && col < N) {
    output[row * N + col] = sum;
  }
}

extern "C" {
void tensor_matmul(float *output, const float *input1, const float *input2,
                   const int M, const int N, const int K) {
  dim3 block_size(TILE_SIZE, TILE_SIZE);
  dim3 num_blocks((N + TILE_SIZE - 1) / TILE_SIZE,
                  (M + TILE_SIZE - 1) / TILE_SIZE);

  matmul_kernel<<<num_blocks, block_size>>>(output, input1, input2, M, N, K);
  cudaDeviceSynchronize();
}
}
