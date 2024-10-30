#include "tensor_ops.cuh"
#include <cuda_runtime.h>

#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void transpose_kernel(float *output, const float *input, int rows,
                                 int cols) {
  __shared__ float tile[TILE_DIM][TILE_DIM + 1];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM && y + j < rows; j += BLOCK_ROWS) {
    if (x < cols) {
      tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * cols + x];
    }
  }

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM && y + j < cols; j += BLOCK_ROWS) {
    if (x < rows) {
      output[(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
    }
  }
}

extern "C" {
void tensor_transpose(float *output, const float *input, int rows, int cols) {
  dim3 dimBlock(TILE_DIM, BLOCK_ROWS);
  dim3 dimGrid((cols + TILE_DIM - 1) / TILE_DIM,
               (rows + TILE_DIM - 1) / TILE_DIM);

  transpose_kernel<<<dimGrid, dimBlock>>>(output, input, rows, cols);
  cudaDeviceSynchronize();
}
}
