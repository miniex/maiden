cmake_minimum_required(VERSION 3.15)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cuda-headers)

add_library(tensor_ops STATIC
   ${CMAKE_CURRENT_SOURCE_DIR}/cuda/tensor_ops.cu
)

set_target_properties(tensor_ops PROPERTIES
   CUDA_SEPARABLE_COMPILATION ON
   POSITION_INDEPENDENT_CODE ON
)
