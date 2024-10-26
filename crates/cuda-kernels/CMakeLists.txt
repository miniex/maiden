cmake_minimum_required(VERSION 3.15)
project(cuda_kernels CUDA C CXX)
find_package(CUDA REQUIRED)

# cuda
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# cpp
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# cuda arch
set(CMAKE_CUDA_ARCHITECTURES 75)

# export compile
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

file(GLOB_RECURSE CUDA_SOURCES "cuda/*.cu")

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda-headers
    ${CUDA_INCLUDE_DIRS}
)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

function(add_cuda_module MODULE_PATH)
    get_filename_component(MODULE_NAME ${MODULE_PATH} NAME)
    file(GLOB MODULE_SOURCES "${MODULE_PATH}/*.cu")
    
    if(MODULE_SOURCES)
        add_library(${MODULE_NAME} STATIC ${MODULE_SOURCES})
        set_target_properties(${MODULE_NAME} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
            POSITION_INDEPENDENT_CODE ON
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
        )
        
        target_link_libraries(${MODULE_NAME}
            PRIVATE
            ${CUDA_LIBRARIES}
            ${CUDA_CUDA_LIBRARY}
            ${CUDA_CUDART_LIBRARY}
        )
    endif()
endfunction()

add_cuda_module(${CMAKE_CURRENT_SOURCE_DIR}/cuda/tensor_ops)
