cmake_minimum_required(VERSION 3.26)
project(CS205-Project LANGUAGES CXX CUDA) # Add CUDA as a language if you're using CUDA

set(CMAKE_CXX_STANDARD 17)

include_directories(/usr/local/cuda/include)

add_executable(CS205-Project
        tensor_aclr.cuh
        tensor_aclr_impl.cuh
        main_aclr.cu
        test_openmp.cpp
)

# OpenMP Configuration
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    message("OPENMP FOUND")
    target_link_libraries(CS205-Project PUBLIC OpenMP::OpenMP_CXX)
endif ()
