cmake_minimum_required(VERSION 3.27)
project(cpp_project)

set(CMAKE_CXX_STANDARD 17)

add_executable(cpp_project
        tensor_impl.h
        main.cpp
        tensor.h)
