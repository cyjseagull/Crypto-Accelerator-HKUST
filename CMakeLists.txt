cmake_minimum_required(VERSION 3.10)
project(crypto-accelerator LANGUAGES CXX CUDA)

find_package(CUDA)
find_library(gmp gmp)

set(CMAKE_CXX_FLAGS "-O3")

add_subdirectory(RapidSV)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}) 

set_target_properties(RapidSV PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(main main.cpp)
set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(main PRIVATE RapidSV ${CUDA_LIBRARIES} gmp)
