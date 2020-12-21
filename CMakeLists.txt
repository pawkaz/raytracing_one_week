cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(rt LANGUAGES CXX CUDA)
 
set(CMAKE_CXX_STANDARD 17)

add_library(render STATIC
  src/render.cu
)

include_directories(/usr/local/cuda/include)
include_directories(src/)

set_property(TARGET render  PROPERTY CUDA_SEPARABLE_COMPILATION ON)

add_executable(app main.cc)

target_link_libraries(app render)