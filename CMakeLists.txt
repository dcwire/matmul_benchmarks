cmake_minimum_required(VERSION 3.18)
project(MatmulBenchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
include_directories(include)

include(CheckLanguage)
check_language(CUDA)

set(SOURCES
    src/main.cpp
)

if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_ARCHITECTURES native)
  # list(APPEND SOURCES src/)
  add_compile_definitions(HAS_CUDA=1)
else()
  add_compile_definitions(HAS_CUDA=0)
endif()

# Adds an executable called benchmark
add_executable(benchmark ${SOURCES})

