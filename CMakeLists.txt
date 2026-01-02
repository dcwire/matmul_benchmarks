cmake_minimum_required(VERSION 3.18)
project(MatmulBenchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
include_directories(include)

include(CheckLanguage)
check_language(CUDA)

set(SOURCES
    src/main.cpp
    src/naive_cpu.cpp
)

if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_ARCHITECTURES native)

  include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

  list(APPEND SOURCES src/naive_gpu.cu src/tiled_matmul.cu src/thread_coarsening_matmul.cu)
  add_compile_definitions(HAS_CUDA=1)
else()
  add_compile_definitions(HAS_CUDA=0)
endif()

# Adds an executable called benchmark
add_executable(benchmark ${SOURCES})

if(CMAKE_CUDA_COMPILER)
  target_link_libraries(benchmark cudart)
endif()