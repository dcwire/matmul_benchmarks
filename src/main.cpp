#include <iostream>
#include <vector> 
#include <chrono>
#include "naive_cpu.hpp"


void fill_random(std::vector<float>& v) {
  for(auto& i: v) i = static_cast<float>(rand()) / RAND_MAX;
}

int main() {
  int N = 1024; // Can move this outside the file. Maybe pass matrix size through the CLI?
  int LIMIT = 4; // Print the first LIMITxLIMIT tile of the matrices 
  size_t bytes = N * N * sizeof(float);

  std::vector<float> h_A(N*N), h_B(N*N), h_C(N*N);
  
  // ===================================================
  // 1. CPU BENCHMARK
  // ===================================================
  printf("Beginning CPU benchmarks for N = %d... \n", N);
  fill_random(h_A);
  fill_random(h_B);
  
  auto cpu_start = std::chrono::high_resolution_clock::now();

  cpu_matmul(h_A, h_B, h_C, N);
  
  auto cpu_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float, std::milli> cpu_duration = cpu_end - cpu_start;
  float cpu_ms = cpu_duration.count();

  std::cout << "Naive CPU time: " << cpu_ms << " ms\n";
  


  return 0;
}
