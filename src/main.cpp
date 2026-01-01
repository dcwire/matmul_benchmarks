#include <iostream>
#include <vector> 
#include <chrono>
#include <string>
#include "naive_cpu.hpp"


void fill_random(std::vector<float>& v) {
  for(auto& i: v) i = static_cast<float>(rand()) / RAND_MAX;
}

void print_matrix_tile(std::vector<float>& A, std::vector<float>& B, std::vector<float>& C, int LIMIT, int N) {
  if (LIMIT > N) {
    std::cerr << "Error: print_matrix_tile, LIMIT greater than matrix size\n";
    return;
  }
  std::cout << "Matmul results for the tile with size: " << LIMIT << "x" << LIMIT << " \n";
  for (int i=0; i<LIMIT; ++i) {
    std::cout << "| ";
    
    for (int j=0; j<LIMIT; ++j) {
      printf("%0.3f ", A[i*N + j]); 
    }

    if (i == (LIMIT / 2)) {
      printf("| X");
    } else {

    std::cout << "|  ";
    }
    std::cout << " | ";
    
    

    for (int j=0; j<LIMIT; ++j) {
      printf("%0.3f ", B[i*N + j]); 
    }

    if (i == (LIMIT / 2)) {
      printf("| =");
    } else {

    std::cout << "|  ";
    }
    std::cout << " | ";

    for (int j=0; j<LIMIT; ++j) {
      printf("%0.3f ", C[i*N + j]); 
    }
    
    std::cout << "| \n";
  }
}

int main() {
  int N = 4; // Can move this outside the file. Maybe pass matrix size through the CLI?
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
  print_matrix_tile(h_A, h_B, h_C, LIMIT, N);
  

  return 0;
}
