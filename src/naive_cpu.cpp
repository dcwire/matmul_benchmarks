#include "naive_cpu.hpp"

void cpu_matmul(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int N) {

  // O(N^3) complexity - N * N * N operations
  for (int i=0; i<N; ++i) {
    for (int j=0; j<N; ++j) {
      C[i*N + j] = 0.0f;
      for (int k=0; k<N; ++k) {
        C[i*N + j] += A[i*N + k] * B[k*N + j];
      }
    }
  }

}
