#include <iostream>
#include <vector> 
#include <chrono>

void fill_random(std::vector<float>& v) {
  for(auto& i: v) i = static_cast<float>(rand()) / RAND_MAX;
}

int main() {
  int N = 1024; // Can move this outside the file. Maybe pass matrix size through the CLI?
  size_t bytes = N * N * sizeof(float);

  std::vector<float> h_A(N*N), h_B(N*N), h_C(N*N);
  
  printf("Beginning CPU benchmarks: \n");
  fill_random(h_A);
  fill_random(h_B);



  return 0;
}
