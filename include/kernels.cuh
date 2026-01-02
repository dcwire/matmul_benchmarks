#pragma once

// Naive matrix multiplication without any optimizations
void launch_naive_matmul(float *A, float *B, float *C, int N);
// Tileed matrix multiplication
void launch_tiled_matmul(float *A, float *B, float *C, int N);

