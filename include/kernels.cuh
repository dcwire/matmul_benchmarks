#pragma once

#define THREADS 128
#define TILE_WIDTH 64
#define COARSE_FACTOR 8

// Naive matrix multiplication without any optimizations
void launch_naive_matmul(float *A, float *B, float *C, int N);
// Tiled matrix multiplication
void launch_tiled_matmul(float *A, float *B, float *C, int N);
// Thread coarsening matrix multiplication
void launch_threadcoa_matmul(float *A, float *B, float *C, int N);

