#include <cuda_runtime.h>
#include "kernels.cuh"


__global__ void naive_matmul(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float pvalue = 0.0f;
        for (int i=0; i<N; i++) {
            pvalue += A[row*N + i] * B[i*N + col];
        }
        C[row*N + col] = pvalue;
    }
}

// TODO: Add error checking
void launch_naive_matmul(float *A, float *B, float *C, int N) {
    dim3 gridDim((N + THREADS - 1)/THREADS, (N + THREADS - 1)/THREADS);
    dim3 blockDim(THREADS, THREADS);

    naive_matmul<<<gridDim, blockDim>>>(A, B, C, N);

}