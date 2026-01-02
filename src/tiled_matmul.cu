#include <cuda_runtime.h>
#include "kernels.cuh"


__global__ void tiled_matmul(float *A, float *B, float *C, int N) {
    
    __shared__ float mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int tx = threadIdx.x;
    int by = blockIdx.y; int ty = threadIdx.y;

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    float pvalue = 0.0f;

    for (int p=0; p < (N / TILE_WIDTH); ++p) {
        if (row < N && (p*TILE_WIDTH + tx) < N) {
            int a_idx = row*N + p*TILE_WIDTH + tx;
            mds[ty][tx] = A[a_idx];
        } else {
            mds[ty][tx] = 0.0f;
        }
        if (col < N && (p*TILE_WIDTH + ty) < N) {
            int b_idx = (p*TILE_WIDTH + ty)*N + col;
            nds[ty][tx] = B[b_idx];
        } else {
            nds[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        
        for (int k=0; k < TILE_WIDTH; ++k) {
            pvalue += mds[ty][k] * nds[k][tx];
        }

        __syncthreads();

        
    }

    C[row*N + col] = pvalue;

}

void launch_tiled_matmul(float *A, float *B, float *C, int N) {
    dim3 gridDim((N + THREADS - 1)/THREADS, (N + THREADS - 1)/THREADS);
    dim3 blockDim(THREADS, THREADS);

    tiled_matmul<<<gridDim, blockDim>>>(A, B, C, N);
}