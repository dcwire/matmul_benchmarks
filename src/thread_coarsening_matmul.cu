#include <cuda_runtime.h>
#include "kernels.cuh"


__global__ void threadcoa_matmul(float *A, float *B, float *C, int N) {
    __shared__ float mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int tx = threadIdx.x;
    int by = blockIdx.y; int ty = threadIdx.y;

    int row = by * blockDim.y + ty;
    int col_start = bx * blockDim.x * COARSE_FACTOR + tx;

    float pvalues[COARSE_FACTOR];
    for (int i=0; i < COARSE_FACTOR; ++i) {
        pvalues[i] = 0.0f;
    }
    
    
        
        
        
    for (int p=0; p < (N / TILE_WIDTH); ++p) {

        if (row < N && (p*TILE_WIDTH + tx) < N) {
            int a_idx = row*N + p*TILE_WIDTH + tx;
            mds[ty][tx] = A[a_idx];
        } else {
            mds[ty][tx] = 0.0f;
        }

        for (int c=0; c < COARSE_FACTOR; ++c) {
            
            int col = col_start + c * TILE_WIDTH;
            if (col < N && (p*TILE_WIDTH + ty) < N) {
                int b_idx = (p*TILE_WIDTH + ty) * N + col;
                nds[ty][tx] = B[b_idx];
            } else {
                nds[ty][tx] = 0.0f;
            }
            
            __syncthreads();
            
            
            for (int k=0; k < TILE_WIDTH; ++k) {
                pvalues[c] += mds[ty][k] * nds[k][tx];
            }

            __syncthreads();
  
        }
    }
    
    for (int c=0; c < COARSE_FACTOR; ++c) {
        C[row*N + col_start + c * TILE_WIDTH] = pvalues[c];
    }
    
}

void launch_threadcoa_matmul(float *A, float *B, float *C, int N) {
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    int COL_BLOCKS = TILE_WIDTH * COARSE_FACTOR;
    // x, y -> x is used for columns
    dim3 gridDim((N + COL_BLOCKS - 1) / COL_BLOCKS, (N + TILE_WIDTH - 1) / TILE_WIDTH);

    threadcoa_matmul<<<gridDim, blockDim>>>(A, B, C, N);

}