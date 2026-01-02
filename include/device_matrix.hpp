#pragma once
#include<cuda_runtime.h>
#include<stdexcept>

// The idea is to manage device memory through a constructor/destructor
// TODO: Add error checking
template <typename T>
class DeviceMatrix {
    int rows, cols;
    T* data;
    DeviceMatrix(int r, int c): rows(r), cols(c) {
        size_t size = r * c * sizeof(T);
        cudaMalloc(&data, size);
    }

    // Copy specific number of bytes
    void copy_from_host(T* h, size_t bytes) {
        size_t size = rows * cols * sizeof(T);
        if (size < bytes) {
            // ERROR
        }
        cudaMemcpy(data, h, bytes, CudaMemcpyHostToDevice);
    }

    // Copy specific number of bytes
    void copy_from_device(T* h, size_t bytes) {
        size_t size = rows * cols * sizeof(T);
        if (size < bytes) {
            // ERROR
        }
        cudaMemcpy(h, data, bytes, CudaMemcpyDeviceToHost);
    } 
}