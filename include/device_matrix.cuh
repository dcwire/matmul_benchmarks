#pragma once
#include <cuda_runtime.h>
#include <stdexcept>

// The idea is to manage device memory through a constructor/destructor
// TODO: Add error checking
template <typename T>
class DeviceMatrix {
public:
    int rows, cols;
    T* data;
    DeviceMatrix(int r, int c): rows(r), cols(c) {
        size_t size = r * c * sizeof(T);
        cudaMalloc(&data, size);
    }

    ~DeviceMatrix() {
        if (data) cudaFree(data);
    }
    
    // Copy specific number of bytes
    void copy_from_host(const std::vector<T>& h, size_t bytes) {
        size_t size = rows * cols * sizeof(T);
        if (size < bytes) {
            // ERROR
        }
        cudaMemcpy(data, h.data(), bytes, cudaMemcpyHostToDevice);
    }

    // Copy specific number of bytes
    void copy_from_device(std::vector<T>& h, size_t bytes) {
        size_t size = rows * cols * sizeof(T);
        if (size < bytes) {
            // ERROR
        }
        cudaMemcpy(h.data(), data, bytes, cudaMemcpyDeviceToHost);
    } 
};
