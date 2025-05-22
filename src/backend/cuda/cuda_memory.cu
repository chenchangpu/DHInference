#include <cuda_runtime.h>
#include <iostream>
#include "backend/cuda/cuda_utils.hpp"

// namespace dhinference {
// namespace backend {

// 初始化CUDA设备
static bool cuda_initialized = false;

static void init_cuda() {
    if (!cuda_initialized) {
        cudaError_t err = cudaSetDevice(0);  // 使用第一个GPU
        if (err != cudaSuccess) {
            std::cerr << "CUDA device initialization failed: " << cudaGetErrorString(err) << std::endl;
            throw std::runtime_error("CUDA device initialization failed");
        }
        cuda_initialized = true;
    }
}

// 显存分配
void* gpu_malloc(size_t size) {
    if (size == 0) {
        std::cerr << "Warning: Attempting to allocate 0 bytes" << std::endl;
        return nullptr;
    }

    init_cuda();

    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed: " << cudaGetErrorString(err) 
                  << " (size: " << size << " bytes)" << std::endl;
        throw std::runtime_error("CUDA malloc failed");
    }
    return ptr;
}

// 显存释放
void gpu_free(void* ptr) {
    if (ptr == nullptr) {
        return;
    }

    cudaError_t err = cudaFree(ptr);
    if (err != cudaSuccess) {
        std::cerr << "CUDA free failed: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA free failed");
    }
}

// CPU -> GPU 数据拷贝
void copy_cpu_to_gpu(void* d_dst, const void* h_src, size_t size) {
    if (d_dst == nullptr || h_src == nullptr || size == 0) {
        std::cerr << "Invalid parameters for copy_cpu_to_gpu" << std::endl;
        return;
    }

    cudaError_t err = cudaMemcpy(d_dst, h_src, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy H2D failed: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA memcpy H2D failed");
    }
    cuda_device_sync();
}

// GPU -> CPU 数据拷贝
void copy_gpu_to_cpu(void* h_dst, const void* d_src, size_t size) {
    if (h_dst == nullptr || d_src == nullptr || size == 0) {
        std::cerr << "Invalid parameters for copy_gpu_to_cpu" << std::endl;
        return;
    }

    cudaError_t err = cudaMemcpy(h_dst, d_src, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy D2H failed: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA memcpy D2H failed");
    }
    cuda_device_sync();
}

// 同步CPU和设备
void cuda_device_sync() {
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA device sync failed: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA device sync failed");
    }
}


// }   // end of backend
// }   // end of dhinference