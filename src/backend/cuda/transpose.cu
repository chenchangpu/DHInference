#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda.h>
#include <algorithm>
#include <assert.h>
#include <float.h>

#include <cuda_runtime.h>
#include <stdio.h>
#include "backend/cuda/cuda_utils.hpp"

// 最高支持4维
__global__ void transpose_kernel(
    const float* input, 
    float* output, 
    const int shape[4],
    const int perm[4],
    int rank,
    size_t total_size
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size) return;

    int old_coords[4];  // 存储旧坐标（假设最高支持4维）
    int new_coords[4];  // 存储新坐标
    int old_stride[4];  // 存储原始stride
    int new_stride[4];  // 存储转置后stride

    // Step 1: 计算旧stride
    old_stride[0] = 1;
    for (int i = 1; i < rank; ++i) {
        old_stride[i] = old_stride[i - 1] * shape[i - 1];
    }

    // Step 2: 计算旧坐标（根据线性索引）
    int remaining = idx;
    for (int i = rank - 1; i >= 0; --i) {
        old_coords[i] = remaining / old_stride[i];
        remaining %= old_stride[i];
    }

    // Step 3: 计算新坐标（根据perm）
    for (int i = 0; i < rank; ++i) {
        new_coords[i] = old_coords[perm[i]];
    }

    // Step 4: 计算转置后stride
    new_stride[0] = 1;
    for (int i = 1; i < rank; ++i) {
        new_stride[i] = new_stride[i - 1] * shape[perm[i - 1]];
    }

    // Step 5: 根据新坐标计算线性索引
    int out_idx = 0;
    for (int i = 0; i < rank; ++i) {
        out_idx += new_coords[i] * new_stride[i];
    }

    // Step 6: 数据拷贝
    output[out_idx] = input[idx];
}

// 优化的2D转置核函数（针对常见的2D转置场景）
__global__ void transpose_2d_kernel(
    const float* input,
    float* output,
    int rows,
    int cols
) {
    __shared__ float shared_mem[32][32 + 1];  // 避免bank conflict
    
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    
    if (x < cols && y < rows) {
        shared_mem[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }
    
    __syncthreads();
    
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    
    if (x < rows && y < cols) {
        output[y * rows + x] = shared_mem[threadIdx.x][threadIdx.y];
    }
}

extern "C" void launch_transpose(
    const float* input, 
    float* output, 
    const int shape[4],
    const int perm[4],
    void* extra_buff,
    int rank
) {
    if (rank == 2 && (perm[0] == 1 && perm[1] == 0)) {
        constexpr int BLOCK_SIZE = 32;
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(
            (shape[1] + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (shape[0] + BLOCK_SIZE - 1) / BLOCK_SIZE
        );
        transpose_2d_kernel<<<grid, block>>>(
            input, output, shape[0], shape[1]
        );
    }
    else{
        size_t total_size = 1;
        for (int i = 0; i < rank; ++i) {
            total_size *= shape[i];
        }
    
        int block_size = 256;
        int grid_size = (total_size + block_size - 1) / block_size;
        // host转移到device，再传入kernel
        copy_cpu_to_gpu(extra_buff, shape, 4 * sizeof(int));
        copy_cpu_to_gpu((void*)((int*)extra_buff + 4), perm, 4 * sizeof(int));
        // Launch kernel
        transpose_kernel<<<grid_size, block_size>>>(input, output, static_cast<const int*>(extra_buff), static_cast<const int*>((int*)extra_buff + 4), rank, total_size);
    }
}

// // CPU 上实现相同的转置逻辑
// void transpose_cpu(
//     const float* input,
//     float* output,
//     const int shape[4],
//     const int perm[4],
//     int rank,
//     size_t total_size
// ) {
//     int old_coords[4];
//     int new_coords[4];
//     int old_stride[4];
//     int new_stride[4];

//     old_stride[0] = 1;
//     for (int i = 1; i < rank; ++i) {
//         old_stride[i] = old_stride[i - 1] * shape[i - 1];
//     }

//     for (size_t idx = 0; idx < total_size; ++idx) {
//         int remaining = idx;
//         for (int i = rank - 1; i >= 0; --i) {
//             old_coords[i] = remaining / old_stride[i];
//             remaining %= old_stride[i];
//         }

//         for (int i = 0; i < rank; ++i) {
//             new_coords[i] = old_coords[perm[i]];
//         }

//         new_stride[0] = 1;
//         for (int i = 1; i < rank; ++i) {
//             new_stride[i] = new_stride[i - 1] * shape[perm[i - 1]];
//         }

//         int out_idx = 0;
//         for (int i = 0; i < rank; ++i) {
//             out_idx += new_coords[i] * new_stride[i];
//         }

//         output[out_idx] = input[idx];
//     }
// }

// // 计算总元素数
// size_t get_total_size(const int shape[4], int rank) {
//     size_t total = 1;
//     for (int i = 0; i < rank; ++i) {
//         total *= shape[i];
//     }
//     return total;
// }

// // 检查两个数组是否几乎相等（允许浮点误差）
// bool compare_arrays(float* h_out_gpu, float* h_out_cpu, size_t size) {
//     for (size_t i = 0; i < size; ++i) {
//         if (fabs(h_out_gpu[i] - h_out_cpu[i]) > 1e-5f) {
//             std::cout << "Mismatch at index " << i << ": "
//                       << h_out_gpu[i] << " vs " << h_out_cpu[i] << std::endl;
//             return false;
//         }
//     }
//     return true;
// }

// // 主函数
// int main() {
//     // 设置参数
//     int rank = 4;
//     int shape[4] = {2, 3, 4, 5};  // 输入形状
//     int perm[4] = {3, 2, 1, 0};   // 转置排列

//     size_t total_size = get_total_size(shape, rank);

//     // 分配内存
//     float* h_in = new float[total_size];
//     float* h_out_cpu = new float[total_size];
//     float* h_out_gpu = new float[total_size];
//     float* d_in, *d_out;
//     cudaMalloc(&d_in, total_size * sizeof(float));
//     cudaMalloc(&d_out, total_size * sizeof(float));

//     // 初始化输入数据
//     for (size_t i = 0; i < total_size; ++i) {
//         h_in[i] = static_cast<float>(rand()) / RAND_MAX;
//     }

//     // CPU 转置
//     transpose_cpu(h_in, h_out_cpu, shape, perm, rank, total_size);

//     // GPU 转置
//     cudaMemcpy(d_in, h_in, total_size * sizeof(float), cudaMemcpyHostToDevice);
//     int* d_shape;
//     int* d_perm;
//     cudaMalloc(&d_shape, 4 * sizeof(int));
//     cudaMalloc(&d_perm, 4 * sizeof(int));
//     cudaMemcpy(d_shape, shape, 4 * sizeof(int), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_perm, perm, 4 * sizeof(int), cudaMemcpyHostToDevice);
//     dim3 threadsPerBlock(256);
//     dim3 numBlocks((total_size + threadsPerBlock.x - 1) / threadsPerBlock.x);
//     transpose_kernel<<<numBlocks, threadsPerBlock>>>(d_in, d_out, d_shape, d_perm, rank, total_size);
//     cudaMemcpy(h_out_gpu, d_out, total_size * sizeof(float), cudaMemcpyDeviceToHost);

//     // 比较结果
//     bool success = compare_arrays(h_out_gpu, h_out_cpu, total_size);
//     if (success) {
//         std::cout << "Test passed!" << std::endl;
//     } else {
//         std::cerr << "Test failed!" << std::endl;
//     }

//     // 释放内存
//     delete[] h_in;
//     delete[] h_out_cpu;
//     delete[] h_out_gpu;
//     cudaFree(d_in);
//     cudaFree(d_out);

//     return success ? 0 : 1;
// }