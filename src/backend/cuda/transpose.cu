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
