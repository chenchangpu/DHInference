#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda.h>
#include <assert.h>
#include <vector>
#include <iostream>
#include <cmath> 

#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define WARP_SIZE 32

template<int warpsize = 32>
__forceinline__ __device__ float warp_reduce_sum(float val){
    #pragma unroll
    for(int mask = warpsize >> 1; mask >= 1; mask >>= 1){
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

// y = Ax
// 一个warp负责一行
// K = 16
template<int warpsize = 32, int threads_per_block = 128>
__global__ void sgemv_k16(float *A, float *x, float *y, int M, int K){
    static_assert(threads_per_block % warpsize == 0);
    constexpr int kwarpsize = 16;                                    // 16
    constexpr int rows_per_warp = warpsize / kwarpsize;
    constexpr int warps_per_block = threads_per_block / warpsize; // 128 / 32 = 4 -> 8 rows

    int klaneid = threadIdx.x % kwarpsize; // 0~15
    int kwarpid = threadIdx.x / kwarpsize; // 0~7
    int m = blockIdx.x * warps_per_block * rows_per_warp + kwarpid;

    if(m < M){
        float *A_start = A + m * K;
        float sum = A_start[klaneid] * x[klaneid];
        sum = warp_reduce_sum<kwarpsize>(sum);
        if(klaneid == 0){
            y[m] = sum;
        }
    }
}

// y = Ax
// 一个warp负责一行
// 32 <= K <= 128
template<int warpsize = 32, int threads_per_block = 128>
__global__ void sgemv_k32(float *A, float *x, float *y, int M, int K){
    static_assert(threads_per_block % warpsize == 0);
    constexpr int warps_per_block = threads_per_block / warpsize; // 128 / 32 = 4

    int laneid = threadIdx.x % warpsize; // 0~31
    int warpid = threadIdx.x / warpsize; // 0~3
    int m = blockIdx.x * warps_per_block + warpid;

    if(m < M){
        float *A_start = A + m * K;
        int nk = K / warpsize;
        float sum = 0.0f;
        for(int i = 0; i < nk; ++i){
            sum += A_start[i * warpsize + laneid] * x[i * warpsize + laneid];
        }
        sum = warp_reduce_sum<warpsize>(sum);
        if(laneid == 0){
            y[m] = sum;
        }
    }
}


// y = Ax
// 一个warp负责一行, 采用float4 fetch
// K >= 128
template<int warpsize = 32, int threads_per_block = 128>
__global__ void sgemv_k128_float4(float *A, float *x, float *y, int M, int K){
    static_assert(threads_per_block % warpsize == 0);
    constexpr int warps_per_block = threads_per_block / warpsize; // 128 / 32 = 4

    int laneid = threadIdx.x % warpsize; // 0~31
    int warpid = threadIdx.x / warpsize; // 0~3
    int m = blockIdx.x * warps_per_block + warpid;

    if(m < M){
        float *A_start = A + m * K;
        int nk = K / warpsize / 4;
        float sum = 0.0f;
        for(int i = 0; i < nk; ++i){
            int idx = (i * warpsize + laneid) * 4;
            float4 a_ = FLOAT4(A_start[idx]);
            float4 x_ = FLOAT4(x[idx]);
            sum += a_.w * x_.w + (a_.x * x_.x + (a_.y * x_.y + (a_.z * x_.z)));         // dahu: fma组合
        }
        sum = warp_reduce_sum<warpsize>(sum);
        if(laneid == 0){
            y[m] = sum;
        }
    }
}

template<const int threads_per_block = 128>
void launch_sgemv(float *A, float *x, float *y, int M, int K) {
    dim3 blockDim(threads_per_block);
    dim3 gridDim;

    assert(K % 16 == 0);

    if(K == 16){
        gridDim = {M / (threads_per_block / 16), 1, 1};
    }
    else{
        gridDim = {M / (threads_per_block / WARP_SIZE), 1, 1};
    }

    if (K == 16) {
        sgemv_k16<WARP_SIZE, threads_per_block><<<gridDim, blockDim>>>(A, x, y, M, K);
    } else if (K <= 128) {
        sgemv_k32<WARP_SIZE, threads_per_block><<<gridDim, blockDim>>>(A, x, y, M, K);
    } else {
        sgemv_k128_float4<WARP_SIZE, threads_per_block><<<gridDim, blockDim>>>(A, x, y, M, K);
    }
}

// 显示实例化
extern "C" void launch_sgemv_96(float *A, float *x, float *y, int M, int K){
    launch_sgemv<96>(A, x, y, M, K);
}
extern "C" void launch_sgemv_128(float *A, float *x, float *y, int M, int K){
    launch_sgemv<128>(A, x, y, M, K);
}
extern "C" void launch_sgemv_256(float *A, float *x, float *y, int M, int K){
    launch_sgemv<256>(A, x, y, M, K);
}