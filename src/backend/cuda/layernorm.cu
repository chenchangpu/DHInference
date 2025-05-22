#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <cuda.h>
#include <algorithm>
#include <assert.h>
#include <float.h>
#include <chrono>

#define kWarpSize 32
#define Waves 32

struct SumOP{
    __device__ __forceinline__ float operator()(float a, float b) const { return a + b; }
};

struct MaxOP{
    __device__ __forceinline__ float operator()(float a, float b) const { return fmaxf(a, b); }
};

template <class ReductionOP, int warpsize = kWarpSize>
__device__ __forceinline__ float WarpAllReduce(float val){
    #pragma unroll
    for(int mask = warpsize >> 1; mask >= 1; mask >>= 1){
        val = ReductionOP()(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

template<int threads_per_block = 128 ,int warpsize = 32>
__device__ __forceinline__ float BlockAllReduceSum(float val){
    constexpr int warps_per_block = threads_per_block / warpsize;
    static_assert(warps_per_block <= warpsize);
    __shared__ float warpsum[warps_per_block];

    const int warpid = threadIdx.x / warpsize;
    const int laneid = threadIdx.x % warpsize;

    val = WarpAllReduce<SumOP, warpsize>(val);
    if(laneid == 0) warpsum[warpid] = val;
    __syncthreads();
    val = laneid < warps_per_block ? warpsum[laneid] : 0.0f;

    return WarpAllReduce<SumOP, warpsize>(val);
}

inline cudaError_t GetNumBlocks(int64_t block_size, int64_t max_blocks, int64_t waves, int* num_blocks) {
    int dev;
    {
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) { return err; }
    }
    int sm_count;
    {
    cudaError_t err = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
    if (err != cudaSuccess) { return err; }
    }
    int tpm;
    {
    cudaError_t err = cudaDeviceGetAttribute(&tpm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
    if (err != cudaSuccess) { return err; }
    }
    *num_blocks = std::max<int>(1, std::min<int64_t>(max_blocks, sm_count * tpm / block_size * waves));
    return cudaSuccess;
}


// 一个warp处理一行，每次只处理一个float
template<int warpsize = 32, int cols_per_thread>
__global__ void layer_norm_float(float *x, float *y, float* g, float* b, int N, int K){
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int bid = blockIdx.x;

    int row = bid * blockDim.y + tidy;

    float *x_start;  
    float *y_start;
    float buff[cols_per_thread];
    float row_sum;
    float row_mean;
    float row_std;

    int load_buff_idx;
    float eps = 1e-5f;

    while(row < N){
        x_start = x + row * K;  
        y_start = y + row * K;
        row_sum = 0.0f;
    
        load_buff_idx = 0;
        // load - 每次只加载一个float
        #pragma unroll
        for(int idx = tidx; idx < K; idx += warpsize){
            buff[load_buff_idx] = x_start[idx];
            row_sum += buff[load_buff_idx];
            load_buff_idx++;
        }
    
        row_sum = WarpAllReduce<SumOP, warpsize>(row_sum);
        row_mean = row_sum / K;

        row_sum = 0.0f;
        #pragma unroll
        for(int i = 0; i < load_buff_idx; ++i){
            row_sum += (buff[i] - row_mean) * (buff[i] - row_mean);
        }
    
        row_sum = WarpAllReduce<SumOP, warpsize>(row_sum);
        // 1. / std
        row_sum = rsqrtf(row_sum / K + eps);
        
        // layer norm: y = g * (x - x_mean) / x_std + b
        #pragma unroll
        for(int i = 0; i < load_buff_idx; ++i){
            int col_idx = tidx + i * warpsize;  // 计算当前元素在行中的列索引
            buff[i] = g[col_idx] * ((buff[i] - row_mean) * row_sum) + b[col_idx];
        }
    
        // store - 每次只存储一个float
        load_buff_idx = 0;
        #pragma unroll
        for(int idx = tidx; idx < K; idx += warpsize){
            y_start[idx] = buff[load_buff_idx++];
        }

        row += blockDim.y * gridDim.x;
    }
}


template<int cols_per_thread>
void dispatch_layernorm(float* d_x, float* d_y, float* g, float* b, int N, int K) {
    constexpr dim3 blockSize(32, 4); // 每个block有128个线程

    int num_blocks;
    int64_t max_blocks = (N + blockSize.y - 1) / blockSize.y;
    if (GetNumBlocks(128, max_blocks, Waves, &num_blocks) != cudaSuccess) {
        printf("GetNumBlocks错误！\n");
        exit(1);
    }

    layer_norm_float<kWarpSize, cols_per_thread><<<num_blocks, blockSize>>>(d_x, d_y, g, b, N, K);
}

extern "C" void launch_layernorm(float* d_x, float* d_y, float* g, float* b, int N, int K) {
    if(K <= 0){
        printf("layernrom错误：cols小于0，错误！\n");
        exit(1);
    }
#define DEFINE_ONE_ELIF(ColsPerThread)                                                        \
else if (K <= (ColsPerThread)*kWarpSize) {                                                 \
    dispatch_layernorm<ColsPerThread>(d_x, d_y, g, b, N, K);                                    \
}
  DEFINE_ONE_ELIF(2)
  DEFINE_ONE_ELIF(3)
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(5)
  DEFINE_ONE_ELIF(6)
  DEFINE_ONE_ELIF(7)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(9)
  DEFINE_ONE_ELIF(10)
  DEFINE_ONE_ELIF(11)
  DEFINE_ONE_ELIF(12)
  DEFINE_ONE_ELIF(13)
  DEFINE_ONE_ELIF(14)
  DEFINE_ONE_ELIF(15)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(17)
  DEFINE_ONE_ELIF(18)
  DEFINE_ONE_ELIF(19)
  DEFINE_ONE_ELIF(20)
  DEFINE_ONE_ELIF(21)
  DEFINE_ONE_ELIF(22)
  DEFINE_ONE_ELIF(23)
  DEFINE_ONE_ELIF(24)
  DEFINE_ONE_ELIF(25)
  DEFINE_ONE_ELIF(26)
  DEFINE_ONE_ELIF(27)
  DEFINE_ONE_ELIF(28)
  DEFINE_ONE_ELIF(29)
  DEFINE_ONE_ELIF(30)
  DEFINE_ONE_ELIF(31)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF     
    else{
        printf("layernrom错误：要求0 < cols <= 1024！\n");
        exit(1);
    }                                                     
}

