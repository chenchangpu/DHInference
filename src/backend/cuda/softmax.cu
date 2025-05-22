#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <cuda.h>
#include <algorithm>
#include <assert.h>
#include <float.h>

#define kWarpSize 32
#define fInf FLT_MAX
#define FLOAT2(pointer) (reinterpret_cast<float2 *>(&(pointer))[0])

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

__device__ __forceinline__ float Exp(float x){
    return __expf(x);
}

__device__ __forceinline__ float Div(float a, float b){
    return __fdividef(a, b);
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

// 一个warp处理一行（softmax）, 适合 cols <= 1024
// blockDim.x = 32, blockDim.y = 4
template<int warpsize = 32, int cols_per_thread>
__global__ void warp_softmax_vec2(float * x, float *y, int rows, int cols){
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int bid = blockIdx.x;

    int row = bid * blockDim.y + tidy;

    float *x_start;  
    float *y_start;
    float buff[cols_per_thread];        // dahu: 注意！寄存器数组不存在地址！编译器在不能保证对齐的情况时，不能reinterpret_cast<float2 *>转换！
    float row_max;
    float row_sum;

    int load_buff_idx;

    while(row < rows){
        x_start = x + row * cols;  
        y_start = y + row * cols;
        row_max = - fInf;
        row_sum = 0.0f;
    
        load_buff_idx = 0;
        // load
        for(int idx = tidx * 2; idx < cols; idx += blockDim.x * 2){ // assert warpsize == blockDim.x
            // dahu: 这样更安全,寄存器不能直接FLOAT2 ！！！！
            // 因为寄存器不存在地址的！分配的寄存器也不连续，无法连续float2存取！！！
            float2 buff_tmp = FLOAT2(x_start[idx]);
            buff[load_buff_idx] = buff_tmp.x;
            buff[load_buff_idx + 1] = buff_tmp.y;
    
            row_max = fmaxf(fmaxf(buff_tmp.x, buff_tmp.y), row_max);
            load_buff_idx += 2;
        }
    
        row_max = WarpAllReduce<MaxOP, warpsize>(row_max);
        for(int i = 0; i < load_buff_idx; ++i){
            buff[i] = Exp(buff[i] - row_max);
            row_sum += buff[i];
        }
    
        row_sum = WarpAllReduce<SumOP, warpsize>(row_sum);
    
        for(int i = 0; i < load_buff_idx; ++i){
            buff[i] = Div(buff[i], row_sum);
        }
    
        // store
        load_buff_idx = 0;
        for(int idx = tidx * 2; idx < cols; idx += blockDim.x * 2){ // assert warpsize == blockDim.x
            float2 buff_tmp;
            buff_tmp.x = buff[load_buff_idx];
            buff_tmp.y = buff[load_buff_idx + 1];
            FLOAT2(y_start[idx]) = buff_tmp;
            load_buff_idx += 2;
        }

        row += blockDim.y * gridDim.x;          // dahu: 哇去！一开始想当然写的gridDim.y，慢死了！！！！！！！！！！！
    }
}

template<int cols_per_thread>
void dispatch_softmax(float* d_x, float* d_y, int rows, int cols) {
    constexpr dim3 blockSize(32, 4); // 每个block有128个线程
    static_assert(blockSize.x == kWarpSize);

    assert(kWarpSize * cols_per_thread >= cols);

    int num_blocks;
    int64_t max_blocks = (rows + blockSize.y - 1) / blockSize.y;
    if (GetNumBlocks(128, max_blocks, Waves, &num_blocks) != cudaSuccess) {
        printf("GetNumBlocks错误！\n");
        exit(1);
    }

    warp_softmax_vec2<kWarpSize, cols_per_thread><<<num_blocks, blockSize>>>(d_x, d_y, rows, cols);
}

// 根据cols选择合适模板参数
// 要求cols % 2 == 0 且 2*32 <= cols <= 1024
extern "C" void launch_softmax(float* d_x, float* d_y, int rows, int cols) {
    if(cols <= 0){
        printf("sofmax错误：cols小于0，错误！\n");
        exit(1);
    }
#define DEFINE_ONE_ELIF(cols_per_thread)                                                        \
else if (cols <= (cols_per_thread)*kWarpSize) {                                                 \
    dispatch_softmax<cols_per_thread>(d_x, d_y, rows, cols);                                    \
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
        printf("softmax错误：要求cols % 2 == 0 且 2*32 <= cols <= 1024！\n");
        exit(1);
    }                                                     
}
