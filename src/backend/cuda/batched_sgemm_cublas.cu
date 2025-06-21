#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda.h>
#include <assert.h>
#include "cublas_v2.h"

extern "C" void launch_batched_sgemm_cublas_default(
    float * __restrict__ A,     // [Batch, M, K]
    float * __restrict__ B,     // [Batch, K, N]
    float * __restrict__ C,     // [Batch, M, N]
    const size_t Batch,
    const size_t M, 
    const size_t N, 
    const size_t K,
    float alpha)
{
    // host
    const float** Aarray = (const float**)malloc(Batch * sizeof(float*));
    const float** Barray = (const float**)malloc(Batch * sizeof(float*));
    float** Carray = (float**)malloc(Batch * sizeof(float*));
    // device
    const float** d_Aarray;
    const float** d_Barray;
    float** d_Carray;
    cudaMalloc(&d_Aarray, Batch * sizeof(float*));
    cudaMalloc(&d_Barray, Batch * sizeof(float*));
    cudaMalloc(&d_Carray, Batch * sizeof(float*));
    // 指针数组赋值
    for(int i = 0; i < Batch; ++i){
        Aarray[i] = (const float*)(A + i * M * K);
        Barray[i] = (const float*)(B + i * K * N);
        Carray[i] = (float*)(C + i * M * N);
    }
    cudaMemcpy(d_Aarray, Aarray, Batch * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Barray, Barray, Batch * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Carray, Carray, Batch * sizeof(float*), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    float beta = 0;
    cublasSgemmBatched(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            N, M, K,              
            &alpha, 
            d_Barray, N,
            d_Aarray, K, 
            &beta,
            d_Carray, N,
            Batch
    );

    cublasDestroy(handle);
}