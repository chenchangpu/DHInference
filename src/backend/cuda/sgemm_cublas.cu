#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda.h>
#include <assert.h>
#include "cublas_v2.h"

extern "C" void launch_sgemm_cublas_default(
    float * __restrict__ A, 
    float * __restrict__ B, 
    float * __restrict__ C, 
    const size_t M, 
    const size_t N, 
    const size_t K,
    float alpha)
{
    cublasHandle_t handle;
    cublasCreate(&handle);

    float beta = 0;
    cublasSgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            N,              // 
            M,              // 
            K,              // 
            &alpha,         //alpha
            B,              //
            N,              //leading dimension
            A,              //
            K,              //leading dimension
            &beta,          //beta
            C,              //C
            N               //C leading dimension
    );

    cublasDestroy(handle);
}