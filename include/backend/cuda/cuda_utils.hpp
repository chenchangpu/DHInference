#ifndef BACKEND_CUDA_UTILS_H
#define BACKEND_CUDA_UTILS_H

#include <cstddef>  // for size_t

// namespace dhinference {
// namespace backend {

#ifdef _WIN32
#define CUDA_API __declspec(dllexport)
#else
#define CUDA_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// 显存管理, 数据拷贝
// cuda_memory.cu
CUDA_API void* gpu_malloc(size_t size);     // 分配显存
CUDA_API void gpu_free(void* ptr);          // 释放显存
CUDA_API void copy_cpu_to_gpu(void* d_dst, const void* h_src, size_t size);
CUDA_API void copy_gpu_to_cpu(void* h_dst, const void* d_src, size_t size);
CUDA_API void copy_gpu_to_gpu(void* h_dst, const void* d_src, size_t size);
CUDA_API void cuda_device_sync();

// ops
// elementwise.cu
CUDA_API void launch_elementwise_add(float* a, float* b, float* c, int n);
CUDA_API void launch_elementwise_sub(float* a, float* b, float* c, int n);
CUDA_API void launch_elementwise_mul(float* a, float* b, float* c, int n);
CUDA_API void launch_elementwise_div(float* a, float* b, float* c, int n);
CUDA_API void launch_elementwise_relu(float* a, float* b, int n);

CUDA_API void launch_elementwise_add_oneflow(float* a, float* b, float* c, int n);
CUDA_API void launch_elementwise_relu_oneflow(float* a, float* b, int n);

// layernorm.cu
CUDA_API void launch_layernorm(float* d_x, float* d_y, float* g, float* b, int N, int K);
CUDA_API void launch_layernorm_oneflow(float* d_x, float* d_y, float* g, float* b, int N, int K);

// rmsnorm.cu
CUDA_API void launch_rmsnorm(float* d_x, float* d_y, float* g, int N, int K);

// sgemm.cu
CUDA_API void launch_sgemm_default(    
    float * __restrict__ A, 
    float * __restrict__ B, 
    float * __restrict__ C, 
    const size_t M, 
    const size_t N, 
    const size_t K,
    float alpha);

CUDA_API void launch_sgemm_cublas_default(    
    float * __restrict__ A, 
    float * __restrict__ B, 
    float * __restrict__ C, 
    const size_t M, 
    const size_t N, 
    const size_t K,
    float alpha);

// sgemv.cu
CUDA_API void launch_sgemv_96(float* A, float* x, float* y, int M, int K);
CUDA_API void launch_sgemv_128(float* A, float* x, float* y, int M, int K);
CUDA_API void launch_sgemv_256(float* A, float* x, float* y, int M, int K);

// softmax.cu
CUDA_API void launch_softmax(float* d_x, float* d_y, int rows, int cols);
CUDA_API void launch_softmax_oneflow(float* d_x, float* d_y, int rows, int cols);

// transpose.cu
CUDA_API void launch_transpose(const float* input, float* output, const int shape[4], const int perm[4], void* extra_buff, int rank);

#ifdef __cplusplus
}
#endif

// }   // end of backend
// }   // end of dhinference

#endif // BACKEND_CUDA_UTILS_H
