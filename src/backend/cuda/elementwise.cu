#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void elementwise_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

__global__ void elementwise_sub(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] - b[idx];
}

__global__ void elementwise_mul(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] * b[idx];
}

__global__ void elementwise_div(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = __fdividef(a[idx], b[idx] + 1e-9);
}

__global__ void elementwise_relu(float* a, float* b, int n) { // b = relu(a)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) b[idx] = fmaxf(a[idx], 0);             // relu(x) = max(x, 0)
}

extern "C" void launch_elementwise_add(float* a, float* b, float* c, int n){
    int num_threads = 128;
    int num_blocks = (n + num_threads - 1) / num_threads;

    elementwise_add<<<num_blocks, num_threads>>>(a, b, c, n);
}

extern "C" void launch_elementwise_sub(float* a, float* b, float* c, int n){
    int num_threads = 128;
    int num_blocks = (n + num_threads - 1) / num_threads;

    elementwise_sub<<<num_blocks, num_threads>>>(a, b, c, n);
}

extern "C" void launch_elementwise_mul(float* a, float* b, float* c, int n){
    int num_threads = 128;
    int num_blocks = (n + num_threads - 1) / num_threads;

    elementwise_mul<<<num_blocks, num_threads>>>(a, b, c, n);
}

extern "C" void launch_elementwise_div(float* a, float* b, float* c, int n){
    int num_threads = 128;
    int num_blocks = (n + num_threads - 1) / num_threads;

    elementwise_div<<<num_blocks, num_threads>>>(a, b, c, n);
}

extern "C" void launch_elementwise_relu(float* a, float* b, int n){
    int num_threads = 128;
    int num_blocks = (n + num_threads - 1) / num_threads;

    elementwise_relu<<<num_blocks, num_threads>>>(a, b, n);
}