#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda.h>
#include <assert.h>


/*
    1. shared memory 的 double buffer
    2. register 的 double buffer
*/

using namespace std;

#define FLOAT4(a) ((reinterpret_cast<float4 *>(&a))[0])

template<
    int BM,                 // 128
    int BN,                 // 128
    int BK,                 // 8
    int TM,                 // 8
    int TN,                 // 8
    int THREADS_PER_BLOCK   // 256
> __global__ void sgemm(
    float * __restrict__ A, 
    float * __restrict__ B, 
    float * __restrict__ C, 
    const size_t M, 
    const size_t N, 
    const size_t K,
    float alpha)            // C = alpha * A * B
    {
    static_assert(BM % TM == 0);
    static_assert(BN % TN == 0);
    static_assert(THREADS_PER_BLOCK * TM * TN == BM * BN);

    const int bx = blockIdx.x;                                    // 2-D Grid
    const int by = blockIdx.y;
    const int tid = threadIdx.x;                                  // 1-D Block
    float *A_start = A + by * BM * K;
    float *B_start = B + bx * BN;
    float *C_start = C + by * BM * N + bx * BN;

    // double buffer
    __shared__ float sA[2][BK][BM];
    __shared__ float sB[2][BK][BN];

    float rC[TM][TN] = {0};
    float rA[2][TM], rB[2][TN];

    constexpr int float4_per_row_A = BK / 4;                            // 8 / 4 = 2
    constexpr int float4_per_row_B = BN / 4;                            // 128 / 4 = 32
    constexpr int rows_per_block_A = THREADS_PER_BLOCK / float4_per_row_A;      // 256 / 2 = 128
    constexpr int rows_per_block_B = THREADS_PER_BLOCK / float4_per_row_B;      // 256 / 32 = 8
    const int load_global_y_idx_A = tid / float4_per_row_A;
    const int load_global_x_idx_A = tid % float4_per_row_A * 4;
    const int load_global_y_idx_B = tid / float4_per_row_B;
    const int load_global_x_idx_B = tid % float4_per_row_B * 4;

    constexpr int threads_per_row = BN / TN;                // 128 / 8 = 16
    const int calc_c_y_idx = tid / threads_per_row * TM;
    const int calc_c_x_idx = tid % threads_per_row * TN;

    float load_g_A[4*BM/rows_per_block_A];
    float load_g_B[4*BK/rows_per_block_B];

    int load_shared_idx = 0;
    // tile-0: k = 0 -> shared[0]
    // copy sA
    #pragma unroll
    for(int row = 0; row < BM; row += rows_per_block_A){
        float4 load_g_A_tmp = FLOAT4(A_start[(row + load_global_y_idx_A) * K + load_global_x_idx_A]);
        sA[0][load_global_x_idx_A][row + load_global_y_idx_A] = load_g_A_tmp.x;
        sA[0][load_global_x_idx_A + 1][row + load_global_y_idx_A] = load_g_A_tmp.y;
        sA[0][load_global_x_idx_A + 2][row + load_global_y_idx_A] = load_g_A_tmp.z;
        sA[0][load_global_x_idx_A + 3][row + load_global_y_idx_A] = load_g_A_tmp.w;
    }
    // copy sB
    #pragma unroll
    for(int row = 0; row < BK; row += rows_per_block_B){
        FLOAT4(sB[0][row + load_global_y_idx_B][load_global_x_idx_B]) = 
                            FLOAT4(B_start[(row + load_global_y_idx_B) * N + load_global_x_idx_B]);
    }
    __syncthreads();

    // register double buffer
    // load sA and sB to rA[0] and rB[0]
    #pragma unroll
    for(int i = 0; i < TM; i += 4){
        FLOAT4(rA[0][i]) = FLOAT4(sA[0][0][calc_c_y_idx + i]);
    }

    // load sB to rB
    #pragma unroll 
    for(int i = 0; i < TN; i += 4){
        FLOAT4(rB[0][i]) = FLOAT4(sB[0][0][calc_c_x_idx + i]);
    }

    // tile-0 in shared[0], start with tile[1]: k = BK
    #pragma unroll
    for(int k = 0; k + BK < K;){
        k += BK;
        load_shared_idx ^= 1;

        // copy from global to shared
        // copy sA
        #pragma unroll
        for(int row = 0, idx = 0; row < BM; row += rows_per_block_A, idx + 4){
            FLOAT4(load_g_A[idx]) = FLOAT4(A_start[(row + load_global_y_idx_A) * K + k + load_global_x_idx_A]);
        }
        // copy sB
        #pragma unroll
        for(int row = 0, idx = 0; row < BK; row += rows_per_block_B, idx + 4){
            FLOAT4(load_g_B[idx]) = FLOAT4(B_start[(k + row + load_global_y_idx_B) * N + load_global_x_idx_B]);
        }
        // __syncthreads();                      // delay

        // calculate rC, direct load from smem
        #pragma unroll
        for(int t = 0; t < BK - 1; ++t){
            // load sA and sB to rA and rB
            // load sA to rA
            #pragma unroll
            for(int i = 0; i < TM; i += 4){
                FLOAT4(rA[(t+1)%2][i]) = FLOAT4(sA[load_shared_idx ^ 1][t+1][calc_c_y_idx + i]);
            }
            // load sB to rB
            #pragma unroll 
            for(int i = 0; i < TN; i += 4){
                FLOAT4(rB[(t+1)%2][i]) = FLOAT4(sB[load_shared_idx ^ 1][t+1][calc_c_x_idx + i]);
            }
            // calculate rC
            #pragma unroll
            for(int i = 0; i < TM; ++i){
                #pragma unroll
                for(int j = 0; j < TN; ++j){
                    rC[i][j] += rA[t%2][i] * rB[t%2][j];
                }
            }
        }

        // register -> shared
        #pragma unroll
        for(int row = 0, idx = 0; row < BM; row += rows_per_block_A, idx += 4){
            sA[load_shared_idx][load_global_x_idx_A][row + load_global_y_idx_A] = load_g_A[idx];
            sA[load_shared_idx][load_global_x_idx_A + 1][row + load_global_y_idx_A] = load_g_A[idx + 1];
            sA[load_shared_idx][load_global_x_idx_A + 2][row + load_global_y_idx_A] = load_g_A[idx + 2];
            sA[load_shared_idx][load_global_x_idx_A + 3][row + load_global_y_idx_A] = load_g_A[idx + 3];
        }
        #pragma unroll
        for(int row = 0, idx = 0; row < BK; row += rows_per_block_B, idx += 4){
            FLOAT4(sB[load_shared_idx][row + load_global_y_idx_B][load_global_x_idx_B]) = 
                            FLOAT4(load_g_B[idx]);
        }
        __syncthreads();

        // load first tile from shared to register
        #pragma unroll
        for(int i = 0; i < TM; i += 4){
            FLOAT4(rA[0][i]) = FLOAT4(sA[load_shared_idx][0][calc_c_y_idx + i]);
        }
        // load sB to rB
        #pragma unroll 
        for(int i = 0; i < TN; i += 4){
            FLOAT4(rB[0][i]) = FLOAT4(sB[load_shared_idx][0][calc_c_x_idx + i]);
        }

        // last register tile compute
        #pragma unroll
        for(int i = 0; i < TM; ++i){
            #pragma unroll
            for(int j = 0; j < TN; ++j){
                rC[i][j] += rA[1][i] * rB[1][j];
            }
        }
    }

    // calculation of last tile
    #pragma unroll
    for(int t = 0; t < BK - 1; ++t){
        #pragma unroll
        for(int i = 0; i < TM; i += 4){
            FLOAT4(rA[(t+1)%2][i]) = FLOAT4(sA[load_shared_idx][t+1][calc_c_y_idx + i]);
        }
        #pragma unroll 
        for(int i = 0; i < TN; i += 4){
            FLOAT4(rB[(t+1)%2][i]) = FLOAT4(sB[load_shared_idx][t+1][calc_c_x_idx + i]);
        }
        #pragma unroll
        for(int i = 0; i < TM; ++i){
            #pragma unroll
            for(int j = 0; j < TN; ++j){
                rC[i][j] += rA[t%2][i] * rB[t%2][j];
            }
        }
    }
    #pragma unroll
    for(int i = 0; i < TM; ++i){
        #pragma unroll
        for(int j = 0; j < TN; ++j){
            rC[i][j] += rA[1][i] * rB[1][j];
        }
    }

    // epilogue: scale
    for(int i = 0; i < TM; ++i){
        #pragma unroll
        for(int j = 0; j < TN; ++j){
            rC[i][j] *= alpha;
        }
    }

    // copy rC to global C
    #pragma unroll
    for(int i = 0; i < TM; ++i){
        #pragma unroll
        for(int j = 0; j < TN; j += 4){
            FLOAT4(C_start[(calc_c_y_idx + i) * N + calc_c_x_idx + j]) = FLOAT4(rC[i][j]);
        }
    }
}


template<
    int BM = 128,                 // 128
    int BN = 128,                 // 128
    int BK = 8,                 // 8
    int TM = 8,                 // 8
    int TN = 8,                 // 8
    int THREADS_PER_BLOCK = 256   // 256
>void launch_sgemm(
    float * __restrict__ A, 
    float * __restrict__ B, 
    float * __restrict__ C, 
    const size_t M, 
    const size_t N, 
    const size_t K,
    float alpha){
    const dim3 num_blocks((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm<BM, BN, BK, TM, TN, THREADS_PER_BLOCK>
        <<<num_blocks, THREADS_PER_BLOCK>>>(A, B, C, M, N, K, alpha);
}

// 显示实例化
extern "C" void launch_sgemm_default(
    float * __restrict__ A, 
    float * __restrict__ B, 
    float * __restrict__ C, 
    const size_t M, 
    const size_t N, 
    const size_t K,
    float alpha)
{
        launch_sgemm<128, 128, 8, 8, 8, 256>(A, B, C, M, N, K, alpha);
}