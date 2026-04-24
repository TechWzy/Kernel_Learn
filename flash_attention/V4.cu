#include <cuda_runtime.h>
#include <stdio.h>

#include <cassert>
#include <cmath>

#include "helper.h"

/*
    由于同一个 warp 仅仅处理 K 和 V 的一行，因此某些条件判断允许采用 warp 内原语同步消除条件判断。此外采取寄存器取代共享内存进一步提高 访存效率.
    此外，本例扩大 qDivide 若干倍，并增加 warp 的数量，每一个 warp 分别处理K 和 V 的不同行，且引入 warp 级别同步进一步提供程序运行效率...
    
    相比于 V3版本，排除暴力求算过程（注意两者的线程数使用量一致相同），运行时间对比如下：
    1. input_seq = 128, dim = 512
        V3:  0.126 ms  &  V4: 0.076 ms
    2. input_seq = 512, dim = 512
        V3:  0.522 ms  &  V4: 0.367 ms
    3. input_seq = 1024, dim = 512
        V3:  1.084 ms  &  V4: 0.993 ms
    4. input_seq = 2048, dim = 512
        V3:  4.369 ms  &  V4: 4.001 ms
    5. input_seq = 4096, dim = 512
        V3: 17.791 ms  &  V4: 13.235 ms 
    6. input_seq = 8192, dim = 512
        V3: 71.439 ms  &  V4: 48.947 ms
    7. input_seq = 16384, dim = 512
        V3: 267.621 ms  &  V4: 194.306 ms
    8. input_seq = 32768, dim = 512
        V3: 1072.727 ms  &  V4: 753.824 ms      （共享内存 和 寄存器等资源开始受限）

    观察发现，V4 的运行时间更占优势，一方面 我们采取 warp 内原语机制大幅度消除了 warp 分支，另一方面 随着 warp 数量增多，活跃 warp 和 符合条件的 warp 将增多，
    有效隐藏访存延迟。
*/

using FP = float;
constexpr int qDivide = 64;
constexpr int input_seq = 65536;
constexpr int dim = 512;
__constant__ float4 c_zero_float4 = {0.0f, 0.0f, 0.0f, 0.0f};

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
__global__ void flash_attention_v2_kernel(FP *Q, FP *K, FP *V, FP *O, int seqlen, FP smScale);

__inline__ __device__
float warpReduceSum(float val) {
    val += __shfl_down_sync(0xFFFFFFFF, val, 16);
    val += __shfl_down_sync(0xFFFFFFFF, val, 8);
    val += __shfl_down_sync(0xFFFFFFFF, val, 4);
    val += __shfl_down_sync(0xFFFFFFFF, val, 2);
    val += __shfl_down_sync(0xFFFFFFFF, val, 1);
    return val;
}

void flash_attention_v2_cuda(FP *Q, FP *K, FP *V, FP *O, int m, int n) {
    constexpr int blockSize = 32 * 4;
    //  n 代表 dim 
    FP scale = 1.f / sqrtf(static_cast<FP>(n));
    dim3 grid((m + qDivide - 1) / qDivide);
    dim3 block(blockSize);
    flash_attention_v2_kernel<<<grid, block>>>(Q, K, V, O, m, scale);
    DEBUG_BLOCK(printf("== v2: O ==\n"); print_device_matrix(O, SEQLEN, DIM););
}

__global__ void flash_attention_v2_kernel(FP *Q, FP *K, FP *V, FP *O, int seqlen, FP Scale) {

    //  当前线程块处理Q的行范围是 [row, row + qDivide)
    const int row = blockIdx.x * qDivide;
    const int row_in_warp = threadIdx.x / warpSize;
    const int tid = threadIdx.x;
    const int warp_idx = tid % warpSize;
    const int width = dim / 4;      //  按照 float4 访存

    __shared__ float sQ[dim];
    __shared__ float sK[4][dim];
    __shared__ float sV[4][dim];
    __shared__ float sO[4][dim];
    __shared__ float WarpMax[4];
    __shared__ float WarpTotal[4];

    #pragma unroll
    for(int i = 0;i < qDivide;i++) {
        //  共享内存 -> 寄存器，采用 warp 内同步...
        float sMax = -INFINITY;
        float total = 0.f;
        //  全部线程加载 Q 和 O，此处涉及线程块同步
        #pragma unroll
        for(int j = tid;j < width;j += blockDim.x) {
            const int index = j * 4;
            FETCH_FLOAT4(sQ[index]) = FETCH_FLOAT4(Q[(row + i) * dim + index]);
        }
        if(i == 0) {
            #pragma unroll
            for(int j = tid;j < width;j += blockDim.x) {
                const int index = j * 4;
                FETCH_FLOAT4(sO[0][index])  = c_zero_float4;
                FETCH_FLOAT4(sO[1][index])  = c_zero_float4;
                FETCH_FLOAT4(sO[2][index])  = c_zero_float4;
                FETCH_FLOAT4(sO[3][index])  = c_zero_float4;
            }
        }
        __syncthreads();
        #pragma unroll
        for(int k = 0;k < seqlen;k += 4) {
            //  当前线程处理 k + row_in_warp 行数据
            #pragma unroll
            for(int j = warp_idx;j < width;j += 32) {
                const int index = j * 4;
                FETCH_FLOAT4(sK[row_in_warp][index]) = FETCH_FLOAT4(K[(k + row_in_warp) * dim + index]);
                FETCH_FLOAT4(sV[row_in_warp][index]) = FETCH_FLOAT4(V[(k + row_in_warp) * dim + index]);
            }

            //  求算 Q * K. 对于同一个线程，加载 sK 和 使用 sK 的位置是一样的，因此此处无需同步
            float sum_warp = 0.f;
            #pragma unroll
            for(int j = warp_idx;j < width;j += 32) {
                const int index = j * 4;
                float4 left = FETCH_FLOAT4(sQ[index]);
                float4 right = FETCH_FLOAT4(sK[row_in_warp][index]);
                sum_warp += left.x * right.x;
                sum_warp += left.y * right.y;
                sum_warp += left.z * right.z;
                sum_warp += left.w * right.w;
            }
            
            //  之所以放在这个位置，是为了避免因同步而浪费线程加载数据和计算的时间...
            //  确保精度一致，保证同一个 Warp 内的所有线程共用同一个 sMax 和 total
            total = __shfl_sync(0xFFFFFFFF, total, 0);
            sMax = __shfl_sync(0xFFFFFFFF, sMax, 0);

            sum_warp = warpReduceSum(sum_warp);
            sum_warp = __shfl_sync(0xFFFFFFFF, sum_warp, 0);        //  避免线程束分化
            sum_warp *= Scale;

            float newMax = max(sMax, sum_warp);
            sum_warp = expf(sum_warp - newMax);
            const float ratio = expf(sMax - newMax);

            //  求算 attn * V
            #pragma unroll
            for(int j = warp_idx;j < width;j += 32) {
                float4 left = FETCH_FLOAT4(sO[row_in_warp][j * 4]);
                float4 right = FETCH_FLOAT4(sV[row_in_warp][j * 4]);
                left.x = left.x * ratio + sum_warp * right.x;
                left.y = left.y * ratio + sum_warp * right.y;
                left.z = left.z * ratio + sum_warp * right.z;
                left.w = left.w * ratio + sum_warp * right.w;
                FETCH_FLOAT4(sO[row_in_warp][j * 4]) = left;
            }

            total = total * ratio + sum_warp;
            sMax = max(sMax, newMax);
        }

        total = __shfl_sync(0xFFFFFFFF, total, 0);
        sMax = __shfl_sync(0xFFFFFFFF, sMax, 0);
        
        //  事实上允许采用 条件判断的方式，仅需要 warp_idx == 0加载数据
        //  为了避免线程束分化，所有线程进行重复操作
        WarpMax[row_in_warp] = sMax;
        WarpTotal[row_in_warp] = total;

        __syncthreads();
        //  4 合 2，观察发现，此处不会出现线程束分化
        if(row_in_warp % 2 == 0) {
            float newMax = max(WarpMax[row_in_warp], WarpMax[row_in_warp + 1]);
            const float ratio1 = expf(WarpMax[row_in_warp] - newMax);
            const float ratio2 = expf(WarpMax[row_in_warp + 1] - newMax);
            total = WarpTotal[row_in_warp] * ratio1 + WarpTotal[row_in_warp + 1] * ratio2;
            #pragma unroll
            for(int j = warp_idx;j < width;j += 32) {
                float4 left = FETCH_FLOAT4(sO[row_in_warp][j * 4]);
                float4 right = FETCH_FLOAT4(sO[row_in_warp + 1][j * 4]);
                left.x = left.x * ratio1 + right.x * ratio2;
                left.y = left.y * ratio1 + right.y * ratio2;
                left.z = left.z * ratio1 + right.z * ratio2;
                left.w = left.w * ratio1 + right.w * ratio2;
                FETCH_FLOAT4(sO[row_in_warp][j * 4]) = left;
            }
            WarpMax[row_in_warp] = newMax;
            WarpTotal[row_in_warp] = total;
        }

        __syncthreads();

        if(row_in_warp % 2 != 0) {
            #pragma unroll
            for(int j = warp_idx;j < width;j += 32) {
                FETCH_FLOAT4(sO[row_in_warp][j * 4]) = c_zero_float4;
            }
        }

        if(row_in_warp == 0) {
            float newMax = max(WarpMax[0], WarpMax[2]);
            const float ratio1 = expf(WarpMax[0] - newMax);
            const float ratio2 = expf(WarpMax[2] - newMax);
            total = WarpTotal[0] * ratio1 + WarpTotal[2] * ratio2;
            #pragma unroll
            for(int j = warp_idx;j < width;j += 32) {
                float4 left = FETCH_FLOAT4(sO[0][j * 4]);
                float4 right = FETCH_FLOAT4(sO[2][j * 4]);
                left.x = left.x * ratio1 + right.x * ratio2;
                left.y = left.y * ratio1 + right.y * ratio2;
                left.z = left.z * ratio1 + right.z * ratio2;
                left.w = left.w * ratio1 + right.w * ratio2;
                left.x /= total;
                left.y /= total;
                left.z /= total;
                left.w /= total;
                FETCH_FLOAT4(O[(row + i) * dim + j * 4]) = left;
                FETCH_FLOAT4(sO[0][j * 4]) = c_zero_float4;
                FETCH_FLOAT4(sO[2][j * 4]) = c_zero_float4;
            }
        }
    }
}

void test_attention() {
    // seqlen
    int m = input_seq;
    // dim
    int n = dim;

    // Host pointer
    float *h_K = new float[m * n];
    float *h_Q = new float[m * n];
    float *h_V = new float[m * n];
    float *h_O = new float[m * n];
    float *h_O2 = new float[m * n];

    // 初始化 K, Q, V
    for (int i = 0; i < m * n; ++i) {
        h_K[i] = static_cast<float>(rand()) / RAND_MAX;
        h_Q[i] = static_cast<float>(rand()) / RAND_MAX;
        h_V[i] = static_cast<float>(rand()) / RAND_MAX;

        DEBUG_BLOCK(h_K[i] = static_cast<float>(i); h_Q[i] = static_cast<float>(i);
                    h_V[i] = static_cast<float>(i););
    }

    DEBUG_BLOCK(printf("== K ==\n"); print_host_matrix(h_K, m, n););

    float *d_K, *d_Q, *d_V, *d_O, *d_O2;
    // Malloc device memory
    cudaMalloc((void **)&d_K, sizeof(float) * m * n);
    cudaMalloc((void **)&d_Q, sizeof(float) * m * n);
    cudaMalloc((void **)&d_V, sizeof(float) * m * n);
    cudaMalloc((void **)&d_O, sizeof(float) * m * n);
    cudaMalloc((void **)&d_O2, sizeof(float) * m * n);

    // Copy data from host to device
    cudaMemcpy(d_K, h_K, sizeof(float) * m * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Q, h_Q, sizeof(float) * m * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, sizeof(float) * m * n, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // for (int i = 0; i < 1; i++) {
    //     // Launch kernel
    //     self_attention_cuda(d_Q, d_K, d_V, d_O, m, n);
    //     CUDA_CHECK(cudaGetLastError());
    // }

    // test flash attention 2
    for (int i = 0; i < 1; i++) {
        flash_attention_v2_cuda(d_Q, d_K, d_V, d_O2, m, n);
        CUDA_CHECK(cudaGetLastError());
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time for kernel execution: %.3f ms \n", milliseconds / 100);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Result back to host
    cudaMemcpy(h_O, d_O, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_O2, d_O2, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
    // bool res = all_close(h_O, h_O2, m, n);
    // if (res) {
    //     printf("is equal\n");
    // } else {
    //     printf("is not equal\n");
    // }

    cudaFree(d_K);
    cudaFree(d_Q);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_O2);
    free(h_Q);
    free(h_K);
    free(h_V);
    free(h_O);
    free(h_O2);
}

int main() {
    int epoch = 1;
    for (int i = 0; i < epoch; i++) {
        test_attention();
    }
  return 0;
}