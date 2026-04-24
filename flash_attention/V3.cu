#include <cuda_runtime.h>
#include <stdio.h>

#include <cassert>
#include <cmath>

#include "helper.h"

/*
    在 V2版本中，线程之间访存无法按照 32 字节合并，V3版本将修复该性能缺陷.
*/

using FP = float;
constexpr int qDivide = 16;
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

    constexpr int blockSize = 32;
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
    const int tid = threadIdx.x;
    const int width = dim / 4;      //  按照 float4 访存

    __shared__ float sQ[dim];
    __shared__ float sK[2][dim];
    __shared__ float sV[2][dim];
    __shared__ float sO[dim];
    //  确保 blockDim.x % warpSize == 0
    __shared__ float sSum;
    __shared__ float sMax;
    __shared__ float total;
    
    #pragma unroll
    for(int i = 0;i < qDivide;i++) {
        
        int work_index = 0; 
        #pragma unroll
        for(int j = tid;j < width;j += blockDim.x) {
            const int index = j * 4;
            FETCH_FLOAT4(sQ[index]) = FETCH_FLOAT4(Q[(row + i) * dim + index]);
            FETCH_FLOAT4(sO[index]) = c_zero_float4;
            FETCH_FLOAT4(sK[work_index][index]) = FETCH_FLOAT4(K[index]);
            FETCH_FLOAT4(sV[work_index][index]) = FETCH_FLOAT4(V[index]);
        }
        if(tid == 0) {
            sMax = -INFINITY;
            total = 0.f;
        }
        __syncthreads();        //  等待线程0 完成初始化
        #pragma unroll
        for(int k = 0;k < seqlen;k++) {
            if(k + 1 < seqlen) {
                #pragma unroll
                for(int j = tid;j < width;j += blockDim.x) {
                    const int index = j * 4;
                    FETCH_FLOAT4(sK[(k + 1) % 2][index]) = FETCH_FLOAT4(K[(k + 1) * dim + index]);
                    FETCH_FLOAT4(sV[(k + 1) % 2][index]) = FETCH_FLOAT4(V[(k + 1) * dim + index]);
                }
            }
            //  求算 Q[i][:] * K[:][j]，采取Warp归约的方式求算
            float sum1 = 0.f;
            #pragma unroll
            for(int j = tid;j < width;j += blockDim.x) {
                const int index = j * 4;
                float4 left = FETCH_FLOAT4(sQ[index]);
                float4 right = FETCH_FLOAT4(sK[work_index][index]);
                sum1 += left.x * right.x;
                sum1 += left.y * right.y;
                sum1 += left.z * right.z;
                sum1 += left.w * right.w;
            }
            sum1 = warpReduceSum(sum1);
            if(tid % warpSize == 0) {       //  同一个 Warp 内仅需要第一个线程存储数据
                sSum = sum1;
            }
            __syncthreads();
            float sum = sSum;
            sum *= Scale;

            float newMax = max(sMax, sum);
            sum = expf(sum - newMax);
            const float ratio = expf(sMax - newMax);

            //  求算 attn * V
            #pragma unroll
            for(int j = tid;j < width;j += blockDim.x) {
                float4 left = FETCH_FLOAT4(sO[j * 4]);
                float4 right = FETCH_FLOAT4(sV[work_index][j * 4]);
                left.x = left.x * ratio + sum * right.x;
                left.y = left.y * ratio + sum * right.y;
                left.z = left.z * ratio + sum * right.z;
                left.w = left.w * ratio + sum * right.w;
                FETCH_FLOAT4(sO[j * 4]) = left;
            }

            if(tid == 0) {
                total = total * ratio + sum;
                sMax = max(sMax, newMax);
            }

            work_index ^= 1;
            __syncthreads();
        }

        float inv_total = 1.0f / total;
        #pragma unroll  
        for(int j = tid;j < width;j += blockDim.x) {
            float4 val = FETCH_FLOAT4(sO[j * 4]);
            val.x *= inv_total;
            val.y *= inv_total;
            val.z *= inv_total;
            val.w *= inv_total;
            FETCH_FLOAT4(O[(row + i) * dim + 4 * j]) = val;
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