#include <cuda_runtime.h>
#include <stdio.h>

#include <cassert>
#include <cmath>

#include "helper.h"



/*
    在 V1 版本中，处理 S 某一行的线程数量很少，观察发现 加载 sQ, sK，sV 这些的数据次数是固定的.
    在 V2 版本中，考虑采取缓存机制 和 float4加载方式。为了节省共享显存资源，每一个线程块一次仅仅处理 S 的一行数据...
*/

using FP = float;
constexpr int qDivide = 16;
constexpr int input_seq = 4096;
constexpr int dim = 512;
__constant__ float4 c_zero_float4 = {0.0f, 0.0f, 0.0f, 0.0f};

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
__global__ void flash_attention_v2_kernel(FP *Q, FP *K, FP *V, FP *O, int seqlen, FP smScale);

__inline__ __device__
float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
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
    const int width = dim / blockDim.x;     //  确保 width % 4 == 0

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
        //  当前加载子块的第 row + i 行
        #pragma unroll
        for(int j = tid * width;j < (tid + 1) * width;j += 4) {     //  加载可能出现 bank confict
            FETCH_FLOAT4(sQ[j]) = FETCH_FLOAT4(Q[(row + i) * dim + j]);
            FETCH_FLOAT4(sO[j]) = c_zero_float4;
            FETCH_FLOAT4(sK[work_index][j]) = FETCH_FLOAT4(K[j]);
            FETCH_FLOAT4(sV[work_index][j]) = FETCH_FLOAT4(V[j]);
        }
        if(tid == 0) {
            sMax = -INFINITY;
            total = 0.f;
        }
        __syncthreads();        //  等待线程0 完成初始化
        #pragma unroll
        for(int k = 0;k < seqlen;k++) {
            if(k + 1 < seqlen) {
                //  缓存下一轮数据
                //  相比于 V1 版本，缓冲机制大幅度减少了同步次数
                #pragma unroll
                for(int j = tid * width;j < (tid + 1) * width;j += 4) {
                    FETCH_FLOAT4(sK[(k + 1) % 2][j]) = FETCH_FLOAT4(K[(k + 1) * dim + j]);
                    FETCH_FLOAT4(sV[(k + 1) % 2][j]) = FETCH_FLOAT4(V[(k + 1) * dim + j]);
                }
            }
            //  求算 Q[i][:] * K[:][j]，采取Warp归约的方式求算
            float sum1 = 0.f;
            #pragma unroll
            for(int j = tid;j < dim;j += blockDim.x) {
                sum1 += sQ[j] * sK[work_index][j];
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
            for(int j = tid;j < dim;j += blockDim.x) {
                //  求算 O[(row + i)][j]
                sO[j] *= ratio;
                sO[j] += sum * sV[work_index][j];
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
        for(int j = tid * width;j < (tid + 1) * width;j += 4) {
            float4 val = FETCH_FLOAT4(sO[j]);
            val.x *= inv_total;
            val.y *= inv_total;
            val.z *= inv_total;
            val.w *= inv_total;
            FETCH_FLOAT4(O[(row + i) * dim + j]) = val;
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