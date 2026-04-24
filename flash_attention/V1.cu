#include <cuda_runtime.h>
#include <stdio.h>

#include <cassert>
#include <cmath>

#include "helper.h"

/*
    朴素版本的 flash attention，只要求功能完整，不要在乎优化细节.
*/

using FP = float;
const int Br = 2;
const int Bc = 2;
constexpr int input_seq = 4096;
const int dim = 512;

__global__ void flash_attention_v2_kernel(FP *Q, FP *K, FP *V, FP *O, int seqlen, FP smScale);

void flash_attention_v2_cuda(FP *Q, FP *K, FP *V, FP *O, int m, int n) {

    //  n 代表 dim 
    FP scale = 1.f / sqrtf(static_cast<FP>(n));

    dim3 grid((input_seq + Br - 1) / Br);     //  每一个线程块独立处理一个 Q 子块
    dim3 block(Br * Bc);
    flash_attention_v2_kernel<<<grid, block>>>(Q, K, V, O, m, scale);

    DEBUG_BLOCK(printf("== v2: O ==\n"); print_device_matrix(O, SEQLEN, DIM););
}

__global__ void flash_attention_v2_kernel(FP *Q, FP *K, FP *V, FP *O, int seqlen, FP Scale) {
    
    //  每一个线程块的形状为 Br * Bc
    //  (nx, ny) 代表线程的坐标位置...
    const int nx = threadIdx.x / Bc;
    const int ny = threadIdx.x % Bc;
    const int row = blockIdx.x * Br + nx;

    if(row >= seqlen) {
        return ;
    }

    __shared__ FP sQ[Br][dim];     
    __shared__ FP sK[Bc][dim];     
    __shared__ FP sV[Bc][dim];      
    __shared__ FP sO[Br][dim];
    __shared__ FP sQK[Br][Bc];  //  计算本轮次的 QK 值
    __shared__ FP sMax[Br];     //  sMax[i] 代表第 blockIdx.x * Br + i 个元素的最大值
    __shared__ FP sSUM[Br];     //  求算 S 同一行的元素和
    #pragma unroll
    for(int i = ny;i < dim;i += Bc) {
        sQ[nx][i] = Q[row * dim + i];
        sO[nx][i] = 0;
    }

    sSUM[nx] = 0;
    sMax[nx] = -INFINITY;   //  上一个轮次的 Max(x)

    #pragma unroll
    for(int i = 0;i < (seqlen + Bc - 1) / Bc;i++) {
        #pragma unroll
        for(int j = nx;j < dim;j += Br) {
            sK[ny][j] = K[(i * Bc + ny) * dim + j];
            sV[ny][j] = V[(i * Bc + ny) * dim + j]; 
        }
        __syncthreads();
        
        //  求算 S(nx, ny)，同一行的 S 值需要被执行 softmax 操作
        FP sum = 0.f;
        #pragma unroll
        for(int j = 0;j < dim;j++) {
            sum += sQ[nx][j] * sK[ny][j];
        }
        sQK[nx][ny] = sum * Scale;
        __syncthreads();

        FP localMax = -INFINITY;
        #pragma unroll
        for(int j = 0;j < Bc;j++) {
            localMax = max(localMax, sQK[nx][j]);
        }
        __syncthreads();
        FP newMax = max(sMax[nx], localMax);
        sQK[nx][ny] = expf(sQK[nx][ny] - newMax);
        __syncthreads();

        FP lSum = 0.f;
        #pragma unroll
        for(int j = 0;j < Bc;j++) {
            lSum += sQK[nx][j];
        }
        __syncthreads();

        const FP ratio = expf(sMax[nx] - newMax);
        #pragma unroll
        for(int j = ny;j < dim;j += Bc) {
            sO[nx][j] = sO[nx][j] * ratio;
            #pragma unroll
            for(int k = 0;k < Bc;k++) {
                sO[nx][j] += sQK[nx][k] * sV[k][j];
            }
        }

        sSUM[nx] = ratio * sSUM[nx] + lSum;
        sMax[nx] = max(sMax[nx], newMax);
        __syncthreads();
    }

    #pragma unroll
    for(int j = ny;j < dim;j += Bc) {
        //  写入 O 的位置为 (row, j)
        O[row * dim + j] = sO[nx][j] / sSUM[nx];
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